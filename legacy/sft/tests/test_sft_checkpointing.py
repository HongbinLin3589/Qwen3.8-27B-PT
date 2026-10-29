"""Check whole-sublayer checkpoint semantics using actual mixed Qwen3.5 CPU layers."""

from contextlib import nullcontext
import copy
from functools import partial
import gc
import weakref

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from legacy.sft.utils.verify_sft_offload import ADAM_OPTIONS, compare_tensors
from legacy.sft.verl.checkpointing import qwen3_5_sublayer_checkpointing


def _model(monkeypatch):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    from transformers import Qwen3_5TextConfig

    # Use HF's real differentiable CPU fallback; no mock attention math or CUDA kernels.
    for name in ("FusedRMSNormGated", "causal_conv1d_fn", "causal_conv1d_update",
                 "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"):
        monkeypatch.setattr(hf, name, None)
    config = Qwen3_5TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=1, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=1, linear_num_value_heads=1, linear_key_head_dim=8, linear_value_head_dim=8,
        layer_types=["linear_attention", "full_attention"], use_cache=False, max_position_embeddings=64,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 1.0,
                         "mrope_section": [3, 3, 2], "mrope_interleaved": True})
    config._attn_implementation = "eager"
    model = hf.Qwen3_5TextModel(config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("include_layernorms", [False, True])
def test_real_mixed_layers_three_adam_steps_preserve_loss_inputs_weights_and_moments(monkeypatch, bf16, include_layernorms):
    torch.manual_seed(729)
    baseline = _model(monkeypatch)
    candidate = copy.deepcopy(baseline)
    left_optim = torch.optim.AdamW(baseline.parameters(), **ADAM_OPTIONS)
    right_optim = torch.optim.AdamW(candidate.parameters(), **ADAM_OPTIONS)
    initial = {name: parameter.detach().clone() for name, parameter in baseline.named_parameters()}
    generator = torch.Generator().manual_seed(9501)
    linear_shapes = [{}, {}]
    for model, shapes in zip((baseline, candidate), linear_shapes):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) or type(module).__name__ == "Qwen3_5RMSNorm":
                def capture(_module, values, *, name=name, shapes=shapes):
                    shapes.setdefault(name, set()).add(tuple(values[0].shape))
                module.register_forward_pre_hook(capture)
    with qwen3_5_sublayer_checkpointing(candidate, True, include_layernorms=include_layernorms) as names:
        assert names == (["layers.0.input_layernorm", "layers.0.linear_attn", "layers.0.post_attention_layernorm", "layers.0.mlp",
                          "layers.1.input_layernorm", "layers.1.self_attn", "layers.1.post_attention_layernorm", "layers.1.mlp"]
                         if include_layernorms else ["layers.0.linear_attn", "layers.0.mlp", "layers.1.self_attn", "layers.1.mlp"])
        for _ in range(3):
            inputs = torch.randn(1, 7, 16, generator=generator)
            results = []
            for model, optimizer in ((baseline, left_optim), (candidate, right_optim)):
                optimizer.zero_grad(set_to_none=True)
                before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                x = inputs.clone().requires_grad_()
                context = torch.autocast("cpu", dtype=torch.bfloat16) if bf16 else nullcontext()
                with context:
                    hidden = model(inputs_embeds=x, use_cache=False).last_hidden_state
                    hidden.retain_grad()
                    loss = hidden.float().square().mean()
                loss.backward()
                gradients = {name: parameter.grad.clone() if parameter.grad is not None else None
                             for name, parameter in model.named_parameters()}
                optimizer.step()
                updates = {name: parameter.detach() - before[name] for name, parameter in model.named_parameters()}
                results.append((loss.detach(), x.grad.clone(), hidden.grad.clone(), gradients, updates))
            expected, actual = results
            torch.testing.assert_close(actual[0], expected[0], rtol=0.002, atol=0.002)
            assert compare_tensors(actual[1], expected[1])["passed"]
            assert compare_tensors(actual[2], expected[2])["passed"]
            for (name, left), (_, right) in zip(baseline.named_parameters(), candidate.named_parameters()):
                if expected[3][name] is None:
                    assert actual[3][name] is None
                    continue
                assert compare_tensors(actual[3][name], expected[3][name])["passed"], name
                assert compare_tensors(right, left)["passed"], name
                assert compare_tensors(actual[4][name], expected[4][name])["passed"], (name, "step_update")
                assert compare_tensors(right - initial[name], left - initial[name])["passed"], (name, "cumulative_update")
                for key in ("exp_avg", "exp_avg_sq"):
                    assert compare_tensors(right_optim.state[right][key], left_optim.state[left][key])["passed"], (name, key)
            assert linear_shapes[0] == linear_shapes[1]


def test_default_is_noop_and_enabled_restores_exact_instance_attributes(monkeypatch):
    model = _model(monkeypatch)
    before = {id(module): module.__dict__.get("forward") for module in model.modules()}
    with qwen3_5_sublayer_checkpointing(model) as names:
        assert names == []
        assert before == {id(module): module.__dict__.get("forward") for module in model.modules()}
    with pytest.raises(RuntimeError, match="audit failure"):
        with qwen3_5_sublayer_checkpointing(model, True):
            raise RuntimeError("audit failure")
    assert before == {id(module): module.__dict__.get("forward") for module in model.modules()}


def test_kwargs_tuple_outputs_eval_and_no_grad_keep_native_calls(monkeypatch):
    import legacy.sft.verl.checkpointing as implementation

    model = _model(monkeypatch)
    attention = model.layers[1].self_attn
    attention.config._attn_implementation = "sdpa"
    hidden = torch.randn(1, 5, 16, requires_grad=True)
    positions = (torch.ones(1, 5, 16), torch.zeros(1, 5, 16))
    options = {"hidden_states": hidden, "position_embeddings": positions, "attention_mask": None,
               "past_key_values": None}
    expected = attention(**options)
    calls = []
    native = implementation.checkpoint

    def observed(*args, **kwargs):
        calls.append(kwargs)
        return native(*args, **kwargs)

    monkeypatch.setattr(implementation, "checkpoint", observed)
    with qwen3_5_sublayer_checkpointing(model, True):
        actual = attention(**options)
        assert isinstance(actual, tuple) and len(actual) == 2
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        assert actual[1] is None and expected[1] is None
        assert calls and calls[-1]["use_reentrant"] is False
        calls.clear()
        with torch.no_grad():
            attention(**options)
        assert not calls
        model.eval()
        attention(**options)
        assert not calls


@pytest.mark.parametrize("include_layernorms", [False, True])
def test_keyword_and_nested_tensor_inputs_are_saved_without_closure_retention(monkeypatch, include_layernorms):
    model = _model(monkeypatch)
    attention = model.layers[1].self_attn
    attention.config._attn_implementation = "sdpa"
    leaf = torch.randn(1, 5, 16, requires_grad=True)
    hidden = leaf * 1.1
    cosine = torch.ones(1, 5, 16)
    sine = torch.zeros(1, 5, 16)
    inputs = (hidden, cosine, sine)
    addresses = {tensor.data_ptr() for tensor in inputs}
    references = [weakref.ref(tensor) for tensor in inputs]
    packed = set()

    def pack(tensor):
        if tensor.data_ptr() in addresses:
            packed.add(tensor.data_ptr())
        # CPU clone stands in for the GPU-to-CPU copy without retaining its source.
        return tensor.detach().clone()

    with qwen3_5_sublayer_checkpointing(model, True, include_layernorms=include_layernorms):
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = attention(hidden_states=hidden, position_embeddings=(cosine, sine),
                               attention_mask=None, past_key_values=None)[0]
        assert packed == addresses
        del inputs, hidden, cosine, sine
        gc.collect()
        assert all(reference() is None for reference in references)
        output.float().square().mean().backward()
    assert leaf.grad is not None and torch.isfinite(leaf.grad).all()


@pytest.mark.parametrize("include_layernorms", [False, True])
def test_keyword_boundary_is_owned_by_outer_checkpoint_not_python_closure(monkeypatch, include_layernorms):
    model = _model(monkeypatch)
    references = []

    def observe(_module, _args, kwargs):
        references.append(weakref.ref(kwargs["hidden_states"]))

    model.layers[1].self_attn.register_forward_pre_hook(observe, with_kwargs=True)
    inputs = torch.randn(1, 5, 16, requires_grad=True)
    with qwen3_5_sublayer_checkpointing(model, True, include_layernorms=include_layernorms), torch.autograd.graph.save_on_cpu(pin_memory=False):
        hidden = model(inputs_embeds=inputs, use_cache=False).last_hidden_state
        gc.collect()
        assert len(references) == 1 and references[0]() is None
        hidden.float().square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize("bf16", [False, True])
def test_actual_norm_saves_only_boundary_with_same_shape_output_and_gradients(monkeypatch, bf16):
    model = _model(monkeypatch)
    norm = model.layers[1].post_attention_layernorm
    native = copy.deepcopy(norm)
    dtype = torch.bfloat16 if bf16 else torch.float32
    inputs = torch.randn(1, 5, 16, dtype=dtype, requires_grad=True)
    expected_inputs = inputs.detach().clone().requires_grad_()
    gradient = torch.randn_like(inputs)
    expected = native(expected_inputs)
    expected.backward(gradient)
    saved = []

    def pack(tensor):
        saved.append((tuple(tensor.shape), tensor.dtype))
        return tensor.detach().clone()

    with qwen3_5_sublayer_checkpointing(model, True, include_layernorms=True):
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            actual = norm(inputs)
        actual.backward(gradient)
    assert saved == [((0,), torch.float32), (tuple(inputs.shape), dtype)]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(inputs.grad, expected_inputs.grad, rtol=0, atol=0)
    torch.testing.assert_close(norm.weight.grad, native.weight.grad, rtol=0, atol=0)


def test_rejects_mutable_caches_nested_enable_and_token_chunks(monkeypatch):
    from legacy.sft.verl.mlp import qwen3_5_mlp_token_chunking

    model = _model(monkeypatch)
    with qwen3_5_sublayer_checkpointing(model, True):
        with pytest.raises(ValueError, match="already enabled"):
            with qwen3_5_sublayer_checkpointing(model, True):
                pass
        with pytest.raises(ValueError, match="mutable attention caches"):
            model.layers[1].self_attn(hidden_states=torch.randn(1, 5, 16), past_key_values=object())
    with qwen3_5_mlp_token_chunking(model, 3):
        with pytest.raises(ValueError, match="token chunking"):
            with qwen3_5_sublayer_checkpointing(model, True):
                pass
    with pytest.raises(ValueError, match="actual Qwen3_5DecoderLayer"):
        with qwen3_5_sublayer_checkpointing(torch.nn.Linear(2, 2), True):
            pass


def test_nested_checkpoint_hides_attention_saved_storage_until_its_own_backward(monkeypatch):
    import torch.utils.checkpoint as implementation

    active_frames = []
    native_hook = implementation._recomputation_hook

    def observed_hook(frame_ref, gid):
        active_frames.append((frame_ref, gid))
        return native_hook(frame_ref, gid)

    monkeypatch.setattr(implementation, "_recomputation_hook", observed_hook)

    class HeavyAttention(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            large = x.repeat(1, 64)
            ctx.save_for_backward(large)
            return x * 1.1

        @staticmethod
        def backward(ctx, gradient):
            large, = ctx.saved_tensors
            return gradient * 1.1 + large.sum() * 0

    def run(nested):
        active_frames.clear()
        observed = []
        x = torch.ones(2, 8, requires_grad=True)

        def attention(values):
            return HeavyAttention.apply(values)

        def mlp(values):
            tensors = [tensor for ref, gid in active_frames if (frame := ref()) is not None
                       for tensor in list(frame.recomputed[gid].values())]
            observed.append(sum(tensor.numel() * tensor.element_size() for tensor in tensors if tensor.numel() == 1024))
            return values.sin()

        def layer(values):
            if nested:
                mixed = checkpoint(attention, values, use_reentrant=False)
                return checkpoint(mlp, mixed, use_reentrant=False)
            return mlp(attention(values))

        checkpoint(layer, x, use_reentrant=False).sum().backward()
        return max(observed), x.grad

    baseline_bytes, baseline_gradient = run(False)
    nested_bytes, nested_gradient = run(True)
    assert baseline_bytes == 4096 and nested_bytes == 0
    torch.testing.assert_close(baseline_gradient, nested_gradient, rtol=0, atol=0)
