"""Check scoped text-MLP chunk precision, Adam updates and activation lifetimes on CPU."""

from collections import Counter
from contextlib import nullcontext
import copy
from functools import partial
import json
import math
from types import SimpleNamespace
import weakref

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

from legacy.sft.verl.mlp import qwen3_5_mlp_token_chunking


ADAM = {"lr": 3e-4, "betas": (0.9, 0.95), "eps": 1e-8,
        "weight_decay": 0.01, "foreach": False, "fused": False}


@pytest.fixture(autouse=True)
def _cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def _mlp(hidden=16, intermediate=40):
    return Qwen3_5MLP(SimpleNamespace(hidden_size=hidden, hidden_act="silu"), intermediate)


class _HFLayer(GradientCheckpointingLayer):
    def __init__(self, outer_checkpoint):
        super().__init__()
        self.mlp = _mlp()
        self.gradient_checkpointing = outer_checkpoint
        self._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False, preserve_rng_state=False)

    def forward(self, x):
        return self.mlp(x)


def _compare(actual, expected, dtype, *, scalar=False, enforce=True):
    actual, expected = actual.detach().double(), expected.detach().double()
    difference = actual - expected
    relative = float(difference.norm() / expected.norm().clamp_min(1e-300))
    normalized = float(difference.abs().max() / expected.abs().max().clamp_min(1e-300))
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if scalar:
        passed = bool(torch.allclose(actual, expected, rtol=0.002 if dtype == torch.bfloat16 else 1e-5,
                                     atol=0.002 if dtype == torch.bfloat16 else 1e-7))
    else:
        passed = (relative <= (0.02 if dtype == torch.bfloat16 else 1e-4)
                  and normalized <= (0.04 if dtype == torch.bfloat16 else 3e-4))
    if enforce:
        assert finite and passed, (relative, normalized)
    return {"passed": finite and passed, "relative_l2": relative, "max_normalized": normalized,
            "max_absolute": float(difference.abs().max())}


def _masked_loss(output, labels, mask, head):
    logits = F.linear(output.float(), head)
    targets = labels[:, 1:].masked_fill(~mask[:, 1:], -100)
    return F.cross_entropy(logits[:, :-1].reshape(-1, head.shape[0]), targets.reshape(-1),
                           reduction="sum") / mask.sum()


@pytest.mark.parametrize("chunk_tokens,dtype,qualified", [
    (128, torch.float32, True), (128, torch.bfloat16, False),
    (1024, torch.float32, False), (1024, torch.bfloat16, False),
    (8192, torch.float32, True), (8192, torch.bfloat16, True),
])
@pytest.mark.parametrize("outer_checkpoint", [False, True])
def test_chunked_outputs_gradients_and_three_adam_steps(dtype, chunk_tokens, qualified, outer_checkpoint):
    """Keep known small-chunk Adam failures visible under unchanged qualification gates."""
    torch.manual_seed(19)
    reference = _HFLayer(outer_checkpoint).train()
    candidate = copy.deepcopy(reference)
    optimizers = [torch.optim.AdamW(model.parameters(), **ADAM) for model in (reference, candidate)]
    # Each selected size has an incomplete final chunk and crosses a sample boundary.
    sequence = chunk_tokens + 1 if chunk_tokens < 8192 else 4107
    head = torch.randn(37, 16, generator=torch.Generator().manual_seed(271))
    worst = {"relative_l2": 0.0, "max_normalized": 0.0, "max_absolute": 0.0}
    failures = []
    with qwen3_5_mlp_token_chunking(candidate, chunk_tokens) as names:
        assert names == ["mlp"]
        for step in range(3):
            generator = torch.Generator().manual_seed(5913 + step * 11)
            values = torch.randn(2, sequence, 16, generator=generator).to(dtype)
            inputs = [values.clone().requires_grad_() for _ in range(2)]
            labels = torch.randint(37, (2, sequence), generator=generator)
            mask = ((torch.arange(sequence)[None, :] + torch.arange(2)[:, None] + step) % 3 != 0)
            mask[:, 0] = False
            mask.reshape(-1)[chunk_tokens - 1:chunk_tokens + 2] = False
            outputs, losses = [], []
            for index, (model, optimizer, tensor) in enumerate(zip((reference, candidate), optimizers, inputs, strict=True)):
                optimizer.zero_grad(set_to_none=True)
                activation_context = torch.autograd.graph.save_on_cpu(pin_memory=False) if index else nullcontext()
                autocast = torch.autocast("cpu", dtype=torch.bfloat16) if dtype == torch.bfloat16 else nullcontext()
                with activation_context, autocast:
                    output = model(tensor)
                loss = _masked_loss(output, labels, mask, head)
                loss.backward()
                outputs.append(output)
                losses.append(loss)
                hidden_mask = torch.cat((mask[:, 1:], torch.zeros(2, 1, dtype=torch.bool)), dim=1)
                assert torch.count_nonzero(tensor.grad[~hidden_mask]) == 0
            metrics = [("output", _compare(outputs[1], outputs[0], dtype, enforce=False)),
                       ("loss", _compare(losses[1], losses[0], dtype, scalar=True, enforce=False)),
                       ("dInput", _compare(inputs[1].grad, inputs[0].grad, dtype, enforce=False))]
            before = [[parameter.detach().clone() for parameter in model.parameters()] for model in (reference, candidate)]
            for (name, actual), (_, expected) in zip(candidate.named_parameters(), reference.named_parameters(), strict=True):
                assert actual.dtype == expected.dtype == torch.float32
                metrics.append((f"dW/{name}", _compare(actual.grad, expected.grad, dtype, enforce=False)))
            for optimizer in optimizers:
                optimizer.step()
            for index, ((name, actual), (_, expected)) in enumerate(zip(candidate.named_parameters(), reference.named_parameters(), strict=True)):
                metrics.append((f"Adam_delta/{name}", _compare(actual - before[1][index], expected - before[0][index], dtype, enforce=False)))
                for key in ("exp_avg", "exp_avg_sq"):
                    metrics.append((f"Adam_{key}/{name}", _compare(optimizers[1].state[actual][key], optimizers[0].state[expected][key], dtype, enforce=False)))
            for name, metric in metrics:
                for key in worst:
                    worst[key] = max(worst[key], metric[key])
                if not metric["passed"]:
                    failures.append({"step": step + 1, "field": name, **metric})
    assert outputs[0].dtype == outputs[1].dtype == dtype
    print(json.dumps({"dtype": str(dtype), "chunk_tokens": chunk_tokens,
                      "outer_nonreentrant_checkpoint": outer_checkpoint, "adam_steps": 3,
                      "worst": worst, "failures": failures, "qualified": not failures,
                      "expected_qualification": qualified}))
    assert (not failures) is qualified, failures
    if not qualified:
        assert failures and all(failure["field"].startswith("Adam_delta/") for failure in failures)


def test_nested_checkpoint_releases_and_recomputes_chunk_intermediates():
    """Weak references prove gate/up outputs are bounded by one chunk and then released."""
    torch.manual_seed(19)
    layer = _HFLayer(True).train()
    values = torch.randn(2, 257, 16, requires_grad=True)
    references, events, handles = [], [], []
    phase = "forward"

    def capture(name):
        def hook(_module, _args, output):
            references.append(weakref.ref(output))
            live_elements = sum(reference().numel() for reference in references if reference() is not None)
            events.append((phase, name, output.numel(), live_elements))
        return hook

    for name in ("gate_proj", "up_proj"):
        handles.append(getattr(layer.mlp, name).register_forward_hook(capture(name)))
    handles.append(layer.mlp.down_proj.register_forward_pre_hook(
        lambda module, args: capture("product")(module, args, args[0])
    ))
    try:
        with qwen3_5_mlp_token_chunking(layer, 128), torch.autograd.graph.save_on_cpu(pin_memory=False):
            output = layer(values)
            assert all(reference() is None for reference in references)
            phase = "backward"
            output.square().mean().backward()
        counts = Counter((event[0], event[1]) for event in events)
        chunks = math.ceil(values.numel() / values.shape[-1] / 128)
        assert counts["forward", "gate_proj"] == counts["forward", "up_proj"] == chunks
        assert counts["forward", "product"] == chunks
        assert counts["backward", "gate_proj"] >= chunks
        assert counts["backward", "up_proj"] >= chunks
        assert max(event[2] for event in events) == 128 * 40
        assert max(event[3] for event in events) <= 3 * 128 * 40
        assert all(reference() is None for reference in references)
        assert values.grad is not None
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_single_whole_mlp_chunk_still_checkpoints_and_preserves_adam(dtype):
    """The whole-MLP candidate retains native GEMM sizes and still recomputes."""
    torch.manual_seed(19)
    reference = _HFLayer(True).train()
    candidate = copy.deepcopy(reference)
    optimizers = [torch.optim.AdamW(model.parameters(), **ADAM) for model in (reference, candidate)]
    calls = []
    handle = candidate.mlp.up_proj.register_forward_hook(lambda _module, args, _out: calls.append(args[0].shape[0]))
    try:
        with qwen3_5_mlp_token_chunking(candidate, 8192):
            for step in range(3):
                generator = torch.Generator().manual_seed(5913 + step * 11)
                values = torch.randn(2, 129, 16, generator=generator).to(dtype)
                inputs = [values.clone().requires_grad_() for _ in range(2)]
                outputs = []
                for index, (model, optimizer, tensor) in enumerate(zip((reference, candidate), optimizers, inputs, strict=True)):
                    optimizer.zero_grad(set_to_none=True)
                    autocast = torch.autocast("cpu", dtype=torch.bfloat16) if dtype == torch.bfloat16 else nullcontext()
                    with torch.autograd.graph.save_on_cpu(pin_memory=False), autocast:
                        output = model(tensor)
                    output.float().square().mean().backward()
                    outputs.append(output)
                torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
                torch.testing.assert_close(inputs[1].grad, inputs[0].grad, rtol=0, atol=0)
                for actual, expected in zip(candidate.parameters(), reference.parameters(), strict=True):
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
                for optimizer in optimizers:
                    optimizer.step()
                for actual, expected in zip(candidate.parameters(), reference.parameters(), strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    for key in ("exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(optimizers[1].state[actual][key], optimizers[0].state[expected][key], rtol=0, atol=0)
        assert len(calls) >= 6 and set(calls) == {258}
    finally:
        handle.remove()


def test_no_grad_chunks_preserve_shape_dtype_and_skip_checkpoint(monkeypatch):
    """Inference uses the same token math without creating checkpoint frames."""
    import legacy.sft.verl.mlp as implementation

    model = _mlp()
    values = torch.randn(3, 5, 16).transpose(0, 1)
    calls = []
    handle = model.up_proj.register_forward_hook(lambda _module, args, _out: calls.append(args[0].shape[0]))
    try:
        with torch.no_grad():
            expected = model(values)
            calls.clear()
            monkeypatch.setattr(implementation, "checkpoint", lambda *args, **kwargs: pytest.fail("no_grad checkpoint"))
            with qwen3_5_mlp_token_chunking(model, 4) as names:
                actual = model(x=values)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
        assert names == [""] and calls == [4, 4, 4, 3]
        assert actual.shape == values.shape and actual.dtype == values.dtype
    finally:
        handle.remove()


def test_scope_restores_class_and_instance_forward_after_failure():
    """Only real text MLPs are patched, and existing instance overrides survive exceptions."""
    first, second = _mlp(), _mlp()
    original = second.forward
    second.forward = original
    unrelated = torch.nn.Linear(16, 16)
    model = torch.nn.ModuleList([first, second, unrelated])
    first_forward, other_forward = first.forward, unrelated.forward
    with pytest.raises(RuntimeError, match="abort"):
        with qwen3_5_mlp_token_chunking(model, 128) as names:
            assert names == ["0", "1"]
            assert first.forward != first_forward and second.forward != original
            assert first.forward._qwenpt_mlp_chunk_tokens == 128
            assert unrelated.forward == other_forward
            raise RuntimeError("abort")
    assert "forward" not in first.__dict__ and first.forward == first_forward
    assert second.__dict__["forward"] is original and unrelated.forward == other_forward


def test_disabled_scope_is_noop_and_enabled_scope_requires_real_text_mlp():
    """An enabled candidate cannot silently miss its target architecture."""
    unrelated = torch.nn.Linear(3, 4)
    native = unrelated.forward
    with qwen3_5_mlp_token_chunking(unrelated) as names:
        assert names == [] and unrelated.forward == native and "forward" not in unrelated.__dict__
    with pytest.raises(ValueError, match="Qwen3_5MLP"):
        with qwen3_5_mlp_token_chunking(unrelated, 128):
            pass
    with pytest.raises(ValueError, match="non-negative"):
        with qwen3_5_mlp_token_chunking(unrelated, -1):
            pass


def test_scope_rejects_existing_sublayer_checkpoint_and_restores_prior_modules():
    """The two checkpoint candidates cannot silently wrap each other."""
    first, second = _mlp(), _mlp()
    first_forward, second_forward = first.forward, second.forward

    def checkpointed(x):
        return second_forward(x)

    checkpointed._qwenpt_sublayer_checkpointed = True
    second.forward = checkpointed
    model = torch.nn.ModuleList([first, second])
    with pytest.raises(ValueError, match="sublayer-checkpointed"):
        with qwen3_5_mlp_token_chunking(model, 8192):
            pass
    assert first.forward == first_forward and "forward" not in first.__dict__
    assert second.forward is checkpointed
