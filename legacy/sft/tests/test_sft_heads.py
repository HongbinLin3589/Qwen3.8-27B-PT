"""Check message masks, next-token shift and global DP loss/gradient equivalence."""

import pytest
import torch
import torch.nn.functional as F

from legacy.sft.verl.sft_heads import prepare_sft_targets, sft_head_loss


def _batch():
    generator = torch.Generator().manual_seed(19)
    hidden = torch.randn(13, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(17, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    ids = torch.randint(17, (13,), generator=generator)
    # A masked historical assistant message remains in context before the answer.
    mask = torch.tensor([0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1])
    offsets = torch.tensor([0, 8, 13])
    return hidden, weight, ids, mask, offsets


def test_matches_per_sample_causal_ce_and_excludes_masked_messages():
    hidden, weight, ids, mask, offsets = _batch()
    targets = prepare_sft_targets(ids, mask, offsets, global_supervised_tokens=int(mask.sum()))
    actual = sft_head_loss(hidden, weight, targets)
    numerator = sum(
        F.cross_entropy(
            F.linear(hidden[start : end - 1], weight),
            ids[start + 1 : end].masked_fill(mask[start + 1 : end] == 0, -100),
            reduction="sum",
        )
        for start, end in zip(offsets[:-1], offsets[1:])
    )
    expected = numerator / mask.sum()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual.backward()
    assert torch.count_nonzero(hidden.grad[~targets.mask]) == 0
    assert torch.count_nonzero(hidden.grad[targets.mask]) > 0
    assert weight.grad is not None and torch.count_nonzero(weight.grad) > 0
    assert not targets.mask[7] and not targets.mask[12]


def test_unequal_dp_ranks_and_microbatches_recover_global_token_mean_gradient():
    hidden, weight, ids, mask, offsets = _batch()
    count = int(mask.sum())
    expected = sft_head_loss(
        hidden, weight, prepare_sft_targets(ids, mask, offsets, global_supervised_tokens=count),
    )
    expected.backward()
    expected_gradient = weight.grad.clone()
    weight.grad = None
    # Each DP rank has a different number of supervised tokens. DDP/FSDP averages
    # rank gradients, while accumulated microbatch losses must share one denominator.
    rank_losses = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        local_targets = prepare_sft_targets(
            ids[start:end], mask[start:end], torch.tensor([0, end - start]),
            global_supervised_tokens=count, dp_size=2,
        )
        rank_losses.append(sft_head_loss(hidden[start:end], weight, local_targets))
    (sum(rank_losses) / 2).backward()
    torch.testing.assert_close(weight.grad, expected_gradient, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(sum(rank_losses) / 2, expected)


def test_all_masked_local_rank_has_zero_finite_loss_and_both_zero_gradients():
    hidden, weight, ids, mask, offsets = _batch()
    targets = prepare_sft_targets(ids, torch.zeros_like(mask), offsets, global_supervised_tokens=7, dp_size=2)
    loss = sft_head_loss(hidden, weight, targets)
    loss.backward()
    assert loss.item() == 0
    assert torch.count_nonzero(hidden.grad) == 0
    assert torch.count_nonzero(weight.grad) == 0


def test_rejects_cross_sample_supervision_instead_of_rolling_it_into_previous_sample():
    _, _, ids, mask, offsets = _batch()
    mask[8] = 1
    with pytest.raises(ValueError, match="first token"):
        prepare_sft_targets(ids, mask, offsets, global_supervised_tokens=7)


def test_rejects_fractional_message_masks():
    _, _, ids, mask, offsets = _batch()
    mask = mask.float()
    mask[2] = 0.5
    with pytest.raises(ValueError, match="binary"):
        prepare_sft_targets(ids, mask, offsets, global_supervised_tokens=7)


@pytest.mark.parametrize("backend", ["verl_torch", "verl_unfused"])
def test_builtin_verl_paths_keep_full_trainable_weight_and_hidden_gradients(backend):
    hidden, weight, ids, mask, offsets = _batch()
    hidden = hidden.detach().float().requires_grad_()
    weight = weight.detach().float().requires_grad_()
    targets = prepare_sft_targets(ids, mask, offsets, global_supervised_tokens=23, dp_size=3)
    native_loss = sft_head_loss(hidden, weight, targets)
    native_loss.backward()
    expected_hidden, expected_weight = hidden.grad.clone(), weight.grad.clone()
    hidden.grad = weight.grad = None
    loss = sft_head_loss(hidden, weight, targets, backend=backend, chunk_size=3)
    loss.backward()
    torch.testing.assert_close(loss, native_loss, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(hidden.grad, expected_hidden, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(weight.grad, expected_weight, rtol=3e-5, atol=3e-6)


def test_qwen_adapter_respects_engine_global_shift_and_packed_arguments(monkeypatch):
    import sys
    from types import SimpleNamespace
    from legacy.sft.verl.sft_heads import fused_qwen3_5_sft_forward, sft_head_logprobs

    hidden = torch.randn(1, 6, 5, dtype=torch.float64, requires_grad=True)
    head = torch.nn.Linear(5, 17, bias=False, dtype=torch.float64)
    captured = {}

    class DecoderOutput(tuple):
        hidden_states = None

    def decoder(input_ids, **kwargs):
        captured.update(kwargs)
        return DecoderOutput((hidden,))

    monkeypatch.setitem(sys.modules, "verl.models.transformers.qwen3_5", SimpleNamespace(
        Qwen3_5CausalLMOutputForPPO=SimpleNamespace,
    ))
    monkeypatch.setitem(sys.modules, "verl.utils.ulysses", SimpleNamespace(
        get_ulysses_sequence_parallel_world_size=lambda: 2,
    ))
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5"), lm_head=head, model=decoder)
    input_ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    globally_shifted = torch.tensor([[8, 9, 10, 11, 12, 13]])
    cu_seqlens = torch.tensor([0, 3, 6])
    output = fused_qwen3_5_sft_forward(
        model, input_ids, shift_labels=globally_shifted, backend="native",
        cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens,
    )
    expected = sft_head_logprobs(hidden, head.weight, globally_shifted, backend="native")
    torch.testing.assert_close(output.log_probs, expected)
    assert captured["cu_seqlens"] is cu_seqlens
    assert captured["cu_seqlens_cpu"] is cu_seqlens
    assert "shift_labels" not in captured
    output.log_probs.sum().backward()
    assert head.weight.grad is not None and torch.count_nonzero(head.weight.grad) > 0
    assert hidden.grad is not None and torch.count_nonzero(hidden.grad) > 0


@pytest.mark.parametrize("shape", [(3, 5), (1, 12)], ids=["dense-batch", "packed-sequence"])
@pytest.mark.parametrize("backend", ["verl_torch", "verl_triton", "liger", "cce_exact"])
def test_fused_module_call_preserves_hooks_shift_shape_gradients_and_forward(monkeypatch, shape, backend):
    import sys
    from types import SimpleNamespace
    from legacy.sft.verl import sft_heads

    generator = torch.Generator().manual_seed(73)
    hidden = torch.randn(*shape, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    head = torch.nn.Linear(7, 19, bias=False, dtype=torch.float64)
    shifted = torch.randint(19, shape, generator=generator)
    upstream = torch.randn(shape, generator=generator, dtype=torch.float64)
    original_forward = head.forward.__func__
    hooks = []
    handles = [head.register_forward_pre_hook(lambda module, args: hooks.append("pre")),
               head.register_forward_hook(lambda module, args, output: hooks.append("post"))]

    def fake_kernel(hidden, weight, labels, *args):
        assert torch.equal(labels, shifted.flatten())
        assert weight is head.weight
        logits = F.linear(hidden, weight)
        return (-F.cross_entropy(logits, labels, reduction="none"),
                -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1))

    class FakeFusedLinear:
        def __init__(self, **kwargs):
            assert kwargs == {"chunk_size": 3, "impl_backend": "torch"}

        def __call__(self, *args):
            return fake_kernel(*args)

    monkeypatch.setitem(sys.modules, "verl.utils.experimental.torch_functional", SimpleNamespace(
        FusedLinearForPPO=FakeFusedLinear,
    ))
    monkeypatch.setitem(sys.modules, "verl.utils.kernel.linear_cross_entropy", SimpleNamespace(
        linear_cross_entropy=fake_kernel,
    ))
    monkeypatch.setattr(sft_heads, "sft_head_logprobs", lambda h, w, y, **kwargs: fake_kernel(h, w, y)[0])
    expected, expected_entropy = fake_kernel(hidden.reshape(-1, 7), head.weight, shifted.flatten())
    expected_loss = (expected * upstream.flatten()).sum()
    if backend in ("verl_torch", "verl_triton"):
        expected_loss = expected_loss + (expected_entropy * upstream.flatten()).sum()
    expected_hidden, expected_weight = torch.autograd.grad(expected_loss, (hidden, head.weight))
    actual, entropy = sft_heads.call_sft_head_module(head, hidden, shifted, backend=backend, chunk_size=3)
    assert actual.shape == shape
    assert hooks == ["pre", "post"]
    assert head.forward.__func__ is original_forward
    torch.testing.assert_close(actual.flatten(), expected, rtol=0, atol=0)
    actual_loss = (actual * upstream).sum()
    if backend in ("verl_torch", "verl_triton"):
        assert entropy.shape == shape
        torch.testing.assert_close(entropy.flatten(), expected_entropy, rtol=0, atol=0)
        actual_loss = actual_loss + (entropy * upstream).sum()
    else:
        assert entropy is None
    actual_loss.backward()
    torch.testing.assert_close(hidden.grad, expected_hidden, rtol=0, atol=0)
    torch.testing.assert_close(head.weight.grad, expected_weight, rtol=0, atol=0)
    for handle in handles:
        handle.remove()


def test_fused_module_forward_restores_after_kernel_exception(monkeypatch):
    from legacy.sft.verl import sft_heads

    head = torch.nn.Linear(5, 17, bias=False)
    original_forward = head.forward.__func__

    def fail(*args, **kwargs):
        raise RuntimeError("kernel failure")

    monkeypatch.setattr(sft_heads, "sft_head_logprobs", fail)
    with pytest.raises(RuntimeError, match="kernel failure"):
        sft_heads.call_sft_head_module(head, torch.randn(1, 3, 5), torch.zeros(1, 3, dtype=torch.long), backend="liger")
    assert head.forward.__func__ is original_forward
    assert "forward" not in vars(head)


@pytest.mark.parametrize("backend", ["verl_torch", "verl_triton", "liger", "cce_exact"])
def test_fused_context_restores_both_forwards_after_exception(monkeypatch, backend):
    import sys
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from legacy.sft.verl.sft_heads import sft_output_head_backend

    class Model:
        def forward(self, input_ids):
            return input_ids

    original_class_forward = Model.forward
    original_torch, original_triton = object(), object()
    module = SimpleNamespace(forward_with_triton_backend=original_triton, forward_with_torch_backend=original_torch)
    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_5.modeling_qwen3_5", SimpleNamespace(
        Qwen3_5ForConditionalGeneration=Model,
    ))
    monkeypatch.setitem(sys.modules, "verl.models.transformers", SimpleNamespace(qwen3_5=module))
    impl = "triton" if backend == "verl_triton" else "torch"
    config = OmegaConf.create({
        "trainer": {"output_head_backend": backend}, "engine": {"strategy": "fsdp2"},
        "model": {"use_fused_kernels": True, "fused_kernel_options": {"impl_backend": impl}},
    })
    with pytest.raises(RuntimeError, match="trainer failure"):
        with sft_output_head_backend(config):
            selected = getattr(module, f"forward_with_{impl}_backend")
            assert selected is not (original_torch if impl == "torch" else original_triton)
            Model.forward = selected
            raise RuntimeError("trainer failure")
    assert Model.forward is original_class_forward
    assert module.forward_with_triton_backend is original_triton
    assert module.forward_with_torch_backend is original_torch


def test_default_backend_remains_native():
    from omegaconf import OmegaConf
    from legacy.sft.verl.sft_heads import BACKENDS, sft_output_head_backend

    assert BACKENDS == ("native", "verl_torch", "verl_triton", "liger", "cce_exact")
    config = OmegaConf.create({"trainer": {}, "model": {"use_fused_kernels": False}})
    with sft_output_head_backend(config):
        assert config.trainer.get("output_head_backend", "native") == "native"
