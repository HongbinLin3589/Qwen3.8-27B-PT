"""Check batched CE against the current CUDA-autocast numerical semantics on CPU."""

import pytest
import torch
import torch.nn.functional as F

from legacy.sft.verl.ce import logprobs_from_logits_chunked, sft_logprobs_backend


def _rowwise_reference(logits, labels):
    """Mirror native BF16 row processing with CUDA's FP32 log-softmax promotion."""
    if logits.dtype in (torch.float32, torch.float64):
        from verl.utils.torch_functional import logprobs_from_logits_v2

        return logprobs_from_logits_v2(logits, labels)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    return torch.stack([
        F.log_softmax(row.float(), dim=-1).gather(0, label.reshape(1)).squeeze(0)
        for row, label in zip(flat_logits, flat_labels, strict=True)
    ]).reshape(labels.shape)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("shape,chunk_size", [((13, 257), 4), ((2, 3, 129), 32), ((5, 17), 1)])
def test_chunked_values_and_gradients_match_rowwise(dtype, shape, chunk_size):
    """Nonuniform upstream gradients cover partial chunks and batched input shapes."""
    generator = torch.Generator().manual_seed(20261020)
    logits = (torch.randn(shape, generator=generator) * 5).to(dtype).requires_grad_()
    other = logits.detach().clone().requires_grad_()
    labels = torch.randint(shape[-1], shape[:-1], generator=generator)
    weights = torch.randn(shape[:-1], generator=generator)
    expected = _rowwise_reference(logits, labels)
    actual = logprobs_from_logits_chunked(other, labels, chunk_size=chunk_size)
    torch.testing.assert_close(actual, expected)
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    torch.testing.assert_close(other.grad, logits.grad)
    torch.testing.assert_close(other.detach(), logits.detach(), rtol=0, atol=0)
    assert actual.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)


def test_chunked_full_native_sft_loss_preserves_mask_and_global_normalization():
    """Exercise actual native SFT loss with jagged masks and a global token count."""
    from tensordict import TensorDict
    from tensordict.tensorclass import NonTensorData
    from verl.workers.utils.losses import sft_loss

    generator = torch.Generator().manual_seed(19)
    logits = torch.randn(13, 127, generator=generator, requires_grad=True)
    other = logits.detach().clone().requires_grad_()
    labels = torch.randint(127, (13,), generator=generator)
    offsets = torch.tensor([0, 5, 13])
    mask = torch.tensor([0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1])
    data = TensorDict({
        "loss_mask": torch.nested.nested_tensor_from_jagged(mask, offsets),
        "pad_mode": NonTensorData("no_padding"),
        "dp_size": 2,
        "batch_num_tokens": 21,
    }, batch_size=[])
    expected_logprobs = _rowwise_reference(logits, labels)
    actual_logprobs = logprobs_from_logits_chunked(other, labels, chunk_size=4)
    expected, _ = sft_loss(None, {
        "log_probs": torch.nested.nested_tensor_from_jagged(expected_logprobs, offsets),
    }, data)
    actual, metrics = sft_loss(None, {
        "log_probs": torch.nested.nested_tensor_from_jagged(actual_logprobs, offsets),
    }, data)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, -(actual_logprobs * mask.roll(-1)).sum() / 21 * 2)
    assert metrics == {}
    actual.backward()
    expected.backward()
    torch.testing.assert_close(other.grad, logits.grad)
    assert torch.count_nonzero(other.grad[mask.roll(-1) == 0]) == 0


def test_chunked_ce_remains_stable_for_large_logits():
    """FP32 reductions avoid BF16 cancellation and remain finite for large shifts."""
    logits = torch.tensor([[1000, 1000, -1000], [-1000, 1000, 1000]], dtype=torch.bfloat16)
    labels = torch.tensor([0, 2])
    actual = logprobs_from_logits_chunked(logits, labels, chunk_size=1)
    torch.testing.assert_close(actual, _rowwise_reference(logits, labels))
    assert torch.isfinite(actual).all()


def test_backend_scope_restores_native_symbol_on_exception():
    """Restrict the override to the engine binding and restore it after failure."""
    from verl.utils import torch_functional
    from verl.workers.engine.fsdp import transformer_impl

    original_engine = transformer_impl.logprobs_from_logits
    original_utility = torch_functional.logprobs_from_logits
    with pytest.raises(RuntimeError, match="stop"):
        with sft_logprobs_backend("chunked", chunk_size=7):
            assert transformer_impl.logprobs_from_logits.func is logprobs_from_logits_chunked
            assert transformer_impl.logprobs_from_logits.keywords == {"chunk_size": 7}
            assert torch_functional.logprobs_from_logits is original_utility
            raise RuntimeError("stop")
    assert transformer_impl.logprobs_from_logits is original_engine


def test_native_backend_leaves_binding_unchanged():
    """The opt-in switch keeps the existing path available for comparison."""
    from verl.workers.engine.fsdp import transformer_impl

    original = transformer_impl.logprobs_from_logits
    with sft_logprobs_backend("native"):
        assert transformer_impl.logprobs_from_logits is original


@pytest.mark.parametrize("backend,chunk_size", [("invalid", 256), ("chunked", 0)])
def test_invalid_backend_configuration_fails_before_training(backend, chunk_size):
    """Reject unsupported backends and chunk lengths at context entry."""
    with pytest.raises(ValueError):
        with sft_logprobs_backend(backend, chunk_size):
            pass
