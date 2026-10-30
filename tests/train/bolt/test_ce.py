"""Check opt-in BOLT output-head numerics, accumulation, and saved activations."""

from inspect import unwrap
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from qwenpt.train.bolt.ce import (
    bolt_logprobs_backend,
    chunked_linear_logprobs,
    chunked_qwen3_5_forward,
)


def _reference(hidden, weight, labels, temperature=1.0):
    logits = F.linear(hidden, weight) / temperature
    if logits.dtype != torch.float64:
        logits = logits.float()
    return -F.cross_entropy(
        logits.reshape(-1, weight.shape[0]), labels.reshape(-1), reduction="none",
    ).reshape(labels.shape)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape,chunk_size", [((13, 7), 1), ((2, 5, 7), 4), ((3, 4, 7), 128)])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_values_and_hidden_gradients_match_dense_reference(dtype, shape, chunk_size, temperature):
    generator = torch.Generator().manual_seed(20261020)
    hidden = torch.randn(shape, dtype=dtype, generator=generator, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    weight = torch.randn(37, shape[-1], dtype=dtype, generator=generator)
    labels = torch.randint(weight.shape[0], shape[:-1], generator=generator)
    upstream = torch.randn(labels.shape, dtype=dtype, generator=generator)
    expected = _reference(hidden, weight, labels, temperature)
    actual = chunked_linear_logprobs(
        other, weight, labels, chunk_size=chunk_size, temperature=temperature,
    )
    tolerance = {"rtol": 2e-12, "atol": 2e-12} if dtype == torch.float64 else {"rtol": 3e-6, "atol": 3e-6}
    torch.testing.assert_close(actual, expected, **tolerance)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    torch.testing.assert_close(other.grad, hidden.grad, **tolerance)
    assert weight.grad is None
    assert actual.shape == labels.shape
    assert actual.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_logits_use_fp32_cross_entropy(dtype):
    generator = torch.Generator().manual_seed(17)
    hidden = torch.randn(5, 8, generator=generator).to(dtype).requires_grad_()
    other = hidden.detach().clone().requires_grad_()
    weight = torch.randn(19, 8, generator=generator).to(dtype)
    labels = torch.randint(19, (5,), generator=generator)
    actual = chunked_linear_logprobs(other, weight, labels, chunk_size=2)
    expected = _reference(hidden, weight, labels)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(other.grad, hidden.grad, rtol=0.01, atol=0.01)


def test_validation_without_gradients_matches_training_forward():
    generator = torch.Generator().manual_seed(11)
    hidden = torch.randn(2, 3, 7, generator=generator, requires_grad=True)
    weight = torch.randn(23, 7, generator=generator)
    labels = torch.randint(23, (2, 3), generator=generator)
    expected = _reference(hidden, weight, labels)
    with torch.no_grad():
        actual = chunked_linear_logprobs(hidden, weight, labels, chunk_size=4)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad
    assert hidden.grad is None
    assert weight.grad is None


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_rejects_invalid_chunk_size(chunk_size):
    with pytest.raises(ValueError):
        chunked_linear_logprobs(torch.zeros(3, 7), torch.zeros(13, 7), torch.zeros(3, dtype=torch.long), chunk_size=chunk_size)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        chunked_linear_logprobs(torch.zeros(3, 7), torch.zeros(13, 7), torch.zeros(3, dtype=torch.long), temperature=temperature)


def test_rejects_trainable_output_weight():
    weight = torch.randn(13, 7, requires_grad=True)
    with pytest.raises(ValueError):
        chunked_linear_logprobs(torch.zeros(3, 7), weight, torch.zeros(3, dtype=torch.long))


@pytest.mark.parametrize("labels", [torch.zeros(3, dtype=torch.float32), torch.zeros(0, dtype=torch.long)])
def test_rejects_invalid_label_tensor(labels):
    with pytest.raises(ValueError):
        chunked_linear_logprobs(torch.zeros(len(labels), 7), torch.zeros(13, 7), labels)


def test_noncontiguous_hidden_states_and_labels_preserve_gradients():
    generator = torch.Generator().manual_seed(71)
    hidden = torch.randn(2, 7, 5, generator=generator).transpose(1, 2).requires_grad_()
    other = hidden.detach().clone().requires_grad_()
    labels = torch.randint(23, (5, 2), generator=generator).t()
    weight = torch.randn(23, 7, generator=generator)
    expected = _reference(hidden, weight, labels)
    actual = chunked_linear_logprobs(other, weight, labels, chunk_size=4)
    torch.testing.assert_close(actual, expected)
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(other.grad, hidden.grad)


@pytest.mark.parametrize("hidden_shape,weight_shape,label_shape", [
    ((2, 3, 7), (13, 7), (6,)),
    ((2, 3, 7), (13, 8), (2, 3)),
    ((2, 3, 7), (2, 13, 7), (2, 3)),
    ((2, 3, 4, 7), (13, 7), (2, 3, 4)),
])
def test_rejects_incompatible_shapes(hidden_shape, weight_shape, label_shape):
    with pytest.raises(ValueError):
        chunked_linear_logprobs(
            torch.zeros(hidden_shape), torch.zeros(weight_shape), torch.zeros(label_shape, dtype=torch.long),
        )


def test_checkpoint_retains_no_vocabulary_sized_activations():
    """A chunk loop alone still saves every CE log-softmax until backward."""
    generator = torch.Generator().manual_seed(21)
    hidden = torch.randn(23, 11, generator=generator, requires_grad=True)
    weight = torch.randn(97, 11, generator=generator)
    labels = torch.randint(97, (23,), generator=generator)
    saved_shapes = []

    def save(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        log_probs = chunked_linear_logprobs(hidden, weight, labels, chunk_size=7)
    assert saved_shapes
    assert not any(len(shape) >= 2 and shape[-1] == weight.shape[0] for shape in saved_shapes), saved_shapes
    log_probs.sum().backward()
    assert torch.isfinite(hidden.grad).all()
    assert torch.count_nonzero(hidden.grad) > 0


def _bolt_batch(log_probs, masks, weights, *, global_batch_size, dp_size):
    from verl.utils.dataset.dataset_utils import SFTTensorCollator
    from verl.utils.tensordict_utils import get_tensordict

    items = [
        {"input_ids": torch.arange(len(mask)), "loss_mask": mask, "bolt_weight": weight.reshape(1)}
        for mask, weight in zip(masks, weights, strict=True)
    ]
    batch = SFTTensorCollator("no_padding")(items)
    data = get_tensordict(batch, non_tensor_dict={
        "pad_mode": "no_padding", "dp_size": dp_size, "global_batch_size": global_batch_size,
    })
    values = torch.cat([row[:len(mask)] for row, mask in zip(log_probs, masks, strict=True)])
    nested = torch.nested.nested_tensor_from_jagged(values, data["loss_mask"].offsets())
    return {"log_probs": nested}, data


def test_microbatch2_accumulation8_and_dp2_preserve_weighted_bolt_objective():
    """Exercise all 32 rows, variable lengths, padding, masks, and causal boundaries."""
    from qwenpt.train.bolt.trainer import bolt_loss

    generator = torch.Generator().manual_seed(20261020)
    lengths = [3 + index % 5 for index in range(32)]
    hidden = torch.randn(32, max(lengths), 9, generator=generator, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    weight = torch.randn(29, 9, generator=generator)
    ids = torch.randint(29, hidden.shape[:-1], generator=generator)
    masks = [torch.tensor(([index % 2, 0] + [1] * (length - 2)), dtype=torch.long)
             for index, length in enumerate(lengths)]
    weights = torch.tensor([0.0, 0.125, 2.0, 5.875] * 8)

    packed_hidden = torch.cat([row[:length] for row, length in zip(hidden, lengths, strict=True)])
    packed_labels = torch.cat([row[:length] for row, length in zip(ids, lengths, strict=True)]).roll(-1)
    reference_values = _reference(packed_hidden, weight, packed_labels)
    reference_rows = list(reference_values.split(lengths))
    expected, _ = bolt_loss(*_bolt_batch(
        reference_rows, masks, weights, global_batch_size=32, dp_size=1,
    ))
    expected.backward()

    rank_losses = []
    for rank in range(2):
        rank_loss = torch.zeros(())
        for accumulation in range(8):
            start = rank * 16 + accumulation * 2
            indices = slice(start, start + 2)
            local_ids = ids[indices]
            log_probs = chunked_linear_logprobs(
                other[indices], weight, local_ids.roll(-1, dims=-1), chunk_size=3,
            )
            loss, _ = bolt_loss(*_bolt_batch(
                log_probs, masks[indices], weights[indices], global_batch_size=32, dp_size=2,
            ))
            rank_loss += loss.detach()
            # FSDP averages the two ranks after their eight backward passes.
            (loss / 2).backward()
        rank_losses.append(rank_loss)

    torch.testing.assert_close(sum(rank_losses) / 2, expected.detach(), rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(other.grad, hidden.grad, rtol=5e-6, atol=5e-7)
    for index, length in enumerate(lengths):
        assert torch.count_nonzero(other.grad[index, length - 1:]) == 0
        if weights[index] == 0:
            assert torch.count_nonzero(other.grad[index]) == 0
    assert weight.grad is None


def _config(**changes):
    config = OmegaConf.create({
        "trainer": {"ce_backend": "chunked_linear", "ce_chunk_size": 7},
        "model": {"use_fused_kernels": True, "use_remove_padding": False,
                  "fused_kernel_options": {"impl_backend": "torch"}},
        "engine": {"strategy": "fsdp2", "ulysses_sequence_parallel_size": 1},
    })
    for key, value in changes.items():
        OmegaConf.update(config, key, value)
    return config


def test_backend_is_opt_in_and_restores_native_forward_after_exception():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    from verl.models.transformers import qwen3_5

    original = qwen3_5.forward_with_torch_backend
    original_class_forward = Qwen3_5ForConditionalGeneration.forward
    with bolt_logprobs_backend(OmegaConf.create({"trainer": {}})):
        assert qwen3_5.forward_with_torch_backend is original
    with pytest.raises(RuntimeError, match="intentional"):
        with bolt_logprobs_backend(_config()):
            patched = qwen3_5.forward_with_torch_backend
            assert patched is not original
            assert unwrap(patched) is chunked_qwen3_5_forward
            # verl installs the selected function on the model class during construction.
            Qwen3_5ForConditionalGeneration.forward = patched
            raise RuntimeError("intentional")
    assert qwen3_5.forward_with_torch_backend is original
    assert Qwen3_5ForConditionalGeneration.forward is original_class_forward


def test_native_backend_installer_binds_self_and_preserves_chunk_configuration(monkeypatch):
    """Calling the class-installed forward catches non-descriptor partial wrappers."""
    from transformers.modeling_outputs import BaseModelOutput
    from verl.models.transformers.monkey_patch import patch_forward_with_backends
    from qwenpt.train.bolt import ce as bolt_ce

    generator = torch.Generator().manual_seed(81)
    hidden = torch.randn(2, 5, 7, generator=generator, requires_grad=True)

    class Decoder(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            return BaseModelOutput(last_hidden_state=hidden, hidden_states=(hidden,))

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="qwen3_5")
            self.model = Decoder()
            self.lm_head = torch.nn.Linear(7, 19, bias=False)
            self.lm_head.requires_grad_(False)

        def forward(self, **kwargs):
            raise AssertionError("native backend installer did not install forward")

    model = DummyModel()
    ids = torch.randint(19, (2, 5), generator=generator)
    expected = _reference(hidden, model.lm_head.weight, ids.roll(-1, dims=-1))
    observed_chunks = []
    native_helper = bolt_ce.chunked_linear_logprobs

    def capture_chunk_size(*args, **kwargs):
        observed_chunks.append(kwargs["chunk_size"])
        return native_helper(*args, **kwargs)

    monkeypatch.setattr(bolt_ce, "chunked_linear_logprobs", capture_chunk_size)
    with bolt_logprobs_backend(_config(**{"trainer.ce_chunk_size": 3})):
        patch_forward_with_backends(model, True, "torch")
        actual = model(input_ids=ids)
    assert observed_chunks == [3]
    torch.testing.assert_close(actual.log_probs, expected)
    actual.log_probs.sum().backward()
    assert torch.isfinite(hidden.grad).all()
    assert torch.count_nonzero(hidden.grad) > 0


@pytest.mark.parametrize("key,value", [
    ("trainer.ce_backend", "unknown"),
    ("trainer.ce_chunk_size", 0),
    ("model.use_fused_kernels", False),
    ("model.use_remove_padding", True),
    ("model.fused_kernel_options.impl_backend", "triton"),
    ("engine.strategy", "fsdp"),
    ("engine.strategy", "megatron"),
    ("engine.ulysses_sequence_parallel_size", 2),
])
def test_backend_rejects_incompatible_configuration_before_patch(key, value):
    from verl.models.transformers import qwen3_5

    original = qwen3_5.forward_with_torch_backend
    with pytest.raises(ValueError):
        with bolt_logprobs_backend(_config(**{key: value})):
            pytest.fail("incompatible backend entered")
    assert qwen3_5.forward_with_torch_backend is original


def _model_stub(hidden):
    from transformers.modeling_outputs import BaseModelOutput

    head = torch.nn.Linear(hidden.shape[-1], 19, bias=False)
    head.requires_grad_(False)
    return SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_5"), lm_head=head,
        model=lambda input_ids, **kwargs: BaseModelOutput(last_hidden_state=hidden, hidden_states=(hidden,)),
    )


@pytest.mark.parametrize("source", ["input_ids", "labels", "shift_labels"])
def test_qwen_forward_preserves_label_source_and_hidden_gradients(source):
    generator = torch.Generator().manual_seed(9)
    hidden = torch.randn(2, 5, 7, generator=generator, requires_grad=True)
    model = _model_stub(hidden)
    ids = torch.randint(19, (2, 5), generator=generator)
    labels = torch.randint(19, (2, 5), generator=generator)
    kwargs = {"input_ids": ids}
    if source != "input_ids":
        kwargs[source] = labels
    expected_labels = labels if source == "shift_labels" else (labels if source == "labels" else ids).roll(-1, dims=-1)
    expected = _reference(hidden, model.lm_head.weight, expected_labels)
    actual = chunked_qwen3_5_forward(model, **kwargs, chunk_size=3)
    torch.testing.assert_close(actual.log_probs, expected)
    assert actual.logits is None
    assert actual.entropy is None
    actual.log_probs.sum().backward()
    assert torch.isfinite(hidden.grad).all()
    assert torch.count_nonzero(hidden.grad) > 0
    assert model.lm_head.weight.grad is None


@pytest.mark.parametrize("incompatibility", ["architecture", "trainable_head", "bias", "lora", "packed"])
def test_qwen_forward_rejects_model_changes_that_would_alter_objective(incompatibility):
    hidden = torch.zeros(2, 3, 7)
    model = _model_stub(hidden)
    kwargs = {"input_ids": torch.zeros(2, 3, dtype=torch.long)}
    if incompatibility == "architecture":
        model.config.model_type = "qwen3_5_moe"
    elif incompatibility == "trainable_head":
        model.lm_head.weight.requires_grad_()
    elif incompatibility == "bias":
        model.lm_head.bias = torch.nn.Parameter(torch.zeros(19), requires_grad=False)
    elif incompatibility == "lora":
        model.lm_head.lora_A = torch.nn.Identity()
    else:
        kwargs["cu_seqlens"] = torch.tensor([0, 3, 6])
    with pytest.raises(ValueError):
        chunked_qwen3_5_forward(model, **kwargs)
