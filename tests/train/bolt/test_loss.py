"""Verify BOLT's sample weighting and native jagged training contract on CPU."""

import torch
from verl.utils.dataset.dataset_utils import SFTTensorCollator
from verl.utils.tensordict_utils import get_tensordict

from qwenpt.train.bolt.trainer import bolt_loss


def build_batch(log_probs, masks, weights, *, global_batch_size=None, dp_size=1):
    """Collate response tensors and attach native trainer scaling metadata."""
    items = [
        {
            "input_ids": torch.arange(len(values)),
            "loss_mask": torch.tensor(mask, dtype=torch.long),
            "bolt_weight": torch.tensor([weight], dtype=values.dtype),
        }
        for values, mask, weight in zip(log_probs, masks, weights, strict=True)
    ]
    batch = SFTTensorCollator("no_padding")(items)
    data = get_tensordict(batch, non_tensor_dict={
        "pad_mode": "no_padding", "dp_size": dp_size,
        "global_batch_size": global_batch_size or len(items),
    })
    output = {"log_probs": torch.nested.as_nested_tensor(log_probs, layout=torch.jagged)}
    return output, data


def example():
    """Return unequal responses whose sample and token means differ."""
    log_probs = [
        torch.tensor(values, dtype=torch.float32, requires_grad=True)
        for values in [[-1, -3, -10, -99], [-2, -6, -99], [-4, -6, -8, -10, -99]]
    ]
    # The first mask entries intentionally expose accidental wrapping between samples.
    return log_probs, [[1, 1, 0, 1], [1, 1, 1], [1, 1, 0, 1, 1]], [2.0, 0.5, 0.0]


def test_sample_mean_and_causal_boundaries_with_native_collator() -> None:
    log_probs, masks, weights = example()
    output, data = build_batch(log_probs, masks, weights)
    assert data.batch_size == torch.Size([3])
    assert data["bolt_weight"].is_nested
    assert [row.tolist() for row in data["bolt_weight"].unbind()] == [[2.0], [0.5], [0.0]]
    assert data["loss_mask"].offsets().tolist() == [0, 4, 7, 12]

    loss, metrics = bolt_loss(output, data)
    # Row token means are 5.5, 4, and 22/3; weights apply before the sample mean.
    torch.testing.assert_close(loss, torch.tensor(13 / 3, dtype=torch.float32))
    assert isinstance(metrics, dict)
    loss.backward()
    expected_gradients = [[-1 / 3, 0, -1 / 3, 0], [-1 / 12, -1 / 12, 0], [0] * 5]
    for values, expected in zip(log_probs, expected_gradients, strict=True):
        torch.testing.assert_close(values.grad, torch.tensor(expected, dtype=torch.float32))


def test_microbatch_accumulation_matches_whole_batch_loss_and_gradients() -> None:
    log_probs, masks, weights = example()
    whole, _ = bolt_loss(*build_batch(log_probs, masks, weights))
    whole.backward()
    expected_gradients = [values.grad.clone() for values in log_probs]
    for values in log_probs:
        values.grad = None

    accumulated = torch.zeros((), dtype=torch.float32)
    for indices in [slice(0, 1), slice(1, 3)]:
        microbatch, _ = bolt_loss(*build_batch(
            log_probs[indices], masks[indices], weights[indices], global_batch_size=3,
        ))
        accumulated += microbatch.detach()
        microbatch.backward()

    torch.testing.assert_close(accumulated, whole.detach())
    for values, expected in zip(log_probs, expected_gradients, strict=True):
        torch.testing.assert_close(values.grad, expected)


def test_data_parallel_average_preserves_global_sample_mean() -> None:
    log_probs, masks, weights = example()
    whole, _ = bolt_loss(*build_batch(log_probs, masks, weights))
    whole.backward()
    expected_gradients = [values.grad.clone() for values in log_probs]
    for values in log_probs:
        values.grad = None

    rank_losses = [
        bolt_loss(*build_batch(
            log_probs[indices], masks[indices], weights[indices], global_batch_size=3, dp_size=2,
        ))[0]
        for indices in [slice(0, 1), slice(1, 3)]
    ]
    averaged = sum(rank_losses) / 2
    torch.testing.assert_close(averaged, whole.detach())
    averaged.backward()
    for values, expected in zip(log_probs, expected_gradients, strict=True):
        torch.testing.assert_close(values.grad, expected)


def test_tiny_causal_model_backward_lowers_weighted_objective(tokenizer) -> None:
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(7)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_positions=32, n_embd=16, n_layer=1, n_head=2,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
    ))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ids = [torch.tensor(tokenizer.encode(text, add_special_tokens=False)) for text in ["1+1=2", "2+3=5."]]
    masks = [[0] + [1] * (len(row) - 1) for row in ids]
    weights = [2.0, 0.25]

    def objective():
        log_probs, expected = [], []
        for row, mask, weight in zip(ids, masks, weights, strict=True):
            logits = model(row[None]).logits[0]
            values = logits.log_softmax(-1).gather(-1, torch.roll(row, -1)[:, None]).squeeze(-1)
            log_probs.append(values)
            supervised = torch.tensor(mask[1:], dtype=values.dtype)
            expected.append(-weight * (values[:-1] * supervised).sum() / supervised.sum())
        loss, _ = bolt_loss(*build_batch(log_probs, masks, weights))
        torch.testing.assert_close(loss, sum(expected) / len(ids))
        return loss

    before = objective()
    before.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    optimizer.step()
    assert objective() < before.detach()
