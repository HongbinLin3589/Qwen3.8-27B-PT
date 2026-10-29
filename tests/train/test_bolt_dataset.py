"""Check complete-group BOLT weights and the native SFT sample contract."""

from copy import deepcopy
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from synthetic_data import write_synthetic_data

from qwenpt.train.verl.bolt_dataset import QwenBOLTDataset, compute_bolt_weights
from qwenpt.train.verl.dataset import QwenSFTDataset


@pytest.fixture
def bolt_paths(tmp_path):
    """Write unequal prompt groups split across two Parquet shards."""
    paths = write_synthetic_data(tmp_path / "sft")
    original = pq.ParquetFile(paths["train"]).read().to_pylist()
    groups = [("a", 0.0), ("b", -1.0), ("c", 5.0), ("a", 2.0), ("b", 1.0), ("a", 1.0)]
    rows = []
    for index, (prompt_id, reward) in enumerate(groups):
        row = deepcopy(original[2] if prompt_id == "b" else original[0])
        row.update(sample_id=f"bolt-{index}", prompt_id=prompt_id, reward=reward)
        rows.append(row)
    result = []
    for index, shard in enumerate((rows[:3], rows[3:])):
        path = tmp_path / f"bolt-{index}.parquet"
        pq.write_table(pa.Table.from_pylist(shard), path)
        result.append(str(path))
    return result


def test_complete_groups_span_shards_before_sample_limit(tokenizer, bolt_paths):
    dataset = QwenBOLTDataset(bolt_paths, tokenizer, {"max_length": 4096, "bolt_beta": 1.0})
    a = torch.tensor([1.0, math.exp(2), math.e], dtype=torch.float64)
    b = torch.tensor([math.exp(-1), math.e], dtype=torch.float64)
    torch.testing.assert_close(dataset.bolt_weights[[0, 3, 5]], 3 * a / a.sum())
    torch.testing.assert_close(dataset.bolt_weights[[1, 4]], 2 * b / b.sum())
    assert dataset.bolt_weights[2] == 1
    capped = QwenBOLTDataset(bolt_paths, tokenizer, {"max_length": 4096, "bolt_beta": 1.0}, max_samples=2)
    assert len(capped) == 2
    assert capped.prompt_ids == {"a", "b", "c"}
    torch.testing.assert_close(capped.bolt_weights, dataset.bolt_weights[:2])
    assert capped.bolt_weights[0] < 1
    assert capped.bolt_weights[1] < 1


@pytest.mark.parametrize("beta", [0.1, 1e-300, 1e300])
def test_equal_rewards_have_unit_weights(beta):
    weights = compute_bolt_weights([7.0, 7.0, 7.0], beta)
    assert weights.dtype == torch.float64
    torch.testing.assert_close(weights, torch.ones(3, dtype=torch.float64))


def test_extreme_finite_rewards_and_small_beta_are_stable():
    weights = compute_bolt_weights([-1e308, 1e308, 1e308], beta=1e-300)
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights, torch.tensor([0.0, 1.5, 1.5], dtype=torch.float64))
    torch.testing.assert_close(compute_bolt_weights([99.0]), torch.ones(1, dtype=torch.float64))


def test_large_rewards_and_temperature_preserve_finite_reward_gaps():
    weights = compute_bolt_weights([-1e308, 1e308], beta=1e308)
    expected = torch.tensor([2 / (1 + math.exp(2)), 2 / (1 + math.exp(-2))], dtype=torch.float64)
    torch.testing.assert_close(weights, expected)


@pytest.mark.parametrize("beta", [0, -0.1, math.nan, math.inf, True, "0.1"])
def test_invalid_beta_is_rejected(beta):
    with pytest.raises(ValueError, match="bolt_beta"):
        compute_bolt_weights([0.0, 1.0], beta)


@pytest.mark.parametrize("rewards", [[], [math.nan], [math.inf], [True], ["1"], [[1.0]], torch.tensor([True])])
def test_invalid_reward_sequences_are_rejected(rewards):
    with pytest.raises(ValueError, match="rewards"):
        compute_bolt_weights(rewards)


@pytest.mark.parametrize("field,value", [
    ("prompt_id", None), ("prompt_id", ""), ("prompt_id", []), ("prompt_id", True),
    ("reward", None), ("reward", "1.0"), ("reward", True), ("reward", math.nan), ("reward", math.inf),
])
def test_invalid_group_or_reward_reports_sample(tmp_path, tokenizer, field, value):
    row = {"sample_id": "bad-bolt", "prompt_id": "p", "reward": 0.0, "messages": []}
    row[field] = value
    path = tmp_path / "bad.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    with pytest.raises(ValueError, match=f"bad-bolt.*{field}"):
        QwenBOLTDataset(str(path), tokenizer, {})


@pytest.mark.parametrize("field", ["prompt_id", "reward"])
def test_required_fields_cannot_be_omitted(tmp_path, tokenizer, field):
    row = {"sample_id": "missing-bolt", "prompt_id": "p", "reward": 0.0, "messages": []}
    row.pop(field)
    path = tmp_path / "missing.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    with pytest.raises(ValueError, match=f"missing-bolt.*{field}"):
        QwenBOLTDataset(str(path), tokenizer, {})


def test_tools_preserve_sft_supervision_and_native_collation(tokenizer, bolt_paths):
    from verl.utils.dataset.dataset_utils import SFTTensorCollator

    dataset = QwenBOLTDataset(bolt_paths, tokenizer, {"max_length": 4096})
    sft = QwenSFTDataset(bolt_paths, tokenizer, {"max_length": 4096})
    items = [dataset[index] for index in [0, 1, 4]]
    for index, item in zip([0, 1, 4], items, strict=True):
        original = sft[index]
        for key in original:
            torch.testing.assert_close(item[key], original[key])
        assert set(item["loss_mask"].tolist()) <= {0, 1}
        assert item["bolt_weight"].shape == (1,)
        assert item["bolt_weight"].dtype == torch.float32
    supervised = tokenizer.decode(items[1]["input_ids"][items[1]["loss_mask"].bool()])
    assert "<function=add>" in supervised
    assert "先调用工具取得结果。" in supervised
    assert "TOOL_RESULT_5" not in supervised
    assert "Add two integers." not in supervised
    batch = SFTTensorCollator("no_padding")(items)
    assert batch["bolt_weight"].is_nested
    torch.testing.assert_close(batch["bolt_weight"].values(), torch.cat([item["bolt_weight"] for item in items]))
    assert batch["input_ids"].is_nested
    assert batch["loss_mask"].is_nested
