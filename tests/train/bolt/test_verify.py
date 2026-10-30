"""Exercise native adapter validation, provenance binding and audit reuse on CPU."""

import json
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qwenpt.train.bolt.data import finalize_bolt_data, load_math_pool
from qwenpt.train.bolt.verify import verify_bolt_splits


@pytest.fixture
def finalized(tmp_path, tokenizer):
    """Export four genuine prompt groups and save an offline tiny Qwen tokenizer."""
    pool_path = tmp_path / "pool.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"messages": [{"role": "user", "content": f"Compute {index} + 1."}], "ground_truth": "1"}
        for index in range(4)
    ]), pool_path)
    groups = [{**row, "enable_thinking": True, "run_fingerprint": "test", "candidates": [
        {"candidate_index": candidate, "content": "Answer is 1.", "reasoning_content": "Add.",
         "reward": float(candidate % 2), "finish_reason": "stop", "generated_tokens": 5}
        for candidate in range(8)
    ]} for row in load_math_pool([str(pool_path)])]
    shard = tmp_path / "shard.jsonl"
    shard.write_text("".join(json.dumps(group) + "\n" for group in groups))
    output = tmp_path / "finalized"
    finalize_bolt_data([str(pool_path)], [str(shard)], str(output), val_ratio=0.25)
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    return output, tokenizer_path


def test_native_rows_checked_and_matching_audit_reused(finalized):
    output, tokenizer_path = finalized
    report = verify_bolt_splits(str(output), str(tokenizer_path))
    assert report["status"] == "passed"
    assert report["splits"]["train"]["rows"] == 24
    assert report["splits"]["validation"]["rows"] == 8
    for split in report["splits"].values():
        assert split["min_supervised_tokens"] > 0
        assert split["mean_bolt_weight"] == pytest.approx(1.0)
    assert json.loads((output / "native_input_check.json").read_text()) == report
    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=AssertionError("retokenized")):
        assert verify_bolt_splits(str(output), str(tokenizer_path)) == report


def test_changed_length_requires_fresh_native_validation(finalized):
    output, tokenizer_path = finalized
    verify_bolt_splits(str(output), str(tokenizer_path))
    with pytest.raises(ValueError, match="max_length=1"):
        verify_bolt_splits(str(output), str(tokenizer_path), max_length=1)


def test_modified_parquet_rejected_before_cached_audit(finalized):
    output, tokenizer_path = finalized
    verify_bolt_splits(str(output), str(tokenizer_path))
    with (output / "train.parquet").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="hashes"):
        verify_bolt_splits(str(output), str(tokenizer_path))
