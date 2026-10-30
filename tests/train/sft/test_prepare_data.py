"""Exercise train-only SFT data preparation with local data and a tiny tokenizer."""

import json
import hashlib
from pathlib import Path
import struct

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qwenpt.train.common.files import sha256
from qwenpt.train.sft.prepare_data import _select_rows, main, prepare_sft_data
from qwenpt.train.common.template import tokenize_sft_sample


@pytest.fixture
def source(tmp_path, tokenizer):
    """Keep validation/test rows invalid so any accidental inclusion fails loudly."""
    root = tmp_path / "sft"
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    for domain in ("math", "code"):
        rows = [{
            "id": f"{domain}-{item}", "domain": domain, "split": "train", "tools": "[]",
            "total_tokens": (100, 6000, 12000)[item % 3], "supervised_tokens": 12345,
            "messages": [
                {"role": "user", "content": "x" * 5000 if item == 5 else "2+2?", "loss_mask": 1.0, "weight": 1.0},
                {"role": "assistant", "content": "4", "loss_mask": 1.0, "weight": 1.0},
            ],
        } for item in range(6)]
        for split in ("train", "validation", "test"):
            path = root / domain / split / "part-00000.parquet"
            path.parent.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist(rows if split == "train" else [{"split": split, "messages": None}]), path)
    return root, tokenizer_path


def test_round_robin_strata_sampling_is_seeded_and_caps_producer_bins():
    """Sampling covers occupied domains and producer bins before returning to a stratum."""
    candidates = [{"domain": domain, "bucket": bucket, "row": item} for domain in ("a", "b") for bucket in (0, 1) for item in range(10)]
    chosen = _select_rows(candidates, 8, 7, None)
    assert chosen == _select_rows(candidates, 8, 7, None)
    assert chosen != _select_rows(candidates, 8, 8, None)
    assert all(sum(row["domain"] == domain and row["bucket"] == bucket for row in chosen) == 2
               for domain in ("a", "b") for bucket in (0, 1))
    capped = _select_rows(candidates, 40, 7, 3)
    assert len(capped) == 6
    assert all(sum(row["bucket"] == bucket for row in capped) == 3 for bucket in (0, 1))
    assert len(_select_rows(candidates, -1, 7, None)) == len(candidates)


def test_pool_preserves_train_rows_and_schema_and_uses_actual_lengths(source, tmp_path, tokenizer):
    """Producer lengths choose candidates, while measured lengths choose output buckets."""
    root, tokenizer_path = source
    output = tmp_path / "prepared"
    manifest = prepare_sft_data(str(root), str(tokenizer_path), str(output))
    assert manifest["status"] == "passed"
    assert manifest["max_samples"] == -1
    assert manifest["counts"] == {"eligible_train_rows": 12, "selected_rows": 12, "validated_rows": 12, "failures": 0}
    assert len(manifest["inputs"]) == 2
    assert len(manifest["outputs"]) == 9
    rows = pq.ParquetFile(output / "pool.parquet").read().to_pylist()
    for row in rows:
        original = pq.ParquetFile(row["benchmark_source_path"]).read().to_pylist()[row["benchmark_source_row"]]
        assert all(row[key] == value for key, value in original.items())
        assert row["tools"] == "[]"
        assert row["supervised_tokens"] == 12345
        assert row["split"] == "train"
        inputs = tokenize_sft_sample(row, tokenizer)
        assert row["tokenized_length"] == len(inputs["input_ids"])
        assert row["tokenized_supervised_tokens"] == int(inputs["loss_mask"].sum())
        assert len(row["sample_fingerprint"]) == len(row["tokenized_fingerprint"]) == 64
        values = inputs["input_ids"].tolist() + inputs["loss_mask"].tolist()
        assert row["tokenized_fingerprint"] == hashlib.sha256(struct.pack("=" + "q" * len(values), *values)).hexdigest()
    assert manifest["actual_length_bins"]["0_4096"]["rows"] == 10
    assert manifest["actual_length_bins"]["4096_8192"]["rows"] == 2
    assert sum(stat["tokens"] for stat in manifest["actual_length_bins"].values()) == sum(row["tokenized_length"] for row in rows)
    assert sum(stat["supervised_tokens"] for stat in manifest["actual_length_bins"].values()) == sum(row["tokenized_supervised_tokens"] for row in rows)
    for entry in manifest["inputs"]:
        assert entry["sha256"] == sha256(Path(entry["path"]))
    assets = manifest["fingerprint"]["tokenizer_files"]
    assert assets["tokenizer.json"] == sha256(tokenizer_path / "tokenizer.json")
    assert set(manifest["fingerprint"]["source_sha256"]) == {"template.py", "prepare_sft_benchmark_data.py"}
    assert manifest["resolved_template_sha256"]
    assert (output / "failures.jsonl").read_text() == ""
    with pytest.raises(FileExistsError):
        prepare_sft_data(str(root), str(tokenizer_path), str(output))


def test_worker_processes_produce_the_same_pool_and_fingerprints(source, tmp_path):
    """Worker scheduling must not affect selected rows, rendering or output order."""
    root, tokenizer_path = source
    manifests = []
    pools = []
    for workers in (1, 2):
        output = tmp_path / f"workers-{workers}"
        manifests.append(prepare_sft_data(str(root), str(tokenizer_path), str(output), workers=workers, max_samples=6, seed=7))
        pools.append(pq.ParquetFile(output / "pool.parquet").read().to_pylist())
    assert pools[0] == pools[1]
    assert manifests[0]["selected_strata"] == manifests[1]["selected_strata"]
    assert manifests[0]["resolved_template_sha256"] == manifests[1]["resolved_template_sha256"]


@pytest.mark.parametrize("bad_field", ["supervision", "split"])
def test_invalid_rows_publish_a_failure_report_without_partial_training_data(source, tmp_path, bad_field):
    """Failed rows are visible and cannot silently disappear into a successful pool."""
    root, tokenizer_path = source
    path = root / "math/train/part-00000.parquet"
    rows = pq.ParquetFile(path).read().to_pylist()
    if bad_field == "supervision":
        rows[0]["messages"][-1]["loss_mask"] = 0.0
    else:
        rows[0]["split"] = "validation"
    pq.write_table(pa.Table.from_pylist(rows), path)
    output = tmp_path / "failed"
    manifest = prepare_sft_data(str(root), str(tokenizer_path), str(output), max_samples=-1)
    assert manifest["status"] == "failed"
    failures = [json.loads(line) for line in (output / "failures.jsonl").read_text().splitlines()]
    assert len(failures) == manifest["counts"]["failures"] == 1
    assert failures[0]["id"] == "math-0"
    assert failures[0]["stage"] == ("tokenize" if bad_field == "supervision" else "metadata")
    assert not list(output.glob("*.parquet"))


@pytest.mark.parametrize("kwargs", [{"workers": 0}, {"max_samples": 0}, {"per_bucket_rows": 0}, {"max_samples": -1, "per_bucket_rows": 5}])
def test_invalid_resource_or_sampling_options_fail_before_preparation(source, tmp_path, kwargs):
    """Explicit sampling caps must not contradict full-data preparation."""
    root, tokenizer_path = source
    with pytest.raises(ValueError):
        prepare_sft_data(str(root), str(tokenizer_path), str(tmp_path / "unused"), **kwargs)


def test_all_invalid_metadata_still_publishes_each_failed_row(source, tmp_path):
    """An entirely invalid delivery receives an auditable report instead of an empty pool."""
    root, tokenizer_path = source
    for path in root.glob("*/train/*.parquet"):
        rows = pq.ParquetFile(path).read().to_pylist()
        for row in rows:
            row["split"] = "validation"
        pq.write_table(pa.Table.from_pylist(rows), path)
    output = tmp_path / "all-invalid"
    manifest = prepare_sft_data(str(root), str(tokenizer_path), str(output))
    assert manifest["status"] == "failed"
    assert manifest["counts"]["selected_rows"] == manifest["counts"]["validated_rows"] == 0
    assert len((output / "failures.jsonl").read_text().splitlines()) == 12
    assert not list(output.glob("*.parquet"))


def test_cli_exits_nonzero_when_actual_lengths_exceed_supported_buckets(source, tmp_path, monkeypatch):
    """The default CLI audits all train rows and cannot publish failed input as usable."""
    import sys

    root, tokenizer_path = source
    output = tmp_path / "overflow"
    monkeypatch.setattr("qwenpt.train.sft.prepare_data.LENGTH_LIMITS", (4096,))
    monkeypatch.setattr(sys, "argv", [
        "prepare", "--data-root", str(root), "--tokenizer", str(tokenizer_path),
        "--output", str(output), "--workers", "1",
    ])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["max_samples"] == -1
    assert manifest["counts"]["failures"] == 2
    assert not list(output.glob("*.parquet"))
