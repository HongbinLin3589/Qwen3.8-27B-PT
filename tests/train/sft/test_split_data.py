"""Verify reproducible SFT splits preserve every input row and training field."""

import hashlib
import json
import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qwenpt.train.sft.split_data import prepare_sft_data

pa.set_cpu_count(1)


def _make_shards(root):
    root.mkdir()
    rows = [{
        "sample_id": f"sample-{index}",
        "split": "train" if index % 2 else "validation",
        "messages": [
            {"role": "user", "content": f"question {index}"},
            {"role": "assistant", "content": "<think> </think>answer" if index == 0
             else f"<think>reason {index}</think>answer"},
        ],
        "enable_thinking": True,
        "tools": [{"type": "function", "function": {"name": "calculate"}}],
        "images": [{"bytes": b"example-image", "mime_type": "image/png"}],
        "gold_answer": str(index),
        "metadata": {"source": "demo", "number": index},
    } for index in range(20)]
    pq.write_table(pa.Table.from_pylist(rows[:10]), root / "a.parquet")
    pq.write_table(pa.Table.from_pylist(rows[10:]), root / "b.parquet")
    return rows


def _read_split(root, name):
    return pq.ParquetFile(root / name).read(use_threads=False).to_pylist()


def test_directory_and_reversed_file_inputs_have_identical_seeded_split(tmp_path):
    source = tmp_path / "source"
    _make_shards(source)
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = prepare_sft_data([str(source)], str(first))
    again = prepare_sft_data(
        [str(source / "b.parquet"), str(source / "a.parquet")], str(second),
    )
    assert manifest == again
    assert manifest["counts"] == {"total": 20, "train": 18, "validation": 2}
    assert manifest["seed"] == 20261020
    expected = list(range(20))
    random.Random(20261020).shuffle(expected)
    assert manifest["split_indices"] == {"validation": expected[:2], "train": expected[2:]}
    for name in ("train.parquet", "val.parquet"):
        assert _read_split(first, name) == _read_split(second, name)
    assert json.loads((first / "split_manifest.json").read_text()) == manifest
    for item in manifest["inputs"]:
        with open(item["path"], "rb") as file:
            assert item["sha256"] == hashlib.file_digest(file, "sha256").hexdigest()


def test_split_is_disjoint_complete_and_preserves_content_and_extra_fields(tmp_path):
    source = tmp_path / "source"
    original = _make_shards(source)
    output = tmp_path / "output"
    manifest = prepare_sft_data([str(source)], str(output), seed=42, val_ratio=0.25)
    train = _read_split(output, "train.parquet")
    validation = _read_split(output, "val.parquet")
    assert manifest["counts"] == {"total": 20, "train": 15, "validation": 5}
    train_ids = {row["sample_id"] for row in train}
    validation_ids = {row["sample_id"] for row in validation}
    assert not train_ids & validation_ids
    assert train_ids | validation_ids == {row["sample_id"] for row in original}
    by_id = {row["sample_id"]: row for row in original}
    for label, rows in (("train", train), ("validation", validation)):
        for row in rows:
            assert row == {**by_id[row["sample_id"]], "split": label}


def test_existing_split_artifact_is_not_overwritten(tmp_path):
    source = tmp_path / "source"
    _make_shards(source)
    output = tmp_path / "output"
    prepare_sft_data([str(source)], str(output))
    before = (output / "train.parquet").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_sft_data([str(source)], str(output), seed=42)
    assert (output / "train.parquet").read_bytes() == before


def test_reuse_requires_matching_inputs_parameters_and_outputs(tmp_path):
    source = tmp_path / "source"
    _make_shards(source)
    output = tmp_path / "output"
    manifest = prepare_sft_data([str(source)], str(output))
    assert prepare_sft_data([str(source)], str(output), reuse_existing=True) == manifest
    for options in ({"seed": 42}, {"val_ratio": 0.2}):
        with pytest.raises(ValueError, match="inputs or parameters differ"):
            prepare_sft_data([str(source)], str(output), reuse_existing=True, **options)
    with (output / "train.parquet").open("ab") as file:
        file.write(b"corrupted")
    with pytest.raises(ValueError, match="output hashes differ"):
        prepare_sft_data([str(source)], str(output), reuse_existing=True)


@pytest.mark.parametrize("ratio", [0, 1, -0.1, float("nan"), 0.001])
def test_invalid_or_empty_split_is_rejected(tmp_path, ratio):
    source = tmp_path / "source"
    _make_shards(source)
    with pytest.raises(ValueError):
        prepare_sft_data([str(source)], str(tmp_path / "output"), val_ratio=ratio)
