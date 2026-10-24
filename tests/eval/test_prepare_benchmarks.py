"""Check frozen banks preserve sample counts, answer mappings, and existing artifacts."""

import csv
import hashlib
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qwenpt.eval import prepare_benchmarks as module


def _cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    cache = tmp_path / "cache"
    cache.mkdir()
    specs = []
    gpqa = {
        "Question": "Which choice is correct?\u2028Select one.",
        "Correct Answer": "correct",
        "Incorrect Answer 1": "wrong 1",
        "Incorrect Answer 2": "wrong 2",
        "Incorrect Answer 3": "wrong 3",
        "Record ID": "question-1",
    }
    for original in module.SOURCES:
        source = {**original, "rows": 1}
        if source["benchmark"] == "gpqa_diamond":
            path = cache / "gpqa.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(gpqa))
                writer.writeheader()
                writer.writerow(gpqa)
        elif source["benchmark"] == "aime2026":
            path = cache / "aime2026.parquet"
            pq.write_table(pa.Table.from_pylist([
                {"problem_idx": 1, "problem": "Compute 1+1.", "answer": 2}
            ]), path)
            source["expected_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            path = cache / f"{source['benchmark']}.arrow"
            table = pa.Table.from_pylist([{"id": "1", "problem": "Compute 1+1.", "answer": "002"}])
            with pa.OSFile(str(path), "wb") as stream:
                with pa.ipc.new_stream(stream, table.schema) as writer:
                    writer.write_table(table)
        source["relative_path"] = path.name
        specs.append(source)
    monkeypatch.setattr(module, "SOURCES", tuple(specs))
    return cache, cache / "aime2026.parquet"


def test_bank_is_deterministic_and_preserves_protocol(tmp_path, monkeypatch):
    cache, aime2026 = _cache(tmp_path, monkeypatch)
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    manifest = module.prepare_benchmarks(first_dir, cache, aime2026_source=aime2026)
    assert module.prepare_benchmarks(first_dir, cache, aime2026_source=aime2026) == manifest
    module.prepare_benchmarks(second_dir, cache, aime2026_source=aime2026)
    for name in ("question_bank.jsonl", "manifest.json", "bank.sha256"):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
    rows = [json.loads(line) for line in (first_dir / "question_bank.jsonl").read_text().splitlines()]
    assert [row["samples"] for row in rows] == [1, 32, 32, 32]
    assert manifest["generations_per_model"] == 97
    gpqa = rows[0]
    assert gpqa["metadata"]["choices"][gpqa["answer"]] == "correct"
    assert gpqa["metadata"]["choice_permutation"] == random.Random(0).sample(range(4), 4)
    assert gpqa["metadata"]["correct_letter"] == gpqa["answer"]
    assert all(row["answer"] == "2" for row in rows[1:])
    assert all(row["prompt"].endswith("\\boxed{}.") for row in rows[1:])
    assert rows[-1]["metadata"]["source_repository"] == "MathArena/aime_2026"
    assert rows[-1]["metadata"]["source_sha256"] == hashlib.sha256(aime2026.read_bytes()).hexdigest()


def test_conflicting_output_is_not_replaced(tmp_path, monkeypatch):
    cache, aime2026 = _cache(tmp_path, monkeypatch)
    output = tmp_path / "output"
    output.mkdir()
    (output / "manifest.json").write_text("existing frozen content")
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        module.prepare_benchmarks(output, cache, aime2026_source=aime2026)
    assert (output / "manifest.json").read_text() == "existing frozen content"
    assert not (output / "question_bank.jsonl").exists()


def test_incomplete_source_is_rejected_before_writing(tmp_path, monkeypatch):
    cache, aime2026 = _cache(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "SOURCES", ({**module.SOURCES[0], "rows": 2},))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Expected 2 rows"):
        module.prepare_benchmarks(output, cache, aime2026_source=aime2026)
    assert not output.exists()


def test_aime2026_requires_exact_pinned_artifact(tmp_path, monkeypatch):
    cache, aime2026 = _cache(tmp_path, monkeypatch)
    aime2026.write_bytes(b"modified artifact")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        module.prepare_benchmarks(tmp_path / "output", cache, aime2026_source=aime2026)
    assert not (tmp_path / "output").exists()
