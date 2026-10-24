"""Freeze cached GPQA Diamond and AIME 2024–2026 into a shared evaluation bank."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


GPQA_PROMPT = """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}"""
AIME_PROMPT = (
    "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
)
SHUFFLE_SEED = 0
SOURCES = (
    {
        "benchmark": "gpqa_diamond",
        "repository": "Idavidrein/gpqa",
        "revision": "633f5ee89ab8ad4522a9f850766b73f62147ffdd",
        "relative_path": "hub/datasets--Idavidrein--gpqa/snapshots/633f5ee89ab8ad4522a9f850766b73f62147ffdd/gpqa_diamond.csv",
        "rows": 198,
        "samples": 1,
    },
    {
        "benchmark": "aime2024",
        "repository": "Maxwell-Jia/AIME_2024",
        "revision": "8d88b2876a82a080e2f172cc9b25d0d9d2cb4792",
        "relative_path": "datasets/Maxwell-Jia___aime_2024/default/0.0.0/8d88b2876a82a080e2f172cc9b25d0d9d2cb4792/aime_2024-train.arrow",
        "rows": 30,
        "samples": 32,
    },
    {
        "benchmark": "aime2025",
        "repository": "math-ai/aime25",
        "revision": "563bb8404243c5f09de6ec262f2db674fe5bce9b",
        "relative_path": "datasets/math-ai___aime25/default/0.0.0/563bb8404243c5f09de6ec262f2db674fe5bce9b/aime25-test.arrow",
        "rows": 30,
        "samples": 32,
    },
    {
        "benchmark": "aime2026",
        "repository": "MathArena/aime_2026",
        "revision": "d61f486805306e6e6b5913504c07763a99e1b112",
        "expected_sha256": "ae72853349a46c5f08666feb42db2d09fb4ac4fd4768c726b29702df21c7768e",
        "rows": 30,
        "samples": 32,
    },
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    if path.suffix == ".parquet":
        return pq.read_table(path).to_pylist()
    with pa.memory_map(str(path), "r") as stream:
        return pa.ipc.open_stream(stream).read_all().to_pylist()


def _make_record(
    row: dict[str, Any], source: dict[str, Any], index: int, rng: random.Random
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_index": index,
        "source_repository": source["repository"],
        "source_revision": source["revision"],
        "source_sha256": source["sha256"],
    }
    if source["benchmark"] == "gpqa_diamond":
        original = [row["Correct Answer"]] + [
            row[f"Incorrect Answer {number}"] for number in range(1, 4)
        ]
        permutation = rng.sample(range(4), 4)
        choices = dict(zip("ABCD", (original[pos] for pos in permutation), strict=True))
        answer = "ABCD"[permutation.index(0)]
        question_id = str(row["Record ID"])
        prompt = GPQA_PROMPT.format(Question=row["Question"], **choices)
        metadata.update(
            choices=choices,
            correct_letter=answer,
            source_record_id=question_id,
            choice_permutation=permutation,
        )
    else:
        question_id = str(row.get("ID", row.get("id", row.get("problem_idx"))))
        answer = str(int(row.get("Answer", row.get("answer"))))
        if not 0 <= int(answer) <= 999:
            raise ValueError(f"Invalid AIME answer for {question_id}: {answer}")
        prompt = AIME_PROMPT.format(problem=row.get("Problem", row.get("problem")).strip())
    return {
        "benchmark": source["benchmark"],
        "question_id": question_id,
        "prompt": prompt,
        "answer": answer,
        "samples": source["samples"],
        "metadata": metadata,
    }


def prepare_benchmarks(
    output_dir: Path, cache_root: Path | None = None, *, aime2026_source: Path
) -> dict[str, Any]:
    """Write a deterministic bank and manifest from cache; reject differing existing outputs.

    Args:
        output_dir: Directory for question_bank.jsonl, manifest.json, and bank.sha256.
        cache_root: Hugging Face cache directory, defaulting to ~/.cache/huggingface.
        aime2026_source: MathArena Parquet at the pinned revision and SHA256.

    Returns:
        The manifest describing all source artifacts and generated bank contents.
    """
    cache_root = (cache_root or Path.home() / ".cache/huggingface").expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    records = []
    sources = []
    rng = random.Random(SHUFFLE_SEED)
    for source in SOURCES:
        path = (
            aime2026_source.expanduser().resolve()
            if source["benchmark"] == "aime2026"
            else cache_root / source["relative_path"]
        )
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if source.get("expected_sha256", source_hash) != source_hash:
            raise ValueError(f"SHA256 mismatch for pinned benchmark source: {path}")
        frozen_source = {**source, "sha256": source_hash}
        rows = _read_rows(path)
        if len(rows) != source["rows"]:
            raise ValueError(f"Expected {source['rows']} rows in {path}, found {len(rows)}")
        prepared = [_make_record(row, frozen_source, index, rng) for index, row in enumerate(rows)]
        ids = [row["question_id"] for row in prepared]
        if len(set(ids)) != len(ids) or any(not value or value == "None" for value in ids):
            raise ValueError(f"Invalid or duplicate question IDs in {path}")
        records.extend(prepared)
        sources.append(
            {
                **{key: value for key, value in source.items() if key != "relative_path"},
                "path": str(path),
                "sha256": source_hash,
                "url": f"https://huggingface.co/datasets/{source['repository']}/tree/{source['revision']}",
            }
        )
    # Escaping Unicode line separators also makes splitlines-based JSONL readers safe.
    bank = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records).encode("utf-8")
    bank_hash = hashlib.sha256(bank).hexdigest()
    manifest = {
        "schema_version": 1,
        "bank_file": "question_bank.jsonl",
        "bank_sha256": bank_hash,
        "questions": len(records),
        "generations_per_model": sum(row["samples"] for row in records),
        "shuffle_seed": SHUFFLE_SEED,
        "prompt_templates": {"gpqa_diamond": GPQA_PROMPT, "aime": AIME_PROMPT},
        "prompt_sources": {
            "gpqa_diamond": "https://github.com/openai/simple-evals/blob/main/common.py",
            "aime": "https://github.com/deepseek-ai/DeepInfer/blob/main/eval/aime/bench.py",
        },
        "sources": sources,
    }
    outputs = {
        "question_bank.jsonl": bank,
        "manifest.json": (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        "bank.sha256": f"{bank_hash}  question_bank.jsonl\n".encode("ascii"),
    }
    # Check all artifacts first so a conflict cannot leave a newly written partial bank.
    for name, content in outputs.items():
        path = output_dir / name
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"Refusing to replace different frozen content: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        path = output_dir / name
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(content)
    return manifest


def main() -> None:
    """Parse local cache and output paths, then print the generated bank manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--aime2026-source", type=Path, required=True,
                        help="Local MathArena AIME2026 Parquet with the pinned SHA256")
    args = parser.parse_args()
    manifest = prepare_benchmarks(args.output_dir, args.cache_root, aime2026_source=args.aime2026_source)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
