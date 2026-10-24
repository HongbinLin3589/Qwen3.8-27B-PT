"""Verify benchmark coverage and publish sample-average accuracy tables."""

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

BENCHMARKS = ("gpqa_diamond", "aime2024", "aime2025", "aime2026")
VARIANTS = ("original", "step_100", "step_200", "step_300", "step_400")


def read_jsonl(path: Path) -> list[dict]:
    """Read a snapshot while a writer may be appending its final record.

    Writers terminate durable records with a newline. Only a malformed final
    unterminated record is ignored; valid JSON without a newline is retained.
    """
    if not path.exists():
        return []
    lines = path.read_bytes().split(b"\n")
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines) - 1:
                raise
    return records


def _atomic_write(path: Path, content: str) -> None:
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def summarize_task(bank: list[dict], groups: list[dict], variant: str, benchmark: str) -> dict:
    """Validate unique questions and exact sample counts, then compute accuracy.

    Every completed AIME question contributes all 32 samples. The reported
    average is the fraction of correct samples, never majority vote or pass@32.
    """
    expected = {str(row["question_id"]): row for row in bank if row["benchmark"] == benchmark}
    seen = set()
    samples = []
    fingerprints = set()
    for group in groups:
        identity = str(group["question_id"])
        if group["benchmark"] != benchmark or group["variant"] != variant:
            raise ValueError("Results contain the wrong benchmark or model variant")
        if identity not in expected or identity in seen:
            raise ValueError(f"Unexpected or duplicate question: {identity}")
        row = expected[identity]
        if group["answer"] != row["answer"]:
            raise ValueError(f"Gold answer differs for {identity}")
        candidates = group["samples"]
        if sorted(item["sample_index"] for item in candidates) != list(range(row["samples"])):
            raise ValueError(f"Missing or duplicate samples for {identity}")
        for item in candidates:
            if any(type(item[key]) is not bool for key in ("correct", "format_error", "thinking_complete")):
                raise ValueError("Grading flags must be booleans")
            if type(item["generated_tokens"]) is not int or item["generated_tokens"] <= 0:
                raise ValueError("A sample has no generated tokens")
        fingerprints.add(group["run_fingerprint"])
        samples.extend(candidates)
        seen.add(identity)
    if len(fingerprints) > 1:
        raise ValueError("A result file mixes multiple generation protocols")
    count = len(samples)
    correct = sum(item["correct"] for item in samples)
    complete = len(seen) == len(expected) and bool(expected)
    accuracy = correct / count if complete else None
    return {
        "variant": variant, "benchmark": benchmark,
        "metric": "accuracy" if benchmark == "gpqa_diamond" else "avg@32",
        "complete": complete,
        "questions": len(seen), "expected_questions": len(expected),
        "samples": count, "expected_samples": sum(row["samples"] for row in expected.values()),
        "correct": correct, "accuracy": accuracy,
        "accuracy_percent": accuracy * 100 if accuracy is not None else None,
        "truncated": sum(item["finish_reason"] == "length" for item in samples),
        "format_errors": sum(item["format_error"] for item in samples),
        "unfinished_thinking": sum(not item["thinking_complete"] for item in samples),
        "mean_generated_tokens": sum(item["generated_tokens"] for item in samples) / count if count else None,
        "run_fingerprint": next(iter(fingerprints), None),
    }


def summarize_run(root: Path) -> dict:
    """Write coverage and complete-task accuracy with gains in percentage points."""
    bank_path = root / "bank/question_bank.jsonl"
    expected_hash = (root / "bank/bank.sha256").read_text().strip().split()[0]
    if hashlib.sha256(bank_path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Frozen question bank checksum differs")
    bank = read_jsonl(bank_path)
    identities = {(row["benchmark"], row["question_id"]) for row in bank}
    counts = {name: sum(row["benchmark"] == name for row in bank) for name in BENCHMARKS}
    if (len(bank) != 288 or len(identities) != len(bank)
            or counts != dict(zip(BENCHMARKS, (198, 30, 30, 30), strict=True))
            or any(row["samples"] != (1 if row["benchmark"] == "gpqa_diamond" else 32) for row in bank)):
        raise ValueError("Expected unique questions: 198 GPQA at n=1 and 30 per AIME year at n=32")
    rows = [summarize_task(bank, read_jsonl(root / "results" / variant / benchmark / "results.jsonl"),
                           variant, benchmark) for variant in VARIANTS for benchmark in BENCHMARKS]
    baseline = {row["benchmark"]: row for row in rows if row["variant"] == "original"}
    for row in rows:
        original = baseline[row["benchmark"]]
        row["delta_pp"] = (row["accuracy_percent"] - original["accuracy_percent"]
                           if row["complete"] and original["complete"] else None)
    report = {"complete": all(row["complete"] for row in rows), "metrics": rows}
    _atomic_write(root / "summary.json", json.dumps(report, indent=2) + "\n")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(root / "summary.csv", stream.getvalue())
    lines = ["# BOLT checkpoint benchmark results", "",
             "GPQA: one sample per question; AIME: mean accuracy over 32 samples per question.", "",
             "Only complete tasks are shown as scores; pending tasks remain blank.", "",
             "Delta is the accuracy difference from the original model in percentage points (pp), not a relative percentage.", "",
             "| Model | GPQA Diamond | Delta (pp) | AIME 2024 avg@32 | Delta (pp) | AIME 2025 avg@32 | Delta (pp) | AIME 2026 avg@32 | Delta (pp) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant in VARIANTS:
        cells = []
        for benchmark in BENCHMARKS:
            row = next(row for row in rows if row["variant"] == variant and row["benchmark"] == benchmark)
            cells.append(f"{row['accuracy_percent']:.2f}%" if row["complete"] else f"pending ({row['questions']}/{row['expected_questions']})")
            cells.append(f"{row['delta_pp']:+.2f}" if row["delta_pp"] is not None else "—")
        lines.append("| " + " | ".join([variant] + cells) + " |")
    _atomic_write(root / "summary.md", "\n".join(lines) + "\n")
    return report


def main() -> None:
    """Summarize a run and optionally require every model/task to be complete."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = summarize_run(args.run_dir)
    print(json.dumps({"complete": report["complete"],
                      "completed_tasks": sum(row["complete"] for row in report["metrics"]),
                      "total_tasks": len(report["metrics"])}))
    if args.require_complete and not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
