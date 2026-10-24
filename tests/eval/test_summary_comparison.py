"""Validate complete benchmark comparisons and safe monitoring snapshots."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from qwenpt.eval import summarize as module


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _write_bank(root):
    bank = [{"benchmark": benchmark, "question_id": f"{benchmark}-{index}",
             "answer": "1", "samples": 1 if benchmark == "gpqa_diamond" else 32}
            for benchmark in module.BENCHMARKS
            for index in range(198 if benchmark == "gpqa_diamond" else 30)]
    _save_bank(root, bank)
    return bank


def _save_bank(root, bank):
    path = root / "bank/question_bank.jsonl"
    _write_jsonl(path, bank)
    (path.parent / "bank.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")


def _groups(bank, variant, benchmark, correct):
    groups = []
    offset = 0
    for row in bank:
        if row["benchmark"] != benchmark:
            continue
        samples = [{"sample_index": index, "correct": offset + index < correct,
                    "format_error": False, "thinking_complete": True,
                    "generated_tokens": 10, "finish_reason": "stop"}
                   for index in range(row["samples"])]
        groups.append({"benchmark": benchmark, "question_id": row["question_id"],
                       "answer": row["answer"], "variant": variant,
                       "run_fingerprint": f"{variant}-{benchmark}", "samples": samples})
        offset += row["samples"]
    return groups


def _write_groups(root, variant, benchmark, groups):
    path = root / "results" / variant / benchmark / "results.jsonl"
    _write_jsonl(path, groups)
    return path


def test_complete_comparison_reports_all_models_accuracy_and_percentage_points(tmp_path):
    """Keep exact coverage and publish percentage-point gains in all formats."""
    bank = _write_bank(tmp_path)
    for variant_index, variant in enumerate(module.VARIANTS):
        for benchmark in module.BENCHMARKS:
            correct = 99 + variant_index * 9 if benchmark == "gpqa_diamond" else 480 + variant_index * 32
            _write_groups(tmp_path, variant, benchmark, _groups(bank, variant, benchmark, correct))

    report = module.summarize_run(tmp_path)

    assert report["complete"]
    assert len(report["metrics"]) == 20
    assert json.loads((tmp_path / "summary.json").read_text()) == report
    with (tmp_path / "summary.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 20
    markdown = (tmp_path / "summary.md").read_text()
    assert "percentage points (pp), not a relative percentage" in markdown
    for index, row in enumerate(report["metrics"]):
        variant_index = module.VARIANTS.index(row["variant"])
        total = 198 if row["benchmark"] == "gpqa_diamond" else 960
        gain = variant_index * (9 if total == 198 else 32)
        assert row["samples"] == row["expected_samples"] == total
        assert row["accuracy"] == pytest.approx(0.5 + gain / total)
        assert row["accuracy_percent"] == pytest.approx(50 + gain / total * 100)
        assert row["delta_pp"] == pytest.approx(gain / total * 100)
        assert float(csv_rows[index]["delta_pp"]) == pytest.approx(row["delta_pp"])
        assert f"| {row['variant']} |" in markdown
        assert f"{row['accuracy_percent']:.2f}%" in markdown
        assert f"{row['delta_pp']:+.2f}" in markdown


def test_incomplete_tasks_and_incomplete_baseline_do_not_publish_scores_or_deltas(tmp_path):
    """Do not mistake partial high accuracy for an official benchmark score."""
    bank = _write_bank(tmp_path)
    original = _groups(bank, "original", "gpqa_diamond", 198)
    _write_groups(tmp_path, "original", "gpqa_diamond", original[:-1])
    checkpoint = _groups(bank, "step_100", "gpqa_diamond", 99)
    _write_groups(tmp_path, "step_100", "gpqa_diamond", checkpoint)

    report = module.summarize_run(tmp_path)

    assert not report["complete"]
    baseline = report["metrics"][0]
    assert baseline["questions"] == 197
    assert baseline["correct"] == 197
    assert baseline["accuracy"] is baseline["accuracy_percent"] is baseline["delta_pp"] is None
    measured = next(row for row in report["metrics"]
                    if row["variant"] == "step_100" and row["benchmark"] == "gpqa_diamond")
    assert measured["accuracy"] == 0.5
    assert measured["delta_pp"] is None
    markdown = (tmp_path / "summary.md").read_text()
    assert "pending (197/198) | —" in markdown
    assert "| step_100 | 50.00% | —" in markdown
    assert "100.00%" not in markdown
    with (tmp_path / "summary.csv").open(newline="") as stream:
        baseline_csv = next(csv.DictReader(stream))
    assert baseline_csv["accuracy"] == baseline_csv["accuracy_percent"] == baseline_csv["delta_pp"] == ""

    _write_groups(tmp_path, "original", "gpqa_diamond", original)
    _write_groups(tmp_path, "step_200", "gpqa_diamond",
                  _groups(bank, "step_200", "gpqa_diamond", 198)[:1])
    report = module.summarize_run(tmp_path)
    measured = next(row for row in report["metrics"]
                    if row["variant"] == "step_100" and row["benchmark"] == "gpqa_diamond")
    assert measured["delta_pp"] == -50.0
    partial = next(row for row in report["metrics"]
                   if row["variant"] == "step_200" and row["benchmark"] == "gpqa_diamond")
    assert partial["accuracy"] is partial["accuracy_percent"] is partial["delta_pp"] is None


@pytest.mark.parametrize("suffix", [b'{"question_id":', b'{"text": "\xe4\xb8'])
def test_read_jsonl_ignores_only_unterminated_partial_final_record(tmp_path, suffix):
    """Concurrent reads tolerate an unfinished JSON object or UTF-8 character."""
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"question_id": "first"}\n' + suffix)
    assert module.read_jsonl(path) == [{"question_id": "first"}]


def test_read_jsonl_preserves_valid_final_record_without_newline(tmp_path):
    """A durable hand-written final object must not disappear from coverage."""
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"question_id": "first"}\n{"question_id": "last"}')
    assert module.read_jsonl(path) == [{"question_id": "first"}, {"question_id": "last"}]


@pytest.mark.parametrize("content", [b'{"broken":\n', b'{"broken":\n{"valid": true}',
                                     b'{"text": "\xe4\xb8\n'])
def test_read_jsonl_rejects_malformed_newline_terminated_records(tmp_path, content):
    """Never turn durable corruption into apparently successful partial progress."""
    path = tmp_path / "results.jsonl"
    path.write_bytes(content)
    with pytest.raises((json.JSONDecodeError, UnicodeDecodeError)):
        module.read_jsonl(path)


@pytest.mark.parametrize("change", ["duplicate_question", "missing_sample", "duplicate_sample", "fingerprint"])
def test_comparison_preserves_coverage_uniqueness_and_fingerprint_checks(tmp_path, change):
    """A result cannot earn an official score by duplicating or mixing runs."""
    bank = _write_bank(tmp_path)
    groups = _groups(bank, "step_100", "aime2024", 480)
    if change == "duplicate_question":
        groups.append(groups[0])
    elif change == "missing_sample":
        groups[0]["samples"].pop()
    elif change == "duplicate_sample":
        groups[0]["samples"][-1]["sample_index"] = 0
    else:
        groups[-1]["run_fingerprint"] = "another-protocol"
    _write_groups(tmp_path, "step_100", "aime2024", groups)
    with pytest.raises(ValueError):
        module.summarize_run(tmp_path)


@pytest.mark.parametrize("change", ["duplicate_question", "sample_count", "checksum"])
def test_comparison_preserves_frozen_bank_checks(tmp_path, change):
    """Reject altered benchmark coverage even if its checksum is refreshed."""
    bank = _write_bank(tmp_path)
    if change == "duplicate_question":
        bank[-1] = bank[-2]
    elif change == "sample_count":
        bank[-1]["samples"] = 31
    else:
        (tmp_path / "bank/bank.sha256").write_text("0" * 64 + "\n")
    if change != "checksum":
        _save_bank(tmp_path, bank)
    with pytest.raises(ValueError):
        module.summarize_run(tmp_path)


def test_summary_files_replace_complete_snapshots_atomically(tmp_path, monkeypatch):
    """A monitor sees the old summary until a complete replacement is ready."""
    _write_bank(tmp_path)
    names = ("summary.json", "summary.csv", "summary.md")
    for name in names:
        (tmp_path / name).write_text("previous snapshot")
    replaced = []
    replace = module.os.replace

    def check_replace(source, destination):
        destination = Path(destination)
        assert destination.read_text() == "previous snapshot"
        assert Path(source).parent == destination.parent
        content = Path(source).read_text()
        if destination.suffix == ".json":
            assert len(json.loads(content)["metrics"]) == 20
        elif destination.suffix == ".csv":
            assert len(list(csv.DictReader(content.splitlines()))) == 20
        else:
            assert "| step_400 |" in content
        replaced.append(destination.name)
        replace(source, destination)

    monkeypatch.setattr(module.os, "replace", check_replace)
    module.summarize_run(tmp_path)
    assert tuple(replaced) == names
    assert not list(tmp_path.glob(".summary.*"))
