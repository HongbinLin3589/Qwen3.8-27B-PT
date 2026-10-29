"""Exercise strict paired benchmark acceptance with small CPU JSON fixtures."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import pytest

from legacy.sft.utils.sft_ab_report import ARMS, build_report, main


def _write_json(path, data):
    path.write_text(json.dumps(data) + "\n")


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def run_dir(tmp_path):
    """Three matched arms differ only in elapsed time and allowed CPU placement."""
    _write_json(tmp_path / "host_plan.json", {
        "gpus": [{"local_rank": 0, "numa_node": 1}, {"local_rank": 1, "numa_node": 1}],
        "effective_physical_cores_per_arm": 8,
        "threads_per_rank": 4,
        "comparison_limits": ["Sixteen CPUs are reserved while eight are active."],
    })
    _write_json(tmp_path / "ce_gpu_validation.json", {
        "passed": True, "completed_cases": 6, "expected_cases": 6,
        "cases": [{"passed": True} for _ in range(6)],
    })
    origin = datetime(2026, 10, 21, 12).timestamp()
    for arm, measured_seconds in zip(ARMS, (12.0, 8.0, 6.0), strict=True):
        directory = tmp_path / arm
        directory.mkdir()
        native = []
        for rank in (0, 1):
            rows = []
            for step in range(1, 12):
                elapsed = (500 if step <= 3 else measured_seconds) + rank
                local_lengths = [2, 3, 4, 5] if rank == 0 else [6, 7, 8, 9]
                digest = hashlib.sha256(f"rank{rank}:step{step}".encode()).hexdigest()
                row = {
                    "step": step, "loss": .5 + step * .01, "grad_norm": .9, "lr": 1e-5 * step / 225,
                    "mfu": .025, "elapsed_seconds": elapsed, "input_sha256": digest,
                    "field_sha256": {"input_ids": digest, "loss_mask": "a" * 64, "position_ids": "b" * 64},
                    "local_sequence_lengths": local_lengths, "global_sequence_lengths": list(range(2, 10)),
                    "global_tokens": 44, "non_finite_metrics": [], "scheduler_last_epoch": step,
                    "started_unix": origin + step * 1000,
                    "finished_unix": origin + step * 1000 + elapsed,
                }
                rows.append(row)
                if rank == 0:
                    native.append({"step": step, "data": {
                        "train/loss": row["loss"], "train/grad_norm": row["grad_norm"], "train/lr": row["lr"],
                        "train/global_tokens": row["global_tokens"], "timing/train_step_seconds": elapsed,
                        "perf/max_memory_allocated_gb": 50.0, "perf/max_memory_reserved_gb": 55.0,
                        "perf/cpu_memory_used_gb": 420.0,
                    }})
            _write_rows(directory / f"benchmark_steps_rank{rank}.jsonl", rows)
            _write_json(directory / f"benchmark_manifest_rank{rank}.json", {
                "status": "benchmark_completed", "rank": rank, "requested_steps": 11, "completed_steps": 11,
                "native_total_training_steps": 2250, "native_steps_per_epoch": 1125,
                "scheduler_last_epoch": 11, "all_recorded_metrics_finite": True, "parameter_sampling_error": None,
                "process_placement": {"rss_bytes": 1234, "numa_resident_pages": {"N0": 10, "N1": 990}},
            })
            _write_json(directory / f"benchmark_parameter_samples_rank{rank}.json", {
                "weight": {"global_shape": [6], "local_shape": [3], "dtype": "torch.float32",
                           "flat_positions": [0, 1, 2], "values": [.1, .2, .3]},
            })
        _write_rows(directory / "metrics.jsonl", native)
        _write_json(directory / "timing_summary.json", {
            "status": "benchmark_completed", "completed_train_steps": 11,
            "initialization_seconds": 20, "fit_seconds": 1600, "run_seconds": 1620,
        })
        header = "timestamp, pci.bus_id, utilization.gpu [%], utilization.memory [%], power.draw [W], memory.used [MiB]\n"
        samples = []
        for timestamp, utilization in ((origin, 0), (origin + 1001, 0), (origin + 4002, 80)):
            stamp = datetime.fromtimestamp(timestamp).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
            for device in ("00000000:3F:00.0", "00000000:AF:00.0"):
                samples.append(f"{stamp}, {device}, {utilization} %, 20 %, 250 W, 51200 MiB\n")
        (directory / "gpu_samples.csv").write_text(header + "".join(samples))
    return tmp_path


def _change_json(path, callback):
    data = json.loads(path.read_text())
    callback(data)
    _write_json(path, data)


def _change_rows(path, callback):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    callback(rows)
    _write_rows(path, rows)


def test_paired_report_uses_max_rank_times_and_excludes_initialization(run_dir):
    report = build_report(run_dir)
    assert report["status"] == "passed", report["errors"]
    assert not report["errors"] and not report["incomplete"]
    baseline = report["arms"]["baseline"]
    assert baseline["measured"]["steps"] == list(range(4, 12))
    assert baseline["measured"]["mean_step_seconds"] == 13.0
    assert baseline["measured"]["median_step_seconds"] == 13.0
    assert baseline["measured"]["tokens_per_second"] == 44 / 13
    assert report["comparisons"]["numa_vs_baseline"]["speedup"]["mean_step_ratio"] == 13 / 9
    assert report["comparisons"]["ce_vs_numa"]["speedup"]["mean_step_ratio"] == 9 / 7
    assert report["comparisons"]["combined_vs_baseline"]["speedup"]["mean_step_ratio"] == 13 / 7
    gpu = baseline["gpu_telemetry_measured_steps"]["devices"]["00000000:3F:00.0"]
    assert gpu["mean_utilization_percent"] == 80
    assert gpu["zero_utilization_fraction"] == 0
    assert baseline["gpu_telemetry_entire_arm"]["devices"]["00000000:3F:00.0"]["zero_utilization_fraction"] == 2 / 3
    assert baseline["ranks"]["0"]["gpu_local_numa_resident_page_fraction"] == .99
    assert baseline["native_peak_memory"]["perf/max_memory_allocated_gb"] == 50
    assert "Sixteen CPUs are reserved while eight are active." in report["comparison_limits"]
    assert report["convergence_proven"] is False


@pytest.mark.parametrize("metric,delta", [("loss", .001), ("grad_norm", .01)])
def test_full_trace_numerical_mismatch_fails_without_relaxing_thresholds(run_dir, metric, delta):
    path = run_dir / "numa_ce/benchmark_steps_rank1.jsonl"
    _change_rows(path, lambda rows: rows[1].__setitem__(metric, rows[1][metric] + delta))
    report = build_report(run_dir)
    assert report["status"] == "failed"
    comparison = report["comparisons"]["ce_vs_numa"]["ranks"]["1"][metric]
    assert comparison["mismatch_indices_one_based"] == [2]
    assert comparison["max_absolute_error"] == pytest.approx(delta)
    assert not comparison["passed"]


def test_parameter_sample_mismatch_reports_absolute_and_rms_error(run_dir):
    path = run_dir / "numa_ce/benchmark_parameter_samples_rank0.json"
    _change_json(path, lambda data: data["weight"]["values"].__setitem__(1, .21))
    report = build_report(run_dir)
    assert report["status"] == "failed"
    comparison = report["comparisons"]["ce_vs_numa"]["ranks"]["0"]["parameter_samples"]
    assert comparison["max_absolute_error"] == pytest.approx(.01)
    assert comparison["rms_difference"] == pytest.approx(.01 / 3 ** .5)


@pytest.mark.parametrize("key,value", [
    ("input_sha256", "f" * 64), ("global_tokens", 99), ("lr", 1e-5),
    ("scheduler_last_epoch", 1), ("loss", None), ("elapsed_seconds", None),
])
def test_protocol_or_nonfinite_trace_difference_fails(run_dir, key, value):
    _change_rows(run_dir / "numa/benchmark_steps_rank1.jsonl", lambda rows: rows[4].__setitem__(key, value))
    report = build_report(run_dir)
    assert report["status"] == "failed"
    assert report["errors"]


@pytest.mark.parametrize("key,value", [
    ("status", "completed"), ("native_total_training_steps", 11), ("native_steps_per_epoch", 11),
    ("scheduler_last_epoch", 10), ("parameter_sampling_error", "bad parameter"),
])
def test_manifest_cannot_claim_completion_with_wrong_native_schedule(run_dir, key, value):
    _change_json(run_dir / "numa/benchmark_manifest_rank0.json", lambda data: data.__setitem__(key, value))
    assert build_report(run_dir)["status"] == "failed"


def test_unexpected_checkpoint_or_validation_fails(run_dir):
    (run_dir / "numa/global_step_11").mkdir()
    _change_rows(run_dir / "numa/metrics.jsonl", lambda rows: rows[0]["data"].__setitem__("val/loss", .2))
    report = build_report(run_dir)
    assert report["status"] == "failed"
    assert any("checkpoint" in error for error in report["errors"])
    assert any("validation" in error for error in report["errors"])


def test_missing_artifact_and_partial_steps_are_incomplete(run_dir):
    (run_dir / "numa_ce/benchmark_parameter_samples_rank1.json").unlink()
    _change_rows(run_dir / "numa_ce/benchmark_steps_rank0.jsonl", lambda rows: rows.pop())
    report = build_report(run_dir)
    assert report["status"] == "incomplete"
    assert any("Missing artifact" in issue for issue in report["incomplete"])
    assert any("10/11" in issue for issue in report["incomplete"])


def test_missing_time_windows_do_not_mix_initialization_into_measured_gpu_stats(run_dir):
    for rank in (0, 1):
        def drop_times(rows):
            for row in rows:
                row.pop("started_unix")
                row.pop("finished_unix")
        _change_rows(run_dir / f"baseline/benchmark_steps_rank{rank}.jsonl", drop_times)
    report = build_report(run_dir)
    assert report["status"] == "passed"
    assert report["arms"]["baseline"]["gpu_telemetry_measured_steps"]["status"] == "unavailable"


def test_cli_writes_report_and_exits_nonzero_for_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sft_ab_report", "--run-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 2
    assert json.loads((tmp_path / "ab_report.json").read_text())["status"] == "incomplete"


def test_cli_success_and_failed_ce_numerics_have_distinct_exit_codes(run_dir, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sft_ab_report", "--run-dir", str(run_dir), "--warmup-steps", "3"])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 0
    _change_json(run_dir / "ce_gpu_validation.json", lambda data: data.__setitem__("passed", False))
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 1


def test_invalid_warmup_is_rejected(run_dir):
    with pytest.raises(ValueError, match="warmup_steps"):
        build_report(run_dir, 11)
