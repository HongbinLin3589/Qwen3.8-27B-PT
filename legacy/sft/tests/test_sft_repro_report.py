"""Test native repeatability decisions independently of any GPU CE microtest."""

from copy import deepcopy
import hashlib
import json
import sys

import pytest

from legacy.sft.utils.sft_ab_report import TOLERANCES
from legacy.sft.utils.sft_repro_report import RUNS, build_report, main


def _json(path, value):
    path.write_text(json.dumps(value) + "\n")


def _rows(path, value):
    path.write_text("".join(json.dumps(row) + "\n" for row in value))


@pytest.fixture
def run_dir(tmp_path):
    runtime = {
        "torch_num_threads": 4, "torch_num_interop_threads": 16,
        "torch_deterministic_algorithms_enabled": False, "torch_deterministic_warn_only": False,
        "cudnn_deterministic": False, "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": True,
        "float32_matmul_precision": "highest", "cpu_capability": "AVX512",
        "environment": {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
                        "RAYON_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": None, "FLA_TILELANG": "1"},
        "versions": {"torch": "2.11.0+cu129", "verl": "0.9.1"},
        "autotuners": [{"object_path": "kernel", "cached_config_count": 1}],
        "coverage": {"not_a_root_cause_proof": True},
    }
    for run in RUNS:
        directory = tmp_path / run
        directory.mkdir()
        native = []
        for rank in (0, 1):
            records = []
            for step in range(1, 7):
                fingerprint = hashlib.sha256(f"rank{rank}/step{step}".encode()).hexdigest()
                row = {
                    "step": step, "loss": .35 + step * .01, "grad_norm": 2.0, "lr": step * 1e-5 / 225,
                    "mfu": .02, "elapsed_seconds": 111.0, "global_tokens": 44,
                    "global_sequence_lengths": list(range(2, 10)),
                    "local_sequence_lengths": list(range(2, 6)) if rank == 0 else list(range(6, 10)),
                    "input_sha256": fingerprint,
                    "field_sha256": {"input_ids": fingerprint, "position_ids": "a" * 64, "loss_mask": "b" * 64},
                    "scheduler_last_epoch": step, "non_finite_metrics": [],
                }
                records.append(row)
                if rank == 0:
                    native.append({"step": step, "data": {
                        "train/loss": row["loss"], "train/grad_norm": row["grad_norm"],
                        "train/lr": row["lr"], "train/global_tokens": 44,
                        "timing/train_step_seconds": 111.0,
                    }})
            _rows(directory / f"benchmark_steps_rank{rank}.jsonl", records)
            _json(directory / f"benchmark_manifest_rank{rank}.json", {
                "status": "benchmark_completed", "rank": rank, "requested_steps": 6, "completed_steps": 6,
                "native_total_training_steps": 2250, "native_steps_per_epoch": 1125,
                "scheduler_last_epoch": 6, "all_recorded_metrics_finite": True,
                "parameter_sampling_error": None, "runtime_diagnostics": deepcopy(runtime),
                "process_placement": {"cpu_affinity": list(range(8)), "rss_bytes": 1024},
            })
            _json(directory / f"benchmark_parameter_samples_rank{rank}.json", {
                "weight": {"global_shape": [6], "local_shape": [3], "dtype": "torch.float32",
                           "flat_positions": [0, 1, 2], "values": [.1, .2, .3]},
            })
        _rows(directory / "metrics.jsonl", native)
        _json(directory / "timing_summary.json", {"status": "benchmark_completed", "completed_train_steps": 6})
    return tmp_path


def _change(path, callback, *, rows=False):
    data = [json.loads(line) for line in path.read_text().splitlines()] if rows else json.loads(path.read_text())
    callback(data)
    (_rows if rows else _json)(path, data)


def test_identical_native_runs_pass_without_ce_validation(run_dir):
    report = build_report(run_dir)
    assert report["status"] == "passed", report
    assert report["comparison_protocol_valid"]
    assert report["native_repeatability_standard_met"]
    assert report["thresholds"] == TOLERANCES
    assert not (run_dir / "ce_gpu_validation.json").exists()
    comparison = report["rank_comparisons"]["0"]
    assert comparison["loss"]["max_absolute_error"] == 0
    assert comparison["grad_norm"]["rms_difference"] == 0
    assert comparison["runtime_configuration_identical"]
    assert len(comparison["steps"]) == 6


def test_native_gradient_mismatch_is_numerical_even_if_ce_microtest_passes(run_dir):
    _json(run_dir / "ce_gpu_validation.json", {"passed": True})
    path = run_dir / "baseline_repeat2/benchmark_steps_rank1.jsonl"
    _change(path, lambda rows: rows[4].__setitem__("grad_norm", 8.0), rows=True)
    report = build_report(run_dir)
    assert report["status"] == "mismatch"
    assert report["comparison_protocol_valid"]
    assert report["native_repeatability_standard_met"] is False
    comparison = report["rank_comparisons"]["1"]
    assert comparison["grad_norm"]["mismatch_indices_one_based"] == [5]
    assert comparison["steps"][4]["grad_norm"]["max_absolute_error"] == 6
    assert comparison["grad_norm"]["atol"] == TOLERANCES["grad_norm"]["atol"]


def test_native_loss_mismatch_and_parameter_sample_mismatch_are_reported(run_dir):
    _change(run_dir / "baseline_repeat2/benchmark_steps_rank1.jsonl",
            lambda rows: rows[2].__setitem__("loss", .5), rows=True)
    _change(run_dir / "baseline_repeat2/benchmark_parameter_samples_rank0.json",
            lambda data: data["weight"]["values"].__setitem__(1, .21))
    report = build_report(run_dir)
    assert report["status"] == "mismatch"
    assert report["rank_comparisons"]["1"]["loss"]["mismatch_indices_one_based"] == [3]
    assert report["rank_comparisons"]["0"]["parameter_samples"]["max_absolute_error"] == pytest.approx(.01)


@pytest.mark.parametrize("key,value", [
    ("input_sha256", "f" * 64), ("lr", 1e-5), ("scheduler_last_epoch", 1),
    ("global_tokens", 100), ("loss", None),
])
def test_input_schedule_or_nonfinite_protocol_failure_does_not_claim_repeatability(run_dir, key, value):
    _change(run_dir / "baseline_repeat2/benchmark_steps_rank1.jsonl",
            lambda rows: rows[2].__setitem__(key, value), rows=True)
    report = build_report(run_dir)
    assert report["status"] == "mismatch"
    assert not report["comparison_protocol_valid"]
    assert report["native_repeatability_standard_met"] is None


@pytest.mark.parametrize("key,value", [
    ("torch_num_threads", 2), ("cudnn_benchmark", True), ("cuda_matmul_allow_tf32", True),
    ("cpu_capability", "AVX2"), ("float32_matmul_precision", "high"),
])
def test_thread_or_algorithm_flag_difference_invalidates_comparison(run_dir, key, value):
    _change(run_dir / "baseline_repeat2/benchmark_manifest_rank1.json",
            lambda data: data["runtime_diagnostics"].__setitem__(key, value))
    report = build_report(run_dir)
    assert report["status"] == "mismatch"
    assert key in report["rank_comparisons"]["1"]["runtime_configuration_differences"]
    assert report["native_repeatability_standard_met"] is None


@pytest.mark.parametrize("mutation", [
    lambda data: data.pop("runtime_diagnostics"),
    lambda data: data["runtime_diagnostics"].pop("cudnn_benchmark"),
    lambda data: data["runtime_diagnostics"].__setitem__("cudnn_benchmark", None),
    lambda data: data["runtime_diagnostics"]["environment"].pop("FLA_TILELANG"),
])
def test_missing_runtime_diagnostics_are_explicitly_incomplete(run_dir, mutation):
    _change(run_dir / "baseline_repeat2/benchmark_manifest_rank1.json", mutation)
    report = build_report(run_dir)
    assert report["status"] == "incomplete"
    assert any("runtime_diagnostics" in item for item in report["incomplete"])
    assert report["native_repeatability_standard_met"] is None


def test_autotuner_cache_variation_is_recorded_without_claiming_a_cause(run_dir):
    _change(run_dir / "baseline_repeat2/benchmark_manifest_rank1.json",
            lambda data: data["runtime_diagnostics"]["autotuners"][0].__setitem__("cached_config_count", 9))
    report = build_report(run_dir)
    assert report["status"] == "passed"
    cache = report["rank_comparisons"]["1"]["autotuners"]
    assert cache["baseline_repeat1"] != cache["baseline_repeat2"]


def test_training_failure_is_distinguished_from_a_numerical_mismatch(run_dir):
    _change(run_dir / "baseline_repeat2/benchmark_manifest_rank1.json", lambda data: data.__setitem__("status", "failed"))
    report = build_report(run_dir)
    assert report["status"] == "incomplete"
    assert report["training_failures"]
    assert not report["numerical_errors"]
    assert report["native_repeatability_standard_met"] is None


def test_missing_or_partial_artifacts_are_incomplete(run_dir):
    (run_dir / "baseline_repeat2/benchmark_parameter_samples_rank1.json").unlink()
    _change(run_dir / "baseline_repeat2/benchmark_steps_rank1.jsonl", lambda rows: rows.pop(), rows=True)
    report = build_report(run_dir)
    assert report["status"] == "incomplete"
    assert any("5/6" in item for item in report["incomplete"])


def test_checkpoint_and_validation_are_forbidden(run_dir):
    (run_dir / "baseline_repeat2/global_step_6").mkdir()
    _change(run_dir / "baseline_repeat2/metrics.jsonl", lambda rows: rows[0]["data"].__setitem__("val/loss", .1), rows=True)
    report = build_report(run_dir)
    assert report["status"] == "mismatch"
    assert any("checkpoint" in item for item in report["protocol_errors"])
    assert any("validation" in item for item in report["protocol_errors"])


def test_cli_pass_mismatch_and_incomplete_exit_codes(run_dir, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sft_repro_report", "--run-dir", str(run_dir), "--steps", "6"])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 0
    _change(run_dir / "baseline_repeat2/benchmark_steps_rank1.jsonl", lambda rows: rows[2].__setitem__("loss", .6), rows=True)
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 1
    (run_dir / "baseline_repeat2/benchmark_manifest_rank1.json").unlink()
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 2
    assert json.loads((run_dir / "repro_report.json").read_text())["status"] == "incomplete"


@pytest.mark.parametrize("steps", [0, -1, 1125])
def test_steps_cannot_trigger_an_epoch_boundary(run_dir, steps):
    with pytest.raises(ValueError, match="steps"):
        build_report(run_dir, steps)
