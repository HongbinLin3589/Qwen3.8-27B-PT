"""Check whether two unchanged native SFT runs satisfy fixed repeatability limits."""

import argparse
import json
import math
from pathlib import Path

from legacy.sft.utils.sft_ab_report import TOLERANCES, _difference, _finite, _read


RUNS = ("baseline_repeat1", "baseline_repeat2")
RANKS = (0, 1)
RUNTIME_KEYS = (
    "torch_num_threads", "torch_num_interop_threads", "torch_deterministic_algorithms_enabled",
    "torch_deterministic_warn_only", "cudnn_deterministic", "cudnn_benchmark",
    "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "float32_matmul_precision",
    "cpu_capability", "environment", "versions",
)
ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS",
    "CUBLAS_WORKSPACE_CONFIG", "FLA_TILELANG",
)


def build_report(run_dir: str | Path, steps: int = 6) -> dict:
    """Return a two-rank native repeatability report without using CE microtests.

    ``passed`` means only that the recorded short native runs meet unchanged
    tolerances; ``mismatch`` identifies differing evidence. Missing diagnostics
    or failed training yield ``incomplete`` instead of a numerical conclusion.
    """
    if not 1 <= steps < 1125:
        raise ValueError("steps must be positive and below the first epoch boundary, 1125")
    root = Path(run_dir).resolve()
    incomplete, protocol_errors, numerical_errors, training_failures = [], [], [], []

    def check(condition, message):
        if not condition:
            protocol_errors.append(message)
        return bool(condition)

    evidence, runs = {}, {}
    expected_steps = list(range(1, steps + 1))
    for run in RUNS:
        directory = root / run
        evidence[run] = {"rows": {}, "parameters": {}, "runtime": {}}
        runs[run] = {"ranks": {}}
        check(not list(directory.glob("global_step*")), f"{run}: unexpected checkpoint artifact")
        for rank in RANKS:
            label = f"{run}/rank{rank}"
            rows = _read(directory / f"benchmark_steps_rank{rank}.jsonl", incomplete, lines=True)
            manifest = _read(directory / f"benchmark_manifest_rank{rank}.json", incomplete)
            parameters = _read(directory / f"benchmark_parameter_samples_rank{rank}.json", incomplete)
            evidence[run]["rows"][rank] = rows
            evidence[run]["parameters"][rank] = parameters
            rank_summary = {"recorded_steps": len(rows) if rows is not None else None}
            runs[run]["ranks"][str(rank)] = rank_summary
            if rows is not None:
                if len(rows) < steps:
                    incomplete.append(f"{label}: only {len(rows)}/{steps} complete optimizer steps")
                check([row.get("step") for row in rows] == expected_steps, f"{label}: wrong optimizer step sequence")
                for row in rows:
                    step = row.get("step")
                    for key in ("loss", "grad_norm", "lr", "mfu", "elapsed_seconds", "global_tokens"):
                        check(_finite(row.get(key)), f"{label}/step{step}: missing or non-finite {key}")
                    check(_finite(row.get("elapsed_seconds")) and row["elapsed_seconds"] > 0,
                          f"{label}/step{step}: elapsed time must be positive")
                    check(row.get("non_finite_metrics") == [], f"{label}/step{step}: non-finite metric flag")
                    check(row.get("scheduler_last_epoch") == step, f"{label}/step{step}: scheduler counter differs")
                    if _finite(row.get("lr")) and isinstance(step, int):
                        check(math.isclose(row["lr"], 1e-5 * step / 225, rel_tol=1e-12, abs_tol=1e-15),
                              f"{label}/step{step}: original 225-step warmup is not preserved")
                    lengths, global_lengths = row.get("local_sequence_lengths"), row.get("global_sequence_lengths")
                    check(isinstance(lengths, list) and len(lengths) == 4
                          and all(isinstance(length, int) and 0 < length <= 8192 for length in lengths),
                          f"{label}/step{step}: local batch/length violates four samples of at most 8192 tokens")
                    check(isinstance(global_lengths, list) and len(global_lengths) == 8
                          and all(isinstance(length, int) and 0 < length <= 8192 for length in global_lengths),
                          f"{label}/step{step}: global batch is not eight valid sequences")
                    if isinstance(global_lengths, list) and all(_finite(value) for value in global_lengths):
                        check(row.get("global_tokens") == sum(global_lengths), f"{label}/step{step}: token total differs")
                    digest, fields = row.get("input_sha256"), row.get("field_sha256")
                    check(isinstance(digest, str) and len(digest) == 64
                          and all(character in "0123456789abcdef" for character in digest),
                          f"{label}/step{step}: invalid input fingerprint")
                    check(isinstance(fields, dict) and {"input_ids", "position_ids", "loss_mask"} <= fields.keys(),
                          f"{label}/step{step}: required field fingerprints are missing")
            runtime = None
            if isinstance(manifest, dict):
                rank_summary["training_status"] = manifest.get("status")
                if manifest.get("status") == "failed":
                    training_failures.append(f"{label}: native training failed; numerical repeatability is not established")
                else:
                    check(manifest.get("status") == "benchmark_completed", f"{label}: native run did not end by benchmark limit")
                expected = {"rank": rank, "requested_steps": steps, "completed_steps": steps,
                            "native_total_training_steps": 2250, "native_steps_per_epoch": 1125,
                            "scheduler_last_epoch": steps, "all_recorded_metrics_finite": True,
                            "parameter_sampling_error": None}
                for key, value in expected.items():
                    check(key in manifest and manifest.get(key) == value, f"{label}: manifest {key} differs from {value!r}")
                rank_summary["process_placement"] = manifest.get("process_placement")
                runtime = manifest.get("runtime_diagnostics")
                if not isinstance(runtime, dict) or not runtime:
                    incomplete.append(f"{label}: runtime_diagnostics are absent")
                    runtime = None
                else:
                    missing = [key for key in RUNTIME_KEYS if key not in runtime]
                    for key in ("torch_num_threads", "torch_num_interop_threads"):
                        if type(runtime.get(key)) is not int or runtime[key] <= 0:
                            missing.append(f"valid {key}")
                    for key in ("torch_deterministic_algorithms_enabled", "torch_deterministic_warn_only",
                                "cudnn_deterministic", "cudnn_benchmark", "cuda_matmul_allow_tf32", "cudnn_allow_tf32"):
                        if type(runtime.get(key)) is not bool:
                            missing.append(f"valid {key}")
                    for key in ("float32_matmul_precision", "cpu_capability"):
                        if not isinstance(runtime.get(key), str) or not runtime[key]:
                            missing.append(f"valid {key}")
                    environment = runtime.get("environment")
                    missing.extend(f"environment.{key}" for key in ENVIRONMENT_KEYS
                                   if not isinstance(environment, dict) or key not in environment)
                    if not isinstance(runtime.get("versions"), dict) or not runtime["versions"]:
                        missing.append("nonempty versions")
                    if missing:
                        incomplete.append(f"{label}: incomplete runtime_diagnostics: {', '.join(missing)}")
                    check(runtime.get("torch_num_threads") == 4, f"{label}: native intraop thread count is not four")
                    rank_summary["runtime_diagnostics"] = runtime
            elif manifest is not None:
                check(False, f"{label}: manifest must be a JSON object")
            evidence[run]["runtime"][rank] = runtime
            if parameters is not None:
                check(isinstance(parameters, dict) and bool(parameters), f"{label}: parameter samples are empty or invalid")
                for name, parameter in parameters.items() if isinstance(parameters, dict) else []:
                    values = parameter.get("values")
                    check(parameter.get("dtype") == "torch.float32", f"{label}/{name}: master parameter dtype is not FP32")
                    check(isinstance(values, list) and bool(values) and all(_finite(value) for value in values),
                          f"{label}/{name}: parameter samples are missing or non-finite")
                    check(isinstance(values, list) and len(values) == len(parameter.get("flat_positions", [])),
                          f"{label}/{name}: sample positions and values differ")
        timing = _read(directory / "timing_summary.json", incomplete)
        runs[run]["timing_summary"] = timing
        if isinstance(timing, dict):
            if timing.get("status") == "failed":
                training_failures.append(f"{run}: timing summary records a training failure")
            else:
                check(timing.get("status") == "benchmark_completed", f"{run}: timing summary has wrong completion status")
            check(timing.get("completed_train_steps") == steps, f"{run}: timing summary has wrong step count")
        native = _read(directory / "metrics.jsonl", incomplete, lines=True)
        if native is not None:
            if len(native) < steps:
                incomplete.append(f"{run}: only {len(native)}/{steps} native metric records")
            check([row.get("step") for row in native] == expected_steps, f"{run}: native metrics step sequence differs")
            for index, row in enumerate(native):
                data = row.get("data", {})
                check(not any(key.startswith("val/") for key in data), f"{run}: unexpected validation metrics")
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        check(_finite(value), f"{run}/step{row.get('step')}: native metric {key} is non-finite")
                rank0 = evidence[run]["rows"][0]
                if rank0 is not None and index < len(rank0):
                    for native_key, key in (("train/loss", "loss"), ("train/grad_norm", "grad_norm"),
                                            ("train/lr", "lr"), ("train/global_tokens", "global_tokens"),
                                            ("timing/train_step_seconds", "elapsed_seconds")):
                        check(data.get(native_key) == rank0[index].get(key), f"{run}/step{row.get('step')}: native {native_key} differs from recorder")
        left, right = evidence[run]["rows"][0], evidence[run]["rows"][1]
        if left is not None and right is not None and len(left) == len(right):
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                check(a.get("global_sequence_lengths") == b.get("global_sequence_lengths"),
                      f"{run}/step{index + 1}: ranks disagree on global batch")
                check((a.get("local_sequence_lengths") or []) + (b.get("local_sequence_lengths") or [])
                      == a.get("global_sequence_lengths"), f"{run}/step{index + 1}: local batches do not compose global batch")

    comparisons = {}
    for rank in RANKS:
        comparison = {"steps": []}
        left, right = (evidence[run]["rows"][rank] for run in RUNS)
        if left is not None and right is not None and len(left) == len(right) == steps:
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                for key in ("input_sha256", "field_sha256", "local_sequence_lengths", "global_sequence_lengths",
                            "global_tokens", "lr", "scheduler_last_epoch"):
                    check(a.get(key) == b.get(key), f"rank{rank}/step{index + 1}: repeats differ in {key}")
                per_step = {"step": index + 1}
                for key in ("loss", "grad_norm"):
                    if _finite(a.get(key)) and _finite(b.get(key)):
                        result = _difference([a[key]], [b[key]], **TOLERANCES[key])
                        per_step[key] = {"repeat1": a[key], "repeat2": b[key], **result}
                comparison["steps"].append(per_step)
            for key in ("loss", "grad_norm"):
                if all(_finite(row.get(key)) for row in left + right):
                    result = _difference([row[key] for row in left], [row[key] for row in right], **TOLERANCES[key])
                    comparison[key] = result
                    if not result["passed"]:
                        numerical_errors.append(f"rank{rank}: native {key} exceeds the unchanged repeatability tolerance")
        runtime1, runtime2 = (evidence[run]["runtime"][rank] for run in RUNS)
        if runtime1 is not None and runtime2 is not None:
            runtime_differences = [key for key in RUNTIME_KEYS if runtime1.get(key) != runtime2.get(key)]
            comparison["runtime_configuration_identical"] = not runtime_differences
            comparison["runtime_configuration_differences"] = runtime_differences
            check(not runtime_differences, f"rank{rank}: runtime thread, algorithm, environment or version settings differ: {runtime_differences}")
            comparison["autotuners"] = {run: evidence[run]["runtime"][rank].get("autotuners") for run in RUNS}
        a, b = (evidence[run]["parameters"][rank] for run in RUNS)
        if isinstance(a, dict) and isinstance(b, dict):
            check(set(a) == set(b), f"rank{rank}: sampled parameter names differ")
            values1, values2 = [], []
            for name in sorted(set(a) & set(b)):
                for key in ("global_shape", "local_shape", "dtype", "flat_positions"):
                    check(a[name].get(key) == b[name].get(key), f"rank{rank}/{name}: sampled parameter {key} differs")
                x, y = a[name].get("values", []), b[name].get("values", [])
                if len(x) == len(y) and all(_finite(value) for value in x + y):
                    values1.extend(x)
                    values2.extend(y)
                else:
                    check(False, f"rank{rank}/{name}: parameter samples cannot be paired")
            result = _difference(values1, values2, **TOLERANCES["parameter_samples"])
            comparison["parameter_samples"] = result
            if not result["passed"]:
                numerical_errors.append(f"rank{rank}: native parameter samples exceed the unchanged tolerance")
        comparisons[str(rank)] = comparison

    ready = not incomplete and not training_failures
    valid = ready and not protocol_errors
    repeatable = not numerical_errors if valid else None
    status = "incomplete" if not ready else "mismatch" if protocol_errors or numerical_errors else "passed"
    return {
        "schema_version": 1, "status": status, "run_dir": str(root), "steps": steps,
        "thresholds": TOLERANCES, "training_failures": training_failures,
        "incomplete": incomplete, "protocol_errors": protocol_errors, "numerical_errors": numerical_errors,
        "comparison_protocol_valid": valid, "native_repeatability_standard_met": repeatable,
        "runs": runs, "rank_comparisons": comparisons,
        "limitations": [
            "This compares two native runs; a CE microbenchmark pass does not establish whole-model trajectory agreement.",
            "A mismatch shows failure of the stated repeatability standard; it does not identify a kernel or other cause.",
            "Only a few parameter-shard elements and global gradient norms are compared, not every parameter or gradient.",
            "These short runs use the original warmup schedule and do not prove convergence or final validation quality.",
            "Autotuner caches are recorded separately; changing cache contents alone is not treated as changing a hyperparameter.",
        ],
    }


def main() -> None:
    """Write ``repro_report.json``; exit 0/1/2 for pass/mismatch/incomplete."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.steps < 1125:
        parser.error("steps must be positive and below the first epoch boundary, 1125")
    report = build_report(args.run_dir, args.steps)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    output = args.run_dir / "repro_report.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(output.resolve()),
                      "native_repeatability_standard_met": report["native_repeatability_standard_met"],
                      "training_failures": report["training_failures"]}))
    raise SystemExit({"passed": 0, "mismatch": 1, "incomplete": 2}[report["status"]])


if __name__ == "__main__":
    main()
