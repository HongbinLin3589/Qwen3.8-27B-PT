"""Verify paired SFT benchmark evidence and summarize measured end-to-end speed."""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics


ARMS = ("baseline", "numa", "numa_ce")
RANKS = (0, 1)
STEPS = 11
TOLERANCES = {
    "loss": {"atol": 1e-5, "rtol": 1e-4},
    "grad_norm": {"atol": 1e-4, "rtol": 1e-3},
    "parameter_samples": {"atol": 1e-6, "rtol": 1e-5},
}


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _read(path: Path, incomplete: list, *, lines=False):
    if not path.is_file():
        incomplete.append(f"Missing artifact: {path}")
        return None
    try:
        text = path.read_text()
        return [json.loads(line) for line in text.splitlines() if line.strip()] if lines else json.loads(text)
    except (OSError, json.JSONDecodeError) as error:
        incomplete.append(f"Unreadable or unfinished artifact {path}: {error}")
        return None


def _difference(reference: list, candidate: list, *, atol: float, rtol: float) -> dict:
    absolute = [abs(actual - expected) for expected, actual in zip(reference, candidate, strict=True)]
    relative = [error / max(abs(expected), 1e-30) for error, expected in zip(absolute, reference, strict=True)]
    mismatches = [index + 1 for index, (error, expected) in enumerate(zip(absolute, reference, strict=True))
                  if error > atol + rtol * abs(expected)]
    return {
        "passed": not mismatches and bool(reference),
        "elements": len(reference),
        "atol": atol,
        "rtol": rtol,
        "max_absolute_error": max(absolute, default=None),
        "max_relative_error": max(relative, default=None),
        "rms_difference": math.sqrt(statistics.mean(error * error for error in absolute)) if absolute else None,
        "mismatched_elements": len(mismatches),
        "mismatch_indices_one_based": mismatches,
    }


def _gpu_samples(path: Path, windows=None) -> dict:
    if not path.is_file():
        return {"status": "unavailable", "reason": "gpu_samples.csv is absent"}
    groups = {}
    try:
        with path.open(newline="") as handle:
            for raw in csv.DictReader(handle):
                row = {key.strip(): value.strip() for key, value in raw.items() if key is not None and value is not None}
                device = row.get("pci.bus_id")
                if not device:
                    continue
                if windows is not None:
                    try:
                        timestamp = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                    except (KeyError, ValueError):
                        continue
                    if not any(start <= timestamp <= finish for start, finish in windows):
                        continue
                samples = groups.setdefault(device, {"utilization_percent": [], "power_watts": [], "memory_mib": []})
                for prefix, name in (("utilization.gpu", "utilization_percent"),
                                     ("power.draw", "power_watts"), ("memory.used", "memory_mib")):
                    value = next((value for key, value in row.items() if key.startswith(prefix)), None)
                    try:
                        number = float(value.split()[0]) if value else None
                    except ValueError:
                        number = None
                    if _finite(number):
                        samples[name].append(number)
    except (OSError, csv.Error) as error:
        return {"status": "unavailable", "reason": str(error)}
    result = {}
    for device, samples in groups.items():
        utilization = samples["utilization_percent"]
        result[device] = {
            "utilization_samples": len(utilization),
            "mean_utilization_percent": statistics.mean(utilization) if utilization else None,
            "zero_utilization_fraction": utilization.count(0.0) / len(utilization) if utilization else None,
            "mean_power_watts": statistics.mean(samples["power_watts"]) if samples["power_watts"] else None,
            "peak_memory_gib": max(samples["memory_mib"]) / 1024 if samples["memory_mib"] else None,
        }
    return {
        "status": "available" if result else "unavailable",
        "scope": "Union of both ranks' measured-step windows" if windows is not None else "Entire arm, including initialization and warmup",
        "timestamp_interpretation": "CSV timestamps use the report host's local timezone; ranks record Unix timestamps",
        "devices": result,
    }


def build_report(run_dir: str | Path, warmup_steps: int = 3) -> dict:
    """Read three two-rank arms and return strict verification and timing results.

    Missing evidence yields ``incomplete``; any numerical or protocol mismatch
    yields ``failed`` once evidence is complete. No tolerance is adapted to data.
    """
    if not 0 <= warmup_steps < STEPS:
        raise ValueError(f"warmup_steps must be between 0 and {STEPS - 1}")
    run_dir = Path(run_dir).resolve()
    errors, incomplete = [], []

    def check(condition, message):
        if not condition:
            errors.append(message)
        return bool(condition)

    host_plan = _read(run_dir / "host_plan.json", incomplete)
    ce_validation = _read(run_dir / "ce_gpu_validation.json", incomplete)
    if isinstance(ce_validation, dict):
        check(ce_validation.get("passed") is True, "CE GPU numerical validation did not pass")
        check(ce_validation.get("completed_cases") == ce_validation.get("expected_cases")
              and ce_validation.get("completed_cases", 0) > 0, "CE GPU validation has unfinished cases")
        check(len(ce_validation.get("cases", [])) == ce_validation.get("expected_cases")
              and all(case.get("passed") is True for case in ce_validation.get("cases", [])),
              "CE GPU validation contains a failed case")
    elif ce_validation is not None:
        check(False, "CE GPU validation must be a JSON object")

    arms, evidence = {}, {}
    expected_steps = list(range(1, STEPS + 1))
    for arm in ARMS:
        directory = run_dir / arm
        arm_evidence = {"rows": {}, "parameters": {}, "manifests": {}}
        arm_report = {"ranks": {}, "gpu_telemetry_entire_arm": _gpu_samples(directory / "gpu_samples.csv")}
        check(not list(directory.glob("global_step*")), f"{arm}: unexpected checkpoint directory/artifact")
        for rank in RANKS:
            label = f"{arm}/rank{rank}"
            rows = _read(directory / f"benchmark_steps_rank{rank}.jsonl", incomplete, lines=True)
            manifest = _read(directory / f"benchmark_manifest_rank{rank}.json", incomplete)
            parameters = _read(directory / f"benchmark_parameter_samples_rank{rank}.json", incomplete)
            arm_evidence["rows"][rank] = rows
            arm_evidence["parameters"][rank] = parameters
            arm_evidence["manifests"][rank] = manifest
            arm_report["ranks"][str(rank)] = {"recorded_steps": len(rows) if rows is not None else None}
            if rows is not None:
                if len(rows) < STEPS:
                    incomplete.append(f"{label}: only {len(rows)}/{STEPS} recorded optimizer steps")
                check([row.get("step") for row in rows] == expected_steps, f"{label}: optimizer step sequence differs from 1..11")
                for row in rows:
                    step = row.get("step")
                    for key in ("loss", "grad_norm", "lr", "mfu", "elapsed_seconds", "global_tokens"):
                        check(_finite(row.get(key)), f"{label}/step{step}: missing or non-finite {key}")
                    check(_finite(row.get("elapsed_seconds")) and row["elapsed_seconds"] > 0,
                          f"{label}/step{step}: elapsed_seconds must be positive")
                    check(row.get("non_finite_metrics") == [], f"{label}/step{step}: non-finite native metrics")
                    check(row.get("scheduler_last_epoch") == step, f"{label}/step{step}: scheduler progress differs")
                    lengths, global_lengths = row.get("local_sequence_lengths"), row.get("global_sequence_lengths")
                    check(isinstance(lengths, list) and len(lengths) == 4
                          and all(isinstance(value, int) and 0 < value <= 8192 for value in lengths),
                          f"{label}/step{step}: local batch is not four valid sequences of at most 8192 tokens")
                    check(isinstance(global_lengths, list) and len(global_lengths) == 8
                          and all(isinstance(value, int) and 0 < value <= 8192 for value in global_lengths),
                          f"{label}/step{step}: global batch is not eight valid sequences")
                    if isinstance(global_lengths, list) and all(_finite(value) for value in global_lengths):
                        check(sum(global_lengths) == row.get("global_tokens"), f"{label}/step{step}: global token sum differs")
                    digest = row.get("input_sha256")
                    check(isinstance(digest, str) and len(digest) == 64
                          and all(character in "0123456789abcdef" for character in digest),
                          f"{label}/step{step}: missing or invalid input fingerprint")
                    fields = row.get("field_sha256")
                    check(isinstance(fields, dict) and {"input_ids", "loss_mask", "position_ids"} <= fields.keys(),
                          f"{label}/step{step}: fingerprint is missing input IDs, positions or loss mask")
                    if _finite(row.get("lr")) and isinstance(step, int):
                        check(math.isclose(row["lr"], 1e-5 * step / 225, rel_tol=1e-12, abs_tol=1e-15),
                              f"{label}/step{step}: LR does not preserve 225-step warmup")
            if isinstance(manifest, dict):
                expected = {"status": "benchmark_completed", "rank": rank, "requested_steps": STEPS,
                            "completed_steps": STEPS, "native_total_training_steps": 2250,
                            "native_steps_per_epoch": 1125, "scheduler_last_epoch": STEPS,
                            "all_recorded_metrics_finite": True, "parameter_sampling_error": None}
                for key, value in expected.items():
                    check(key in manifest and manifest.get(key) == value,
                          f"{label}: manifest {key}={manifest.get(key)!r}, expected {value!r}")
                arm_report["ranks"][str(rank)]["process_placement"] = manifest.get("process_placement")
                placement = manifest.get("process_placement") or {}
                if isinstance(host_plan, dict):
                    gpu = next((gpu for gpu in host_plan.get("gpus", []) if gpu.get("local_rank") == rank), None)
                    pages = placement.get("numa_resident_pages", {})
                    if gpu and pages and all(_finite(value) for value in pages.values()) and sum(pages.values()) > 0:
                        arm_report["ranks"][str(rank)]["gpu_local_numa_resident_page_fraction"] = pages.get(f"N{gpu['numa_node']}", 0) / sum(pages.values())
            elif manifest is not None:
                check(False, f"{label}: manifest must be a JSON object")
            if parameters is not None:
                check(isinstance(parameters, dict) and bool(parameters), f"{label}: empty or invalid parameter samples")
                for name, parameter in parameters.items() if isinstance(parameters, dict) else []:
                    check(parameter.get("dtype") == "torch.float32", f"{label}/{name}: parameter shard is not FP32")
                    values = parameter.get("values")
                    check(isinstance(values, list) and bool(values) and all(_finite(value) for value in values),
                          f"{label}/{name}: missing or non-finite parameter samples")
                    check(isinstance(values, list) and len(values) == len(parameter.get("flat_positions", [])),
                          f"{label}/{name}: sample positions and values differ in length")
        timing = _read(directory / "timing_summary.json", incomplete)
        native_rows = _read(directory / "metrics.jsonl", incomplete, lines=True)
        arm_report["timing_summary"] = timing
        if isinstance(timing, dict):
            check(timing.get("status") == "benchmark_completed" and timing.get("completed_train_steps") == STEPS,
                  f"{arm}: timing summary is not an 11-step benchmark completion")
        elif timing is not None:
            check(False, f"{arm}: timing summary must be a JSON object")
        if native_rows is not None:
            if len(native_rows) < STEPS:
                incomplete.append(f"{arm}: only {len(native_rows)}/{STEPS} native metric records")
            check([row.get("step") for row in native_rows] == expected_steps, f"{arm}: native metrics do not cover steps 1..11 exactly")
            peak_memory = {}
            for index, native in enumerate(native_rows):
                data = native.get("data", {})
                check(not any(key.startswith("val/") for key in data), f"{arm}: unexpected validation metrics")
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        check(_finite(value), f"{arm}/native step{native.get('step')}: non-finite {key}")
                    if key in ("perf/max_memory_allocated_gb", "perf/max_memory_reserved_gb", "perf/cpu_memory_used_gb") and _finite(value):
                        peak_memory[key] = max(peak_memory.get(key, 0), value)
                rank0 = arm_evidence["rows"][0]
                if rank0 is not None and index < len(rank0):
                    for native_key, key in (("train/loss", "loss"), ("train/grad_norm", "grad_norm"),
                                            ("train/lr", "lr"), ("train/global_tokens", "global_tokens"),
                                            ("timing/train_step_seconds", "elapsed_seconds")):
                        check(data.get(native_key) == rank0[index].get(key),
                              f"{arm}/native step{native.get('step')}: {native_key} differs from rank0 benchmark")
            arm_report["native_peak_memory"] = peak_memory
            arm_report["cpu_memory_note"] = "Native perf/cpu_memory_used_gb measures the entire node, not this job's RSS"
        rank_rows = arm_evidence["rows"]
        if all(rank_rows[rank] is not None and len(rank_rows[rank]) == STEPS for rank in RANKS):
            for index in range(STEPS):
                left, right = rank_rows[0][index], rank_rows[1][index]
                check(left.get("global_sequence_lengths") == right.get("global_sequence_lengths"),
                      f"{arm}/step{index + 1}: ranks disagree on global sequences")
                check((left.get("local_sequence_lengths") or []) + (right.get("local_sequence_lengths") or [])
                      == left.get("global_sequence_lengths"), f"{arm}/step{index + 1}: rank-local batches do not compose global batch")
            elapsed = [[rank_rows[rank][index].get("elapsed_seconds") for rank in RANKS]
                       for index in range(warmup_steps, STEPS)]
            measured = [max(values) for values in elapsed] if all(_finite(value) and value > 0 for values in elapsed for value in values) else []
            tokens = [row.get("global_tokens") for row in rank_rows[0][warmup_steps:]]
            if measured and all(_finite(value) and value > 0 for value in measured + tokens):
                mfus = [row.get("mfu") for row in rank_rows[0][warmup_steps:]]
                arm_report["measured"] = {
                    "steps": list(range(warmup_steps + 1, STEPS + 1)),
                    "max_rank_elapsed_seconds": measured,
                    "mean_step_seconds": statistics.mean(measured),
                    "median_step_seconds": statistics.median(measured),
                    "min_step_seconds": min(measured),
                    "max_step_seconds": max(measured),
                    "sum_step_seconds": sum(measured),
                    "global_tokens": sum(tokens),
                    "tokens_per_second": sum(tokens) / sum(measured),
                    "mean_rank0_native_mfu": statistics.mean(mfus) if all(_finite(value) for value in mfus) else None,
                }
            windows = []
            for index in range(warmup_steps, STEPS):
                starts = [rank_rows[rank][index].get("started_unix") for rank in RANKS]
                finishes = [rank_rows[rank][index].get("finished_unix") for rank in RANKS]
                if all(_finite(value) for value in starts + finishes) and all(start <= finish for start, finish in zip(starts, finishes, strict=True)):
                    windows.append((min(starts), max(finishes)))
            if len(windows) == STEPS - warmup_steps:
                arm_report["gpu_telemetry_measured_steps"] = _gpu_samples(directory / "gpu_samples.csv", windows)
            else:
                arm_report["gpu_telemetry_measured_steps"] = {"status": "unavailable", "reason": "Complete Unix step windows are absent"}
        arms[arm], evidence[arm] = arm_report, arm_evidence

    comparisons = {}
    for reference_name, candidate_name, name in (("baseline", "numa", "numa_vs_baseline"),
                                                 ("numa", "numa_ce", "ce_vs_numa"),
                                                 ("baseline", "numa_ce", "combined_vs_baseline")):
        comparison = {"reference": reference_name, "candidate": candidate_name, "ranks": {}}
        for rank in RANKS:
            reference, candidate = evidence[reference_name], evidence[candidate_name]
            left, right = reference["rows"][rank], candidate["rows"][rank]
            rank_comparison = {}
            if left is not None and right is not None and len(left) == len(right) == STEPS:
                for index, (expected, actual) in enumerate(zip(left, right, strict=True)):
                    for key in ("input_sha256", "field_sha256", "local_sequence_lengths",
                                "global_sequence_lengths", "global_tokens", "lr", "scheduler_last_epoch"):
                        check(expected.get(key) == actual.get(key), f"{name}/rank{rank}/step{index + 1}: {key} differs")
                for key in ("loss", "grad_norm"):
                    values = [row.get(key) for row in left + right]
                    if all(_finite(value) for value in values):
                        result = _difference([row[key] for row in left], [row[key] for row in right], **TOLERANCES[key])
                        rank_comparison[key] = result
                        check(result["passed"], f"{name}/rank{rank}: {key} exceeds fixed numerical tolerance")
            expected_parameters, actual_parameters = reference["parameters"][rank], candidate["parameters"][rank]
            if isinstance(expected_parameters, dict) and isinstance(actual_parameters, dict):
                check(set(expected_parameters) == set(actual_parameters), f"{name}/rank{rank}: sampled parameter names differ")
                expected_values, actual_values = [], []
                for parameter_name in sorted(set(expected_parameters) & set(actual_parameters)):
                    expected, actual = expected_parameters[parameter_name], actual_parameters[parameter_name]
                    for key in ("global_shape", "local_shape", "dtype", "flat_positions"):
                        check(expected.get(key) == actual.get(key), f"{name}/rank{rank}/{parameter_name}: {key} differs")
                    a, b = expected.get("values", []), actual.get("values", [])
                    if len(a) == len(b) and all(_finite(value) for value in a + b):
                        expected_values.extend(a)
                        actual_values.extend(b)
                    else:
                        check(False, f"{name}/rank{rank}/{parameter_name}: sample values are incomparable")
                result = _difference(expected_values, actual_values, **TOLERANCES["parameter_samples"])
                rank_comparison["parameter_samples"] = result
                check(result["passed"], f"{name}/rank{rank}: parameter samples exceed fixed numerical tolerance")
            comparison["ranks"][str(rank)] = rank_comparison
        reference_timing, candidate_timing = arms[reference_name].get("measured"), arms[candidate_name].get("measured")
        if reference_timing and candidate_timing:
            paired = [a / b for a, b in zip(reference_timing["max_rank_elapsed_seconds"], candidate_timing["max_rank_elapsed_seconds"], strict=True)]
            comparison["speedup"] = {
                "mean_step_ratio": reference_timing["mean_step_seconds"] / candidate_timing["mean_step_seconds"],
                "median_step_ratio": reference_timing["median_step_seconds"] / candidate_timing["median_step_seconds"],
                "paired_step_ratios": paired,
                "paired_ratio_median": statistics.median(paired),
                "improved": candidate_timing["mean_step_seconds"] < reference_timing["mean_step_seconds"],
            }
        comparisons[name] = comparison

    limits = [
        "Eleven early optimizer steps, including three excluded timing warmups, do not establish convergence, final validation quality or full-run duration.",
        "All compared updates remain in the original 225-step LR warmup; cosine decay is not exercised.",
        "Arms execute in a fixed sequence on one allocation; cache, thermal and other workload drift are not randomized away.",
        "Parameter checks sample a few local shard elements, not every model weight or gradient.",
        "Speedups use the maximum synchronized rank wall time for each matched step; GPU CSV covers a wider interval.",
    ]
    if isinstance(host_plan, dict):
        limits.extend(host_plan.get("comparison_limits", []))
    return {
        "schema_version": 1,
        "status": "incomplete" if incomplete else "failed" if errors else "passed",
        "run_dir": str(run_dir),
        "expected_steps_per_arm": STEPS,
        "warmup_steps_discarded": warmup_steps,
        "thresholds": TOLERANCES,
        "incomplete": incomplete,
        "errors": errors,
        "arms": arms,
        "comparisons": comparisons,
        "host_plan": host_plan,
        "ce_gpu_validation_passed": ce_validation.get("passed") if isinstance(ce_validation, dict) else None,
        "comparison_limits": limits,
        "convergence_proven": False,
    }


def main() -> None:
    """Write ``ab_report.json`` and fail if required evidence is incomplete or differs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=3)
    args = parser.parse_args()
    if not 0 <= args.warmup_steps < STEPS:
        parser.error(f"warmup-steps must be between 0 and {STEPS - 1}")
    report = build_report(args.run_dir, args.warmup_steps)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    output = args.run_dir / "ab_report.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(output.resolve()),
                      "incomplete": len(report["incomplete"]), "errors": len(report["errors"])}))
    raise SystemExit({"passed": 0, "failed": 1, "incomplete": 2}[report["status"]])


if __name__ == "__main__":
    main()
