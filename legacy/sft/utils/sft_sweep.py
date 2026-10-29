"""Launch isolated native SFT performance arms and verify their recorded workload."""

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import statistics
import subprocess
import sys
from time import perf_counter


OUTPUT_HEADS = ("native", "verl_torch", "verl_triton", "liger", "cce_exact")


@dataclass(frozen=True)
class SweepConfig:
    """Describe a frozen dataset pool and the independent native-training arms."""

    repo: Path
    model: Path
    train_files: tuple[Path, ...]
    val_files: tuple[Path, ...]
    output: Path
    gpu_mask: tuple[int, ...]
    global_batches: tuple[int, ...] = (24,)
    micro_batches: tuple[int, ...] = (1,)
    attn_backends: tuple[str, ...] = ("sdpa",)
    dynamic: tuple[bool, ...] = (False,)
    length_bucketing: tuple[bool, ...] = (False,)
    output_head_backends: tuple[str, ...] = ("native",)
    sequence_parallel_sizes: tuple[int, ...] = (1,)
    remove_padding: tuple[bool, ...] = (False,)
    mode: str = "screen"
    stage: str = "sweep"
    warmup_steps: int = 2
    measure_steps: int = 4
    warmup_passes: int = 1
    measure_passes: int = 1
    max_length: int = 8192
    max_token_len_per_gpu: int = 8192
    timeout_seconds: float = 1800
    seed: int = 20261028
    numa_binding: bool = False
    python: str = sys.executable

    def validate(self, pool_rows: int) -> None:
        """Reject ambiguous devices, incompatible batches and incomplete matched pools."""
        if not self.gpu_mask or len(set(self.gpu_mask)) != len(self.gpu_mask) or min(self.gpu_mask) < 0:
            raise ValueError("gpu_mask must explicitly contain distinct non-negative physical GPU IDs")
        if self.mode not in ("screen", "matched"):
            raise ValueError("mode must be screen or matched")
        if not self.stage or not self.train_files:
            raise ValueError("A stage and at least one training file are required")
        if self.warmup_steps < 0 or self.warmup_passes < 0:
            raise ValueError("Warmup counts cannot be negative")
        if min(self.measure_steps, self.measure_passes, self.max_length, self.max_token_len_per_gpu) <= 0:
            raise ValueError("Measurement counts and token limits must be positive")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        choices = (self.global_batches, self.micro_batches, self.attn_backends, self.dynamic,
                   self.length_bucketing, self.output_head_backends,
                   self.sequence_parallel_sizes, self.remove_padding)
        if any(not values or len(set(values)) != len(values) for values in choices):
            raise ValueError("Matrix dimensions must contain distinct values and cannot be empty")
        if not set(self.attn_backends) <= {"sdpa", "flash_attention_2"}:
            raise ValueError("Attention backends must be sdpa or flash_attention_2")
        if not set(self.output_head_backends) <= set(OUTPUT_HEADS):
            raise ValueError(f"Output heads must be selected from {OUTPUT_HEADS}")
        for parallel, padding, attention in itertools.product(
            self.sequence_parallel_sizes, self.remove_padding, self.attn_backends,
        ):
            if parallel not in (1, 2, 4, 8) or len(self.gpu_mask) % parallel:
                raise ValueError("Sequence parallel size must be 1, 2, 4 or 8 and divide the GPU count")
            if parallel > 1 and not padding:
                raise ValueError("Sequence parallel size > 1 requires remove_padding=true")
            if parallel > 1 and attention != "flash_attention_2":
                raise ValueError("Sequence parallel size > 1 requires flash_attention_2 for attention communication")
        for batch, micro, parallel, dynamic in itertools.product(
            self.global_batches, self.micro_batches, self.sequence_parallel_sizes, self.dynamic,
        ):
            dp_size = len(self.gpu_mask) // parallel
            divisor = dp_size if dynamic else dp_size * micro
            if min(batch, micro) <= 0 or batch % divisor:
                raise ValueError("Global batch must be divisible by DP size, and by DP * micro batch in static mode")
            if pool_rows < batch:
                raise ValueError("The frozen pool must contain at least one complete global batch")
            if self.mode == "matched" and pool_rows % batch:
                raise ValueError("Matched mode requires pool rows divisible by every global batch")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _pool_identity(files: tuple[Path, ...]) -> dict:
    import pyarrow.parquet as pq

    identities = []
    for path in files:
        identity = _file_identity(path)
        identity["rows"] = pq.ParquetFile(path).metadata.num_rows
        identities.append(identity)
    return {"files": identities, "rows": sum(row["rows"] for row in identities),
            "sha256": _digest(identities)}


def build_arms(config: SweepConfig, pool_rows: int) -> list[dict]:
    """Build concrete torchrun commands; screen windows may consume different samples."""
    config.validate(pool_rows)
    arms = []
    dimensions = itertools.product(config.global_batches, config.micro_batches, config.attn_backends,
                                   config.dynamic, config.length_bucketing, config.output_head_backends,
                                   config.sequence_parallel_sizes, config.remove_padding)
    for index, (batch, micro, attention, dynamic, bucketing, head, parallel, padding) in enumerate(dimensions):
        steps_per_pass = pool_rows // batch
        warmup = config.warmup_steps if config.mode == "screen" else config.warmup_passes * steps_per_pass
        measured = config.measure_steps if config.mode == "screen" else config.measure_passes * steps_per_pass
        requested = warmup + measured
        name = (f"{index:03d}-g{batch}-m{micro}-{attention}-d{int(dynamic)}-b{int(bucketing)}"
                f"-{head}-sp{parallel}-p{int(padding)}")
        directory = config.output.resolve() / name
        options = {
            "model.path": str(config.model.resolve()),
            "model.use_remove_padding": padding,
            "model.enable_gradient_checkpointing": True,
            "model.lora_rank": 0,
            "model.use_fused_kernels": head != "native",
            "model.fused_kernel_options.impl_backend": "triton" if head == "verl_triton" else "torch",
            "+model.override_config.attn_implementation": attention,
            "engine.strategy": "fsdp2", "engine.fsdp_size": len(config.gpu_mask),
            "engine.ulysses_sequence_parallel_size": parallel,
            "engine.model_dtype": "float32", "engine.offload_policy": False,
            "engine.param_offload": False, "engine.optimizer_offload": False,
            "engine.use_torch_compile": False, "engine.use_no_sync_for_gradient_accumulation": False,
            "engine.seed": config.seed,
            "data.train_files": [str(path.resolve()) for path in config.train_files],
            # Native SFT always validates and saves at its final step, even with frequencies -1.
            "data.val_files": None,
            "data.custom_cls.path": "pkg://qwenpt.train.verl.dataset",
            "data.custom_cls.name": "QwenSFTDataset",
            "data.enable_thinking_default": True,
            "+data.loss_scope": "all_assistant",
            "+data.apply_chat_template_kwargs.preserve_thinking": True,
            "data.pad_mode": "no_padding", "data.truncation": "error",
            "data.max_length": config.max_length,
            "data.train_batch_size": batch, "data.micro_batch_size_per_gpu": micro,
            "data.max_token_len_per_gpu": config.max_token_len_per_gpu,
            "data.use_dynamic_bsz": dynamic, "data.num_workers": 0,
            "+data.length_bucketing": bucketing,
            "optim.lr": 1e-5, "optim.lr_scheduler_type": "cosine",
            "optim.lr_warmup_steps_ratio": 0.1,
            "trainer.default_local_dir": str(directory),
            "trainer.project_name": "sft-performance-sweep", "trainer.experiment_name": name,
            "trainer.total_epochs": math.ceil((requested + 1) / steps_per_pass),
            "trainer.total_training_steps": max(1000, requested + 1),
            "trainer.seed": config.seed, "trainer.n_gpus_per_node": len(config.gpu_mask),
            "trainer.logger": [], "trainer.save_freq": -1, "trainer.test_freq": -1,
            "trainer.resume_mode": "disable", "+trainer.benchmark_steps": requested,
            "+trainer.output_head_backend": head, "+trainer.numa_binding": config.numa_binding,
            "hydra.run.dir": str(directory / "hydra"), "hydra.job.chdir": False,
        }
        command = [config.python, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                   f"--nproc-per-node={len(config.gpu_mask)}", "--module", "legacy.sft.entrypoint"]
        command.extend(f"{key}={json.dumps(value, separators=(',', ':'))}" for key, value in options.items())
        environment = {
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, config.gpu_mask)),
            "PYTHONPATH": os.pathsep.join((str(config.repo.resolve()), str(config.repo.resolve() / "src"))),
            "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled", "WANDB_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
            "CUDA_HOME": "/usr/local/cuda-12.8", "FLA_TILELANG": "1",
            "TRITON_CACHE_DIR": str(config.output.resolve() / "kernel_cache" / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(config.output.resolve() / "kernel_cache" / "inductor"),
        }
        arms.append({"name": name, "directory": str(directory), "command": command,
                     "shell_command": shlex.join(command), "environment": environment,
                     "global_batch": batch, "micro_batch": micro, "attention": attention,
                     "dynamic": dynamic, "length_bucketing": bucketing, "output_head_backend": head,
                     "sequence_parallel_size": parallel, "data_parallel_size": len(config.gpu_mask) // parallel,
                     "remove_padding": padding,
                     "warmup_steps": warmup, "measure_steps": measured, "requested_steps": requested,
                     "measured_samples_expected": measured * batch,
                     "mode": config.mode, "stage": config.stage})
    return arms


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _unique_dp_rows(rows: list[dict], num_gpus: int, parallel: int) -> list[tuple[int, dict]]:
    if parallel not in (1, 2, 4, 8) or num_gpus % parallel:
        raise ValueError("Recorded sequence parallel topology cannot divide world size")
    expected_dp = num_gpus // parallel
    groups = {}
    for rank, row in enumerate(rows):
        dp_size = row.get("data_parallel_size")
        sp_size = row.get("sequence_parallel_size")
        dp_rank = row.get("data_parallel_rank")
        if parallel == 1:
            dp_size = num_gpus if dp_size is None else dp_size
            sp_size = 1 if sp_size is None else sp_size
            dp_rank = rank if dp_rank is None else dp_rank
        if (type(dp_size) is not int or dp_size != expected_dp
                or type(sp_size) is not int or sp_size != parallel
                or type(dp_rank) is not int or not 0 <= dp_rank < expected_dp):
            raise ValueError(f"rank{rank}: recorded data/sequence parallel topology differs from arm")
        groups.setdefault(dp_rank, []).append(row)
    if set(groups) != set(range(expected_dp)) or any(len(group) != parallel for group in groups.values()):
        raise ValueError("Recorded ranks do not form complete sequence-parallel copies of each DP rank")
    for dp_rank, group in groups.items():
        first = group[0]
        if parallel > 1 and not isinstance(first.get("sample_sha256"), list):
            raise ValueError(f"DP rank{dp_rank}: SP copies lack per-sample fingerprints")
        for replica in group[1:]:
            for key in ("input_sha256", "sample_sha256", "local_sequence_lengths", "local_supervised_tokens"):
                if replica.get(key) != first.get(key):
                    raise ValueError(f"DP rank{dp_rank}: sequence-parallel copies differ in {key}")
    return [(rank, groups[rank][0]) for rank in sorted(groups)]


def summarize_arm(directory: Path, arm: dict, num_gpus: int) -> dict:
    """Count each DP batch once while timing and memory include every SP/world rank."""
    ranks = []
    requested = arm["requested_steps"]
    for rank in range(num_gpus):
        rows = [json.loads(line) for line in (directory / f"benchmark_steps_rank{rank}.jsonl").read_text().splitlines()
                if line.strip()]
        manifest = json.loads((directory / f"benchmark_manifest_rank{rank}.json").read_text())
        if manifest.get("status") != "benchmark_completed" or manifest.get("completed_steps") != requested:
            raise ValueError(f"rank{rank}: controlled benchmark did not complete all requested steps")
        if [row.get("step") for row in rows] != list(range(1, requested + 1)):
            raise ValueError(f"rank{rank}: missing, repeated or unexpected recorded steps")
        for row in rows:
            if not _finite(row.get("elapsed_seconds")) or row["elapsed_seconds"] <= 0:
                raise ValueError(f"rank{rank}: invalid synchronized elapsed time")
            if row.get("non_finite_metrics"):
                raise ValueError(f"rank{rank}: non-finite training metrics")
            lengths = row.get("local_sequence_lengths")
            if not isinstance(lengths, list) or not lengths or any(type(length) is not int or length <= 0 for length in lengths):
                raise ValueError(f"rank{rank}: invalid local sequence lengths")
            fingerprint = row.get("input_sha256")
            if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
                raise ValueError(f"rank{rank}: missing valid input fingerprint")
        ranks.append(rows)
    parallel = arm.get("sequence_parallel_size", 1)
    unique_steps = [_unique_dp_rows([rows[step] for rows in ranks], num_gpus, parallel)
                    for step in range(requested)]
    start = arm["warmup_steps"]
    measured = [rows[start:] for rows in ranks]
    seconds = [max(rows[step]["elapsed_seconds"] for rows in measured) for step in range(arm["measure_steps"])]
    token_counts, sequence_lengths, samples, batch_hashes, supervised = [], [], [], [], []
    memory = {"peak_memory_allocated_bytes": [], "peak_memory_reserved_bytes": []}
    sample_hashes_available = supervised_available = True
    for step in range(arm["measure_steps"]):
        unique = unique_steps[start + step]
        local_tokens = sum(sum(row["local_sequence_lengths"]) for _, row in unique)
        token_counts.append(local_tokens)
        for rank, rows in enumerate(measured):
            row = rows[step]
            global_tokens = row.get("global_tokens")
            if global_tokens is not None and global_tokens != local_tokens:
                raise ValueError("Recorded global_tokens differs from summed rank-local input tokens")
            batch_hashes.append([start + step + 1, rank, row["input_sha256"]])
            for key, values in memory.items():
                value = row.get(key)
                if type(value) is int and value >= 0:
                    values.append(value)
        for _dp_rank, row in unique:
            lengths = row["local_sequence_lengths"]
            sequence_lengths.extend(lengths)
            hashes = row.get("sample_sha256")
            if (not isinstance(hashes, list) or len(hashes) != len(lengths)
                    or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes)):
                sample_hashes_available = False
            else:
                samples.extend(hashes)
            count = row.get("local_supervised_tokens")
            if type(count) is not int or not 0 <= count <= sum(lengths):
                supervised_available = False
            else:
                supervised.append(count)
    if len(sequence_lengths) != arm["measured_samples_expected"]:
        raise ValueError("Measured sample count differs from the declared global-batch window")
    elapsed = sum(seconds)
    return {
        "world_size": num_gpus, "data_parallel_size": num_gpus // parallel,
        "sequence_parallel_size": parallel,
        "measured_steps": len(seconds), "measured_samples": len(sequence_lengths),
        "measured_input_tokens": sum(token_counts), "synchronized_train_seconds": elapsed,
        "input_tokens_per_second": sum(token_counts) / elapsed,
        "measured_supervised_tokens": sum(supervised) if supervised_available else None,
        "supervised_tokens_per_second": sum(supervised) / elapsed if supervised_available else None,
        "mean_step_seconds": statistics.mean(seconds), "median_step_seconds": statistics.median(seconds),
        "max_step_seconds": max(seconds), "step_seconds": seconds, "step_input_tokens": token_counts,
        "sequence_length_min": min(sequence_lengths), "sequence_length_max": max(sequence_lengths),
        "sequence_length_mean": statistics.mean(sequence_lengths),
        "input_trace_sha256": _digest(batch_hashes),
        "sample_multiset_sha256": _digest(sorted(samples)) if sample_hashes_available else None,
        "sample_sha256": sorted(samples) if sample_hashes_available else None,
        **{key: max(values) if len(values) == len(seconds) * num_gpus else None for key, values in memory.items()},
        "timing_scope": "Sum of synchronized native train calls; initialization, dataloading and teardown excluded",
    }


def _stderr_tail(path: Path, count: int = 16000) -> str:
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - count))
        return handle.read().decode(errors="replace")


def _stop_process_group(process: subprocess.Popen) -> int:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()


def run_arm(config: SweepConfig, arm: dict) -> dict:
    """Run one process group, retain its logs and classify resource failures explicitly."""
    directory = Path(arm["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    _write_json(directory / "command.json", arm)
    environment = os.environ.copy()
    for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "FLA_CI_ENV", "FLA_DISABLE_BACKEND_DISPATCH"):
        environment.pop(key, None)
    environment.update(arm["environment"])
    started = perf_counter()
    timed_out = False
    with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(arm["command"], cwd=config.repo, env=environment,
                                   stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            returncode = process.wait(timeout=config.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = _stop_process_group(process)
        except BaseException:
            # An interrupted sweep must not leave detached torchrun workers using GPUs.
            _stop_process_group(process)
            raise
    result = {"name": arm["name"], "status": "completed", "returncode": returncode,
              "wall_seconds": perf_counter() - started, "configuration": arm,
              "stdout": str(directory / "stdout.log"), "stderr": str(directory / "stderr.log")}
    error = _stderr_tail(directory / "stderr.log")
    if timed_out:
        result.update(status="timeout", error=error)
    elif returncode != 0:
        oom = re.search(r"CUDA out of memory|torch(?:\.cuda)?\.OutOfMemoryError|CUDA error: out of memory", error, re.I)
        result.update(status="oom" if oom else "failed", error=error)
    else:
        try:
            result["measurement"] = summarize_arm(directory, arm, len(config.gpu_mask))
        except (OSError, ValueError, KeyError, TypeError) as failure:
            result.update(status="invalid_evidence", error=f"{type(failure).__name__}: {failure}", stderr_tail=error)
    _write_json(directory / "result.json", result)
    return result


def build_report(results: list[dict], mode: str) -> dict:
    """Keep screened speeds descriptive and require matching workloads for final comparisons."""
    results = copy.deepcopy(results)
    completed = [result for result in results if result["status"] == "completed"]
    reference = completed[0] if completed else None
    for result in completed:
        measurement = result["measurement"]
        expected = reference["measurement"]
        same_samples = (measurement["sample_multiset_sha256"] is not None
                        and measurement["sample_multiset_sha256"] == expected["sample_multiset_sha256"])
        same_tokens = measurement["measured_input_tokens"] == expected["measured_input_tokens"]
        result["workload_comparison"] = {
            "reference": reference["name"], "same_sample_multiset": same_samples,
            "same_input_tokens": same_tokens,
            "comparable": mode == "matched" and same_samples and same_tokens,
        }
        if mode == "matched" and not (same_samples and same_tokens):
            result["status"] = "workload_mismatch"
    failed = any(result["status"] in ("failed", "invalid_evidence", "workload_mismatch") for result in results)
    status = "failed" if failed else "completed" if completed else "no_completed_arms"
    return {"schema_version": 1, "mode": mode, "status": status,
            "completed_arms": sum(result["status"] == "completed" for result in results),
            "resource_failure_arms": sum(result["status"] in ("oom", "timeout") for result in results),
            "comparison_note": ("Screening uses the same frozen pool; measured sample sets may differ."
                                if mode == "screen" else "Matched windows cover complete frozen-pool passes."),
            "convergence_note": "Short benchmark scheduling and warmup provide no convergence evidence.",
            "arms": results}


def run_sweep(config: SweepConfig, *, dry_run: bool = False) -> dict:
    """Freeze local inputs, write reviewable commands and optionally execute each arm."""
    if not (config.repo / "legacy/sft/entrypoint.py").is_file():
        raise ValueError("repo must contain legacy/sft/entrypoint.py")
    if not (config.model / "config.json").is_file():
        raise ValueError("model must be a local model directory containing config.json")
    pool = _pool_identity(config.train_files)
    validation = [_file_identity(path) for path in config.val_files]
    arms = build_arms(config, pool["rows"])
    config.output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "stage": config.stage, "mode": config.mode,
                "config": json.loads(json.dumps(asdict(config), default=str)),
                "train_pool": pool, "validation_files_not_loaded": validation, "arms": arms,
                "dry_run": dry_run}
    _write_json(config.output / "sweep_manifest.json", manifest)
    if dry_run:
        report = {"status": "dry_run", "stage": config.stage, "mode": config.mode, "arms": arms}
    else:
        results = []
        for arm in arms:
            # A run must not silently compare a producer rewriting the pool between arms.
            if _pool_identity(config.train_files) != pool:
                raise ValueError("Frozen training pool changed during the sweep")
            results.append(run_arm(config, arm))
            _write_json(config.output / "sweep_report.json", build_report(results, config.mode))
        report = build_report(results, config.mode)
    _write_json(config.output / "sweep_report.json", report)
    return report


def _csv(value: str, convert=str) -> tuple:
    return tuple(convert(part.strip()) for part in value.split(",") if part.strip())


def _boolean(value: str) -> bool:
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("Boolean matrix values must be true or false")
    return value.lower() == "true"


def main(argv: list[str] | None = None) -> int:
    """Parse a CPU-only dry run or a sequential matrix of native SFT subprocesses."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "model", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--train-files", type=Path, nargs="+", required=True)
    parser.add_argument("--val-files", type=Path, nargs="+", default=[])
    parser.add_argument("--gpu-mask", required=True, help="Explicit physical IDs, e.g. 2,3,4,5,6,7")
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--global-batches", default="24")
    parser.add_argument("--micro-batches", default="1")
    parser.add_argument("--attn-backends", default="sdpa")
    parser.add_argument("--dynamic", default="false")
    parser.add_argument("--length-bucketing", default="false")
    parser.add_argument("--output-head-backends", default="native")
    parser.add_argument("--sequence-parallel-sizes", default="1")
    parser.add_argument("--remove-padding", default="false")
    parser.add_argument("--mode", choices=("screen", "matched"), default="screen")
    parser.add_argument("--stage", default="sweep")
    for name, default in (("warmup-steps", 2), ("measure-steps", 4), ("warmup-passes", 1),
                          ("measure-passes", 1), ("max-length", 8192), ("max-token-len-per-gpu", 8192),
                          ("seed", 20261028)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--numa-binding", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    values = vars(args).copy()
    dry_run = values.pop("dry_run")
    num_gpus = values.pop("num_gpus")
    try:
        for key in ("gpu_mask", "global_batches", "micro_batches", "sequence_parallel_sizes"):
            values[key] = _csv(values[key], int)
        for key in ("attn_backends", "output_head_backends"):
            values[key] = _csv(values[key])
        for key in ("dynamic", "length_bucketing", "remove_padding"):
            values[key] = _csv(values[key], _boolean)
        for key in ("train_files", "val_files"):
            values[key] = tuple(values[key])
        if num_gpus is not None and num_gpus != len(values["gpu_mask"]):
            raise ValueError("num_gpus must equal the explicit gpu_mask length")
        report = run_sweep(SweepConfig(**values), dry_run=dry_run)
    except (ValueError, OSError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}))
    return int(report["status"] not in ("completed", "dry_run"))


if __name__ == "__main__":
    raise SystemExit(main())
