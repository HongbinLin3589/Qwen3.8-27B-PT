"""Append production SFT evidence without changing optimizer or checkpoint behavior."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
from time import time, time_ns
import traceback
from typing import Any

import torch


def _source_fingerprint(row: dict) -> str:
    cached = row.get("sample_fingerprint")
    if cached is not None:
        return cached

    def encode(value):
        if isinstance(value, bytes):
            return {"bytes_hex": value.hex()}
        raise TypeError(f"Unsupported source fingerprint value: {type(value).__name__}")

    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=encode)
    return hashlib.sha256(payload.encode()).hexdigest()


def _cpu_memory() -> dict[str, int]:
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return {
        "process_rss_bytes": resident_pages * os.sysconf("SC_PAGE_SIZE"),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


class SFTProgressRecorder:
    """Flush each rank's source rows, real token counts and optimizer-step outcomes.

    Construct after native checkpoint loading. ``before_batch`` removes the
    loader's CPU-only provenance before the engine sees it. Each launch needs a
    new ``QWENPT_SFT_ATTEMPT_ID``; unset IDs are generated per rank automatically.
    """

    def __init__(self, trainer: Any, *, rank: int | None = None, attempt_id: str | None = None) -> None:
        self.rank = int(os.environ.get("RANK", "0")) if rank is None else rank
        self.attempt_id = attempt_id or os.environ.get("QWENPT_SFT_ATTEMPT_ID") or f"{time_ns()}-{os.getpid()}"
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.attempt_id):
            raise ValueError("SFT attempt IDs may contain only letters, digits, '.', '_' and '-'")
        self.rows = trainer.train_dataset.rows
        self.resume_global_step = int(trainer.resume_global_step)
        self.completed_steps = 0
        self._seen: set[int] = set()
        self._pending: dict | None = None
        directory = Path(trainer.config.trainer.default_local_dir) / "progress"
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{self.attempt_id}.rank{self.rank}.jsonl"
        self._handle = self.path.open("x", buffering=1)
        configured_resume = trainer.config.trainer.get("resume_from_path")
        self._write({
            "event": "attempt_started", "resume_global_step": self.resume_global_step,
            "resume_source": configured_resume or (
                str(directory.parent / f"global_step_{self.resume_global_step}") if self.resume_global_step else None
            ),
            "source_rows": len(self.rows), "steps_per_epoch": trainer.steps_per_epoch,
            "total_training_steps": trainer.total_training_steps,
            "sampler_configuration": getattr(trainer.train_sampler, "_configuration", None),
            "data_parallel_rank": trainer.engine.get_data_parallel_rank(),
            "data_parallel_size": trainer.engine.get_data_parallel_size(),
            "sequence_parallel_size": getattr(trainer.engine, "ulysses_sequence_parallel_size", 1),
            "train_files": list(trainer.config.data.train_files)
            if not isinstance(trainer.config.data.train_files, str) else [trainer.config.data.train_files],
            **_cpu_memory(),
        })

    def _write(self, record: dict) -> None:
        self._handle.write(json.dumps({
            "attempt_id": self.attempt_id, "rank": self.rank, "unix_time": time(), **record,
        }, allow_nan=False) + "\n")
        self._handle.flush()

    def before_batch(self, data) -> None:
        """Validate and record source indices, excluding zero-loss internal padding."""
        from qwenpt.train.common.template import tokenized_fingerprint

        if self._pending is not None:
            raise RuntimeError("The previous production batch has no recorded outcome")
        indices = data.pop("sft_source_index")
        indices = indices.values() if indices.is_nested else indices.reshape(-1)
        indices = indices.detach().cpu().tolist()
        ids = data["input_ids"]
        masks = data["loss_mask"]
        ids = ids.unbind() if ids.is_nested or ids.ndim > 1 else (ids,)
        masks = masks.unbind() if masks.is_nested or masks.ndim > 1 else (masks,)
        if len(indices) != len(ids) or len(masks) != len(ids):
            raise ValueError("Source provenance must identify each rank-local sample")
        samples, padding_samples, padding_tokens = [], 0, 0
        for index, tokens, mask in zip(indices, ids, masks, strict=True):
            input_tokens, supervised_tokens = tokens.numel(), int(mask.sum().item())
            if index == -1:
                if supervised_tokens or input_tokens != 16:
                    raise ValueError("Internal SFT tail padding must contain 16 zero-loss tokens")
                padding_samples += 1
                padding_tokens += input_tokens
                continue
            if not 0 <= index < len(self.rows) or supervised_tokens <= 0:
                raise ValueError("Real SFT provenance needs a valid source row and nonempty supervision")
            if index in self._seen or any(item["source_index"] == index for item in samples):
                raise ValueError(f"SFT source row {index} was repeated within this attempt")
            fingerprint = tokenized_fingerprint(tokens, mask)
            row = self.rows[index]
            if row.get("tokenized_fingerprint") not in (None, fingerprint):
                raise ValueError(f"SFT source row {index} disagrees with its audited token fingerprint")
            samples.append({
                "source_index": index, "source_fingerprint": _source_fingerprint(row),
                "tokenized_fingerprint": fingerprint,
                "input_tokens": input_tokens, "supervised_tokens": supervised_tokens,
            })
        self._pending = {
            "global_step": self.resume_global_step + self.completed_steps + 1,
            "samples": samples, "real_samples": len(samples),
            "real_input_tokens": sum(item["input_tokens"] for item in samples),
            "real_supervised_tokens": sum(item["supervised_tokens"] for item in samples),
            "padding_samples": padding_samples, "padding_input_tokens": padding_tokens,
        }
        self._write({"event": "batch_started", **self._pending})
        if torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats()

    def after_batch(self, output, elapsed_seconds: float) -> None:
        """Flush the successful optimizer step and cheap CPU/CUDA memory statistics."""
        from verl.utils import tensordict_utils as tu

        if self._pending is None:
            raise RuntimeError("Production batch has no source evidence")
        metrics = tu.get(output, "metrics", default={}) if output is not None else {}
        values, invalid = {}, []
        for key in ("loss", "grad_norm", "lr"):
            value = metrics.get(key, metrics.get(f"train/{key}"))
            value = float(value) if value is not None else None
            if value is not None and not math.isfinite(value):
                invalid.append(key)
                value = None
            values[key] = value
        if invalid:
            raise FloatingPointError(f"Non-finite SFT step metrics: {invalid}")
        memory = _cpu_memory()
        if torch.cuda.is_initialized():
            memory.update(
                memory_allocated_bytes=torch.cuda.memory_allocated(),
                memory_reserved_bytes=torch.cuda.memory_reserved(),
                peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        self._write({"event": "step_completed", **self._pending, "elapsed_seconds": elapsed_seconds,
                     **values, **memory})
        self._seen.update(item["source_index"] for item in self._pending["samples"])
        self.completed_steps += 1
        self._pending = None

    def fail(self, error: BaseException) -> None:
        """Flush failure evidence without CUDA reads that could hang after a rank error."""
        self._write({
            "event": "step_failed" if self._pending is not None else "attempt_failed",
            **(self._pending or {}), "error_type": type(error).__name__, "error": str(error),
            "traceback": "".join(traceback.format_exception(error)), **_cpu_memory(),
        })
        self._pending = None

    def finish(self, status: str) -> None:
        """Close this attempt; native checkpoint completion determines durable progress."""
        if self._handle.closed:
            return
        self._write({"event": "attempt_finished", "status": status, "completed_steps": self.completed_steps,
                     "last_completed_global_step": self.resume_global_step + self.completed_steps})
        self._handle.close()


def audit_sft_progress(
    run_dir: str | Path, manifest: dict, *, world_size: int, expected_rows: int | None = None,
) -> dict:
    """Verify the effective resumed epoch against a complete train-only token audit.

    New attempts discard older evidence beyond their restored checkpoint step.
    Only the resulting completed branch contributes source rows and token totals.
    Native checkpoint/archive verification remains a separate required check.
    """
    counts = manifest["counts"]
    rows = counts["validated_rows"]
    if (manifest["status"] != "passed" or manifest["max_samples"] != -1 or counts["failures"]
            or rows <= 0 or rows != counts["eligible_train_rows"] or rows != counts["selected_rows"]
            or (expected_rows is not None and rows != expected_rows)):
        raise ValueError("Progress audit requires a passed, complete, unfiltered train manifest")
    bins = list(manifest["actual_length_bins"].values())
    expected_input = sum(bucket["tokens"] for bucket in bins)
    expected_supervised = sum(bucket["supervised_tokens"] for bucket in bins)
    if sum(bucket["rows"] for bucket in bins) != rows:
        raise ValueError("Manifest length bins disagree with the source row count")
    attempts: dict[int, list[list[dict]]] = {}
    truncated = []
    for path in sorted((Path(run_dir) / "progress").glob("*.jsonl")):
        records = []
        lines = path.read_text().splitlines(keepends=True)
        for line_number, line in enumerate(lines):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if line_number == len(lines) - 1 and not line.endswith("\n"):
                    truncated.append(str(path))
                    break
                raise ValueError(f"Corrupted progress evidence: {path}:{line_number + 1}") from None
        if not records or records[0].get("event") != "attempt_started":
            raise ValueError(f"Progress file has no attempt header: {path}")
        header = records[0]
        if any(record["rank"] != header["rank"] or record["attempt_id"] != header["attempt_id"] for record in records):
            raise ValueError(f"Mixed rank/attempt evidence: {path}")
        attempts.setdefault(header["rank"], []).append(records)
    if set(attempts) != set(range(world_size)):
        raise ValueError("Progress evidence is missing a rank or contains an unexpected rank")
    effective, final_steps = {}, set()
    for rank, histories in attempts.items():
        branch = {}
        prior_config = None
        for records in sorted(histories, key=lambda history: history[0]["unix_time"]):
            header = records[0]
            if (header["source_rows"] != rows or header["data_parallel_size"] != world_size
                    or header["data_parallel_rank"] != rank or header["sequence_parallel_size"] != 1):
                raise ValueError("Production progress requires matching source rows and SP1/DP world size")
            resume = header["resume_global_step"]
            if resume < 0 or set(range(1, resume + 1)) - set(branch):
                raise ValueError(f"Rank {rank} resumed without complete prior checkpoint-step evidence")
            if resume and prior_config != header["sampler_configuration"]:
                raise ValueError("A resumed attempt changed the sampler contract")
            branch = {step: record for step, record in branch.items() if step <= resume}
            next_step = resume + 1
            for record in records[1:]:
                if record["event"] != "step_completed":
                    continue
                if record["global_step"] != next_step:
                    raise ValueError(f"Rank {rank} has nonconsecutive optimizer-step evidence")
                branch[next_step] = record
                next_step += 1
            prior_config = header["sampler_configuration"]
        final = records[-1]
        target_steps = header["total_training_steps"]
        if (final.get("event") != "attempt_finished" or final.get("status") != "completed"
                or final.get("last_completed_global_step") != target_steps
                or set(branch) != set(range(1, target_steps + 1))):
            raise ValueError(f"Rank {rank} has no completed full-epoch terminal evidence")
        if target_steps != header["steps_per_epoch"]:
            raise ValueError("Progress audit requires exactly one complete epoch")
        final_steps.add(target_steps)
        effective[rank] = branch
    if len(final_steps) != 1:
        raise ValueError("Ranks disagree on the complete epoch step count")
    seen, input_tokens, supervised_tokens, padding_samples, padding_tokens = Counter(), 0, 0, 0, 0
    for branch in effective.values():
        for record in branch.values():
            for key in ("loss", "grad_norm", "lr", "elapsed_seconds"):
                value = record.get(key)
                if value is None or not math.isfinite(value) or (key == "elapsed_seconds" and value <= 0):
                    raise ValueError(f"Effective progress has missing or non-finite {key}")
            samples = record["samples"]
            batch_input = sum(sample["input_tokens"] for sample in samples)
            batch_supervised = sum(sample["supervised_tokens"] for sample in samples)
            if (record["real_samples"] != len(samples) or record["real_input_tokens"] != batch_input
                    or record["real_supervised_tokens"] != batch_supervised
                    or record["padding_samples"] < 0 or record["padding_input_tokens"] != record["padding_samples"] * 16):
                raise ValueError("Progress token aggregates disagree with their sample evidence")
            for sample in samples:
                index = sample["source_index"]
                if (not 0 <= index < rows or not 0 < sample["supervised_tokens"] <= sample["input_tokens"]
                        or not sample["source_fingerprint"] or not sample["tokenized_fingerprint"]):
                    raise ValueError("Invalid real-source progress evidence")
                seen[index] += 1
            input_tokens += batch_input
            supervised_tokens += batch_supervised
            padding_samples += record["padding_samples"]
            padding_tokens += record["padding_input_tokens"]
    if len(seen) != rows or any(count != 1 for count in seen.values()):
        raise ValueError("The effective epoch contains duplicated or missing real source rows")
    if input_tokens != expected_input or supervised_tokens != expected_supervised:
        raise ValueError("Effective real token totals disagree with the complete train manifest")
    return {
        "status": "passed", "real_samples": rows, "real_input_tokens": input_tokens,
        "real_supervised_tokens": supervised_tokens, "padding_samples": padding_samples,
        "padding_input_tokens": padding_tokens, "world_size": world_size,
        "completed_global_steps": final_steps.pop(), "truncated_old_attempt_files": truncated,
        "checkpoint_archives_verified": False,
    }


def main() -> None:
    """Audit the complete production epoch, writing a small success/failure report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--expected-rows", type=int, default=100000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = audit_sft_progress(
            args.run_dir, json.loads(args.manifest.read_text()),
            world_size=args.world_size, expected_rows=args.expected_rows,
        )
    except (ValueError, KeyError, TypeError) as error:
        report = {"status": "failed", "error": str(error)}
    with args.manifest.open("rb") as handle:
        report["manifest_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    output = args.output or args.run_dir / "training_progress_audit.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
