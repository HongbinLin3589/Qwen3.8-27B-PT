"""Record paired SFT benchmarks without changing native optimization or scheduling."""

import hashlib
import json
import math
import os
from pathlib import Path
from time import time

import torch


class BenchmarkFinished(RuntimeError):
    """Signal that all requested native optimizer steps have already completed."""


class BenchmarkRecorder:
    """Record rank-local batches, metrics and small parameter samples without collectives.

    Args:
        run_dir: Directory receiving the rank-specific benchmark artifacts.
        steps: Number of complete optimizer steps before controlled termination.
        rank: Global process rank used to keep artifact names independent.
    """

    def __init__(self, run_dir: str | Path, steps: int, rank: int) -> None:
        if steps <= 0:
            raise ValueError("Benchmark steps must be positive")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.steps = steps
        self.rank = rank
        self.records: list[dict] = []
        self._pending: dict | None = None
        self._head_capture_handle = None
        self._head_capture_sample = None
        self.steps_path = self.run_dir / f"benchmark_steps_rank{rank}.jsonl"
        # Mixing a previous attempt into an arm would invalidate paired comparisons.
        self.steps_path.open("x").close()

    def before_batch(self, *args, **kwargs) -> None:
        """Fingerprint the native ``data`` argument or stop before step N + 1."""
        if len(self.records) >= self.steps:
            raise BenchmarkFinished(f"Completed {self.steps} benchmark optimizer steps")
        if self._pending is not None:
            raise RuntimeError("Previous benchmark batch has no recorded output")
        data = kwargs["data"] if "data" in kwargs else args[0]
        from verl.utils import tensordict_utils as tu

        digest = hashlib.sha256()
        field_hashes = {}
        for key in ("input_ids", "labels", "loss_mask", "position_ids", "attention_mask"):
            if key not in data.keys():
                continue
            field_digest = hashlib.sha256()
            self._hash_tensor(field_digest, data[key])
            field_hashes[key] = field_digest.hexdigest()
            digest.update(key.encode())
            digest.update(field_digest.digest())
        input_ids = data["input_ids"]
        samples = input_ids.unbind() if input_ids.is_nested or input_ids.ndim > 1 else (input_ids,)
        if self._head_capture_handle is not None:
            if len(samples) != 1:
                raise ValueError("Head-input capture requires one rank-local sample and microbatch=1")
            loss_mask = data["loss_mask"]
            loss_mask = loss_mask.unbind()[0] if loss_mask.is_nested or loss_mask.ndim > 1 else loss_mask
            self._head_capture_sample = {
                "input_ids": samples[0].detach().cpu().clone(),
                "loss_mask": loss_mask.detach().cpu().clone(),
                "offsets": torch.tensor([0, samples[0].numel()]),
            }
        sample_hashes = []
        for index in range(len(samples)):
            sample_digest = hashlib.sha256()
            for key in ("input_ids", "loss_mask", "position_ids"):
                if key not in data.keys():
                    continue
                tensor = data[key]
                values = tensor.unbind()[index] if tensor.is_nested or tensor.ndim > 1 else tensor
                sample_digest.update(key.encode())
                self._hash_tensor(sample_digest, values)
            sample_hashes.append(sample_digest.hexdigest())
        global_lengths = tu.get(data, "global_token_num", default=None)
        if isinstance(global_lengths, torch.Tensor):
            global_lengths = global_lengths.detach().cpu().tolist()
        self._pending = {
            "input_sha256": digest.hexdigest(),
            "field_sha256": field_hashes,
            "sample_sha256": sample_hashes,
            "local_sequence_lengths": [sample.numel() for sample in samples],
            "local_supervised_tokens": int(data["loss_mask"].sum().item()) if "loss_mask" in data.keys() else None,
            "global_sequence_lengths": global_lengths,
            "global_tokens": sum(global_lengths) if global_lengths is not None else None,
        }
        if torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats()

    def after_batch(self, output, elapsed: float, trainer) -> None:
        """Write one successful native step after its synchronized timer has stopped."""
        from verl.utils import tensordict_utils as tu

        if self._pending is None:
            raise RuntimeError("Benchmark batch must be fingerprinted before recording output")
        metrics = tu.get(output, "metrics", default={}) if output is not None else {}
        finished_unix = time()
        record = {
            "step": len(self.records) + 1, "elapsed_seconds": float(elapsed),
            "finished_unix": finished_unix, "started_unix": finished_unix - float(elapsed),
            **self._pending,
        }
        engine = getattr(trainer, "engine", None)
        dp_rank = getattr(engine, "get_data_parallel_rank", None)
        dp_size = getattr(engine, "get_data_parallel_size", None)
        record.update(
            data_parallel_rank=int(dp_rank()) if callable(dp_rank) else self.rank,
            data_parallel_size=int(dp_size()) if callable(dp_size) else None,
            sequence_parallel_size=int(getattr(engine, "ulysses_sequence_parallel_size", 1)),
        )
        if torch.cuda.is_initialized():
            record.update(
                peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        invalid = []
        for key in ("loss", "grad_norm", "lr", "mfu"):
            value = metrics.get(key, metrics.get(f"train/{key}"))
            value = float(value) if value is not None else None
            if value is not None and not math.isfinite(value):
                invalid.append(key)
                value = None
            record[key] = value
        scheduler = getattr(trainer.engine, "lr_scheduler", None)
        record["scheduler_last_epoch"] = getattr(scheduler, "last_epoch", None)
        record["non_finite_metrics"] = invalid
        with self.steps_path.open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        self.records.append(record)
        self._pending = None
        if invalid:
            raise FloatingPointError(f"Non-finite benchmark metrics: {invalid}")

    def finish(self, trainer, status: str) -> dict:
        """Save local samples and process placement, retaining failures in the manifest."""
        runtime_diagnostics = None
        if os.environ.get("QWENPT_SFT_RUNTIME_DIAGNOSTICS") == "1":
            try:
                from legacy.sft.utils.sft_runtime import collect_runtime_diagnostics

                runtime_diagnostics = collect_runtime_diagnostics()
            except Exception as error:
                runtime_diagnostics = {"collection_error": f"{type(error).__name__}: {error}"}
        samples_path = self.run_dir / f"benchmark_parameter_samples_rank{self.rank}.json"
        sampling_error = None
        parameter_samples = None
        if trainer is not None and status == "failed":
            # A failed rank may leave peers inside NCCL; copying CUDA samples can
            # then block before Python reports the original training exception.
            sampling_error = "Skipped parameter sampling after native training failure"
        elif trainer is not None:
            try:
                parameter_samples = self._parameter_samples(trainer.engine.module)
                samples_path.write_text(json.dumps(parameter_samples, indent=2, allow_nan=False) + "\n")
            except Exception as error:
                # A diagnostic failure must not replace the original training exception.
                sampling_error = f"{type(error).__name__}: {error}"
        engine = getattr(trainer, "engine", None)
        scheduler = getattr(engine, "lr_scheduler", None)
        manifest = {
            "schema_version": 1,
            "status": status,
            "rank": self.rank,
            "requested_steps": self.steps,
            "completed_steps": len(self.records),
            "native_total_training_steps": getattr(trainer, "total_training_steps", None),
            "native_steps_per_epoch": getattr(trainer, "steps_per_epoch", None),
            "scheduler_last_epoch": getattr(scheduler, "last_epoch", None),
            "all_recorded_metrics_finite": all(not row["non_finite_metrics"] for row in self.records),
            "steps_file": self.steps_path.name,
            "parameter_samples_file": samples_path.name if parameter_samples is not None and not sampling_error else None,
            "parameter_sampling_error": sampling_error,
            "parameter_sampling_note": "Seven fixed named local shards; first, middle and last elements only",
            "process_placement": self._process_placement(),
            "runtime_diagnostics": runtime_diagnostics,
            "head_input_capture": "head_input_rank0.pt" if (self.run_dir / "head_input_rank0.pt").is_file() else None,
        }
        if self._head_capture_handle is not None:
            self._head_capture_handle.remove()
            self._head_capture_handle = None
        manifest_path = self.run_dir / f"benchmark_manifest_rank{self.rank}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        return manifest

    def capture_head_input(self, trainer) -> None:
        """Save rank zero's first real decoder output with its unshifted targets.

        Capture is limited to an unfused, one-sample microbatch; its disk copy is
        part of the first warmup step and is excluded from steady-state timing.
        """
        if self.rank != 0:
            return
        heads = [module for name, module in trainer.engine.module.named_modules()
                 if name == "lm_head" or name.endswith(".lm_head")]
        if len(heads) != 1:
            raise ValueError("Expected exactly one unfused lm_head for activation capture")

        def capture(module, args):
            if self._head_capture_sample is None:
                raise RuntimeError("Head-input capture ran before its source batch was recorded")
            hidden = args[0].detach().reshape(-1, args[0].shape[-1]).cpu()
            if len(hidden) != self._head_capture_sample["input_ids"].numel():
                raise ValueError("Captured decoder positions do not match the source sample")
            torch.save({**self._head_capture_sample, "hidden": hidden}, self.run_dir / "head_input_rank0.pt")
            self._head_capture_handle.remove()
            self._head_capture_handle = None
            self._head_capture_sample = None

        self._head_capture_handle = heads[0].register_forward_pre_hook(capture)

    @staticmethod
    def _parameter_samples(module) -> dict:
        parameters = list(module.named_parameters())
        chosen = sorted({round(i * (len(parameters) - 1) / 6) for i in range(7)}) if parameters else []
        parameter_samples = {}
        for index in chosen:
            name, parameter = parameters[index]
            local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
            local = local.detach()
            count = local.numel()
            positions = sorted({
                max(0, min(count - 1, center + offset))
                for center in (0, count // 2, count - 1)
                for offset in (-1, 0, 1)
            }) if count else []
            # Index individual elements so even a strided shard never gets fully cloned.
            selected = []
            for position in positions:
                remaining = position
                coordinates = []
                for size in reversed(local.shape):
                    remaining, coordinate = divmod(remaining, size)
                    coordinates.append(coordinate)
                selected.append(local[tuple(reversed(coordinates))])
            values = torch.stack(selected).float().cpu().tolist() if selected else []
            parameter_samples[name] = {
                "global_shape": list(parameter.shape),
                "local_shape": list(local.shape),
                "dtype": str(local.dtype),
                "flat_positions": positions,
                "values": values,
            }
        return parameter_samples

    @staticmethod
    def _process_placement() -> dict:
        placement = {"pid": os.getpid(), "cpu_affinity": sorted(os.sched_getaffinity(0))}
        try:
            status = {}
            for line in Path("/proc/self/status").read_text().splitlines():
                key, _, value = line.partition(":")
                status[key] = value.strip()
            placement.update({
                "rss_bytes": int(status["VmRSS"].split()[0]) * 1024,
                "peak_rss_bytes": int(status["VmHWM"].split()[0]) * 1024,
                "threads": int(status["Threads"]),
                "cpus_allowed_list": status["Cpus_allowed_list"],
                "mems_allowed_list": status["Mems_allowed_list"],
            })
            pages = {}
            for line in Path("/proc/self/numa_maps").read_text().splitlines():
                for field in line.split():
                    key, _, value = field.partition("=")
                    if key.startswith("N") and key[1:].isdigit() and value.isdigit():
                        pages[key] = pages.get(key, 0) + int(value)
            placement["numa_resident_pages"] = pages
        except OSError as error:
            placement["read_error"] = f"{type(error).__name__}: {error}"
        return placement

    @classmethod
    def _hash_tensor(cls, digest, tensor: torch.Tensor) -> None:
        if tensor.is_nested:
            samples = tensor.unbind()
            digest.update(f"nested:{len(samples)}".encode())
            for sample in samples:
                cls._hash_tensor(digest, sample)
            return
        values = tensor.detach().cpu().contiguous()
        digest.update(json.dumps({"shape": list(values.shape), "dtype": str(values.dtype)}).encode())
        digest.update(values.reshape(-1).view(torch.uint8).numpy().tobytes())
