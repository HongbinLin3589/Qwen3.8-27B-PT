"""Read back a fresh BOLT smoke run's metrics and native FSDP2 LoRA checkpoint."""

import argparse
from collections import Counter
from collections.abc import Iterator, Mapping
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import zipfile

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def verify_metrics(path: Path, expected_step: int) -> dict:
    """Require every fresh training step and finite training/final validation metrics."""
    training, validation = {}, {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        step, values = record["step"], record["data"]
        if type(step) is not int or not 1 <= step <= expected_step:
            raise ValueError(f"Unexpected metrics step: {step!r}")
        for name in ("train/loss", "train/grad_norm", "train/lr", "val/loss"):
            if name in values:
                value = values[name]
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid {name} at step {step}: {value!r}")
        if "train/loss" in values:
            if step in training or not all(name in values for name in ("train/grad_norm", "train/lr")):
                raise ValueError(f"Duplicate or incomplete training metrics at step {step}")
            training[step] = {name: values[name] for name in ("train/loss", "train/grad_norm", "train/lr")}
        if "val/loss" in values:
            if step in validation:
                raise ValueError(f"Duplicate validation metrics at step {step}")
            validation[step] = values["val/loss"]
    if sorted(training) != list(range(1, expected_step + 1)) or expected_step not in validation:
        raise ValueError("Missing training steps or final validation loss")
    if not any(values["train/grad_norm"] > 0 and values["train/lr"] > 0 for values in training.values()):
        raise ValueError("No training step has both a positive gradient norm and learning rate")
    return {
        "training": training, "validation": validation,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def verify_schedule(root: Path, metrics: dict, expected_step: int) -> dict:
    """Compare each post-update logged LR with verl's configured cosine schedule."""
    from omegaconf import OmegaConf

    config_path = root / "hydra/.hydra/config.yaml"
    config = OmegaConf.load(config_path)
    optimizer = config.optim
    if optimizer.lr_scheduler_type != "cosine":
        raise ValueError("Schedule verification requires optim.lr_scheduler_type=cosine")
    total_steps = config.trainer.total_training_steps
    if total_steps is not None and total_steps != expected_step:
        raise ValueError("Configured training steps differ from the expected final step")
    warmup_steps = optimizer.lr_warmup_steps
    if warmup_steps <= 0:
        warmup_steps = int(optimizer.lr_warmup_steps_ratio * expected_step)
    base_lr = float(optimizer.lr)
    min_ratio = float(optimizer.min_lr_ratio or 0.0)
    cycles = float(optimizer.num_cycles)
    zero_indexed = optimizer.zero_indexed_step

    def learning_rate(step):
        if not zero_indexed:
            step += 1
        if step < warmup_steps:
            return base_lr * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, expected_step - warmup_steps)
        value = math.cos(math.pi * cycles * 2 * progress) * (1 - min_ratio) * 0.5 + (1 + min_ratio) * 0.5
        return base_lr * max(min_ratio, value)

    expected_lrs = {}
    for step in range(1, expected_step + 1):
        expected = learning_rate(step)
        logged = metrics["training"][step]["train/lr"]
        if not math.isclose(logged, expected, rel_tol=1e-7, abs_tol=max(1e-15, abs(base_lr) * 1e-9)):
            raise ValueError(f"Incorrect logged LR at step {step}: {logged}, expected {expected}")
        expected_lrs[step] = expected
    return {
        "scheduler": "cosine", "base_lr": base_lr, "warmup_steps": warmup_steps,
        "total_steps": expected_step, "min_lr_ratio": min_ratio, "num_cycles": cycles,
        "zero_indexed_step": zero_indexed, "first_update_lr": learning_rate(0),
        "expected_logged_lrs": expected_lrs,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }


def verify_checkpoints(root: Path, expected_steps: list[int], expected_step: int, world_size: int) -> dict:
    """Require the exact retained checkpoints and CRC-readable native shard artifacts."""
    if (not expected_steps or len(set(expected_steps)) != len(expected_steps)
            or min(expected_steps) <= 0 or max(expected_steps) != expected_step):
        raise ValueError("Expected checkpoint steps must be unique, positive and end at the expected final step")
    expected_names = {f"global_step_{step}" for step in expected_steps}
    actual_names = {path.name for path in root.glob("global_step_*")}
    if actual_names != expected_names:
        raise ValueError(f"Retained checkpoints differ: missing={sorted(expected_names - actual_names)}, "
                         f"extra={sorted(actual_names - expected_names)}")
    tracker = int((root / "latest_checkpointed_iteration.txt").read_text().strip())
    if tracker != expected_step:
        raise ValueError(f"Checkpoint tracker is {tracker}, expected {expected_step}")
    files = [f"{kind}_world_size_{world_size}_rank_{rank}.pt"
             for kind in ("model", "optim", "extra_state") for rank in range(world_size)]
    files += [f"data_{rank}.pt" for rank in range(world_size)]
    retained = {}
    for step in sorted(expected_steps):
        checkpoint = root / f"global_step_{step}"
        config = json.loads((checkpoint / "fsdp_config.json").read_text())
        if config != {"FSDP_version": 2, "world_size": world_size}:
            raise ValueError(f"Incorrect FSDP configuration at step {step}")
        metadata = json.loads((checkpoint / "lora_train_meta.json").read_text())
        if metadata.get("r", 0) <= 0 or metadata.get("lora_alpha", 0) <= 0:
            raise ValueError(f"Missing LoRA metadata at step {step}")
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            if not json.loads((checkpoint / "huggingface" / name).read_text()):
                raise ValueError(f"Empty Hugging Face artifact at step {step}: {name}")
        sizes = {}
        for name in files:
            path = checkpoint / name
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Missing or empty checkpoint artifact: {path}")
            with zipfile.ZipFile(path) as archive:
                entries = archive.namelist()
                if not all(any(item.endswith("/" + leaf) for item in entries)
                           for leaf in ("data.pkl", "version", "byteorder")):
                    raise ValueError(f"Invalid Torch archive metadata: {path}")
                if name.startswith(("model_", "optim_")) and not any("/data/" in item for item in entries):
                    raise ValueError(f"Missing tensor storage in checkpoint: {path}")
                if archive.testzip() is not None:
                    raise ValueError(f"Checkpoint CRC check failed: {path}")
            sizes[name] = path.stat().st_size
        retained[step] = {"file_bytes": sizes, "zip_crc": "passed"}
    return {"steps": sorted(expected_steps), "tracker": tracker, "checkpoints": retained}


def local_tensors(value) -> Iterator[torch.Tensor]:
    """Yield checkpoint tensors' local CPU storage without gathering DTensors."""
    if isinstance(value, DTensor):
        yield value.to_local().detach().cpu()
    elif hasattr(value, "local_shards"):
        for shard in value.local_shards():
            yield shard.tensor.detach().cpu()
    elif isinstance(value, torch.Tensor):
        yield value.detach().cpu()
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from local_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from local_tensors(item)


def tensor_statistics(value) -> dict:
    """Reject nonfinite local tensors and count their stored/nonzero elements."""
    count, elements, nonzero = 0, 0, 0
    for tensor in local_tensors(value):
        if not torch.isfinite(tensor).all().item():
            raise ValueError("Checkpoint contains a nonfinite tensor")
        count += 1
        elements += tensor.numel()
        nonzero += torch.count_nonzero(tensor).item()
    return {"tensors": count, "elements": elements, "nonzero_elements": nonzero}


def verify_rank(root: Path, expected_step: int, world_size: int, rank: int) -> dict:
    """Read this rank's native model, optimizer, RNG/scheduler and loader states."""
    checkpoint = root / f"global_step_{expected_step}"
    fsdp = json.loads((checkpoint / "fsdp_config.json").read_text())
    if fsdp != {"FSDP_version": 2, "world_size": world_size}:
        raise ValueError(f"Unexpected FSDP configuration: {fsdp}")
    metadata = json.loads((checkpoint / "lora_train_meta.json").read_text())
    if metadata.get("r", 0) <= 0 or metadata.get("lora_alpha", 0) <= 0:
        raise ValueError("Missing positive LoRA rank/alpha metadata")
    tracker = int((root / "latest_checkpointed_iteration.txt").read_text().strip())
    if tracker != expected_step:
        raise ValueError(f"Checkpoint tracker is {tracker}, expected {expected_step}")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not json.loads((checkpoint / "huggingface" / name).read_text()):
            raise ValueError(f"Empty Hugging Face artifact: {name}")

    files = {}

    def load(name):
        path = checkpoint / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing or empty checkpoint file: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files[name] = {"bytes": path.stat().st_size, "sha256": digest}
        # Native FSDP checkpoints serialize DTensor/DeviceMesh and Python RNG state.
        return torch.load(path, map_location="cpu", weights_only=False)

    suffix = f"world_size_{world_size}_rank_{rank}.pt"
    model = load(f"model_{suffix}")
    if not isinstance(model, dict) or not model or any("lora_" not in name and ".adapter_" not in name for name in model):
        raise ValueError("Model shard is empty or contains non-LoRA parameters")
    model_stats = tensor_statistics(model)
    b_stats = tensor_statistics({name: value for name, value in model.items() if "lora_B" in name})
    model_keys = sorted(model)
    del model

    optimizer = load(f"optim_{suffix}")
    states = optimizer.get("state", {})
    if not states or not optimizer.get("param_groups"):
        raise ValueError("Optimizer has no parameter state/groups")
    steps = []
    for state in states.values():
        step_tensors = list(local_tensors(state.get("step")))
        step = step_tensors[0].item() if len(step_tensors) == 1 else state.get("step")
        if not isinstance(step, Real) or not math.isfinite(step) or int(step) != step or not 1 <= step <= expected_step:
            raise ValueError(f"Invalid optimizer update step: {step!r}")
        if "exp_avg" not in state or "exp_avg_sq" not in state:
            raise ValueError("AdamW parameter state lacks its momentum tensors")
        steps.append(int(step))
    if max(steps) != expected_step:
        raise ValueError("Optimizer did not reach the expected update step")
    optimizer_stats = tensor_statistics(optimizer)
    momentum_stats = tensor_statistics([state["exp_avg_sq"] for state in states.values()])
    del optimizer

    extra = load(f"extra_state_{suffix}")
    scheduler = extra.get("lr_scheduler")
    if not isinstance(scheduler, dict) or not extra.get("rng") or scheduler.get("last_epoch") != expected_step:
        raise ValueError("Missing RNG or incorrect learning-rate scheduler progress")
    extra_stats = tensor_statistics(extra)
    loader = load(f"data_{rank}.pt")
    if not isinstance(loader, dict) or loader.get("_num_yielded") != expected_step:
        raise ValueError("Missing or incorrect stateful dataloader progress for this fresh smoke run")
    return {
        "rank": rank, "files": files, "lora_metadata": metadata,
        "model_keys": model_keys, "model": model_stats, "lora_B": b_stats,
        "optimizer": optimizer_stats, "optimizer_steps": dict(Counter(steps)),
        "optimizer_parameters_with_state": len(steps), "second_moment": momentum_stats,
        "extra": extra_stats, "scheduler_last_epoch": scheduler["last_epoch"],
        "loader_batches_yielded": loader["_num_yielded"],
    }


def main() -> None:
    """Verify a completed fresh smoke checkpoint in a matching torchrun world."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--check-schedule", action="store_true")
    parser.add_argument("--expected-checkpoints", type=int, nargs="+")
    args = parser.parse_args()
    if args.expected_step <= 0 or args.world_size <= 0 or int(os.environ.get("WORLD_SIZE", "0")) != args.world_size:
        parser.error("Use torchrun with the requested positive world size and expected step")
    torch.set_num_threads(1)
    if args.backend == "nccl":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(args.backend, timeout=timedelta(minutes=5))
    try:
        rank = dist.get_rank()
        root = Path(args.output_dir).resolve()
        result = {"rank": rank}
        try:
            result["checkpoint"] = verify_rank(root, args.expected_step, args.world_size, rank)
            if rank == 0:
                result["metrics"] = verify_metrics(root / "metrics.jsonl", args.expected_step)
                if args.check_schedule:
                    result["schedule"] = verify_schedule(root, result["metrics"], args.expected_step)
                if args.expected_checkpoints is not None:
                    result["retained_checkpoints"] = verify_checkpoints(
                        root, args.expected_checkpoints, args.expected_step, args.world_size
                    )
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        results = [None] * args.world_size
        dist.all_gather_object(results, result)
        errors = [item["error"] for item in results if "error" in item]
        if not errors:
            first = results[0]["checkpoint"]
            if any(item["checkpoint"]["model_keys"] != first["model_keys"]
                   or item["checkpoint"]["lora_metadata"] != first["lora_metadata"] for item in results):
                errors.append("Model parameter keys or LoRA metadata differ between ranks")
            for field in ("model", "lora_B", "second_moment"):
                if sum(item["checkpoint"][field]["nonzero_elements"] for item in results) <= 0:
                    errors.append(f"All ranks have zero {field} tensors")
        if rank == 0:
            report = {
                "status": "failed" if errors else "passed", "errors": errors,
                "checked_at": datetime.now(timezone.utc).isoformat(), "output_dir": str(root),
                "expected_step": args.expected_step, "world_size": args.world_size, "backend": args.backend,
                "metrics": results[0].get("metrics"),
                "ranks": [item.get("checkpoint", item) for item in results],
                "scope": "Metric and checkpoint readback; does not restore a model or measure task accuracy.",
            }
            for key in ("schedule", "retained_checkpoints"):
                if key in results[0]:
                    report[key] = results[0][key]
            destination = root / "training_smoke_check.json"
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            temporary.replace(destination)
            print(json.dumps({"status": report["status"], "report": str(destination), "errors": errors}), flush=True)
        if errors:
            raise RuntimeError("BOLT smoke verification failed: " + "; ".join(errors))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
