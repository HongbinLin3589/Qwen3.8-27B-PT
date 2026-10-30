"""Verify a completed SFT run from metrics and CPU-loaded checkpoint contents."""

import argparse
import json
import math
from pathlib import Path

import torch


def _local(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if hasattr(value, "to_local") else value


def main() -> None:
    """Check the requested run and always write its verification.json report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=1125)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--train-rows", type=int, default=9000)
    parser.add_argument("--validation-rows", type=int, default=1000)
    args = parser.parse_args()
    run = args.run.resolve()
    report = {"run": str(run), "passed": False, "errors": [], "ranks": [], "counts": {
        "expected_steps": args.expected_steps, "world_size": args.world_size,
        "train_rows_expected": args.train_rows, "validation_rows_expected": args.validation_rows,
    }, "validation_coverage_verified": False, "slurm_status_verified": False}

    def require(condition, message):
        if not condition:
            report["errors"].append(message)

    def load(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    try:
        torch.set_num_threads(1)
        require(args.train_rows % args.world_size == 0, "Training rows are not divisible by world size")
        records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
        training = [record for record in records if "train/loss" in record.get("data", {})]
        steps = [record["step"] for record in training]
        require(steps == list(range(1, args.expected_steps + 1)), "Training steps are missing, duplicated, or out of order")
        report["counts"].update({"metric_records": len(records), "train_steps": len(training)})
        losses = []
        for record in training:
            for key in ("train/loss", "train/grad_norm"):
                value = record["data"].get(key)
                require(isinstance(value, (int, float)) and math.isfinite(value), f"Step {record['step']}: invalid {key}")
            loss = record["data"]["train/loss"]
            if isinstance(loss, (int, float)) and math.isfinite(loss):
                losses.append(loss)
        if losses:
            report["train_loss"] = {"first": losses[0], "last": losses[-1], "min": min(losses),
                                    "max": max(losses), "mean": math.fsum(losses) / len(losses),
                                    "last_50_mean": math.fsum(losses[-50:]) / len(losses[-50:])}
        validations = [record for record in records if "val/loss" in record.get("data", {})]
        require(bool(validations), "No validation loss was recorded")
        if validations:
            final = validations[-1]
            value = final["data"]["val/loss"]
            finite = isinstance(value, (int, float)) and math.isfinite(value)
            require(final["step"] == args.expected_steps and finite, "Final validation step/loss is invalid")
            report["validation_loss"] = value if finite else None
            report["validation_step"] = final["step"]
        report["peak_memory_gb"] = {key: max((record["data"][key] for record in records if key in record.get("data", {})), default=None)
                                     for key in ("perf/max_memory_allocated_gb", "perf/max_memory_reserved_gb", "perf/cpu_memory_used_gb")}
        latest = int((run / "latest_checkpointed_iteration.txt").read_text().strip())
        require(latest == args.expected_steps, f"Checkpoint tracker is {latest}, expected {args.expected_steps}")
        report["latest_checkpointed_iteration"] = latest
        checkpoint = run / f"global_step_{args.expected_steps}"
        fsdp = json.loads((checkpoint / "fsdp_config.json").read_text())
        require(fsdp.get("FSDP_version") == 2 and fsdp.get("world_size") == args.world_size, "FSDP version/world size mismatch")
        lora = json.loads((checkpoint / "lora_train_meta.json").read_text())
        require(lora.get("r") == 8 and lora.get("lora_alpha") == 16, "LoRA rank/alpha mismatch")
        report.update({"fsdp_config": fsdp, "lora_train_meta": lora})
        language_b_nonzero = 0
        with torch.no_grad():
            for rank in range(args.world_size):
                suffix = f"world_size_{args.world_size}_rank_{rank}.pt"
                paths = {kind: checkpoint / f"{kind}_{suffix}" for kind in ("model", "optim", "extra_state")}
                paths["data"] = checkpoint / f"data_{rank}.pt"
                for path in paths.values():
                    require(path.is_file() and path.stat().st_size > 0, f"Missing/empty checkpoint file: {path.name}")
                state = load(paths["model"])
                require(bool(state), f"Rank {rank}: empty model state")
                stats = {"rank": rank, "model_tensors": len(state), "lora_elements": 0,
                         "lora_nonzero_elements": 0, "language_model_lora_B_nonzero_elements": 0}
                for name, value in state.items():
                    require("lora_" in name or ".adapter_" in name, f"Rank {rank}: unexpected base-model tensor {name}")
                    value = _local(value)
                    require(bool(torch.isfinite(value).all()), f"Rank {rank}: nonfinite model tensor {name}")
                    nonzero = int(torch.count_nonzero(value))
                    stats["lora_elements"] += value.numel()
                    stats["lora_nonzero_elements"] += nonzero
                    if "language_model" in name and "lora_B" in name:
                        stats["language_model_lora_B_nonzero_elements"] += nonzero
                language_b_nonzero += stats["language_model_lora_B_nonzero_elements"]
                del state
                optimizer = load(paths["optim"])
                initialized, optimizer_steps = 0, set()
                for param, entry in optimizer["state"].items():
                    if not entry:
                        continue
                    initialized += 1
                    step = entry.get("step")
                    step = float(_local(step)) if isinstance(step, torch.Tensor) else step
                    require(step == args.expected_steps, f"Rank {rank}: optimizer state {param} has step {step}")
                    optimizer_steps.add(step)
                    for key in ("exp_avg", "exp_avg_sq"):
                        require(key in entry and bool(torch.isfinite(_local(entry[key])).all()), f"Rank {rank}: invalid {key} for {param}")
                require(initialized > 0, f"Rank {rank}: optimizer has no initialized state")
                stats.update({"optimizer_initialized_states": initialized, "optimizer_steps": sorted(optimizer_steps)})
                del optimizer
                extra = load(paths["extra_state"])
                scheduler_step = extra["lr_scheduler"]["last_epoch"]
                require(scheduler_step == args.expected_steps, f"Rank {rank}: scheduler step is {scheduler_step}")
                data = load(paths["data"])
                # Current torchdata single-process iterator nests BatchSampler state here.
                yielded = data["_num_yielded"]
                samples = data["_sampler_iter_state"]["samples_yielded"]
                require(yielded == args.expected_steps, f"Rank {rank}: dataloader yielded {yielded} batches")
                require(samples == args.train_rows // args.world_size, f"Rank {rank}: dataloader yielded {samples} samples")
                stats.update({"scheduler_last_epoch": scheduler_step, "dataloader_num_yielded": yielded,
                              "dataloader_samples_yielded": samples})
                report["ranks"].append(stats)
        report["language_model_lora_B_nonzero_elements"] = language_b_nonzero
        require(language_b_nonzero > 0, "All language-model LoRA B tensors remain zero")
        report["counts"]["train_rows_observed"] = sum(rank["dataloader_samples_yielded"] for rank in report["ranks"])
    except Exception as error:
        report["errors"].append(repr(error))
    report["passed"] = not report["errors"]
    run.mkdir(parents=True, exist_ok=True)
    (run / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
