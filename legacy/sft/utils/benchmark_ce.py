"""Compare native and chunked SFT CE numerics, CUDA time, and peak memory."""

import argparse
import gc
import json
import math
import os
import statistics
from pathlib import Path
from time import perf_counter

import torch

from legacy.sft.verl.ce import logprobs_from_logits_chunked


VOCAB_SIZE = 248320
CASES = (
    {"name": "bf16_short", "tokens": 37, "dtype": "bfloat16"},
    {"name": "fp32_short", "tokens": 37, "dtype": "float32"},
    {"name": "bf16_dynamic", "tokens": 19, "dtype": "bfloat16", "scale": 20, "shift": 128},
    {"name": "bf16_noncontiguous", "tokens": 31, "dtype": "bfloat16", "noncontiguous": True},
    {"name": "fp32_noncontiguous", "tokens": 31, "dtype": "float32", "noncontiguous": True},
    {"name": "bf16_full_8192", "tokens": 8192, "dtype": "bfloat16"},
)


def _compare(actual, expected, *, rtol, atol):
    actual, expected = actual.float(), expected.float()
    difference = (actual - expected).abs()
    finite = torch.isfinite(actual).all() & torch.isfinite(expected).all()
    close = torch.isclose(actual, expected, rtol=rtol, atol=atol)
    max_error = difference.max().item()
    return {
        "passed": bool(finite.item() and close.all().item()),
        "rtol": rtol,
        "atol": atol,
        "max_absolute_error": max_error if math.isfinite(max_error) else float("inf"),
        "mismatched_elements": int((~close).sum().item()),
        "elements": actual.numel(),
        "all_finite": bool(finite.item()),
    }


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _compare_gradients(actual, expected_cpu, mask, *, rtol, atol, chunk_size):
    result = {
        "passed": True, "rtol": rtol, "atol": atol, "max_absolute_error": 0.0,
        "mismatched_elements": 0, "elements": actual.numel(), "all_finite": True,
        "masked_gradient_nonzero": 0,
    }
    shifted_mask = mask.roll(-1)
    # Keep the full reference gradient on the host; only one comparison block uses CUDA.
    for start in range(0, actual.shape[0], chunk_size):
        block = actual[start : start + chunk_size]
        reference = expected_cpu[start : start + chunk_size].to(actual.device)
        comparison = _compare(block, reference, rtol=rtol, atol=atol)
        result["passed"] &= comparison["passed"]
        result["all_finite"] &= comparison["all_finite"]
        result["max_absolute_error"] = max(result["max_absolute_error"], comparison["max_absolute_error"])
        result["mismatched_elements"] += comparison["mismatched_elements"]
        excluded = shifted_mask[start : start + chunk_size] == 0
        result["masked_gradient_nonzero"] += int(torch.count_nonzero(block[excluded]).item())
        del reference, block
    result["passed"] &= result["masked_gradient_nonzero"] == 0
    return result


def _run_backend(logits, labels, data, backend, *, chunk_size, warmup, repeats):
    from verl.utils.torch_functional import logprobs_from_logits
    from verl.workers.utils.losses import sft_loss

    seconds, forwards, backwards, allocated, reserved = [], [], [], [], []
    baseline_bytes = torch.cuda.memory_allocated(logits.device)
    for iteration in range(warmup + repeats):
        logits.grad = None
        torch.cuda.synchronize(logits.device)
        torch.cuda.reset_peak_memory_stats(logits.device)
        started = perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if backend == "native":
                log_probs = logprobs_from_logits(logits, labels, inplace_backward=False)
            else:
                log_probs = logprobs_from_logits_chunked(logits, labels, chunk_size=chunk_size)
            nested = torch.nested.nested_tensor_from_jagged(log_probs, data["loss_mask"].offsets())
            loss, _ = sft_loss(None, {"log_probs": nested}, data)
        torch.cuda.synchronize(logits.device)
        forward_finished = perf_counter()
        loss.backward()
        torch.cuda.synchronize(logits.device)
        finished = perf_counter()
        if iteration >= warmup:
            forwards.append(forward_finished - started)
            backwards.append(finished - forward_finished)
            seconds.append(finished - started)
            allocated.append(torch.cuda.max_memory_allocated(logits.device))
            reserved.append(torch.cuda.max_memory_reserved(logits.device))
        values_cpu = log_probs.detach().cpu()
        loss_cpu = loss.detach().cpu()
        del nested, log_probs, loss
    return {
        "forward_backward_seconds": seconds,
        "forward_seconds": forwards,
        "backward_seconds": backwards,
        "mean_seconds": statistics.mean(seconds),
        "median_seconds": statistics.median(seconds),
        "peak_allocated_bytes": max(allocated),
        "peak_reserved_bytes": max(reserved),
        "baseline_allocated_bytes": baseline_bytes,
        "peak_extra_allocated_bytes": max(allocated) - baseline_bytes,
        "loss": loss_cpu.item(),
    }, values_cpu, loss_cpu


def benchmark_case(spec: dict, *, device, seed, chunk_size, warmup, repeats) -> dict:
    """Compare one deterministic input, retaining only one CUDA logits and gradient."""
    from tensordict import TensorDict
    from tensordict.tensorclass import NonTensorData

    generator = torch.Generator(device=device).manual_seed(seed)
    tokens, dtype = spec["tokens"], getattr(torch, spec["dtype"])
    noncontiguous = spec.get("noncontiguous", False)
    shape = (VOCAB_SIZE, tokens) if noncontiguous else (tokens, VOCAB_SIZE)
    logits = torch.randn(shape, dtype=dtype, device=device, generator=generator)
    logits.mul_(spec.get("scale", 1)).add_(spec.get("shift", 0))
    if noncontiguous:
        logits = logits.t()
    logits.requires_grad_()
    labels = torch.randint(VOCAB_SIZE, (tokens,), device=device, generator=generator)
    mask = (torch.rand(tokens, device=device, generator=generator) > 0.25).long()
    mask[0], mask[-1] = 0, 1
    count = int(mask.sum().item())
    # Simulate two DP ranks with equal token counts; use the actual native loss.
    data = TensorDict({
        "loss_mask": torch.nested.nested_tensor_from_jagged(mask, torch.tensor([0, tokens], device=device)),
        "pad_mode": NonTensorData("no_padding"), "dp_size": 2, "batch_num_tokens": count * 2,
    }, batch_size=[])
    sample_before = logits.detach()[:: max(tokens // 17, 1), ::4096].clone()
    native, reference_values, reference_loss = _run_backend(
        logits, labels, data, "native", chunk_size=chunk_size, warmup=warmup, repeats=repeats,
    )
    reference_gradient_cpu = logits.grad.detach().cpu()
    logits.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    chunked, values, loss = _run_backend(
        logits, labels, data, "chunked", chunk_size=chunk_size, warmup=warmup, repeats=repeats,
    )
    value_comparison = _compare(values, reference_values, rtol=1e-5, atol=2e-5)
    loss_comparison = _compare(loss.reshape(1), reference_loss.reshape(1), rtol=1e-5, atol=2e-5)
    # Scale the absolute gradient tolerance by native loss normalization.
    gradient_comparison = _compare_gradients(
        logits.grad, reference_gradient_cpu, mask,
        rtol=0.02 if dtype == torch.bfloat16 else 3e-5,
        atol=(3e-6 if dtype == torch.bfloat16 else 2e-6) / count,
        chunk_size=chunk_size,
    )
    input_unchanged = torch.equal(sample_before, logits.detach()[:: max(tokens // 17, 1), ::4096])
    result = {
        **spec, "vocab_size": VOCAB_SIZE, "seed": seed, "chunk_size": chunk_size,
        "contiguous": logits.is_contiguous(), "supervised_tokens": count,
        "warmup_iterations": warmup, "measured_iterations": repeats,
        "native": native, "chunked": chunked,
        "log_probs_comparison": value_comparison, "masked_loss_comparison": loss_comparison,
        "logits_gradient_comparison": gradient_comparison,
        "sampled_input_unchanged": input_unchanged,
        "median_speedup": native["median_seconds"] / chunked["median_seconds"],
        "performance_improved": chunked["median_seconds"] < native["median_seconds"],
        "passed": all(item["passed"] for item in (value_comparison, loss_comparison, gradient_comparison))
        and input_unchanged,
    }
    del logits, reference_gradient_cpu, data
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    """Run CUDA CE checks and atomically write JSON; fail on any numerical mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.chunk_size < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("chunk-size/repeats must be positive and warmup must be nonnegative")
    if not torch.cuda.is_available():
        parser.error("CUDA is required; run this benchmark in the Slurm GPU allocation")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    from verl.utils import torch_functional

    report = {
        "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "device_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "cuda_autocast_dtype": "bfloat16", "seed": args.seed,
        "native_flash_ce_available": torch_functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE,
        "native_flash_ce_disabled_env": os.environ.get("VERL_DISABLE_FLASH_ATTN_CE", "0"),
        "limitations": [
            "CE microbenchmark only: excludes model forward/backward, FSDP/offload, CPU optimizer and NUMA.",
            "FP32 uses short inputs; full 8192-by-248320 performance is tested in BF16.",
            "Numerical agreement uses tolerances, not bitwise equality; BF16 gradients allow 2% relative error.",
            "The reference full gradient resides on CPU; comparison and host transfers are excluded from timing.",
            "Native always precedes chunked; use full-training A/B to assess end-to-end speed.",
            "Memory peaks cover CE and its input/gradient, excluding a loaded model.",
        ],
        "cases": [],
    }
    for index, spec in enumerate(CASES):
        print(f"CE benchmark: {spec['name']}", flush=True)
        try:
            result = benchmark_case(
                spec, device=device, seed=args.seed + index, chunk_size=args.chunk_size,
                warmup=args.warmup, repeats=args.repeats,
            )
        except Exception as error:
            result = {**spec, "passed": False, "error": f"{type(error).__name__}: {error}"}
            gc.collect()
            torch.cuda.empty_cache()
        report["cases"].append(result)
        report["passed"] = len(report["cases"]) == len(CASES) and all(case["passed"] for case in report["cases"])
        report["completed_cases"] = len(report["cases"])
        report["expected_cases"] = len(CASES)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)
        print(json.dumps({key: result[key] for key in ("name", "passed", "median_speedup", "error") if key in result}), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
