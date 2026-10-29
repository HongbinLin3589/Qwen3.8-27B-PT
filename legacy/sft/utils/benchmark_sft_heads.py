"""Compare trainable SFT output-head accuracy and warmed CUDA head-step cost.

This is an output-head microbenchmark, not a full-model training benchmark.
Both the hidden-state gradient and every lm_head weight-gradient element are checked.
"""

import argparse
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
from time import perf_counter

import torch

from legacy.sft.verl.sft_heads import PROBE_BACKENDS, prepare_sft_targets, sft_head_loss


# Fixed before running any kernels. A failure is reported, never used to relax these.
TOLERANCES = {
    "float32": {"loss_rtol": 1e-5, "loss_atol": 2e-5, "gradient_relative_l2": 1e-4, "gradient_max_normalized": 3e-4},
    "bfloat16": {"loss_rtol": 2e-3, "loss_atol": 2e-3, "gradient_relative_l2": 0.02, "gradient_max_normalized": 0.04},
}


def _compare_gradient(actual, expected_cpu, *, dtype, chunk_rows=1024):
    tolerance = TOLERANCES[dtype]
    squared_error, squared_reference, max_error, max_reference, non_finite = 0.0, 0.0, 0.0, 0.0, 0
    # Compare every element without keeping two vocabulary-sized gradients on CUDA.
    for start in range(0, actual.shape[0], chunk_rows):
        value = actual[start : start + chunk_rows].float()
        reference = expected_cpu[start : start + chunk_rows].to(device=actual.device, dtype=torch.float32)
        difference = value - reference
        non_finite += int((~torch.isfinite(value) | ~torch.isfinite(reference)).sum().item())
        squared_error += difference.double().square().sum().item()
        squared_reference += reference.double().square().sum().item()
        max_error = max(max_error, difference.abs().max().item())
        max_reference = max(max_reference, reference.abs().max().item())
    relative_l2 = math.sqrt(squared_error / squared_reference) if squared_reference else float(bool(squared_error))
    max_normalized = max_error / max_reference if max_reference else float(bool(max_error))
    return {
        "elements_compared": actual.numel(), "non_finite_elements": non_finite,
        "relative_l2_error": relative_l2, "max_absolute_error": max_error,
        "max_reference_absolute": max_reference, "max_normalized_error": max_normalized,
        "passed": non_finite == 0 and relative_l2 <= tolerance["gradient_relative_l2"]
        and max_normalized <= tolerance["gradient_max_normalized"],
    }


def _compare_loss(actual, expected, dtype):
    tolerance = TOLERANCES[dtype]
    absolute = abs(actual - expected)
    return {
        "absolute_error": absolute, "reference": expected, "actual": actual,
        "passed": math.isfinite(actual) and math.isfinite(expected)
        and absolute <= tolerance["loss_atol"] + tolerance["loss_rtol"] * abs(expected),
    }


def _clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def _run_accuracy(hidden_cpu, weight_cpu, targets, *, backend, dtype, device, chunk_size, references):
    hidden = hidden_cpu.to(device=device, dtype=dtype).requires_grad_()
    weight = weight_cpu.to(device=device, dtype=dtype).requires_grad_()
    loss = sft_head_loss(hidden, weight, targets, backend=backend, chunk_size=chunk_size)
    loss.backward()
    torch.cuda.synchronize(device)
    result = {
        "loss": loss.item(), "hidden_dtype": str(hidden.dtype), "weight_dtype": str(weight.dtype),
        "loss_dtype": str(loss.dtype), "hidden_gradient_dtype": str(hidden.grad.dtype),
        "weight_gradient_dtype": str(weight.grad.dtype),
        "all_hidden_gradient_elements": hidden.numel(), "all_weight_gradient_elements": weight.numel(),
    }
    result["masked_hidden_gradient_nonzero"] = int(torch.count_nonzero(hidden.grad[~targets.mask]).item())
    for name, reference in references.items():
        comparisons = {
            "loss": _compare_loss(loss.item(), reference["loss"], str(dtype).removeprefix("torch.")),
            "hidden_gradient": _compare_gradient(hidden.grad, reference["hidden_gradient"], dtype=str(dtype).removeprefix("torch.")),
            "weight_gradient": _compare_gradient(weight.grad, reference["weight_gradient"], dtype=str(dtype).removeprefix("torch.")),
        }
        comparisons["passed"] = all(item["passed"] for item in comparisons.values())
        result[f"vs_{name}"] = comparisons
    saved_reference = None
    if backend == "native":
        saved_reference = {
            "loss": loss.item(), "hidden_gradient": hidden.grad.detach().cpu(), "weight_gradient": weight.grad.detach().cpu(),
        }
    result["passed"] = result["masked_hidden_gradient_nonzero"] == 0 and all(
        comparison["passed"] for key, comparison in result.items() if key.startswith("vs_")
    )
    del hidden, weight, loss
    _clear_cuda()
    return result, saved_reference


def _run_timing(hidden_cpu, weight_cpu, targets, *, backend, dtype, device, chunk_size, warmup, repeats, step_lr):
    hidden = hidden_cpu.to(device=device, dtype=dtype).requires_grad_()
    weight = weight_cpu.to(device=device, dtype=dtype).requires_grad_()
    forwards, backwards, steps, peak_allocated, peak_reserved = [], [], [], [], []
    baseline = torch.cuda.memory_allocated(device)
    for iteration in range(warmup + repeats):
        hidden.grad = weight.grad = None
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        loss = sft_head_loss(hidden, weight, targets, backend=backend, chunk_size=chunk_size)
        torch.cuda.synchronize(device)
        forward_finished = perf_counter()
        loss.backward()
        torch.cuda.synchronize(device)
        backward_finished = perf_counter()
        if step_lr:
            with torch.no_grad():
                weight.add_(weight.grad, alpha=-step_lr)
        torch.cuda.synchronize(device)
        finished = perf_counter()
        if iteration >= warmup:
            forwards.append(forward_finished - started)
            backwards.append(backward_finished - forward_finished)
            steps.append(finished - started)
            peak_allocated.append(torch.cuda.max_memory_allocated(device))
            peak_reserved.append(torch.cuda.max_memory_reserved(device))
        del loss
    median = statistics.median(steps)
    result = {
        "warmup_iterations": warmup, "measured_iterations": repeats,
        "forward_seconds": forwards, "backward_seconds": backwards, "head_step_seconds": steps,
        "median_head_step_seconds": median, "mean_head_step_seconds": statistics.mean(steps),
        "effective_input_tokens_per_second": hidden.shape[0] / median,
        "supervised_tokens_per_second": int(targets.mask.sum().item()) / median,
        "peak_allocated_bytes": max(peak_allocated), "peak_reserved_bytes": max(peak_reserved),
        "baseline_allocated_bytes": baseline, "peak_extra_allocated_bytes": max(peak_allocated) - baseline,
        "head_sgd_step_lr": step_lr,
    }
    del hidden, weight
    _clear_cuda()
    return result


def _weight(args):
    if args.model_path is None:
        generator = torch.Generator().manual_seed(args.seed)
        return torch.randn(args.vocab_size, args.hidden_size, generator=generator, dtype=torch.bfloat16).mul_(0.02), {
            "source": "synthetic_normal_std_0.02", "hidden_size": args.hidden_size, "vocab_size": args.vocab_size,
        }
    from safetensors import safe_open

    index_path = args.model_path / "model.safetensors.index.json"
    if index_path.exists():
        mapping = json.loads(index_path.read_text())["weight_map"]
        names = [name for name in mapping if name.endswith("lm_head.weight")]
        if len(names) != 1:
            raise ValueError(f"Expected one untied lm_head weight; found {names}")
        key, shard = names[0], args.model_path / mapping[names[0]]
    else:
        shard = args.model_path / "model.safetensors"
        with safe_open(shard, framework="pt", device="cpu") as handle:
            names = [name for name in handle.keys() if name.endswith("lm_head.weight")]
        if len(names) != 1:
            raise ValueError(f"Expected one untied lm_head weight; found {names}")
        key = names[0]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(key)
    return weight, {"source": "checkpoint_lm_head", "model_path": str(args.model_path), "key": key, "shard": shard.name}


def _inputs(tokens, hidden_size, vocab_size, seed):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(tokens, hidden_size, dtype=torch.bfloat16, generator=generator)
    ids = torch.randint(vocab_size, (tokens,), generator=generator)
    # Two packed examples each contain prompt, masked historical assistant, and
    # supervised answer spans. The second sample's first token is never a label.
    boundaries = [0, tokens // 2, tokens]
    mask = torch.zeros(tokens, dtype=torch.long)
    for start, end in zip(boundaries, boundaries[1:]):
        length = end - start
        mask[start + max(1, length // 4) : start + length // 2] = 1
        mask[start + max(1, 3 * length // 4) : end] = 1
    return hidden, ids, mask, torch.tensor(boundaries)


def _versions():
    result = {}
    for name in ("torch", "triton", "verl", "liger-kernel", "cut-cross-entropy", "flash-attn", "transformers"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    """Run paired accuracy/performance probes and write JSON after each completed arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--input-file", type=Path, help="Optional torch snapshot with hidden/input_ids/loss_mask/offsets")
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--tokens", type=int, nargs="+", default=[127, 2048, 8192])
    parser.add_argument("--dtypes", nargs="+", choices=TOLERANCES, default=["float32", "bfloat16"])
    parser.add_argument("--backends", nargs="+", choices=PROBE_BACKENDS, default=list(PROBE_BACKENDS))
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--dp-size", type=int, default=8)
    parser.add_argument("--global-supervised-tokens", type=int, help="Actual captured optimizer batch token denominator across DP")
    parser.add_argument("--head-sgd-step-lr", type=float, default=0.0)
    parser.add_argument("--accuracy-only", action="store_true")
    args = parser.parse_args()
    if min(args.tokens) < 8 or args.warmup < 0 or args.repeats < 1 or args.dp_size < 1 or args.chunk_size < 1:
        parser.error("tokens must be >=8, repeats/dp-size/chunk-size positive, warmup nonnegative")
    if args.global_supervised_tokens is not None and args.global_supervised_tokens < 1:
        parser.error("global-supervised-tokens must be positive")
    if not math.isfinite(args.head_sgd_step_lr) or args.head_sgd_step_lr < 0:
        parser.error("head-sgd-step-lr must be finite and nonnegative")
    if not torch.cuda.is_available():
        parser.error("CUDA is required; run only on the allocated/explicitly authorized GPU")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    weight_cpu, weight_info = _weight(args)
    captured = torch.load(args.input_file, map_location="cpu", weights_only=True) if args.input_file else None
    if captured is not None:
        args.tokens = [captured["hidden"].shape[0]]
    from verl.utils import torch_functional as verl_functional

    report = {
        "schema_version": 1, "scope": "output_head_microbenchmark_not_full_model_training",
        "status": "running", "passed": False,
        "expected_arms": len(args.tokens) * len(args.dtypes) * (1 + len(set(args.backends) - {"native"})),
        "completed_arms": 0,
        "versions": _versions(), "cuda_runtime": torch.version.cuda, "device_name": torch.cuda.get_device_name(device),
        "verl_flash_ce_available": verl_functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE,
        "verl_disable_flash_attn_ce_env": os.environ.get("VERL_DISABLE_FLASH_ATTN_CE", "0"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "weight": {**weight_info, "shape": list(weight_cpu.shape), "dtype": str(weight_cpu.dtype)},
        "hidden_source": str(args.input_file) if captured is not None else "synthetic_bf16_normal_std_1_then_same_quantized_inputs_for_fp32_reference",
        "tolerances_fixed_before_execution": TOLERANCES,
        "git_source": "working_tree_files_not_installed_packages; record repository commit with full-model run",
        "chunk_size": args.chunk_size,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "semantics": {
            "label_shift": "torch.roll(input_ids, -1) once before any backend",
            "mask_shift": "torch.roll(binary_message_aware_loss_mask, -1) once",
            "sample_boundaries": "first token of every sample masked; final hidden of every sample unsupervised",
            "normalization": "local_loss_sum * dp_size / global_optimizer_batch_supervised_tokens",
            "global_denominator_source": "CLI actual optimizer-batch count" if args.global_supervised_tokens else "synthetic unequal ranks: local_supervised * dp_size + 7",
            "native_reference": "full F.linear logits + FP32 torch F.cross_entropy",
            "verl_unfused": "full F.linear logits + actual verl.logprobs_from_logits (Flash CE when enabled)",
            "trainable_lm_head": True, "weight_and_hidden_gradient_comparison": "all_elements_streamed_no_sampling",
            "cce": "cce_exact; filter_eps=None; filter_e_grad=False; filter_c_grad=False; accum_e/c_fp32=True",
            "liger": "FLCE reduction=none; external nonuniform binary message mask; requires >=0.8.5",
            "timing": "CUDA synchronized warmed forward+backward, optional SGD head update; no full transformer/FSDP/AdamW",
            "input_tokens_per_second": "all actual input tokens, including context/masked messages; excludes padding",
            "memory": "CUDA allocator allocated/reserved peaks of this head-only process, including inputs/gradients",
        },
        "cases": [],
    }
    backends = ["native", *dict.fromkeys(backend for backend in args.backends if backend != "native")]
    for tokens in args.tokens:
        if captured is None:
            hidden_cpu, ids_cpu, mask_cpu, offsets_cpu = _inputs(tokens, weight_cpu.shape[1], weight_cpu.shape[0], args.seed + tokens)
        else:
            hidden_cpu, ids_cpu, mask_cpu, offsets_cpu = (captured[key] for key in ("hidden", "input_ids", "loss_mask", "offsets"))
        global_count = args.global_supervised_tokens or int(mask_cpu.sum()) * args.dp_size + 7
        target = prepare_sft_targets(
            ids_cpu.to(device), mask_cpu.to(device), offsets_cpu.to(device),
            global_supervised_tokens=global_count, dp_size=args.dp_size,
        )
        case = {
            "tokens": tokens, "supervised_tokens": int(mask_cpu.sum()), "seed": args.seed + tokens,
            "dp_size": args.dp_size, "global_supervised_tokens": global_count,
            "local_loss_normalizer": target.normalizer, "arms": [],
        }
        report["cases"].append(case)
        reference_fp32 = None
        try:
            _, reference_fp32 = _run_accuracy(
                hidden_cpu, weight_cpu, target, backend="native", dtype=torch.float32, device=device,
                chunk_size=args.chunk_size, references={},
            )
        except Exception as error:
            case["fp32_reference_error"] = f"{type(error).__name__}: {error}"
            _clear_cuda()
        for dtype_name in args.dtypes:
            reference_dtype = None
            for backend in backends:
                arm = {"backend": backend, "dtype": dtype_name}
                case["arms"].append(arm)
                print(json.dumps({"tokens": tokens, **arm, "status": "started"}), flush=True)
                try:
                    references = {"fp32_reference": reference_fp32} if reference_fp32 is not None else {}
                    if reference_dtype is not None:
                        references["same_dtype_native"] = reference_dtype
                    accuracy, native_reference = _run_accuracy(
                        hidden_cpu, weight_cpu, target, backend=backend, dtype=getattr(torch, dtype_name), device=device,
                        chunk_size=args.chunk_size, references=references,
                    )
                    if backend == "native":
                        reference_dtype = native_reference
                    arm["accuracy"] = accuracy
                    arm["passed"] = accuracy["passed"] and reference_fp32 is not None
                    if not args.accuracy_only:
                        arm["performance"] = _run_timing(
                            hidden_cpu, weight_cpu, target, backend=backend, dtype=getattr(torch, dtype_name),
                            device=device, chunk_size=args.chunk_size, warmup=args.warmup, repeats=args.repeats,
                            step_lr=args.head_sgd_step_lr,
                        )
                except Exception as error:
                    arm.update({"passed": False, "error": f"{type(error).__name__}: {error}"})
                    _clear_cuda()
                report["completed_arms"] = sum("passed" in arm for item in report["cases"] for arm in item["arms"])
                report["passed"] = report["completed_arms"] == report["expected_arms"] and all(
                    arm.get("passed", False) for item in report["cases"] for arm in item["arms"]
                )
                _write(args.output, report)
                print(json.dumps({key: arm[key] for key in ("backend", "dtype", "passed", "error") if key in arm}), flush=True)
            del reference_dtype
        del reference_fp32, hidden_cpu, target
        _clear_cuda()
    report["status"] = "complete"
    _write(args.output, report)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
