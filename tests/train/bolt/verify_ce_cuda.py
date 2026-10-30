"""Gate chunked BOLT CE against dense CUDA CE before a Slurm training run.

Run with torchrun on allocated GPUs; each rank writes an independent JSON report.
The short tensors test numerical parity and do not estimate 8K training memory.
"""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from qwenpt.train.bolt.ce import chunked_linear_logprobs


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual, expected = actual.double().flatten(), expected.double().flatten()
    difference = actual - expected
    denominator = expected.norm().clamp_min(torch.finfo(torch.float64).tiny)
    return {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (difference.norm() / denominator).item(),
        "cosine": F.cosine_similarity(actual[None], expected[None]).item(),
    }


def _measure(
    initial_hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    upstream: torch.Tensor,
    *,
    chunk_size: int | None,
    training: bool,
) -> dict:
    hidden = initial_hidden.detach().clone().requires_grad_(training)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    grad_context = nullcontext() if training else torch.no_grad()
    amp_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if hidden.dtype == torch.bfloat16 else nullcontext()
    )
    with grad_context, amp_context:
        if chunk_size is None:
            logits = F.linear(hidden, weight).float()
            values = -F.cross_entropy(
                logits.reshape(-1, weight.shape[0]), labels.flatten(), reduction="none",
            ).reshape(labels.shape)
        else:
            values = chunked_linear_logprobs(hidden, weight, labels, chunk_size=chunk_size)
        objective = (values * upstream).sum()
    if training:
        objective.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return {
        "log_probs": values.detach().float().cpu(),
        "hidden_grad": hidden.grad.detach().float().cpu() if training else None,
        "weighted_objective": objective.item(),
        "requires_grad": values.requires_grad,
        "peak_allocated_bytes": peak,
        "peak_increment_bytes": peak - baseline,
    }


def _case(
    *, batch: int, sequence: int, hidden_size: int, vocabulary: int,
    dtype: torch.dtype, chunk_size: int, device: torch.device,
) -> dict:
    weight = torch.randn(vocabulary, hidden_size, device=device, dtype=dtype) * 0.02
    initial_hidden = torch.randn(batch, sequence, hidden_size, device=device, dtype=dtype)
    labels = torch.randint(vocabulary, (batch, sequence), device=device)
    labels.flatten()[:4] = torch.tensor([0, 1, vocabulary - 2, vocabulary - 1], device=device)
    upstream = torch.linspace(-1.5, 1.75, labels.numel(), device=device).reshape(labels.shape)
    upstream.flatten()[::7] = 0
    reference = _measure(initial_hidden, weight, labels, upstream, chunk_size=None, training=True)
    chunked = _measure(initial_hidden, weight, labels, upstream, chunk_size=chunk_size, training=True)
    inference = _measure(initial_hidden, weight, labels, upstream, chunk_size=chunk_size, training=False)
    inference_reference = _measure(initial_hidden, weight, labels, upstream, chunk_size=None, training=False)
    low_precision = dtype == torch.bfloat16
    atol, rtol = (2e-3, 1e-3) if low_precision else (5e-6, 1e-5)
    grad_limit = 0.02 if low_precision else 2e-5
    finite = all(
        torch.isfinite(result[key]).all().item()
        for result in (reference, chunked, inference, inference_reference)
        for key in ("log_probs", "hidden_grad") if result[key] is not None
    )
    if not finite:
        raise FloatingPointError("CE values or hidden gradients contain nonfinite values")
    gradient_error = _errors(chunked["hidden_grad"], reference["hidden_grad"])
    forward_passed = torch.allclose(chunked["log_probs"], reference["log_probs"], atol=atol, rtol=rtol)
    inference_passed = torch.allclose(
        inference["log_probs"], inference_reference["log_probs"], atol=atol, rtol=rtol,
    )
    gradient_passed = gradient_error["relative_l2"] <= grad_limit
    no_grad_passed = not inference["requires_grad"]
    passed = finite and forward_passed and inference_passed and gradient_passed and no_grad_passed
    result = {
        "status": "passed" if passed else "failed",
        "hidden_shape": list(initial_hidden.shape), "vocabulary": vocabulary,
        "dtype": str(dtype), "chunk_size": chunk_size, "head_frozen": not weight.requires_grad,
        "upstream_gradient": {"minimum": upstream.min().item(), "maximum": upstream.max().item()},
        "tolerances": {"log_probs_atol": atol, "log_probs_rtol": rtol, "gradient_relative_l2": grad_limit},
        "checks": {
            "finite": finite, "forward": forward_passed, "hidden_gradient": gradient_passed,
            "inference": inference_passed, "no_grad": no_grad_passed,
        },
        "log_probs_error": _errors(chunked["log_probs"], reference["log_probs"]),
        "hidden_gradient_error": gradient_error,
        "inference_error": _errors(inference["log_probs"], inference_reference["log_probs"]),
        "memory_scope": "Short CE parity tensors only; not an 8K model-training memory measurement.",
        "paths": {
            name: {key: value for key, value in measured.items() if key not in ("log_probs", "hidden_grad")}
            for name, measured in (
                ("dense_training", reference), ("chunked_training", chunked),
                ("chunked_inference", inference), ("dense_inference", inference_reference),
            )
        },
    }
    return result


def main() -> None:
    """Validate CE parity on LOCAL_RANK's GPU and write a per-rank JSON gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for per-rank JSON reports")
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    seed = 20261020 + rank
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"ce_cuda_parity_rank{rank}.json"
    report = {"status": "failed", "rank": rank, "local_rank": local_rank, "seed": seed, "cases": {}}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("This gate must run on an allocated Slurm CUDA GPU")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        report["gpu"] = torch.cuda.get_device_name(device)
        report["cases"]["bf16_real_vocabulary"] = _case(
            batch=2, sequence=128, hidden_size=5120, vocabulary=248320,
            dtype=torch.bfloat16, chunk_size=128, device=device,
        )
        torch.cuda.empty_cache()
        report["cases"]["fp32_small"] = _case(
            batch=2, sequence=17, hidden_size=64, vocabulary=257,
            dtype=torch.float32, chunk_size=11, device=device,
        )
        if any(case["status"] != "passed" for case in report["cases"].values()):
            raise AssertionError("Chunked CE numerical parity gate failed; inspect the per-rank report")
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": report["status"], "rank": rank, "report": str(report_path)}), flush=True)


if __name__ == "__main__":
    main()
