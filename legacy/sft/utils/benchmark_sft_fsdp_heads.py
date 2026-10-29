"""Audit nested FSDP2 head hooks, global gradients and AdamW updates on tiny inputs."""

import argparse
from datetime import timedelta
import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard
import torch.nn.functional as F

from legacy.sft.verl.sft_heads import call_sft_head_module, prepare_sft_targets, sft_head_logprobs


ARMS = ("module", "bypass_linear", "bypass_verl_torch", "bypass_verl_triton", "bypass_liger", "bypass_cce_exact",
        "module_verl_torch", "module_verl_triton", "module_liger", "module_cce_exact")


def _info(tensor):
    return {"dtype": str(tensor.dtype), "type": type(tensor).__name__, "shape": list(tensor.shape)}


def _compare(actual, expected):
    actual, expected = actual.detach().double(), expected.detach().double()
    difference = actual - expected
    squared_reference = expected.square().sum()
    relative = float((difference.square().sum() / squared_reference.clamp_min(1e-300)).sqrt())
    max_normalized = float(difference.abs().max() / expected.abs().max().clamp_min(1e-300))
    dot = (actual * expected).sum()
    return {
        "elements": actual.numel(), "relative_l2": relative, "max_absolute": float(difference.abs().max()),
        "max_normalized": max_normalized, "dot_ratio_to_reference": float(dot / squared_reference.clamp_min(1e-300)),
        "cosine": float(dot / (actual.norm() * expected.norm()).clamp_min(1e-300)),
        "passed": relative <= 0.02 and max_normalized <= 0.04,
    }


def _full(tensor):
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


class AuditHead(torch.nn.Linear):
    """Record the parameter and activation dtypes after FSDP's module pre-hook."""

    def __init__(self, hidden_size, vocab_size):
        super().__init__(hidden_size, vocab_size, bias=False)
        self.observations = []

    def forward(self, hidden, labels=None, backend=None):
        """Record the real module-call dtypes and perform the standard projection."""
        self.observations.append({"hidden": _info(hidden), "weight": _info(self.weight),
                                  "autocast": torch.is_autocast_enabled("cuda")})
        if backend is not None:
            return sft_head_logprobs(hidden.reshape(-1, hidden.shape[-1]), self.weight, labels,
                                     backend=backend, chunk_size=8)
        return F.linear(hidden, self.weight)


class TinyModel(torch.nn.Module):
    """Reproduce an active FSDP root with separately sharded backbone and head."""

    def __init__(self, hidden_size, vocab_size, arm):
        super().__init__()
        self.backbone = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lm_head = AuditHead(hidden_size, vocab_size)
        self.arm = arm
        self.observations = []
        self.last_hidden = None

    def forward(self, inputs, targets):
        """Call the real backbone, then the head module or its bypassed weight."""
        hidden = self.backbone(inputs)
        hidden.retain_grad()
        self.last_hidden = hidden
        observation = {"decoder_hidden": _info(hidden), "head_weight_outside": _info(self.lm_head.weight),
                       "autocast": torch.is_autocast_enabled("cuda")}
        if self.arm == "module":
            logits = self.lm_head(hidden)
            log_probs = -F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.labels, reduction="none")
        elif self.arm.startswith("module_"):
            def observe_head(head, inputs):
                head.observations.append({"hidden": _info(inputs[0]), "weight": _info(head.weight),
                                          "autocast": torch.is_autocast_enabled("cuda")})

            with self.lm_head.register_forward_pre_hook(observe_head):
                log_probs, _ = call_sft_head_module(self.lm_head, hidden, targets.labels.reshape(hidden.shape[:-1]),
                                                   backend=self.arm.removeprefix("module_"), chunk_size=8)
            log_probs = log_probs.flatten()
        else:
            weight = self.lm_head.weight.full_tensor()
            kernel_hidden = hidden.to(weight.dtype)
            observation.update(kernel_hidden=_info(kernel_hidden), kernel_weight=_info(weight))
            backend = "native" if self.arm == "bypass_linear" else self.arm.removeprefix("bypass_")
            log_probs = sft_head_logprobs(kernel_hidden.reshape(-1, hidden.shape[-1]), weight, targets.labels,
                                        backend=backend, chunk_size=8)
        self.observations.append(observation)
        return -(log_probs * targets.mask).sum() * targets.normalizer


def _batch(step, rank, device, hidden_size, vocab_size):
    length = 17 + rank * 6
    generator = torch.Generator(device=device).manual_seed(9501 + step * 71 + rank * 113)
    inputs = torch.randn(1, length, hidden_size, device=device, generator=generator, requires_grad=True)
    ids = torch.randint(vocab_size, (length,), device=device, generator=generator)
    mask = ((torch.arange(length, device=device) + step + rank) % 3 != 0).long()
    mask[0] = 0
    count = mask.sum()
    dist.all_reduce(count)
    targets = prepare_sft_targets(ids, mask, torch.tensor([0, length], device=device),
                                  global_supervised_tokens=int(count), dp_size=dist.get_world_size())
    return inputs, targets


def _arm(arm, args, mesh, device, initial_backbone, initial_head):
    rank, world = dist.get_rank(), dist.get_world_size()
    model = TinyModel(args.hidden_size, args.vocab_size, arm).to(device)
    with torch.no_grad():
        model.backbone.weight.copy_(initial_backbone)
        model.lm_head.weight.copy_(initial_head)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    options = dict(mesh=mesh, mp_policy=policy, shard_placement_fn=lambda param: Shard(0))
    fully_shard(model.backbone, **options)
    fully_shard(model.lm_head, **options)
    fully_shard(model, **options)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                                  weight_decay=0.01, foreach=False)
    ref_backbone = initial_backbone.detach().clone().requires_grad_()
    ref_head = initial_head.detach().clone().requires_grad_()
    ref_optimizer = torch.optim.AdamW([ref_backbone, ref_head], lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                                      weight_decay=0.01, foreach=False)
    steps = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)
        inputs, targets = _batch(step, rank, device, args.hidden_size, args.vocab_size)
        reference_inputs = inputs.detach().clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reference_hidden = F.linear(reference_inputs, ref_backbone)
            reference_hidden.retain_grad()
            backend = "native" if arm in ("module", "bypass_linear") else arm.split("_", 1)[1]
            if args.reference_backend == "same":
                reference_log_probs = sft_head_logprobs(reference_hidden.reshape(-1, args.hidden_size),
                                                       ref_head.to(torch.bfloat16), targets.labels,
                                                       backend=backend, chunk_size=8)
            else:
                reference_log_probs = sft_head_logprobs(reference_hidden.reshape(-1, args.hidden_size), ref_head,
                                                       targets.labels, backend="native")
            reference_loss = -(reference_log_probs * targets.mask).sum() * targets.normalizer
        reference_loss.backward()
        local_head_gradient = ref_head.grad.detach().clone()
        local_gradients = [torch.empty_like(local_head_gradient) for _ in range(world)]
        dist.all_gather(local_gradients, local_head_gradient)
        shard_rows = args.vocab_size // world
        stitched = torch.cat([gradient[index * shard_rows:(index + 1) * shard_rows]
                              for index, gradient in enumerate(local_gradients)])
        for parameter in (ref_backbone, ref_head):
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(world)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(inputs, targets)
        loss.backward()
        torch.cuda.synchronize(device)
        full_gradient = _full(model.lm_head.weight.grad)
        record = {
            "step": step + 1, "loss": float(loss.detach()), "reference_loss": float(reference_loss.detach()),
            "loss_passed": bool(torch.isclose(loss.detach().float(), reference_loss.detach().float(), rtol=0.002, atol=0.002)),
            "hidden_gradient_vs_serial_local": _compare(model.last_hidden.grad, reference_hidden.grad),
            "weight_gradient_vs_serial_global_mean": _compare(full_gradient, ref_head.grad),
            "weight_gradient_vs_stitched_rank_local": _compare(full_gradient, stitched),
            "masked_hidden_gradient_nonzero": int(torch.count_nonzero(model.last_hidden.grad.reshape(-1, args.hidden_size)[~targets.mask])),
            "observed_root_and_kernel": model.observations[-1],
            "observed_head_module": model.lm_head.observations[-1] if model.lm_head.observations else None,
            "grad_dtype": str(model.lm_head.weight.grad.dtype),
            "global_supervised_tokens": round(world / targets.normalizer),
        }
        optimizer.step()
        ref_optimizer.step()
        full_parameter = _full(model.lm_head.weight).detach()
        record["adam_parameter_vs_serial"] = _compare(full_parameter, ref_head)
        record["adam_update_vs_serial"] = _compare(full_parameter - initial_head, ref_head - initial_head)
        moment = optimizer.state[model.lm_head.weight]["exp_avg"]
        record["adam_first_moment_vs_serial"] = _compare(_full(moment), ref_optimizer.state[ref_head]["exp_avg"])
        steps.append(record)
        del inputs, reference_inputs, reference_hidden, reference_loss, reference_log_probs
        del loss, local_gradients, local_head_gradient, stitched, full_gradient, full_parameter
    passed = all(row["loss_passed"] and row["weight_gradient_vs_serial_global_mean"]["passed"]
                 and row["hidden_gradient_vs_serial_local"]["passed"]
                 and row["adam_update_vs_serial"]["passed"] for row in steps)
    result = {"arm": arm, "rank": rank, "passed": passed, "steps": steps}
    del optimizer, model, ref_optimizer, ref_backbone, ref_head
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return result


def main() -> None:
    """Run the nested FSDP2 fixture under torchrun and save every complete tensor comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--reference-backend", choices=("native", "same"), default="native")
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    world = dist.get_world_size()
    if world != 2 or args.vocab_size % world:
        raise ValueError("This fixture requires exactly two DP GPUs and an evenly sharded vocabulary")
    mesh = init_device_mesh("cuda", (world,))
    generator = torch.Generator(device=device).manual_seed(729)
    initial_backbone = torch.randn(args.hidden_size, args.hidden_size, device=device, generator=generator) * 0.03
    initial_head = torch.randn(args.vocab_size, args.hidden_size, device=device, generator=generator) * 0.05
    report = {"scope": "nested_FSDP2_FP32_master_BF16_compute_FP32_reduce", "world_size": world,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "reference_backend": args.reference_backend,
              "torch": torch.__version__, "tolerance": {"relative_l2": 0.02, "max_normalized": 0.04,
                                                         "loss_rtol": 0.002, "loss_atol": 0.002},
              "root_and_backbone_and_head_fully_sharded": True, "arms": []}
    for arm in args.arms:
        if dist.get_rank() == 0:
            print(json.dumps({"arm": arm, "status": "started"}), flush=True)
        result = _arm(arm, args, mesh, device, initial_backbone, initial_head)
        gathered = [None] * world
        dist.all_gather_object(gathered, result)
        report["arms"].extend(gathered)
        if dist.get_rank() == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"arm": arm, "passed_all_ranks": all(row["passed"] for row in gathered)}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
