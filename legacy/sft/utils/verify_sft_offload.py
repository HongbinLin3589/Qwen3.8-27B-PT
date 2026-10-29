"""Compare nested FSDP2 CPU and saved-tensor offload against the same GPU kernel."""

import argparse
from contextlib import ExitStack, nullcontext
from datetime import timedelta
from functools import partial
import gc
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard
from torch.utils.checkpoint import checkpoint
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding,
)

from legacy.sft.verl.sft_heads import call_sft_head_module, prepare_sft_targets


TOLERANCES = {"loss_rtol": 0.002, "loss_atol": 0.002, "relative_l2": 0.02,
              "max_normalized": 0.04, "adam_rtol": 1e-5, "adam_atol": 1e-7}
ADAM_OPTIONS = {"lr": 3e-4, "betas": (0.9, 0.95), "eps": 1e-8,
                "weight_decay": 0.01, "foreach": False, "fused": False}


def compare_tensors(actual: torch.Tensor, expected: torch.Tensor, *, strict: bool = False) -> dict:
    """Compare every element using fixed BF16 gates or strict FP32 Adam tolerances."""
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    if actual.shape != expected.shape:
        raise ValueError("Tensor comparison requires identical shapes")
    difference = actual - expected
    relative = float(difference.norm() / expected.norm().clamp_min(1e-300))
    normalized = float(difference.abs().max() / expected.abs().max().clamp_min(1e-300))
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    passed = bool(torch.allclose(actual, expected, rtol=TOLERANCES["adam_rtol"],
                                 atol=TOLERANCES["adam_atol"])) if strict else (
        relative <= TOLERANCES["relative_l2"] and normalized <= TOLERANCES["max_normalized"]
    )
    return {"shape": list(actual.shape), "elements": actual.numel(), "relative_l2": relative,
            "max_normalized": normalized, "max_absolute": float(difference.abs().max()),
            "bitwise_equal": bool(torch.equal(actual, expected)), "finite": finite,
            "passed": finite and passed}


def batch_definition(step: int, rank: int, micro: int, hidden_size: int, vocab_size: int) -> dict:
    """Create reproducible unequal rank/micro masks, including an all-masked local microbatch."""
    length = 17 + rank * 6 + micro * 4
    generator = torch.Generator().manual_seed(9501 + step * 71 + rank * 113 + micro * 193)
    inputs = torch.randn(1, length, hidden_size, generator=generator)
    ids = torch.randint(vocab_size, (length,), generator=generator)
    mask = ((torch.arange(length) + step + rank + micro) % 3 != 0).long()
    offsets = torch.tensor([0, length // 2, length])
    mask[offsets[:-1]] = 0
    mask[3:6] = 0
    if rank == 1 and micro == 1:
        mask.zero_()
    return {"inputs": inputs, "ids": ids, "mask": mask, "offsets": offsets}


class CheckpointBackbone(GradientCheckpointingLayer):
    """Use the same non-reentrant layer-call checkpoint wrapper as Transformers."""

    def __init__(self, hidden_size: int) -> None:
        """Build two trainable projections and enable native checkpoint recomputation."""
        super().__init__()
        self.in_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.gradient_checkpointing = True
        self._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        self.forward_calls = 0
        self.is_recompute = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply a token-local nonlinear backbone, counting forward and recompute calls."""
        self.forward_calls += 1
        # This fixture executes exactly one training forward and one recompute per microbatch.
        self.is_recompute = self.forward_calls % 2 == 0
        try:
            return self.out_proj(torch.nn.functional.silu(self.in_proj(inputs)))
        finally:
            self.is_recompute = False


class OffloadAuditModel(torch.nn.Module):
    """Exercise an active root and separately sharded checkpointed backbone and head."""

    def __init__(self, hidden_size: int, vocab_size: int, backend: str, *, qwen_backbone: bool = False) -> None:
        """Build the same FP32 parameter layout for GPU and CPU-offloaded arms."""
        super().__init__()
        self.root_gain = torch.nn.Parameter(torch.ones(hidden_size))
        self.backbone = QwenAuditBackbone(hidden_size) if qwen_backbone else CheckpointBackbone(hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.backend = backend
        self.last_hidden = None

    def forward(self, inputs, targets):
        """Run the real head module hooks and preserve shifted global-token normalization."""
        hidden = self.backbone(inputs * self.root_gain)
        hidden.retain_grad()
        self.last_hidden = hidden
        backend = "verl_unfused" if self.backend == "native" else self.backend
        log_probs, _ = call_sft_head_module(self.lm_head, hidden, targets.labels.view(hidden.shape[:-1]),
                                           backend=backend, chunk_size=8)
        return -(log_probs.flatten() * targets.mask).sum() * targets.normalizer


class CountingQwenDecoder(Qwen3_5DecoderLayer):
    """Count actual HF decoder forward/recompute calls without changing its equations."""

    def __init__(self, config, layer_idx):
        """Create the real decoder and retain HF's non-reentrant outer checkpoint."""
        super().__init__(config, layer_idx)
        self.forward_calls = 0
        self.gradient_checkpointing = True
        self._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)

    def forward(self, *args, **kwargs):
        """Count the unchanged decoder computation, including checkpoint recomputations."""
        self.forward_calls += 1
        return super().forward(*args, **kwargs)


class QwenAuditBackbone(torch.nn.Module):
    """Use two actual full-attention Qwen decoders with unchanged sublayer GEMM shapes."""

    def __init__(self, hidden_size):
        """Construct an attention/MLP fixture; it does not test packed FLA/SP correctness."""
        super().__init__()
        config = Qwen3_5TextConfig(hidden_size=hidden_size, intermediate_size=4 * hidden_size,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=64,
            layer_types=["full_attention"] * 2, use_cache=False, max_position_embeddings=4096,
            rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
                "partial_rotary_factor": 1.0, "mrope_section": [11, 11, 10], "mrope_interleaved": True})
        config._attn_implementation = "sdpa"
        self.layers = torch.nn.ModuleList([CountingQwenDecoder(config, index) for index in range(2)])
        self.norm = Qwen3_5RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.rotary = Qwen3_5TextRotaryEmbedding(config)
        self.is_recompute = False

    @property
    def forward_calls(self):
        """Sum the real decoder calls across both layers."""
        return sum(layer.forward_calls for layer in self.layers)

    def forward(self, hidden):
        """Run ordinary causal SDPA, Qwen normalization and whole MLP equations."""
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        rope_positions = positions.unsqueeze(0).expand(3, hidden.shape[0], -1)
        embeddings = self.rotary(hidden, rope_positions)
        for layer in self.layers:
            hidden = layer(hidden, position_embeddings=embeddings, attention_mask=None,
                           position_ids=positions, past_key_values=None, use_cache=False)
        return self.norm(hidden)


def _full_cpu(tensor):
    if tensor is None:
        raise RuntimeError("A trainable audit tensor has no gradient")
    if isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    return tensor.detach().cpu().clone()


def _initial_state(args):
    if args.candidate_sublayer_checkpointing:
        # Preserve the original seed; a new architecture is explicit in the report, not a seed search.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(729)
            model = OffloadAuditModel(args.hidden_size, args.vocab_size, "native", qwen_backbone=True)
        return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    generator = torch.Generator().manual_seed(729)
    return {"root_gain": torch.ones(args.hidden_size),
            "backbone.in_proj.weight": torch.randn(args.hidden_size, args.hidden_size,
                                                    generator=generator) * 0.03,
            "backbone.out_proj.weight": torch.randn(args.hidden_size, args.hidden_size,
                                                     generator=generator) * 0.03,
            "lm_head.weight": torch.randn(args.vocab_size, args.hidden_size, generator=generator) * 0.05}


def _saved_tensor_context(receipt, backbone):
    context = torch.autograd.graph.save_on_cpu(pin_memory=False)
    native_pack = context.pack_hook

    def audited_pack(tensor):
        packed = native_pack(tensor)
        receipt["tensors"] += 1
        receipt["cuda_tensors"] += int(tensor.device.type == "cuda")
        receipt["cuda_bytes"] += tensor.numel() * tensor.element_size() if tensor.device.type == "cuda" else 0
        if backbone.is_recompute and tensor.device.type == "cuda":
            receipt["recompute_cuda_tensors"] += 1
            receipt["recompute_cuda_bytes"] += tensor.numel() * tensor.element_size()
        receipt["copies_preserve_dtype_and_values"] &= bool(
            packed[1].device.type == "cpu" and packed[1].dtype == tensor.dtype
            and torch.equal(packed[1], tensor.detach().cpu())
        )
        return packed

    context.pack_hook = audited_pack
    return context


def write_sharded_checkpoint(model, optimizer, path: Path, completed_step: int) -> dict:
    """Save CPU local state shards and the completed optimizer-step counter."""
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_state, optimizer_state = get_state_dict(model, optimizer, options=options)

    def tensor_devices(value):
        if isinstance(value, torch.Tensor):
            return {value.to_local().device.type if isinstance(value, DTensor) else value.device.type}
        if isinstance(value, dict):
            return set().union(*(tensor_devices(item) for item in value.values()))
        if isinstance(value, (list, tuple)):
            return set().union(*(tensor_devices(item) for item in value))
        return set()

    devices = tensor_devices((model_state, optimizer_state))
    if devices != {"cpu"}:
        raise RuntimeError(f"Expected CPU checkpoint shards, found {devices}")
    payload = {"scope": "torch_sharded_state_dict_not_native_trainer_checkpoint",
               "model": model_state, "optimizer": optimizer_state, "global_step": completed_step,
               "cpu_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {"path": str(path), "bytes": path.stat().st_size, "global_step": completed_step,
            "scope": payload["scope"], "local_tensor_devices": sorted(devices)}


def restore_sharded_checkpoint(model, optimizer, path: Path, expected_step: int) -> int:
    """Restore this fixture's own local shard file into a fresh model and optimizer."""
    payload = torch.load(path, weights_only=False)
    if payload["scope"] != "torch_sharded_state_dict_not_native_trainer_checkpoint":
        raise ValueError("Unrecognized offload fixture checkpoint")
    if payload["global_step"] != expected_step:
        raise ValueError("Checkpoint optimizer-step counter does not match the continuation")
    set_state_dict(model, optimizer, model_state_dict=payload["model"], optim_state_dict=payload["optimizer"],
                   options=StateDictOptions(full_state_dict=False, cpu_offload=True))
    torch.set_rng_state(payload["cpu_rng"])
    if payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload["global_step"]


def _build_arm(args, mesh, device, initial, backend, offload, patch_stack):
    model = OffloadAuditModel(args.hidden_size, args.vocab_size, backend,
                             qwen_backbone=args.candidate_sublayer_checkpointing)
    if offload and args.candidate_reentrant:
        model.backbone._gradient_checkpointing_func = partial(checkpoint, use_reentrant=True)
    model.load_state_dict(initial)
    model.to("cpu" if offload else device)
    options = {"mesh": mesh, "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                                               reduce_dtype=torch.float32),
               "shard_placement_fn": lambda param: Shard(0), "reshard_after_forward": True}
    if offload:
        options["offload_policy"] = CPUOffloadPolicy(pin_memory=False)
    if args.candidate_sublayer_checkpointing:
        for layer in model.backbone.layers:
            fully_shard(layer, **options)
    else:
        fully_shard(model.backbone, **options)
    fully_shard(model.lm_head, **options)
    fully_shard(model, **options)
    # Synchronize and reduce-scatter each microbatch, matching the intended production path.
    model.set_requires_gradient_sync(True)
    optimizer = torch.optim.AdamW(model.parameters(), **ADAM_OPTIONS)
    from legacy.sft.verl.checkpointing import qwen3_5_sublayer_checkpointing

    model.sublayer_checkpoint_names = patch_stack.enter_context(qwen3_5_sublayer_checkpointing(
        model, enabled=bool(offload and args.candidate_sublayer_checkpointing),
        include_layernorms=bool(offload and getattr(args, "candidate_checkpoint_layernorms", False))))
    model.gemm_shapes = {}
    model.norm_shapes = {}
    if args.candidate_sublayer_checkpointing:
        def observe_shape(name, shapes):
            def observed(module, inputs):
                shapes.setdefault(name, set()).add(tuple(inputs[0].shape))
            return observed

        for name, module in model.backbone.named_modules():
            shapes = model.gemm_shapes if isinstance(module, torch.nn.Linear) else (
                model.norm_shapes if isinstance(module, Qwen3_5RMSNorm) else None)
            if shapes is not None:
                handle = module.register_forward_pre_hook(observe_shape(name, shapes))
                patch_stack.callback(handle.remove)
    return model, optimizer


def _run_arm(args, mesh, device, initial, backend, *, offload):
    with ExitStack() as patch_stack:
        return _run_arm_scoped(args, mesh, device, initial, backend, offload=offload, patch_stack=patch_stack)


def _run_arm_scoped(args, mesh, device, initial, backend, *, offload, patch_stack):
    rank, world = dist.get_rank(), dist.get_world_size()
    model, optimizer = _build_arm(args, mesh, device, initial, backend, offload, patch_stack)
    snapshots = []
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        model.gemm_shapes.clear()
        model.norm_shapes.clear()
        batches = [batch_definition(step, rank, micro, args.hidden_size, args.vocab_size)
                   for micro in range(args.microbatches)]
        global_count = torch.tensor(sum(int(batch["mask"].sum()) for batch in batches))
        dist.all_reduce(global_count)
        micros = []
        call_start = model.backbone.forward_calls
        saved_tensor_receipt = {"tensors": 0, "cuda_tensors": 0, "cuda_bytes": 0,
                                "recompute_cuda_tensors": 0, "recompute_cuda_bytes": 0,
                                "copies_preserve_dtype_and_values": True}
        context = _saved_tensor_context(saved_tensor_receipt, model.backbone) if offload else nullcontext()
        with context:
            for batch in batches:
                inputs = batch["inputs"].to(device).requires_grad_()
                targets = prepare_sft_targets(batch["ids"].to(device), batch["mask"].to(device),
                                              batch["offsets"].to(device),
                                              global_supervised_tokens=int(global_count), dp_size=world)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(inputs, targets)
                loss.backward()
                torch.cuda.synchronize(device)
                micros.append({"loss": _full_cpu(loss), "dInputs": _full_cpu(inputs.grad),
                               "dHidden": _full_cpu(model.last_hidden.grad),
                               "dW_accumulated": _full_cpu(model.lm_head.weight.grad),
                               "shifted_mask": targets.mask.detach().cpu(),
                               "labels": targets.labels.detach().cpu(),
                               "normalizer": targets.normalizer,
                               "local_supervised_tokens": int(batch["mask"].sum())})
        del inputs, targets, loss
        gradients = {name: _full_cpu(parameter.grad) for name, parameter in model.named_parameters()}
        before = {name: _full_cpu(parameter) for name, parameter in model.named_parameters()}
        placements = {name: {"param_device": str(parameter.device), "param_dtype": str(parameter.dtype),
                              "grad_device": str(parameter.grad.device), "grad_dtype": str(parameter.grad.dtype)}
                      for name, parameter in model.named_parameters()}
        optimizer.step()
        after = {name: _full_cpu(parameter) for name, parameter in model.named_parameters()}
        moments = {name: {key: _full_cpu(optimizer.state[parameter][key]) for key in ("exp_avg", "exp_avg_sq")}
                   for name, parameter in model.named_parameters()}
        adam_steps = {name: float(optimizer.state[parameter]["step"]) for name, parameter in model.named_parameters()}
        snapshots.append({"step": step + 1, "global_supervised_tokens": int(global_count),
                          "micros": micros, "gradients": gradients, "before": before, "parameters": after,
                          "moments": moments, "placements": placements,
                          "checkpoint_use_reentrant": bool(offload and args.candidate_reentrant),
                          "sublayer_checkpoint_names": list(model.sublayer_checkpoint_names),
                          "candidate_sublayer_checkpointing": bool(offload and args.candidate_sublayer_checkpointing),
                          "candidate_checkpoint_layernorms": bool(offload and getattr(args, "candidate_checkpoint_layernorms", False)),
                          "gemm_input_shapes": {name: sorted(shapes) for name, shapes in model.gemm_shapes.items()},
                          "norm_input_shapes": {name: sorted(shapes) for name, shapes in model.norm_shapes.items()},
                          "adam_steps": adam_steps,
                          "saved_tensor_offload": saved_tensor_receipt,
                          "backbone_forward_calls": model.backbone.forward_calls - call_start})
        if offload and step + 1 == args.roundtrip_step:
            path = args.output.parent / f"{args.output.stem}-{backend}-checkpoint-rank{rank}.pt"
            receipt = write_sharded_checkpoint(model, optimizer, path, completed_step=step + 1)
            del model, optimizer
            gc.collect()
            torch.cuda.empty_cache()
            # Start from zero parameters and empty Adam states so an ineffective restore cannot pass.
            blank = {name: torch.zeros_like(value) for name, value in initial.items()}
            model, optimizer = _build_arm(args, mesh, device, blank, backend, offload, patch_stack)
            restored_step = restore_sharded_checkpoint(model, optimizer, path, expected_step=step + 1)
            checks = {}
            for name, parameter in model.named_parameters():
                checks[f"{name}/parameters"] = compare_tensors(_full_cpu(parameter), after[name], strict=True)
                for key in ("exp_avg", "exp_avg_sq"):
                    checks[f"{name}/{key}"] = compare_tensors(_full_cpu(optimizer.state[parameter][key]),
                                                              moments[name][key], strict=True)
                checks[f"{name}/adam_step"] = {"passed": float(optimizer.state[parameter]["step"]) == adam_steps[name],
                                                 "actual": float(optimizer.state[parameter]["step"]),
                                                 "expected": adam_steps[name]}
            for check in checks.values():
                if "bitwise_equal" in check:
                    check["passed"] &= check["bitwise_equal"]
            receipt.update(restored_global_step=restored_step, checks=checks,
                           passed=all(row["passed"] for row in checks.values()))
            snapshots[-1]["checkpoint_roundtrip"] = receipt
        if rank == 0:
            print(json.dumps({"backend": backend, "offload": offload, "completed_step": step + 1}), flush=True)
    memory = {"max_allocated": torch.cuda.max_memory_allocated(device),
              "max_reserved": torch.cuda.max_memory_reserved(device)}
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return snapshots, memory


def _compare_steps(actual, expected, initial):
    records = []
    for candidate, reference in zip(actual, expected, strict=True):
        checks = {}
        for index, (micro, ref_micro) in enumerate(zip(candidate["micros"], reference["micros"], strict=True)):
            prefix = f"micro{index}"
            checks[f"{prefix}/loss"] = {"passed": bool(torch.isclose(micro["loss"], ref_micro["loss"],
                rtol=TOLERANCES["loss_rtol"], atol=TOLERANCES["loss_atol"])),
                "actual": float(micro["loss"]), "expected": float(ref_micro["loss"])}
            for key in ("dInputs", "dHidden", "dW_accumulated"):
                checks[f"{prefix}/{key}"] = compare_tensors(micro[key], ref_micro[key])
            masked_nonzero = int(torch.count_nonzero(micro["dHidden"].flatten(0, 1)[~micro["shifted_mask"]]))
            checks[f"{prefix}/message_mask"] = {"passed": masked_nonzero == 0,
                                                  "masked_hidden_gradient_nonzero": masked_nonzero}
            checks[f"{prefix}/targets"] = {"passed": bool(torch.equal(micro["labels"], ref_micro["labels"])
                and torch.equal(micro["shifted_mask"], ref_micro["shifted_mask"])
                and micro["normalizer"] == ref_micro["normalizer"])}
        for name in initial:
            for key in ("gradients", "parameters"):
                checks[f"{name}/{key}"] = compare_tensors(candidate[key][name], reference[key][name])
            checks[f"{name}/cumulative_update"] = compare_tensors(candidate["parameters"][name] - initial[name],
                                                                   reference["parameters"][name] - initial[name])
            checks[f"{name}/step_update"] = compare_tensors(candidate["parameters"][name] - candidate["before"][name],
                                                             reference["parameters"][name] - reference["before"][name])
            for key in ("exp_avg", "exp_avg_sq"):
                checks[f"{name}/{key}"] = compare_tensors(candidate["moments"][name][key], reference["moments"][name][key])
            checks[f"{name}/adam_step"] = {"passed": candidate["adam_steps"][name] == reference["adam_steps"][name]
                == candidate["step"], "candidate": candidate["adam_steps"][name],
                "reference": reference["adam_steps"][name]}
        checks["checkpoint_recomputation"] = {"passed": candidate["backbone_forward_calls"] >= 2 * len(candidate["micros"])
            and candidate["backbone_forward_calls"] == reference["backbone_forward_calls"],
            "candidate": candidate["backbone_forward_calls"], "reference": reference["backbone_forward_calls"]}
        placements = candidate["placements"]
        checks["cpu_fp32_storage"] = {"passed": all(row["param_device"] == "cpu" and row["grad_device"] == "cpu"
            and row["param_dtype"] == "torch.float32" and row["grad_dtype"] == "torch.float32"
            for row in placements.values()), "parameters": placements}
        receipt = candidate["saved_tensor_offload"]
        checks["saved_tensor_cpu_copies"] = {"passed": receipt["cuda_tensors"] > 0
            and receipt["copies_preserve_dtype_and_values"], **receipt}
        if candidate["checkpoint_use_reentrant"]:
            checks["reentrant_recompute_cpu_copies"] = {"passed": receipt["recompute_cuda_tensors"] > 0,
                "cuda_tensors": receipt["recompute_cuda_tensors"], "cuda_bytes": receipt["recompute_cuda_bytes"]}
        if candidate["candidate_sublayer_checkpointing"]:
            expected_sublayers = 8 if candidate.get("candidate_checkpoint_layernorms", False) else 4
            checks["actual_sublayer_scope"] = {"passed": len(candidate["sublayer_checkpoint_names"]) == expected_sublayers
                and not reference["sublayer_checkpoint_names"], "names": candidate["sublayer_checkpoint_names"]}
            checks["unchanged_gemm_input_shapes"] = {"passed": bool(candidate["gemm_input_shapes"])
                and candidate["gemm_input_shapes"] == reference["gemm_input_shapes"],
                "actual": candidate["gemm_input_shapes"], "reference": reference["gemm_input_shapes"]}
            if candidate.get("candidate_checkpoint_layernorms", False):
                checks["unchanged_norm_input_shapes"] = {"passed": bool(candidate["norm_input_shapes"])
                    and candidate["norm_input_shapes"] == reference["norm_input_shapes"],
                    "actual": candidate["norm_input_shapes"], "reference": reference["norm_input_shapes"]}
        if "checkpoint_roundtrip" in candidate:
            checks["checkpoint_roundtrip"] = candidate["checkpoint_roundtrip"]
        records.append({"step": candidate["step"], "global_supervised_tokens": candidate["global_supervised_tokens"],
                        "passed": all(check["passed"] for check in checks.values()), "checks": checks})
    return records


def replay_cpu_adam(initial: dict, gpu_steps: list) -> list:
    """Replay identical full GPU gradients on CPU to isolate FP32 Adam device arithmetic."""
    parameters = {name: torch.nn.Parameter(value.clone()) for name, value in initial.items()}
    optimizer = torch.optim.AdamW(parameters.values(), **ADAM_OPTIONS)
    records = []
    for reference in gpu_steps:
        before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        for name, parameter in parameters.items():
            parameter.grad = reference["gradients"][name].clone()
        optimizer.step()
        checks = {}
        for name, parameter in parameters.items():
            checks[f"{name}/parameters"] = compare_tensors(parameter, reference["parameters"][name], strict=True)
            checks[f"{name}/step_update"] = compare_tensors(parameter - before[name],
                reference["parameters"][name] - reference["before"][name], strict=True)
            for key in ("exp_avg", "exp_avg_sq"):
                checks[f"{name}/{key}"] = compare_tensors(optimizer.state[parameter][key],
                                                          reference["moments"][name][key], strict=True)
        records.append({"step": reference["step"], "passed": all(row["passed"] for row in checks.values()),
                        "checks": checks})
    return records


def main() -> None:
    """Run GPU baselines and CPU-offloaded candidates serially on exactly two GPUs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", choices=("native", "cce_exact"), nargs="+", default=["native", "cce_exact"])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--microbatches", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--roundtrip-step", type=int, default=1)
    parser.add_argument("--candidate-reentrant", action="store_true",
                        help="Only the CPU-offloaded candidate uses reentrant GC; GPU baseline remains non-reentrant")
    parser.add_argument("--candidate-sublayer-checkpointing", action="store_true",
                        help="Use two actual Qwen full-attention decoders; only the offload candidate adds inner sublayer checkpoints")
    parser.add_argument("--candidate-checkpoint-layernorms", action="store_true",
                        help="Also checkpoint both decoder norms; requires --candidate-sublayer-checkpointing")
    parser.add_argument("--save-tensors", action="store_true")
    args = parser.parse_args()
    if args.steps < 3 or args.microbatches < 2 or args.hidden_size < 2 or args.hidden_size % 2 or args.vocab_size % 2:
        parser.error("Require >=3 steps, >=2 microbatches, and positive even hidden/vocabulary dimensions")
    if not 1 <= args.roundtrip_step < args.steps:
        parser.error("The checkpoint roundtrip must leave at least one subsequent optimizer step")
    if args.candidate_reentrant and args.candidate_sublayer_checkpointing:
        parser.error("Compare reentrant and sublayer checkpointing as separate candidates")
    if args.candidate_checkpoint_layernorms and not args.candidate_sublayer_checkpointing:
        parser.error("Norm checkpointing requires --candidate-sublayer-checkpointing")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    # Match verl: offloaded DTensor norms/state reads require a CPU collective backend.
    dist.init_process_group("cpu:gloo,cuda:nccl", timeout=timedelta(minutes=5))
    try:
        if dist.get_world_size() != 2:
            raise ValueError("This fixture requires exactly two DP GPUs")
        mesh = init_device_mesh("cuda", (2,))
        initial = _initial_state(args)
        report = {"scope": "nested_FSDP2_same_kernel_CPU_offload_nonreentrant_GC_saved_tensors_CPU",
                  "world_size": 2, "steps": args.steps, "microbatches": args.microbatches,
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
                  "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "checkpointing_source_sha256": hashlib.sha256(
                      (Path(__file__).resolve().parents[1] / "verl" / "checkpointing.py").read_bytes()
                  ).hexdigest(),
                  "checkpoint_scope": "torch_sharded_state_dict_not_native_trainer_checkpoint",
                  "native_trainer_and_stateful_loader_resume_validated": False,
                  "checkpoint_after_step": args.roundtrip_step,
                  "gpu_baseline_checkpoint_use_reentrant": False,
                  "offload_candidate_checkpoint_use_reentrant": args.candidate_reentrant,
                  "candidate_reentrant": args.candidate_reentrant, "reference_reentrant": False,
                  "candidate_sublayer_checkpointing": args.candidate_sublayer_checkpointing,
                  "candidate_checkpoint_layernorms": args.candidate_checkpoint_layernorms,
                  "reference_checkpoint_layernorms": False,
                  "reference_sublayer_checkpointing": False, "initial_seed": 729,
                  "backbone_architecture": "actual_Qwen3_5DecoderLayer_2_full_attention_SDPA" if args.candidate_sublayer_checkpointing
                      else "two_linear_silu_backbone",
                  "linear_attention_packing_or_sequence_parallel_validated": False,
                  "tolerances": TOLERANCES, "adam": ADAM_OPTIONS, "backends": []}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for backend in args.backends:
            reference, ref_memory = _run_arm(args, mesh, device, initial, backend, offload=False)
            candidate, memory = _run_arm(args, mesh, device, initial, backend, offload=True)
            comparisons = _compare_steps(candidate, reference, initial)
            adam_replay = replay_cpu_adam(initial, reference)
            result = {"backend": backend, "rank": dist.get_rank(), "steps": comparisons,
                      "strict_fp32_cpu_adam_replay_of_gpu_gradients": adam_replay,
                      "gpu_memory": ref_memory, "offload_memory": memory,
                      "passed": all(row["passed"] for row in comparisons + adam_replay)}
            if args.save_tensors:
                path = args.output.parent / f"{args.output.stem}-{backend}-rank{dist.get_rank()}.pt"
                torch.save({"initial": initial, "gpu_reference": reference, "offload_candidate": candidate}, path)
                result["full_tensor_snapshot"] = str(path)
            gathered = [None, None]
            dist.all_gather_object(gathered, result)
            report["backends"].extend(gathered)
            report["passed"] = all(row["passed"] for row in report["backends"])
            if dist.get_rank() == 0:
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({"backend": backend, "passed_all_ranks": all(row["passed"] for row in gathered)}), flush=True)
            del reference, candidate
        if not report["passed"]:
            raise SystemExit(1)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
