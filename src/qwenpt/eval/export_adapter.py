"""Export trusted FSDP2 LoRA shards and merge PEFT adapters into HF models on CPU."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import torch
from torch.distributed.tensor import DTensor
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def merge_shards(shards: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Reconstruct one-dimensional FSDP shards without starting a process group."""
    if not shards or not shards[0]:
        raise ValueError("Checkpoint has no adapter tensors")
    keys = set(shards[0])
    if any(set(shard) != keys for shard in shards):
        raise ValueError("Model shard keys differ across ranks")
    merged = {}
    for name in sorted(keys):
        values = [shard[name] for shard in shards]
        first = values[0]
        if isinstance(first, DTensor):
            placements = first.placements
            mesh = first.device_mesh.mesh
            if mesh.ndim != 1 or sorted(mesh.tolist()) != list(range(len(shards))) or len(placements) != 1:
                raise ValueError(f"Unsupported FSDP mesh for {name}")
            for value in values:
                if (not isinstance(value, DTensor) or value.placements != placements
                        or value.shape != first.shape or value.dtype != first.dtype
                        or not torch.equal(value.device_mesh.mesh, mesh)):
                    raise ValueError(f"Inconsistent DTensor metadata for {name}")
            local = [values[rank].to_local().detach().cpu() for rank in mesh.tolist()]
            placement = placements[0]
            if placement.is_shard():
                tensor = torch.cat(local, dim=placement.dim)
            elif placement.is_replicate():
                tensor = local[0]
                if any(not torch.equal(tensor, other) for other in local[1:]):
                    raise ValueError(f"Replicated tensors differ for {name}")
            else:
                raise ValueError(f"Unsupported DTensor placement for {name}: {placement}")
            if tensor.shape != first.shape:
                raise ValueError(f"Reconstructed tensor shape differs for {name}")
        else:
            if not all(isinstance(value, torch.Tensor) and not isinstance(value, DTensor) for value in values):
                raise ValueError(f"Non-tensor or inconsistent shard type for {name}")
            tensor = first.detach().cpu()
            if any(value.dtype != first.dtype or not torch.equal(tensor, value.detach().cpu())
                   for value in values[1:]):
                raise ValueError(f"Replicated tensors differ for {name}")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Nonfinite adapter tensor: {name}")
        merged[name] = tensor.contiguous()
    return merged


def adapter_weights(state: dict[str, torch.Tensor], rank: int) -> dict[str, torch.Tensor]:
    """Validate complete LoRA A/B pairs and remove PEFT's internal adapter name."""
    pairs, result = {}, {}
    for name, tensor in state.items():
        match = re.fullmatch(r"(base_model\.model\..+)\.lora_([AB])\.default\.weight", name)
        if match is None or tensor.ndim != 2:
            raise ValueError(f"Unsupported adapter parameter: {name}")
        module, side = match.groups()
        if tensor.shape[0 if side == "A" else 1] != rank:
            raise ValueError(f"LoRA rank differs from metadata for {name}")
        pairs.setdefault(module, {})[side] = tensor
        result[name.replace(".default.weight", ".weight")] = tensor
    if any(set(pair) != {"A", "B"} for pair in pairs.values()):
        raise ValueError("Checkpoint has incomplete LoRA A/B pairs")
    return result


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_adapter(checkpoint: Path, output: Path, base_model: str, *, text_only: bool = False) -> dict:
    """Export a trusted checkpoint, verify exact tensor roundtrip, and return its audit."""
    checkpoint, output = checkpoint.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite adapter output: {output}")
    fsdp_path = checkpoint / "fsdp_config.json"
    metadata_path = checkpoint / "lora_train_meta.json"
    fsdp = json.loads(fsdp_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    world_size = fsdp.get("world_size")
    if fsdp.get("FSDP_version") != 2 or type(world_size) is not int or world_size <= 0:
        raise ValueError("Expected a native FSDP2 checkpoint with positive world size")
    rank, alpha = metadata.get("r"), metadata.get("lora_alpha")
    if type(rank) is not int or rank <= 0 or not isinstance(alpha, (int, float)) or alpha <= 0:
        raise ValueError("Missing positive LoRA rank/alpha metadata")
    if metadata.get("task_type") != "CAUSAL_LM":
        raise ValueError("Expected a CAUSAL_LM adapter")
    paths = [checkpoint / f"model_world_size_{world_size}_rank_{index}.pt" for index in range(world_size)]
    # Native shards contain DTensor/DeviceMesh Python objects, so only load trusted local training artifacts.
    shards = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    state = adapter_weights(merge_shards(shards), rank)
    del shards
    excluded = sorted(name for name in state if text_only and ".visual." in name)
    weights = {name: tensor for name, tensor in state.items() if name not in excluded}
    if not weights:
        raise ValueError("No adapter tensors remain after selecting text parameters")
    config = {
        "base_model_name_or_path": base_model, "bias": "none", "fan_in_fan_out": False,
        "inference_mode": True, "lora_alpha": alpha, "lora_dropout": 0.0,
        "peft_type": "LORA", "r": rank, "task_type": "CAUSAL_LM",
        "target_modules": sorted({name.split(".")[-3] for name in weights}),
        "use_dora": False, "use_rslora": False,
    }
    report = {
        "checkpoint": str(checkpoint), "base_model": base_model, "world_size": world_size,
        "lora_metadata": metadata, "text_only": text_only, "excluded_visual_keys": excluded,
        "source_sha256": {path.name: _sha256(path) for path in [fsdp_path, metadata_path, *paths]},
        "tensors": {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                    for name, tensor in weights.items()},
        "tensor_count": len(weights), "parameter_count": sum(tensor.numel() for tensor in weights.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        model_path = staging / "adapter_model.safetensors"
        save_file(weights, model_path, metadata={"format": "pt"})
        restored = load_file(model_path)
        if (set(restored) != set(weights)
                or any(restored[name].dtype != tensor.dtype or not torch.equal(restored[name], tensor)
                       for name, tensor in weights.items())):
            raise ValueError("Exported adapter failed exact tensor roundtrip")
        config_path = staging / "adapter_config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        report["output_sha256"] = {path.name: _sha256(path) for path in (model_path, config_path)}
        (staging / "export_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(output)
    return report


def merge_base(adapter: Path, base_model: Path, output: Path) -> dict:
    """Merge all adapter pairs into a new sharded HF model with bounded CPU memory."""
    adapter, base_model, output = adapter.resolve(), base_model.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite merged model output: {output}")
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text())
    if (config.get("peft_type") != "LORA" or config.get("fan_in_fan_out")
            or config.get("use_dora") or config.get("use_rslora") or config.get("bias") != "none"):
        raise ValueError("Only ordinary linear LoRA adapters with bias=none are supported")
    rank, alpha = config["r"], config["lora_alpha"]
    if type(rank) is not int or rank <= 0 or not isinstance(alpha, (int, float)) or alpha <= 0:
        raise ValueError("Missing positive LoRA rank/alpha")
    weights_path = adapter / "adapter_model.safetensors"
    weights = load_file(weights_path)
    pairs = {}
    for name, tensor in weights.items():
        match = re.fullmatch(r"base_model\.model\.(.+)\.lora_([AB])\.weight", name)
        if match is None or tensor.ndim != 2:
            raise ValueError(f"Unsupported adapter parameter: {name}")
        module, side = match.groups()
        if tensor.shape[0 if side == "A" else 1] != rank or not torch.isfinite(tensor).all().item():
            raise ValueError(f"Invalid adapter tensor: {name}")
        pairs.setdefault(module + ".weight", {})[side] = tensor
    if not pairs or any(set(pair) != {"A", "B"} for pair in pairs.values()):
        raise ValueError("Adapter has incomplete LoRA A/B pairs")
    index_path = base_model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    if set(pairs) - set(weight_map):
        raise ValueError(f"Adapter targets absent from base model: {sorted(set(pairs) - set(weight_map))}")
    report = {
        "adapter": str(adapter), "base_model": str(base_model), "lora_rank": rank, "lora_alpha": alpha,
        "adapter_sha256": _sha256(weights_path), "adapter_config_sha256": _sha256(config_path),
        "base_index_sha256": _sha256(index_path), "merge_dtype": "float32", "shards": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        processed, updated_count = set(), 0
        for shard_name in sorted(set(weight_map.values())):
            source = base_model / shard_name
            state = load_file(source)
            expected = {name for name, filename in weight_map.items() if filename == shard_name}
            if set(state) != expected or processed.intersection(state):
                raise ValueError(f"Base shard keys disagree with index: {shard_name}")
            modified = []
            for name in sorted(expected.intersection(pairs)):
                pair, base = pairs[name], state[name]
                if base.ndim != 2 or (pair["B"].shape[0], pair["A"].shape[1]) != tuple(base.shape):
                    raise ValueError(f"LoRA/base shape mismatch: {name}")
                # FP32 accumulation avoids rounding a small learned delta before adding it to the BF16 base.
                value = base.float().addmm(pair["B"].float(), pair["A"].float(), alpha=alpha / rank)
                if not torch.isfinite(value).all().item():
                    raise ValueError(f"Nonfinite merged weight: {name}")
                state[name] = value.to(base.dtype).contiguous()
                modified.append(name)
            destination = staging / shard_name
            save_file(state, destination, metadata={"format": "pt"})
            with safe_open(destination, framework="pt", device="cpu") as restored:
                if set(restored.keys()) != expected:
                    raise ValueError(f"Merged shard keys disagree with index: {shard_name}")
                for name in expected:
                    if list(restored.get_slice(name).get_shape()) != list(state[name].shape):
                        raise ValueError(f"Merged shard shape roundtrip failed: {name}")
                for name in modified:
                    if not torch.equal(restored.get_tensor(name), state[name]):
                        raise ValueError(f"Merged tensor exact roundtrip failed: {name}")
            report["shards"][shard_name] = {
                "bytes": destination.stat().st_size, "sha256": _sha256(destination),
                "tensor_count": len(state), "modified_keys": modified,
            }
            processed.update(state)
            updated_count += len(modified)
            del state
            print(json.dumps({"merged_shard": shard_name, "updated_tensors": len(modified)}), flush=True)
        if processed != set(weight_map) or updated_count != len(pairs):
            raise ValueError("Merged model does not cover every base key and adapter pair")
        for path in base_model.iterdir():
            if path.is_file() and path.suffix != ".safetensors" and path.name != "crc32.txt":
                shutil.copyfile(path, staging / path.name)
        report.update({"base_tensor_count": len(processed), "modified_tensor_count": updated_count})
        manifest = staging / "merge_manifest.json"
        manifest.write_text(json.dumps(report, indent=2) + "\n")
        (staging / "manifest.complete").write_text(json.dumps({"merge_manifest_sha256": _sha256(manifest)}) + "\n")
        staging.rename(output)
    return report


def main() -> None:
    """Export native LoRA shards or merge an adapter into a new sharded HF model."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Export trusted native FSDP2 LoRA shards")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--base-model", required=True)
    export.add_argument("--text-only", action="store_true", help="Exclude visual adapters for text evaluation; record all exclusions")
    merge = commands.add_parser("merge", help="Merge a complete PEFT adapter into each base-model shard")
    merge.add_argument("--adapter", type=Path, required=True)
    merge.add_argument("--base-model", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    if args.command == "export":
        report = export_adapter(args.checkpoint, args.output, args.base_model, text_only=args.text_only)
        print(json.dumps({key: report[key] for key in ("checkpoint", "tensor_count", "parameter_count", "text_only")}))
    else:
        report = merge_base(args.adapter, args.base_model, args.output)
        print(json.dumps({key: report[key] for key in ("base_tensor_count", "modified_tensor_count")}))


if __name__ == "__main__":
    main()
