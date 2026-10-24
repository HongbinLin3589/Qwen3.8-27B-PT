"""Exercise native DTensor reconstruction and exact PEFT export without a GPU."""

import json

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from safetensors.torch import load_file

from qwenpt.eval.export_adapter import adapter_weights, export_adapter, merge_base, merge_shards


def dtensor(tensor, rank):
    """Build a serialized-style FSDP shard without initializing any process group."""
    mesh = DeviceMesh("cpu", [0, 1], mesh_dim_names=("fsdp",), _init_backend=False, _rank=rank)
    spec = DTensorSpec(mesh, (Shard(0),), TensorMeta(tensor.shape, tensor.stride(), tensor.dtype))
    return DTensor(tensor.chunk(2)[rank].clone(), spec, requires_grad=False)


@pytest.fixture
def checkpoint(tmp_path):
    """Write two real FSDP DTensor shards with text and vision LoRA pairs."""
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "fsdp_config.json").write_text(json.dumps({"FSDP_version": 2, "world_size": 2}))
    (root / "lora_train_meta.json").write_text(json.dumps({"r": 2, "lora_alpha": 4, "task_type": "CAUSAL_LM"}))
    state = {}
    for module in ("model.language_model.layers.0.self_attn.q_proj", "model.visual.blocks.0.attn.qkv"):
        state[f"base_model.model.{module}.lora_A.default.weight"] = torch.arange(8).reshape(2, 4).to(torch.bfloat16)
        state[f"base_model.model.{module}.lora_B.default.weight"] = torch.arange(12).reshape(6, 2).to(torch.bfloat16)
    for rank in range(2):
        torch.save({name: dtensor(tensor, rank) for name, tensor in state.items()},
                   root / f"model_world_size_2_rank_{rank}.pt")
    return root, state


def test_exact_native_dtensor_export_and_source_preservation(checkpoint, tmp_path):
    """Require an exact PEFT roundtrip, untouched source files and no overwrite."""
    root, state = checkpoint
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    output = tmp_path / "adapter"
    report = export_adapter(root, output, "/model/base")
    restored = load_file(output / "adapter_model.safetensors")
    for name, tensor in state.items():
        assert torch.equal(restored[name.replace(".default.weight", ".weight")], tensor)
    assert report["tensor_count"] == 4
    assert report["parameter_count"] == 40
    assert report["excluded_visual_keys"] == []
    config = json.loads((output / "adapter_config.json").read_text())
    assert (config["r"], config["lora_alpha"]) == (2, 4)
    assert config["target_modules"] == ["q_proj", "qkv"]
    assert before == {path.name: path.read_bytes() for path in root.iterdir()}
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        export_adapter(root, output, "/model/base")


def test_text_only_export_records_every_visual_exclusion(checkpoint, tmp_path):
    """Require explicit text selection to enumerate every omitted visual tensor."""
    root, _ = checkpoint
    output = tmp_path / "text_adapter"
    report = export_adapter(root, output, "/model/base", text_only=True)
    assert report["tensor_count"] == 2
    assert len(report["excluded_visual_keys"]) == 2
    assert all(".visual." in name for name in report["excluded_visual_keys"])
    assert all(".visual." not in name for name in load_file(output / "adapter_model.safetensors"))


def test_native_merge_rejects_rank_key_mismatch_and_nonfinite():
    """Reject missing parameters, nonfinite values and inconsistent replicas."""
    with pytest.raises(ValueError, match="keys differ"):
        merge_shards([{"a": torch.ones(2)}, {"b": torch.ones(2)}])
    with pytest.raises(ValueError, match="Nonfinite"):
        merge_shards([{"a": torch.tensor([float("nan")])}])
    with pytest.raises(ValueError, match="Replicated tensors differ"):
        merge_shards([{"a": torch.ones(2)}, {"a": torch.zeros(2)}])


def test_native_merge_rejects_incomplete_dtensor_shape():
    """Reject local DTensor storage that disagrees with its global shape."""
    value = torch.arange(8).reshape(2, 4).float()
    valid = dtensor(value, 0)
    invalid = DTensor(value, valid._spec, requires_grad=False)
    with pytest.raises(ValueError, match="Reconstructed tensor shape differs"):
        merge_shards([{"a": valid}, {"a": invalid}])


def test_lora_pairs_and_rank_must_match():
    """Reject missing A/B partners and adapter ranks that contradict metadata."""
    key = "base_model.model.layer.q_proj.lora_A.default.weight"
    with pytest.raises(ValueError, match="incomplete"):
        adapter_weights({key: torch.ones(2, 4)}, 2)
    with pytest.raises(ValueError, match="rank differs"):
        adapter_weights({key: torch.ones(2, 4)}, 8)


def test_sharded_merge_updates_all_pairs_and_preserves_unadapted_weights(checkpoint, tmp_path):
    """Verify text and visual merges, base preservation and completion markers."""
    from safetensors.torch import save_file

    root, state = checkpoint
    adapter = tmp_path / "adapter"
    export_adapter(root, adapter, "/model/base")
    base = tmp_path / "base"
    base.mkdir()
    tensors = {name.split("base_model.model.", 1)[1].split(".lora_A.", 1)[0] + ".weight":
               torch.arange(24).reshape(6, 4).to(torch.bfloat16)
               for name in state if ".lora_A." in name}
    tensors["model.embed_tokens.weight"] = torch.ones(3, 4, dtype=torch.bfloat16)
    filenames = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    weight_map = {name: filenames[index % 2] for index, name in enumerate(tensors)}
    for filename in filenames:
        save_file({name: tensor for name, tensor in tensors.items() if weight_map[name] == filename}, base / filename)
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (base / "config.json").write_text('{"model_type":"test"}')
    merged = tmp_path / "merged"
    report = merge_base(adapter, base, merged)
    assert report["base_tensor_count"] == 3
    assert report["modified_tensor_count"] == 2
    assert json.loads((merged / "manifest.complete").read_text())["merge_manifest_sha256"]
    for name, tensor in tensors.items():
        restored = load_file(merged / weight_map[name])[name]
        if "embed_tokens" in name:
            assert torch.equal(restored, tensor)
        else:
            prefix = "base_model.model." + name.removesuffix(".weight")
            delta = state[prefix + ".lora_B.default.weight"].float() @ state[prefix + ".lora_A.default.weight"].float() * 2
            assert torch.equal(restored, (tensor.float() + delta).to(tensor.dtype))
        assert torch.equal(load_file(base / weight_map[name])[name], tensor)
    assert (merged / "config.json").read_bytes() == (base / "config.json").read_bytes()
