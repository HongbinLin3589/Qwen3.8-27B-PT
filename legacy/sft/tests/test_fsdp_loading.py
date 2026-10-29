"""Check the loader correction stays scoped to explicit CPU migrations."""

import sys
from types import MethodType, ModuleType

import pytest
import torch

from legacy.sft.verl.fsdp_loading import fsdp2_cpu_load_blocking


class _TrackingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def to(self, *args, **kwargs):
        self.calls.append((args, kwargs, torch._C._nn._parse_to(*args, **kwargs)))
        return self


def _install_engine(monkeypatch, loader):
    names = ["verl", "verl.workers", "verl.workers.engine", "verl.workers.engine.fsdp"]
    for name in names:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, attribute = name.rsplit(".", 1)
            setattr(sys.modules[parent], attribute, module)
    engine = ModuleType(f"{names[-1]}.transformer_impl")
    engine.fsdp2_load_full_state_dict = loader
    monkeypatch.setitem(sys.modules, engine.__name__, engine)
    sys.modules[names[-1]].transformer_impl = engine
    return engine


def test_disabled_context_does_not_import_or_patch(monkeypatch):
    """Other strategies must not require the optional engine import."""
    monkeypatch.setitem(sys.modules, "verl.workers.engine.fsdp", None)
    with fsdp2_cpu_load_blocking(False):
        pass


@pytest.mark.parametrize("policy", [object(), False])
def test_only_offloaded_loader_root_cpu_transfer_blocks(monkeypatch, policy):
    """Retain native DCP arguments and CUDA work while waiting for final D2H."""
    model, unrelated = _TrackingModule(), _TrackingModule()
    state, mesh, result = object(), object(), object()
    seen = []

    def native_load(model, full_state, device_mesh=None, cpu_offload=None):
        seen.append((model, full_state, device_mesh, cpu_offload))
        model.to(device="cuda:0", non_blocking=True)
        unrelated.to("cpu", non_blocking=True)
        model.to("cpu", non_blocking=True)
        return result

    engine = _install_engine(monkeypatch, native_load)
    module_to, tensor_to = torch.nn.Module.to, torch.Tensor.to
    with fsdp2_cpu_load_blocking(True):
        assert engine.fsdp2_load_full_state_dict is not native_load
        assert engine.fsdp2_load_full_state_dict(model, state, mesh, policy) is result
        assert "to" not in model.__dict__
    assert engine.fsdp2_load_full_state_dict is native_load
    assert seen == [(model, state, mesh, policy)]
    assert model.calls[0][:2] == ((), {"device": "cuda:0", "non_blocking": True})
    assert model.calls[0][2][2] is True
    assert model.calls[1][2][0] == torch.device("cpu")
    assert model.calls[1][2][2] is False
    assert unrelated.calls[0][2][2] is True
    assert torch.nn.Module.to is module_to and torch.Tensor.to is tensor_to


def test_nonoffloaded_load_never_installs_instance_method(monkeypatch):
    """The native policy uses presence rather than truthiness to select offload."""
    model = _TrackingModule()

    def native_load(model, full_state, device_mesh=None, cpu_offload=None):
        assert "to" not in model.__dict__
        model.to("cpu", non_blocking=True)

    engine = _install_engine(monkeypatch, native_load)
    with fsdp2_cpu_load_blocking(True):
        engine.fsdp2_load_full_state_dict(model, {})
    assert model.calls[0][2][2] is True


def test_real_cpu_module_preserves_values_and_dtype(monkeypatch):
    """The rewritten overload must remain a valid native Module.to call."""
    model = torch.nn.Linear(3, 2, bias=False)
    expected = model.weight.detach().to(torch.bfloat16)

    def native_load(model, full_state, device_mesh=None, cpu_offload=None):
        return model.to("cpu", torch.bfloat16, True)

    engine = _install_engine(monkeypatch, native_load)
    with fsdp2_cpu_load_blocking(True):
        assert engine.fsdp2_load_full_state_dict(model, {}, cpu_offload=object()) is model
    torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)
    assert "to" not in model.__dict__


@pytest.mark.parametrize("overload", ["device", "positional", "tensor", "memory_format"])
def test_cpu_transfer_preserves_module_to_overloads(monkeypatch, overload):
    """Use the same argument parser as Module.to to retain dtype and layout."""
    model = _TrackingModule()
    options = {
        "device": ((torch.device("cpu"),), {"non_blocking": True}),
        "positional": (("cpu", torch.bfloat16, True), {}),
        "tensor": ((torch.empty(1, dtype=torch.float64),), {"non_blocking": True}),
        "memory_format": (("cpu",), {"non_blocking": True, "memory_format": torch.channels_last}),
    }
    args, kwargs = options[overload]
    expected = torch._C._nn._parse_to(*args, **kwargs)

    def native_load(model, full_state, device_mesh=None, cpu_offload=None):
        model.to(*args, **kwargs)
        # Dtype-only and already blocking transfers must retain their raw call.
        model.to(dtype=torch.float64, non_blocking=True)
        model.to("cpu", non_blocking=False)

    engine = _install_engine(monkeypatch, native_load)
    with fsdp2_cpu_load_blocking(True):
        engine.fsdp2_load_full_state_dict(model, {}, cpu_offload=object())
    actual = model.calls[0][2]
    assert actual == (expected[0], expected[1], False, expected[3])
    assert model.calls[1][:2] == ((), {"dtype": torch.float64, "non_blocking": True})
    assert model.calls[2][:2] == (("cpu",), {"non_blocking": False})


@pytest.mark.parametrize("instance_override", [False, True])
def test_exception_restores_exact_engine_and_instance_state(monkeypatch, instance_override):
    """An initialization failure must not leak the loader or model wrapper."""
    model = _TrackingModule()
    if instance_override:
        def custom_to(self, *args, **kwargs):
            return _TrackingModule.to(self, *args, **kwargs)

        model.to = MethodType(custom_to, model)
    original = model.__dict__.get("to")

    def native_load(model, full_state, device_mesh=None, cpu_offload=None):
        model.to("cpu", non_blocking=True)
        raise RuntimeError("native initialization failure")

    engine = _install_engine(monkeypatch, native_load)
    with pytest.raises(RuntimeError, match="native initialization failure"):
        with fsdp2_cpu_load_blocking(True):
            engine.fsdp2_load_full_state_dict(model, {}, cpu_offload=object())
    assert engine.fsdp2_load_full_state_dict is native_load
    assert ("to" in model.__dict__) is instance_override
    assert model.__dict__.get("to") is original
    assert model.calls[0][2][2] is False
