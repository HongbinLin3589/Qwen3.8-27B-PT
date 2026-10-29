"""Check the opt-in FSDP2 prefetch policy without allocating CUDA tensors."""

from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch.distributed.fsdp as fsdp

from legacy.sft.entrypoint import _configure_fsdp2_backward_prefetch


class _ShardedModule:
    def __init__(self):
        self.requests = []

    def set_modules_to_backward_prefetch(self, modules):
        self.requests.append(modules)


def _config(policy="default", strategy="fsdp2"):
    return OmegaConf.create({"trainer": {"fsdp2_backward_prefetch": policy},
                             "engine": {"strategy": strategy}})


def test_default_prefetch_leaves_modules_untouched(monkeypatch):
    """Keep existing runs on Torch's default policy."""
    module = _ShardedModule()
    monkeypatch.setattr(fsdp, "FSDPModule", _ShardedModule)
    model = SimpleNamespace(named_modules=lambda: iter([("", module)]))
    assert _configure_fsdp2_backward_prefetch(_config(), model) == []
    assert module.requests == []


def test_current_prefetch_targets_every_sharded_unit(monkeypatch):
    """Use the current unit instead of the empty list that restores default prefetch."""
    first, second = _ShardedModule(), _ShardedModule()
    monkeypatch.setattr(fsdp, "FSDPModule", _ShardedModule)
    model = SimpleNamespace(named_modules=lambda: iter([
        ("", first), ("plain", object()), ("layer", second),
    ]))
    assert _configure_fsdp2_backward_prefetch(_config("current"), model) == ["", "layer"]
    assert first.requests == [[first]]
    assert second.requests == [[second]]


@pytest.mark.parametrize("policy,strategy", [("off", "fsdp2"), ("current", "fsdp")])
def test_prefetch_rejects_unsupported_configuration(policy, strategy):
    """Reject ambiguous modes and the FSDP1 implementation."""
    with pytest.raises(ValueError, match="fsdp2_backward_prefetch"):
        _configure_fsdp2_backward_prefetch(_config(policy, strategy), object())


def test_current_prefetch_requires_sharded_modules():
    """Fail when the policy cannot affect any initialized FSDP2 unit."""
    model = SimpleNamespace(named_modules=lambda: iter([]))
    with pytest.raises(ValueError, match="initialized FSDP2"):
        _configure_fsdp2_backward_prefetch(_config("current"), model)
