"""Scope blocking CPU migration to verl's FSDP2 full-state initialization."""

from contextlib import contextmanager
from functools import wraps
from types import MethodType
from typing import Iterator

import torch


@contextmanager
def _blocking_cpu_transfers(model: torch.nn.Module) -> Iterator[None]:
    native_to = model.to
    had_instance_to = "to" in model.__dict__
    instance_to = model.__dict__.get("to")

    @wraps(native_to)
    def to_cpu_blocking(_model, *args, **kwargs):
        # Use Module.to's own parser so Tensor/dtype/positional overloads keep
        # their meaning while changing only an explicit asynchronous CPU move.
        device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None and device.type == "cpu" and non_blocking:
            options = {"device": device, "dtype": dtype, "non_blocking": False}
            if memory_format is not None:
                options["memory_format"] = memory_format
            return native_to(**options)
        return native_to(*args, **kwargs)

    model.to = MethodType(to_cpu_blocking, model)
    try:
        yield
    finally:
        if had_instance_to:
            model.to = instance_to
        else:
            del model.to


@contextmanager
def fsdp2_cpu_load_blocking(enabled: bool) -> Iterator[None]:
    """Preserve native DCP loading while waiting for its final CPU migration.

    FSDP2 immediately repads uneven local shards in ``Module._apply``.
    An asynchronous device-to-CPU move can leave that CPU repadding reading
    unfinished data, so the loader's root-module CPU transfer must finish first.

    Args:
        enabled: Apply this correctness fix while constructing a CPU-offloaded
            FSDP2 engine; False leaves every method and import untouched.

    Yields:
        None. The engine import and the individual model's ``to`` attribute are
        restored on every exit; no vendor file or global Tensor method changes.
    """
    if not enabled:
        yield
        return
    from verl.workers.engine.fsdp import transformer_impl

    native_load = transformer_impl.fsdp2_load_full_state_dict

    @wraps(native_load)
    def load(model, full_state, device_mesh=None, cpu_offload=None):
        # Match native verl: CPU offload is selected by the policy's presence.
        if cpu_offload is None:
            return native_load(model, full_state, device_mesh=device_mesh, cpu_offload=cpu_offload)
        with _blocking_cpu_transfers(model):
            return native_load(model, full_state, device_mesh=device_mesh, cpu_offload=cpu_offload)

    transformer_impl.fsdp2_load_full_state_dict = load
    try:
        yield
    finally:
        transformer_impl.fsdp2_load_full_state_dict = native_load
