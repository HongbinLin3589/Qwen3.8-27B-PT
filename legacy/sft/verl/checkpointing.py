"""Scope nested non-reentrant checkpoints to whole Qwen3.5 text sublayers."""

from contextlib import contextmanager
from functools import wraps
from types import MethodType
from typing import Iterator

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.checkpoint import checkpoint


def _sublayer_forward(native_forward):
    @wraps(native_forward)
    def forward(module, *args, **kwargs):
        if not module.training or not torch.is_grad_enabled():
            return native_forward(*args, **kwargs)
        if any(kwargs.get(key) is not None for key in ("cache_params", "past_key_values", "past_key_value")):
            raise ValueError("Checkpointed Qwen3.5 training requires disabled mutable attention caches")
        # Checkpoint saves only positional Tensor leaves; keyword/nested tensors would
        # otherwise stay in its Python closure and bypass the parent's CPU offload.
        flat_inputs, input_spec = tree_flatten((args, kwargs))

        def run(*flat_values):
            positional, keyword = tree_unflatten(flat_values, input_spec)
            return native_forward(*positional, **keyword)

        return checkpoint(run, *flat_inputs, use_reentrant=False)

    forward._qwenpt_sublayer_checkpointed = True
    return forward


@contextmanager
def qwen3_5_sublayer_checkpointing(
    model: torch.nn.Module, enabled: bool = False, include_layernorms: bool = False,
) -> Iterator[list[str]]:
    """Temporarily checkpoint each actual text decoder's attention and whole MLP.

    Args:
        model: Initialized Qwen3.5 modules, optionally sharded at decoder boundaries.
        enabled: False preserves every module attribute and training operation.
        include_layernorms: Also checkpoint both decoder norms without changing their shapes.

    Yields:
        Actual patched module names for run receipts; evaluation remains native.
    """
    if not enabled:
        yield []
        return
    from torch.distributed.fsdp import FSDPModule
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    targets = []
    for name, decoder in model.named_modules():
        if not isinstance(decoder, Qwen3_5DecoderLayer):
            continue
        attention = "linear_attn" if decoder.layer_type == "linear_attention" else "self_attn"
        child_names = ("input_layernorm", attention, "post_attention_layernorm", "mlp") if include_layernorms else (attention, "mlp")
        for child_name in child_names:
            child = getattr(decoder, child_name)
            function = getattr(child.forward, "__func__", child.forward)
            if isinstance(child, FSDPModule):
                raise ValueError("Sublayer checkpointing requires FSDP boundaries around decoders, not their children")
            if getattr(function, "_qwenpt_sublayer_checkpointed", False):
                raise ValueError("Qwen3.5 sublayer checkpointing is already enabled")
            if getattr(function, "_qwenpt_mlp_chunk_tokens", 0) or function.__module__ == "legacy.sft.verl.mlp":
                raise ValueError("Whole-sublayer checkpointing cannot be mixed with MLP token chunking")
            targets.append((f"{name}.{child_name}".lstrip("."), child))
    if not targets:
        raise ValueError("Sublayer checkpointing requires actual Qwen3_5DecoderLayer text modules")
    originals = []
    try:
        for _, module in targets:
            native_forward = module.forward
            originals.append((module, "forward" in module.__dict__, module.__dict__.get("forward")))
            module.forward = MethodType(_sublayer_forward(native_forward), module)
        yield [name for name, _ in targets]
    finally:
        for module, had_forward, original in reversed(originals):
            if had_forward:
                module.forward = original
            else:
                del module.forward
