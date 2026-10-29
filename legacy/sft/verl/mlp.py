"""Scope token chunking and recomputation to Qwen3.5 text MLP instances."""

from contextlib import contextmanager
from operator import index
from types import MethodType
from typing import Iterator

import torch
from torch.utils.checkpoint import checkpoint


def _chunked_forward(native_forward, chunk_tokens: int):
    """Keep each gate/up activation local to one recomputed token chunk."""
    def forward(_module, x):
        shape = x.shape
        flattened = x.reshape(-1, shape[-1])
        outputs = []
        for chunk in flattened.split(chunk_tokens, dim=0):
            if torch.is_grad_enabled():
                # Qwen3_5MLP has no stochastic layers; preserve the outer HF GC mode.
                output = checkpoint(native_forward, chunk, use_reentrant=False, preserve_rng_state=False)
            else:
                output = native_forward(chunk)
            outputs.append(output)
        return torch.cat(outputs, dim=0).reshape(shape)

    forward._qwenpt_mlp_chunk_tokens = chunk_tokens
    return forward


@contextmanager
def qwen3_5_mlp_token_chunking(model: torch.nn.Module, chunk_tokens: int = 0) -> Iterator[list[str]]:
    """Temporarily checkpoint token chunks inside initialized Qwen3.5 text MLPs.

    Args:
        model: An initialized module containing actual ``Qwen3_5MLP`` instances.
        chunk_tokens: Maximum tokens per MLP call; zero preserves native forwards.

    Yields:
        Names of patched text MLPs, for optional run receipts.

    Numerical qualification is specific to chunk size and compute dtype because
    splitting GEMMs can change floating-point gradient accumulation.
    """
    chunk_tokens = index(chunk_tokens)
    if chunk_tokens < 0:
        raise ValueError("chunk_tokens must be non-negative")
    if chunk_tokens == 0:
        yield []
        return
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

    originals, names = [], []
    try:
        for name, module in model.named_modules():
            if not isinstance(module, Qwen3_5MLP):
                continue
            if getattr(module.forward, "_qwenpt_sublayer_checkpointed", False):
                raise ValueError("MLP token chunking cannot wrap a sublayer-checkpointed forward")
            native_forward = module.forward
            originals.append((module, "forward" in module.__dict__, module.__dict__.get("forward")))
            module.forward = MethodType(_chunked_forward(native_forward, chunk_tokens), module)
            names.append(name)
        if not names:
            raise ValueError("Token chunking requires initialized Qwen3_5MLP text modules")
        yield names
    finally:
        for module, had_forward, original in reversed(originals):
            if had_forward:
                module.forward = original
            else:
                del module.forward
