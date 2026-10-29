"""Optional batched cross entropy for verl's standard FSDP SFT output path."""

from contextlib import contextmanager
from functools import partial
from typing import Iterator
from unittest.mock import patch

import torch
import torch.nn.functional as F


def logprobs_from_logits_chunked(
    logits: torch.Tensor,
    labels: torch.Tensor,
    inplace_backward: bool = True,
    *,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Return target log probabilities, batching at most ``chunk_size`` tokens.

    ``logits`` has shape ``(..., vocabulary)`` and ``labels`` contains valid
    vocabulary indices with shape ``logits.shape[:-1]``.
    Low precision logits use FP32 CE, matching native log-softmax under CUDA
    autocast; FP64 is retained.
    ``inplace_backward`` is accepted for verl compatibility; inputs are unchanged.
    """
    if chunk_size < 1:
        raise ValueError("CE chunk_size must be positive")
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    losses = []
    for block, targets in zip(flat_logits.split(chunk_size), flat_labels.split(chunk_size), strict=True):
        # Bound the temporary FP32 cast instead of converting the full vocabulary matrix.
        if block.dtype in (torch.float16, torch.bfloat16):
            block = block.float()
        losses.append(F.cross_entropy(block, targets, reduction="none"))
    return -torch.cat(losses).reshape(labels.shape)


@contextmanager
def sft_logprobs_backend(backend: str = "native", chunk_size: int = 256) -> Iterator[None]:
    """Select native or chunked CE for the enclosed FSDP SFT calls, restoring on exit."""
    if backend == "native":
        yield
        return
    if backend != "chunked":
        raise ValueError(f"Unknown SFT CE backend: {backend!r}")
    if chunk_size < 1:
        raise ValueError("CE chunk_size must be positive")
    from verl.workers.engine.fsdp import transformer_impl

    replacement = partial(logprobs_from_logits_chunked, chunk_size=chunk_size)
    with patch.object(transformer_impl, "logprobs_from_logits", replacement):
        yield
