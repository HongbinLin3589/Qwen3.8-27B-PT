"""Opt-in bounded-memory output projection for BOLT's frozen Qwen3.5 head."""

from contextlib import contextmanager
from functools import wraps
import math
from typing import Iterator
from unittest.mock import patch

import torch
from torch.distributed.tensor import DTensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def chunked_linear_logprobs(
    hidden_states: torch.Tensor,
    vocab_weights: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = 128,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return target log-probabilities without retaining sequence-wide logits.

    ``hidden_states`` is ``(..., hidden)``, frozen ``vocab_weights`` is
    ``(vocabulary, hidden)``, and integer ``labels`` is ``hidden_states.shape[:-1]``.
    Training recomputes each projection/CE block during backward; validation
    releases it immediately, bounding vocabulary-sized temporaries by chunk size.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("CE chunk_size must be a positive integer")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("CE temperature must be finite and positive")
    if vocab_weights.requires_grad:
        raise ValueError("Chunked BOLT CE requires frozen output weights")
    if hidden_states.ndim not in (2, 3) or vocab_weights.ndim != 2:
        raise ValueError("Expected 2D/3D hidden states and 2D output weights")
    if hidden_states.shape[-1] != vocab_weights.shape[-1] or labels.shape != hidden_states.shape[:-1]:
        raise ValueError("Hidden states, output weights and labels have incompatible shapes")
    if labels.dtype != torch.long or not labels.numel():
        raise ValueError("CE labels must be a nonempty long tensor")

    def block_logprobs(hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden, vocab_weights)
        if logits.dtype != torch.float64:
            logits = logits.float()
        return -F.cross_entropy(logits / temperature, targets, reduction="none")

    chunks = []
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    targets = labels.reshape(-1)
    for block, block_targets in zip(hidden.split(chunk_size), targets.split(chunk_size), strict=True):
        if torch.is_grad_enabled() and hidden_states.requires_grad:
            # Chunking CE alone still retains all its vocabulary-sized activations.
            values = checkpoint(
                block_logprobs, block, block_targets, use_reentrant=False, preserve_rng_state=False,
            )
        else:
            values = block_logprobs(block, block_targets)
        chunks.append(values)
    return torch.cat(chunks).reshape(labels.shape)


def chunked_qwen3_5_forward(
    self,
    input_ids: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    temperature: float = 1.0,
    shift_labels: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    *,
    chunk_size: int = 128,
    **kwargs,
):
    """Return per-token log-probabilities through verl's Qwen3.5 fused adapter.

    Only a frozen, bias-free head without an adapter is supported; gathering its
    weight once avoids one FSDP all-gather for every token block.
    """
    from verl.models.transformers.qwen3_5 import Qwen3_5CausalLMOutputForPPO

    if self.config.model_type != "qwen3_5":
        raise ValueError("Chunked BOLT CE currently supports only dense Qwen3.5")
    head = self.lm_head
    if (not isinstance(head, torch.nn.Linear) or hasattr(head, "lora_A")
            or head.bias is not None or any(parameter.requires_grad for parameter in head.parameters())):
        raise ValueError("Chunked BOLT CE requires a frozen, bias-free lm_head without LoRA")
    if cu_seqlens is not None or cu_seqlens_cpu is not None:
        raise ValueError("Chunked BOLT CE currently requires dense microbatches")
    targets = shift_labels
    if targets is None:
        source = labels if labels is not None else input_ids
        if source is None:
            raise ValueError("Chunked BOLT CE requires input_ids or labels")
        targets = torch.roll(source, shifts=-1, dims=-1)

    outputs = self.model(input_ids, **kwargs)
    weight = head.weight
    if isinstance(weight, DTensor):
        weight = weight.full_tensor()
    if not getattr(self, "bolt_ce_backend_logged", False):
        print(
            f"BOLT CE: backend=chunked_linear chunk_size={chunk_size} "
            f"frozen_head={tuple(weight.shape)} hidden={tuple(outputs[0].shape)}",
            flush=True,
        )
        self.bolt_ce_backend_logged = True
    log_probs = chunked_linear_logprobs(
        outputs[0].to(weight.dtype), weight, targets, chunk_size=chunk_size, temperature=temperature,
    )
    return Qwen3_5CausalLMOutputForPPO(log_probs=log_probs, hidden_states=outputs.hidden_states)


@contextmanager
def bolt_logprobs_backend(config) -> Iterator[None]:
    """Enable chunked linear CE only within an explicitly configured BOLT run."""
    backend = config.trainer.get("ce_backend", "native")
    if backend == "native":
        yield
        return
    if backend != "chunked_linear":
        raise ValueError(f"Unknown BOLT CE backend: {backend!r}")
    chunk_size = config.trainer.get("ce_chunk_size", 128)
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("CE chunk_size must be a positive integer")
    if (not config.model.use_fused_kernels or config.model.fused_kernel_options.impl_backend != "torch"
            or config.model.use_remove_padding or config.engine.strategy != "fsdp2"
            or config.engine.ulysses_sequence_parallel_size != 1):
        raise ValueError("Chunked BOLT CE requires torch fused kernels, dense microbatches and FSDP2 without SP")

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    from verl.models.transformers import qwen3_5

    @wraps(chunked_qwen3_5_forward)
    def replacement(model, *args, **kwargs):
        return chunked_qwen3_5_forward(model, *args, chunk_size=chunk_size, **kwargs)

    with (
        patch.object(qwen3_5, "forward_with_torch_backend", replacement),
        patch.object(Qwen3_5ForConditionalGeneration, "forward", Qwen3_5ForConditionalGeneration.forward),
    ):
        yield
