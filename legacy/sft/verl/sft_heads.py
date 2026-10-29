"""Equivalent trainable SFT output heads for explicit numerical and timing probes."""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import importlib.metadata
from types import MethodType
from typing import Iterator
from unittest.mock import patch

import torch
import torch.nn.functional as F


BACKENDS = ("native", "verl_torch", "verl_triton", "liger", "cce_exact")
PROBE_BACKENDS = ("native", "verl_unfused", *BACKENDS[1:])


@dataclass(frozen=True)
class SFTTargets:
    """Next-token labels, supervision and native verl DP normalization.

    ``labels`` and ``mask`` are already shifted to align with hidden states.
    ``normalizer`` is ``dp_size / global_supervised_tokens``, so FSDP's averaged
    gradients recover the global token mean even with unequal rank/microbatch sizes.
    """

    labels: torch.Tensor
    mask: torch.Tensor
    normalizer: float


def prepare_sft_targets(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    offsets: torch.Tensor,
    *,
    global_supervised_tokens: int,
    dp_size: int = 1,
) -> SFTTargets:
    """Prepare packed targets, preserving message masks and excluding sample boundaries.

    Args:
        input_ids: Concatenated one-dimensional sample token IDs.
        loss_mask: Binary, unshifted token mask produced by the message-aware dataset.
        offsets: Sample offsets, including zero and the total number of tokens.
        global_supervised_tokens: Full optimizer batch's supervised token count across DP.
        dp_size: Number of data-parallel ranks whose gradients are averaged.

    Returns:
        Targets matching verl's no-padding label roll and SFT loss-mask roll.
    """
    if input_ids.ndim != 1 or input_ids.dtype != torch.long or loss_mask.shape != input_ids.shape:
        raise ValueError("input_ids and loss_mask must be matching flat tensors with int64 input_ids")
    if offsets.ndim != 1 or offsets.numel() < 2 or offsets.dtype != torch.long:
        raise ValueError("offsets must be a one-dimensional int64 sample-boundary tensor")
    boundaries = offsets.detach().cpu().tolist()
    if boundaries[0] != 0 or boundaries[-1] != input_ids.numel() or any(
        end <= start for start, end in zip(boundaries, boundaries[1:])
    ):
        raise ValueError("offsets must increase from zero to the packed token count")
    if not bool(((loss_mask == 0) | (loss_mask == 1)).all()):
        raise ValueError("Only binary message/token loss masks are supported")
    if bool(loss_mask[offsets[:-1]].any()):
        raise ValueError("Every sample's first token must be unsupervised to prevent cross-sample targets")
    if global_supervised_tokens < 1 or dp_size < 1:
        raise ValueError("global_supervised_tokens and dp_size must be positive")
    return SFTTargets(input_ids.roll(-1), loss_mask.bool().roll(-1), dp_size / global_supervised_tokens)


def sft_head_loss(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: SFTTargets,
    *,
    backend: str = "native",
    chunk_size: int = 512,
) -> torch.Tensor:
    """Compute a masked SFT scalar loss with both hidden and output-weight gradients.

    Args:
        hidden: Packed hidden states with shape ``(tokens, hidden_size)``.
        weight: Trainable lm_head matrix with shape ``(vocabulary, hidden_size)``.
        targets: Shifted labels, binary mask and DP/global-batch normalization.
        backend: Unfused reference, either native verl fusion, Liger FLCE or exact CCE.
        chunk_size: Token chunk size for verl's built-in torch fused output head.

    Returns:
        Local supervised loss sum multiplied by ``dp_size / global_supervised_tokens``.
    """
    log_probs = sft_head_logprobs(hidden, weight, targets.labels, backend=backend, chunk_size=chunk_size)
    return -(log_probs * targets.mask).sum() * targets.normalizer


def sft_head_logprobs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    backend: str,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Return per-token log probabilities, retaining arbitrary masked upstream gradients.

    Hidden states may be packed matrices or dense ``(batch, sequence, hidden)``
    tensors; labels must already be next-token shifted and match the leading shape.
    Both the hidden state and the lm_head weights remain trainable.
    """
    if backend not in PROBE_BACKENDS:
        raise ValueError(f"Unknown SFT output-head backend: {backend!r}")
    if hidden.ndim not in (2, 3) or weight.ndim != 2 or hidden.shape[-1] != weight.shape[-1]:
        raise ValueError("hidden and weight must share their hidden dimension")
    if labels.shape != hidden.shape[:-1]:
        raise ValueError("Shifted targets must have one entry per hidden-state token")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    shape = labels.shape
    hidden = hidden.reshape(-1, hidden.shape[-1])
    labels = labels.reshape(-1)
    if backend in ("native", "verl_unfused"):
        logits = F.linear(hidden, weight)
        if backend == "verl_unfused":
            from verl.utils.torch_functional import logprobs_from_logits

            log_probs = logprobs_from_logits(logits, labels, inplace_backward=True)
        else:
            if logits.dtype != torch.float64:
                logits = logits.float()
            log_probs = -F.cross_entropy(logits, labels, reduction="none")
    elif backend == "verl_torch":
        from verl.utils.experimental.torch_functional import FusedLinearForPPO

        log_probs, _ = FusedLinearForPPO(chunk_size=chunk_size, impl_backend="torch")(
            hidden, weight, labels,
        )
    elif backend == "verl_triton":
        from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

        log_probs, _ = linear_cross_entropy(hidden, weight, labels, 1.0, "none")
    elif backend == "liger":
        from packaging.version import Version
        from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

        if Version(importlib.metadata.version("liger-kernel")) < Version("0.8.5"):
            raise RuntimeError("Masked per-token Liger FLCE requires >=0.8.5 for correct lm_head gradients")
        log_probs = -LigerFusedLinearCrossEntropyLoss(reduction="none")(weight, hidden, labels)
    else:
        from cut_cross_entropy import linear_cross_entropy

        log_probs = -linear_cross_entropy(
            hidden, weight, labels, reduction="none", shift=0,
            impl="cce_exact", filter_eps=None, accum_e_fp32=True, accum_c_fp32=True,
            filter_e_grad=False, filter_c_grad=False,
        )
    return log_probs.reshape(shape)


def fused_qwen3_5_sft_forward(
    self,
    input_ids: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    temperature: float = 1.0,
    shift_labels: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    *,
    backend: str,
    chunk_size: int = 512,
    **kwargs,
):
    """Use the Qwen3.5 decoder with a trainable per-token fused SFT output head.

    This adapter preserves the engine's already shifted labels and packed-sequence
    arguments; the existing verl SFT loss still applies message masks and global
    DP normalization outside the model. Entropy/RL losses are intentionally unsupported.
    """
    from verl.models.transformers.qwen3_5 import Qwen3_5CausalLMOutputForPPO
    from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size

    if self.config.model_type != "qwen3_5" or temperature != 1.0:
        raise ValueError("This SFT output-head adapter requires dense Qwen3.5 and unit temperature")
    if self.lm_head.bias is not None or hasattr(self.lm_head, "lora_A"):
        raise ValueError("This SFT output-head adapter requires a bias-free head without LoRA")
    if not self.lm_head.weight.requires_grad:
        raise ValueError("This SFT comparison requires a trainable lm_head weight")
    if cu_seqlens is not None:
        kwargs["cu_seqlens"] = cu_seqlens
    if cu_seqlens_cpu is not None:
        kwargs["cu_seqlens_cpu"] = cu_seqlens_cpu
    targets = shift_labels
    if targets is None:
        if get_ulysses_sequence_parallel_world_size() > 1:
            raise ValueError("Sequence parallelism requires globally shifted engine-provided labels")
        source = labels if labels is not None else input_ids
        if source is None:
            raise ValueError("SFT output-head adapter requires input_ids or labels")
        targets = source.roll(-1, dims=-1)
    outputs = self.model(input_ids, **kwargs)
    log_probs, entropy = call_sft_head_module(self.lm_head, outputs[0], targets,
                                             backend=backend, chunk_size=chunk_size)
    return Qwen3_5CausalLMOutputForPPO(log_probs=log_probs, entropy=entropy, hidden_states=outputs.hidden_states)


def _sft_head_module_forward(head, hidden, labels, *, backend, chunk_size):
    from torch.distributed.tensor import DTensor

    weight = head.weight
    if isinstance(weight, DTensor):
        raise RuntimeError("SFT fusion must execute inside the lm_head FSDP module hooks")
    shape = labels.shape
    if hidden.shape[:-1] != shape:
        raise ValueError("Shifted targets must have one entry per hidden-state token")
    hidden, labels = hidden.reshape(-1, hidden.shape[-1]), labels.reshape(-1)
    entropy = None
    if backend == "verl_torch":
        from verl.utils.experimental.torch_functional import FusedLinearForPPO

        log_probs, entropy = FusedLinearForPPO(chunk_size=chunk_size, impl_backend="torch")(
            hidden, weight, labels,
        )
    elif backend == "verl_triton":
        from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

        log_probs, entropy = linear_cross_entropy(hidden, weight, labels, 1.0, "none")
    else:
        log_probs = sft_head_logprobs(hidden, weight, labels, backend=backend, chunk_size=chunk_size)
    # Triton's kernel returns a flat vector; the dense engine requires B x S.
    return log_probs.reshape(shape), entropy.reshape(shape) if entropy is not None else None


def call_sft_head_module(head, hidden, labels, *, backend, chunk_size=512):
    """Execute a fused kernel through the real head's FSDP hooks and restore forward.

    Reading a sharded weight with ``full_tensor()`` bypasses BF16 casting and DP
    gradient averaging. A module call preserves FSDP's parameter lifecycle without
    patching global Linear classes or changing the chosen kernel's mathematics.
    """
    with patch.object(head, "forward", MethodType(_sft_head_module_forward, head)):
        return head(hidden, labels, backend=backend, chunk_size=chunk_size)


@contextmanager
def sft_output_head_backend(config) -> Iterator[None]:
    """Select an opt-in SFT output head around native verl trainer construction/run.

    ``trainer.output_head_backend`` selects native, verl_torch, verl_triton, liger
    or cce_exact. Fused choices require ``model.use_fused_kernels=true`` and a
    matching ``model.fused_kernel_options.impl_backend`` (torch for Liger/CCE).
    All fused arms retain the chosen upstream kernel and execute it through the
    actual head module so FSDP2 casts and averages gradients correctly.
    """
    backend = config.trainer.get("output_head_backend", "native")
    if backend not in BACKENDS:
        raise ValueError(f"Unknown SFT output-head backend: {backend!r}")
    if backend == "native":
        if config.model.use_fused_kernels:
            raise ValueError("native output-head reference requires model.use_fused_kernels=false")
        yield
        return
    impl = "triton" if backend == "verl_triton" else "torch"
    if not config.model.use_fused_kernels or config.model.fused_kernel_options.impl_backend != impl:
        raise ValueError(f"{backend} requires model.use_fused_kernels=true and impl_backend={impl}")
    if config.engine.strategy != "fsdp2":
        raise ValueError("The trainable SFT output-head adapter has only been prepared for FSDP2")
    chunk_size = config.trainer.get("output_head_chunk_size", 512)
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    from verl.models.transformers import qwen3_5

    @wraps(fused_qwen3_5_sft_forward)
    def replacement(model, *args, **kwargs):
        return fused_qwen3_5_sft_forward(model, *args, backend=backend, chunk_size=chunk_size, **kwargs)
    # Restore the Transformers class after verl's construction-time monkey patch.
    with (
        patch.object(qwen3_5, f"forward_with_{impl}_backend", replacement),
        patch.object(Qwen3_5ForConditionalGeneration, "forward", Qwen3_5ForConditionalGeneration.forward),
    ):
        yield
