"""Prompt-group reward weighting over the complete SFT conversation format."""

from collections.abc import Sequence
import math
from numbers import Real
from typing import Any

import torch

from qwenpt.train.verl.dataset import QwenSFTDataset


def compute_bolt_weights(rewards: Sequence[float] | torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """Return float64 ``N * softmax(rewards / beta)`` for one complete prompt group.

    Rewards retain their original numeric scale and beta must be finite and positive.
    Centering before division avoids overflow for small temperatures.
    """
    if isinstance(beta, bool) or not isinstance(beta, Real) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("bolt_beta must be a finite positive number")
    if isinstance(rewards, torch.Tensor):
        if rewards.dtype == torch.bool or rewards.is_complex():
            raise ValueError("BOLT rewards must be finite real numbers")
    elif any(isinstance(reward, bool) or not isinstance(reward, Real) for reward in rewards):
        raise ValueError("BOLT rewards must be finite real numbers")
    values = torch.as_tensor(rewards, dtype=torch.float64)
    if values.ndim != 1 or not values.numel() or not torch.isfinite(values).all():
        raise ValueError("BOLT rewards must be a nonempty one-dimensional sequence of finite real numbers")
    # Scale large temperatures first so subtracting opposite finite extremes
    # cannot overflow before an otherwise well-conditioned division.
    scaled = values / max(1.0, beta)
    logits = (scaled - scaled.max()) / min(1.0, beta)
    return values.numel() * torch.softmax(logits, dim=0)


class QwenBOLTDataset(QwenSFTDataset):
    """Read complete trajectories and attach fixed weights from each prompt group.

    The flat Parquet rows use the SFT messages/tools/media fields plus a scalar
    ``reward`` and a ``prompt_id`` identifying the same conditioning input,
    including tools and media. Groups span all provided files; their weights
    are computed before applying ``max_samples``.
    """

    def __init__(
        self,
        parquet_files: str | Sequence[str],
        tokenizer: Any,
        config: Any,
        processor: Any = None,
        max_samples: int = -1,
    ) -> None:
        super().__init__(parquet_files, tokenizer, config, processor, max_samples=-1)
        self.bolt_beta = config.get("bolt_beta", 0.1)
        groups: dict[str | int, list[int]] = {}
        for index, row in enumerate(self.rows):
            identity = row.get("sample_id", index)
            prompt_id = row.get("prompt_id")
            if isinstance(prompt_id, bool) or not isinstance(prompt_id, (str, int)) or prompt_id == "":
                raise ValueError(f"BOLT sample {identity!r}: prompt_id must be a nonempty string or integer")
            reward = row.get("reward")
            if isinstance(reward, bool) or not isinstance(reward, Real) or not math.isfinite(reward):
                raise ValueError(f"BOLT sample {identity!r}: reward must be a finite real number")
            groups.setdefault(prompt_id, []).append(index)
        self.prompt_ids = set(groups)
        self.bolt_weights = torch.empty(len(self.rows), dtype=torch.float64)
        for indices in groups.values():
            rewards = [self.rows[index]["reward"] for index in indices]
            self.bolt_weights[indices] = compute_bolt_weights(rewards, self.bolt_beta)
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
            self.bolt_weights = self.bolt_weights[:max_samples]

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return SFT tensors with a one-element weight for verl's jagged collator."""
        result = super().__getitem__(index)
        result["bolt_weight"] = self.bolt_weights[index].reshape(1).to(dtype=torch.float32)
        return result
