"""Parquet SFT dataset implementing verl's unpadded sample contract."""

from collections.abc import Sequence
from typing import Any

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from qwenpt.train.verl.template import tokenize_sft_sample


class QwenSFTDataset(Dataset):
    """Read prepared SFT Parquet rows into unpadded tensors for native verl.

    Args mirror verl's ``data.custom_cls`` constructor. ``config`` controls
    max_length, loss_scope and enable_thinking_default; truncation is rejected
    so tool interactions and supervised answers cannot be cut silently.
    """

    def __init__(
        self,
        parquet_files: str | Sequence[str],
        tokenizer: Any,
        config: Any,
        processor: Any = None,
        max_samples: int = -1,
    ) -> None:
        if config.get("pad_mode", "no_padding") != "no_padding":
            raise ValueError("QwenSFTDataset requires data.pad_mode=no_padding")
        if config.get("truncation", "error") != "error":
            raise ValueError("QwenSFTDataset requires truncation=error; prepare complete shorter samples instead")
        kwargs = config.get("apply_chat_template_kwargs", {})
        if kwargs.get("reasoning_effort") is not None:
            raise ValueError("Leave reasoning_effort unset to use the template default")
        if kwargs.get("preserve_thinking", True) is not True:
            raise ValueError("SFT preserves historical reasoning; preserve_thinking must be true")
        unsupported = set(kwargs) - {"reasoning_effort", "preserve_thinking"}
        if unsupported:
            raise ValueError(f"Unsupported template kwargs: {sorted(unsupported)}")
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = int(config.get("max_length", 2048))
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        self.loss_scope = config.get("loss_scope", "all_assistant")
        self.enable_thinking_default = config.get("enable_thinking_default", True)
        if self.enable_thinking_default is None:
            self.enable_thinking_default = True
        paths = [parquet_files] if isinstance(parquet_files, str) else list(parquet_files)
        if not paths:
            raise ValueError("At least one Parquet file is required")
        self.rows = []
        for path in paths:
            # Read each file independently: directory partition names are not data columns.
            self.rows.extend(pq.ParquetFile(path).read().to_pylist())
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError("The SFT dataset is empty")

    def __len__(self) -> int:
        """Return the number of rows after the optional sample limit."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one unpadded example, reporting its identity on invalid input."""
        try:
            inputs = tokenize_sft_sample(
                self.rows[index], self.tokenizer, self.processor, self.loss_scope, self.enable_thinking_default
            )
            length = len(inputs["input_ids"])
            if length > self.max_length:
                raise ValueError(f"Sequence length {length} exceeds max_length={self.max_length}")
            position_ids = self.position_ids(inputs)
            result = {
                "input_ids": inputs.pop("input_ids"),
                "position_ids": position_ids,
                "loss_mask": inputs.pop("loss_mask"),
            }
            inputs.pop("attention_mask", None)
            inputs.pop("mm_token_type_ids", None)
            inputs.pop("token_type_ids", None)
            if inputs or self.processor is not None:
                result["multi_modal_inputs"] = inputs
            return result
        except (ValueError, TypeError, KeyError) as error:
            identity = self.rows[index].get("sample_id", index)
            raise ValueError(f"SFT sample {identity!r}: {error}") from error

    def position_ids(self, inputs: dict[str, Any]) -> torch.Tensor:
        """Return text or multimodal position IDs for the tokenized inputs."""
        input_ids = inputs["input_ids"]
        text = torch.arange(len(input_ids), dtype=torch.long)
        if self.processor is None:
            return text
        from verl.models.transformers.qwen3_vl import get_rope_index

        vision = get_rope_index(
            self.processor,
            input_ids=input_ids,
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            attention_mask=inputs["attention_mask"],
        )
        return torch.cat((text.unsqueeze(0), vision), dim=0)
