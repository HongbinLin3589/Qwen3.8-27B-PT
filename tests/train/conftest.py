"""Offline tokenizer fixtures using the pinned official Qwen template."""

from pathlib import Path

import pytest
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

from tests.train.common.synthetic_data import write_synthetic_data


@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizerFast:
    """Build a tiny byte-level tokenizer without downloading any model assets."""
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>",
               "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
    backend = ByteLevelBPETokenizer()
    backend.train_from_iterator(
        ["system user assistant tool\n", "数学助手答案是五。推理过程。", "add function parameter tool_call",
         "Reasoning effort is set to xhigh. Please think carefully through the task."] * 4,
        vocab_size=384, special_tokens=special,
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend._tokenizer, pad_token=special[0], eos_token="<|im_end|>",
        chat_template=(Path(__file__).parent / "common/fixtures/chat_template.jinja").read_text(),
    )


@pytest.fixture
def sample_paths(tmp_path: Path) -> dict[str, str]:
    """Create small Parquet fixtures for dataset tests."""
    return write_synthetic_data(tmp_path / "data")
