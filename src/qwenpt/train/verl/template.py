"""Render complete Qwen conversations and identify assistant supervision."""

from copy import deepcopy
from io import BytesIO
import json
import re
from typing import Any

from PIL import Image
import torch


def assistant_loss_mask(input_ids: torch.Tensor, tokenizer: Any, loss_scope: str = "all_assistant") -> torch.Tensor:
    """Mask assistant bodies and closing tokens in a one-dimensional Qwen sequence.

    ``tokenizer`` must register the Qwen ChatML boundary tokens; ``loss_scope``
    selects all assistant turns or only the last one. Prepared messages must not
    contain literal ChatML boundary tokens inside their content.
    """
    if loss_scope not in ("all_assistant", "last_assistant"):
        raise ValueError("loss_scope must be all_assistant or last_assistant")
    if input_ids.ndim != 1:
        raise ValueError("input_ids must be one-dimensional")
    start = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(start) != 1 or len(end) != 1 or start == end:
        raise ValueError("The tokenizer must register Qwen ChatML boundary tokens")
    header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    ids = input_ids.tolist()
    spans = []
    cursor = 0
    while cursor < len(ids):
        try:
            opening = ids.index(start[0], cursor)
        except ValueError:
            break
        try:
            closing = ids.index(end[0], opening + 1)
        except ValueError as error:
            raise ValueError("Rendered conversation contains an unterminated ChatML turn") from error
        if start[0] in ids[opening + 1 : closing]:
            raise ValueError("Rendered conversation contains nested ChatML boundaries")
        if ids[opening : opening + len(header)] == header:
            spans.append((opening + len(header), closing + 1))
        cursor = closing + 1
    if not spans:
        raise ValueError("Rendered conversation contains no assistant supervision")
    mask = torch.zeros_like(input_ids)
    for first, last in spans[-1:] if loss_scope == "last_assistant" else spans:
        mask[first:last] = 1
    return mask


def tokenize_sft_sample(
    sample: dict[str, Any],
    tokenizer: Any,
    processor: Any = None,
    loss_scope: str = "all_assistant",
    enable_thinking_default: bool = True,
) -> dict[str, Any]:
    """Tokenize a prepared Parquet sample into model inputs plus loss_mask.

    Data validity is guaranteed by the data producer. Template arguments omit
    reasoning_effort, retaining default xhigh; media requires a processor.
    """
    if not isinstance(enable_thinking_default, bool):
        raise ValueError("enable_thinking_default must be a boolean")
    has_media = bool(sample.get("images") or sample.get("videos"))
    if has_media and processor is None:
        raise ValueError("A multimodal processor is required for images or videos")
    thinking = sample.get("enable_thinking")
    if thinking is None:
        thinking = enable_thinking_default
    messages = template_messages(sample)
    if processor is not None:
        messages = processor_messages(messages, sample)
    renderer = processor if processor is not None else tokenizer
    inputs = dict(
        renderer.apply_chat_template(
            messages,
            tools=sample.get("tools") or None,
            enable_thinking=thinking,
            preserve_thinking=True,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    )
    input_ids = inputs["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("The processor must return exactly one tokenized conversation")
    inputs["input_ids"] = input_ids[0]
    inputs["attention_mask"] = inputs.get("attention_mask", torch.ones_like(input_ids))[0]
    inputs["loss_mask"] = assistant_loss_mask(inputs["input_ids"], tokenizer, loss_scope)
    return inputs


def template_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Copy sample messages into the structured representation expected by Qwen."""
    messages = deepcopy(sample["messages"])
    for message in messages:
        if message["role"] != "assistant":
            continue
        # Qwen expects reasoning and tool arguments as separate structured fields.
        content = message.get("content") or ""
        if not message.get("reasoning_content"):
            match = re.fullmatch(r"\s*<think>(.*?)</think>(.*)", content, flags=re.DOTALL)
            if match is not None:
                message["reasoning_content"], message["content"] = match.groups()
        for call in message.get("tool_calls") or []:
            function = call["function"]
            if isinstance(function["arguments"], str):
                function["arguments"] = json.loads(function["arguments"])
    return messages


def processor_messages(messages: list[dict[str, Any]], sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Fill media placeholders from sample in place and return processor messages."""
    media = {kind: iter(sample.get(kind + "s") or []) for kind in ("image", "video")}
    for message in messages:
        parts = []
        for segment in re.split(r"(<image>|<video>)", message.get("content") or ""):
            if segment in ("<image>", "<video>"):
                kind = segment[1:-1]
                entry = next(media[kind])
                value = entry.get(kind)
                if value is None and entry.get("bytes") is not None:
                    if kind != "image":
                        raise ValueError("Embedded video bytes require a decoded video or file path")
                    with Image.open(BytesIO(entry["bytes"])) as image:
                        value = image.convert("RGB")
                if value is None:
                    value = entry["path"]
                parts.append({"type": kind, kind: value})
            elif segment:
                parts.append({"type": "text", "text": segment})
        message["content"] = parts
    return messages
