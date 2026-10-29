"""Exercise Parquet-to-token supervision with small synthetic conversations."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from synthetic_data import write_synthetic_data

from qwenpt.train.verl.dataset import QwenSFTDataset
from qwenpt.train.verl.template import assistant_loss_mask, tokenize_sft_sample


@pytest.fixture
def sample_paths(tmp_path: Path) -> dict[str, str]:
    """Create small Parquet fixtures for dataset tests."""
    return write_synthetic_data(tmp_path / "data")


def test_empty_effort_keeps_template_default(tokenizer, sample_paths) -> None:
    dataset = QwenSFTDataset(sample_paths["train"], tokenizer, {"max_length": 4096})
    item = dataset[0]
    text = tokenizer.decode(item["input_ids"])
    supervised = tokenizer.decode(item["input_ids"][item["loss_mask"].bool()])
    assert "Reasoning effort is set to xhigh." in text
    assert "Reasoning effort" not in supervised
    assert "数学助手" not in supervised
    assert "把两个整数相加，得到 5。" in supervised
    assert "答案是 5。" in supervised
    assert text.count("<think>") == 1
    assert item["position_ids"].shape == item["input_ids"].shape
    assert not item["loss_mask"][0]
    row = dict(dataset.rows[0])
    row.pop("reasoning_effort")
    assert torch.equal(tokenize_sft_sample(row, tokenizer)["input_ids"], item["input_ids"])


def test_tools_and_historical_reasoning_are_masked_correctly(tokenizer, sample_paths) -> None:
    dataset = QwenSFTDataset(sample_paths["train"], tokenizer, {"max_length": 4096})
    item = dataset[2]
    supervised = tokenizer.decode(item["input_ids"][item["loss_mask"].bool()])
    assert "先调用工具取得结果。" in supervised
    assert "<function=add>" in supervised
    assert "TOOL_RESULT_5" not in supervised
    assert "Add two integers." not in supervised
    assert supervised.count("<|im_end|>") == 2
    last = QwenSFTDataset(sample_paths["train"], tokenizer, {
        "max_length": 4096, "loss_scope": "last_assistant",
    })[2]
    last_text = tokenizer.decode(last["input_ids"][last["loss_mask"].bool()])
    assert "答案是 5。" in last_text
    assert "先调用工具取得结果。" not in last_text
    assert "<tool_call>" not in last_text


def test_template_accepts_structured_and_inline_reasoning(tokenizer) -> None:
    prefix = [{"role": "user", "content": "3 + 2?"}]
    raw = {"messages": prefix + [{"role": "assistant", "content": "<think>求和。</think>5"}]}
    structured = {"messages": prefix + [{"role": "assistant", "reasoning_content": "求和。", "content": "5"}]}
    left = tokenize_sft_sample(raw, tokenizer)
    right = tokenize_sft_sample(structured, tokenizer)
    assert torch.equal(left["input_ids"], right["input_ids"])
    assert torch.equal(left["loss_mask"], right["loss_mask"])


def test_length_errors_name_the_sample_and_do_not_truncate(tokenizer, sample_paths) -> None:
    dataset = QwenSFTDataset(sample_paths["train"], tokenizer, {"max_length": 1})
    with pytest.raises(ValueError, match="synthetic-math.*exceeds max_length"):
        dataset[0]
    with pytest.raises(ValueError, match="truncation=error"):
        QwenSFTDataset(sample_paths["train"], tokenizer, {"truncation": "right"})
    with pytest.raises(ValueError, match="pad_mode=no_padding"):
        QwenSFTDataset(sample_paths["train"], tokenizer, {"pad_mode": "right"})


def test_multiple_files_and_sample_limit(tokenizer, sample_paths) -> None:
    dataset = QwenSFTDataset(list(sample_paths.values()), tokenizer, {"max_length": 4096}, max_samples=4)
    assert len(dataset) == 4
    assert dataset.rows[3]["sample_id"] == "synthetic-math"


@pytest.mark.parametrize("max_samples", [-1, 0, 1, 1024, 1025, 1101, 2050, 4096])
def test_batched_parquet_preserves_nested_rows_order_and_sample_limit(tmp_path, tokenizer, monkeypatch, max_samples):
    """Keep every field and the original global prefix when nested rows span read batches."""
    paths = write_synthetic_data(tmp_path / "original", include_image=True)
    originals = pq.ParquetFile(paths["train"]).read().to_pylist()
    rows = [deepcopy(originals[index % len(originals)]) | {"sample_id": f"row-{index}"} for index in range(2051)]
    files, expected = [], []
    for index, file_rows in enumerate((rows[:1100], rows[1100:])):
        path = tmp_path / f"part-{index}.parquet"
        pq.write_table(pa.Table.from_pylist(file_rows), path, row_group_size=257)
        expected.extend(pq.ParquetFile(path).read().to_pylist())
        files.append(str(path))

    def reject_full_table_read(*args, **kwargs):
        raise pa.ArrowNotImplementedError("Nested data conversions not implemented for chunked array outputs")

    iter_batches = pq.ParquetFile.iter_batches
    batch_rows = []

    def record_batches(self, *args, **kwargs):
        for batch in iter_batches(self, *args, **kwargs):
            batch_rows.append(batch.num_rows)
            yield batch

    monkeypatch.setattr(pq.ParquetFile, "read", reject_full_table_read)
    monkeypatch.setattr(pq.ParquetFile, "iter_batches", record_batches)
    dataset = QwenSFTDataset(files, tokenizer, {"max_length": 4096}, max_samples=max_samples)
    expected = expected[:max_samples] if max_samples > 0 else expected
    assert dataset.rows == expected
    assert [row["sample_id"] for row in dataset.rows] == [f"row-{index}" for index in range(len(expected))]
    assert len(dataset) == len(expected)
    assert sum(batch_rows) == len(rows)
    assert len(batch_rows) > len(files)
    assert max(batch_rows) <= 1024


def test_embedded_image_requires_processor(tmp_path, tokenizer) -> None:
    paths = write_synthetic_data(tmp_path / "images", include_image=True)
    rows = pq.read_table(paths["train"]).to_pylist()
    assert rows[3]["images"][0]["bytes"].startswith(b"\x89PNG")
    dataset = QwenSFTDataset(paths["train"], tokenizer, {"max_length": 4096})
    with pytest.raises(ValueError, match="synthetic-image.*processor is required"):
        dataset[3]
    with pytest.raises(FileExistsError):
        write_synthetic_data(tmp_path / "images")


@pytest.fixture
def processor(tokenizer):
    """Build a local processor for the real text and image batching paths."""
    from transformers import Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor

    return Qwen3VLProcessor(
        tokenizer=tokenizer, chat_template=tokenizer.chat_template,
        image_processor=Qwen2VLImageProcessor(
            patch_size=16, min_pixels=1024, max_pixels=1024,
        ),
        video_processor=Qwen3VLVideoProcessor(),
    )


@pytest.mark.parametrize("indices", [[0, 1], [0], [0, 3], [3, 0], [3, 3]], ids=[
    "text-batch", "text-single", "mixed-text-first", "mixed-image-first", "identical-images",
])
def test_processor_batches_reach_native_trainer_tensordict(
    tmp_path, tokenizer, processor, monkeypatch, indices,
) -> None:
    """Run processor batches through the entrypoint and native TensorDict bridge."""
    from omegaconf import OmegaConf
    from tensordict import set_capture_non_tensor_stack
    from tensordict.tensorclass import NonTensorStack
    from verl.utils.dataset.dataset_utils import SFTTensorCollator
    from verl.utils.model import extract_multi_modal_inputs
    from verl.utils.tensordict_utils import get_tensordict

    from qwenpt.train.sft import main

    paths = write_synthetic_data(tmp_path / "mixed", include_image=True)
    dataset = QwenSFTDataset(paths["train"], tokenizer, {"max_length": 4096}, processor=processor)
    text, image = dataset[0], dataset[3]
    assert text["multi_modal_inputs"] == {}
    assert "pixel_values" in image["multi_modal_inputs"]
    assert image["position_ids"].shape == (4, len(image["input_ids"]))
    assert text["position_ids"].shape == (4, len(text["input_ids"]))
    image_token = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    assert not image["loss_mask"][image["input_ids"] == image_token].any()
    by_index = {index: dataset[index] for index in set(indices)}
    items = [by_index[index] for index in indices]
    called = []

    def run_batch(config):
        batch = SFTTensorCollator("no_padding")(items)
        assert batch["input_ids"].is_nested
        assert isinstance(batch["multi_modal_inputs"], NonTensorStack)
        data = get_tensordict(batch, non_tensor_dict={"pad_mode": "no_padding"})
        assert data.batch_size == torch.Size([len(items)])
        media = extract_multi_modal_inputs(data["multi_modal_inputs"])
        image_inputs = [item["multi_modal_inputs"] for item in items if item["multi_modal_inputs"]]
        if image_inputs:
            assert set(media) == set(image_inputs[0])
            for key in media:
                torch.testing.assert_close(media[key], torch.cat([row[key] for row in image_inputs]))
        else:
            assert media == {}
        called.append(True)

    monkeypatch.setattr("qwenpt.train.sft.run_sft", run_batch)
    # Reproduce the affected legacy mode even if the shell default was changed.
    with set_capture_non_tensor_stack(True):
        main.__wrapped__(OmegaConf.create({"device": "cpu"}))
    assert called == [True]


def test_malformed_rendered_turn_is_rejected(tokenizer) -> None:
    ids = torch.tensor(tokenizer.encode("<|im_start|>assistant\nmissing close", add_special_tokens=False))
    with pytest.raises(ValueError, match="unterminated"):
        assistant_loss_mask(ids, tokenizer)


def test_template_mapping_preserves_delivered_sample(tokenizer, sample_paths) -> None:
    sample = pq.read_table(sample_paths["train"]).to_pylist()[2]
    original = deepcopy(sample)
    tokenize_sft_sample(sample, tokenizer)
    assert sample == original
    assert isinstance(sample["messages"][1]["tool_calls"][0]["function"]["arguments"], str)


def test_delivered_json_tools_and_message_mask_preserve_context(tokenizer, sample_paths) -> None:
    """A masked historical call remains input context without receiving loss."""
    import json

    sample = pq.read_table(sample_paths["train"]).to_pylist()[2]
    sample["tools"] = json.dumps(sample["tools"])
    assistants = [message for message in sample["messages"] if message["role"] == "assistant"]
    assistants[0]["loss_mask"] = 0
    assistants[1]["loss_mask"] = 1
    item = tokenize_sft_sample(sample, tokenizer)
    text = tokenizer.decode(item["input_ids"])
    supervised = tokenizer.decode(item["input_ids"][item["loss_mask"].bool()])
    assert "先调用工具取得结果。" in text
    assert "先调用工具取得结果。" not in supervised
    assert "<function=add>" not in supervised
    assert "答案是 5。" in supervised
    assert supervised.count("<|im_end|>") == 1


def test_all_masked_sample_is_rejected(tokenizer) -> None:
    sample = {"tools": "[]", "messages": [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "context", "loss_mask": 0},
    ]}
    with pytest.raises(ValueError, match="no supervised tokens"):
        tokenize_sft_sample(sample, tokenizer)


def test_nullable_thinking_uses_training_default(tokenizer, sample_paths) -> None:
    sample = pq.read_table(sample_paths["train"]).to_pylist()[0]
    sample["enable_thinking"] = None
    item = tokenize_sft_sample(sample, tokenizer)
    assert "Reasoning effort is set to xhigh." in tokenizer.decode(item["input_ids"])
    sample["enable_thinking"] = False
    sample["messages"][-1]["content"] = "答案是 5。"
    item = tokenize_sft_sample(sample, tokenizer)
    assert "Reasoning effort is set to xhigh." not in tokenizer.decode(item["input_ids"])


def test_native_collator_and_sft_loss_match_causal_cross_entropy(tokenizer, sample_paths) -> None:
    from tensordict import TensorDict
    from tensordict.tensorclass import NonTensorData
    from transformers import GPT2Config, GPT2LMHeadModel
    from verl.utils.dataset.dataset_utils import SFTTensorCollator
    from verl.workers.utils.losses import sft_loss

    torch.manual_seed(7)
    torch.set_num_threads(1)
    dataset = QwenSFTDataset(sample_paths["train"], tokenizer, {"max_length": 4096}, max_samples=2)
    items = [dataset[index] for index in range(2)]
    batch = SFTTensorCollator("no_padding")(items)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_positions=2048, n_embd=16, n_layer=1, n_head=2,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
    ))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def losses():
        log_probs, sums = [], []
        for item in items:
            ids = item["input_ids"]
            logits = model(ids[None]).logits[0]
            next_tokens = torch.roll(ids, shifts=-1)
            values = logits.log_softmax(-1).gather(-1, next_tokens[:, None]).squeeze(-1)
            log_probs.append(values)
            sums.append(-(values[:-1] * item["loss_mask"][1:]).sum())
        count = sum(int(item["loss_mask"].sum()) for item in items)
        data = TensorDict({
            "loss_mask": batch["loss_mask"], "pad_mode": NonTensorData("no_padding"),
            "dp_size": 1, "batch_num_tokens": count,
        }, batch_size=[])
        nested = torch.nested.as_nested_tensor(log_probs, layout=torch.jagged)
        native, _ = sft_loss(None, {"log_probs": nested}, data)
        return native, sum(sums) / count

    loss, expected = losses()
    torch.testing.assert_close(loss, expected)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    after, _ = losses()
    assert after < loss.detach()


@pytest.mark.parametrize("mask_mode", ["present", "missing", "none"])
def test_attention_mask_fallback_only_allocates_when_missing(
    tokenizer, sample_paths, monkeypatch, mask_mode,
) -> None:
    """Preserve supplied masks and the existing None error without unused allocation."""
    sample = pq.read_table(sample_paths["train"]).to_pylist()[2]
    original = deepcopy(sample)
    expected = tokenize_sft_sample(sample, tokenizer)
    rendered = {key: value for key, value in expected.items() if key != "loss_mask"}
    rendered["input_ids"] = expected["input_ids"].unsqueeze(0)
    if mask_mode == "present":
        expected["attention_mask"] = expected["attention_mask"].clone()
        expected["attention_mask"][0] = 0
        rendered["attention_mask"] = expected["attention_mask"].unsqueeze(0)
    elif mask_mode == "missing":
        rendered.pop("attention_mask")
    else:
        rendered["attention_mask"] = None
    monkeypatch.setattr(tokenizer, "apply_chat_template", lambda *args, **kwargs: rendered)
    ones_like = Mock(wraps=torch.ones_like)
    monkeypatch.setattr(torch, "ones_like", ones_like)

    if mask_mode == "none":
        with pytest.raises(TypeError, match="'NoneType' object is not subscriptable"):
            tokenize_sft_sample(sample, tokenizer)
    else:
        actual = tokenize_sft_sample(sample, tokenizer)
        assert actual.keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert ones_like.call_count == (1 if mask_mode == "missing" else 0)
    if mask_mode == "missing":
        assert ones_like.call_args.args[0] is rendered["input_ids"]
    assert sample == original
