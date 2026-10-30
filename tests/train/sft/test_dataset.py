"""Verify shared dataset batches against the native SFT trainer and loss."""

import pytest
import torch

from qwenpt.train.common.dataset import QwenSFTDataset
from tests.train.common.synthetic_data import write_synthetic_data


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

    from qwenpt.train.sft.entrypoint import main

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

    monkeypatch.setattr("qwenpt.train.sft.entrypoint.run_sft", run_batch)
    # Reproduce the affected legacy mode even if the shell default was changed.
    with set_capture_non_tensor_stack(True):
        main.__wrapped__(OmegaConf.create({"device": "cpu"}))
    assert called == [True]


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
