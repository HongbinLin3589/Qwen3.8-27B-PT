"""Verify that captured decoder activations retain their actual supervision."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from legacy.sft.utils.sft_benchmark import BenchmarkRecorder


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(3, 19, bias=False)


def test_real_head_capture_preserves_hidden_and_unshifted_message_mask(tmp_path):
    model = _Model()
    recorder = BenchmarkRecorder(tmp_path, 2, 0)
    recorder.capture_head_input(SimpleNamespace(engine=SimpleNamespace(module=model)))
    ids = torch.tensor([[2, 3, 5, 7, 11]])
    mask = torch.tensor([[0, 1, 0, 1, 1]])
    data = TensorDict({"input_ids": ids, "loss_mask": mask, "position_ids": torch.arange(5)[None]}, [])
    recorder.before_batch(data=data)
    hidden = torch.randn(1, 5, 3)
    model.lm_head(hidden)
    snapshot = torch.load(tmp_path / "head_input_rank0.pt", weights_only=True)
    torch.testing.assert_close(snapshot["hidden"], hidden[0])
    assert torch.equal(snapshot["input_ids"], ids[0])
    assert torch.equal(snapshot["loss_mask"], mask[0])
    assert snapshot["offsets"].tolist() == [0, 5]
    # A later forward must not replace the input used for the reference loss.
    model.lm_head(hidden + 1)
    torch.testing.assert_close(torch.load(tmp_path / "head_input_rank0.pt", weights_only=True)["hidden"], hidden[0])


def test_capture_rejects_ambiguous_multi_sample_microbatch(tmp_path):
    recorder = BenchmarkRecorder(tmp_path, 2, 0)
    recorder.capture_head_input(SimpleNamespace(engine=SimpleNamespace(module=_Model())))
    data = TensorDict({"input_ids": torch.ones(2, 5, dtype=torch.long),
                       "loss_mask": torch.ones(2, 5, dtype=torch.long)}, [])
    with pytest.raises(ValueError, match="one rank-local sample"):
        recorder.before_batch(data=data)
