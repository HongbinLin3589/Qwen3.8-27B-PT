"""Check BOLT RNG initialization and native multi-rank sampler seeding on CPU."""

import random
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from qwenpt.train.bolt.entrypoint import main


@pytest.fixture
def native_trainer_stub(monkeypatch):
    """Keep native loaders and real LoRA initialization without building a GPU model."""
    from peft import LoraConfig
    from peft.tuners.lora.layer import Linear
    from verl.trainer.sft_trainer import SFTTrainer
    from verl.utils import device, distributed

    captured = SimpleNamespace(trainers=[], events=[], rank=0)
    monkeypatch.setattr(device, "auto_set_device", lambda config: None)
    monkeypatch.setattr(distributed, "initialize_global_process_group", lambda: captured.events.append("initialize"))
    monkeypatch.setattr(distributed, "destroy_global_process_group", lambda: captured.events.append("destroy"))

    def initialize(self, config):
        captured.events.append("trainer")
        self.config = config
        self.initial_random = (random.random(), np.random.random(), torch.rand(4))
        adapter = Linear(torch.nn.Linear(6, 4, bias=False), "default", LoraConfig(r=2, lora_alpha=4), r=2, lora_alpha=4)
        self.initial_lora = adapter.lora_A["default"].weight.detach().clone()
        self.engine = SimpleNamespace(get_data_parallel_rank=lambda: captured.rank, get_data_parallel_size=lambda: 2)
        self.train_dataset, self.val_dataset = list(range(32)), list(range(16))
        self._build_dataloader()
        captured.trainers.append(self)

    monkeypatch.setattr(SFTTrainer, "__init__", initialize)
    monkeypatch.setattr(SFTTrainer, "fit", lambda self: captured.events.append("fit"))
    return captured


def config(seed):
    """Return the minimal CPU config required by the actual BOLT entry point."""
    return OmegaConf.create({
        "trainer": {"seed": seed, "device": "cpu"},
        "engine": {"seed": 42, "full_determinism": False},
        "data": {
            "custom_cls": {"path": "pkg://qwenpt.train.bolt.dataset", "name": "QwenBOLTDataset"},
            "train_batch_size": 8, "pad_mode": "no_padding", "num_workers": 0,
        },
    })


def test_rngs_and_lora_initialization_repeat_with_the_same_seed(native_trainer_stub):
    """Seed all CPU RNGs before real model/LoRA parameter initialization."""
    for seed in (20261020, 20261020, 20261021):
        settings = config(seed)
        main.__wrapped__(settings)
        assert settings.engine.seed == settings.trainer.seed == seed
        assert settings.engine.full_determinism is False
    first, repeat, different = native_trainer_stub.trainers
    assert first.initial_random[:2] == repeat.initial_random[:2]
    assert first.initial_random[:2] != different.initial_random[:2]
    torch.testing.assert_close(first.initial_random[2], repeat.initial_random[2], rtol=0, atol=0)
    torch.testing.assert_close(first.initial_lora, repeat.initial_lora, rtol=0, atol=0)
    assert not torch.equal(first.initial_random[2], different.initial_random[2])
    assert not torch.equal(first.initial_lora, different.initial_lora)
    assert list(first.train_sampler) == list(repeat.train_sampler)
    assert list(first.train_sampler) != list(different.train_sampler)
    assert native_trainer_stub.events == ["initialize", "trainer", "fit", "destroy"] * 3


def test_cuda_seeds_are_queued_without_enabling_deterministic_algorithms(monkeypatch, native_trainer_stub):
    """Exercise Transformers CUDA seeding safely on CPU and preserve kernel policy."""
    cuda_seeds, deterministic_calls = [], []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", cuda_seeds.append)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *args, **kwargs: deterministic_calls.append(args))
    main.__wrapped__(config(20261020))
    assert cuda_seeds and set(cuda_seeds) == {20261020}
    assert not deterministic_calls


def test_multi_rank_sampling_repeats_and_changes_with_seed_and_epoch(native_trainer_stub):
    """Retain disjoint complete rank shards and native per-epoch shuffling."""
    orders = {}
    for seed in (20261020, 20261021):
        for rank in (0, 1):
            native_trainer_stub.rank = rank
            main.__wrapped__(config(seed))
            trainer = native_trainer_stub.trainers[-1]
            assert trainer.train_sampler.seed == trainer.val_sampler.seed == seed
            for epoch in (0, 1):
                trainer.train_sampler.set_epoch(epoch)
                order = list(trainer.train_sampler)
                assert order == list(trainer.train_sampler)
                orders[seed, rank, epoch] = order
            assert list(trainer.val_sampler) == list(range(rank, 16, 2))
    for seed in (20261020, 20261021):
        for epoch in (0, 1):
            first, second = (orders[seed, rank, epoch] for rank in (0, 1))
            assert not set(first) & set(second)
            assert sorted(first + second) == list(range(32))
        assert orders[seed, 0, 0] != orders[seed, 0, 1]
    assert orders[20261020, 0, 0] != orders[20261021, 0, 0]
