"""Check SFT seeding and wall timing without model or GPU initialization."""

import json
import random
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from legacy.sft.entrypoint import main, run_sft


@pytest.mark.parametrize("padding,attention", [(False, "flash_attention_2"), (True, "sdpa")])
def test_sequence_parallel_rejects_incorrect_attention_before_initializing(padding, attention):
    config = OmegaConf.create({
        "engine": {"ulysses_sequence_parallel_size": 2},
        "model": {"use_remove_padding": padding, "override_config": {"attn_implementation": attention}},
    })
    with pytest.raises(ValueError, match="Sequence parallel SFT requires"):
        run_sft(config)


@pytest.fixture
def native_trainer_stub(monkeypatch):
    """Keep verl's real dataloader while replacing its model and process group."""
    from tensordict import capture_non_tensor_stack
    from verl.trainer.sft_trainer import SFTTrainer
    from verl.utils import device, distributed, tensordict_utils as tu

    captured = SimpleNamespace(
        trainers=[], events=[], outputs=[], returned=[], calls=[], fail_fit=False, fail_init=False,
        during_init=lambda: None, during_fit=lambda: None,
    )
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr(device, "auto_set_device", lambda config: None)
    monkeypatch.setattr(distributed, "initialize_global_process_group", lambda: captured.events.append("initialize"))
    monkeypatch.setattr(distributed, "destroy_global_process_group", lambda: captured.events.append("destroy"))

    def initialize(self, config):
        captured.events.append("trainer")
        captured.during_init()
        if captured.fail_init:
            raise RuntimeError("initialization failed")
        self.config = config
        self.initial_random = (random.random(), np.random.random(), torch.rand(4))
        self.engine = SimpleNamespace(get_data_parallel_rank=lambda: 0, get_data_parallel_size=lambda: 2)
        self.train_dataset = list(range(32))
        self.val_dataset = list(range(8))
        self._build_dataloader()
        self.training_client = SimpleNamespace(train_batch=train_batch)
        captured.trainers.append(self)

    def train_batch(*args, **kwargs):
        captured.calls.append((args, kwargs))
        output = tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": {"loss": 1.25, "lr": 1e-5}})
        captured.outputs.append(output)
        return output

    def fit(self):
        captured.events.append("fit")
        captured.during_fit()
        self.capture_non_tensor_stack = capture_non_tensor_stack()
        for data in range(3):
            captured.returned.append(self.training_client.train_batch(data=data))
        if captured.fail_fit:
            raise RuntimeError("training failed")

    monkeypatch.setattr(SFTTrainer, "__init__", initialize)
    monkeypatch.setattr(SFTTrainer, "fit", fit)
    return captured


def _config(seed, run_dir):
    return OmegaConf.create({
        "trainer": {"seed": seed, "device": "cpu", "default_local_dir": str(run_dir)},
        "engine": {"strategy": "fsdp2", "offload_policy": False},
        "data": {"train_batch_size": 8, "pad_mode": "no_padding", "num_workers": 0},
    })


def test_seed_repeats_initial_rngs_and_native_sampler_order(native_trainer_stub, tmp_path):
    """Seed RNGs before trainer initialization and control epoch shuffling."""
    for seed in (20261020, 20261020, 20261021):
        run_sft(_config(seed, tmp_path))
    first, repeat, different = native_trainer_stub.trainers
    assert first.initial_random[:2] == repeat.initial_random[:2]
    assert first.initial_random[:2] != different.initial_random[:2]
    torch.testing.assert_close(first.initial_random[2], repeat.initial_random[2], rtol=0, atol=0)
    assert not torch.equal(first.initial_random[2], different.initial_random[2])
    assert first.train_sampler.seed == first.val_sampler.seed == 20261020
    assert list(first.train_sampler) == list(repeat.train_sampler)
    assert list(first.train_sampler) != list(different.train_sampler)
    epoch_zero = list(first.train_sampler)
    first.train_sampler.set_epoch(1)
    repeat.train_sampler.set_epoch(1)
    assert list(first.train_sampler) == list(repeat.train_sampler)
    assert list(first.train_sampler) != epoch_zero
    assert native_trainer_stub.events == ["initialize", "trainer", "fit", "destroy"] * 3


def test_seed_covers_cuda_without_enabling_deterministic_algorithms(monkeypatch, native_trainer_stub, tmp_path):
    """Queue CUDA RNG seeds safely on CPU without changing kernel policies."""
    seeds, deterministic_calls = [], []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", seeds.append)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *args, **kwargs: deterministic_calls.append(args))
    run_sft(_config(20261020, tmp_path))
    assert seeds and set(seeds) == {20261020}
    assert not deterministic_calls


def test_seeded_native_entrypoint_preserves_media_context(native_trainer_stub, tmp_path):
    """The seeded native trainer still runs inside the TensorDict compatibility context."""
    from tensordict import capture_non_tensor_stack, set_capture_non_tensor_stack
    from verl.trainer.sft_trainer import SFTTrainer

    with set_capture_non_tensor_stack(True):
        main.__wrapped__(_config(20261020, tmp_path))
        assert capture_non_tensor_stack() is True
    trainer = native_trainer_stub.trainers[0]
    assert isinstance(trainer, SFTTrainer)
    assert trainer.capture_non_tensor_stack is False
    assert trainer.train_sampler.seed == 20261020


def test_seeded_runner_releases_process_group_on_failure(native_trainer_stub, tmp_path):
    """A native training failure must still release its distributed process group."""
    native_trainer_stub.fail_fit = True
    with pytest.raises(RuntimeError, match="training failed"):
        run_sft(_config(20261020, tmp_path))
    assert native_trainer_stub.events[-1] == "destroy"
    summary = json.loads((tmp_path / "timing_summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["completed_train_steps"] == 3
    assert summary["fit_seconds"] >= summary["train_steps_seconds"]


def test_native_step_outputs_and_wall_timing_are_preserved(monkeypatch, native_trainer_stub, tmp_path):
    """Add one metric to each original output and report the full native fit wall time."""
    from verl.utils import tensordict_utils as tu

    ticks = iter(range(10))
    monkeypatch.setattr("legacy.sft.entrypoint.perf_counter", lambda: next(ticks))
    run_sft(_config(20261020, tmp_path))
    assert native_trainer_stub.calls == [((), {"data": data}) for data in range(3)]
    assert all(original is returned for original, returned in zip(native_trainer_stub.outputs, native_trainer_stub.returned))
    for output in native_trainer_stub.outputs:
        assert tu.get(output, "metrics") == {"loss": 1.25, "lr": 1e-5, "timing/train_step_seconds": 1}
    assert json.loads((tmp_path / "timing_summary.json").read_text()) == {
        "status": "completed",
        "completed_train_steps": 3,
        "train_steps_seconds": 3,
        "train_step_seconds_mean": 1,
        "initialization_seconds": 1,
        "fit_seconds": 7,
        "run_seconds": 9,
    }


def test_cuda_step_timing_synchronizes_both_boundaries(monkeypatch, native_trainer_stub, tmp_path):
    """Synchronize each CUDA step without using a GPU in this test."""
    synchronized = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synchronized.append(len(native_trainer_stub.calls)))
    config = _config(20261020, tmp_path)
    config.trainer.device = "cuda"
    run_sft(config)
    assert synchronized == [0, 1, 1, 2, 2, 3]


def test_initialization_failure_still_records_timing(native_trainer_stub, tmp_path):
    """Record initialization failure without claiming that fit completed."""
    native_trainer_stub.fail_init = True
    with pytest.raises(RuntimeError, match="initialization failed"):
        run_sft(_config(20261020, tmp_path))
    summary = json.loads((tmp_path / "timing_summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["completed_train_steps"] == 0
    assert summary["initialization_seconds"] is summary["fit_seconds"] is None
    assert summary["run_seconds"] >= 0
    assert native_trainer_stub.events[-1] == "destroy"


def test_only_rank_zero_writes_timing_summary(monkeypatch, native_trainer_stub, tmp_path):
    """Prevent distributed workers from overwriting the same summary file."""
    monkeypatch.setenv("RANK", "1")
    run_sft(_config(20261020, tmp_path))
    assert not (tmp_path / "timing_summary.json").exists()


def test_benchmark_stops_before_extra_native_step(monkeypatch, native_trainer_stub, tmp_path):
    """Stop the short run without changing the full native training schedule."""
    from verl.trainer.sft_trainer import SFTTrainer
    from verl.utils import tensordict_utils as tu

    def fit(self):
        self.total_training_steps = 2250
        self.steps_per_epoch = 1125
        self.engine.module = torch.nn.Linear(2, 2)
        self.engine.lr_scheduler = SimpleNamespace(last_epoch=0)
        for _ in range(3):
            data = tu.get_tensordict(tensor_dict={"input_ids": torch.tensor([[1, 2]])})
            self.training_client.train_batch(data=data)

    monkeypatch.setattr(SFTTrainer, "fit", fit)
    config = _config(20261020, tmp_path)
    config.trainer.benchmark_steps = 2
    run_sft(config)
    assert len(native_trainer_stub.calls) == 2
    assert json.loads((tmp_path / "timing_summary.json").read_text())["status"] == "benchmark_completed"
    manifest = json.loads((tmp_path / "benchmark_manifest_rank0.json").read_text())
    assert manifest["completed_steps"] == 2
    assert manifest["native_total_training_steps"] == 2250
    assert manifest["native_steps_per_epoch"] == 1125
    assert manifest["parameter_sampling_error"] is None


@pytest.mark.parametrize("fail_init", [False, True])
def test_unpinned_cpu_offload_is_scoped_to_trainer_creation(native_trainer_stub, tmp_path, fail_init):
    """Return native unpinned policies during initialization and always restore the factory."""
    from torch.distributed.fsdp import CPUOffloadPolicy
    from verl.workers.engine.fsdp import transformer_impl

    original_factory = transformer_impl.CPUOffloadPolicy
    observed = []

    def during_init():
        observed.append("initialize")
        policy = transformer_impl.CPUOffloadPolicy(pin_memory=True)
        assert isinstance(policy, CPUOffloadPolicy)
        assert policy.pin_memory is False
        assert torch.distributed.fsdp.CPUOffloadPolicy is original_factory

    def during_fit():
        observed.append("fit")
        assert transformer_impl.CPUOffloadPolicy is original_factory

    native_trainer_stub.during_init = during_init
    native_trainer_stub.during_fit = during_fit
    native_trainer_stub.fail_init = fail_init
    config = _config(20261020, tmp_path)
    config.trainer.cpu_offload_pin_memory = False
    config.engine.offload_policy = True
    if fail_init:
        with pytest.raises(RuntimeError, match="initialization failed"):
            run_sft(config)
    else:
        run_sft(config)
    assert transformer_impl.CPUOffloadPolicy is original_factory
    assert observed == (["initialize"] if fail_init else ["initialize", "fit"])
    assert native_trainer_stub.events[-1] == "destroy"


@pytest.mark.parametrize("pin_memory, strategy, offload", [
    (None, "fsdp2", True),
    (True, "fsdp2", True),
    (False, "fsdp2", False),
    (False, "fsdp", True),
])
def test_cpu_offload_factory_is_unchanged_without_opt_in(native_trainer_stub, tmp_path, pin_memory, strategy, offload):
    """Leave the native factory untouched by default and outside FSDP2 CPU offload."""
    from verl.workers.engine.fsdp import transformer_impl

    original_factory = transformer_impl.CPUOffloadPolicy

    def check_factory():
        assert transformer_impl.CPUOffloadPolicy is original_factory
        assert transformer_impl.CPUOffloadPolicy(pin_memory=True).pin_memory is True

    native_trainer_stub.during_init = check_factory
    native_trainer_stub.during_fit = check_factory
    config = _config(20261020, tmp_path)
    config.engine.strategy = strategy
    config.engine.offload_policy = offload
    if pin_memory is not None:
        config.trainer.cpu_offload_pin_memory = pin_memory
    run_sft(config)
    assert transformer_impl.CPUOffloadPolicy is original_factory
