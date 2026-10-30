"""Check SFT seeding and wall timing without model or GPU initialization."""

import json
from pathlib import Path
import random
from types import MethodType, SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from qwenpt.train.sft.entrypoint import main, run_sft


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


@pytest.fixture
def progress_stub(monkeypatch, native_trainer_stub):
    """Record runtime auditing callbacks without file I/O or source-row fixtures."""
    from qwenpt.train.sft import progress as sft_progress
    from qwenpt.train.sft import sampler
    from verl.utils import tensordict_utils as tu

    recorded = SimpleNamespace(before_data=[], after_outputs=[], failed=[], statuses=[], errors={})
    events = native_trainer_stub.events

    def before_batch(data):
        events.append(("before", len(native_trainer_stub.calls)))
        recorded.before_data.append(data)
        if "before" in recorded.errors:
            raise recorded.errors["before"]

    def after_batch(output, elapsed):
        events.append(("after", len(native_trainer_stub.calls)))
        recorded.after_outputs.append(output)
        if output is not None:
            assert tu.get(output, "metrics")["timing/train_step_seconds"] == elapsed
        if "after" in recorded.errors:
            raise recorded.errors["after"]

    def fail(error):
        events.append("fail")
        recorded.failed.append(error)
        if "fail" in recorded.errors:
            raise recorded.errors["fail"]

    def finish(status):
        events.append("finish")
        recorded.statuses.append(status)
        if "finish" in recorded.errors:
            raise recorded.errors["finish"]

    recorded.before_batch = before_batch
    recorded.after_batch = after_batch
    recorded.fail = fail
    recorded.finish = finish

    def initialize(trainer):
        events.append("progress")
        recorded.trainer = trainer
        return recorded

    monkeypatch.setattr(sft_progress, "SFTProgressRecorder", initialize)
    monkeypatch.setattr(sampler, "build_length_bucket_dataloader", lambda *args, **kwargs: (None, None))
    return recorded


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
    monkeypatch.setattr("qwenpt.train.sft.entrypoint.perf_counter", lambda: next(ticks))
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
        "timing_scope": "Host wall time around native train_batch; no explicit per-step CUDA synchronization",
    }


def test_cuda_step_timing_does_not_force_synchronization(monkeypatch, native_trainer_stub, tmp_path):
    """Keep production timing from synchronizing CUDA on every batch."""
    synchronized = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synchronized.append(len(native_trainer_stub.calls)))
    config = _config(20261020, tmp_path)
    config.trainer.device = "cuda"
    run_sft(config)
    assert synchronized == []


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



@pytest.mark.parametrize("option,value", [
    ("model.use_fused_kernels", True),
    ("model.use_liger", True),
    ("model.enable_activation_offload", True),
    ("engine.offload_policy", True),
    ("engine.param_offload", True),
    ("engine.optimizer_offload", True),
    ("trainer.output_head_backend", "cce_exact"),
    ("trainer.ce_backend", "chunked"),
    ("trainer.ce_chunk_size", 128),
    ("trainer.output_head_chunk_size", 256),
    ("trainer.benchmark_steps", 2),
    ("trainer.capture_head_input", True),
    ("trainer.cpu_offload_pin_memory", False),
    ("trainer.checkpoint_activation_offload", True),
    ("trainer.checkpoint_use_reentrant", True),
    ("trainer.checkpoint_sublayers", True),
    ("trainer.checkpoint_layernorms", True),
    ("trainer.mlp_chunk_tokens", 8192),
    ("trainer.fsdp2_backward_prefetch", "current"),
])
def test_production_rejects_archived_options_before_initialization(
    native_trainer_stub, tmp_path, option, value,
):
    """Prevent an old experimental config from silently changing its behavior."""
    config = _config(20261020, tmp_path)
    OmegaConf.update(config, option, value, force_add=True)
    for entrypoint in (run_sft, main.__wrapped__):
        with pytest.raises(ValueError, match="unsupported experimental options") as error:
            entrypoint(config)
        assert option in str(error.value)
    assert native_trainer_stub.events == []
    assert not (tmp_path / "timing_summary.json").exists()


def test_native_config_preserves_regular_gradient_checkpointing(native_trainer_stub, tmp_path):
    """Native HF checkpointing remains available after extra checkpoint scopes move."""
    config = _config(20261020, tmp_path)
    config.model = {
        "use_fused_kernels": False,
        "use_liger": False,
        "enable_activation_offload": False,
        "enable_gradient_checkpointing": True,
    }
    config.engine.param_offload = False
    config.engine.optimizer_offload = False
    config.trainer.output_head_backend = "native"
    config.trainer.ce_backend = "native"
    config.trainer.checkpoint_sublayers = False
    config.trainer.checkpoint_layernorms = False
    run_sft(config)
    assert native_trainer_stub.trainers[0].config is config
    assert config.model.enable_gradient_checkpointing is True
    assert native_trainer_stub.events == ["initialize", "trainer", "fit", "destroy"]


def test_progress_callbacks_surround_native_steps(native_trainer_stub, progress_stub, tmp_path):
    """Audit the native outputs after timing metrics are attached, then close before teardown."""
    config = _config(20261020, tmp_path)
    config.trainer.progress_tracking = True
    config.data.length_bucketing = True
    run_sft(config)
    assert progress_stub.before_data == [0, 1, 2]
    assert all(original is audited for original, audited in zip(
        native_trainer_stub.outputs, progress_stub.after_outputs, strict=True,
    ))
    assert progress_stub.failed == []
    assert progress_stub.statuses == ["completed"]
    assert native_trainer_stub.events == [
        "initialize", "trainer", "progress", "fit",
        ("before", 0), ("after", 1), ("before", 1), ("after", 2), ("before", 2), ("after", 3),
        "finish", "destroy",
    ]


@pytest.mark.parametrize("phase,completed_steps", [("before", 0), ("native", 0), ("after", 1)])
def test_progress_step_failure_preserves_count_and_original_exception(
    monkeypatch, native_trainer_stub, progress_stub, tmp_path, phase, completed_steps,
):
    """Only native calls that return contribute timing, including an audit failure afterward."""
    from verl.trainer.sft_trainer import SFTTrainer

    failure = RuntimeError(f"{phase} failed")
    config = _config(20261020, tmp_path)
    config.trainer.progress_tracking = True
    config.data.length_bucketing = True
    if phase == "native":
        initialize = SFTTrainer.__init__

        def initialize_with_failure(self, config):
            initialize(self, config)

            def train_batch(*args, **kwargs):
                native_trainer_stub.calls.append((args, kwargs))
                raise failure

            self.training_client.train_batch = train_batch

        monkeypatch.setattr(SFTTrainer, "__init__", initialize_with_failure)
    else:
        progress_stub.errors[phase] = failure

    with pytest.raises(RuntimeError) as raised:
        run_sft(config)
    assert raised.value is failure
    assert progress_stub.failed == [failure]
    assert progress_stub.statuses == ["failed"]
    assert native_trainer_stub.events[-3:] == ["fail", "finish", "destroy"]
    assert len(native_trainer_stub.calls) == (0 if phase == "before" else 1)
    assert len(progress_stub.after_outputs) == (1 if phase == "after" else 0)
    summary = json.loads((tmp_path / "timing_summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["completed_train_steps"] == completed_steps
    assert (summary["train_step_seconds_mean"] is None) == (completed_steps == 0)


@pytest.mark.parametrize("failures", [
    ("finish",), ("destroy",), ("summary",), ("finish", "destroy"),
    ("finish", "destroy", "summary"), ("fit",), ("fit", "fail"),
    ("fit", "finish", "destroy", "summary"), ("fit", "fail", "finish", "destroy", "summary"),
])
def test_cleanup_keeps_order_status_and_exception_precedence(
    monkeypatch, native_trainer_stub, progress_stub, tmp_path, failures,
):
    """Always attempt later cleanup and retain its existing exception precedence and chain."""
    from verl.utils import distributed

    errors = {phase: RuntimeError(f"{phase} failed") for phase in failures}
    config = _config(20261020, tmp_path)
    config.trainer.progress_tracking = True
    config.data.length_bucketing = True
    progress_stub.errors.update(errors)

    def during_fit():
        if "fit" in errors:
            raise errors["fit"]

    def destroy():
        native_trainer_stub.events.append("destroy")
        if "destroy" in errors:
            raise errors["destroy"]

    write_text = Path.write_text

    def write_summary(path, text, *args, **kwargs):
        if path.name == "timing_summary.json":
            native_trainer_stub.events.append("summary")
            if "summary" in errors:
                raise errors["summary"]
        return write_text(path, text, *args, **kwargs)

    native_trainer_stub.during_fit = during_fit
    monkeypatch.setattr(distributed, "destroy_global_process_group", destroy)
    monkeypatch.setattr(Path, "write_text", write_summary)
    with pytest.raises(RuntimeError) as raised:
        run_sft(config)
    exception = raised.value
    for phase in reversed(failures):
        assert exception is errors[phase]
        exception = exception.__context__
    assert exception is None
    assert native_trainer_stub.events[-3:] == ["finish", "destroy", "summary"]
    expected_status = "failed" if "fit" in errors else "completed"
    assert progress_stub.statuses == [expected_status]
    assert progress_stub.failed == ([errors["fit"]] if "fit" in errors else [])
    if "fit" in errors:
        assert native_trainer_stub.events[-4] == "fail"
    if "summary" not in errors:
        summary = json.loads((tmp_path / "timing_summary.json").read_text())
        assert summary["status"] == expected_status
        assert summary["completed_train_steps"] == (0 if "fit" in errors else 3)


@pytest.mark.parametrize("return_none", [False, True])
def test_timing_wrapper_preserves_bound_method_arguments_and_return_identity(
    monkeypatch, native_trainer_stub, progress_stub, tmp_path, return_none,
):
    """Forward positional and keyword objects once without rebinding the native client."""
    from verl.trainer.sft_trainer import SFTTrainer
    from verl.utils import tensordict_utils as tu

    initialize = SFTTrainer.__init__
    data, extra, option = object(), object(), object()
    observed, returned = [], []
    output = None if return_none else tu.get_tensordict(
        tensor_dict={}, non_tensor_dict={"metrics": {"loss": 1.25}},
    )
    client = SimpleNamespace()

    def native_train_batch(self, *args, **kwargs):
        observed.append((self, args, kwargs))
        return output

    client.train_batch = MethodType(native_train_batch, client)

    def initialize_bound_client(self, config):
        initialize(self, config)
        self.training_client = client

    def fit(self):
        returned.append(self.training_client.train_batch(data, extra, option=option))
        returned.append(self.training_client.train_batch(data=data, option=option))

    monkeypatch.setattr(SFTTrainer, "__init__", initialize_bound_client)
    monkeypatch.setattr(SFTTrainer, "fit", fit)
    config = _config(20261020, tmp_path)
    config.trainer.progress_tracking = True
    config.data.length_bucketing = True
    run_sft(config)
    assert observed == [(client, (data, extra), {"option": option}), (client, (), {"data": data, "option": option})]
    assert len(returned) == 2 and all(result is output for result in returned)
    assert all(result is output for result in progress_stub.after_outputs)
    assert len(progress_stub.before_data) == 2 and all(result is data for result in progress_stub.before_data)
