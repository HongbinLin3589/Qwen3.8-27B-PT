"""Validate paired short-run artifacts without a GPU or process group."""

import json
from types import SimpleNamespace

import pytest
import torch
from tensordict import set_capture_non_tensor_stack
from verl.utils import tensordict_utils as tu

from legacy.sft.utils.sft_benchmark import BenchmarkFinished, BenchmarkRecorder


@pytest.fixture(autouse=True)
def no_collectives(monkeypatch):
    """The recorder must not add rank synchronization to native training."""
    def fail(*args, **kwargs):
        raise AssertionError("Benchmark recorder performed a collective")

    for name in ("all_reduce", "all_gather", "all_gather_into_tensor", "reduce_scatter_tensor", "broadcast", "barrier"):
        monkeypatch.setattr(torch.distributed, name, fail)
    with set_capture_non_tensor_stack(False):
        yield


def _batch(samples=None):
    samples = samples if samples is not None else [[1, 2], [3, 4, 5]]
    nested = lambda values: torch.nested.nested_tensor(values, layout=torch.jagged)
    return tu.get_tensordict(
        tensor_dict={
            "input_ids": nested([torch.tensor(sample) for sample in samples]),
            "loss_mask": nested([torch.ones(len(sample)) for sample in samples]),
            "position_ids": nested([torch.arange(len(sample)) for sample in samples]),
        },
        non_tensor_dict={"global_token_num": [2, 3, 4, 5], "irrelevant_timing": 123.0},
    )


def _trainer():
    module = torch.nn.Linear(4, 3)
    module.register_parameter("strided", torch.nn.Parameter(torch.arange(12.).reshape(3, 4).t()))
    return SimpleNamespace(
        engine=SimpleNamespace(module=module, lr_scheduler=SimpleNamespace(last_epoch=1)),
        total_training_steps=2250,
        steps_per_epoch=1125,
    )


def _output(loss=.3):
    return tu.get_tensordict({}, {"metrics": {"loss": loss, "grad_norm": .9, "lr": 1e-5 / 225, "mfu": .025}})


def test_jagged_fingerprints_repeat_and_detect_input_changes(tmp_path):
    """Fresh symbolic jagged dimensions do not change hashes of identical batches."""
    a, b, changed = [BenchmarkRecorder(tmp_path / name, 1, 0) for name in ("a", "b", "changed")]
    a.before_batch(data=_batch())
    b.before_batch(_batch())
    changed.before_batch(data=_batch([[1, 9], [3, 4, 5]]))
    assert a._pending == b._pending
    assert a._pending["input_sha256"] != changed._pending["input_sha256"]
    assert a._pending["field_sha256"]["loss_mask"] == changed._pending["field_sha256"]["loss_mask"]


def test_fingerprint_includes_sequence_boundaries_and_ignores_metadata(tmp_path):
    """Equal flat tokens with different query boundaries must compare differently."""
    a, b, changed = [BenchmarkRecorder(tmp_path / name, 1, 0) for name in ("a", "b", "changed")]
    data = _batch()
    a.before_batch(data=data)
    tu.assign_non_tensor(data, irrelevant_timing=456.0)
    b.before_batch(data=data)
    changed.before_batch(data=_batch([[1], [2, 3, 4, 5]]))
    assert a._pending["input_sha256"] == b._pending["input_sha256"]
    assert a._pending["input_sha256"] != changed._pending["input_sha256"]


def test_limit_stops_before_extra_native_step_and_records_global_tokens(tmp_path):
    """Exactly N complete calls preserve original metrics and full-run scheduling."""
    recorder = BenchmarkRecorder(tmp_path, 2, 0)
    trainer, output = _trainer(), _output()
    native_calls = []

    def wrapped_native():
        recorder.before_batch(data=_batch())
        native_calls.append(1)
        recorder.after_batch(output, 1.2, trainer)

    wrapped_native()
    wrapped_native()
    with pytest.raises(BenchmarkFinished):
        wrapped_native()
    assert len(native_calls) == 2
    assert set(tu.get(output, "metrics")) == {"loss", "grad_norm", "lr", "mfu"}
    records = [json.loads(line) for line in recorder.steps_path.read_text().splitlines()]
    assert [row["step"] for row in records] == [1, 2]
    assert records[0]["global_tokens"] == 14
    assert records[0]["local_sequence_lengths"] == [2, 3]
    assert records[0]["lr"] == 1e-5 / 225
    manifest = recorder.finish(trainer, "benchmark_completed")
    assert manifest["completed_steps"] == 2
    assert manifest["native_total_training_steps"] == 2250
    assert manifest["native_steps_per_epoch"] == 1125
    assert manifest["scheduler_last_epoch"] == 1
    assert manifest["all_recorded_metrics_finite"]
    assert manifest["process_placement"]["rss_bytes"] > 0
    assert manifest["process_placement"]["cpu_affinity"]
    assert manifest["process_placement"]["numa_resident_pages"]


def test_parameter_samples_support_strided_parameters_and_local_shards(tmp_path):
    """Only the selected local elements are read, including noncontiguous shards."""
    trainer = _trainer()
    recorder = BenchmarkRecorder(tmp_path / "dense", 1, 0)
    manifest = recorder.finish(trainer, "benchmark_completed")
    samples = json.loads((recorder.run_dir / manifest["parameter_samples_file"]).read_text())
    for name, values in samples.items():
        parameter = dict(trainer.engine.module.named_parameters())[name].detach().reshape(-1)
        assert values["values"] == parameter[values["flat_positions"]].tolist()
        assert len(values["values"]) <= 9

    class LocalShard:
        shape = torch.Size([8, 3])

        def to_local(self):
            return torch.arange(12.).reshape(3, 4).t()

    trainer.engine.module = SimpleNamespace(named_parameters=lambda: [("sharded", LocalShard())])
    recorder = BenchmarkRecorder(tmp_path / "local", 1, 1)
    manifest = recorder.finish(trainer, "benchmark_completed")
    samples = json.loads((recorder.run_dir / manifest["parameter_samples_file"]).read_text())
    assert samples["sharded"]["global_shape"] == [8, 3]
    assert samples["sharded"]["local_shape"] == [4, 3]


def test_initialization_and_sampling_failures_keep_a_manifest(tmp_path):
    """Diagnostic sampling cannot mask the earlier native training exception."""
    recorder = BenchmarkRecorder(tmp_path / "initialization", 1, 0)
    manifest = recorder.finish(None, "failed")
    assert manifest["status"] == "failed"
    assert manifest["completed_steps"] == 0
    assert manifest["parameter_samples_file"] is None
    assert manifest["native_total_training_steps"] is None

    def broken_parameters():
        raise RuntimeError("sampling failed")

    trainer = _trainer()
    trainer.engine.module.named_parameters = broken_parameters
    recorder = BenchmarkRecorder(tmp_path / "sampling", 1, 0)
    manifest = recorder.finish(trainer, "benchmark_completed")
    assert manifest["parameter_sampling_error"] == "RuntimeError: sampling failed"
    assert manifest["parameter_samples_file"] is None


def test_failed_cleanup_does_not_synchronize_parameter_samples(monkeypatch, tmp_path):
    recorder = BenchmarkRecorder(tmp_path, 1, 0)
    monkeypatch.setattr(recorder, "_parameter_samples", lambda _module: pytest.fail("Failure cleanup sampled parameters"))
    manifest = recorder.finish(_trainer(), "failed")
    assert manifest["parameter_samples_file"] is None
    assert manifest["parameter_sampling_error"] == "Skipped parameter sampling after native training failure"


def test_runtime_diagnostics_do_not_change_rng_or_initialize_cuda(monkeypatch, tmp_path):
    """Inspect the numerical runtime only when the diagnostic switch is enabled."""
    monkeypatch.setenv("QWENPT_SFT_RUNTIME_DIAGNOSTICS", "1")
    trainer = _trainer()
    before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.is_initialized()
    manifest = BenchmarkRecorder(tmp_path, 1, 0).finish(trainer, "failed")
    runtime = manifest["runtime_diagnostics"]
    assert "collection_error" not in runtime
    assert runtime["torch_num_threads"] == torch.get_num_threads()
    assert runtime["torch_num_interop_threads"] == torch.get_num_interop_threads()
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.cuda.is_initialized() == cuda_before


def test_nonfinite_metrics_are_saved_and_rejected(tmp_path):
    """A nonfinite native update is recorded before benchmark termination fails."""
    recorder, trainer = BenchmarkRecorder(tmp_path, 1, 0), _trainer()
    recorder.before_batch(data=_batch())
    with pytest.raises(FloatingPointError, match="loss"):
        recorder.after_batch(_output(float("nan")), 1.0, trainer)
    record = json.loads(recorder.steps_path.read_text())
    assert record["loss"] is None
    assert record["non_finite_metrics"] == ["loss"]
    assert not recorder.finish(trainer, "failed")["all_recorded_metrics_finite"]


def test_dense_bfloat16_and_optional_labels_can_be_fingerprinted(tmp_path):
    """Byte views support BF16 tensors without NumPy's unsupported BF16 conversion."""
    data = tu.get_tensordict(
        {"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[2, 3]]),
         "loss_mask": torch.tensor([[0., 1.]], dtype=torch.bfloat16)},
        {"global_token_num": [2, 3]},
    )
    recorder = BenchmarkRecorder(tmp_path, 1, 0)
    recorder.before_batch(data)
    assert set(recorder._pending["field_sha256"]) == {"input_ids", "labels", "loss_mask"}


@pytest.mark.parametrize("steps", [0, -1])
def test_nonpositive_limits_are_rejected(tmp_path, steps):
    with pytest.raises(ValueError, match="positive"):
        BenchmarkRecorder(tmp_path, steps, 0)
