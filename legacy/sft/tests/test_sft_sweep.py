"""Check isolated sweep commands and measured workload accounting without CUDA."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from legacy.sft.utils import sft_sweep as module


def _config(tmp_path):
    repo = tmp_path / "repo"
    (repo / "legacy/sft").mkdir(parents=True)
    (repo / "legacy/sft/entrypoint.py").write_text("")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    train = tmp_path / "pool.parquet"
    pq.write_table(pa.table({"sample_id": list(range(16))}), train)
    return module.SweepConfig(repo=repo, model=model, train_files=(train,), val_files=(),
                              output=tmp_path / "sweep", gpu_mask=(2, 3), global_batches=(4,),
                              micro_batches=(1,), warmup_steps=2, measure_steps=2)


def _records(directory: Path, arm: dict):
    directory.mkdir(parents=True)
    for rank in range(2):
        rows = []
        for step in range(1, arm["requested_steps"] + 1):
            lengths = ([2, 3] if rank == 0 else [4, 5]) if step == 3 else ([6, 7] if rank == 0 else [8, 9])
            elapsed = (2, 3)[rank] if step == 3 else (4, 1)[rank]
            digest = f"{rank * 100 + step:064x}"
            rows.append({"step": step, "elapsed_seconds": elapsed if step > 2 else 1000,
                         "input_sha256": digest, "sample_sha256": [digest, digest],
                         "local_sequence_lengths": lengths,
                         "global_tokens": 14 if step == 3 else 30,
                         "local_supervised_tokens": sum(lengths) // 2,
                         "peak_memory_allocated_bytes": (rank + 1) * 100 + step if step > 2 else 999999,
                         "peak_memory_reserved_bytes": (rank + 1) * 200 + step if step > 2 else 999999,
                         "non_finite_metrics": []})
        (directory / f"benchmark_steps_rank{rank}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows))
        (directory / f"benchmark_manifest_rank{rank}.json").write_text(json.dumps({
            "status": "benchmark_completed", "completed_steps": arm["requested_steps"]}))


def test_dry_run_freezes_pool_and_emits_isolated_native_commands(tmp_path, monkeypatch):
    config = replace(_config(tmp_path), output_head_backends=("native", "liger", "verl_triton", "cce_exact"))
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("Dry run launched a process"))
    report = module.run_sweep(config, dry_run=True)
    assert report["status"] == "dry_run"
    manifest = json.loads((config.output / "sweep_manifest.json").read_text())
    assert manifest["train_pool"]["rows"] == 16
    assert len(manifest["train_pool"]["files"][0]["sha256"]) == 64
    for arm in report["arms"]:
        options = dict(value.split("=", 1) for value in arm["command"] if "=" in value)
        assert arm["environment"]["CUDA_VISIBLE_DEVICES"] == "2,3"
        assert arm["command"][arm["command"].index("--module") + 1] == "legacy.sft.entrypoint"
        assert arm["environment"]["PYTHONPATH"].split(":") == [str(config.repo.resolve()), str(config.repo.resolve() / "src")]
        assert options["--nproc-per-node"] == "2"
        assert options["engine.offload_policy"] == "false"
        assert options["engine.model_dtype"] == '"float32"'
        assert options["engine.use_no_sync_for_gradient_accumulation"] == "false"
        assert options["model.enable_gradient_checkpointing"] == "true"
        assert options["model.use_remove_padding"] == "false"
        assert options["trainer.logger"] == "[]"
        assert options["data.val_files"] == "null"
        assert int(options["trainer.total_training_steps"]) > arm["requested_steps"]
        assert options["+trainer.benchmark_steps"] == "4"
        assert json.loads(options["model.use_fused_kernels"]) == (arm["output_head_backend"] != "native")
        expected = "triton" if arm["output_head_backend"] == "verl_triton" else "torch"
        assert json.loads(options["model.fused_kernel_options.impl_backend"]) == expected


def test_matched_windows_cover_identical_complete_pool_passes(tmp_path):
    config = replace(_config(tmp_path), mode="matched", global_batches=(4, 8), micro_batches=(1, 2))
    arms = module.build_arms(config, pool_rows=16)
    assert {arm["measured_samples_expected"] for arm in arms} == {16}
    assert {(arm["global_batch"], arm["warmup_steps"], arm["measure_steps"]) for arm in arms} == {
        (4, 4, 4), (8, 2, 2)}
    with pytest.raises(ValueError, match="divisible by every global batch"):
        module.build_arms(config, pool_rows=18)


def test_all_output_head_commands_compose_with_native_verl_configuration(tmp_path):
    from hydra import compose, initialize_config_module

    config = replace(_config(tmp_path), output_head_backends=module.OUTPUT_HEADS)
    for arm in module.build_arms(config, pool_rows=16):
        overrides = arm["command"][arm["command"].index("legacy.sft.entrypoint") + 1:]
        with initialize_config_module(config_module="verl.trainer.config", version_base=None):
            composed = compose(config_name="sft_trainer_engine", overrides=overrides)
        assert composed.trainer.output_head_backend == arm["output_head_backend"]
        assert composed.trainer.benchmark_steps == arm["requested_steps"]
        assert composed.data.val_files is None
        assert not composed.engine.offload_policy


def test_sequence_parallel_batch_contract_uses_dp_size(tmp_path):
    config = replace(_config(tmp_path), sequence_parallel_sizes=(2,), remove_padding=(True,),
                     attn_backends=("flash_attention_2",),
                     global_batches=(2,), micro_batches=(2,))
    arm = module.build_arms(config, pool_rows=16)[0]
    assert arm["sequence_parallel_size"] == 2
    assert arm["data_parallel_size"] == 1
    options = dict(value.split("=", 1) for value in arm["command"] if "=" in value)
    assert options["engine.fsdp_size"] == "2"
    assert options["engine.ulysses_sequence_parallel_size"] == "2"
    assert options["model.use_remove_padding"] == "true"


def test_dynamic_batch_contract_ignores_static_micro_size(tmp_path):
    config = replace(_config(tmp_path), gpu_mask=tuple(range(8)), global_batches=(8,),
                     micro_batches=(2,), dynamic=(True,))
    arm = module.build_arms(config, pool_rows=16)[0]
    assert arm["data_parallel_size"] == 8
    assert arm["dynamic"] is True
    with pytest.raises(ValueError, match="divisible"):
        module.build_arms(replace(config, dynamic=(False, True)), pool_rows=16)
    with pytest.raises(ValueError, match="divisible"):
        module.build_arms(replace(config, global_batches=(9,)), pool_rows=18)


def test_sequence_parallel_rejects_sdpa_without_communication(tmp_path):
    config = replace(_config(tmp_path), sequence_parallel_sizes=(2,), remove_padding=(True,))
    with pytest.raises(ValueError, match="requires flash_attention_2"):
        module.build_arms(config, pool_rows=16)


@pytest.mark.parametrize("sizes,padding,match", [
    ((2,), (False,), "requires remove_padding"),
    ((4,), (True,), "divide the GPU count"),
    ((3,), (True,), "must be 1, 2, 4 or 8"),
])
def test_invalid_sequence_parallel_topology_rejected(tmp_path, sizes, padding, match):
    with pytest.raises(ValueError, match=match):
        module.build_arms(replace(_config(tmp_path), sequence_parallel_sizes=sizes, remove_padding=padding), 16)


@pytest.mark.parametrize("change,match", [
    ({"gpu_mask": (2, 2)}, "distinct"),
    ({"global_batches": (3,)}, "divisible"),
    ({"micro_batches": (3,)}, "divisible"),
    ({"output_head_backends": ("cce",)}, "Output heads"),
    ({"timeout_seconds": float("nan")}, "finite"),
    ({"measure_steps": 0}, "positive"),
])
def test_invalid_matrix_rejected_before_process_launch(tmp_path, change, match):
    with pytest.raises(ValueError, match=match):
        module.build_arms(replace(_config(tmp_path), **change), pool_rows=16)


def test_summary_uses_slowest_rank_per_step_and_excludes_warmup(tmp_path):
    config = _config(tmp_path)
    arm = module.build_arms(config, pool_rows=16)[0]
    directory = Path(arm["directory"])
    _records(directory, arm)
    report = module.summarize_arm(directory, arm, num_gpus=2)
    assert report["step_seconds"] == [3, 4]
    assert report["step_input_tokens"] == [14, 30]
    assert report["measured_input_tokens"] == 44
    assert report["input_tokens_per_second"] == pytest.approx(44 / 7)
    assert report["measured_supervised_tokens"] == 20
    assert report["supervised_tokens_per_second"] == pytest.approx(20 / 7)
    assert report["peak_memory_allocated_bytes"] == 204
    assert report["peak_memory_reserved_bytes"] == 404
    assert report["measured_samples"] == 8
    assert len(report["sample_multiset_sha256"]) == 64


def test_summary_rejects_duplicated_global_tokens_and_incomplete_ranks(tmp_path):
    config = _config(tmp_path)
    arm = module.build_arms(config, pool_rows=16)[0]
    directory = Path(arm["directory"])
    _records(directory, arm)
    path = directory / "benchmark_steps_rank1.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["global_tokens"] *= 2
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="global_tokens"):
        module.summarize_arm(directory, arm, num_gpus=2)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
    with pytest.raises(ValueError, match="recorded steps"):
        module.summarize_arm(directory, arm, num_gpus=2)


def _sp_records(tmp_path):
    config = replace(_config(tmp_path), gpu_mask=(2, 3, 4, 5), sequence_parallel_sizes=(2,),
                     remove_padding=(True,), attn_backends=("flash_attention_2",))
    arm = module.build_arms(config, pool_rows=16)[0]
    directory = Path(arm["directory"])
    _records(directory, arm)
    reference = module.summarize_arm(directory, {**arm, "sequence_parallel_size": 1}, num_gpus=2)
    originals = [[json.loads(line) for line in (directory / f"benchmark_steps_rank{rank}.jsonl").read_text().splitlines()]
                 for rank in range(2)]
    for rank in range(4):
        rows = json.loads(json.dumps(originals[rank % 2]))
        for row in rows:
            row.update(data_parallel_rank=rank % 2, data_parallel_size=2, sequence_parallel_size=2)
            if rank == 3 and row["step"] > 2:
                row["elapsed_seconds"] = 7 if row["step"] == 3 else 5
                row["peak_memory_allocated_bytes"] += 1000
        (directory / f"benchmark_steps_rank{rank}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows))
        (directory / f"benchmark_manifest_rank{rank}.json").write_text(json.dumps({
            "status": "benchmark_completed", "completed_steps": arm["requested_steps"]}))
    return directory, arm, reference


def test_sp_replicas_do_not_double_tokens_samples_or_supervision(tmp_path):
    directory, arm, reference = _sp_records(tmp_path)
    report = module.summarize_arm(directory, arm, num_gpus=4)
    assert report["measured_input_tokens"] == reference["measured_input_tokens"] == 44
    assert report["measured_samples"] == reference["measured_samples"] == 8
    assert report["measured_supervised_tokens"] == reference["measured_supervised_tokens"] == 20
    assert report["sample_multiset_sha256"] == reference["sample_multiset_sha256"]
    assert report["step_seconds"] == [7, 5]
    assert report["input_tokens_per_second"] == pytest.approx(44 / 12)
    assert report["peak_memory_allocated_bytes"] == 1204
    comparison = module.build_report([
        {"name": "sp1", "status": "completed", "measurement": reference},
        {"name": "sp2", "status": "completed", "measurement": report},
    ], "matched")
    assert comparison["arms"][1]["workload_comparison"]["comparable"]


@pytest.mark.parametrize("key,value,match", [
    ("sample_sha256", ["f" * 64, "f" * 64], "copies differ"),
    ("data_parallel_size", 3, "topology differs"),
    ("data_parallel_rank", 0, "complete sequence-parallel copies"),
])
def test_sp_duplicate_or_topology_mismatch_rejected(tmp_path, key, value, match):
    directory, arm, _ = _sp_records(tmp_path)
    path = directory / "benchmark_steps_rank3.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1][key] = value
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=match):
        module.summarize_arm(directory, arm, num_gpus=4)


def test_matched_comparison_requires_sample_multiset_and_token_totals():
    def result(name, sample_hash, tokens=44):
        return {"name": name, "status": "completed", "measurement": {
            "sample_multiset_sha256": sample_hash, "measured_input_tokens": tokens}}

    report = module.build_report([result("first", "same"), result("different-groups", "same")], "matched")
    assert report["status"] == "completed"
    assert report["arms"][1]["workload_comparison"]["comparable"]
    report = module.build_report([result("first", "same"), result("changed", "other")], "matched")
    assert report["status"] == "failed"
    assert report["arms"][1]["status"] == "workload_mismatch"
    report = module.build_report([result("first", "same"), result("changed", "other")], "screen")
    assert report["status"] == "completed"
    assert not report["arms"][1]["workload_comparison"]["comparable"]


def test_all_resource_failures_cannot_claim_a_successful_benchmark():
    results = [{"name": "too-large", "status": "oom"}, {"name": "stalled", "status": "timeout"}]
    report = module.build_report(results, "screen")
    assert report["status"] == "no_completed_arms"
    assert report["completed_arms"] == 0
    assert report["resource_failure_arms"] == 2


def test_incremental_report_does_not_mutate_measured_evidence():
    results = [{"name": "missing-hashes", "status": "completed", "measurement": {
        "sample_multiset_sha256": None, "measured_input_tokens": 44}}]
    report = module.build_report(results, "matched")
    assert report["status"] == "failed"
    assert results[0]["status"] == "completed"


@pytest.mark.parametrize("stderr,expected", [
    ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate", "oom"),
    ("ValueError: malformed training sample", "failed"),
    ("torch.distributed.DistBackendError: NCCL error", "failed"),
])
def test_child_errors_keep_logs_and_distinguish_oom(tmp_path, monkeypatch, stderr, expected):
    config = _config(tmp_path)
    arm = module.build_arms(config, pool_rows=16)[0]

    class Process:
        def __init__(self, _command, **kwargs):
            assert kwargs["start_new_session"]
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2,3"
            kwargs["stderr"].write(stderr)

        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    result = module.run_arm(config, arm)
    assert result["status"] == expected
    assert stderr in result["error"]
    assert Path(result["stderr"]).read_text() == stderr


def test_timeout_stops_entire_child_process_group(tmp_path, monkeypatch):
    config = _config(tmp_path)
    arm = module.build_arms(config, pool_rows=16)[0]
    killed = []

    class Process:
        pid = 12345

        def __init__(self, *_args, **_kwargs):
            self.calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls < 3:
                raise subprocess.TimeoutExpired("torchrun", timeout)
            return -9

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    monkeypatch.setattr(module.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    result = module.run_arm(config, arm)
    assert result["status"] == "timeout"
    assert killed == [(12345, module.signal.SIGTERM), (12345, module.signal.SIGKILL)]


def test_launch_failure_is_not_mislabeled_as_oom(tmp_path, monkeypatch):
    config = _config(tmp_path)
    arm = module.build_arms(config, pool_rows=16)[0]

    def fail(*_args, **_kwargs):
        raise FileNotFoundError("missing benchmark interpreter")

    monkeypatch.setattr(module.subprocess, "Popen", fail)
    with pytest.raises(FileNotFoundError, match="interpreter"):
        module.run_arm(config, arm)
