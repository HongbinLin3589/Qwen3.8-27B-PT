"""Check SFT monitoring evidence and notification delivery without Slurm or GPUs."""

import json
import subprocess
import sys
import zipfile

import pytest

from legacy.sft.utils import monitor_sft as monitor


def _record(step, seconds=100.0, **changes):
    data = {"train/loss": 0.4, "train/grad_norm": 1.5, "timing/train_step_seconds": seconds}
    data.update(changes)
    return {"step": step, "data": data}


def _write_metrics(run_dir, records):
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))


def _write_checkpoint(run_dir, step):
    directory = run_dir / f"global_step_{step}"
    directory.mkdir(parents=True)
    names = [f"{kind}_world_size_2_rank_{rank}.pt"
             for kind in ("model", "optim", "extra_state") for rank in (0, 1)]
    names += [f"data_{rank}.pt" for rank in (0, 1)]
    for name in names:
        with zipfile.ZipFile(directory / name, "w") as archive:
            for member, value in {"data.pkl": b"\x80\x02}.", "version": b"3\n", "byteorder": b"little"}.items():
                archive.writestr(f"archive/{member}", value)
            archive.writestr("archive/data/0", bytes(4))
    (directory / "fsdp_config.json").write_text(json.dumps({"FSDP_version": 2, "world_size": 2}))
    huggingface = directory / "huggingface"
    huggingface.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (huggingface / name).write_text(json.dumps({"model_type": "qwen3_5"}))
    (run_dir / "latest_checkpointed_iteration.txt").write_text(str(step))
    return directory


def _job(state="RUNNING", elapsed=3600, exit_code="0:0"):
    return {"job_id": "742968", "state": state, "exit_code": exit_code,
            "elapsed_seconds": elapsed, "start": "2026-10-21T01:00:00", "end": "Unknown"}


@pytest.fixture
def completed_run(tmp_path):
    """Build small structural artifacts rather than allocating model tensors."""
    records = [_record(step) for step in range(1, 2251)]
    records.append({"step": 2250, "data": {"val/loss": 0.25}})
    _write_metrics(tmp_path, records)
    for step in (1125, 2250):
        _write_checkpoint(tmp_path, step)
    (tmp_path / "timing_summary.json").write_text(json.dumps({
        "status": "completed", "completed_train_steps": 2250,
        "train_steps_seconds": 225000, "train_step_seconds_mean": 100,
        "initialization_seconds": 200, "fit_seconds": 226000, "run_seconds": 226200,
    }))
    return tmp_path


def test_step_statistics_exclude_validation_and_keep_cold_start():
    summary = monitor.summarize_steps([
        _record(1, 140), _record(2, 100), {"step": 2, "data": {"val/loss": 0.3}}, _record(3, 120),
    ])
    assert summary["latest_step"] == summary["logged_steps"] == 3
    assert summary["train_steps_seconds"] == 360
    assert summary["first_step_seconds"] == 140
    assert summary["mean_seconds"] == summary["median_seconds"] == 120
    assert summary["post_first_mean_seconds"] == 110
    assert summary["p90_seconds"] == summary["p95_seconds"] == 140
    assert summary["missing_steps"] == summary["nonfinite_steps"] == []


@pytest.mark.parametrize("key,value", [
    ("train/loss", float("nan")), ("train/grad_norm", float("inf")),
    ("timing/train_step_seconds", None), ("train/loss", "nan"),
    ("timing/train_step_seconds", 0), ("timing/train_step_seconds", -1),
])
def test_nonfinite_metrics_and_step_gaps_are_reported(key, value):
    summary = monitor.summarize_steps([_record(1), _record(3, **{key: value})])
    assert summary["nonfinite_steps"] == [3]
    assert summary["missing_steps"] == [2]
    assert summary["train_steps_seconds"] == 100


def test_duplicate_training_steps_are_not_hidden_by_summary(tmp_path):
    _write_metrics(tmp_path, [_record(1), _record(2), _record(2)])
    snapshot = monitor.collect_snapshot(tmp_path, _job())
    assert snapshot["stats"]["duplicate_steps"] == [2]
    assert "invalid_metrics" in {event["key"] for event in monitor.detect_events(snapshot)}


def test_tracker_and_directory_alone_do_not_verify_checkpoint(tmp_path):
    (tmp_path / "global_step_1125").mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("1125")
    result = monitor.checkpoint_status(tmp_path, 1125, 1126, False)
    assert not result["complete"]
    assert any("model_world_size_2_rank_1.pt" in error for error in result["errors"])


def test_closed_checkpoint_waits_for_save_barrier_evidence(tmp_path):
    _write_checkpoint(tmp_path, 1125)
    waiting = monitor.checkpoint_status(tmp_path, 1125, 1125, False)
    assert not waiting["complete"]
    assert waiting["errors"] == ["waiting for subsequent training progress or clean job exit"]
    complete = monitor.checkpoint_status(tmp_path, 1125, 1126, False)
    assert complete["complete"]
    assert len(complete["file_bytes"]) == 8
    assert complete["total_bytes"] == sum(complete["file_bytes"].values())


def test_final_checkpoint_is_confirmed_by_clean_exit(tmp_path):
    _write_checkpoint(tmp_path, 2250)
    assert not monitor.checkpoint_status(tmp_path, 2250, 2250, False)["complete"]
    assert monitor.checkpoint_status(tmp_path, 2250, 2250, True)["complete"]


@pytest.mark.parametrize("fault", [
    "missing_rank", "truncated_zip", "zip_without_metadata", "model_without_storage",
    "wrong_fsdp", "missing_config", "missing_tokenizer", "tracker_behind",
])
def test_checkpoint_requires_complete_rank_archives_and_configuration(tmp_path, fault):
    directory = _write_checkpoint(tmp_path, 1125)
    model = directory / "model_world_size_2_rank_1.pt"
    if fault == "missing_rank":
        model.unlink()
    elif fault == "truncated_zip":
        model.write_bytes(model.read_bytes()[:-32])
    elif fault in {"zip_without_metadata", "model_without_storage"}:
        with zipfile.ZipFile(model, "w") as archive:
            if fault == "zip_without_metadata":
                archive.writestr("archive/data/0", bytes(4))
            else:
                for name in ("data.pkl", "version", "byteorder"):
                    archive.writestr(f"archive/{name}", b"x")
    elif fault == "wrong_fsdp":
        (directory / "fsdp_config.json").write_text('{"FSDP_version": 2, "world_size": 1}')
    elif fault == "missing_config":
        (directory / "huggingface/config.json").unlink()
    elif fault == "missing_tokenizer":
        (directory / "huggingface/tokenizer.json").unlink()
    elif fault == "tracker_behind":
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("1000")
    result = monitor.checkpoint_status(tmp_path, 1125, 1126, False)
    assert not result["complete"]
    assert result["errors"]


def test_checkpoint_checks_only_zip_directory(tmp_path, monkeypatch):
    _write_checkpoint(tmp_path, 1125)

    def reject_payload_read(*args, **kwargs):
        raise AssertionError("Monitoring must not read hundreds of GiB of tensor storage")

    monkeypatch.setattr(zipfile.ZipFile, "read", reject_payload_read)
    monkeypatch.setattr(zipfile.ZipFile, "testzip", reject_payload_read)
    assert monitor.checkpoint_status(tmp_path, 1125, 1126, False)["complete"]


def test_completed_run_requires_all_artifacts_and_reports_measured_values(completed_run):
    snapshot = monitor.collect_snapshot(completed_run, _job("COMPLETED", elapsed=226300))
    assert snapshot["final_success"]
    assert snapshot["final_issues"] == []
    assert snapshot["job"]["elapsed_seconds"] == 226300
    assert snapshot["stats"]["mean_seconds"] == 100
    assert snapshot["val_loss"] == 0.25
    assert all(checkpoint["complete"] for checkpoint in snapshot["checkpoints"].values())
    assert monitor.detect_events(snapshot) == [{"key": "terminal", "kind": "completed", "detail": []}]


@pytest.mark.parametrize("fault", [
    "missing_validation", "earlier_validation", "nan_validation", "string_validation",
    "missing_timing", "step_gap", "duplicate_step", "failed_slurm",
])
def test_partial_or_failed_runs_are_never_reported_successful(completed_run, fault):
    records = [json.loads(line) for line in (completed_run / "metrics.jsonl").read_text().splitlines()]
    job = _job("COMPLETED")
    if fault == "missing_validation":
        records.pop()
    elif fault == "earlier_validation":
        records[-1]["step"] = 1125
    elif fault == "nan_validation":
        records[-1]["data"]["val/loss"] = float("nan")
    elif fault == "string_validation":
        records[-1]["data"]["val/loss"] = "invalid"
    elif fault == "missing_timing":
        (completed_run / "timing_summary.json").unlink()
    elif fault == "step_gap":
        records.pop(100)
    elif fault == "duplicate_step":
        records.insert(0, records[0])
    elif fault == "failed_slurm":
        job = _job("FAILED", exit_code="1:0")
    _write_metrics(completed_run, records)
    snapshot = monitor.collect_snapshot(completed_run, job)
    assert not snapshot["final_success"]
    assert snapshot["final_issues"]
    assert monitor.detect_events(snapshot)[0]["kind"] == "failed"


def test_inflight_partial_json_line_does_not_break_monitor(tmp_path):
    _write_metrics(tmp_path, [_record(1)])
    with (tmp_path / "metrics.jsonl").open("a") as stream:
        stream.write('{"step":2,"data":')
    snapshot = monitor.collect_snapshot(tmp_path, _job())
    assert snapshot["stats"]["latest_step"] == 1


@pytest.mark.parametrize("projected,key", [(72 * 3600 + 1, "eta_over_limit"), (71 * 3600, "time_margin_risk")])
def test_time_risk_events_reserve_time_for_validation_and_saving(tmp_path, projected, key):
    _write_metrics(tmp_path, [_record(step) for step in range(1, 31)])
    snapshot = monitor.collect_snapshot(tmp_path, _job())
    snapshot["projections_seconds"] = {"recent30_mean_seconds": projected}
    assert key in {event["key"] for event in monitor.detect_events(snapshot)}


def test_early_normal_progress_stays_quiet(tmp_path):
    _write_metrics(tmp_path, [_record(step) for step in range(1, 6)])
    snapshot = monitor.collect_snapshot(tmp_path, _job())
    assert monitor.detect_events(snapshot) == []


def test_pending_allocation_stays_quiet_before_training(tmp_path):
    snapshot = monitor.collect_snapshot(tmp_path, _job("PENDING", elapsed=0))
    assert snapshot["stats"]["latest_step"] == 0
    assert monitor.detect_events(snapshot) == []


def test_notification_uses_event_job_id_for_restarted_run(tmp_path, monkeypatch):
    event_path = tmp_path / "event_terminal.json"
    monitor.write_json(event_path, {"job_id": "744999", "kind": "completed"})
    calls = []

    def enqueue(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="queued\n")

    monkeypatch.setattr(monitor.subprocess, "run", enqueue)
    assert monitor.queue_event("/test/codex", "test-thread", event_path) == "queued"
    assert calls[0][:5] == ["/test/codex", "queue", "--thread", "test-thread", "--message"]
    assert calls[0][5].startswith("SFT_MONITOR_EVENT_744999：")
    assert "742968" not in calls[0][5]
    assert str(event_path) in calls[0][5]


def test_invalid_validation_type_produces_anomaly_without_read_failure(tmp_path):
    _write_metrics(tmp_path, [_record(1), {"step": 1, "data": {"val/loss": "invalid"}}])
    snapshot = monitor.collect_snapshot(tmp_path, _job())
    assert "invalid_validation" in {event["key"] for event in monitor.detect_events(snapshot)}


def test_written_reports_are_standard_json_even_for_nonfinite_metrics(tmp_path):
    report = tmp_path / "report.json"
    monitor.write_json(report, {"nested": [float("nan"), {"loss": float("inf")} ]})

    def reject_constant(value):
        raise AssertionError(f"Nonstandard JSON constant: {value}")

    parsed = json.loads(report.read_text(), parse_constant=reject_constant)
    assert isinstance(parsed["nested"][0], str)
    assert isinstance(parsed["nested"][1]["loss"], str)


def test_missing_checkpoint_alert_requires_later_training_progress(tmp_path):
    _write_metrics(tmp_path, [_record(step) for step in range(1, 1126)])
    snapshot = monitor.collect_snapshot(tmp_path, _job())
    assert "checkpoint_missing_1125" not in {event["key"] for event in monitor.detect_events(snapshot)}
    _write_metrics(tmp_path, [_record(step) for step in range(1, 1127)])
    snapshot = monitor.collect_snapshot(tmp_path, _job())
    assert "checkpoint_missing_1125" in {event["key"] for event in monitor.detect_events(snapshot)}


def test_checkpoint_retention_overflow_alerts_and_prevents_final_success(completed_run):
    (completed_run / "global_step_1").mkdir()
    running = monitor.collect_snapshot(completed_run, _job())
    assert running["retention_exceeded"]
    assert set(running["checkpoint_directories"]) == {"global_step_1", "global_step_1125", "global_step_2250"}
    assert "checkpoint_retention" in {event["key"] for event in monitor.detect_events(running)}
    terminal = monitor.collect_snapshot(completed_run, _job("COMPLETED"))
    assert not terminal["final_success"]
    assert "more than two checkpoint directories were retained" in terminal["final_issues"]


def test_deadline_warning_includes_final_validation_phase(completed_run):
    threshold = monitor.LIMIT_SECONDS - 1800
    before = monitor.collect_snapshot(completed_run, _job(elapsed=threshold - 1))
    assert before["stats"]["latest_step"] == 2250
    assert "deadline_near" not in {event["key"] for event in monitor.detect_events(before)}
    near = monitor.collect_snapshot(completed_run, _job(elapsed=threshold))
    assert "deadline_near" in {event["key"] for event in monitor.detect_events(near)}
    terminal = monitor.collect_snapshot(completed_run, _job("COMPLETED", elapsed=threshold))
    assert {event["key"] for event in monitor.detect_events(terminal)} == {"terminal"}


def _run_once(monkeypatch, run_dir):
    monkeypatch.setattr(sys, "argv", ["monitor_sft", "--job-id", "742968", "--run-dir", str(run_dir),
                                     "--thread-id", "test-thread", "--once"])
    monitor.main()


def test_notification_is_deduplicated_across_monitor_restarts(tmp_path, monkeypatch):
    _write_metrics(tmp_path, [_record(step) for step in range(1, 31)])
    monkeypatch.setattr(monitor, "query_job", lambda job_id: _job(elapsed=100000))
    monkeypatch.setattr(monitor.shutil, "which", lambda name: "/test/codex")
    delivered = []
    monkeypatch.setattr(monitor, "queue_event", lambda *args: delivered.append(args[-1]) or "queued")
    _run_once(monkeypatch, tmp_path)
    _run_once(monkeypatch, tmp_path)
    assert len(delivered) == 1
    state = json.loads((tmp_path / "monitor/state.json").read_text())
    assert state["delivered_event_keys"] == ["eta_over_limit"]
    assert state["pending_event_keys"] == []


def test_failed_notification_is_retried_without_marking_it_delivered(tmp_path, monkeypatch):
    _write_metrics(tmp_path, [_record(step) for step in range(1, 31)])
    monkeypatch.setattr(monitor, "query_job", lambda job_id: _job(elapsed=100000))
    monkeypatch.setattr(monitor.shutil, "which", lambda name: "/test/codex")
    attempts = []

    def enqueue(*args):
        attempts.append(args[-1])
        if len(attempts) == 1:
            raise subprocess.CalledProcessError(1, ["codex", "queue"])
        return "queued"

    monkeypatch.setattr(monitor, "queue_event", enqueue)
    _run_once(monkeypatch, tmp_path)
    failed = json.loads((tmp_path / "monitor/state.json").read_text())
    assert failed["delivered_event_keys"] == []
    assert failed["pending_event_keys"] == ["eta_over_limit"]
    _run_once(monkeypatch, tmp_path)
    _run_once(monkeypatch, tmp_path)
    assert len(attempts) == 2
    state = json.loads((tmp_path / "monitor/state.json").read_text())
    assert state["delivered_event_keys"] == ["eta_over_limit"]
    assert state["last_notification_error"] is None


def test_terminal_report_is_persisted_and_notified_once(completed_run, monkeypatch):
    monkeypatch.setattr(monitor, "query_job", lambda job_id: _job("COMPLETED", elapsed=226300))
    monkeypatch.setattr(monitor.shutil, "which", lambda name: "/test/codex")
    delivered = []
    monkeypatch.setattr(monitor, "queue_event", lambda *args: delivered.append(args[-1]) or "queued")
    _run_once(monkeypatch, completed_run)
    _run_once(monkeypatch, completed_run)
    assert len(delivered) == 1
    assert delivered[0].name == "event_terminal.json"
    state = json.loads((completed_run / "monitor/state.json").read_text())
    assert state["finished"]
    report = json.loads((completed_run / "monitor/final_report.json").read_text())
    assert report["final_success"]
    assert report["validation_step"] == 2250
    assert report["val_loss"] == 0.25


def test_read_failure_requires_three_polls_before_notification(tmp_path, monkeypatch):
    def unavailable(job_id):
        raise subprocess.TimeoutExpired("sacct", 20)

    monkeypatch.setattr(monitor, "query_job", unavailable)
    monkeypatch.setattr(monitor.shutil, "which", lambda name: "/test/codex")
    delivered = []
    monkeypatch.setattr(monitor, "queue_event", lambda *args: delivered.append(args[-1]) or "queued")
    for _ in range(2):
        _run_once(monkeypatch, tmp_path)
    assert delivered == []
    _run_once(monkeypatch, tmp_path)
    _run_once(monkeypatch, tmp_path)
    assert len(delivered) == 1
    state = json.loads((tmp_path / "monitor/state.json").read_text())
    assert state["consecutive_errors"] == 4
    assert state["delivered_event_keys"] == ["monitor_read_failure"]
