"""Exercise quiet evaluation monitoring, result integrity and notification retries."""

import hashlib
import json
import subprocess

import pytest

from qwenpt.eval import monitor


def _sample(index=0):
    return dict(sample_index=index, correct=True, format_error=False, thinking_complete=True,
                generated_tokens=20, finish_reason="stop", text="answer")


def _group(variant="original", benchmark="gpqa_diamond", identity="q0"):
    return dict(variant=variant, benchmark=benchmark, question_id=identity, answer="1",
                run_fingerprint="test", samples=[_sample(index) for index in range(1 if benchmark == "gpqa_diamond" else 32)])


def _write_groups(path, groups):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(group) + "\n" for group in groups))


@pytest.fixture
def suite(tmp_path, monkeypatch):
    """Create current-attempt jobs and a small bank without Slurm or inference."""
    bank = [dict(question_id="q0", benchmark=benchmark, answer="1", samples=1 if benchmark == "gpqa_diamond" else 32)
            for benchmark in monitor.BENCHMARKS]
    bank_path = tmp_path / "bank/question_bank.jsonl"
    _write_groups(bank_path, bank)
    (bank_path.parent / "bank.sha256").write_text(hashlib.sha256(bank_path.read_bytes()).hexdigest())
    submission = [dict(variant=variant, benchmark=benchmark, job_id=str(index))
                  for index, (variant, benchmark) in enumerate(
                      ((variant, benchmark) for variant in monitor.VARIANTS for benchmark in monitor.BENCHMARKS), 1)]
    (tmp_path / "submission.json").write_text(json.dumps({"jobs": submission}))
    jobs = {item["job_id"]: dict(job_id=item["job_id"], state="PENDING", exit_code="0:0",
                                elapsed_seconds=0, start="Unknown", end="Unknown", node="") for item in submission}
    monkeypatch.setattr(monitor, "log_evidence", lambda job_id, log_dir: dict(paths=[], fatal_lines=[], engine_ready=False))
    monkeypatch.setattr(monitor, "query_jobs", lambda job_ids: jobs)
    return tmp_path, submission, jobs


def test_pending_qos_and_normal_progress_stay_quiet(suite):
    root, submission, jobs = suite
    assert monitor.detect_events(monitor.collect_snapshot(root, submission, jobs)) == []
    item = submission[0]
    jobs[item["job_id"]].update(state="RUNNING", elapsed_seconds=200)
    path = root / "results/original/gpqa_diamond/results.jsonl"
    _write_groups(path, [_group()])
    assert monitor.detect_events(monitor.collect_snapshot(root, submission, jobs)) == []


@pytest.mark.parametrize("state", ["FAILED", "OUT_OF_MEMORY", "TIMEOUT", "CANCELLED", "NODE_FAIL"])
def test_failure_states_are_actionable_and_previous_attempts_ignored(suite, state):
    root, submission, jobs = suite
    submission[0]["previous_attempts"] = [{"job_id": "old", "state": "FAILED"}]
    assert monitor.detect_events(monitor.collect_snapshot(root, submission, jobs)) == []
    jobs["1"].update(state=state, exit_code="1:0")
    events = monitor.detect_events(monitor.collect_snapshot(root, submission, jobs))
    assert len(events) == 1
    assert events[0]["job_id"] == "1"
    assert any(state in issue for issue in events[0]["detail"])


def test_only_unterminated_inflight_tail_is_ignored_without_modifying_results(tmp_path):
    path = tmp_path / "results.jsonl"
    payload = json.dumps(_group()).encode() + b'\n{"partial":'
    path.write_bytes(payload)
    assert monitor.read_groups(path, allow_partial=True) == [_group()]
    assert path.read_bytes() == payload
    with pytest.raises(ValueError, match="Malformed completed record"):
        monitor.read_groups(path)
    path.write_bytes(payload + b"\n")
    with pytest.raises(ValueError, match="Malformed completed record"):
        monitor.read_groups(path, allow_partial=True)


@pytest.mark.parametrize("fault", ["missing", "truncated", "duplicate", "wrong_samples", "wrong_variant"])
def test_clean_slurm_exit_does_not_hide_invalid_result_coverage(suite, fault):
    root, submission, jobs = suite
    jobs["1"].update(state="COMPLETED", elapsed_seconds=100)
    path = root / "results/original/gpqa_diamond/results.jsonl"
    groups = [_group()]
    if fault == "duplicate":
        groups.append(_group())
    elif fault == "wrong_samples":
        groups[0]["samples"].append(_sample(1))
    elif fault == "wrong_variant":
        groups[0]["variant"] = "step_100"
    if fault != "missing":
        _write_groups(path, groups)
    if fault == "truncated":
        with path.open("a") as stream:
            stream.write('{"partial":')
    task = monitor.collect_snapshot(root, submission, jobs)["tasks"][0]
    assert task["issues"]
    assert monitor.detect_events({"tasks": [task]})[0]["kind"] == "anomaly"


def test_running_stall_allowances_distinguish_initialization_and_generation(suite, monkeypatch):
    root, submission, jobs = suite
    jobs["1"].update(state="RUNNING", elapsed_seconds=1801)
    assert "initialization" in monitor.collect_snapshot(root, submission, jobs)["tasks"][0]["issues"][0]
    monkeypatch.setattr(monitor, "log_evidence", lambda job_id, log_dir: dict(paths=[], fatal_lines=[], engine_ready=True))
    assert monitor.collect_snapshot(root, submission, jobs)["tasks"][0]["issues"] == []
    jobs["1"]["elapsed_seconds"] = 5401
    assert "generation" in monitor.collect_snapshot(root, submission, jobs)["tasks"][0]["issues"][0]


def test_fatal_log_scan_ignores_informational_strings_and_old_attempts(tmp_path):
    (tmp_path / "eval_1.out").write_text("INFO 10-21 23:00:00 config RuntimeError: diagnostic\n"
                                        "WARNING 10-21 23:00:00 ValueError: handled\n"
                                        "init engine (profile, create kv cache, warmup model) took 50 s\n")
    (tmp_path / "eval_old.err").write_text("RuntimeError: old failed attempt\n")
    assert monitor.log_evidence("1", tmp_path)["fatal_lines"] == []
    assert monitor.log_evidence("1", tmp_path)["engine_ready"]
    (tmp_path / "eval_1.err").write_text("(EngineCore pid=10) RuntimeError: CUDA initialization failed\n")
    assert len(monitor.log_evidence("1", tmp_path)["fatal_lines"]) == 1


def test_frozen_monitor_source_uses_project_logs_from_run_path(suite, monkeypatch):
    root, submission, jobs = suite
    monkeypatch.setattr(monitor, "__file__", str(root / "monitor_source/src/qwenpt/eval/monitor.py"))
    paths = []
    def evidence(job_id, log_dir):
        paths.append(log_dir)
        return dict(paths=[], fatal_lines=[], engine_ready=False)
    monkeypatch.setattr(monitor, "log_evidence", evidence)
    monitor.collect_snapshot(root, submission, jobs)
    assert set(paths) == {root.resolve().parents[2] / "slurm_log/bolt-benchmarks"}


def test_stall_event_reoccurs_only_after_different_completed_progress(suite):
    root, submission, jobs = suite
    jobs["1"].update(state="RUNNING", elapsed_seconds=2000)
    snapshot = monitor.collect_snapshot(root, submission, jobs)
    event = monitor.detect_events(snapshot)[0]
    assert monitor.detect_events(snapshot)[0]["key"] == event["key"]
    snapshot["tasks"][0]["coverage"]["questions"] = 1
    assert monitor.detect_events(snapshot)[0]["key"] != event["key"]


def test_scheduler_query_requires_all_latest_rows_and_excludes_steps(monkeypatch):
    output = "1|CANCELLED by 20|0:15|100|start|end|node|\n1.batch|FAILED|1:0|90|start|end|node|\n"
    monkeypatch.setattr(monitor.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 0, stdout=output))
    assert monitor.query_jobs(["1"])["1"]["state"] == "CANCELLED"
    with pytest.raises(RuntimeError, match="Missing accounting rows"):
        monitor.query_jobs(["1", "2"])


def test_scheduler_failure_notifies_after_three_failures_and_only_once(suite, monkeypatch):
    root, _, _ = suite
    def fail(job_ids):
        raise subprocess.TimeoutExpired("sacct", 30)
    monkeypatch.setattr(monitor, "query_jobs", fail)
    queued = []
    monkeypatch.setattr(monitor, "queue_event", lambda *args: queued.append(args) or "ok")
    state = {}
    monitor.poll_once(root, state, "thread", "codex")
    monitor.poll_once(root, state, "thread", "codex")
    assert queued == []
    monitor.poll_once(root, state, "thread", "codex")
    monitor.poll_once(root, state, "thread", "codex")
    assert len(queued) == 1
    assert state["consecutive_query_failures"] == 4


def test_notification_failure_is_retried_without_marking_delivered(suite, monkeypatch):
    root, _, jobs = suite
    jobs["1"].update(state="FAILED", exit_code="1:0")
    calls = []
    def queue(*args):
        calls.append(args)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired("codex", 30)
        return "queued"
    monkeypatch.setattr(monitor, "queue_event", queue)
    state = {}
    monitor.poll_once(root, state, "thread", "codex")
    assert state["delivered_event_keys"] == []
    assert len(state["pending_event_keys"]) == 1
    monitor.poll_once(root, state, "thread", "codex")
    monitor.poll_once(root, state, "thread", "codex")
    assert len(calls) == 2
    assert len(state["delivered_event_keys"]) == 1
    assert not state["finished"]


def test_no_notify_preserves_pending_event_for_real_delivery(suite, monkeypatch):
    root, _, jobs = suite
    jobs["1"].update(state="FAILED", exit_code="1:0")
    monkeypatch.setattr(monitor, "queue_event", lambda *args: pytest.fail("Unexpected notification"))
    state = monitor.poll_once(root, {}, "thread", None, no_notify=True)
    assert len(state["pending_event_keys"]) == 1
    assert state["delivered_event_keys"] == []


def test_completion_requires_final_validation_and_fires_once(suite, monkeypatch):
    root, submission, jobs = suite
    for item in submission:
        jobs[item["job_id"]]["state"] = "COMPLETED"
        _write_groups(root / "results" / item["variant"] / item["benchmark"] / "results.jsonl",
                      [_group(item["variant"], item["benchmark"])])
    monkeypatch.setattr(monitor, "finalize_run", lambda *args: {"status": "complete"})
    queued = []
    monkeypatch.setattr(monitor, "queue_event", lambda *args: queued.append(args) or "ok")
    state = monitor.poll_once(root, {}, "thread", "codex")
    assert state["finished"]
    assert state["delivered_event_keys"] == ["suite_completed"]
    monitor.poll_once(root, state, "thread", "codex")
    assert len(queued) == 1


def test_new_failure_after_stall_produces_a_distinct_event(suite):
    root, submission, jobs = suite
    jobs["1"].update(state="RUNNING", elapsed_seconds=2000)
    first = monitor.detect_events(monitor.collect_snapshot(root, submission, jobs))[0]
    jobs["1"].update(state="FAILED", exit_code="1:0")
    second = monitor.detect_events(monitor.collect_snapshot(root, submission, jobs))[0]
    assert first["key"] != second["key"]


def test_finalizer_does_not_accept_small_bank_even_with_clean_jobs(suite):
    root, submission, jobs = suite
    for job in jobs.values():
        job["state"] = "COMPLETED"
    with pytest.raises(ValueError, match="Expected unique questions"):
        monitor.finalize_run(root, monitor.collect_snapshot(root, submission, jobs))


def test_queue_message_has_fixed_instructions_and_artifact_path(tmp_path, monkeypatch):
    calls = []
    def enqueue(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="queued\n")
    monkeypatch.setattr(monitor.subprocess, "run", enqueue)
    path = tmp_path / "event_suite_completed.json"
    assert monitor.queue_event("/verified/codex", "chat-id", path) == "queued"
    assert calls[0][:5] == ["/verified/codex", "queue", "--thread", "chat-id", "--message"]
    assert str(path) in calls[0][5]
    assert "正常运行和已通知事件保持静默" in calls[0][5]
    assert "提升百分点" in calls[0][5]
