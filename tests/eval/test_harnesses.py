from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from qwenpt.eval import harnesses as h


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for name in ("task-c", "task-a", "task-b"):
        task = root / "tasks" / name
        (task / "tests").mkdir(parents=True)
        (task / "task.toml").write_text('version = "1.0"\n', encoding="utf-8")
        (task / "instruction.md").write_text("A synthetic test task", encoding="utf-8")
        (task / "tests" / "test.sh").write_text("exit 0\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], check=True)
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return {"id": "fixture_terminal", "engine": "harbor", "checkout": str(root),
            "revision": revision, "engine_version": "0.25.0", "expected_count": 3,
            "subset_count": 2, "model_revision": "checkpoint-sha256-123", "attempts_full": 2}


def receipt(task, split, output, monkeypatch):
    monkeypatch.setenv("BENCHMARK_MODEL", "local-27b")
    monkeypatch.setenv("BENCHMARK_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("BENCHMARK_API_KEY", "test-secret-not-for-artifacts")
    manifest = h.prepare_harness(task, split, output)
    runtime, env = h._runtime(manifest)
    h._json(output / "runtime.json", runtime)
    h._json(output / "exit.json", {"returncode": 0, "manifest_fingerprint": manifest["fingerprint"]})
    return manifest, runtime, env


def trial(output, manifest, runtime, task_id, suffix="one", reward=1, error=None):
    native = {"id": f"{task_id}-{suffix}", "task_name": task_id,
              "trial_name": f"{task_id}__{suffix}", "finished_at": "2026-10-29T00:00:01Z",
              "config": {"task": {"path": manifest["task_paths"][task_id]},
                         "agent": {"name": manifest["agent"], "model_name": runtime["model"]}},
              "verifier_result": {"rewards": {"reward": reward}},
              "exception_info": {"exception_type": error} if error else None}
    path = output / "jobs" / "frozen" / native["trial_name"] / "result.json"
    h._json(path, native)
    return path


def test_selection_is_stable_not_first_n_and_full_includes_all(corpus, tmp_path):
    one = h.prepare_harness(corpus, "subset", tmp_path / "a")
    two = h.prepare_harness(corpus, "subset", tmp_path / "b")
    assert one == two
    assert len(one["task_ids"]) == 2
    assert len(h.prepare_harness(corpus, "full", tmp_path / "c")["task_ids"]) == 3
    with pytest.raises(ValueError, match="differs"):
        h.prepare_harness(corpus, "full", tmp_path / "a")


def test_revision_count_and_dirty_data_are_rejected(corpus, tmp_path):
    for change, match in [({"revision": "a" * 40}, "HEAD"), ({"expected_count": 2}, "count"),
                          ({"revision": "main"}, "SHA"), ({"subset_count": 4}, "exceeds")]:
        with pytest.raises(ValueError, match=match):
            h.prepare_harness(dict(corpus, **change), "subset", tmp_path / "out")
    (Path(corpus["checkout"]) / "tasks/task-a/instruction.md").write_text("Changed prompt", encoding="utf-8")
    with pytest.raises(ValueError, match="changes"):
        h.prepare_harness(corpus, "subset", tmp_path / "out")


def test_explicit_ids_and_no_glob_expansion(corpus, tmp_path):
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps(["task-b", "task-a"]), encoding="utf-8")
    task = dict(corpus, subset_ids_file=str(ids))
    manifest = h.prepare_harness(task, "subset", tmp_path / "out")
    assert manifest["task_ids"] == ["task-a", "task-b"]
    ids.write_text(json.dumps(["task-*", "task-a"]), encoding="utf-8")
    with pytest.raises(ValueError, match="known task"):
        h.prepare_harness(task, "subset", tmp_path / "different")


def test_native_job_config_has_exact_paths_no_secrets(corpus, tmp_path, monkeypatch):
    output = tmp_path / "out"
    manifest, runtime, env = receipt(corpus, "subset", output, monkeypatch)
    job = h.native_job_config(manifest, output, runtime)
    assert job["n_attempts"] == 1
    assert {Path(item["path"]).name for item in job["tasks"]} == set(manifest["task_ids"])
    assert job["verifier"]["disable"] is False
    assert job["retry"]["max_retries"] == 0
    assert job["agents"][0]["model_api_key_name"] == "BENCHMARK_API_KEY"
    assert job["agents"][0]["model_api_format"] == "openai_chat_completions"
    assert env["OPENAI_API_KEY"] == "test-secret-not-for-artifacts"
    assert "test-secret" not in json.dumps([job, runtime, manifest])


def test_deepswe_requires_pier_and_forwards_model_endpoint(corpus, tmp_path, monkeypatch):
    task = dict(corpus, engine="pier", engine_version="0.3.1", agent_version="2.4.6")
    manifest, runtime, env = receipt(task, "full", tmp_path / "out", monkeypatch)
    job = h.native_job_config(manifest, tmp_path / "out", runtime)
    assert job["agents"][0]["name"] == "mini-swe-agent"
    assert "model_api_key_name" not in job["agents"][0]
    assert job["agents"][0]["kwargs"]["model_class"] == "litellm"
    assert env["OPENAI_API_BASE"] == "http://localhost:8000/v1"
    with pytest.raises(ValueError, match="newer than"):
        h.prepare_harness(dict(task, engine_version="0.3.0"), "subset", tmp_path / "old")


def test_partial_and_infra_trials_never_publish_full_score(corpus, tmp_path, monkeypatch):
    output = tmp_path / "out"
    manifest, runtime, _ = receipt(corpus, "subset", output, monkeypatch)
    trial(output, manifest, runtime, manifest["task_ids"][0])
    partial = h.score_harness(corpus, "subset", output)
    assert partial["complete"] is False
    assert partial["final_mean_score"] is None
    assert partial["score_lower_bound"] == 0.5
    assert partial["missing_attempts"] == 1
    trial(output, manifest, runtime, manifest["task_ids"][1], reward=0, error="SandboxBuildFailedError")
    failed = h.score_harness(corpus, "subset", output)
    assert failed["infra_errors"] == 1
    assert failed["attempts_scored"] == 1
    assert failed["complete"] is False


def test_model_budget_failure_is_counted_only_with_native_reward(corpus, tmp_path, monkeypatch):
    output = tmp_path / "out"
    manifest, runtime, _ = receipt(corpus, "subset", output, monkeypatch)
    for name in manifest["task_ids"]:
        trial(output, manifest, runtime, name, reward=0, error="AgentTimeoutError")
    result = h.score_harness(corpus, "subset", output)
    assert result["complete"] is True
    assert result["model_failures"] == 2
    assert result["final_mean_score"] == 0


def test_attempts_are_not_extra_independent_tasks(corpus, tmp_path, monkeypatch):
    output = tmp_path / "out"
    manifest, runtime, _ = receipt(corpus, "full", output, monkeypatch)
    for name in manifest["task_ids"]:
        trial(output, manifest, runtime, name, suffix="one", reward=1)
        trial(output, manifest, runtime, name, suffix="two", reward=0)
    result = h.score_harness(corpus, "full", output)
    assert result["expected_tasks"] == 3
    assert result["expected_attempts"] == 6
    assert result["final_mean_score"] == 0.5
    trial(output, manifest, runtime, manifest["task_ids"][0], suffix="extra")
    with pytest.raises(ValueError, match="exceed"):
        h.score_harness(corpus, "full", output)


@pytest.mark.parametrize("field", ["task_path", "model", "trial_id"])
def test_reject_foreign_or_duplicate_native_results(corpus, tmp_path, monkeypatch, field):
    output = tmp_path / "out"
    manifest, runtime, _ = receipt(corpus, "subset", output, monkeypatch)
    path = trial(output, manifest, runtime, manifest["task_ids"][0])
    data = json.loads(path.read_text(encoding="utf-8"))
    if field == "task_path":
        data["config"]["task"]["path"] = str(tmp_path / "foreign")
    elif field == "model":
        data["config"]["agent"]["model_name"] = "openai/different"
    else:
        data["id"] = None
    h._json(path, data)
    with pytest.raises(ValueError):
        h.score_harness(corpus, "subset", output)


def test_run_dispatches_native_config_and_redacts_console(corpus, tmp_path, monkeypatch):
    output = tmp_path / "out"
    manifest, runtime, _ = receipt(corpus, "subset", output, monkeypatch)
    real_run = subprocess.run
    calls = []
    monkeypatch.setattr(h.shutil, "which", lambda value: "/bin/harbor")

    def fake_runner(argv, **kwargs):
        if argv[0] == "git":
            return real_run(argv, **kwargs)
        calls.append((argv, kwargs))
        if argv[1] == "--version":
            return subprocess.CompletedProcess(argv, 0, "0.25.0\n", "")
        assert argv[1:3] == ["run", "--config"]
        assert kwargs.get("shell") is not True
        assert kwargs["env"]["OPENAI_API_KEY"] == "test-secret-not-for-artifacts"
        for name in manifest["task_ids"]:
            trial(output, manifest, runtime, name)
        return subprocess.CompletedProcess(argv, 0, "test-secret-not-for-artifacts\n", "")

    monkeypatch.setattr(h.subprocess, "run", fake_runner)
    result = h.run_harness(corpus, "subset", output)
    assert result["complete"] is True
    assert result["final_mean_score"] == 1
    assert "test-secret" not in (output / "stdout.log").read_text(encoding="utf-8")
    assert "[REDACTED]" in (output / "stdout.log").read_text(encoding="utf-8")
    assert not (output / ".harness.lock").exists()
    assert len(calls) == 2


def test_endpoint_credentials_and_unknown_settings_are_refused(corpus, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Unknown"):
        h.prepare_harness(dict(corpus, api_key="secret"), "subset", tmp_path / "out")
    manifest, _, _ = receipt(corpus, "subset", tmp_path / "out", monkeypatch)
    monkeypatch.setenv("BENCHMARK_BASE_URL", "https://user:secret@example.test/v1")
    with pytest.raises(ValueError, match="userinfo"):
        h._runtime(manifest)


def test_native_namespaced_name_is_matched_to_local_path(corpus, tmp_path, monkeypatch):
    root = Path(corpus["checkout"])
    for name in ("task-a", "task-b", "task-c"):
        (root / "tasks" / name / "task.toml").write_text(
            f'[task]\nname = "organization/{name}"\n[metadata]\ncategory = "{name[-1]}"\n', encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "native names"], check=True)
    task = dict(corpus, revision=subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip())
    output = tmp_path / "out"
    manifest, runtime, _ = receipt(task, "subset", output, monkeypatch)
    assert len(set(manifest["strata"].values())) == 2
    for name in manifest["task_ids"]:
        path = trial(output, manifest, runtime, name)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["task_name"] = "organization/" + name
        h._json(path, data)
    assert h.score_harness(task, "subset", output)["complete"] is True


def test_preflight_without_keys_reports_missing_runner_and_docker(corpus, tmp_path, monkeypatch):
    monkeypatch.delenv("BENCHMARK_API_KEY", raising=False)
    monkeypatch.setattr(h.shutil, "which", lambda value: None)
    result = h.preflight_harness(corpus, "subset", tmp_path / "out")
    assert result["ready"] is False
    assert result["model_api_checked"] is False
    assert [row["component"] for row in result["checks"]] == ["dataset", "runner", "environment", "endpoint_configuration"]


def test_pier_requires_pinned_agent_version(corpus, tmp_path):
    with pytest.raises(ValueError, match="agent_version"):
        h.prepare_harness(dict(corpus, engine="pier", engine_version="0.3.1"), "subset", tmp_path / "out")


@pytest.mark.parametrize("receipt_value", [None, {"returncode": None}, {"returncode": True}, {"returncode": 1}])
def test_interrupted_or_invalid_exit_receipt_never_certifies_native_completion(corpus, tmp_path, monkeypatch, receipt_value):
    output = tmp_path / "out"
    manifest, runtime, _ = receipt(corpus, "subset", output, monkeypatch)
    for name in manifest["task_ids"]:
        trial(output, manifest, runtime, name)
    if receipt_value is None:
        (output / "exit.json").unlink()
    else:
        h._json(output / "exit.json", {**receipt_value, "manifest_fingerprint": manifest["fingerprint"]})
    result = h.score_harness(corpus, "subset", output)
    assert not result["complete"]
    assert result["final_mean_score"] is None


def test_explicit_unauthenticated_local_server_uses_nonsecret_framework_key(corpus, tmp_path, monkeypatch):
    task = dict(corpus, api_key_required=False)
    manifest, _, _ = receipt(task, "subset", tmp_path / "out", monkeypatch)
    monkeypatch.delenv("BENCHMARK_API_KEY")
    runtime, env = h._runtime(manifest)
    assert runtime["authentication"] == "framework_placeholder"
    assert env["OPENAI_API_KEY"] == env["BENCHMARK_API_KEY"] == "EMPTY"
    assert runtime["model_api_format"] == "openai_chat_completions"


def test_harness_checkout_subdirectory_cannot_claim_repository_root(corpus, tmp_path):
    with pytest.raises(ValueError, match="repository root"):
        h.prepare_harness(dict(corpus, checkout=str(Path(corpus["checkout"]) / "tasks")), "subset", tmp_path / "out")


def test_full_chat_route_is_normalized_for_native_sdks(corpus, tmp_path, monkeypatch):
    manifest, _, _ = receipt(corpus, "subset", tmp_path / "out", monkeypatch)
    monkeypatch.setenv("BENCHMARK_BASE_URL", "http://localhost:8000/v1/chat/completions")
    runtime, env = h._runtime(manifest)
    assert runtime["base_url"] == "http://localhost:8000/v1"
    assert env["OPENAI_BASE_URL"] == env["OPENAI_API_BASE"] == runtime["base_url"]
