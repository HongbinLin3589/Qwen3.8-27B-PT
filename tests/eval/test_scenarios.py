"""Native scenario contracts: selection, provenance, coverage and API routing.

These tests use native-shaped artifacts and stub environment runtimes. They do
not start browsers, office applications, VMs, emulators, or paid model calls.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from qwenpt.eval import scenarios


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def configured(tmp_path, monkeypatch):
    checkout = tmp_path / "native"
    checkout.mkdir()
    monkeypatch.setattr(scenarios, "_checkout_errors", lambda *_: [])
    monkeypatch.setattr(scenarios, "_dependency_errors", lambda *_: [])
    monkeypatch.setenv("SCENE_MODEL", "Qwen3.8-27B-step-100")
    monkeypatch.setenv("SCENE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("SCENE_KEY", "secret-scene-test-value")
    source = tmp_path / "tasks.json"
    write(source, {"libreoffice_calc": ["c1", "c2", "c3"], "libreoffice_writer": ["w1", "w2"]})
    task = {"id": "office", "engine": "osworld_verified", "checkout": str(checkout),
            "revision": "a" * 40, "tasks_path": str(source), "expected_count": 5,
            "subset_count": 2, "seed": 42, "model_revision": "step-100",
            "domains": ["libreoffice_calc", "libreoffice_writer"],
            "model_env": "SCENE_MODEL", "base_url_env": "SCENE_URL", "api_key_env": "SCENE_KEY"}
    return task, tmp_path / "out"


def export_scores(output, plan, records, attempt=0, **extra):
    write(output / f"attempt-{attempt:03d}" / "native_scores.json",
          {"selection_fingerprint": plan["selection_fingerprint"], "records": records, **extra})
    write(output / f"attempt-{attempt:03d}" / "execution.json",
          {"selection_fingerprint": plan["selection_fingerprint"], "returncode": 0})


def browser(configured):
    task, output = configured
    task.update(id="browser", engine="webarena_verified", expected_count=3, subset_count=2)
    task.pop("domains")
    write(Path(task["tasks_path"]), [{"task_id": i, "sites": ["shopping"], "intent": "Find the price"} for i in range(3)])
    native_config = output.parent / "environment.json"
    write(native_config, {"environments": {"__SHOPPING__": {"urls": ["http://localhost:7770"]}}})
    task["native_config_file"] = str(native_config)
    task["native_results_path"] = str(output.parent / "native_results")
    task["native_run_manifest_file"] = str(output.parent / "native_run.json")
    return task, output


def artifact_manifest(task, plan):
    write(Path(task["native_run_manifest_file"]), {key: plan[key] for key in
          ("engine", "revision", "model_revision", "split", "attempts", "task_ids", "selection_fingerprint")})
    for key in plan["task_ids"]:
        root = Path(task["native_results_path"]) / key
        write(root / "agent_response.json", {"task_type": "RETRIEVE", "status": "SUCCESS", "retrieved_data": []})
        write(root / "network.har", {"log": {"entries": []}})


def test_native_subset_is_stable_stratified_and_full_ignores_subset_file(configured):
    task, output = configured
    small = scenarios.prepare_scenario(task, "subset", output)
    assert small["ready"] and small["selected_count"] == 2
    assert {key.split("/")[0] for key in small["task_ids"]} == {"libreoffice_calc", "libreoffice_writer"}
    assert scenarios.prepare_scenario(task, "subset", output)["task_ids"] == small["task_ids"]
    task["subset_ids_file"] = str(output.parent / "does-not-exist.json")
    full = scenarios.prepare_scenario(task, "full", output.parent / "full")
    assert full["selected_count"] == 5 and not full["complete"]


def test_native_expected_count_and_explicit_unknown_ids_are_rejected(configured):
    task, output = configured
    task["expected_count"] = 6
    with pytest.raises(ValueError, match="coverage mismatch"):
        scenarios.prepare_scenario(task, "full", output)
    task["expected_count"] = 5
    ids = output.parent / "ids.json"
    write(ids, ["libreoffice_calc/not-in-native-metadata"])
    task["subset_ids_file"] = str(ids)
    with pytest.raises(ValueError, match="unknown task IDs"):
        scenarios.prepare_scenario(task, "subset", output)


def test_explicit_subset_must_match_declared_subset_count(configured):
    task, output = configured
    task["subset_count"] = 5
    ids = output.parent / "ids.json"
    write(ids, ["libreoffice_calc/c1"])
    task["subset_ids_file"] = str(ids)
    with pytest.raises(ValueError, match="subset_count requires 5 IDs, found 1"):
        scenarios.prepare_scenario(task, "subset", output)


def test_zero_is_model_failure_but_missing_and_infra_are_not_scores(configured):
    task, output = configured
    plan = scenarios.prepare_scenario(task, "full", output)
    ids = plan["task_ids"]
    export_scores(output, plan, [{"task_id": ids[0], "status": "scored", "score": 0},
                                {"task_id": ids[1], "status": "scored", "score": 1},
                                {"task_id": ids[2], "status": "infra_error"}])
    result = scenarios.score_scenario(task, "full", output)
    assert result["completed_samples"] == 2 and result["missing_samples"] == 3
    assert result["mean_on_scored"] == 0.5 and result["final_mean_score"] is None
    assert not result["complete"] and not result["full_coverage"]
    export_scores(output, plan, [{"task_id": key, "status": "scored", "score": 0} for key in ids])
    result = scenarios.score_scenario(task, "full", output)
    assert result["complete"] and result["full_coverage"] and result["final_mean_score"] == 0


def test_model_task_and_native_score_provenance_cannot_be_reused(configured):
    task, output = configured
    plan = scenarios.prepare_scenario(task, "subset", output)
    wrong_model = {**task, "model_revision": "step-200"}
    with pytest.raises(ValueError, match="does not match"):
        scenarios.score_scenario(wrong_model, "subset", output)
    write(output / "attempt-000" / "native_scores.json", {"selection_fingerprint": "stale", "records": []})
    write(output / "attempt-000" / "execution.json", {"selection_fingerprint": plan["selection_fingerprint"], "returncode": 0})
    with pytest.raises(ValueError, match="provenance differs"):
        scenarios.score_scenario(task, "subset", output)
    export_scores(output, plan, [{"task_id": "unknown-task", "status": "scored", "score": 1}])
    with pytest.raises(ValueError, match="unknown or duplicate"):
        scenarios.score_scenario(task, "subset", output)


def test_browser_requires_matching_model_and_task_artifact_manifest(configured):
    task, output = browser(configured)
    plan = scenarios.prepare_scenario(task, "subset", output)
    result = scenarios.run_scenario(task, "subset", output)
    assert result["status"] == "needs_native_artifacts" and not result["complete"]
    artifact_manifest(task, plan)
    manifest_path = Path(task["native_run_manifest_file"])
    manifest = json.loads(manifest_path.read_text())
    manifest["model_revision"] = "wrong-checkpoint"
    write(manifest_path, manifest)
    with pytest.raises(ValueError, match="model_revision"):
        scenarios.score_scenario(task, "subset", output)
    manifest["model_revision"] = task["model_revision"]
    manifest["task_ids"] = ["0"]
    write(manifest_path, manifest)
    with pytest.raises(ValueError, match="task_ids"):
        scenarios.score_scenario(task, "subset", output)


def test_browser_missing_trace_cannot_count_old_scores_as_complete(configured):
    task, output = browser(configured)
    plan = scenarios.prepare_scenario(task, "subset", output)
    artifact_manifest(task, plan)
    export_scores(output, plan, [{"task_id": key, "status": "scored", "score": 1} for key in plan["task_ids"]])
    assert scenarios.score_scenario(task, "subset", output)["complete"]
    (Path(task["native_results_path"]) / plan["task_ids"][0] / "network.har").unlink()
    result = scenarios.score_scenario(task, "subset", output)
    assert not result["complete"] and result["completed_samples"] == 0


def test_changed_native_artifacts_require_reevaluation(configured):
    task, output = browser(configured)
    plan = scenarios.prepare_scenario(task, "subset", output)
    artifact_manifest(task, plan)
    path = Path(task["native_results_path"]) / plan["task_ids"][0] / "network.har"
    export_scores(output, plan, [], artifact_sha256={str(path): scenarios._file_hash(path)})
    write(path, {"log": {"entries": ["new"]}})
    with pytest.raises(ValueError, match="artifact changed"):
        scenarios.score_scenario(task, "subset", output)


def test_missing_checkout_or_dependency_never_starts_model_process(configured, monkeypatch):
    task, output = configured
    monkeypatch.setattr(scenarios, "_dependency_errors", lambda *_: ["Native dependency missing: desktop_env"])
    monkeypatch.setattr(scenarios.subprocess, "run", lambda *_a, **_k: pytest.fail("No process should start"))
    result = scenarios.run_scenario(task, "subset", output)
    assert not result["ready"] and not result["complete"]


@pytest.mark.parametrize("execution", [{"returncode": 2}, {"returncode": None, "status": "timeout"},
                                      {"returncode": None, "status": "running"}])
def test_failed_native_process_cannot_reuse_old_success_scores(configured, execution):
    task, output = configured
    plan = scenarios.prepare_scenario(task, "subset", output)
    export_scores(output, plan, [{"task_id": key, "status": "scored", "score": 1} for key in plan["task_ids"]])
    write(output / "attempt-000" / "execution.json", {**execution, "selection_fingerprint": plan["selection_fingerprint"]})
    result = scenarios.score_scenario(task, "subset", output)
    assert not result["complete"] and result["completed_samples"] == 0 and result["final_mean_score"] is None


def test_execution_from_another_plan_is_rejected(configured):
    task, output = configured
    plan = scenarios.prepare_scenario(task, "subset", output)
    export_scores(output, plan, [])
    write(output / "attempt-000" / "execution.json", {"returncode": 0, "selection_fingerprint": "wrong"})
    with pytest.raises(ValueError, match="execution provenance"):
        scenarios.score_scenario(task, "subset", output)


def test_preflight_detects_absent_native_artifacts_before_launch(configured, monkeypatch):
    task, output = browser(configured)
    monkeypatch.setattr(scenarios.subprocess, "run", lambda *_a, **_k: pytest.fail("No native process should start"))
    result = scenarios.preflight_scenario(task, "subset", output)
    assert not result["ready"] and result["status"] == "needs_native_artifacts"


def test_inline_credentials_and_opaque_commands_are_rejected(configured):
    task, output = configured
    for extra in ({"api_key": "bad"}, {"settings": {"password": "bad"}}, {"command": "anything"}):
        with pytest.raises(ValueError, match="inline secrets"):
            scenarios.prepare_scenario({**task, **extra}, "subset", output)


def test_native_logs_and_plans_never_save_api_key(configured, monkeypatch):
    task, output = configured
    monkeypatch.setattr(scenarios.subprocess, "run", lambda *_a, **_k: SimpleNamespace(
        stdout="response secret-scene-test-value", stderr="failure", returncode=2))
    result = scenarios.run_scenario(task, "subset", output)
    assert not result["complete"]
    for path in output.rglob("*"):
        if path.is_file():
            assert "secret-scene-test-value" not in path.read_text()
    assert "[REDACTED]" in (output / "attempt-000" / "native.log").read_text()


def test_osworld_bridge_uses_configured_model_and_native_meta_not_gpt_default(configured, monkeypatch):
    task, output = configured
    plan = scenarios.prepare_scenario(task, "subset", output)
    destination = output / "attempt-000"
    destination.mkdir()
    class PromptAgent:
        pass
    calls = []
    monkeypatch.setitem(sys.modules, "mm_agents", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mm_agents.agent", SimpleNamespace(PromptAgent=PromptAgent))
    monkeypatch.setattr(scenarios, "_client", lambda _: SimpleNamespace(
        complete=lambda messages, parameters: calls.append((messages, parameters)) or {"text": "DONE"}))
    def native_run(path, run_name):
        assert sys.argv[sys.argv.index("--model") + 1] == "Qwen3.8-27B-step-100"
        meta = json.loads(Path(sys.argv[sys.argv.index("--test_all_meta_path") + 1]).read_text())
        assert sum(map(len, meta.values())) == 2
        assert PromptAgent().call_llm({"messages": [{"role": "user", "content": "task"}], "max_tokens": 123}) == "DONE"
        result_dir = Path(sys.argv[sys.argv.index("--result_dir") + 1])
        for key in plan["task_ids"]:
            path = result_dir / "pyautogui" / "screenshot" / "Qwen3.8-27B-step-100" / key / "result.txt"
            path.parent.mkdir(parents=True)
            path.write_text("1")
    monkeypatch.setattr(scenarios.runpy, "run_path", native_run)
    records = scenarios._native_osworld(plan, 0, destination)
    assert len(records) == 2 and all(item["score"] == 1 for item in records)
    assert calls[0][1] == {"max_tokens": 123}


def test_spreadsheet_recalc_failure_and_actual_case_count(configured, monkeypatch):
    task, output = configured
    task.update(id="sheets", engine="spreadsheetbench_verified", expected_count=1, subset_count=1,
                recalculation_backend="libreoffice", soffice=sys.executable,
                native_results_path=str(output.parent / "artifacts"))
    task.pop("domains")
    write(Path(task["tasks_path"]), [{"id": "s1", "spreadsheet_path": "spreadsheet/s1",
          "instruction_type": "Cell-Level Manipulation", "answer_position": "Sheet1!A1"}])
    gold = output.parent / "spreadsheet" / "s1" / "1_s1_answer.xlsx"
    gold.parent.mkdir(parents=True)
    gold.write_bytes(b"gold fixture")
    gold.with_name("1_s1_input.xlsx").write_bytes(b"input fixture")
    artifact = Path(task["native_results_path"]) / "s1" / "1_s1_output.xlsx"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"prediction fixture")
    plan = scenarios.prepare_scenario(task, "subset", output)
    calls = []
    recalc_ok = [False]
    def module(name, path):
        if "recalc" in name:
            return SimpleNamespace(find_libreoffice=lambda: sys.executable,
                                   just_open_libreoffice=lambda *_: recalc_ok[0])
        return SimpleNamespace(compare_workbooks=lambda *args: calls.append(args) or (True, ""))
    monkeypatch.setattr(scenarios, "_load_module", module)
    first = scenarios._native_spreadsheet(plan, 0, output / "attempt-000")
    assert first[0]["status"] == "infra_error" and not calls
    recalc_ok[0] = True
    second = scenarios._native_spreadsheet(plan, 0, output / "attempt-000")
    assert second[0]["score"] == 1 and len(calls) == 1  # Verified instance has one case, not hard-coded three.
    assert artifact.read_bytes() == b"prediction fixture" and gold.read_bytes() == b"gold fixture"


def test_android_uses_native_state_scores_and_generic_model_bridge(configured, monkeypatch):
    task, output = configured
    task.update(id="phone", engine="androidworld", expected_count=2, subset_count=2,
                adb_path=str(Path(sys.executable)))
    task.pop("domains")
    write(Path(task["tasks_path"]), [{"task_name": "ClockStopWatchPaused", "difficulty": "easy"},
                                    {"task_name": "MarkorCreateNote", "difficulty": "medium"}])
    plan = scenarios.prepare_scenario(task, "subset", output)
    destination = output / "attempt-000"
    calls = []
    class NativeCheckpointer:
        def __init__(self, directory):
            self.directory = directory
        def save_episodes(self, episodes, name):
            pass
    class NativeRegistry:
        ANDROID_WORLD_FAMILY = "android_world"
        def get_registry(self, family):
            assert family == "android_world"
            return dict.fromkeys(plan["task_ids"])
    def create_suite(registry, **kwargs):
        assert kwargs["tasks"] == plan["task_ids"] and kwargs["n_task_combinations"] == 1
        return SimpleNamespace()
    env = SimpleNamespace(close=lambda: calls.append("closed"))
    def native_run(suite, agent, checkpointer, demo_mode):
        assert agent.wrapper.predict("Observe screen")[0] == "action"
        checkpointer.save_episodes([{"task_template": plan["task_ids"][0], "is_successful": 0,
                                    "exception_info": None}], plan["task_ids"][0] + "_0")
        checkpointer.save_episodes([{"task_template": plan["task_ids"][1], "is_successful": float("nan"),
                                    "exception_info": "environment down"}], plan["task_ids"][1] + "_0")
    monkeypatch.setitem(sys.modules, "android_world", SimpleNamespace(
        checkpointer=SimpleNamespace(IncrementalCheckpointer=NativeCheckpointer),
        registry=SimpleNamespace(TaskRegistry=NativeRegistry),
        suite_utils=SimpleNamespace(create_suite=create_suite, run=native_run)))
    monkeypatch.setitem(sys.modules, "android_world.agents", SimpleNamespace(
        m3a=SimpleNamespace(M3A=lambda e, wrapper: SimpleNamespace(env=e, wrapper=wrapper))))
    monkeypatch.setitem(sys.modules, "android_world.env", SimpleNamespace(
        env_launcher=SimpleNamespace(load_and_setup_env=lambda **kwargs: env)))
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=None))
    monkeypatch.setattr(scenarios, "_client", lambda _: SimpleNamespace(
        complete=lambda messages, parameters: {"text": "action"}))
    records = scenarios._native_android(plan, 0, destination)
    assert records[0]["status"] == "scored" and records[0]["score"] == 0
    assert records[1]["status"] == "infra_error" and records[1]["score"] is None
    assert calls == ["closed"]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1, "1"])
def test_invalid_native_scores_are_rejected(configured, score):
    task, output = configured
    plan = scenarios.prepare_scenario(task, "subset", output)
    export_scores(output, plan, [{"task_id": plan["task_ids"][0], "status": "scored", "score": score}])
    with pytest.raises(ValueError, match="finite"):
        scenarios.score_scenario(task, "subset", output)
