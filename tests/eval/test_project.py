import json
from pathlib import Path

import pytest
import yaml

from qwenpt.eval.project import catalog, execute_project, load_project


def fixture_project(tmp_path, *, missing=False):
    rows = [{"benchmark": "science", "question_id": str(i), "prompt": "Pick A", "answer": "A",
             "mock_response": "Answer: A", "metadata": {"stratum": str(i % 2)}} for i in range(4)]
    bank = tmp_path / "science.jsonl"
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = {"version": 1, "suite": "science", "bank": "science.jsonl",
              "dataset": {"name": "fixture", "revision": "fixture-v1", "split": "test", "expected_counts": {"science": 4}},
              "model": {"revision": "fixture-checkpoint"}, "generation": {}, "samples": 1,
              "subset": {"seed": 42, "counts": {"science": 2}},
              "benchmarks": {"science": {"scorer": {"type": "multiple_choice"}}}}
    (tmp_path / "science.yaml").write_text(yaml.safe_dump(config))
    tasks = {"science": {"kind": "bank", "profile": "core", "capability": "science_reasoning",
                          "subset_count": 2, "full_count": 4, "config": "science.yaml"}}
    if missing:
        tasks["code"] = {"kind": "bank", "profile": "core", "capability": "code_execution",
                         "subset_count": 1, "full_count": 2, "config": "missing-code.yaml"}
    project = tmp_path / "project.yaml"
    project.write_text(yaml.safe_dump({"version": 1, "project": "unit-fixture", "tasks": tasks}))
    return load_project(project)


def test_project_has_general_capability_matrix():
    config = load_project(Path(__file__).parents[2] / "configs/eval/project.yaml")
    result = catalog(config)
    assert len(result["tasks"]) == 11
    assert not result["missing_required_tasks"]
    assert sum(task["subset_count"] for task in result["tasks"].values() if task["profile"] == "core") == 150
    assert sum(task["subset_count"] for task in result["tasks"].values() if task["profile"] == "scenarios") == 20
    assert sum(task["full_count"] for task in result["tasks"].values() if task["profile"] == "core") == 955


@pytest.mark.parametrize("split,count", [("subset", 2), ("full", 4)])
def test_project_non_math_end_to_end_and_mock_never_certifies_model(tmp_path, split, count):
    config = fixture_project(tmp_path)
    output = tmp_path / split
    result = execute_project(config, "run", split, output, profile="all", model_revision="checkpoint-7", backend="mock")
    assert result["complete"]
    assert result["tasks"]["science"]["benchmarks"]["science"]["selected_questions"] == count
    scores = execute_project(config, "score", split, output, profile="all", model_revision="checkpoint-7")
    assert scores["complete"]
    assert scores["tasks"]["science"]["benchmarks"]["science"]["final_mean_score"] == 1
    assert not scores["all_required_complete"]


def test_missing_task_stays_visible_and_prevents_any_generation(tmp_path, monkeypatch):
    config = fixture_project(tmp_path, missing=True)
    def forbidden(*args, **kwargs):
        pytest.fail("Paid/API generation started before all tasks passed preflight")
    monkeypatch.setattr("qwenpt.eval.project.suite.run", forbidden)
    result = execute_project(config, "run", "subset", tmp_path / "out", profile="core", model_revision="checkpoint-7")
    assert result["status"] == "preflight_failed"
    assert not result["complete"]
    assert set(result["tasks"]) == {"science", "code"}
    assert result["tasks"]["code"]["status"] == "blocked"


def test_model_or_split_change_requires_new_directory(tmp_path):
    config = fixture_project(tmp_path)
    execute_project(config, "plan", "subset", tmp_path / "out", profile="core", model_revision="checkpoint-7")
    with pytest.raises(ValueError, match="differs"):
        execute_project(config, "plan", "full", tmp_path / "out", profile="core", model_revision="checkpoint-7")
    with pytest.raises(ValueError, match="differs"):
        execute_project(config, "plan", "subset", tmp_path / "out", profile="core", model_revision="checkpoint-8")


def test_declared_full_count_cannot_mask_smaller_bank(tmp_path):
    config = fixture_project(tmp_path)
    config["tasks"]["science"]["full_count"] = 5
    result = execute_project(config, "plan", "full", tmp_path / "out", profile="core", model_revision="checkpoint-7")
    assert not result["complete"]
    assert "registered task/full count" in result["tasks"]["science"]["message"]


def test_explicit_frozen_bank_ids_keep_required_quota(tmp_path):
    config = fixture_project(tmp_path)
    path = tmp_path / "science.yaml"
    bank = yaml.safe_load(path.read_text())
    bank["subset"] = {"ids_file": "ids.json"}
    path.write_text(yaml.safe_dump(bank))
    ids = [{"benchmark": "science", "question_id": "1"}, {"benchmark": "science", "question_id": "3"}]
    (tmp_path / "ids.json").write_text(json.dumps(ids))
    result = execute_project(config, "plan", "subset", tmp_path / "out", profile="core", model_revision="checkpoint-7")
    assert result["complete"]
    (tmp_path / "ids.json").write_text(json.dumps(ids[:1]))
    result = execute_project(config, "plan", "subset", tmp_path / "new-out", profile="core", model_revision="checkpoint-7")
    assert not result["complete"]


def test_deleting_required_task_cannot_certify_whole_profile(tmp_path, monkeypatch):
    config = fixture_project(tmp_path)
    config["project"] = "qwen38-general-v1"
    result = execute_project(config, "plan", "full", tmp_path / "out", profile="all", model_revision="checkpoint-7")
    assert not result["complete"]
    assert "ifbench" in result["missing_required_tasks"]
    assert "androidworld" in result["missing_required_tasks"]


def test_harness_doctor_propagates_unready(tmp_path, monkeypatch):
    config = fixture_project(tmp_path)
    config["tasks"]["terminal"] = {"kind": "harness", "profile": "core", "capability": "terminal",
                                    "subset_count": 1, "full_count": 2, "settings": {"id": "terminal"}}
    monkeypatch.setattr("qwenpt.eval.harnesses.prepare_harness", lambda *args: {"task_ids": ["a"]})
    monkeypatch.setattr("qwenpt.eval.harnesses.preflight_harness", lambda *args: {"ready": False})
    result = execute_project(config, "doctor", "subset", tmp_path / "out", profile="core", model_revision="checkpoint-7")
    assert not result["complete"]
    assert result["tasks"]["terminal"]["status"] == "blocked"
