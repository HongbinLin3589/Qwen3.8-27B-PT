import hashlib
import json

import pytest
import yaml

from qwenpt.eval.prepare_general import normalize_gpqa, normalize_matharena, prepare_general
from qwenpt.eval.suite import load_config, prepare


def test_gpqa_stable_options_and_science_domain():
    rows = [{"Record ID": str(i), "Question": "science", "Correct Answer": "correct",
             "Incorrect Answer 1": "x", "Incorrect Answer 2": "y", "Incorrect Answer 3": "z",
             "Subdomain": "Chemistry"} for i in range(6)]
    forward = normalize_gpqa(rows)
    reverse = normalize_gpqa(rows[::-1])
    assert {row["question_id"]: row for row in forward} == {row["question_id"]: row for row in reverse}
    assert all(row["options"][row["answer"]] == "correct" for row in forward)
    assert forward[0]["metadata"]["stratum"] == "Chemistry"


def test_matharena_keeps_symbolic_answer_instead_of_aime_integer_cast():
    rows = normalize_matharena([{"problem_idx": 4, "problem": "Solve", "answer": r"-\frac{1}{21}"}], "hmmt_feb_2026")
    assert rows[0]["answer"] == r"-\frac{1}{21}"


def source_config(tmp_path):
    data = tmp_path / "rows.json"
    data.write_text(json.dumps([{"problem_idx": i, "problem": f"Question {i}", "answer": "1"} for i in range(3)]))
    source = {"kind": "matharena", "repository": "fixture/math", "revision": "a" * 40,
              "files": ["rows.json"], "local_files": ["rows.json"],
              "file_hashes": {"rows.json": {"sha256": hashlib.sha256(data.read_bytes()).hexdigest()}},
              "release_count": 3, "expected_count": 3, "subset_count": 2, "split": "test"}
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "sources": {"math": source}}))
    return path


def test_preparation_freezes_source_and_independent_subset_full(tmp_path):
    path = source_config(tmp_path)
    result = prepare_general(path, tmp_path / "prepared", model_revision="checkpoint-7")
    assert result["complete"]
    config = load_config(result["tasks"]["math"]["config"])
    assert len(prepare(config, "subset")[0]) == 2
    assert len(prepare(config, "full")[0]) == 3
    assert prepare_general(path, tmp_path / "prepared", model_revision="checkpoint-7")["complete"]


def test_same_count_wrong_source_is_rejected_before_freezing(tmp_path):
    path = source_config(tmp_path)
    data = tmp_path / "rows.json"
    data.write_text(data.read_text().replace("Question 1", "Different!"))
    result = prepare_general(path, tmp_path / "prepared")
    assert not result["complete"]
    assert "upstream hash" in result["tasks"]["math"]["message"]
    assert not (tmp_path / "prepared/math.jsonl").exists()


def test_unknown_preparation_task_is_not_silently_ignored(tmp_path):
    with pytest.raises(ValueError, match="Unknown"):
        prepare_general(source_config(tmp_path), tmp_path / "prepared", tasks=["not-there"])
