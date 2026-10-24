"""Behavioral checks for explicit coverage, replay, resume and safe config."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from qwenpt.eval import suite
from qwenpt.eval.cli import main


@pytest.fixture
def configured(tmp_path):
    rows = [{"benchmark": "choice", "question_id": str(i), "prompt": "Pick A or B",
             "answer": "A", "stratum": "x" if i < 4 else "y", "mock_response": "A"}
            for i in range(8)]
    bank = tmp_path / "bank.jsonl"
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = {"version": 1, "suite": "test", "bank": "bank.jsonl",
              "dataset": {"name": "test", "revision": "v1", "split": "test", "expected_counts": {"choice": 8}},
              "model": {"revision": "step-100", "base_url": "http://localhost:8000/v1", "model": "student",
                        "api_key_env": "TEST_BENCH_KEY"},
              "generation": {"max_tokens": 32}, "samples": 1,
              "subset": {"seed": 42, "counts": {"choice": 4}, "stratify_by": "stratum"},
              "benchmarks": {"choice": {"scorer": {"type": "multiple_choice"}}}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path, suite.load_config(path), rows


def test_subset_is_balanced_deterministic_and_row_order_independent(configured):
    _, config, rows = configured
    selected, manifest = suite.select_questions(config, rows, "subset")
    reversed_rows, _ = suite.select_questions(config, list(reversed(rows)), "subset")
    assert selected == reversed_rows
    assert [row["stratum"] for row in selected].count("x") == 2
    assert manifest["selected_counts"] == {"choice": 4}
    all_rows, full = suite.select_questions(config, rows, "full")
    assert len(all_rows) == 8 and full["selected_counts"] == full["available_counts"]
    assert full["selection_fingerprint"] != manifest["selection_fingerprint"]


def test_subset_limits_and_complete_bank_validation(configured):
    path, config, _ = configured
    config["subset"]["counts"]["choice"] = 9
    with pytest.raises(ValueError, match="exceeds"):
        suite.prepare(config, "subset")
    config["dataset"]["expected_counts"]["choice"] = 10
    with pytest.raises(ValueError, match="coverage"):
        suite.prepare(config, "full")
    with pytest.raises(SystemExit):
        main(["plan", "--config", str(path), "--split", "full", "--output", str(path.parent / "out"), "--limit", "1"])


def test_explicit_id_subset_and_invalid_ids(configured, tmp_path):
    _, config, _ = configured
    path = tmp_path / "ids.json"
    path.write_text('[{"benchmark":"choice","question_id":"7"}]', encoding="utf-8")
    config["subset"] = {"ids_file": str(path)}
    rows, selection = suite.prepare(config, "subset")
    assert [row["question_id"] for row in rows] == ["7"]
    assert selection["method"] == "explicit_ids"
    path.write_text('[{"benchmark":"choice","question_id":"missing"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown IDs"):
        suite.prepare(config, "subset")
    # A subset ID file never truncates full mode, nor is it required by full.
    assert len(suite.prepare(config, "full")[0]) == 8


def test_mock_full_subset_and_resume_are_separate(configured, tmp_path):
    _, config, _ = configured
    output = tmp_path / "run"
    planned = suite.plan(config, "subset", output)
    result = suite.run(config, "subset", output, "mock")
    assert result["complete"] and result["is_mock"] and not result["full_coverage"]
    contents = (output / "predictions.jsonl").read_bytes()
    assert suite.run(config, "subset", output, "mock")["complete"]
    assert contents == (output / "predictions.jsonl").read_bytes()
    graded = suite.score(config, "subset", output)
    assert graded["benchmarks"]["choice"]["final_mean_score"] == 1
    assert suite.score(config, "subset", output) == graded
    assert planned["selection_fingerprint"] == graded["selection_fingerprint"]
    with pytest.raises(ValueError, match="differs"):
        suite.run(config, "full", output, "mock")
    full = tmp_path / "full"
    assert suite.run(config, "full", full, "mock")["full_coverage"]
    assert suite.score(config, "full", full)["full_coverage"]


def test_changed_generation_revision_and_bank_reject_resume(configured, tmp_path):
    _, config, _ = configured
    output = tmp_path / "run"
    suite.run(config, "full", output, "mock")
    changed = deepcopy(config)
    changed["model"]["revision"] = "step-200"
    with pytest.raises(ValueError, match="differs"):
        suite.run(changed, "full", output, "mock")
    with pytest.raises(ValueError, match="revision differs"):
        suite.score(changed, "full", output)
    changed = deepcopy(config)
    changed["generation"]["max_tokens"] = 64
    with pytest.raises(ValueError, match="parameters differ"):
        suite.score(changed, "full", output)
    with Path(config["bank"]).open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="differs"):
        suite.run(config, "full", output, "mock")


def test_partial_api_failure_is_not_full_and_retry_only_missing(configured, tmp_path, monkeypatch):
    _, config, _ = configured
    monkeypatch.setenv("TEST_BENCH_KEY", "never-save-this-key")
    class Client:
        fail = True
        calls = 0
        def complete(self, *_):
            self.calls += 1
            if self.fail:
                self.fail = False
                raise RuntimeError("raw provider error never-save-this-key")
            return {"text": "A", "reasoning": "", "finish_reason": "stop", "usage": {}, "response_model": "student"}
    client = Client()
    output = tmp_path / "partial"
    summary = suite.run(config, "full", output, client=client)
    assert not summary["complete"] and not summary["full_coverage"]
    assert summary["benchmarks"]["choice"]["missing_samples"] == 1
    graded = suite.score(config, "full", output)
    assert graded["benchmarks"]["choice"]["final_mean_score"] is None
    assert suite.run(config, "full", output, client=client)["complete"]
    assert client.calls == 9
    assert suite.score(config, "full", output)["complete"]
    for path in output.rglob("*.json*"):
        assert "never-save-this-key" not in path.read_text(encoding="utf-8")


def test_missing_key_fails_before_generating_artifacts(configured, tmp_path, monkeypatch):
    _, config, _ = configured
    monkeypatch.delenv("TEST_BENCH_KEY", raising=False)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="key environment variable"):
        suite.run(config, "full", output)
    assert not output.exists()


def test_judge_regrading_uses_separate_tag(configured, tmp_path, monkeypatch):
    _, config, _ = configured
    output = tmp_path / "run"
    suite.run(config, "subset", output, "mock")
    config["benchmarks"]["choice"]["scorer"] = {"type": "llm_judge", "rubric": "Check answer against reference."}
    config["judge"] = {"revision": "judge-v1", "model": "judge", "base_url": "http://localhost:9999/v1", "api_key_env": "TEST_JUDGE_KEY"}
    monkeypatch.setenv("TEST_JUDGE_KEY", "judge-secret-value")
    class Judge:
        calls = 0
        def complete(self, *_):
            self.calls += 1
            return {"text": '{"score":1,"reason":"matches"}', "finish_reason": "stop", "usage": {}, "response_model": "judge"}
    judge = Judge()
    assert suite.score(config, "subset", output, "judge-v1", judge)["complete"]
    assert judge.calls == 4
    suite.score(config, "subset", output, "judge-v1", judge)
    assert judge.calls == 4
    config["judge"]["revision"] = "judge-v2"
    with pytest.raises(ValueError, match="differs"):
        suite.score(config, "subset", output, "judge-v1", judge)
    assert suite.score(config, "subset", output, "judge-v2", judge)["complete"]
    assert judge.calls == 8
    for path in output.rglob("*.json*"):
        assert "judge-secret-value" not in path.read_text(encoding="utf-8")


def test_predictions_tampering_and_jsonl_damage_fail(configured, tmp_path):
    _, config, _ = configured
    output = tmp_path / "run"
    suite.run(config, "full", output, "mock")
    suite.score(config, "full", output)
    path = output / "predictions.jsonl"
    records = suite.read_records(path)
    records[0]["text"] = "B"
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    with pytest.raises(ValueError, match="cached grade"):
        suite.score(config, "full", output)
    path.write_bytes(b'{"ok":true}\n{"interrupted')
    assert suite.read_records(path) == [{"ok": True}]
    path.write_bytes(b'{"ok":true}\nINVALID\n')
    with pytest.raises(ValueError, match="Corrupt completed"):
        suite.read_records(path)


def test_config_rejects_secrets_typos_invalid_yaml_and_placeholders(configured, tmp_path):
    source, config, _ = configured
    original = yaml.safe_load(source.read_text(encoding="utf-8"))
    for field in ("api_key", "token", "apiKey", "password"):
        broken = deepcopy(original)
        broken["model"][field] = "secret-not-in-error"
        source.write_text(yaml.safe_dump(broken), encoding="utf-8")
        with pytest.raises(ValueError) as exc:
            suite.load_config(source)
        assert "secret-not-in-error" not in str(exc.value)
    source.write_text('api_key: [secret-not-in-error\n', encoding="utf-8")
    with pytest.raises(ValueError, match="content omitted"):
        suite.load_config(source)
    config["model"]["revision"] = "FILL_WITH_CHECKPOINT"
    with pytest.raises(ValueError, match="placeholder"):
        suite.run(config, "full", tmp_path / "run", "mock")


def test_sample_precedence_and_env_file(configured, tmp_path, monkeypatch):
    _, config, _ = configured
    config["samples"] = 2
    config["benchmarks"]["choice"]["samples"] = 3
    assert suite.run(config, "subset", tmp_path / "run", "mock")["benchmarks"]["choice"]["expected_samples"] == 12
    env = tmp_path / "local.env"
    env.write_text('TEST_BENCH_KEY="from-file"\nTEST_LITERAL=${NO_EXPANSION}\n', encoding="utf-8")
    monkeypatch.setenv("TEST_BENCH_KEY", "already-set")
    monkeypatch.delenv("TEST_LITERAL", raising=False)
    suite.load_env(env)
    import os
    assert os.environ["TEST_BENCH_KEY"] == "already-set"
    assert os.environ["TEST_LITERAL"] == "${NO_EXPANSION}"
    monkeypatch.delenv("TEST_LITERAL")


def test_duplicate_ids_and_tool_messages_fail_preflight(configured):
    _, config, rows = configured
    bank = Path(config["bank"])
    bank.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        suite.load_bank(config)
    rows[0]["messages"] = [{"role": "tool", "content": "result"}]
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="message role"):
        suite.load_bank(config)


def test_end_to_end_http_transport_and_rule_score(configured, tmp_path, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    _, config, _ = configured
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, self.headers["Authorization"], body))
            response = json.dumps({"model": "student", "choices": [{"message": {
                "role": "assistant", "content": "Answer: A", "reasoning_content": "private reasoning"},
                "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 3}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config["model"]["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"
        monkeypatch.setenv("TEST_BENCH_KEY", "local-http-test-key")
        output = tmp_path / "http-run"
        assert suite.run(config, "subset", output)["complete"]
        summary = suite.score(config, "subset", output)
        assert summary["benchmarks"]["choice"]["final_mean_score"] == 1
        assert len(received) == 4
        for path, authorization, body in received:
            assert path == "/v1/chat/completions" and authorization == "Bearer local-http-test-key"
            assert body["model"] == "student" and body["n"] == 1 and body["stream"] is False
            assert "answer" not in body and "api_key" not in body
        for path in output.rglob("*.json*"):
            assert "local-http-test-key" not in path.read_text(encoding="utf-8")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_budget_exhausted_without_final_answer_is_not_missing(configured, tmp_path, monkeypatch):
    from qwenpt.eval.api_backend import _parse_response
    _, config, _ = configured
    monkeypatch.setenv("TEST_BENCH_KEY", "fake-test-key")
    response = _parse_response({"choices": [{"message": {"role": "assistant", "content": None,
                               "reasoning_content": "unfinished reasoning"}, "finish_reason": "length"}]})
    class Client:
        def complete(self, *_):
            return response
    output = tmp_path / "truncated"
    result = suite.run(config, "subset", output, client=Client())
    assert result["complete"]
    assert result["benchmarks"]["choice"]["length_limited_samples"] == 4
    result = suite.score(config, "subset", output)
    assert result["complete"] and result["benchmarks"]["choice"]["final_mean_score"] == 0
