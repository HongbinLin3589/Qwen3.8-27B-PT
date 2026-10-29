"""Transport-level sampling independence and retry/resume compatibility."""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread

import pytest

from qwenpt.eval import suite


def configuration(tmp_path):
    bank = tmp_path / "bank.jsonl"
    bank.write_text(json.dumps({"benchmark": "choice", "question_id": "stable-question",
                               "prompt": "Pick A or B", "answer": "A", "mock_response": "A"}) + "\n")
    return {"version": 1, "suite": "sampling-test", "bank": str(bank),
            "dataset": {"name": "test", "revision": "test-v1", "split": "test", "expected_counts": {"choice": 1}},
            "model": {"revision": "checkpoint-1", "base_url": "http://127.0.0.1:8000/v1",
                      "model": "student", "api_key_required": False, "retries": 1},
            "generation": {"seed": 20261024, "temperature": 0.8, "max_tokens": 32},
            "samples": 3, "subset": {"seed": 7, "counts": {"choice": 1}},
            "benchmarks": {"choice": {"scorer": {"type": "multiple_choice"}}}}


def test_same_question_gets_unique_seeds_but_transient_retry_and_resume_reuse_seed(tmp_path, monkeypatch):
    config, requests = configuration(tmp_path), []
    monkeypatch.setattr("qwenpt.eval.api_backend.time.sleep", lambda *_: None)
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            if len(requests) == 1:
                self.send_response(429)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = json.dumps({"model": "student", "system_fingerprint": "backend-version-1",
                "choices": [{"message": {"role": "assistant", "content": "Answer: A"}, "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config["model"]["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"
        output = tmp_path / "run"
        assert suite.run(config, "full", output)["complete"]
        assert len(requests) == 4
        assert requests[0] == requests[1]  # HTTP retry is the very same sample.
        seeds = [item["seed"] for item in requests[1:]]
        assert len(set(seeds)) == 3
        records = suite.read_records(output / "predictions.jsonl")
        assert [item["request_seed"] for item in records] == seeds
        assert all(item["system_fingerprint"] == "backend-version-1" for item in records)
        # Simulate an interruption just before the final sample was persisted.
        (output / "predictions.jsonl").write_text("".join(json.dumps(item) + "\n" for item in records[:2]))
        assert suite.run(config, "full", output)["complete"]
        assert len(requests) == 5 and requests[-1] == requests[3]
        assert suite.score(config, "full", output)["benchmarks"]["choice"]["final_mean_score"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_seed_mapping_is_order_independent_and_stable_across_splits_and_checkpoints(tmp_path):
    config = configuration(tmp_path)
    params = config["generation"]
    keys = [("choice", question, index) for question in ("q1", "q2") for index in range(32)]
    forward = {key: suite.sample_parameters(params, *key)["seed"] for key in keys}
    reverse = {key: suite.sample_parameters(params, *key)["seed"] for key in reversed(keys)}
    assert forward == reverse
    assert len({forward[key] for key in keys if key[1] == "q1"}) == 32
    assert forward[("choice", "q1", 0)] != forward[("choice", "q2", 0)]
    assert all(type(seed) is int and 0 <= seed < 2 ** 31 for seed in forward.values())
    suite.run(config, "subset", tmp_path / "subset", "mock")
    changed = deepcopy(config)
    changed["model"]["revision"] = "checkpoint-2"
    suite.run(changed, "full", tmp_path / "full", "mock")
    first = suite.read_records(tmp_path / "subset/predictions.jsonl")
    second = suite.read_records(tmp_path / "full/predictions.jsonl")
    assert [item["request_seed"] for item in first] == [item["request_seed"] for item in second]


@pytest.mark.parametrize("parameters", [{}, {"seed": None}])
def test_seed_is_not_forced_on_providers_without_seed_support(parameters):
    assert "seed" not in suite.sample_parameters(parameters, "task", "question", 0)


def test_extra_body_seed_cannot_bypass_derived_sampling(tmp_path):
    config = configuration(tmp_path)
    config["generation"] = {"extra_body": {"seed": 7, "chat_template_kwargs": {"enable_thinking": True}}}
    params = suite.generation_parameters(config)["choice"]
    requests = [suite.sample_parameters(params, "choice", "q", i) for i in range(3)]
    assert len({request["seed"] for request in requests}) == 3
    assert all("extra_body" not in request and request["chat_template_kwargs"]["enable_thinking"] for request in requests)


@pytest.mark.parametrize("override", [{"seed": 9}, {"extra_body": {"seed": 9}}, {"seed": None}, {"extra_body": {"seed": None}}])
def test_benchmark_seed_override_works_across_direct_and_sdk_style_parameters(tmp_path, override):
    config = configuration(tmp_path)
    config["generation"] = {"extra_body": {"seed": 7, "chat_template_kwargs": {"enable_thinking": True}}}
    config["benchmarks"]["choice"]["generation"] = override
    params = suite.generation_parameters(config)["choice"]
    expected = override.get("seed", override.get("extra_body", {}).get("seed"))
    assert params.get("seed") == expected
    assert params["chat_template_kwargs"] == {"enable_thinking": True}


@pytest.mark.parametrize("seed", [True, "7", -1, 2 ** 63, 0.25])
def test_invalid_seed_fails_before_generation(tmp_path, seed):
    config = configuration(tmp_path)
    config["generation"]["seed"] = seed
    with pytest.raises(ValueError, match="seed"):
        suite.run(config, "full", tmp_path / "out", "mock")
    assert not (tmp_path / "out").exists()


def test_cached_seed_cannot_be_changed_and_old_fixed_seed_averages_require_regeneration(tmp_path):
    config = configuration(tmp_path)
    output = tmp_path / "run"
    suite.run(config, "full", output, "mock")
    path = output / "predictions.jsonl"
    records = suite.read_records(path)
    records[0]["request_seed"] += 1
    path.write_text("".join(json.dumps(item) + "\n" for item in records))
    with pytest.raises(ValueError, match="seed differs"):
        suite.score(config, "full", output)
    with pytest.raises(ValueError, match="seed differs"):
        suite.run(config, "full", output, "mock")
    legacy = {"generation": {"choice": {"seed": 7}}, "samples": {"choice": 3}}
    with pytest.raises(ValueError, match="Legacy cached samples"):
        suite._validate_sampling({}, legacy)
    suite._validate_sampling({}, {"generation": {"choice": {}}, "samples": {"choice": 3}})


def test_filtered_partial_correct_answer_counts_zero_and_refusal_is_preserved(tmp_path):
    config = configuration(tmp_path)
    config["samples"] = 1
    class Filtered:
        def complete(self, *_):
            return {"text": "Answer: A", "finish_reason": "content_filter", "reasoning": "", "usage": {}, "refusal": None}
    output = tmp_path / "filtered"
    result = suite.run(config, "full", output, client=Filtered())
    assert result["complete"] and result["benchmarks"]["choice"]["content_filtered_samples"] == 1
    scores = suite.score(config, "full", output)
    assert scores["complete"] and scores["benchmarks"]["choice"]["final_mean_score"] == 0
    assert scores["benchmarks"]["choice"]["metrics"]["accuracy"]["mean_on_scored"] == 0
    class Refused:
        def complete(self, *_):
            return {"text": "", "finish_reason": "stop", "refusal": "I cannot answer.", "reasoning": "", "usage": {}}
    output = tmp_path / "refused"
    result = suite.run(config, "full", output, client=Refused())
    assert result["complete"] and result["benchmarks"]["choice"]["refused_samples"] == 1
    assert suite.score(config, "full", output)["benchmarks"]["choice"]["final_mean_score"] == 0
