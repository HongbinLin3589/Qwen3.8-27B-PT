"""Exercise durable benchmark generation, resume rejection, and interrupted writes."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from legacy.benchmark import run_benchmark as module


def _score(text, benchmark, answer):
    return {"predicted": text, "correct": text == answer,
            "thinking_complete": True, "format_error": False}


@pytest.fixture
def fake_grader(monkeypatch):
    monkeypatch.setitem(sys.modules, "legacy.benchmark.scoring", SimpleNamespace(score_completion=_score))


def _args(tmp_path):
    return argparse.Namespace(
        bank=tmp_path / "bank.jsonl", model=tmp_path / "model", variant="step_100",
        benchmark="aime2024", output=tmp_path / "output", variant_fingerprint="adapter-hash",
        max_new_tokens=512, max_model_len=1024, max_num_seqs=8, max_num_batched_tokens=8192,
        gpu_memory_utilization=0.92, temperature=1.0, top_p=0.95, top_k=20,
        seed=20261021, batch_prompts=None, limit=None, samples_override=None, enforce_eager=False,
    )


def _row(question_id="first"):
    return {"benchmark": "aime2024", "question_id": question_id, "prompt": "Compute 1+1.",
            "answer": "2", "samples": 2, "metadata": {}}


def _group(args, row):
    return {"benchmark": args.benchmark, "question_id": row["question_id"], "answer": row["answer"],
            "variant": args.variant, "run_fingerprint": "run-hash", "prompt_tokens": 3,
            "samples": [{"sample_index": index, "text": "2", "generated_tokens": 1,
                         "finish_reason": "stop", **_score("2", args.benchmark, row["answer"])}
                        for index in range(row["samples"])]}


def test_generation_resumes_without_reloading_gpu(tmp_path, monkeypatch, fake_grader):
    args = _args(tmp_path)
    rows = [_row(), _row("second")]
    args.bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    args.model.mkdir()
    (args.model / "generation_config.json").write_text('{"eos_token_id": [7, 8]}')
    calls = {"loaded": 0, "seeds": [], "shutdown": []}

    class Tokenizer:
        chat_template = "template"

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] and kwargs["preserve_thinking"]
            assert kwargs["reasoning_effort"] == "xhigh"
            return [1, 2, 3]

    class Engine:
        def __init__(self, **kwargs):
            calls["loaded"] += 1
            assert kwargs["tensor_parallel_size"] == 1
            assert kwargs["generation_config"] == "vllm"
            self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(
                shutdown=lambda **kwargs: calls["shutdown"].append(kwargs)))

        def generate(self, prompts, sampling_params, use_tqdm):
            calls["seeds"].extend(param.seed for param in sampling_params)
            return [SimpleNamespace(outputs=[
                SimpleNamespace(index=index, text="2", token_ids=[9], finish_reason="stop")
                for index in reversed(range(param.n))]) for param in sampling_params]

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: Tokenizer())))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Engine, SamplingParams=SimpleNamespace))
    monkeypatch.setattr(module, "_manifest", lambda *_: {"run_fingerprint": "run-hash", "config": {}})
    module.run_benchmark(args)
    saved = [json.loads(line) for line in (args.output / "results.jsonl").read_text().splitlines()]
    assert len(saved) == 2
    assert [sample["sample_index"] for sample in saved[0]["samples"]] == [0, 1]
    assert calls == {"loaded": 1, "seeds": [20261021, 20261022], "shutdown": [{"timeout": 30}]}
    module.run_benchmark(args)
    assert calls["loaded"] == 1


def test_resume_rejects_duplicate_incomplete_or_changed_grades(tmp_path, fake_grader):
    args = _args(tmp_path)
    row = _row()
    group = _group(args, row)
    tokens = {row["question_id"]: [1, 2, 3]}
    with pytest.raises(ValueError, match="duplicate"):
        module._validate_groups([group, group], [row], tokens, args, "run-hash")
    group["samples"].pop()
    with pytest.raises(ValueError, match="Incomplete sample coverage"):
        module._validate_groups([group], [row], tokens, args, "run-hash")
    group = _group(args, row)
    group["samples"][1]["correct"] = False
    with pytest.raises(ValueError, match="Stored grade differs"):
        module._validate_groups([group], [row], tokens, args, "run-hash")


def test_recover_only_interrupted_final_json_write(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"question_id":"first"}\n{"quest')
    assert module._read_groups(path) == [{"question_id": "first"}]
    assert path.read_bytes() == b'{"question_id":"first"}\n'
    path.write_bytes(b'{"question_id":"first"}\ninvalid\n')
    with pytest.raises(ValueError, match="Corrupt completed result"):
        module._read_groups(path)
    path.write_bytes(b'invalid\n{"question_id":"second"}\n')
    with pytest.raises(ValueError, match="Corrupt completed result"):
        module._read_groups(path)


def test_manifest_mismatch_does_not_replace_outputs(tmp_path):
    manifest = {"run_fingerprint": "first", "config": {}}
    module._bind_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="manifest differs"):
        module._bind_manifest(tmp_path, {"run_fingerprint": "second", "config": {}})
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_bank_duplicates_and_invalid_sample_counts_fail(tmp_path):
    args = _args(tmp_path)
    row = _row()
    args.bank.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Duplicate question"):
        module._load_questions(args)
    args.bank.write_text(json.dumps(row) + "\n")
    args.samples_override = 0
    with pytest.raises(ValueError, match="Invalid sample count"):
        module._load_questions(args)


def test_manifest_fingerprint_binds_bank_variant_and_sampling(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.bank.write_text(json.dumps(_row()) + "\n")
    args.model.mkdir()
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "test-runtime")
    tokenizer = SimpleNamespace(chat_template="template")
    original = module._manifest(args, tokenizer)
    assert module._manifest(args, tokenizer) == original
    args.variant_fingerprint = "other-adapter"
    assert module._manifest(args, tokenizer)["run_fingerprint"] != original["run_fingerprint"]
    args.variant_fingerprint = "adapter-hash"
    args.max_new_tokens = 256
    assert module._manifest(args, tokenizer)["run_fingerprint"] != original["run_fingerprint"]
    args.max_new_tokens = 512
    args.bank.write_text(json.dumps(_row("changed-question")) + "\n")
    assert module._manifest(args, tokenizer)["run_fingerprint"] != original["run_fingerprint"]
