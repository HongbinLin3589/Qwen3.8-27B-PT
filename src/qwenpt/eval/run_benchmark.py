"""Generate resumable GPQA and AIME benchmark groups with the verified vLLM runtime."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Any


BENCHMARKS = ("gpqa_diamond", "aime2024", "aime2025", "aime2026")
SCORE_FIELDS = ("predicted", "correct", "thinking_complete", "format_error")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_questions(args: argparse.Namespace) -> list[dict[str, Any]]:
    bank = Path(args.bank).expanduser().resolve()
    if bank.is_dir():
        bank /= "question_bank.jsonl"
    args.bank = bank
    selected = []
    seen = set()
    with bank.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["benchmark"], row["question_id"])
            if key in seen:
                raise ValueError(f"Duplicate question in bank: {key}")
            seen.add(key)
            if row["benchmark"] != args.benchmark:
                continue
            if not isinstance(row["question_id"], str) or not row["question_id"]:
                raise ValueError("Question IDs must be nonempty strings")
            if not isinstance(row["prompt"], str) or not row["prompt"].strip():
                raise ValueError(f"Missing prompt: {key}")
            if not isinstance(row["answer"], str) or not row["answer"]:
                raise ValueError(f"Missing answer: {key}")
            samples = args.samples_override if args.samples_override is not None else row["samples"]
            if type(samples) is not int or samples <= 0:
                raise ValueError(f"Invalid sample count: {key}")
            selected.append({**row, "samples": samples})
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        selected = selected[:args.limit]
    if not selected:
        raise ValueError(f"No questions selected for {args.benchmark}")
    return selected


def _manifest(args: argparse.Namespace, tokenizer: Any) -> dict[str, Any]:
    module_dir = Path(__file__).parent
    config = {
        "schema_version": 1,
        "bank": str(args.bank),
        "bank_sha256": _sha256(args.bank),
        "benchmark": args.benchmark,
        "variant": args.variant,
        "variant_fingerprint": args.variant_fingerprint,
        "model": str(args.model),
        "model_metadata_sha256": {
            name: _sha256(args.model / name)
            for name in ("config.json", "model.safetensors.index.json", "generation_config.json",
                         "tokenizer_config.json", "tokenizer.json", "merge_manifest.json")
            if (args.model / name).is_file()
        },
        "code_sha256": {
            **{name: _sha256(module_dir / name) for name in ("run_benchmark.py", "scoring.py")},
            "bolt_math.py": _sha256(module_dir.parent / "train" / "utils" / "bolt_math.py"),
        },
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "runtime": {
            name: importlib.metadata.version(name) for name in (
                "vllm", "torch", "transformers", "math-verify", "latex2sympy2-extended",
            )
        },
        "sampling": {
            "seed": args.seed,
            "seed_rule": "seed_plus_benchmark_question_index",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "samples_override": args.samples_override,
            "enable_thinking": True,
            "preserve_thinking": True,
            "reasoning_effort": "xhigh",
            "generation_config": "vllm",
        },
        "engine": {
            "tensor_parallel_size": 1,
            "dtype": "bfloat16",
            "language_model_only": True,
            "enable_prefix_caching": True,
            **{name: getattr(args, name) for name in (
                "max_model_len", "max_num_seqs", "max_num_batched_tokens",
                "gpu_memory_utilization", "enforce_eager", "batch_prompts",
            )},
        },
        "limit": args.limit,
    }
    encoded = json.dumps(config, sort_keys=True, allow_nan=False).encode()
    return {"run_fingerprint": hashlib.sha256(encoded).hexdigest(), "config": config}


def _bind_manifest(output: Path, manifest: dict[str, Any]) -> None:
    path = output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f"Existing run manifest differs: {path}")
        return
    if (output / "results.jsonl").exists():
        raise FileExistsError("Refusing results without an identifying manifest")
    temporary = output / "manifest.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _read_groups(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    groups = []
    with path.open("rb+") as stream:
        while line := stream.readline():
            start = stream.tell() - len(line)
            try:
                group = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Only an unterminated final write can be discarded during resume.
                if line.endswith(b"\n") or stream.read(1):
                    raise ValueError(f"Corrupt completed result in {path}") from None
                stream.truncate(start)
                stream.flush()
                os.fsync(stream.fileno())
                break
            groups.append(group)
            if not line.endswith(b"\n"):
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
    return groups


def _validate_groups(
    groups: list[dict[str, Any]], questions: list[dict[str, Any]],
    tokenized: dict[str, list[int]], args: argparse.Namespace, fingerprint: str,
) -> set[str]:
    from qwenpt.eval.scoring import score_completion

    expected = {row["question_id"]: row for row in questions}
    done: set[str] = set()
    for group in groups:
        question_id = group["question_id"]
        if question_id not in expected or question_id in done:
            raise ValueError(f"Unexpected or duplicate result: {question_id}")
        row = expected[question_id]
        for name, value in (
            ("benchmark", args.benchmark), ("variant", args.variant),
            ("run_fingerprint", fingerprint), ("answer", row["answer"]),
            ("prompt_tokens", len(tokenized[question_id])),
        ):
            if group.get(name) != value:
                raise ValueError(f"Result {question_id} has mismatched {name}")
        samples = group["samples"]
        if len(samples) != row["samples"]:
            raise ValueError(f"Incomplete sample coverage: {question_id}")
        indices = [sample["sample_index"] for sample in samples]
        if any(type(index) is not int for index in indices) or sorted(indices) != list(range(row["samples"])):
            raise ValueError(f"Invalid sample indices: {question_id}")
        for sample in samples:
            if not isinstance(sample["text"], str):
                raise ValueError(f"Invalid completion text: {question_id}")
            generated_tokens = sample["generated_tokens"]
            if type(generated_tokens) is not int or not 0 <= generated_tokens <= args.max_new_tokens:
                raise ValueError(f"Invalid generated token count: {question_id}")
            if sample["finish_reason"] not in ("stop", "length"):
                raise ValueError(f"Unfinished completion: {question_id}")
            score = score_completion(sample["text"], args.benchmark, row["answer"])
            if any(sample.get(name) != score[name] for name in SCORE_FIELDS):
                raise ValueError(f"Stored grade differs from grader: {question_id}")
            if any(type(sample.get(name)) is not bool for name in SCORE_FIELDS[1:]):
                raise ValueError(f"Invalid stored score flags: {question_id}")
        done.add(question_id)
    return done


def _run_locked(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from qwenpt.eval.scoring import score_completion

    questions = _load_questions(args)
    args.model = Path(args.model).expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenized = {}
    for row in questions:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], enable_thinking=True,
            reasoning_effort="xhigh", preserve_thinking=True,
            add_generation_prompt=True, tokenize=True, return_dict=False,
        )
        if len(ids) + args.max_new_tokens > args.max_model_len:
            raise ValueError(f"Question {row['question_id']} exceeds context budget")
        tokenized[row["question_id"]] = ids
    manifest = _manifest(args, tokenizer)
    _bind_manifest(args.output, manifest)
    path = args.output / "results.jsonl"
    groups = _read_groups(path)
    fingerprint = manifest["run_fingerprint"]
    done = _validate_groups(groups, questions, tokenized, args, fingerprint)
    remaining = [(index, row) for index, row in enumerate(questions) if row["question_id"] not in done]
    print(json.dumps({"variant": args.variant, "benchmark": args.benchmark,
                      "questions": len(questions), "completed_questions": len(done),
                      "remaining_questions": len(remaining), "run_fingerprint": fingerprint}), flush=True)
    if not remaining:
        return
    # Native kernel builders resolve helpers through this inference interpreter's PATH.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    if os.environ.get("CXX"):
        runtime = Path(os.environ["CXX"]).parent.parent / "lib64"
        if (runtime / "libstdc++.so.6").is_file():
            os.environ["LD_LIBRARY_PATH"] = str(runtime) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    from vllm import LLM, SamplingParams

    eos = json.loads((args.model / "generation_config.json").read_text())["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
    engine = LLM(
        model=str(args.model), tensor_parallel_size=1, dtype="bfloat16", language_model_only=True,
        max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=True,
        seed=args.seed, generation_config="vllm", enforce_eager=args.enforce_eager,
    )
    try:
        with path.open("a", encoding="utf-8") as stream:
            for first in range(0, len(remaining), args.batch_prompts):
                batch = remaining[first:first + args.batch_prompts]
                prompts = [{"prompt_token_ids": tokenized[row["question_id"]]} for _, row in batch]
                parameters = [SamplingParams(
                    n=row["samples"], temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    max_tokens=args.max_new_tokens, seed=args.seed + index, stop_token_ids=eos,
                ) for index, row in batch]
                outputs = engine.generate(prompts, sampling_params=parameters, use_tqdm=False)
                if len(outputs) != len(batch):
                    raise ValueError("Inference engine returned an incomplete question batch")
                for (_, row), result in zip(batch, outputs, strict=True):
                    samples = [{
                        "sample_index": completion.index, "text": completion.text,
                        "generated_tokens": len(completion.token_ids), "finish_reason": completion.finish_reason,
                        **score_completion(completion.text, args.benchmark, row["answer"]),
                    } for completion in sorted(result.outputs, key=lambda item: item.index)]
                    group = {
                        "benchmark": args.benchmark, "question_id": row["question_id"], "answer": row["answer"],
                        "variant": args.variant, "run_fingerprint": fingerprint,
                        "prompt_tokens": len(tokenized[row["question_id"]]), "samples": samples,
                    }
                    _validate_groups([group], [row], tokenized, args, fingerprint)
                    stream.write(json.dumps(group, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    done.add(row["question_id"])
                    print(json.dumps({"variant": args.variant, "benchmark": args.benchmark,
                                      "completed_questions": len(done), "questions": len(questions),
                                      "question_id": row["question_id"], "samples": len(samples),
                                      "correct": sum(sample["correct"] for sample in samples),
                                      "length_limited": sum(sample["finish_reason"] == "length" for sample in samples)}),
                          flush=True)
    finally:
        pending_error = sys.exception()
        try:
            # The non-daemon EngineCore otherwise can block a completed Slurm job's exit.
            engine.llm_engine.engine_core.shutdown(timeout=30)
        except Exception as shutdown_error:
            if pending_error is None:
                raise
            pending_error.add_note(f"Engine shutdown also failed: {shutdown_error}")
    if done != {row["question_id"] for row in questions}:
        raise ValueError("Benchmark ended without complete question coverage")


def run_benchmark(args: argparse.Namespace) -> None:
    """Lock one output directory and generate or resume complete question groups.

    Args:
        args: Parsed generation, model, benchmark, and output arguments.
    """
    args.output = Path(args.output).expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.batch_prompts is None:
        args.batch_prompts = 8 if args.benchmark == "gpqa_diamond" else 1
    if min(args.max_new_tokens, args.max_model_len, args.max_num_seqs,
           args.max_num_batched_tokens, args.batch_prompts) <= 0:
        raise ValueError("Generation and batch limits must be positive")
    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run_locked(args)


def main() -> None:
    """Parse one model and benchmark job, then run its resumable GPU generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant-fingerprint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=36864)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--batch-prompts", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples-override", type=int)
    parser.add_argument("--enforce-eager", action="store_true")
    run_benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
