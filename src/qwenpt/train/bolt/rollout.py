"""Generate resumable fixed BOLT candidates with vLLM, then export grouped splits."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Any

from qwenpt.train.bolt.data import MATH_PROMPT_SUFFIX, finalize_bolt_data, load_math_pool, validate_bolt_group
from qwenpt.train.bolt.math import parse_reference, score_math, split_completion
from qwenpt.train.common.files import input_paths, sha256


def read_completed_groups(path: Path) -> list[dict]:
    """Read durable whole-group records, recovering only an interrupted final write."""
    if not path.exists():
        return []
    groups = []
    with path.open("rb+") as stream:
        while line := stream.readline():
            boundary = stream.tell() - len(line)
            try:
                group = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if stream.read(1):
                    raise ValueError(f"Corrupt group before end of {path}") from None
                stream.truncate(boundary)
                break
            groups.append(group)
            if not line.endswith(b"\n"):
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
    return groups


def generation_config(args: argparse.Namespace, tokenizer: Any) -> dict:
    """Bind resumable shards to their source, model, template and sampling protocol."""
    model = Path(args.model).resolve()
    module_dir = Path(__file__).parent
    return {
        "format_version": 1,
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in input_paths(args.data)],
        "model": str(model),
        "model_metadata_sha256": {
            name: sha256(model / name) for name in (
                "config.json", "model.safetensors.index.json", "generation_config.json",
                "tokenizer_config.json", "tokenizer.json",
            )
            if (model / name).is_file()
        },
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        # Keep manifest labels stable while hashing the relocated implementations.
        "code_sha256": {name: sha256(module_dir / name.removeprefix("bolt_")) for name in (
            "bolt_rollout.py", "bolt_data.py", "bolt_math.py",
        )},
        "runtime": {name: importlib.metadata.version(name) for name in (
            "vllm", "torch", "transformers", "math-verify", "latex2sympy2-extended",
        )},
        "num_shards": args.num_shards,
        "limit": args.limit,
        "num_candidates": args.num_candidates,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "math_prompt_suffix": MATH_PROMPT_SUFFIX,
        "engine": {name: getattr(args, name) for name in (
            "max_num_seqs", "max_num_batched_tokens", "batch_prompts",
            "gpu_memory_utilization", "enforce_eager",
        )},
        "enable_thinking": True,
        "reasoning_effort": "template_default_xhigh",
        "grader": "apex_math_accuracy_final_content_v1",
        "reward_values": {"correct": 1.0, "incorrect": 0.0},
    }


def generate_shard(args: argparse.Namespace) -> None:
    """Lock one output shard so duplicate submissions cannot mix append streams."""
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"shard_{args.shard_id:05d}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        generate_locked_shard(args)


def generate_locked_shard(args: argparse.Namespace) -> None:
    """Generate and grade a locked shard's complete groups, saving each durably."""
    # Native kernel builders invoke tools such as ninja from this interpreter's environment.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    if os.environ.get("CXX"):
        runtime = Path(os.environ["CXX"]).parent.parent / "lib64"
        if (runtime / "libstdc++.so.6").is_file():
            # Fresh engine subprocesses must load the runtime matching their JIT compiler.
            os.environ["LD_LIBRARY_PATH"] = str(runtime) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard_id must be within [0, num_shards)")
    if args.num_candidates <= 0 or args.batch_prompts <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Candidate, batch and generation limits must be positive")
    pool = load_math_pool(args.data)
    if args.limit > 0:
        pool = pool[:args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = generation_config(args, tokenizer)
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    shard = output / f"shard_{args.shard_id:05d}.jsonl"
    metadata = shard.with_suffix(".manifest.json")
    expected_manifest = {"run_fingerprint": fingerprint, "shard_id": args.shard_id, "config": config}
    if metadata.exists():
        if json.loads(metadata.read_text()) != expected_manifest:
            raise ValueError(f"Existing shard configuration differs: {metadata}")
    else:
        if shard.exists():
            raise FileExistsError(f"Unidentified existing shard: {shard}")
        with metadata.open("x") as stream:
            json.dump(expected_manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    selected = pool[args.shard_id::args.num_shards]
    expected = {row["prompt_id"]: row for row in selected}
    completed = read_completed_groups(shard)
    done = set()
    for group in completed:
        prompt_id = group["prompt_id"]
        if prompt_id not in expected or prompt_id in done or group.get("run_fingerprint") != fingerprint:
            raise ValueError(f"Unexpected or repeated prompt in {shard}: {prompt_id}")
        if len(group["candidates"]) != args.num_candidates:
            raise ValueError(f"Incomplete candidates in {shard}: {prompt_id}")
        validate_bolt_group(group, expected[prompt_id], args.num_candidates)
        done.add(prompt_id)
    remaining = [row for row in selected if row["prompt_id"] not in done]
    print(json.dumps({"shard": args.shard_id, "assigned_prompts": len(selected),
                      "completed_prompts": len(done), "remaining_prompts": len(remaining),
                      "run_fingerprint": fingerprint}), flush=True)
    if not remaining:
        return
    tokenized = {}
    for row in remaining:
        parse_reference(row["ground_truth"])
        ids = tokenizer.apply_chat_template(
            row["messages"], enable_thinking=True, preserve_thinking=True,
            add_generation_prompt=True, tokenize=True, return_dict=False,
        )
        if len(ids) + args.max_new_tokens > args.max_model_len:
            raise ValueError(f"Prompt {row['prompt_id']} exceeds the requested context budget")
        tokenized[row["prompt_id"]] = ids
    engine = LLM(
        model=args.model, tensor_parallel_size=1, dtype="bfloat16", language_model_only=True,
        max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=True,
        seed=args.seed, generation_config="vllm", enforce_eager=args.enforce_eager,
    )
    try:
        eos = json.loads((Path(args.model) / "generation_config.json").read_text())["eos_token_id"]
        eos = eos if isinstance(eos, list) else [eos]
        with shard.open("a") as stream:
            for first in range(0, len(remaining), args.batch_prompts):
                batch = remaining[first:first + args.batch_prompts]
                prompts = [{"prompt_token_ids": tokenized[row["prompt_id"]]} for row in batch]
                parameters = [SamplingParams(
                    n=args.num_candidates, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    max_tokens=args.max_new_tokens, seed=args.seed + row["source_index"], stop_token_ids=eos,
                ) for row in batch]
                outputs = engine.generate(prompts, sampling_params=parameters, use_tqdm=False)
                if len(outputs) != len(batch):
                    raise ValueError("The inference engine returned an incomplete prompt batch")
                for row, result in zip(batch, outputs, strict=True):
                    if len(result.outputs) != args.num_candidates:
                        raise ValueError(f"Expected {args.num_candidates} candidates for {row['prompt_id']}")
                    candidates = []
                    for completion in sorted(result.outputs, key=lambda item: item.index):
                        assistant = split_completion(completion.text, enable_thinking=True)
                        candidates.append({
                            "candidate_index": completion.index, **assistant, "text": completion.text,
                            "reward": score_math(assistant["content"], row["ground_truth"]),
                            "finish_reason": completion.finish_reason,
                            "generated_tokens": len(completion.token_ids),
                        })
                    group = {**row, "run_fingerprint": fingerprint, "enable_thinking": True,
                             "prompt_tokens": len(tokenized[row["prompt_id"]]), "candidates": candidates}
                    validate_bolt_group(group, row, args.num_candidates)
                    stream.write(json.dumps(group, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    done.add(row["prompt_id"])
                    print(json.dumps({"shard": args.shard_id, "completed_prompts": len(done),
                                      "assigned_prompts": len(selected), "source_index": row["source_index"],
                                      "correct": sum(item["reward"] for item in candidates),
                                      "length_limited": sum(item["finish_reason"] == "length" for item in candidates)}),
                          flush=True)
    finally:
        pending_error = sys.exception()
        try:
            # vLLM's non-daemon engine can block interpreter exit unless explicitly closed.
            engine.llm_engine.engine_core.shutdown(timeout=30)
        except Exception as shutdown_error:
            if pending_error is None:
                raise
            pending_error.add_note(f"Engine shutdown also failed: {shutdown_error}")


def main() -> None:
    """Run one resumable generation shard or verify and publish all BOLT splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--data", nargs="+", required=True)
    generate.add_argument("--model", required=True)
    generate.add_argument("--output", required=True, help="Directory of independent JSONL generation shards")
    generate.add_argument("--shard-id", type=int, default=0)
    generate.add_argument("--num-shards", type=int, default=8)
    generate.add_argument("--num-candidates", type=int, default=8)
    generate.add_argument("--max-new-tokens", type=int, default=8192)
    generate.add_argument("--max-model-len", type=int, default=12288)
    generate.add_argument("--max-num-seqs", type=int, default=16)
    generate.add_argument("--max-num-batched-tokens", type=int, default=8192)
    generate.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    generate.add_argument("--batch-prompts", type=int, default=4)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--top-k", type=int, default=20)
    generate.add_argument("--seed", type=int, default=20261020)
    generate.add_argument("--limit", type=int, default=-1, help="Optional prompt limit for smoke tests")
    generate.add_argument("--enforce-eager", action="store_true")
    generate.add_argument("--finalize-output", help="Publish splits when every shard has a completion receipt")
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--data", nargs="+", required=True)
    finalize.add_argument("--shards", nargs="+", required=True)
    finalize.add_argument("--output", required=True)
    finalize.add_argument("--num-candidates", type=int, default=8)
    finalize.add_argument("--seed", type=int, default=20261020)
    finalize.add_argument("--val-ratio", type=float, default=0.1)
    finalize.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    if args.command == "generate":
        generate_shard(args)
        if args.finalize_output and args.limit <= 0:
            from qwenpt.train.bolt.complete import complete_shard

            complete_shard(args)
    else:
        manifest = finalize_bolt_data(
            args.data, args.shards, args.output, candidates_per_prompt=args.num_candidates,
            seed=args.seed, val_ratio=args.val_ratio, reuse_existing=args.reuse_existing,
        )
        print(json.dumps(manifest["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
