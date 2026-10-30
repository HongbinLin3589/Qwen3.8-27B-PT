"""Validate offline math rollouts and export complete BOLT prompt groups."""

import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import random
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from qwenpt.train.common.files import input_paths, sha256

MATH_PROMPT_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def load_math_pool(data: list[str]) -> list[dict]:
    """Add APEX's answer-format instruction and assign stable prompt IDs and indices.

    Each row must contain a nonempty user/system conversation and a nonempty
    string ``ground_truth``; duplicate conversations are rejected.
    """
    rows = []
    prompt_ids = set()
    for path in input_paths(data):
        for row in pq.read_table(path, use_threads=False).to_pylist():
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"Math pool row {len(rows)}: messages must be a nonempty list")
            for message in messages:
                if (not isinstance(message, dict) or message.get("role") not in {"system", "user"}
                        or not isinstance(message.get("content"), str) or not message["content"].strip()):
                    raise ValueError(f"Math pool row {len(rows)}: expected nonempty user/system text messages")
            if messages[-1]["role"] != "user":
                raise ValueError(f"Math pool row {len(rows)}: prompt must end with a user message")
            messages = [dict(message) for message in messages]
            # APEX's strict verifier needs an explicit final-answer anchor.
            if not messages[-1]["content"].endswith(MATH_PROMPT_SUFFIX):
                messages[-1]["content"] += MATH_PROMPT_SUFFIX
            ground_truth = row.get("ground_truth")
            if not isinstance(ground_truth, str) or not ground_truth.strip():
                raise ValueError(f"Math pool row {len(rows)}: ground_truth must be a nonempty string")
            canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            prompt_id = "math-" + hashlib.sha256(canonical.encode()).hexdigest()
            if prompt_id in prompt_ids:
                raise ValueError(f"Duplicate prompt in math pool: {prompt_id}")
            prompt_ids.add(prompt_id)
            rows.append({
                "prompt_id": prompt_id, "source_index": len(rows),
                "messages": messages, "ground_truth": ground_truth,
            })
    if not rows:
        raise ValueError("Math pool contains no rows")
    return rows


def validate_bolt_group(group: dict, expected: dict, candidates_per_prompt: int) -> list[dict]:
    """Validate one complete generated group and return its ordered BOLT rows."""
    prompt_id = expected["prompt_id"]
    for key in ("prompt_id", "source_index", "messages", "ground_truth"):
        if group.get(key) != expected[key]:
            raise ValueError(f"Group {prompt_id}: {key} does not match the math pool")
    if isinstance(group.get("source_index"), bool) or not isinstance(group.get("source_index"), int):
        raise ValueError(f"Group {prompt_id}: source_index must be an integer")
    if not isinstance(group.get("enable_thinking"), bool):
        raise ValueError(f"Group {prompt_id}: enable_thinking must be a boolean")
    candidates = group.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != candidates_per_prompt:
        raise ValueError(f"Group {prompt_id}: expected {candidates_per_prompt} candidates")
    ordered = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError(f"Group {prompt_id}: each candidate must be an object")
        index = candidate.get("candidate_index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < candidates_per_prompt:
            raise ValueError(f"Group {prompt_id}: invalid candidate_index {index!r}")
        if index in ordered:
            raise ValueError(f"Group {prompt_id}: duplicate candidate_index {index}")
        reward = candidate.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, Real) or not math.isfinite(reward):
            raise ValueError(f"Group {prompt_id} candidate {index}: reward must be a finite real number")
        content = candidate.get("content")
        reasoning = candidate.get("reasoning_content")
        if not isinstance(content, str) or not isinstance(reasoning, str) or not (content.strip() or reasoning.strip()):
            raise ValueError(f"Group {prompt_id} candidate {index}: empty or invalid assistant supervision")
        finish_reason = candidate.get("finish_reason")
        tokens = candidate.get("generated_tokens")
        if not isinstance(finish_reason, str) or not finish_reason:
            raise ValueError(f"Group {prompt_id} candidate {index}: finish_reason must be a nonempty string")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            raise ValueError(f"Group {prompt_id} candidate {index}: generated_tokens must be a positive integer")
        ordered[index] = {
            "messages": expected["messages"] + [{
                "role": "assistant", "content": content, "reasoning_content": reasoning,
            }],
            "prompt_id": prompt_id, "reward": float(reward),
            "sample_id": f"{prompt_id}-c{index}", "ground_truth": expected["ground_truth"],
            "enable_thinking": group["enable_thinking"],
            "finish_reason": finish_reason, "generated_tokens": tokens,
        }
    return [ordered[index] for index in range(candidates_per_prompt)]


def finalize_bolt_data(
    data: list[str], shards: list[str], output: str, *, candidates_per_prompt: int = 8,
    seed: int = 20261020, val_ratio: float = 0.1, reuse_existing: bool = False,
) -> dict:
    """Validate complete JSONL groups, split by prompt ID, and publish BOLT Parquet.

    ``data`` is the original math pool, and each line in ``shards`` contains one
    whole prompt group. Existing outputs are reused only when their input,
    parameter, and output hashes match and ``reuse_existing`` is enabled.
    """
    if isinstance(candidates_per_prompt, bool) or not isinstance(candidates_per_prompt, int) or candidates_per_prompt < 1:
        raise ValueError("candidates_per_prompt must be a positive integer")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    pool_paths = input_paths(data)
    shard_paths = sorted({Path(value).resolve() for value in shards})
    if not shard_paths or any(not path.is_file() for path in shard_paths):
        raise ValueError("shards must contain existing JSONL files")
    parameters = {
        "format_version": 1,
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in pool_paths],
        "shards": [{"path": str(path), "sha256": sha256(path)} for path in shard_paths],
        "candidates_per_prompt": candidates_per_prompt, "seed": seed, "val_ratio": val_ratio,
    }
    output_dir = Path(output).resolve()
    artifacts = [output_dir / name for name in ("train.parquet", "val.parquet", "split_manifest.json")]
    existing = [path for path in artifacts if path.exists()]
    if existing:
        if not reuse_existing or len(existing) != len(artifacts):
            raise FileExistsError(f"Refusing to overwrite existing or incomplete BOLT split: {output_dir}")
        manifest = json.loads(artifacts[2].read_text())
        if any(manifest.get(key) != value for key, value in parameters.items()):
            raise ValueError(f"Existing BOLT split inputs or parameters differ in {output_dir}")
        if manifest.get("output_sha256") != {path.name: sha256(path) for path in artifacts[:2]}:
            raise ValueError(f"Existing BOLT split output hashes differ in {output_dir}")
        return manifest

    pool = load_math_pool(data)
    expected = {row["prompt_id"]: row for row in pool}
    groups = {}
    run_fingerprint = None
    for path in shard_paths:
        with path.open() as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    group = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from error
                if not isinstance(group, dict) or group.get("prompt_id") not in expected:
                    raise ValueError(f"Unknown prompt group in {path}:{line_number}")
                prompt_id = group["prompt_id"]
                if prompt_id in groups:
                    raise ValueError(f"Duplicate prompt group: {prompt_id}")
                fingerprint = group.get("run_fingerprint")
                if not isinstance(fingerprint, str) or not fingerprint:
                    raise ValueError(f"Group {prompt_id}: run_fingerprint must be a nonempty string")
                if run_fingerprint is not None and fingerprint != run_fingerprint:
                    raise ValueError(f"Group {prompt_id}: run_fingerprint differs across shards")
                run_fingerprint = fingerprint
                groups[prompt_id] = validate_bolt_group(group, expected[prompt_id], candidates_per_prompt)
    if len(groups) != len(pool):
        raise ValueError(f"Missing prompt groups: expected {len(pool)}, found {len(groups)}")

    prompt_ids = [row["prompt_id"] for row in pool]
    random.Random(seed).shuffle(prompt_ids)
    val_count = int(len(prompt_ids) * val_ratio)
    if not 0 < val_count < len(prompt_ids):
        raise ValueError("The requested split must contain both training and validation prompt groups")
    split_ids = {"train": prompt_ids[val_count:], "validation": prompt_ids[:val_count]}
    output_files = {"train": "train.parquet", "validation": "val.parquet"}
    manifest = {
        **parameters,
        "run_fingerprint": run_fingerprint,
        "counts": {
            "total_prompts": len(pool), "train_prompts": len(pool) - val_count,
            "validation_prompts": val_count, "total_rows": len(pool) * candidates_per_prompt,
            "train_rows": (len(pool) - val_count) * candidates_per_prompt,
            "validation_rows": val_count * candidates_per_prompt,
        },
        "split_prompt_ids": split_ids, "output_files": output_files,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    # Write and read back both parts before publishing the manifest as the commit marker.
    with tempfile.TemporaryDirectory(prefix=".bolt-split-", dir=output_dir) as temporary:
        staging = Path(temporary)
        for split, ids in split_ids.items():
            table = pa.Table.from_pylist([row for prompt_id in ids for row in groups[prompt_id]])
            destination = staging / output_files[split]
            pq.write_table(table, destination)
            if not pq.read_table(destination, use_threads=False).equals(table):
                raise ValueError(f"BOLT split read-back mismatch: {split}")
        manifest["output_sha256"] = {
            filename: sha256(staging / filename) for filename in output_files.values()
        }
        (staging / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        for destination in artifacts:
            os.replace(staging / destination.name, destination)
    return manifest
