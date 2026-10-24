"""Export BOLT shards sharing one protocol across different execution routes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from qwenpt.train.utils.bolt_data import load_math_pool, validate_bolt_group
from qwenpt.train.utils.prepare_data import input_paths, sha256


def finalize_execution_family(
    data: list[str], shards: list[str], output: str, *, seed: int = 20261020,
    val_ratio: float = 0.1, reuse_existing: bool = False,
) -> dict:
    """Validate complete routed shards and publish grouped splits without editing raw records.

    Only ``num_shards`` may differ between generation configurations. Existing
    exports require matching source hashes and split parameters before reuse.
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    inputs = [{"path": str(path), "sha256": sha256(path)} for path in input_paths(data)]
    paths = sorted({Path(value).resolve() for value in shards})
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("shards must contain existing JSONL files")
    configurations, sources, protocol = {}, [], None
    module_dir = Path(__file__).parent
    current_code = {name: sha256(module_dir / name) for name in (
        "bolt_rollout.py", "bolt_data.py", "bolt_math.py",
    )}
    for path in paths:
        metadata_path = path.with_suffix(".manifest.json")
        metadata = json.loads(metadata_path.read_text())
        config = metadata["config"]
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        if metadata.get("run_fingerprint") != fingerprint:
            raise ValueError(f"Generation fingerprint does not match configuration: {path}")
        count, index = config.get("num_shards"), metadata.get("shard_id")
        if (type(count) is not int or count < 1 or type(index) is not int or not 0 <= index < count):
            raise ValueError(f"Invalid shard routing: {path}")
        candidate_protocol = {key: value for key, value in config.items() if key != "num_shards"}
        if protocol is None:
            protocol = candidate_protocol
        elif json.dumps(candidate_protocol, sort_keys=True) != json.dumps(protocol, sort_keys=True):
            raise ValueError(f"Generation protocols differ beyond num_shards: {path}")
        if config.get("inputs") != inputs or config.get("code_sha256") != current_code:
            raise ValueError(f"Generation input or code hashes differ from current files: {path}")
        if config.get("num_candidates") != 8 or config.get("limit") != -1:
            raise ValueError("Recovery requires unrestricted generation with eight candidates")
        model = Path(config["model"])
        model_hashes = config.get("model_metadata_sha256")
        if not model_hashes or any(sha256(model / name) != digest for name, digest in model_hashes.items()):
            raise ValueError(f"Generation model metadata hashes differ: {path}")
        template = model / "chat_template.jinja"
        if ("chat_template_sha256" in config and template.is_file()
                and sha256(template) != config["chat_template_sha256"]):
            raise ValueError(f"Generation chat template hash differs: {path}")
        configurations[fingerprint] = config
        sources.append({"path": str(path), "sha256": sha256(path),
                        "manifest_sha256": sha256(metadata_path), "run_fingerprint": fingerprint,
                        "shard_id": index, "num_shards": count})
    bindings = {
        "format_version": 2, "inputs": inputs, "shards": sources,
        "generation_configs": configurations,
        "finalizer_sha256": sha256(Path(__file__).resolve()),
        "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        "seed": seed, "val_ratio": val_ratio, "candidates_per_prompt": 8,
    }
    destination = Path(output).resolve()
    artifacts = [destination / name for name in ("train.parquet", "val.parquet", "split_manifest.json")]
    existing = [path for path in artifacts if path.exists()]
    if existing:
        if not reuse_existing or len(existing) != len(artifacts):
            raise FileExistsError(f"Refusing to overwrite existing or incomplete BOLT split: {destination}")
        manifest = json.loads(artifacts[2].read_text())
        if any(manifest.get(key) != value for key, value in bindings.items()):
            raise ValueError("Existing recovery export inputs or parameters differ")
        if manifest.get("output_sha256") != {path.name: sha256(path) for path in artifacts[:2]}:
            raise ValueError("Existing recovery export output hashes differ")
        return manifest
    pool = load_math_pool(data)
    expected = {row["prompt_id"]: row for row in pool}
    groups = {}
    for source in sources:
        assigned = {row["prompt_id"] for row in pool
                    if row["source_index"] % source["num_shards"] == source["shard_id"]}
        seen = set()
        with Path(source["path"]).open() as stream:
            for line_number, line in enumerate(stream, 1):
                group = json.loads(line)
                prompt_id = group.get("prompt_id")
                if prompt_id not in assigned:
                    raise ValueError(f"Unexpected prompt or wrong shard routing: {source['path']}:{line_number}")
                if prompt_id in groups:
                    raise ValueError(f"Duplicate prompt group: {prompt_id}")
                if group.get("run_fingerprint") != source["run_fingerprint"]:
                    raise ValueError(f"Group fingerprint differs from its generation manifest: {prompt_id}")
                groups[prompt_id] = validate_bolt_group(group, expected[prompt_id], 8)
                seen.add(prompt_id)
        if seen != assigned:
            raise ValueError(f"Incomplete assigned shard: {source['path']}")
    if set(groups) != set(expected):
        raise ValueError(f"Incomplete prompt coverage: expected {len(pool)}, found {len(groups)}")
    prompt_ids = [row["prompt_id"] for row in pool]
    random.Random(seed).shuffle(prompt_ids)
    validation_count = int(len(prompt_ids) * val_ratio)
    if not 0 < validation_count < len(prompt_ids):
        raise ValueError("Both training and validation must contain prompt groups")
    split_ids = {"train": prompt_ids[validation_count:], "validation": prompt_ids[:validation_count]}
    files = {"train": "train.parquet", "validation": "val.parquet"}
    manifest = {
        **bindings, "split_prompt_ids": split_ids, "output_files": files,
        "counts": {"total_prompts": len(pool), "train_prompts": len(pool) - validation_count,
                   "validation_prompts": validation_count, "total_rows": len(pool) * 8,
                   "train_rows": (len(pool) - validation_count) * 8, "validation_rows": validation_count * 8},
    }
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bolt-recovery-", dir=destination) as temporary:
        staging = Path(temporary)
        for split, ids in split_ids.items():
            table = pa.Table.from_pylist([row for prompt_id in ids for row in groups[prompt_id]])
            target = staging / files[split]
            pq.write_table(table, target)
            if not pq.read_table(target, use_threads=False).equals(table):
                raise ValueError(f"Recovery export read-back mismatch: {split}")
        manifest["output_sha256"] = {name: sha256(staging / name) for name in files.values()}
        (staging / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        for path in artifacts:
            os.replace(staging / path.name, path)
    return manifest


def main() -> None:
    """Export an execution family, then verify it with the actual training runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    manifest = finalize_execution_family(args.data, args.shards, args.output, reuse_existing=args.reuse_existing)
    config = next(iter(manifest["generation_configs"].values()))
    training_python = Path(__file__).resolve().parents[4] / ".venv/bin/python"
    subprocess.run([
        str(training_python), "-m", "qwenpt.train.utils.bolt_verify", "--output", args.output,
        "--tokenizer", config["model"], "--max-length", str(config["max_model_len"]),
    ], check=True)
    print(json.dumps({"output": args.output, **manifest["counts"]}), flush=True)


if __name__ == "__main__":
    main()
