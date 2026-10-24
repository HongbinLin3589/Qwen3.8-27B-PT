"""Let the final successful rollout shard export splits within its allocation."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from qwenpt.train.utils.bolt_data import finalize_bolt_data
from qwenpt.train.utils.bolt_rollout import read_completed_groups
from qwenpt.train.utils.bolt_verify import verify_bolt_splits
from qwenpt.train.utils.prepare_data import sha256


def complete_shard(args: argparse.Namespace) -> None:
    """Write a verified completion receipt and merge only after all shards finish."""
    directory = Path(args.output).resolve()
    shard = directory / f"shard_{args.shard_id:05d}.jsonl"
    metadata = json.loads(shard.with_suffix(".manifest.json").read_text())
    receipt = shard.with_suffix(".done.json")
    prompts = len(read_completed_groups(shard))
    value = {"run_fingerprint": metadata["run_fingerprint"], "sha256": sha256(shard), "prompts": prompts}
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, receipt)
    files = [directory / f"shard_{index:05d}.jsonl" for index in range(args.num_shards)]
    if not all(path.with_suffix(".done.json").exists() for path in files):
        return
    output = Path(args.finalize_output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "finalize.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        for path in files:
            completed = json.loads(path.with_suffix(".done.json").read_text())
            if completed["run_fingerprint"] != value["run_fingerprint"] or completed["sha256"] != sha256(path):
                raise ValueError(f"Rollout completion receipt does not match: {path}")
        manifest = finalize_bolt_data(
            args.data, [str(path) for path in files], str(output),
            candidates_per_prompt=args.num_candidates, seed=args.seed, val_ratio=0.1, reuse_existing=True,
        )
        print(json.dumps({"exported": str(output), **manifest["counts"]}), flush=True)
        training_python = Path(__file__).resolve().parents[4] / ".venv/bin/python"
        if Path(sys.executable) == training_python:
            audit = verify_bolt_splits(str(output), args.model, max_length=args.max_model_len)
        else:
            # Validate with the actual trainer's tokenizer/runtime, not vLLM's newer Transformers.
            subprocess.run([
                str(training_python), "-m", "qwenpt.train.utils.bolt_verify", "--output", str(output),
                "--tokenizer", args.model, "--max-length", str(args.max_model_len),
            ], check=True)
            audit = json.loads((output / "native_input_check.json").read_text())
        print(json.dumps({"native_input_check": audit}, ensure_ascii=False), flush=True)
