"""Prepare train-only SFT pools and audit their actual token lengths."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
from importlib.metadata import version
from itertools import islice
import json
import multiprocessing
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from qwenpt.train.utils.prepare_data import sha256


LENGTH_LIMITS = (4096, 8192, 16384, 32768, 65536, 131072, 262144)
_TOKENIZER = None
_TEMPLATE_HASH = None
_EXTRA_FIELDS = (
    pa.field("tokenized_length", pa.int64()),
    pa.field("tokenized_supervised_tokens", pa.int64()),
    pa.field("sample_fingerprint", pa.string()),
    pa.field("tokenized_fingerprint", pa.string()),
    pa.field("benchmark_source_path", pa.string()),
    pa.field("benchmark_source_row", pa.int64()),
)


def _json_bytes(value: Any) -> bytes:
    def encode(item):
        if isinstance(item, bytes):
            return {"bytes_hex": item.hex()}
        raise TypeError(f"Unsupported sample fingerprint value: {type(item).__name__}")

    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=encode).encode()


def _bucket_name(bucket: int) -> str:
    if bucket == len(LENGTH_LIMITS):
        return "overflow"
    lower = 0 if bucket == 0 else LENGTH_LIMITS[bucket - 1]
    return f"{lower}_{LENGTH_LIMITS[bucket]}"


def _initialize_worker(tokenizer_path: str) -> None:
    global _TOKENIZER, _TEMPLATE_HASH
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    from transformers import AutoTokenizer

    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    _TEMPLATE_HASH = hashlib.sha256(_json_bytes(_TOKENIZER.chat_template)).hexdigest()


def _tokenize_row(task: tuple[dict, str, int]) -> dict:
    from qwenpt.train.verl.template import tokenize_sft_sample, tokenized_fingerprint

    row, source_path, source_row = task
    identity = {
        "source_path": source_path, "source_row": source_row,
        "id": row.get("id", row.get("sample_id")), "domain": row.get("domain"),
    }
    try:
        if row.get("split") != "train":
            raise ValueError("Only original train rows may enter the prepared pool")
        inputs = tokenize_sft_sample(row, _TOKENIZER)
        length = len(inputs["input_ids"])
        mask = inputs["loss_mask"]
        if length > LENGTH_LIMITS[-1]:
            raise ValueError(f"Actual sequence length {length} exceeds {LENGTH_LIMITS[-1]}")
        if mask.shape != inputs["input_ids"].shape or not mask.any() or mask[0]:
            raise ValueError("Invalid token supervision shape, empty supervision or supervised first token")
        fingerprint = tokenized_fingerprint(inputs["input_ids"], mask)
        return {
            "ok": True, "tokenized_length": length, "tokenized_supervised_tokens": int(mask.sum()),
            "sample_fingerprint": hashlib.sha256(_json_bytes(row)).hexdigest(),
            "tokenized_fingerprint": fingerprint, "template_sha256": _TEMPLATE_HASH,
        }
    except Exception as error:
        return {"ok": False, **identity, "stage": "tokenize", "error": f"{type(error).__name__}: {error}"[:2000]}


def _scan_inputs(paths: list[Path]) -> tuple[list[dict], list[dict], list[dict]]:
    candidates, inputs, failures = [], [], []
    for file_index, path in enumerate(paths):
        part = pq.ParquetFile(path)
        inputs.append({"path": str(path), "sha256": sha256(path), "rows": part.metadata.num_rows})
        required = {"messages", "total_tokens", "split"}
        if not required <= set(part.schema_arrow.names):
            raise ValueError(f"Missing required input columns in {path}: {sorted(required - set(part.schema_arrow.names))}")
        columns = [key for key in ("domain", "split", "total_tokens", "id", "sample_id") if key in part.schema_arrow.names]
        offset = 0
        for batch in part.iter_batches(batch_size=1024, columns=columns, use_threads=False):
            for local_index, row in enumerate(batch.to_pylist()):
                identity = {
                    "source_path": str(path), "source_row": offset + local_index,
                    "id": row.get("id", row.get("sample_id")), "domain": path.parent.parent.name,
                }
                length = row["total_tokens"]
                if (row["split"] != "train" or row.get("domain", identity["domain"]) != identity["domain"]
                        or isinstance(length, bool) or not isinstance(length, int) or length <= 0):
                    failures.append({**identity, "stage": "metadata", "error": "Invalid original split, domain or producer total_tokens"})
                    continue
                candidates.append({
                    "file_index": file_index, "row": offset + local_index,
                    "domain": identity["domain"], "bucket": bisect_left(LENGTH_LIMITS, length),
                })
            offset += len(batch)
    return candidates, inputs, failures


def _select_rows(candidates: list[dict], max_samples: int, seed: int, per_bucket_rows: int | None) -> list[dict]:
    rng = random.Random(seed)
    strata = defaultdict(list)
    for item in candidates:
        strata[(item["domain"], item["bucket"])].append(item)
    for rows in strata.values():
        rng.shuffle(rows)
    keys = sorted(strata)
    rng.shuffle(keys)
    selected, bucket_counts = [], Counter()
    limit = len(candidates) if max_samples == -1 else max_samples
    while keys and len(selected) < limit:
        remaining = []
        for key in keys:
            if len(selected) == limit:
                break
            if per_bucket_rows is not None and bucket_counts[key[1]] >= per_bucket_rows:
                continue
            selected.append(strata[key].pop())
            bucket_counts[key[1]] += 1
            if strata[key]:
                remaining.append(key)
        keys = remaining
    return selected


def _selected_tasks(paths: list[Path], selected: list[dict]):
    selected_by_file = defaultdict(set)
    for item in selected:
        selected_by_file[item["file_index"]].add(item["row"])
    for file_index, path in enumerate(paths):
        wanted = selected_by_file[file_index]
        if not wanted:
            continue
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, use_threads=False):
            relative = [item for item in range(len(batch)) if offset + item in wanted]
            if relative:
                for relative_index, row in zip(relative, batch.take(pa.array(relative)).to_pylist(), strict=True):
                    yield row, str(path), offset + relative_index
            offset += len(batch)


def prepare_sft_data(
    data_root: str, tokenizer: str, output: str, *, workers: int = 1,
    max_samples: int = -1, seed: int = 20261020, per_bucket_rows: int | None = None,
) -> dict:
    """Audit the complete original train split unless bounded sampling is explicit.

    Sampling balances occupied (domain, producer-length-bin) strata; the optional
    cap applies to each producer length bin across domains.
    Output buckets instead use actual rendered lengths, with inclusive upper bounds.
    Any metadata or selected-row validation failure publishes a failed manifest
    and failures.jsonl, without publishing a partial pool or bucket Parquet files.
    The local tokenizer snapshot is loaded once per worker and never downloaded.
    """
    if workers <= 0 or (max_samples != -1 and max_samples <= 0):
        raise ValueError("workers must be positive and max_samples must be positive or -1")
    if per_bucket_rows is not None and (per_bucket_rows <= 0 or max_samples == -1):
        raise ValueError("per_bucket_rows must be positive and cannot cap a full-data (-1) preparation")
    root, tokenizer_path, destination = Path(data_root).resolve(), Path(tokenizer).resolve(), Path(output).resolve()
    if not tokenizer_path.is_dir():
        raise ValueError("tokenizer must identify a local snapshot directory")
    paths = sorted(root.glob("*/train/*.parquet"))
    if not paths:
        raise ValueError(f"No domain/train/*.parquet files found under {root}")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite SFT preparation artifacts in {destination}")
    candidates, inputs, failures = _scan_inputs(paths)
    selected = _select_rows(candidates, max_samples, seed, per_bucket_rows)
    if not selected and not failures:
        raise ValueError("No train rows were selected")
    schema = pa.unify_schemas([pq.ParquetFile(path).schema_arrow for path in paths], promote_options="permissive")
    if any(field.name in schema.names for field in _EXTRA_FIELDS):
        raise ValueError("SFT inputs already contain generated fingerprint or token-length columns")
    schema = pa.schema(list(schema) + list(_EXTRA_FIELDS), metadata=schema.metadata)
    tokenizer_files = sorted(path for path in tokenizer_path.rglob("*") if path.is_file() and (
        path.name in {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt"}
        or path.suffix in {".jinja", ".model"}
    ))
    source_dir = Path(__file__).resolve().parent.parent
    # Retain the provenance label consumed by existing audit manifests after the module rename.
    fingerprint = {
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_files": {str(path.relative_to(tokenizer_path)): sha256(path) for path in tokenizer_files},
        "source_sha256": {"template.py": sha256(source_dir / "verl/template.py"), "prepare_sft_benchmark_data.py": sha256(Path(__file__))},
        "versions": {name: version(name) for name in ("transformers", "torch", "pyarrow")},
        "template_options": {"enable_thinking_default": True, "preserve_thinking": True, "loss_scope": "all_assistant", "reasoning_effort": None},
    }
    statistics = {name: {"rows": 0, "tokens": 0, "supervised_tokens": 0, "domains": {}}
                  for name in map(_bucket_name, range(len(LENGTH_LIMITS)))}
    template_hashes = set()
    accepted = 0
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=destination) as staging_name:
        staging = Path(staging_name)
        with ExitStack() as stack:
            pool_writer = stack.enter_context(pq.ParquetWriter(staging / "pool.parquet", schema))
            bucket_writers = {name: stack.enter_context(pq.ParquetWriter(staging / f"bucket_{name}.parquet", schema)) for name in statistics}
            if workers == 1:
                _initialize_worker(str(tokenizer_path))
                executor = None
            else:
                executor = stack.enter_context(ProcessPoolExecutor(
                    max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                    initializer=_initialize_worker, initargs=(str(tokenizer_path),),
                ))
            tasks = iter(_selected_tasks(paths, selected))
            while batch := list(islice(tasks, max(1, workers * 2))):
                results = map(_tokenize_row, batch) if executor is None else executor.map(_tokenize_row, batch, chunksize=1)
                rows, bins = [], defaultdict(list)
                for (row, source_path, source_row), result in zip(batch, results, strict=True):
                    if not result["ok"]:
                        failures.append(result)
                        continue
                    template_hashes.add(result.pop("template_sha256"))
                    result.pop("ok")
                    enriched = row | result | {"benchmark_source_path": source_path, "benchmark_source_row": source_row}
                    rows.append(enriched)
                    name = _bucket_name(bisect_left(LENGTH_LIMITS, result["tokenized_length"]))
                    bins[name].append(enriched)
                    stat = statistics[name]
                    stat["rows"] += 1
                    stat["tokens"] += result["tokenized_length"]
                    stat["supervised_tokens"] += result["tokenized_supervised_tokens"]
                    domain = row.get("domain", Path(source_path).parent.parent.name)
                    stat["domains"][domain] = stat["domains"].get(domain, 0) + 1
                    accepted += 1
                if rows:
                    pool_writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                    for name, values in bins.items():
                        bucket_writers[name].write_table(pa.Table.from_pylist(values, schema=schema))
        if len(template_hashes) > 1:
            failures.append({"stage": "template", "error": "Workers did not produce one consistent resolved template fingerprint"})
        manifest = {
            "format_version": 1, "status": "failed" if failures else "passed", "data_root": str(root),
            "inputs": inputs, "seed": seed, "max_samples": max_samples, "per_bucket_rows": per_bucket_rows,
            "workers": workers, "sampling": "round-robin domain/producer-length strata; seeded shuffle within strata",
            "length_bin_upper_bounds_inclusive": LENGTH_LIMITS, "fingerprint": fingerprint,
            "resolved_template_sha256": next(iter(template_hashes)) if len(template_hashes) == 1 else None,
            "counts": {"eligible_train_rows": len(candidates), "selected_rows": len(selected), "validated_rows": accepted, "failures": len(failures)},
            "selected_strata": {f"{domain}/{_bucket_name(bucket)}": count for (domain, bucket), count in sorted(Counter((row["domain"], row["bucket"]) for row in selected).items())},
            "actual_length_bins": statistics,
            "outputs": {},
        }
        failure_path = destination / "failures.jsonl"
        failure_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures))
        manifest["outputs"]["failures.jsonl"] = {"sha256": sha256(failure_path), "rows": len(failures)}
        if not failures:
            for path in sorted(staging.glob("*.parquet")):
                target = destination / path.name
                path.replace(target)
                manifest["outputs"][path.name] = {"sha256": sha256(target), "rows": pq.ParquetFile(target).metadata.num_rows}
        (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    """Parse full train auditing and optional sampling; report failures via exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="Selected train rows; -1 audits the complete original train split (default)")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--per-bucket-rows", type=int,
                        help="Optional producer-bin cap; requires an explicit bounded --max-samples")
    args = parser.parse_args()
    manifest = prepare_sft_data(**vars(args))
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"], "output": args.output}))
    if manifest["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
