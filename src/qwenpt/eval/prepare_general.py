"""Prepare the Feishu core's instruction, science, code and math datasets."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess

from .suite import _name, _no_secrets, _positive, bind, digest, file_digest, write_json


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(rows, list):
            raise ValueError("Dataset JSON must be an array")
        return rows
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def normalize_gpqa(rows: list[dict], seed: int = 0) -> list[dict]:
    result = []
    for row in rows:
        question_id = str(row["Record ID"])
        choices = [row["Correct Answer"], *(row[f"Incorrect Answer {n}"] for n in range(1, 4))]
        order = list(range(4))
        random.Random(f"{seed}:{question_id}").shuffle(order)
        options = dict(zip("ABCD", [choices[n] for n in order]))
        prompt = row["Question"] + "\n\n" + "\n".join(f"{key}) {value}" for key, value in options.items())
        prompt += "\nEnd with a separate line: Answer: LETTER"
        result.append({"benchmark": "gpqa_diamond", "question_id": question_id, "prompt": prompt,
                       "answer": "ABCD"[order.index(0)], "options": options,
                       "metadata": {"stratum": row.get("Subdomain", row.get("High-level domain", "unknown")),
                                    "choice_permutation": order, "prompt_protocol": "qwenpt-gpqa-v1"}})
    return result


def normalize_matharena(rows: list[dict], benchmark: str) -> list[dict]:
    result = []
    for row in rows:
        answer = str(row["answer"])
        if not answer.strip():
            raise ValueError("MathArena row is missing its final answer")
        categories = row.get("problem_type", [])
        result.append({"benchmark": benchmark, "question_id": str(row["problem_idx"]),
                       "prompt": row["problem"] + "\nPut the final answer in \\boxed{}.", "answer": answer,
                       "metadata": {"stratum": "|".join(categories) if categories else row.get("source", "unknown"),
                                    "prompt_protocol": "qwenpt-matharena-v1"}})
    return result


def _checkout(path: Path, url: str, revision: str, download: bool):
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Official grader requires a full immutable Git revision")
    if not path.exists():
        if not download:
            raise ValueError(f"Missing official checkout: {path}; use prepare --download or clone the pinned source")
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none", url, str(path)], check=True, timeout=240)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", revision], check=True, timeout=180)


def _source_files(source: dict, base: Path, download: bool) -> tuple[list[dict], list[dict]]:
    revision = source["revision"]
    if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Dataset revision must be a full Hugging Face commit")
    files = source.get("files")
    if (not isinstance(files, list) or not files or any(not isinstance(name, str) or not name.strip() for name in files)
            or len(files) != len(set(files))):
        raise ValueError("Source files must be a nonempty list of unique file names")
    local = source.get("local_files", [])
    if not isinstance(local, list) or any(not isinstance(name, str) or not name.strip() for name in local):
        raise ValueError("Source local_files must be a list of file paths")
    paths = [(base / name).resolve() for name in local]
    if download and not paths:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ValueError("Install huggingface_hub for prepare --download; alternatively set local_files") from None
        paths = [Path(hf_hub_download(repo_id=source["repository"], repo_type="dataset", revision=revision,
                                      filename=name)) for name in source["files"]]
    if not paths:
        raise ValueError("No local_files configured; use prepare --download to fetch the pinned dataset")
    if len(paths) != len(source["files"]):
        raise ValueError("local_files must cover every file in the pinned release")
    rows, provenance = [], []
    for name, path in zip(source["files"], paths):
        expected = source.get("file_hashes", {}).get(name)
        if not isinstance(expected, dict) or not expected or set(expected) - {"sha256", "git_blob"}:
            raise ValueError("Each source file requires its upstream sha256 or Git blob hash")
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{%d}" % (64 if key == "sha256" else 40), value)
               for key, value in expected.items()):
            raise ValueError("Source file hash must be a complete lowercase hexadecimal digest")
        raw = path.read_bytes()
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        actual_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if expected.get("sha256", actual_sha256) != actual_sha256 or expected.get("git_blob", actual_blob) != actual_blob:
            raise ValueError(f"Source file does not match the frozen upstream hash: {name}")
        rows.extend(read_rows(path))
        provenance.append({"file": name, "sha256": actual_sha256, "expected_upstream": expected})
    return rows, provenance


def prepare_general(config_path: str, output: str, *, download: bool = False,
                    tasks: list[str] | None = None, model_revision: str = "FILL_CHECKPOINT_REVISION") -> dict:
    import yaml
    config_path = Path(config_path).resolve()
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        raise ValueError("Invalid source YAML syntax (content omitted)") from None
    if not isinstance(config, dict):
        raise ValueError("Source config must be an object")
    _no_secrets(config)
    if config.get("version") != 1 or not isinstance(config.get("sources"), dict):
        raise ValueError("Expected version: 1 and a sources mapping")
    if type(config.get("seed", 20261017)) is not int:
        raise ValueError("Source subset seed must be an integer")
    selected = tasks or list(config["sources"])
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(config["sources"]):
        raise ValueError("Unknown or duplicate preparation task")
    base, output = config_path.parent, Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"complete": True, "tasks": {}}
    for task in selected:
        try:
            _name(task, "source task")
            source = config["sources"][task]
            if not isinstance(source, dict) or source.get("kind") not in {"ifbench", "gpqa", "matharena", "livecodebench"}:
                raise ValueError("Each source must declare a supported preparation kind")
            if _positive(source.get("subset_count"), "subset_count") > _positive(source.get("expected_count"), "expected_count"):
                raise ValueError("Source subset_count exceeds its full expected_count")
            if source["kind"] == "ifbench":
                from .ifbench import prepare_ifbench
                checkout = (base / source["checkout"]).resolve()
                _checkout(checkout, "https://github.com/allenai/IFBench.git", source["revision"], download)
                rows, provenance = prepare_ifbench(str(checkout), source["revision"])
                scorer = {"type": "ifbench", "checkout": str(checkout), "revision": source["revision"], "mode": "loose"}
            else:
                raw, files = _source_files(source, base, download)
                if len(raw) != source["release_count"]:
                    raise ValueError(f"Pinned release has {len(raw)} rows, expected {source['release_count']}")
                provenance = {"repository": source["repository"], "revision": source["revision"], "files": files,
                              "split": source["split"], "release_count": len(raw)}
                if source["kind"] == "gpqa":
                    rows = normalize_gpqa(raw, source.get("option_seed", 0))
                    scorer = {"type": "multiple_choice"}
                elif source["kind"] == "matharena":
                    rows = normalize_matharena(raw, task)
                    scorer = {"type": "math_equivalence"}
                elif source["kind"] == "livecodebench":
                    from .livecodebench import normalize_livecodebench, validate_checkout
                    start, end = source.get("start_date"), source.get("end_date")
                    for date in (start, end):
                        if date:
                            datetime.strptime(date, "%Y-%m-%d")
                    if start and end and start > end:
                        raise ValueError("LiveCodeBench start_date is after end_date")
                    raw = [row for row in raw if (not start or row["contest_date"][:10] >= start)
                           and (not end or row["contest_date"][:10] <= end)]
                    rows = normalize_livecodebench(raw)
                    checkout = (base / source["grader_checkout"]).resolve()
                    _checkout(checkout, "https://github.com/LiveCodeBench/LiveCodeBench.git", source["grader_revision"], download)
                    validate_checkout(str(checkout), source["grader_revision"])
                    scorer = {"type": "livecodebench", "checkout": str(checkout),
                              "revision": source["grader_revision"], "image_env": "LCB_SANDBOX_IMAGE", "timeout": 6}
                    provenance.update(version_tag=source["version_tag"], start_date=start, end_date=end)
                else:
                    raise ValueError("Unknown dataset preparation kind")
            for row in rows:
                row["benchmark"] = task
            ids = [row["question_id"] for row in rows]
            if len(ids) != len(set(ids)) or len(rows) != source["expected_count"]:
                raise ValueError(f"Invalid {task} coverage: {len(rows)} rows; expected {source['expected_count']}")
            bank = output / f"{task}.jsonl"
            content = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
            if bank.exists() and bank.read_text(encoding="utf-8") != content:
                raise ValueError(f"Refusing to overwrite a different frozen bank: {bank}")
            bank.write_text(content, encoding="utf-8")
            manifest = {"benchmark": task, "provenance": provenance, "ids": ids,
                        "count": len(rows), "bank_sha256": file_digest(bank)}
            bind(output / f"{task}.source.json", manifest)
            suite = {"version": 1, "suite": task, "bank": bank.name,
                     "dataset": {"name": task, "revision": digest(provenance), "split": source.get("split", "test"),
                                 "expected_counts": {task: len(rows)}},
                     "model": {"revision": model_revision, "base_url_env": "BENCHMARK_BASE_URL",
                               "model_env": "BENCHMARK_MODEL", "api_key_env": "BENCHMARK_API_KEY", "api_key_required": True},
                     "generation": {"temperature": 0.0 if task == "ifbench" else 0.6, "max_tokens": 16384},
                     "samples": 1, "subset": {"seed": config.get("seed", 20261017), "stratify_by": "stratum",
                                               "counts": {task: source["subset_count"]}},
                     "benchmarks": {task: {"scorer": scorer}}}
            target = output / f"{task}.yaml"
            text = yaml.safe_dump(suite, sort_keys=False, allow_unicode=True)
            if target.exists() and target.read_text(encoding="utf-8") != text:
                raise ValueError("Prepared suite already exists with different configuration; use a new output directory")
            target.write_text(text, encoding="utf-8")
            report["tasks"][task] = {"ready": True, "count": len(rows), "subset_count": source["subset_count"],
                                     "config": str(target), "source_manifest": str(output / f"{task}.source.json")}
        except (ValueError, OSError, ImportError, KeyError, TypeError, subprocess.SubprocessError) as error:
            report["complete"] = False
            message = str(error)
            for key, value in os.environ.items():
                if value and any(part in key.upper() for part in ("TOKEN", "KEY", "SECRET", "PASSWORD")):
                    message = message.replace(value, "[REDACTED]")
            report["tasks"][task] = {"ready": False, "error_type": type(error).__name__, "message": message}
    write_json(output / "preparation_summary.json", report)
    return report
