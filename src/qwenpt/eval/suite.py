"""Portable, manifest-bound subset/full evaluation of prepared question banks.

This API does not alter the Linux/vLLM BOLT runner or its fixed job protocol.
Generation and scoring are deliberately separate so a judge can be replaced
without paying to regenerate model answers.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

SUITE_VERSION = 1
SCORERS = {"exact_match", "multiple_choice", "math_equivalence", "llm_judge", "ifbench", "livecodebench"}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _revision(value: Any, label: str) -> str:
    if (not isinstance(value, str) or not value.strip()
            or value.lower() in {"latest", "main", "master"}
            or value.upper().startswith(("FILL", "REPLACE", "TODO", "YOUR_", "CHANGE_ME"))):
        raise ValueError(f"{label} must identify a frozen release/checkpoint, not a placeholder or moving branch")
    return value


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"{label} must be a simple nonempty identifier")
    return value


def _no_secrets(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in {"api_key", "apikey", "api-key", "authorization", "access_token",
                                   "token", "secret", "client_secret", "password", "headers", "extra_headers"}:
                raise ValueError("Credentials/headers must not be embedded in configuration; use api_key_env")
            _no_secrets(child)
    elif isinstance(value, list):
        for child in value:
            _no_secrets(child)


def load_env(path: str | Path) -> None:
    """Read a simple KEY=value file without interpolation or overwriting env vars."""
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid environment-file entry at line {number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_config(path: str | Path) -> dict:
    import yaml

    path = Path(path).expanduser().resolve()
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        raise ValueError("Invalid YAML syntax in suite config (content omitted)") from None
    if not isinstance(config, dict):
        raise ValueError("Suite config must be an object")
    allowed = {"version", "suite", "bank", "dataset", "model", "judge", "generation", "subset", "samples", "benchmarks"}
    if set(config) - allowed:
        raise ValueError("Unknown suite configuration field(s)")
    _no_secrets(config)
    if type(config.get("version")) is not int or config["version"] != SUITE_VERSION:
        raise ValueError("Unsupported suite config version; expected 1")
    _name(config.get("suite"), "suite")
    if not isinstance(config.get("bank"), str) or not config["bank"]:
        raise ValueError("bank path is required")
    config["bank"] = str((path.parent / config["bank"]).resolve())
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("dataset metadata is required")
    for key in ("name", "revision", "split"):
        if not isinstance(dataset.get(key), str) or not dataset[key].strip():
            raise ValueError(f"dataset.{key} is required")
    _revision(dataset["revision"], "dataset.revision")
    benchmarks = config.get("benchmarks")
    if not isinstance(benchmarks, dict) or not benchmarks:
        raise ValueError("benchmarks must be a nonempty mapping")
    expected = dataset.get("expected_counts")
    if not isinstance(expected, dict) or set(expected) != set(benchmarks):
        raise ValueError("dataset.expected_counts must cover exactly the configured benchmarks")
    _positive(config.get("samples", 1), "samples")
    for name, settings in benchmarks.items():
        _name(name, "benchmark")
        _positive(expected[name], "expected_counts")
        if not isinstance(settings, dict) or set(settings) - {"scorer", "samples", "generation", "description"}:
            raise ValueError(f"Invalid benchmark settings: {name}")
        _positive(settings.get("samples", config.get("samples", 1)), "samples")
        scorer = settings.get("scorer")
        if not isinstance(scorer, dict) or scorer.get("type") not in SCORERS:
            raise ValueError(f"Unsupported scorer for {name}; use an explicit supported scorer")
        if scorer["type"] in {"ifbench", "livecodebench"}:
            if not isinstance(scorer.get("checkout"), str) or not scorer["checkout"]:
                raise ValueError(f"{name} requires its official grader checkout")
            scorer["checkout"] = str((path.parent / scorer["checkout"]).resolve())
            if not re.fullmatch(r"[0-9a-f]{40}", scorer.get("revision", "")):
                raise ValueError(f"{name} requires a full immutable official grader revision")
        if scorer["type"] == "llm_judge":
            if scorer.get("mode", "correctness") not in {"correctness", "rubric"}:
                raise ValueError("Judge mode must be correctness or rubric")
            if scorer.get("structured_output", "none") not in {"none", "json_object", "json_schema"}:
                raise ValueError("Unknown judge structured_output mode")
        if not isinstance(settings.get("generation", {}), dict):
            raise ValueError("benchmark generation parameters must be an object")
    for key in ("model", "judge", "generation", "subset"):
        if key in config and not isinstance(config[key], dict):
            raise ValueError(f"{key} must be an object")
    endpoint_fields = {"base_url", "model", "base_url_env", "model_env", "api_key_env",
                       "api_key_required", "timeout", "retries", "revision"}
    for key in ("model", "judge"):
        if set(config.get(key, {})) - endpoint_fields:
            raise ValueError(f"Unknown {key} endpoint field; credentials must use api_key_env")
    subset = config.get("subset", {})
    if set(subset) - {"seed", "counts", "stratify_by", "ids_file"}:
        raise ValueError("Unknown subset configuration field(s)")
    if "ids_file" in subset:
        if "counts" in subset:
            raise ValueError("Use subset.ids_file or subset.counts, not both")
        if not isinstance(subset["ids_file"], str) or not subset["ids_file"]:
            raise ValueError("subset.ids_file must be a nonempty path")
        subset["ids_file"] = str((path.parent / subset["ids_file"]).resolve())
    config["_config_path"] = str(path)
    return config


def load_bank(config: dict) -> list[dict]:
    rows, seen = [], set()
    with Path(config["bank"]).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Invalid bank row {number}")
            benchmark, question_id = row.get("benchmark"), row.get("question_id")
            if not isinstance(benchmark, str) or not isinstance(question_id, str) or not question_id.strip():
                raise ValueError(f"Missing benchmark/question_id in bank row {number}")
            key = (benchmark, question_id)
            if key in seen:
                raise ValueError(f"Duplicate bank question: {key}")
            seen.add(key)
            if benchmark not in config["benchmarks"]:
                continue
            if "messages" in row:
                messages = row["messages"]
                if not isinstance(messages, list) or not messages:
                    raise ValueError(f"Invalid messages: {key}")
                for message in messages:
                    if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
                        raise ValueError(f"Invalid message role: {key}")
                    if not isinstance(message.get("content"), (str, list)):
                        raise ValueError(f"Invalid message content: {key}")
                    if message.get("tool_calls") or message.get("function_call"):
                        raise ValueError("Tool conversations require an official agent harness")
            elif not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                raise ValueError(f"Missing prompt/messages: {key}")
            scorer = config["benchmarks"][benchmark]["scorer"]
            if scorer["type"] not in {"ifbench", "livecodebench"} and (
                    scorer["type"] != "llm_judge" or scorer.get("mode", "correctness") == "correctness"):
                if row.get("reference", row.get("answer")) in (None, "", []):
                    raise ValueError(f"Missing reference/answer: {key}")
            if scorer["type"] == "llm_judge":
                rubric = scorer.get("rubric", row.get("rubric"))
                if not isinstance(rubric, str) or not rubric.strip():
                    raise ValueError(f"Missing rubric: {key}")
            if not isinstance(row.get("metadata", {}), dict):
                raise ValueError(f"metadata must be an object: {key}")
            rows.append(row)
    actual = dict(Counter(row["benchmark"] for row in rows))
    if actual != config["dataset"]["expected_counts"]:
        raise ValueError(f"Bank coverage does not match expected_counts: actual={actual}")
    return rows


def _rank(seed: int, *parts: str) -> str:
    return digest([seed, *parts])


def select_questions(config: dict, rows: list[dict], split: str) -> tuple[list[dict], dict]:
    if split not in {"subset", "full"}:
        raise ValueError("split must explicitly be subset or full")
    selection: dict[str, Any] = {"mode": split}
    if split == "full":
        selected = list(rows)
    else:
        settings = config.get("subset", {})
        seed = settings.get("seed", 20261024)
        if type(seed) is not int:
            raise ValueError("subset.seed must be an integer")
        selection["seed"] = seed
        if settings.get("ids_file"):
            path = Path(settings["ids_file"])
            ids = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(ids, list) or not ids:
                raise ValueError("subset IDs must be a nonempty array of benchmark/question_id objects")
            keys = []
            for item in ids:
                if not isinstance(item, dict) or set(item) != {"benchmark", "question_id"}:
                    raise ValueError("Invalid subset ID object")
                if not all(isinstance(value, str) and value for value in item.values()):
                    raise ValueError("Subset IDs must be nonempty strings")
                keys.append((item["benchmark"], item["question_id"]))
            available = {(row["benchmark"], row["question_id"]): row for row in rows}
            if len(set(keys)) != len(keys) or any(key not in available for key in keys):
                raise ValueError("Duplicate or unknown IDs in subset manifest")
            selected = [available[key] for key in keys]
            if {row["benchmark"] for row in selected} != set(config["benchmarks"]):
                raise ValueError("Subset IDs must include every configured benchmark")
            selection.update(method="explicit_ids", ids_sha256=file_digest(path))
        else:
            counts = settings.get("counts")
            if not isinstance(counts, dict) or set(counts) != set(config["benchmarks"]):
                raise ValueError("subset.counts must cover exactly the configured benchmarks")
            stratum_field = settings.get("stratify_by", "stratum")
            if not isinstance(stratum_field, str) or not stratum_field:
                raise ValueError("subset.stratify_by must be a nonempty field name")
            selected = []
            for benchmark in sorted(counts):
                count = _positive(counts[benchmark], "subset count")
                pool = [row for row in rows if row["benchmark"] == benchmark]
                if count > len(pool):
                    raise ValueError(f"Subset count exceeds available questions: {benchmark}")
                buckets: dict[str, list[dict]] = defaultdict(list)
                for row in pool:
                    metadata = row.get("metadata") or {}
                    stratum = row.get(stratum_field, metadata.get(stratum_field, "all"))
                    if not isinstance(stratum, (str, int)):
                        raise ValueError("Strata must be strings or integers")
                    buckets[str(stratum)].append(row)
                for stratum, bucket in buckets.items():
                    bucket.sort(key=lambda row: _rank(seed, benchmark, row["question_id"]))
                ordered = sorted(buckets, key=lambda item: _rank(seed, benchmark, item))
                chosen, depth = [], 0
                while len(chosen) < count:
                    for stratum in ordered:
                        if depth < len(buckets[stratum]):
                            chosen.append(buckets[stratum][depth])
                            if len(chosen) == count:
                                break
                    depth += 1
                selected.extend(chosen)
            selection.update(method="balanced_strata_sha256", counts=counts, stratify_by=stratum_field)
    selected.sort(key=lambda row: (row["benchmark"], row["question_id"]))
    selection.update(
        schema_version=SUITE_VERSION, suite=config["suite"], dataset=config["dataset"],
        bank_sha256=file_digest(Path(config["bank"])),
        available_counts=dict(Counter(row["benchmark"] for row in rows)),
        selected_counts=dict(Counter(row["benchmark"] for row in selected)),
        selected_ids=[{"benchmark": row["benchmark"], "question_id": row["question_id"]} for row in selected],
    )
    selection["selection_fingerprint"] = digest(selection)
    return selected, selection


@contextmanager
def output_lock(output: Path):
    """OS-backed lock, automatically released even when a process is killed."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("Output directory is already in use by another evaluation") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def bind(path: Path, data: dict) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != data:
            raise ValueError(f"Existing {path.name} differs; use a new output directory or score tag")
    else:
        write_json(path, data)


def append_json(path: Path, data: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_records(path: Path) -> list[dict]:
    """Recover only an interrupted, unterminated final JSONL record."""
    if not path.exists():
        return []
    records = []
    with path.open("rb+") as stream:
        while line := stream.readline():
            position = stream.tell() - len(line)
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if line.endswith(b"\n") or stream.read(1):
                    raise ValueError(f"Corrupt completed record in {path.name}") from None
                stream.truncate(position)
                break
            if not isinstance(row, dict):
                raise ValueError(f"Invalid record in {path.name}")
            records.append(row)
            if not line.endswith(b"\n"):
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
    return records


def _samples(config: dict, row: dict) -> int:
    return config["benchmarks"][row["benchmark"]].get("samples", config.get("samples", 1))


def _expected(config: dict, rows: list[dict]) -> dict[tuple[str, str, int], dict]:
    return {(row["benchmark"], row["question_id"], i): row
            for row in rows for i in range(_samples(config, row))}


def _record_key(record: dict) -> tuple[str, str, int]:
    if type(record.get("sample_index")) is not int:
        raise ValueError("Invalid sample index")
    return record.get("benchmark"), record.get("question_id"), record["sample_index"]


def _validate_records(records: list[dict], expected: dict, fingerprint: str, *, scoring=False) -> dict:
    done = {}
    for record in records:
        key = _record_key(record)
        if key not in expected or key in done or record.get("run_fingerprint") != fingerprint:
            raise ValueError("Unknown, duplicate or incompatible cached result")
        if scoring:
            value = record.get("grade", {}).get("score")
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid cached score")
        elif (not isinstance(record.get("text"), str)
              or not record["text"].strip() and record.get("finish_reason") != "length"
              or record.get("finish_reason") not in {"stop", "length"}):
            raise ValueError("Invalid cached completion")
        done[key] = record
    return done


def _summary(config: dict, selection: dict, expected: dict, done: dict, backend: str, stage: str) -> dict:
    benchmarks = {}
    for name in config["benchmarks"]:
        keys = [key for key in expected if key[0] == name]
        results = [done[key] for key in keys if key in done]
        item = {"available_questions": selection["available_counts"][name],
                "selected_questions": selection["selected_counts"][name],
                "expected_samples": len(keys), "completed_samples": len(results),
                "missing_samples": len(keys) - len(results)}
        if stage == "score":
            item["metrics"] = {}
            metrics = {record["grade"]["metric"] for record in results}
            for metric in metrics:
                scores = [record["grade"]["score"] for record in results if record["grade"]["metric"] == metric]
                item["metrics"][metric] = {"mean_on_scored": sum(scores) / len(scores), "scored_samples": len(scores)}
            # A partial run never receives a final mean that hides missing items.
            item["final_mean_score"] = (sum(r["grade"]["score"] for r in results) / len(keys)
                                        if len(results) == len(keys) else None)
            if results and all("ifbench" in record["grade"] for record in results):
                item["ifbench"] = {}
                for mode in ("strict", "loose"):
                    parts = [record["grade"]["ifbench"][mode] for record in results]
                    total = sum(part["instruction_total"] for part in parts)
                    correct = sum(part["instruction_correct"] for part in parts)
                    item["ifbench"][mode] = {"instruction_correct": correct, "instruction_total": total,
                        "instruction_accuracy_on_scored": correct / total,
                        "prompt_accuracy_on_scored": sum(p["prompt_accuracy"] for p in parts) / len(parts)}
        else:
            item["length_limited_samples"] = sum(r["finish_reason"] == "length" for r in results)
            item["elapsed_seconds"] = sum(r.get("elapsed_seconds", 0) for r in results)
        benchmarks[name] = item
    return {"schema_version": SUITE_VERSION, "suite": config["suite"], "split": selection["mode"],
            "dataset": config["dataset"], "selection_fingerprint": selection["selection_fingerprint"],
            "backend": backend, "is_mock": backend == "mock", "stage": stage,
            "complete": len(done) == len(expected),
            "full_coverage": selection["mode"] == "full" and len(done) == len(expected),
            "benchmarks": benchmarks}


def prepare(config: dict, split: str) -> tuple[list[dict], dict]:
    _no_secrets(config)
    return select_questions(config, load_bank(config), split)


def preflight_scorers(config: dict) -> dict:
    """Resolve official scorer provenance before a run, without calling a model."""
    result = {}
    for name, settings in config["benchmarks"].items():
        scorer = settings["scorer"]
        if scorer["type"] == "ifbench":
            from .ifbench import preflight_ifbench
            result[name] = preflight_ifbench(scorer["checkout"], scorer["revision"])
        elif scorer["type"] == "livecodebench":
            from .livecodebench import preflight_livecodebench
            result[name] = preflight_livecodebench(scorer)
    return result


def plan(config: dict, split: str, output: str | Path) -> dict:
    """Freeze selection offline. Full means the complete declared bank/split."""
    _, selection = prepare(config, split)
    output = Path(output)
    with output_lock(output):
        bind(output / "selection.json", selection)
    return selection


def run(config: dict, split: str, output: str | Path, backend: str = "api", client=None) -> dict:
    """Generate selected questions, preserving complete samples across retries."""
    from .api_backend import ChatClient, endpoint_metadata, resolve_endpoint

    if backend not in {"api", "mock"}:
        raise ValueError("backend must be api or mock")
    rows, selection = prepare(config, split)
    model = config.get("model", {})
    revision = model.get("revision")
    _revision(revision, "model.revision")
    if backend == "mock":
        if any(not isinstance(row.get("mock_response"), str) or not row["mock_response"].strip() for row in rows):
            raise ValueError("Mock mode requires explicit mock_response in every selected row")
        endpoint = {"model": "mock", "revision": revision}
    else:
        endpoint = endpoint_metadata(model)
        resolve_endpoint(model)
        client = client or ChatClient(model)  # validate credentials before writing artifacts
    expected = _expected(config, rows)
    generation = {name: {**config.get("generation", {}), **settings.get("generation", {})}
                  for name, settings in config["benchmarks"].items()}
    for params in generation.values():
        if set(params) & {"messages", "model", "stream", "n", "api_key", "headers"}:
            raise ValueError("Generation settings may not override messages/model/stream/n/credentials")
        if "max_tokens" in params and "max_completion_tokens" in params:
            raise ValueError("Choose max_tokens or max_completion_tokens, not both")
    binding = {"schema_version": SUITE_VERSION, "selection_fingerprint": selection["selection_fingerprint"],
               "backend": backend, "endpoint": endpoint, "model_revision": revision, "model_config": model,
               "generation": generation,
               "samples": {name: settings.get("samples", config.get("samples", 1))
                           for name, settings in config["benchmarks"].items()},
               "code_sha256": {name: file_digest(Path(__file__).with_name(name))
                               for name in ("suite.py", "api_backend.py")}}
    manifest = {"run_fingerprint": digest(binding), "config": binding}
    output = Path(output)
    with output_lock(output):
        if (output / "predictions.jsonl").exists() and not (output / "manifest.json").exists():
            raise ValueError("Predictions without a generation manifest cannot be resumed")
        bind(output / "selection.json", selection)
        bind(output / "manifest.json", manifest)
        done = _validate_records(read_records(output / "predictions.jsonl"), expected, manifest["run_fingerprint"])
        for key, row in expected.items():
            if key in done:
                continue
            started = time.monotonic()
            try:
                if backend == "mock":
                    result = {"text": row["mock_response"], "reasoning": "", "finish_reason": "stop",
                              "usage": {}, "response_model": "mock"}
                else:
                    messages = row.get("messages") or [{"role": "user", "content": row["prompt"]}]
                    result = client.complete(messages, generation[key[0]])
                record = {"benchmark": key[0], "question_id": key[1], "sample_index": key[2],
                          "run_fingerprint": manifest["run_fingerprint"], "elapsed_seconds": time.monotonic() - started,
                          **{field: result.get(field) for field in ("text", "reasoning", "finish_reason", "usage", "response_model")}}
                _validate_records([record], expected, manifest["run_fingerprint"])
                append_json(output / "predictions.jsonl", record)
                done[key] = record
            except (ValueError, RuntimeError, OSError) as error:
                # Never persist arbitrary transport errors or remote bodies (may contain credentials).
                append_json(output / "errors.jsonl", {"benchmark": key[0], "question_id": key[1],
                            "sample_index": key[2], "stage": "generation", "error_type": type(error).__name__,
                            "message": "Generation failed; verify endpoint, request parameters and server health"})
        summary = _summary(config, selection, expected, done, backend, "generation")
        summary["run_fingerprint"] = manifest["run_fingerprint"]
        summary["model"] = endpoint
        write_json(output / "generation_summary.json", summary)
    return summary


def score(config: dict, split: str, output: str | Path, score_tag: str = "default", judge=None) -> dict:
    """Score cached predictions; a new judge/rubric requires a new score tag."""
    from .api_backend import ChatClient, endpoint_metadata, resolve_endpoint
    from .api_scoring import SCORER_VERSION, score_answer

    _name(score_tag, "score tag")
    rows, selection = prepare(config, split)
    expected = _expected(config, rows)
    official_scorers = preflight_scorers(config)
    output = Path(output)
    if not (output / "manifest.json").exists():
        raise ValueError("Generation manifest missing; run generation first")
    uses_judge = any(s["scorer"]["type"] == "llm_judge" for s in config["benchmarks"].values())
    judge_metadata = None
    if uses_judge:
        if not config.get("judge"):
            raise ValueError("LLM scoring requires explicit judge configuration")
        _revision(config["judge"].get("revision"), "judge.revision")
        judge_metadata = endpoint_metadata(config["judge"])
        resolve_endpoint(config["judge"])
        judge = judge or ChatClient(config["judge"])
    with output_lock(output):
        # Scoring can change only scoring settings, never coverage or generation identity.
        bind(output / "selection.json", selection)
        generation = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if generation.get("run_fingerprint") != digest(generation.get("config")):
            raise ValueError("Generation manifest fingerprint is invalid")
        current_samples = {name: s.get("samples", config.get("samples", 1)) for name, s in config["benchmarks"].items()}
        if generation["config"]["samples"] != current_samples:
            raise ValueError("Configured sample counts differ from generation")
        if generation["config"]["model_revision"] != config.get("model", {}).get("revision"):
            raise ValueError("Configured model revision differs from generation")
        if generation["config"]["model_config"] != config.get("model", {}):
            raise ValueError("Configured model endpoint differs from generation")
        parameters = {name: {**config.get("generation", {}), **settings.get("generation", {})}
                      for name, settings in config["benchmarks"].items()}
        if generation["config"]["generation"] != parameters:
            raise ValueError("Configured generation parameters differ from generation")
        if generation["config"]["selection_fingerprint"] != selection["selection_fingerprint"]:
            raise ValueError("Generation selection fingerprint mismatch")
        predictions = _validate_records(read_records(output / "predictions.jsonl"), expected, generation["run_fingerprint"])
        score_dir = output / "scores" / score_tag
        score_dir.mkdir(parents=True, exist_ok=True)
        binding = {"generation_fingerprint": generation["run_fingerprint"],
                   "scorers": {name: s["scorer"] for name, s in config["benchmarks"].items()},
                   "judge": judge_metadata, "judge_revision": config.get("judge", {}).get("revision"),
                   "scorer_version": SCORER_VERSION,
                   "official_scorers": official_scorers,
                   "code_sha256": {name: file_digest(Path(__file__).with_name(name))
                                   for name in ("suite.py", "api_scoring.py", "api_backend.py", "scoring.py")}}
        for name in ("ifbench.py", "livecodebench.py", "lcb_worker.py"):
            binding["code_sha256"][name] = file_digest(Path(__file__).with_name(name))
        if any(s["scorer"]["type"] == "math_equivalence" for s in config["benchmarks"].values()):
            import importlib.metadata
            binding["code_sha256"]["bolt_math.py"] = file_digest(Path(__file__).parent.parent / "train" / "utils" / "bolt_math.py")
            binding["math_dependencies"] = {name: importlib.metadata.version(name)
                                             for name in ("math-verify", "latex2sympy2-extended")}
        manifest = {"run_fingerprint": digest(binding), "config": binding}
        if (score_dir / "scores.jsonl").exists() and not (score_dir / "manifest.json").exists():
            raise ValueError("Scores without an identifying manifest cannot be resumed")
        bind(score_dir / "manifest.json", manifest)
        done = _validate_records(read_records(score_dir / "scores.jsonl"), expected, manifest["run_fingerprint"], scoring=True)
        for key, record in done.items():
            if key not in predictions or record.get("prediction_sha256") != digest(predictions[key]):
                raise ValueError("A cached grade no longer matches its source prediction")
        for key, prediction in predictions.items():
            if key in done:
                continue
            try:
                question = expected[key]
                grade = score_answer(question, prediction["text"], config["benchmarks"][key[0]]["scorer"], judge)
                record = {"benchmark": key[0], "question_id": key[1], "sample_index": key[2],
                          "run_fingerprint": manifest["run_fingerprint"], "prediction_sha256": digest(prediction), "grade": grade}
                _validate_records([record], expected, manifest["run_fingerprint"], scoring=True)
                append_json(score_dir / "scores.jsonl", record)
                done[key] = record
            except (ValueError, RuntimeError, OSError, ImportError) as error:
                append_json(score_dir / "errors.jsonl", {"benchmark": key[0], "question_id": key[1],
                            "sample_index": key[2], "stage": "scoring", "error_type": type(error).__name__,
                            "message": "Scoring failed; verify scorer dependencies or judge response/configuration"})
        summary = _summary(config, selection, expected, done, generation["config"]["backend"], "score")
        summary.update(score_tag=score_tag, run_fingerprint=manifest["run_fingerprint"], judge=judge_metadata,
                       model=generation["config"]["endpoint"])
        write_json(score_dir / "summary.json", summary)
    return summary
