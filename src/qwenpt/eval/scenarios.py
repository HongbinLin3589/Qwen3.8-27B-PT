"""Pinned native adapters for browser, spreadsheet, desktop and phone evaluation.

WebArena-Verified and SpreadsheetBench consume artifacts from the project's
agent; they do not pretend to contain an autonomous browser/Excel agent.
OSWorld's PromptAgent and AndroidWorld's M3A retain their native execution and
state evaluators; only their hard-coded model transport is replaced.

Audited upstream interfaces (2026-10-29):
https://github.com/ServiceNow/webarena-verified (6473f72db5dcefc97b5725b59e734504edc28a21)
https://github.com/RUCKBReasoning/SpreadsheetBench (49b73a94775fb489063f60ca1865e3a650079a79)
https://github.com/xlang-ai/OSWorld (044bc33901cb893808e3f91ebc9a17ed8e54b44c)
https://github.com/google-research/android_world (55ddaf5edc6f6c25673aed343907d5b9dca66f84)
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlsplit

if not __package__:
    # Native workers are launched by absolute script path in their own venv.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ENGINES = {"webarena_verified", "spreadsheetbench_verified", "osworld_verified", "androidworld"}
ARTIFACT_ENGINES = {"webarena_verified", "spreadsheetbench_verified"}
OFFICE_DOMAINS = ("libreoffice_calc", "libreoffice_impress", "libreoffice_writer")
DEFAULT_TASKS = {"webarena_verified": "assets/dataset/webarena-verified.json",
                 "osworld_verified": "evaluation_examples/test_all.json",
                 "androidworld": "android_world/task_metadata.json"}


def _read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identifier(value: Any, label: str) -> str:
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in {".", ".."}:
        raise ValueError(f"Invalid {label}; expected a path-safe identifier")
    return value


def _positive(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validate_task(task: dict) -> None:
    if task.get("engine") not in ENGINES:
        raise ValueError("Unknown native scenario engine")
    common = {"id", "engine", "checkout", "revision", "tasks_path", "expected_count", "subset_count", "seed",
              "subset_ids_file", "executable", "model_revision", "model_env", "base_url_env", "api_key_env",
              "api_key_required", "attempts_subset", "attempts_full", "timeout", "api_timeout"}
    engine_fields = {
        "webarena_verified": {"native_config_file", "native_results_path", "native_run_manifest_file"},
        "spreadsheetbench_verified": {"native_results_path", "native_run_manifest_file", "recalculation_backend", "soffice"},
        "osworld_verified": {"domains", "provider_name", "vm_path", "observation_type", "action_space", "max_steps",
                             "max_tokens", "temperature", "top_p"},
        "androidworld": {"adb_path", "console_port", "max_tokens", "temperature"},
    }
    if type(task.get("seed", 42)) is not int:
        raise ValueError("scenario.seed must be an integer")
    for field in ("max_steps", "max_tokens", "console_port"):
        if field in task:
            _positive(task[field], field)
    for field in ("timeout", "api_timeout"):
        value = task.get(field, 120)
        if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
            raise ValueError(f"scenario.{field} must be positive and finite")
    _identifier(task.get("id", task["engine"]), "scenario id")
    if not re.fullmatch(r"[0-9a-f]{40}", str(task.get("revision", ""))):
        raise ValueError("scenario.revision must be the exact 40-character checkout commit")
    revision = task.get("model_revision", "")
    if (not isinstance(revision, str) or not revision.strip() or revision.lower() in {"main", "master", "latest"}
            or revision.upper().startswith(("FILL", "TODO", "REPLACE", "YOUR_"))):
        raise ValueError("scenario.model_revision must identify a frozen model/checkpoint")
    for name in ("model_env", "base_url_env", "api_key_env"):
        if name in task and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(task[name])):
            raise ValueError(f"{name} must be an environment-variable name")
    forbidden = {"api_key", "token", "password", "authorization", "headers", "command", "shell", "argv"}
    def inspect(value):
        if isinstance(value, dict):
            if forbidden.intersection(str(key).lower() for key in value):
                raise ValueError("Use environment variables and named native adapters, not inline secrets or shell commands")
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
    inspect(task)
    if set(task) - common - engine_fields[task["engine"]]:
        raise ValueError("Unknown native scenario setting; it would not reach the official runner")


def _checkout_errors(checkout: Path, revision: str) -> list[str]:
    if not checkout.is_dir():
        return ["Pinned native checkout is missing"]
    try:
        root = subprocess.run(["git", "-C", str(checkout), "rev-parse", "--show-toplevel"],
                              capture_output=True, text=True, timeout=15, check=True)
        if Path(root.stdout.strip()).resolve() != checkout.resolve():
            return ["Native checkout must be the repository root"]
        result = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=15, check=True)
        if result.stdout.strip() != revision:
            return ["Native checkout HEAD does not match configured revision"]
        dirty = subprocess.run(["git", "-C", str(checkout), "diff", "--quiet", "HEAD", "--"],
                               capture_output=True, timeout=15)
        if dirty.returncode:
            return ["Native checkout has tracked modifications; commit and pin them before evaluation"]
    except (OSError, subprocess.SubprocessError):
        return ["Cannot verify native Git checkout"]
    return []


def _inventory(task: dict, checkout: Path) -> tuple[Path, dict[str, dict]]:
    engine = task["engine"]
    source = Path(task.get("tasks_path", checkout / DEFAULT_TASKS.get(engine, "dataset.json"))).resolve()
    data = _read(source)
    rows: dict[str, dict] = {}
    if engine == "osworld_verified":
        if not isinstance(data, dict):
            raise ValueError("OSWorld task metadata must map domain to native task IDs")
        domains = task.get("domains", list(OFFICE_DOMAINS))
        if not isinstance(domains, list) or not domains or len(set(domains)) != len(domains):
            raise ValueError("OSWorld domains must be a nonempty unique list")
        for domain in domains:
            _identifier(domain, "OSWorld domain")
            if domain not in data or not isinstance(data[domain], list):
                raise ValueError("Configured OSWorld domain is missing from native metadata")
            for native_id in data[domain]:
                native_id = _identifier(native_id, "OSWorld task id")
                key = f"{domain}/{native_id}"
                if key in rows:
                    raise ValueError("Duplicate native scenario task ID")
                rows[key] = {"id": native_id, "domain": domain}
    else:
        if not isinstance(data, list) or not data:
            raise ValueError("Native task inventory must be a nonempty JSON list")
        key_field = {"webarena_verified": "task_id", "spreadsheetbench_verified": "id", "androidworld": "task_name"}[engine]
        for row in data:
            if not isinstance(row, dict) or key_field not in row:
                raise ValueError(f"Native task inventory is missing {key_field}")
            key = _identifier(row[key_field], "native task id")
            if key in rows:
                raise ValueError("Duplicate native scenario task ID")
            rows[key] = row
    expected = _positive(task.get("expected_count"), "expected_count")
    if len(rows) != expected:
        raise ValueError(f"Native full inventory coverage mismatch: expected {expected}, found {len(rows)}")
    return source, rows


def _select(task: dict, rows: dict, split: str) -> list[str]:
    if split == "full":
        return sorted(rows)
    if task.get("subset_ids_file"):
        selected = _read(task["subset_ids_file"])
        if not isinstance(selected, list) or not selected or not all(isinstance(x, str) for x in selected):
            raise ValueError("subset_ids_file must contain a nonempty list of native IDs")
        if len(set(selected)) != len(selected) or set(selected) - set(rows):
            raise ValueError("Scenario subset contains duplicate or unknown task IDs")
        expected = _positive(task.get("subset_count"), "subset_count")
        if len(selected) != expected:
            raise ValueError(f"Scenario explicit subset coverage mismatch: subset_count requires {expected} IDs, found {len(selected)}")
        return sorted(selected)
    count = _positive(task.get("subset_count"), "subset_count")
    if count > len(rows):
        raise ValueError("Scenario subset_count exceeds the full inventory")
    seed = task.get("seed", 42)
    # Round-robin source domains/categories, using only native metadata labels.
    groups: dict[str, list[str]] = {}
    for key, row in rows.items():
        group = str(row.get("domain", row.get("instruction_type", row.get("sites", row.get("difficulty", "all")))))
        groups.setdefault(group, []).append(key)
    for keys in groups.values():
        keys.sort(key=lambda key: _hash([seed, task["engine"], key]))
    selected = []
    while len(selected) < count:
        for group in sorted(groups):
            if groups[group] and len(selected) < count:
                selected.append(groups[group].pop(0))
    return sorted(selected)


def _endpoint(task: dict) -> dict:
    return {"model_env": task.get("model_env", "BENCHMARK_MODEL"),
            "base_url_env": task.get("base_url_env", "BENCHMARK_BASE_URL"),
            "api_key_env": task.get("api_key_env", "BENCHMARK_API_KEY"),
            "api_key_required": task.get("api_key_required", True),
            "revision": task["model_revision"], "timeout": task.get("api_timeout", 120), "retries": 2}


def _resolved_endpoint(task: dict) -> dict:
    from qwenpt.eval.api_backend import endpoint_metadata
    endpoint = _endpoint(task)
    result = {key: os.environ.get(endpoint[key + "_env"], "") for key in ("model", "base_url")}
    if result["base_url"]:
        # Endpoint validation is shared with the single-turn client. Artifact
        # scoring may omit a model service entirely, so preserve that case.
        if result["model"]:
            metadata = endpoint_metadata(endpoint)
            result.update(model=metadata["model"], base_url=metadata["base_url"])
        else:
            from qwenpt.eval.api_backend import _safe_url
            result["base_url"] = _safe_url(result["base_url"])
    return result


def _adapter_files() -> dict:
    return {name: _file_hash(Path(__file__).with_name(name)) for name in ("scenarios.py", "api_backend.py")}


def _check_adapter(plan: dict) -> None:
    if plan.get("adapter_files") != _adapter_files():
        raise ValueError("Scenario adapter/transport changed; use a new output directory for this protocol")


def _python(task: dict) -> str:
    candidate = str(task.get("executable", sys.executable))
    found = shutil.which(candidate)
    if found:
        return found
    if Path(candidate).is_file():
        return str(Path(candidate).resolve())
    raise ValueError("Native Python executable is unavailable")


def _dependency_errors(task: dict, checkout: Path, python: str) -> list[str]:
    modules = {"webarena_verified": ["webarena_verified", "pydantic", "typer"],
               "spreadsheetbench_verified": ["openpyxl", "numpy", "tqdm"],
               "osworld_verified": ["desktop_env", "mm_agents", "openai", "PIL", "requests"],
               "androidworld": ["android_world", "android_env", "absl", "PIL", "numpy"]}[task["engine"]]
    if task["engine"] == "spreadsheetbench_verified" and task.get("recalculation_backend") == "win32com":
        modules += ["win32com"]
    code = ("import importlib.util,json,sys;sys.path[:0]=json.loads(sys.argv[1]);"
            "print(json.dumps([x for x in json.loads(sys.argv[2]) if importlib.util.find_spec(x) is None]))")
    try:
        probe = subprocess.run([python, "-c", code, json.dumps([str(checkout), str(checkout / "src")]),
                                json.dumps(modules)], capture_output=True, text=True, timeout=20, check=True)
        missing = json.loads(probe.stdout)
        return ["Native Python dependency missing: " + name for name in missing]
    except (OSError, subprocess.SubprocessError, ValueError):
        return ["Native Python dependency probe failed"]


def prepare_scenario(task: dict, split: str, output: str | Path) -> dict:
    """Freeze native IDs and commands without calling models or starting environments."""
    _validate_task(task)
    if split not in {"subset", "full"}:
        raise ValueError("split must be subset or full")
    output = Path(output).resolve()
    checkout = Path(task["checkout"]).resolve()
    errors = _checkout_errors(checkout, task["revision"])
    if errors:
        return {"id": task.get("id", task["engine"]), "engine": task["engine"], "split": split,
                "ready": False, "complete": False, "status": "unavailable", "blockers": errors}
    source, rows = _inventory(task, checkout)
    selected = _select(task, rows, split)
    attempts = _positive(task.get("attempts_" + split, 1), "attempts_" + split)
    python = _python(task)
    engine = task["engine"]
    pinned_files = {str(source): _file_hash(source)}
    if engine == "spreadsheetbench_verified":
        for key in selected:
            row = rows[key]
            if not isinstance(row.get("spreadsheet_path"), str):
                raise ValueError("Spreadsheet inventory needs spreadsheet_path for every task")
            folder = (source.parent / row["spreadsheet_path"]).resolve()
            if not folder.is_relative_to(source.parent):
                raise ValueError("Spreadsheet native data path leaves the dataset")
            answers = list(folder.glob(f"*_{key}_answer.xlsx"))
            if not answers:
                errors.append(f"Missing native spreadsheet answer workbooks: {key}")
            for answer in answers:
                pinned_files[str(answer)] = _file_hash(answer)
                original = answer.with_name(answer.name.replace("_answer.xlsx", "_input.xlsx"))
                if original.is_file():
                    pinned_files[str(original)] = _file_hash(original)
                else:
                    errors.append(f"Missing native spreadsheet input workbook: {key}")
    for field in ("native_config_file", "subset_ids_file"):
        if task.get(field) and (field != "subset_ids_file" or split == "subset"):
            path = Path(task[field]).resolve()
            pinned_files[str(path)] = _file_hash(path)
    plan = {"id": task.get("id", engine), "engine": engine, "split": split,
            "adapter_files": _adapter_files(),
            "revision": task["revision"], "model_revision": task["model_revision"],
            "task_ids": selected, "available_count": len(rows), "selected_count": len(selected),
            "inventory_path": str(source),
            "attempts": attempts, "expected_samples": len(selected) * attempts,
            "pinned_files": pinned_files, "generation": "external_native_artifacts" if engine in ARTIFACT_ENGINES else "native_agent",
            "endpoint": _endpoint(task), "resolved_endpoint": _resolved_endpoint(task),
            "source_scope": task.get("domains", list(OFFICE_DOMAINS)) if engine == "osworld_verified" else "declared_native_inventory",
            "task": dict(task), "output": str(output), "checkout": str(checkout), "python": python}
    plan["selection_fingerprint"] = _hash({k: v for k, v in plan.items() if k not in {"output", "python"}})
    saved = output / "scenario_plan.json"
    if saved.exists() and _read(saved).get("selection_fingerprint") != plan["selection_fingerprint"]:
        raise ValueError("Scenario output belongs to a different model, selection or native configuration")
    commands = []
    worker = str(Path(__file__).resolve())
    for attempt in range(attempts):
        attempt_dir = output / f"attempt-{attempt:03d}"
        commands.append({"attempt": attempt, "cwd": str(checkout),
                         "argv": [python, worker, "native", "--plan", str(saved), "--attempt", str(attempt)],
                         "artifact_dir": str(attempt_dir)})
    plan["commands"] = commands
    plan["blockers"] = errors + _dependency_errors(task, checkout, python)
    if engine == "webarena_verified" and not task.get("native_config_file"):
        plan["blockers"].append("WebArena environment native_config_file is required")
    if engine == "spreadsheetbench_verified":
        if not task.get("recalculation_backend") in {"libreoffice", "win32com"}:
            plan["blockers"].append("Specify the spreadsheet recalculation_backend: libreoffice or win32com")
        elif task["recalculation_backend"] == "libreoffice" and not (shutil.which(str(task.get("soffice", "libreoffice"))) or Path(str(task.get("soffice", ""))).is_file()):
            plan["blockers"].append("LibreOffice executable unavailable; specify soffice")
    if engine == "androidworld" and not Path(task.get("adb_path", "")).is_file():
        plan["blockers"].append("AndroidWorld requires an installed adb_path and a prepared emulator")
    if engine not in ARTIFACT_ENGINES:
        for key in ("model_env", "base_url_env"):
            if not os.environ.get(plan["endpoint"][key]):
                plan["blockers"].append("Missing model environment variable: " + plan["endpoint"][key])
        if plan["endpoint"]["api_key_required"] and not os.environ.get(plan["endpoint"]["api_key_env"]):
            plan["blockers"].append("Missing API key environment variable: " + plan["endpoint"]["api_key_env"])
        try:
            from .api_backend import ChatClient, resolve_endpoint
            resolve_endpoint(plan["endpoint"])
            ChatClient(plan["endpoint"])
        except (ValueError, TypeError):
            plan["blockers"].append("Invalid model endpoint, API key or request timeout configuration")
    plan["ready"] = not plan["blockers"]
    plan["complete"] = False
    plan["status"] = "prepared" if plan["ready"] else "unavailable"
    _write(saved, plan)
    # Selected native metadata never includes a key or the contents of native config files.
    _write(output / "selected_tasks.json", {key: rows[key] for key in selected})
    if engine in ARTIFACT_ENGINES:
        _write(output / "native_run_manifest.template.json", {field: plan[field] for field in
               ("engine", "revision", "model_revision", "split", "attempts", "task_ids", "selection_fingerprint")})
    return plan


def _artifact_root(plan: dict, attempt: int) -> Path:
    root = Path(plan["task"].get("native_results_path", Path(plan["output"]) / "native-artifacts"))
    return root if plan["attempts"] == 1 else root / f"attempt-{attempt:03d}"


def _validate_import(plan: dict) -> list[str]:
    file = plan["task"].get("native_run_manifest_file")
    if not file or not Path(file).is_file():
        return ["Provide native_run_manifest_file and artifacts generated by the selected model/agent"]
    manifest = _read(file)
    for field in ("engine", "revision", "model_revision", "split", "attempts", "selection_fingerprint"):
        if manifest.get(field) != plan[field]:
            raise ValueError(f"Native artifact provenance mismatch: {field}")
    ids = manifest.get("task_ids")
    if not isinstance(ids, list) or len(ids) != len(set(ids)) or set(ids) != set(plan["task_ids"]):
        raise ValueError("Native artifact provenance mismatch: task_ids")
    errors = []
    rows = _read(Path(plan["output"]) / "selected_tasks.json") if plan["engine"] == "spreadsheetbench_verified" else {}
    for attempt in range(plan["attempts"]):
        root = _artifact_root(plan, attempt)
        for key in plan["task_ids"]:
            if plan["engine"] == "webarena_verified":
                if not all((root / key / name).is_file() for name in ("agent_response.json", "network.har")):
                    errors.append(f"Missing native browser response/HAR: {key}, attempt {attempt}")
            else:
                data_root = Path(plan["inventory_path"]).parent.resolve()
                native_dir = (data_root / rows[key]["spreadsheet_path"]).resolve()
                if not native_dir.is_relative_to(data_root):
                    raise ValueError("Spreadsheet native data path leaves the dataset")
                answers = sorted(native_dir.glob(f"*_{key}_answer.xlsx"))
                if not answers:
                    errors.append(f"Missing native spreadsheet answer workbooks: {key}")
                for answer in answers:
                    case = answer.name.split("_", 1)[0]
                    if not (root / key / f"{case}_{key}_output.xlsx").is_file():
                        errors.append(f"Missing native spreadsheet output: {key}, case {case}, attempt {attempt}")
    return errors


def _redact(text: str) -> str:
    for name, value in os.environ.items():
        if value and any(word in name.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            text = text.replace(value, "[REDACTED]")
    return text


def preflight_scenario(task: dict, split: str, output: str | Path) -> dict:
    """Check runtime dependencies and imported artifacts before any model call."""
    plan = prepare_scenario(task, split, output)
    if not plan.get("ready"):
        return plan
    if plan["engine"] in ARTIFACT_ENGINES:
        missing = _validate_import(plan)
        if missing:
            return {**plan, "ready": False, "complete": False, "status": "needs_native_artifacts", "blockers": missing}
    return plan


def run_scenario(task: dict, split: str, output: str | Path, *, dry_run: bool = False) -> dict:
    """Run audited native entry points; never execute arbitrary shell configuration."""
    plan = preflight_scenario(task, split, output)
    if not plan.get("ready") or dry_run:
        return plan
    for command in plan["commands"]:
        destination = Path(command["artifact_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        _write(destination / "execution.json", {"returncode": None, "status": "running",
               "selection_fingerprint": plan["selection_fingerprint"]})
        try:
            environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
            completed = subprocess.run(command["argv"], cwd=command["cwd"], capture_output=True, text=True,
                                       encoding="utf-8", errors="replace",
                                       timeout=task.get("timeout", 86400), env=environment)
            # Native frameworks can print service errors; credentials never enter saved logs.
            (destination / "native.log").write_text(_redact(completed.stdout + completed.stderr), encoding="utf-8")
            _write(destination / "execution.json", {"returncode": completed.returncode,
                   "elapsed_seconds": time.monotonic() - started, "selection_fingerprint": plan["selection_fingerprint"]})
        except subprocess.TimeoutExpired:
            _write(destination / "execution.json", {"returncode": None, "status": "timeout",
                   "elapsed_seconds": time.monotonic() - started, "selection_fingerprint": plan["selection_fingerprint"]})
        except OSError:
            _write(destination / "execution.json", {"returncode": None, "status": "launch_error",
                   "selection_fingerprint": plan["selection_fingerprint"]})
    return score_scenario(task, split, output)


def _score_number(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Native score must be finite and between zero and one")
    return float(value)


def score_scenario(task: dict, split: str, output: str | Path) -> dict:
    """Read task-level native results; missing or infra-failed tasks keep coverage incomplete."""
    output = Path(output).resolve()
    saved = output / "scenario_plan.json"
    if not saved.is_file():
        raise ValueError("Prepare the scenario before importing/scoring results")
    plan = _read(saved)
    _check_adapter(plan)
    if plan["split"] != split or plan["revision"] != task.get("revision") or plan["model_revision"] != task.get("model_revision"):
        raise ValueError("Scenario model/revision/split does not match the saved plan")
    if plan["task"] != task:
        raise ValueError("Scenario settings differ from the frozen plan")
    # Rebuild selection/protocol but do not require runtime dependencies for pure score import.
    source, inventory = _inventory(task, Path(plan["checkout"]))
    if _file_hash(source) != plan["pinned_files"][str(source)] or _select(task, inventory, split) != plan["task_ids"]:
        raise ValueError("Scenario task inventory or selected task IDs changed")
    for file, expected in plan["pinned_files"].items():
        if not Path(file).is_file() or _file_hash(Path(file)) != expected:
            raise ValueError("Pinned native evaluation input changed after planning")
    import_errors = _validate_import(plan) if plan["engine"] in ARTIFACT_ENGINES else []
    records = []
    for attempt in range(plan["attempts"]):
        directory = output / f"attempt-{attempt:03d}"
        result_file = directory / "native_scores.json"
        execution_file = directory / "execution.json"
        execution = _read(execution_file) if execution_file.is_file() else None
        if execution is not None and execution.get("selection_fingerprint") != plan["selection_fingerprint"]:
            raise ValueError("Native execution provenance differs from the frozen plan")
        execution_ok = execution is not None and type(execution.get("returncode")) is int and execution["returncode"] == 0
        payload = _read(result_file) if execution_ok and result_file.is_file() and not import_errors else None
        if payload is not None and (not isinstance(payload, dict) or payload.get("selection_fingerprint") != plan["selection_fingerprint"]):
            raise ValueError("Native score export provenance differs from the frozen plan")
        if payload:
            for file, expected_hash in payload.get("artifact_sha256", {}).items():
                if not Path(file).is_file() or _file_hash(Path(file)) != expected_hash:
                    raise ValueError("Native source artifact changed after evaluation")
        native = payload.get("records", []) if payload else []
        if not isinstance(native, list):
            raise ValueError("Native score export must be a JSON list")
        by_id = {}
        for item in native:
            key = item.get("task_id")
            if key not in plan["task_ids"] or key in by_id:
                raise ValueError("Native score export contains unknown or duplicate task IDs")
            by_id[key] = item
        for key in plan["task_ids"]:
            default_status = "infra_error" if execution is not None and not execution_ok else "missing"
            item = by_id.get(key, {"task_id": key, "status": default_status})
            status = item.get("status")
            if status not in {"scored", "infra_error", "missing", "parse_error"}:
                raise ValueError("Unrecognized native result status")
            score = _score_number(item.get("score")) if status == "scored" else None
            records.append({"task_id": key, "attempt": attempt, "status": status, "score": score,
                            "evidence": item.get("evidence")})
    scored = [item for item in records if item["status"] == "scored"]
    complete = len(scored) == plan["expected_samples"]
    summary = {"id": plan["id"], "engine": plan["engine"], "split": split,
               "selection_fingerprint": plan["selection_fingerprint"], "model_revision": plan["model_revision"],
               "generation": plan["generation"], "source_scope": plan["source_scope"],
               "expected_samples": plan["expected_samples"], "completed_samples": len(scored),
               "missing_samples": plan["expected_samples"] - len(scored),
               "statuses": dict(Counter(item["status"] for item in records)),
               "complete": complete, "full_coverage": complete and split == "full",
               "mean_on_scored": sum(item["score"] for item in scored) / len(scored) if scored else None,
               "final_mean_score": sum(item["score"] for item in scored) / len(scored) if complete else None,
               "is_mock": False, "records": records}
    if import_errors:
        summary.update(status="needs_native_artifacts", blockers=import_errors)
    _write(output / "scenario_summary.json", summary)
    return summary


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Native module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _client(plan: dict):
    # The worker can run in a dedicated native venv without installing qwenpt.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from qwenpt.eval.api_backend import ChatClient
    client = ChatClient(plan["endpoint"])
    log_path = plan.get("_request_log")
    if not log_path:
        return client
    class LoggedClient:
        def complete(self, messages, parameters):
            started = time.monotonic()
            try:
                result = client.complete(messages, parameters)
            except Exception as error:
                self.record({"status": "error", "error": type(error).__name__,
                             "elapsed_seconds": time.monotonic() - started})
                raise
            self.record({"status": "complete", "usage": result.get("usage", {}),
                         "response_model": result.get("response_model"), "finish_reason": result.get("finish_reason"),
                         "elapsed_seconds": time.monotonic() - started})
            return result
        def record(self, item):
            with Path(log_path).open("a", encoding="utf-8") as stream:
                stream.write(_redact(json.dumps(item, ensure_ascii=False, allow_nan=False)) + "\n")
    return LoggedClient()


def _native_osworld(plan: dict, attempt: int, destination: Path) -> list[dict]:
    checkout = Path(plan["checkout"])
    sys.path.insert(0, str(checkout))
    from mm_agents.agent import PromptAgent
    client = _client(plan)
    # Native PromptAgent's Qwen branch otherwise routes to DashScope. Keep its
    # prompts/history/action parser and replace only the model request function.
    def call_llm(self, payload):
        parameters = {key: payload[key] for key in ("max_tokens", "temperature", "top_p") if key in payload}
        return client.complete(payload["messages"], parameters)["text"]
    PromptAgent.call_llm = call_llm
    model = os.environ[plan["endpoint"]["model_env"]]
    if any(part in {".", ".."} for part in model.replace("\\", "/").split("/")):
        raise ValueError("Unsafe model identifier for native result paths")
    selected: dict[str, list[str]] = {}
    for key in plan["task_ids"]:
        domain, task_id = key.split("/", 1)
        selected.setdefault(domain, []).append(task_id)
    meta = destination / "test_all_meta.json"
    _write(meta, selected)
    task = plan["task"]
    observation = task.get("observation_type", "screenshot")
    action = task.get("action_space", "pyautogui")
    result_dir = destination / "results"
    args = [str(checkout / "run.py"), "--test_all_meta_path", str(meta),
            "--test_config_base_dir", str(checkout / "evaluation_examples"),
            "--result_dir", str(result_dir), "--model", model,
            "--provider_name", task.get("provider_name", "docker"), "--headless",
            "--observation_type", observation, "--action_space", action,
            "--max_steps", str(task.get("max_steps", 50)), "--max_tokens", str(task.get("max_tokens", 2048)),
            "--temperature", str(task.get("temperature", 0)), "--top_p", str(task.get("top_p", 0.95))]
    if task.get("vm_path"):
        args += ["--path_to_vm", task["vm_path"]]
    (checkout / "logs").mkdir(exist_ok=True)
    sys.argv = args
    runpy.run_path(str(checkout / "run.py"), run_name="__main__")
    records = []
    for key in plan["task_ids"]:
        path = result_dir / action / observation / model / key / "result.txt"
        if path.is_file():
            try:
                records.append({"task_id": key, "status": "scored", "score": _score_number(float(path.read_text().strip())), "evidence": str(path)})
            except ValueError:
                records.append({"task_id": key, "status": "parse_error", "evidence": str(path)})
        else:
            error = path.with_name("error.json")
            records.append({"task_id": key, "status": "infra_error" if error.exists() else "missing", "evidence": str(error if error.exists() else path)})
    return records


def _native_android(plan: dict, attempt: int, destination: Path) -> list[dict]:
    checkout = Path(plan["checkout"])
    sys.path.insert(0, str(checkout))
    from android_world import checkpointer, registry, suite_utils
    from android_world.agents import m3a
    from android_world.env import env_launcher
    from PIL import Image
    client = _client(plan)
    task = plan["task"]
    class ModelBridge:
        def predict(self, text_prompt):
            return self.predict_mm(text_prompt, [])
        def predict_mm(self, text_prompt, images):
            content = [{"type": "text", "text": text_prompt}]
            for array in images:
                buffer = io.BytesIO()
                Image.fromarray(array).convert("RGB").save(buffer, format="JPEG")
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}})
            result = client.complete([{"role": "user", "content": content}],
                                     {"max_tokens": task.get("max_tokens", 2048), "temperature": task.get("temperature", 0)})
            return result["text"], None, result
    previous = destination / "native_scores.json"
    export = _read(previous) if previous.exists() else {}
    if export and export.get("selection_fingerprint") != plan["selection_fingerprint"]:
        raise ValueError("Existing AndroidWorld export belongs to another plan")
    records: dict[str, dict] = {item["task_id"]: item for item in export.get("records", [])}
    class ExportCheckpointer(checkpointer.IncrementalCheckpointer):
        def save_episodes(self, episodes, task_name):
            super().save_episodes(episodes, task_name)
            for episode in episodes:
                key = episode["task_template"]
                failed = episode.get("exception_info") is not None
                records[key] = {"task_id": key, "status": "infra_error" if failed else "scored",
                                "score": None if failed else _score_number(episode["is_successful"]),
                                "evidence": str(Path(self.directory) / f"{task_name}.pkl.gz")}
            _write(destination / "native_scores.json", {"selection_fingerprint": plan["selection_fingerprint"], "records": list(records.values())})
    family = registry.TaskRegistry().get_registry(registry.TaskRegistry.ANDROID_WORLD_FAMILY)
    if set(plan["task_ids"]) - set(family):
        raise ValueError("Selected AndroidWorld tasks are not in the pinned native registry")
    suite = suite_utils.create_suite(family, n_task_combinations=1,
                                    seed=task.get("seed", 42) + attempt, tasks=plan["task_ids"], use_identical_params=True)
    suite.suite_family = registry.TaskRegistry.ANDROID_WORLD_FAMILY
    env = env_launcher.load_and_setup_env(console_port=task.get("console_port", 5554), emulator_setup=False,
                                         adb_path=task["adb_path"])
    try:
        agent = m3a.M3A(env, ModelBridge())
        agent.name = "qwenpt_m3a_openai_compatible"
        agent.transition_pause = None
        suite_utils.run(suite, agent, checkpointer=ExportCheckpointer(str(destination / "checkpoints")), demo_mode=False)
    finally:
        env.close()
    return list(records.values())


def _native_webarena(plan: dict, attempt: int, destination: Path) -> list[dict]:
    root = _artifact_root(plan, attempt)
    sys.path.insert(0, str(Path(plan["checkout"]) / "src"))
    from webarena_verified.api import WebArenaVerified
    from webarena_verified.types.config import WebArenaVerifiedConfig
    config_data = _read(plan["task"]["native_config_file"])
    config_data["test_data_file"] = plan["inventory_path"]
    config = WebArenaVerifiedConfig.model_validate(config_data)
    evaluator = WebArenaVerified(config=config)
    records = []
    for key in plan["task_ids"]:
        try:
            result = evaluator.evaluate_task(task_id=int(key), agent_response=root / key / "agent_response.json", network_trace=root / key / "network.har")
            result_path = destination / key / "eval_result.json"
            _write(result_path, result.model_dump(mode="json"))
            status = str(result.status).lower()
            if status not in {"success", "failure", "partial_match", "error"}:
                raise ValueError("Unknown WebArena evaluation status")
            records.append({"task_id": key, "status": "infra_error" if status == "error" else "scored",
                            "score": None if status == "error" else _score_number(result.score), "evidence": str(result_path)})
        except Exception:
            records.append({"task_id": key, "status": "infra_error"})
    return records


def _native_spreadsheet(plan: dict, attempt: int, destination: Path) -> list[dict]:
    checkout = Path(plan["checkout"])
    evaluator = _load_module("qwenpt_native_spreadsheet_evaluator", checkout / "evaluation" / "evaluation.py")
    recalculator = _load_module("qwenpt_native_spreadsheet_recalc", checkout / "evaluation" / "open_spreadsheet.py")
    source = Path(plan["task"]["tasks_path"])
    data_root = source.parent
    rows = _read(Path(plan["output"]) / "selected_tasks.json")
    artifacts = _artifact_root(plan, attempt)
    backend = plan["task"]["recalculation_backend"]
    soffice = (plan["task"].get("soffice") or recalculator.find_libreoffice()) if backend == "libreoffice" else None
    results = []
    for key in plan["task_ids"]:
        row = rows[key]
        native_dir = (data_root / row["spreadsheet_path"]).resolve()
        if not native_dir.is_relative_to(data_root.resolve()):
            raise ValueError("Spreadsheet native data path leaves the dataset")
        answers = sorted(native_dir.glob(f"*_{key}_answer.xlsx"))
        record = {"task_id": key, "status": "scored", "score": None, "evidence": []}
        scores = []
        if not answers:
            record["status"] = "infra_error"
        for answer in answers:
            number = answer.name.split("_", 1)[0]
            produced = artifacts / key / f"{number}_{key}_output.xlsx"
            if not produced.is_file():
                record["status"] = "missing"
                break
            # Recalculate copies, never mutate archived model outputs or gold workbooks.
            local = destination / key / produced.name
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(produced, local)
            try:
                if backend == "libreoffice":
                    recalculated = recalculator.just_open_libreoffice(str(local), soffice)
                else:
                    recalculated = recalculator.just_open_win32com(str(local))
                if not recalculated:
                    raise RuntimeError("Native formula recalculation failed")
                passed, _ = evaluator.compare_workbooks(str(answer), str(local), row["instruction_type"], row["answer_position"])
                scores.append(bool(passed))
                record["evidence"].append(str(local))
            except Exception:
                record["status"] = "infra_error"
                break
        if record["status"] == "scored":
            record["score"] = float(all(scores))
        results.append(record)
    return results


def _native(plan_file: str, attempt: int) -> None:
    plan = _read(plan_file)
    _check_adapter(plan)
    if not 0 <= attempt < plan["attempts"]:
        raise ValueError("Native attempt outside frozen plan")
    for file, expected in plan["pinned_files"].items():
        if _file_hash(Path(file)) != expected:
            raise ValueError("Pinned native input changed after planning")
    if _checkout_errors(Path(plan["checkout"]), plan["revision"]):
        raise ValueError("Native checkout changed after planning")
    if _resolved_endpoint(plan["task"]) != plan["resolved_endpoint"]:
        raise ValueError("Model endpoint changed after planning")
    destination = Path(plan["output"]) / f"attempt-{attempt:03d}"
    destination.mkdir(parents=True, exist_ok=True)
    plan["_request_log"] = str(destination / "model_calls.jsonl")
    drivers = {"osworld_verified": _native_osworld, "androidworld": _native_android,
               "webarena_verified": _native_webarena, "spreadsheetbench_verified": _native_spreadsheet}
    records = drivers[plan["engine"]](plan, attempt, destination)
    hashes = {}
    if plan["engine"] in ARTIFACT_ENGINES:
        root = _artifact_root(plan, attempt)
        for key in plan["task_ids"]:
            for file in (root / key).glob("*"):
                if file.is_file() and file.suffix.lower() in {".har", ".json", ".xlsx"}:
                    hashes[str(file.resolve())] = _file_hash(file)
    _write(destination / "native_scores.json", {"selection_fingerprint": plan["selection_fingerprint"], "artifact_sha256": hashes, "records": records})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Internal native scenario worker")
    parser.add_argument("command", choices=["native"])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    try:
        _native(args.plan, args.attempt)
    except Exception as error:
        # Native exceptions may include service response bodies. Keep details in
        # native evidence, while the worker's stderr is deliberately secret-free.
        print(json.dumps({"status": "infra_error", "error": type(error).__name__}), file=sys.stderr)
        raise SystemExit(2)
