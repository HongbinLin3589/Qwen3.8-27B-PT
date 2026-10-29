"""Coordinate every required capability without dropping unavailable tasks."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re

from . import suite

REQUIRED = {
    "ifbench": ("bank", "core"), "gpqa_diamond": ("bank", "core"), "livecodebench": ("bank", "core"),
    "hmmt_feb_2026": ("bank", "core"), "apex_shortlist": ("bank", "core"),
    "terminal_bench_2_1": ("harness", "core"), "deepswe_1_1": ("harness", "core"),
    "webarena_verified": ("scenario", "scenarios"), "spreadsheetbench_verified": ("scenario", "scenarios"),
    "osworld_verified": ("scenario", "scenarios"), "androidworld": ("scenario", "scenarios"),
}


def load_project(path: str | Path) -> dict:
    import yaml
    path = Path(path).resolve()
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        raise ValueError("Invalid project YAML syntax (content omitted)") from None
    if not isinstance(config, dict):
        raise ValueError("Project config must be an object")
    suite._no_secrets(config)
    if config.get("version") != 1 or not isinstance(config.get("tasks"), dict) or not config["tasks"]:
        raise ValueError("Project config requires version: 1 and nonempty tasks")
    if set(config) - {"version", "project", "requirements", "model_revision_env", "tasks"}:
        raise ValueError("Unknown project configuration field")
    suite._name(config.get("project"), "project")
    revision_env = config.get("model_revision_env", "BENCHMARK_REVISION")
    if not isinstance(revision_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", revision_env):
        raise ValueError("model_revision_env must be an environment-variable name")
    for name, task in config["tasks"].items():
        suite._name(name, "task")
        if not isinstance(task, dict):
            raise ValueError("Each project task must be an object")
        if task.get("kind") not in {"bank", "harness", "scenario"} or task.get("profile") not in {"core", "scenarios"}:
            raise ValueError("Each task must declare its kind and core/scenarios profile")
        if not isinstance(task.get("capability"), str):
            raise ValueError("Task capability is required")
        suite._positive(task.get("subset_count"), "subset_count")
        suite._positive(task.get("full_count"), "full_count")
        if task["subset_count"] > task["full_count"]:
            raise ValueError("Task subset exceeds full count")
        if config.get("project") == "qwen38-general-v1" and name in REQUIRED:
            if (task["kind"], task["profile"]) != REQUIRED[name]:
                raise ValueError("Required capability changed kind/profile; do not replace native tasks with chat grading")
        if task["kind"] == "bank":
            if not isinstance(task.get("config"), str) or not task["config"].strip():
                raise ValueError("Bank task requires a configuration path")
            task["config"] = str((path.parent / task["config"]).resolve())
        else:
            settings = task.get("settings", {})
            if not isinstance(settings, dict):
                raise ValueError("Native task settings must be an object")
            if settings.get("expected_count") != task["full_count"] or settings.get("subset_count") != task["subset_count"]:
                raise ValueError("Native task settings must match the registered subset/full counts")
            settings["id"] = name
            path_fields = ["checkout", "subset_ids_file", "native_config_file", "inventory_file", "artifacts_dir",
                           "task_config_file", "metadata_file", "dataset_path", "results_dir", "native_results_path",
                           "native_run_manifest_file", "vm_path", "adb_path"]
            if task["kind"] == "scenario" and settings.get("tasks_path"):
                path_fields.append("tasks_path")
            for key in path_fields:
                if settings.get(key):
                    settings[key] = str((path.parent / settings[key]).resolve())
            for key in ("executable", "soffice"):
                if settings.get(key) and ("/" in settings[key] or "\\" in settings[key]):
                    settings[key] = str((path.parent / settings[key]).resolve())
            task["settings"] = settings
    config["_path"] = str(path)
    return config


def catalog(config: dict) -> dict:
    return {"project": config["project"], "requirements": config.get("requirements"),
            "missing_required_tasks": sorted(set(REQUIRED) - set(config["tasks"])) if config["project"] == "qwen38-general-v1" else [],
            "tasks": {name: {key: value for key, value in task.items() if key not in {"config", "settings"}}
                      for name, task in config["tasks"].items()}}


def _revision(config, override):
    revision = override or os.environ.get(config.get("model_revision_env", "BENCHMARK_REVISION"), "")
    return suite._revision(revision, "model_revision / BENCHMARK_REVISION")


def _bank(name, task, revision):
    config = suite.load_config(task["config"])
    if set(config["benchmarks"]) != {name} or config["dataset"]["expected_counts"][name] != task["full_count"]:
        raise ValueError("Prepared bank does not match the registered task/full count")
    subset = config.get("subset", {})
    count = subset.get("counts", {}).get(name)
    if subset.get("ids_file"):
        ids = json.loads(Path(subset["ids_file"]).read_text(encoding="utf-8"))
        count = len(ids) if isinstance(ids, list) else None
    if count != task["subset_count"]:
        raise ValueError("Prepared subset count differs from the capability matrix")
    config["model"]["revision"] = revision
    if any(item["scorer"]["type"] == "llm_judge" for item in config["benchmarks"].values()):
        raise ValueError("The initial Feishu core/scenario protocol does not allow an extra LLM judge")
    return config


def _error(error):
    message = str(error)
    for key, value in os.environ.items():
        if value and any(part in key.upper() for part in ("TOKEN", "KEY", "SECRET", "PASSWORD")):
            message = message.replace(value, "[REDACTED]")
    return {"complete": False, "status": "blocked", "error_type": type(error).__name__, "message": message}


def execute_project(config: dict, command: str, split: str, output: str | Path, *, profile: str,
                    tasks: list[str] | None = None, model_revision: str | None = None,
                    backend: str = "api", score_tag: str = "default") -> dict:
    if command not in {"plan", "doctor", "run", "score"} or split not in {"subset", "full"}:
        raise ValueError("Unknown project operation/split")
    if profile not in {"core", "scenarios", "all"}:
        raise ValueError("Choose core, scenarios or all explicitly")
    available = [name for name, task in config["tasks"].items() if profile == "all" or task["profile"] == profile]
    selected = tasks or available
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(available):
        raise ValueError("Task selection must be unique names within the requested profile")
    revision = _revision(config, model_revision)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    binding = {"project": config["project"], "requirements": config.get("requirements"), "split": split,
               "profile": profile, "selected_tasks": selected, "model_revision": revision,
               "tasks": {name: config["tasks"][name] for name in selected}}
    with suite.output_lock(output):
        suite.bind(output / "project_manifest.json", binding)
    report = {"project": config["project"], "stage": command, "split": split, "profile": profile,
              "model_revision": revision, "selected_tasks": selected,
              "omitted_profile_tasks": sorted(set(available) - set(selected)), "tasks": {},
              "is_mock": backend == "mock", "complete": False, "all_required_complete": False}
    required = {name for name, (_, group) in REQUIRED.items() if profile == "all" or group == profile}
    missing_required = sorted(required - set(config["tasks"])) if config["project"] == "qwen38-general-v1" else []
    report["missing_required_tasks"] = missing_required
    if command == "run":
        readiness = execute_project(config, "doctor", split, output, profile=profile, tasks=tasks,
                                    model_revision=revision, backend=backend, score_tag=score_tag)
        if not readiness["complete"]:
            report.update(tasks=readiness["tasks"], status="preflight_failed", profile_coverage_complete=False)
            suite.write_json(output / "project_run.json", report)
            return report
    for name in selected:
        task = config["tasks"][name]
        target = output / name
        try:
            if task["kind"] == "bank":
                bank = _bank(name, task, revision)
                if command in {"plan", "doctor"}:
                    result = suite.plan(bank, split, target)
                    result = {"complete": True, "status": "planned", "selection": result}
                    if command == "doctor":
                        result["generation"] = suite.preflight_generation(bank, backend=backend)
                        result["graders"] = suite.preflight_scorers(bank)
                elif command == "run":
                    suite.preflight_scorers(bank)
                    result = suite.run(bank, split, target, backend)
                else:
                    result = suite.score(bank, split, target, score_tag)
            elif task["kind"] == "harness":
                from .harnesses import prepare_harness, run_harness, score_harness, preflight_harness
                settings = deepcopy(task["settings"])
                settings["model_revision"] = revision
                if backend == "mock" and command == "run":
                    raise ValueError("Native environment tasks cannot use synthetic mock answers")
                if command in {"plan", "doctor"}:
                    result = {"complete": True, "status": "planned", "selection": prepare_harness(settings, split, target)}
                    if command == "doctor":
                        result["runtime"] = preflight_harness(settings, split, target)
                        result["complete"] = result["runtime"].get("ready") is True
                        result["status"] = "ready" if result["complete"] else "blocked"
                elif command == "run":
                    result = run_harness(settings, split, target)
                else:
                    result = score_harness(settings, split, target)
            else:
                from .scenarios import prepare_scenario, preflight_scenario, run_scenario, score_scenario
                settings = deepcopy(task["settings"])
                settings["model_revision"] = revision
                if backend == "mock" and command == "run":
                    raise ValueError("Interactive scenario tasks cannot use synthetic mock answers")
                if command in {"plan", "doctor"}:
                    native = (preflight_scenario if command == "doctor" else prepare_scenario)(settings, split, target)
                    result = {"complete": native.get("ready", False), "status": "planned", "selection": native}
                elif command == "run":
                    result = run_scenario(settings, split, target)
                else:
                    result = score_scenario(settings, split, target)
            # Native runners may expose a more detailed status. A result must
            # explicitly certify completion; lack of a boolean is never success.
            result.setdefault("complete", False)
            report["tasks"][name] = {"capability": task["capability"],
                                     "expected_questions": task["subset_count" if split == "subset" else "full_count"], **result}
        except (ValueError, RuntimeError, OSError, ImportError, KeyError, TypeError) as error:
            report["tasks"][name] = {"capability": task["capability"], **_error(error)}
    report["complete"] = not missing_required and all(item.get("complete") is True for item in report["tasks"].values())
    report["profile_coverage_complete"] = report["complete"] and not report["omitted_profile_tasks"]
    # score does not select a generation backend. Preserve the provenance of
    # cached predictions even when the caller leaves --backend at its default.
    report["is_mock"] = (backend == "mock" or any(item.get("is_mock") is True for item in report["tasks"].values()))
    report["all_required_complete"] = (command == "score" and split == "full" and profile == "all"
                                       and config["project"] == "qwen38-general-v1" and not missing_required
                                       and report["profile_coverage_complete"] and not report["is_mock"])
    # Different capability metrics are kept separate; no misleading pooled score.
    suite.write_json(output / f"project_{command}.json", report)
    return report
