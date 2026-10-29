"""Frozen task selection and native Harbor/Pier execution.

Terminal-Bench uses Harbor; DeepSWE 1.1 requires Pier > 0.3.0 for its
separate verifier environment. We generate their native JobConfig JSON with
explicit local TaskConfig paths, then import native per-trial result.json.
No benchmark instruction is flattened into a single chat prompt.

Schemas inspected: harbor 0.25.0 and datacurve-pier 0.3.1. See their official
src/{harbor,pier}/models/{job,trial}/config.py and models/trial/result.py.
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
import shutil
import subprocess
import tomllib
from urllib.parse import urlsplit


SOURCES = {
    "terminal_bench_2_1": "https://www.tbench.ai/news/terminal-bench-2-1",
    "deepswe_1_1": "https://github.com/datacurve-ai/deep-swe",
    "harbor": "https://github.com/harbor-framework/harbor",
    "pier": "https://github.com/datacurve-ai/pier",
}
# These need an interactive agent/environment integration, not a chat scorer.
# Expose honest requirements for the project catalog rather than claiming that
# an arbitrary shell command or result JSON makes an official adapter complete.
SCENARIO_REQUIREMENTS = {
    "webarena_verified": {
        "source": "https://github.com/ServiceNow/webarena-verified",
        "requires": "WebArena sites, browser agent, authenticated site state and official evaluator",
    },
    "spreadsheetbench_verified": {
        "source": "https://github.com/RUCKBReasoning/SpreadsheetBench",
        "requires": "Spreadsheet files, editing agent, recalculation backend and official evaluator",
    },
    "osworld_verified": {
        "source": "https://github.com/xlang-ai/OSWorld",
        "requires": "VM snapshots, screenshot/control agent and application-specific evaluators",
    },
    "androidworld": {
        "source": "https://github.com/google-research/android_world",
        "requires": "Android emulator, device control agent and official state checks",
    },
}
_MODEL_FAILURES = {
    "AgentTimeoutError", "ContextWindowExceededError", "OutputTokenExceededError",
    "ContextLengthExceededError", "OutputLengthExceededError", "AgentSafetyRefusalError",
    "ContentPolicyViolationError",
}
_FIELDS = {
    "id", "engine", "checkout", "revision", "tasks_path", "expected_count", "subset_count",
    "seed", "subset_ids_file", "executable", "engine_version", "agent", "agent_version",
    "model_env", "base_url_env", "api_key_env", "model_revision", "attempts_full",
    "attempts_subset", "concurrency", "environment", "timeout_multiplier", "reward_key", "stratify_by",
    "api_key_required",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _frozen(value, name):
    if (not isinstance(value, str) or not value.strip() or value.lower() in {"main", "master", "latest"}
            or value.upper().startswith(("FILL", "REPLACE", "TODO", "YOUR_"))):
        raise ValueError(f"{name} must be a frozen version, not a placeholder")
    return value


def _git(checkout, *args):
    result = subprocess.run(["git", "-C", str(checkout), *args], capture_output=True,
                            text=True, encoding="utf-8", check=False)
    if result.returncode:
        raise ValueError("Harness dataset must be a readable local Git checkout")
    return result.stdout.strip()


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _same_or_write(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Frozen {path.name} differs; use a new output directory")
    else:
        _json(path, value)


def prepare_harness(task: dict, split: str, output: str | Path) -> dict:
    """Freeze the complete local corpus and selected IDs without executing it.

    ``task`` keys: id, engine, checkout, revision (full Git SHA), engine_version,
    expected_count, subset_count, model_revision. Optional keys are documented
    in ``_FIELDS``. Each task is a direct child of checkout/tasks_path. Selection
    uses a frozen explicit ID file, or SHA256(seed, benchmark, task ID).
    """
    if not isinstance(task, dict) or set(task) - _FIELDS:
        raise ValueError("Unknown harness configuration field(s)")
    if split not in {"subset", "full"}:
        raise ValueError("Harness split must be subset or full")
    identifier = task.get("id")
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier):
        raise ValueError("Harness id must be a simple identifier")
    engine = task.get("engine")
    if engine not in {"harbor", "pier"}:
        raise ValueError("Supported execution engines are harbor and pier")
    version = _frozen(task.get("engine_version"), "engine_version")
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("engine_version must be an exact installed package version")
    if engine == "pier" and tuple(map(int, version.split(".")[:3])) <= (0, 3, 0):
        raise ValueError("DeepSWE 1.1 requires datacurve-pier newer than 0.3.0")
    revision = task.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Harness dataset revision must be a full lowercase Git commit SHA")
    _frozen(task.get("model_revision"), "model_revision")
    if not isinstance(task.get("checkout"), str) or not task["checkout"].strip():
        raise ValueError("Harness checkout is required")
    checkout = Path(task["checkout"]).expanduser().resolve()
    if Path(_git(checkout, "rev-parse", "--show-toplevel")).resolve() != checkout:
        raise ValueError("Harness checkout must be the repository root")
    if _git(checkout, "rev-parse", "HEAD") != revision:
        raise ValueError("Harness dataset HEAD does not match the frozen revision")
    relative = Path(task.get("tasks_path", "tasks"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("tasks_path must stay within the checkout")
    tasks_path = (checkout / relative).resolve()
    if not tasks_path.is_relative_to(checkout) or not tasks_path.is_dir():
        raise ValueError("Harness tasks_path must be an existing directory in the checkout")
    if _git(checkout, "status", "--porcelain", "--untracked-files=all", "--", relative.as_posix()):
        raise ValueError("Harness task checkout has changes; freeze a clean source revision")
    entries = _git(checkout, "ls-files", "-v", "--", relative.as_posix()).splitlines()
    if any(not line.startswith("H ") for line in entries):
        raise ValueError("Harness tasks require a complete checkout without sparse or assume-unchanged entries")
    # Enumerate only native task directories. Glob filters and first-N are never
    # used when dispatching, so metacharacters cannot silently widen a subset.
    task_paths = {p.name: p for p in tasks_path.iterdir()
                  if p.is_dir() and (p / "task.toml").is_file()}
    if any(p.is_symlink() or not p.resolve().is_relative_to(checkout) for p in task_paths.values()):
        raise ValueError("Native task paths must not escape the frozen checkout")
    if len(task_paths) != _positive(task.get("expected_count"), "expected_count"):
        raise ValueError("Harness corpus count does not match expected_count")
    if any(not (p / "instruction.md").is_file() or not (p / "tests").is_dir()
           for p in task_paths.values()):
        raise ValueError("Each native task must contain instruction.md and tests/")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) for name in task_paths):
        raise ValueError("Native task names must be simple identifiers")
    task_info = {name: tomllib.loads((path / "task.toml").read_text(encoding="utf-8"))
                 for name, path in task_paths.items()}
    task_names = {name: info.get("task", {}).get("name", name) for name, info in task_info.items()}
    if len(set(task_names.values())) != len(task_names):
        raise ValueError("Native task.toml names must be unique")
    stratify_by = task.get("stratify_by", "language" if engine == "pier" else "category")
    if not isinstance(stratify_by, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", stratify_by):
        raise ValueError("stratify_by must name a task metadata field")
    strata = {name: str(info.get("metadata", {}).get(stratify_by, "unspecified"))
              for name, info in task_info.items()}
    seed = task.get("seed", 20261017)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    selected = sorted(task_paths)
    if split == "subset":
        count = _positive(task.get("subset_count"), "subset_count")
        if count > len(selected):
            raise ValueError("subset_count exceeds the frozen corpus")
        if task.get("subset_ids_file"):
            selected = json.loads(Path(task["subset_ids_file"]).read_text(encoding="utf-8-sig"))
            if (not isinstance(selected, list) or any(not isinstance(x, str) for x in selected)
                    or len(selected) != count or len(set(selected)) != len(selected)
                    or not set(selected) <= task_paths.keys()):
                raise ValueError("subset_ids_file must contain exactly subset_count unique known task IDs")
            selected = sorted(selected)
        else:
            groups = defaultdict(list)
            for name in selected:
                groups[strata[name]].append(name)
            for group in groups.values():
                group.sort(key=lambda name: _digest([seed, identifier, name]))
            group_order = sorted(groups, key=lambda label: _digest([seed, identifier, label]))
            selected = []
            while len(selected) < count:
                for label in group_order:
                    if groups[label] and len(selected) < count:
                        selected.append(groups[label].pop(0))
            selected.sort()
    defaults = {"harbor": "terminus-2", "pier": "mini-swe-agent"}
    agent = task.get("agent", defaults[engine])
    if agent != defaults[engine]:
        raise ValueError(f"This adapter supports {defaults[engine]} for {engine}; another agent needs its own protocol")
    attempts = _positive(task.get("attempts_" + split, 5 if split == "full" and engine == "harbor" else 1),
                         "attempts_" + split)
    environment = task.get("environment", "docker")
    if environment not in {"docker", "modal", "daytona"}:
        raise ValueError("Unsupported harness environment")
    multiplier = task.get("timeout_multiplier", 1.0)
    if type(multiplier) not in {int, float} or not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("timeout_multiplier must be finite and positive")
    env_names = {key: task.get(key, default) for key, default in (
        ("model_env", "BENCHMARK_MODEL"), ("base_url_env", "BENCHMARK_BASE_URL"),
        ("api_key_env", "BENCHMARK_API_KEY"))}
    if any(not isinstance(v, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", v)
           for v in env_names.values()):
        raise ValueError("Endpoint fields must name environment variables")
    if type(task.get("api_key_required", True)) is not bool:
        raise ValueError("api_key_required must be boolean")
    if "agent_version" in task:
        _frozen(task["agent_version"], "agent_version")
    elif engine == "pier":
        raise ValueError("Pier requires agent_version to pin the separately installed mini-swe-agent")
    config = dict(task)
    config.update(checkout=str(checkout), tasks_path=str(relative), engine_version=version,
                  agent=agent, environment=environment, concurrency=_positive(task.get("concurrency", 1), "concurrency"),
                  timeout_multiplier=multiplier, **env_names)
    manifest = {"version": 1, "benchmark": identifier, "split": split,
                "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "model_api_format": "openai_chat_completions",
                "engine": engine, "engine_version": version, "agent": agent,
                "dataset_revision": revision, "model_revision": task["model_revision"],
                "dataset_count": len(task_paths), "selected_count": len(selected),
                "task_ids": selected, "native_task_names": {name: task_names[name] for name in selected},
                "stratify_by": stratify_by, "strata": {name: strata[name] for name in selected},
                "attempts_per_task": attempts, "seed": seed,
                "config": config, "source_tree": _git(checkout, "rev-parse", f"HEAD:{relative.as_posix()}"),
                "task_paths": {name: str(task_paths[name]) for name in selected}}
    manifest["fingerprint"] = _digest(manifest)
    _same_or_write(Path(output) / "harness_manifest.json", manifest)
    return manifest


def preflight_harness(task: dict, split: str, output: str | Path, *, check_environment: bool = True) -> dict:
    """Check source, endpoint configuration, runner and Docker without model calls."""
    manifest = prepare_harness(task, split, output)
    checks = [{"component": "dataset", "ready": True, "revision": manifest["dataset_revision"]}]
    executable = task.get("executable", manifest["engine"])
    if not isinstance(executable, str) or not shutil.which(executable):
        checks.append({"component": "runner", "ready": False, "reason": "executable_missing"})
    else:
        try:
            result = subprocess.run([executable, "--version"], capture_output=True, text=True,
                                    check=False, timeout=30)
            checks.append({"component": "runner", "ready": result.returncode == 0 and result.stdout.strip() == manifest["engine_version"],
                           "expected_version": manifest["engine_version"],
                           "installed_version": result.stdout.strip() if result.returncode == 0 else None})
        except (OSError, subprocess.TimeoutExpired):
            checks.append({"component": "runner", "ready": False, "reason": "version_check_failed"})
    if check_environment:
        if manifest["config"]["environment"] == "docker":
            ready = False
            if shutil.which("docker"):
                try:
                    result = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                                            capture_output=True, text=True, check=False, timeout=30)
                    ready = result.returncode == 0 and bool(result.stdout.strip())
                except (OSError, subprocess.TimeoutExpired):
                    pass
            checks.append({"component": "environment", "ready": ready, "provider": "docker",
                           "reason": None if ready else "docker_daemon_unavailable"})
        else:
            checks.append({"component": "environment", "ready": False,
                           "provider": manifest["config"]["environment"], "reason": "cloud_credentials_and_capacity_require_separate_validation"})
    try:
        _runtime(manifest)
        checks.append({"component": "endpoint_configuration", "ready": True})
    except (ValueError, TypeError):
        checks.append({"component": "endpoint_configuration", "ready": False,
                       "reason": "model_base_url_or_key_configuration_invalid"})
    return {"benchmark": manifest["benchmark"], "split": split, "ready": all(item["ready"] for item in checks),
            "checks": checks, "model_api_checked": False, "endpoint_config_checked": True,
            "manifest_fingerprint": manifest["fingerprint"]}


def _runtime(manifest: dict) -> tuple[dict, dict]:
    from .api_backend import resolve_endpoint
    config = manifest["config"]
    endpoint = resolve_endpoint({key: config[key] for key in ("model_env", "base_url_env", "api_key_env")}
                                | {"api_key_required": config.get("api_key_required", True)})
    model, base, key = endpoint["model"], endpoint["base_url"], endpoint["api_key"]
    # The portable client also accepts the full route; native SDKs append it.
    # Normalize the shared setting to an API root before handing it to them.
    if base.endswith("/chat/completions"):
        base = base.removesuffix("/chat/completions")
    if any(c.isspace() for c in model) or model.startswith("-"):
        raise ValueError("Harness model must be a nonempty API model ID without whitespace")
    model = model if model.startswith("openai/") else "openai/" + model
    # Pier mini-swe-agent forwards these host variables into its sandbox. Harbor
    # Terminus receives base_url / model_api_key_name in AgentConfig below.
    # Pier's OpenAI library requires a key string even for unauthenticated local
    # servers. The explicit opt-out uses a nonsecret framework placeholder.
    env = dict(os.environ)
    framework_key = key or "EMPTY"
    env.update(OPENAI_API_KEY=framework_key, OPENAI_API_BASE=base, OPENAI_BASE_URL=base)
    env[config["api_key_env"]] = framework_key
    runtime = {"model": model, "base_url": base, "model_revision": manifest["model_revision"],
               "model_api_format": "openai_chat_completions",
               "authentication": "environment_key" if key else "framework_placeholder",
               "manifest_fingerprint": manifest["fingerprint"]}
    return runtime, env


def native_job_config(manifest: dict, output: str | Path, runtime: dict) -> dict:
    """Build native Harbor/Pier JobConfig, containing no credential value."""
    output = Path(output).resolve()
    config = manifest["config"]
    agent = {"name": manifest["agent"], "model_name": runtime["model"], "kwargs": {}}
    if config.get("agent_version"):
        agent["kwargs"]["version"] = config["agent_version"]
    if manifest["engine"] == "harbor":
        agent.update(base_url=runtime["base_url"], model_api_key_name=config["api_key_env"],
                     model_api_format="openai_chat_completions")
    else:
        # Pier 0.3.1's auto mode chooses the Responses API for openai/* models.
        # vLLM/SGLang services in this interface expose Chat Completions.
        agent["kwargs"]["model_class"] = "litellm"
    job = {"job_name": "frozen", "jobs_dir": str(output / "jobs"),
           "n_attempts": manifest["attempts_per_task"],
           "n_concurrent_trials": config["concurrency"],
           "timeout_multiplier": config["timeout_multiplier"],
           "retry": {"max_retries": 0}, "environment": {"type": config["environment"]},
           "agents": [agent], "verifier": {"disable": False},
           "tasks": [{"path": manifest["task_paths"][name], "source": manifest["benchmark"]}
                     for name in manifest["task_ids"]]}
    return job


@contextmanager
def _lock(output):
    path = output / ".harness.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError("Harness output is locked by another run; inspect before removing a stale lock") from None
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def run_harness(task: dict, split: str, output: str | Path) -> dict:
    """Execute an installed official runner; native verifier performs grading.

    This launches containers/API calls, so callers should invoke it only for a
    requested real run. A partial preexisting native job is never erased or
    silently retried; preserve its logs and use a new output directory for a
    deliberately authorized retry. Native resume is not imported by this version.
    """
    output = Path(output).resolve()
    manifest = prepare_harness(task, split, output)
    runtime, env = _runtime(manifest)
    executable = task.get("executable", manifest["engine"])
    if not isinstance(executable, str) or not executable or not shutil.which(executable):
        package = "datacurve-pier" if manifest["engine"] == "pier" else "harbor"
        raise RuntimeError(f"Install {package}=={manifest['engine_version']} in an isolated eval environment")
    installed = subprocess.run([executable, "--version"], capture_output=True, text=True, check=False)
    if installed.returncode or installed.stdout.strip() != manifest["engine_version"]:
        raise RuntimeError("Installed harness version does not match engine_version")
    with _lock(output):
        _same_or_write(output / "runtime.json", runtime)
        job_config = native_job_config(manifest, output, runtime)
        _same_or_write(output / "native_job.json", job_config)
        job_path = output / "jobs" / "frozen"
        if job_path.exists():
            summary = score_harness(task, split, output)
            if summary["complete"]:
                return summary
            raise RuntimeError("Native job already exists and is incomplete; preserve its logs and use a new output directory for a deliberate retry (native resume is not imported by this version)")
        argv = [executable, "run", "--config", str(output / "native_job.json")]
        _json(output / "invocation.json", {"argv": argv, "manifest_fingerprint": manifest["fingerprint"]})
        _json(output / "exit.json", {"returncode": None, "status": "running",
                                      "manifest_fingerprint": manifest["fingerprint"]})
        result = subprocess.run(argv, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        # Keep console diagnostics without leaking the supplied inference key.
        for name, content in (("stdout.log", result.stdout), ("stderr.log", result.stderr)):
            for variable, secret in env.items():
                if secret and secret != "EMPTY" and any(word in variable.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                    content = content.replace(secret, "[REDACTED]")
            (output / name).write_text(content, encoding="utf-8")
        _json(output / "exit.json", {"returncode": result.returncode, "manifest_fingerprint": manifest["fingerprint"]})
        summary = score_harness(task, split, output)
        if result.returncode:
            summary.update(complete=False, final_mean_score=None, runner_exit_code=result.returncode)
            _json(output / "harness_metrics.json", summary)
        return summary


def score_harness(task: dict, split: str, output: str | Path) -> dict:
    """Import native verifier scores; incomplete/infra trials retain denominator."""
    output = Path(output).resolve()
    manifest = prepare_harness(task, split, output)
    runtime_path = output / "runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else None
    if runtime and runtime.get("manifest_fingerprint") != manifest["fingerprint"]:
        raise ValueError("Runtime receipt does not match harness manifest")
    attempts = manifest["attempts_per_task"]
    counts, records, seen = Counter(), [], set()
    root = output / "jobs" / "frozen"
    for path in sorted(root.glob("*/result.json")):
        if not runtime:
            raise ValueError("Native results require the saved runtime receipt")
        result = json.loads(path.read_text(encoding="utf-8"))
        trial_config = result.get("config", {})
        native_path = trial_config.get("task", {}).get("path")
        by_path = {str(Path(value).resolve()): key for key, value in manifest["task_paths"].items()}
        name = by_path.get(str(Path(native_path).resolve())) if isinstance(native_path, str) else None
        trial_id = result.get("id")
        if name is None:
            raise ValueError("Native result task path does not match frozen selection")
        if (result.get("task_name") != manifest["native_task_names"][name]
                or not isinstance(trial_id, str) or not trial_id or trial_id in seen):
            raise ValueError("Unknown task or duplicate/missing native trial ID")
        seen.add(trial_id)
        counts[name] += 1
        if counts[name] > attempts:
            raise ValueError("Native results exceed frozen attempts; do not mix independent runs")
        agent = trial_config.get("agent", {})
        if agent.get("name") != manifest["agent"] or agent.get("model_name") != runtime["model"]:
            raise ValueError("Native result agent/model does not match runtime receipt")
        exceptions = [result.get("exception_info")]
        exceptions.extend(item.get("exception_info") for item in (result.get("step_results") or []))
        errors = [item.get("exception_type", "UnknownError") for item in exceptions if item]
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        value = rewards.get(task.get("reward_key", "reward"))
        valid = (type(value) in {int, float} and math.isfinite(value) and value in {0, 1}
                 and all(error in _MODEL_FAILURES for error in errors) and bool(result.get("finished_at")))
        status = "passed" if valid and value else "model_failure" if valid else "infra_error"
        records.append({"task_id": name, "native_task_name": result["task_name"], "trial_id": trial_id, "status": status,
                        "score": float(value) if valid else None, "errors": errors,
                        "result_file": str(path.relative_to(output)),
                        "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    expected = len(manifest["task_ids"]) * attempts
    scored = sum(item["score"] is not None for item in records)
    successes = sum(item["score"] or 0 for item in records)
    complete = scored == expected and all(counts[name] == attempts for name in manifest["task_ids"])
    exit_path = output / "exit.json"
    exit_receipt = json.loads(exit_path.read_text(encoding="utf-8")) if exit_path.exists() else None
    if (not exit_receipt or exit_receipt.get("manifest_fingerprint") != manifest["fingerprint"]
            or type(exit_receipt.get("returncode")) is not int or exit_receipt["returncode"] != 0):
        complete = False
    summary = {"benchmark": manifest["benchmark"], "split": split,
               "metric": "native_verifier_avg_success", "complete": complete,
               "expected_tasks": len(manifest["task_ids"]), "expected_attempts": expected,
               "attempts_recorded": len(records), "attempts_scored": scored,
               "infra_errors": sum(item["status"] == "infra_error" for item in records),
               "model_failures": sum(item["status"] == "model_failure" for item in records),
               "missing_attempts": expected - len(records),
               "score_lower_bound": successes / expected,
               "final_mean_score": successes / expected if complete else None,
               "manifest_fingerprint": manifest["fingerprint"], "trials": records}
    _json(output / "harness_metrics.json", summary)
    return summary
