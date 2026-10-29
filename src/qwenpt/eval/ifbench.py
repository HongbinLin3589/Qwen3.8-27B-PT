"""IFBench data and the upstream deterministic strict/loose evaluators.

The evaluator is loaded from a clean, explicitly pinned allenai/IFBench Git
checkout. No instruction checker is reimplemented here and no LLM is used.
Install that checkout's dependencies and NLTK resources before scoring.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
from importlib.metadata import version as package_version
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable


IFBENCH_REPOSITORY = "https://github.com/allenai/IFBench"
IFBENCH_REVISION = "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d"
IFBENCH_DATA_SHA256 = "d2ada7da94a38cfe406351614c4e686846ed2da6d1b339db95fa5ead19554a4a"
NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}


def _constraints(row: dict) -> tuple[list[str], list[dict]]:
    ids, kwargs = row.get("instruction_id_list"), row.get("kwargs")
    if (not isinstance(ids, list) or not ids
            or any(not isinstance(item, str) or not item.strip() for item in ids)):
        raise ValueError("IFBench requires a nonempty instruction_id_list")
    if (not isinstance(kwargs, list) or len(kwargs) != len(ids)
            or any(not isinstance(item, dict) for item in kwargs)):
        raise ValueError("IFBench kwargs must align one-to-one with instruction_id_list")
    try:
        json.dumps(kwargs, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError("IFBench kwargs must contain finite JSON values") from None
    return deepcopy(ids), deepcopy(kwargs)


def normalize_ifbench(rows: Iterable[dict], benchmark: str = "ifbench") -> list[dict]:
    """Normalize official single-turn rows without altering their prompts.

    A reference answer is deliberately absent: correctness is the conjunction
    of the official constraint checkers, not answer matching or a model judge.
    """
    normalized, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("IFBench rows must be objects")
        key, prompt = row.get("key"), row.get("prompt")
        if isinstance(key, bool) or not isinstance(key, (str, int)) or not str(key).strip():
            raise ValueError("IFBench requires the original nonempty key")
        question_id = str(key)
        if question_id in seen:
            raise ValueError(f"Duplicate IFBench key: {question_id}")
        seen.add(question_id)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("IFBench requires a nonempty prompt")
        ids, kwargs = _constraints(row)
        normalized.append({
            "benchmark": benchmark,
            "question_id": question_id,
            "prompt": prompt,
            "metadata": {
                "stratum": "+".join(sorted(set(item.split(":", 1)[0] for item in ids))),
                "ifbench": {"key": key, "instruction_id_list": ids, "kwargs": kwargs},
            },
        })
    if not normalized:
        raise ValueError("IFBench source contains no tasks")
    return normalized


def _git(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments], check=True, capture_output=True,
            text=True, encoding="utf-8", timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        raise ValueError("Cannot inspect IFBench checkout with Git; use a local Git clone") from None
    return result.stdout.strip()


def _validate_checkout(checkout: str | Path, revision: str) -> Path:
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("IFBench scorer revision must be a full immutable Git commit SHA")
    root = Path(checkout).expanduser().resolve()
    if not (root / "evaluation_lib.py").is_file() or not (root / "ifbench" / "__init__.py").is_file():
        raise ValueError("IFBench checkout must contain evaluation_lib.py and the ifbench package")
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("IFBench checkout must be the root of its own Git clone")
    if _git(root, "rev-parse", "HEAD") != revision:
        raise ValueError("IFBench checkout HEAD does not match the configured scorer revision")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("IFBench checkout has modified tracked files; use a clean pinned checkout")
    return root


def prepare_ifbench(checkout: str | Path, revision: str = IFBENCH_REVISION) -> tuple[list[dict], dict]:
    """Read the complete official test file from a pinned clean checkout.

    This performs no model call, package installation, or network download.
    The validated upstream file's count and hash belong in the suite manifest.
    """
    root = _validate_checkout(checkout, revision)
    path = root / "data" / "IFBench_test.jsonl"
    # Match the repository blob across Windows Git's autocrlf checkouts.
    data = path.read_bytes().replace(b"\r\n", b"\n")
    checksum = hashlib.sha256(data).hexdigest()
    rows = normalize_ifbench(json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip())
    if revision == IFBENCH_REVISION and (checksum != IFBENCH_DATA_SHA256 or len(rows) != 300):
        raise ValueError("Pinned IFBench release must match its official 300-task test file hash")
    return rows, {
        "repository": IFBENCH_REPOSITORY, "revision": revision,
        "split": "test", "path": "data/IFBench_test.jsonl",
        "sha256": checksum, "hash_normalization": "LF", "count": len(rows), "license": "ODC-BY-1.0",
    }


def _require_nltk_resources(root: Path) -> None:
    try:
        import nltk
    except ImportError:
        raise RuntimeError("IFBench dependencies are missing; install the pinned checkout with pip") from None
    # Upstream adds this directory on import. Check it before importing upstream
    # so its automatic downloader is never triggered during an evaluation.
    search_paths = [str(root / "ifbench" / ".nltk_data"), *nltk.data.path]
    missing = []
    for name, resource in NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource, paths=search_paths)
        except LookupError:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "IFBench NLTK data missing; prepare it before scoring with: python -m nltk.downloader "
            + " ".join(missing)
        )


@lru_cache(maxsize=4)
def _load_evaluator(checkout: str, revision: str):
    root = _validate_checkout(checkout, revision)
    _require_nltk_resources(root)
    for name, module in list(sys.modules.items()):
        if name == "ifbench" or name.startswith("ifbench."):
            module_file = getattr(module, "__file__", None)
            if not module_file or not Path(module_file).resolve().is_relative_to(root / "ifbench"):
                raise RuntimeError("A different IFBench package is already loaded; use a fresh scoring process")
    module_name = "_qwenpt_official_ifbench_" + revision
    spec = importlib.util.spec_from_file_location(module_name, root / "evaluation_lib.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Needed by upstream dataclasses.
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(module)
    except ImportError:
        sys.modules.pop(module_name, None)
        raise RuntimeError("IFBench dependencies are missing; install the pinned checkout with pip") from None
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.path.remove(str(root))
    return module


def preflight_ifbench(checkout: str | Path, revision: str = IFBENCH_REVISION) -> dict:
    """Validate exact code provenance, dependencies, data, and registry loading."""
    root = _validate_checkout(checkout, revision)
    module = _load_evaluator(str(root), revision)
    _, source = prepare_ifbench(root, revision)
    return {**source, "checker_count": len(module.instructions_registry.INSTRUCTION_DICT),
            "scoring": "official_strict_and_loose", "llm_judge": False,
            "dependencies": {name: package_version(name) for name in
                             ("nltk", "langdetect", "emoji", "syllapy", "immutabledict")}}


def score_ifbench(question: dict, final_text: str, scorer: dict) -> dict:
    """Invoke both upstream graders; loose prompt accuracy is the primary default.

    ``scorer`` requires ``checkout`` and ``revision``; ``mode`` is either
    ``loose`` (paper default) or ``strict``. Extra metrics retain instruction
    numerators and denominators for correct micro-aggregation across prompts.
    """
    if not isinstance(final_text, str):
        raise ValueError("IFBench final_text must be a string")
    mode = scorer.get("mode", "loose")
    if mode not in {"strict", "loose"}:
        raise ValueError("IFBench mode must be strict or loose")
    if not isinstance(scorer.get("checkout"), str) or not scorer["checkout"].strip():
        raise ValueError("IFBench scorer requires a local pinned checkout")
    revision = scorer.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("IFBench scorer requires an immutable 40-character revision")
    metadata = question.get("metadata", {}).get("ifbench")
    if not isinstance(metadata, dict):
        raise ValueError("IFBench row is missing metadata.ifbench constraints")
    ids, kwargs = _constraints(metadata)
    prompt = question.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("IFBench row requires its original prompt")
    module = _load_evaluator(str(Path(scorer["checkout"]).expanduser().resolve()), revision)
    if any(item not in module.instructions_registry.INSTRUCTION_DICT for item in ids):
        raise ValueError("IFBench constraint ID is absent from the pinned official registry")
    # The official CLI evaluates strict before loose on the same InputExample:
    # strict removes null kwargs in place. Preserve that behavior on a copy.
    example = module.InputExample(key=metadata.get("key", question.get("question_id")),
                                  instruction_id_list=ids, prompt=prompt, kwargs=kwargs)
    responses = {prompt: final_text}
    strict = module.test_instruction_following_strict(example, responses)
    loose = module.test_instruction_following_loose(example, responses)
    outcomes = {}
    for name, result in (("strict", strict), ("loose", loose)):
        flags = result.follow_instruction_list
        if (not isinstance(flags, list) or len(flags) != len(ids)
                or any(type(flag) is not bool for flag in flags)
                or type(result.follow_all_instructions) is not bool
                or result.follow_all_instructions != all(flags)):
            raise ValueError("Official IFBench evaluator returned malformed constraint results")
        outcomes[name] = {
            "prompt_accuracy": float(result.follow_all_instructions),
            "instruction_correct": sum(flags), "instruction_total": len(flags),
            "instruction_accuracy": sum(flags) / len(flags),
            "follow_instruction_list": flags,
        }
    correct = bool(outcomes[mode]["prompt_accuracy"])
    return {"score": float(correct), "correct": correct,
            "metric": f"ifbench_prompt_{mode}_accuracy", "predicted": final_text,
            "ifbench": {"revision": revision, "instruction_id_list": ids, **outcomes}}
