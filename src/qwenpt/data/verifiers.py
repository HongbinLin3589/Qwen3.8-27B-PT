"""Verifiers turn (task, solution) into pass/fail + reward, and stamp every
result with a verifier version so each stored trajectory records which
verifier graded it (step 4)."""
from __future__ import annotations

import hashlib
import re
import json
import os
import subprocess
import sys
import tempfile
from typing import Optional

from .backends.base import Container, Sandbox
from .schema import Solution, Task, VerifyResult

VERIFIER_CODE_VERSION = "2"

_RUNNER = r'''
import json, sys, time
sys.setrecursionlimit(100000)
spec = json.load(open(sys.argv[1]))
ns = {}
if spec.get("prelude"):
    exec(spec["prelude"], ns)
exec(compile(open(sys.argv[2]).read(), "solution.py", "exec"), ns)
name = spec["function"]
f = ns[name] if name in ns or "Solution" not in ns else getattr(ns["Solution"](), name)


def norm(x):  # tuples -> lists, as the expected outputs come from JSON
    try:
        return json.loads(json.dumps(x))
    except (TypeError, ValueError):
        return x


res = []
for t in spec["tests"]:
    try:
        out = norm(f(*t["input"]))
        ok = out == t["output"] or ("alt_output" in t and out == t["alt_output"])
        res.append({"ok": ok, "got": repr(out)[:200]})
    except BaseException as e:
        res.append({"ok": False, "err": type(e).__name__ + ": " + str(e)[:200]})
print("\n__RESULT__" + json.dumps(res))
'''


_ASSERT_RUNNER = r'''
import json, sys
sys.setrecursionlimit(100000)
spec = json.load(open(sys.argv[1]))
ns = {}
if spec.get("prelude"):
    exec(spec["prelude"], ns)
exec(compile(open(sys.argv[2]).read(), "solution.py", "exec"), ns)
if spec.get("setup"):
    exec(spec["setup"], ns)
res = []
for a in spec["tests"]:
    try:
        exec(a, ns)
        res.append({"ok": True})
    except BaseException as e:
        res.append({"ok": False, "err": (type(e).__name__ + ": " + str(e))[:200], "test": a[:200]})
print("\n__RESULT__" + json.dumps(res))
'''


def _test_files_digest(repo_dir: Optional[str]) -> str:
    if not repo_dir or not os.path.isdir(repo_dir):
        return ""
    h = hashlib.sha256()
    for d, _, files in sorted(os.walk(repo_dir)):
        for f in sorted(files):
            p = os.path.join(d, f)
            rel = os.path.relpath(p, repo_dir)
            if "test" in rel.lower():
                h.update(rel.encode())
                with open(p, "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()


def verifier_version(task: Task) -> str:
    """Content hash of everything that decides pass/fail. Changing a test,
    the test command or the verifier code changes the version."""
    if task.verifier.version:
        return task.verifier.version
    v = task.verifier
    payload = json.dumps({"type": v.type, "function": v.function, "tests": v.tests,
                          "tl": v.time_limit_sec, "cmd": task.env.test_cmd, "answer": v.answer,
                          "code": VERIFIER_CODE_VERSION, "prelude": task.meta.get("code_prelude"),
                          "setup": task.meta.get("test_setup_code"),
                          "files": _test_files_digest(task.env.repo_dir)},
                         sort_keys=True, ensure_ascii=False, default=str)
    task.verifier.version = f"{v.type}@{hashlib.sha256(payload.encode()).hexdigest()[:12]}"
    return task.verifier.version


def verify_python_function(task: Task, solution: Optional[Solution]) -> VerifyResult:
    """`python_function`: tests are {"input", "output"} pairs.
    `python_asserts`: tests are assert statements (MBPP / HumanEval style);
    task.meta["test_setup_code"] runs before them."""
    ver = verifier_version(task)
    code = (solution.code if solution else None) or ""
    tests = task.verifier.tests
    if not code.strip():
        return VerifyResult(False, 0.0, ver, {"error": "empty solution"})
    asserts = task.verifier.type == "python_asserts"
    with tempfile.TemporaryDirectory() as d:
        sp, cp = os.path.join(d, "spec.json"), os.path.join(d, "solution.py")
        with open(sp, "w") as f:
            json.dump({"function": task.verifier.function or "solve", "tests": tests,
                       "setup": task.meta.get("test_setup_code"),
                       "prelude": task.meta.get("code_prelude")}, f)
        with open(cp, "w") as f:
            f.write(code)
        limit = task.verifier.time_limit_sec * max(1, len(tests)) + 2
        with open(os.path.join(d, "runner.py"), "w") as f:
            f.write(_ASSERT_RUNNER if asserts else _RUNNER)
        try:
            r = subprocess.run([sys.executable, "-I", os.path.join(d, "runner.py"), sp, cp],
                               cwd=d, capture_output=True, text=True, timeout=limit)
        except subprocess.TimeoutExpired:
            return VerifyResult(False, 0.0, ver, {"error": "timeout"})
    if "__RESULT__" not in r.stdout:
        return VerifyResult(False, 0.0, ver, {"error": r.stderr[-500:]})
    res = json.loads(r.stdout.split("__RESULT__", 1)[1])
    n_ok = sum(x["ok"] for x in res)
    passed = n_ok == len(res) and len(res) > 0
    return VerifyResult(passed, 1.0 if passed else 0.0, ver,
                        {"n_pass": n_ok, "n_tests": len(res),
                         "first_fail": next((x for x in res if not x["ok"]), None)})


def verify_stdio(task: Task, solution: Optional[Solution]) -> VerifyResult:
    """`stdio`: tests are {"input", "output"} strings; the solution is a whole
    program run once per test with `input` on stdin. Outputs are compared
    token by token (whitespace-insensitive), then as floats with 1e-6 tolerance.
    Stops at the first failing test."""
    ver = verifier_version(task)
    code = (solution.code if solution else None) or ""
    if not code.strip():
        return VerifyResult(False, 0.0, ver, {"error": "empty solution"})
    tests = task.verifier.tests
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "solution.py")
        with open(cp, "w") as f:
            f.write(code)
        for i, t in enumerate(tests):
            try:
                r = subprocess.run([sys.executable, "-I", cp], input=t["input"], cwd=d, capture_output=True,
                                   text=True, timeout=task.verifier.time_limit_sec)
            except subprocess.TimeoutExpired:
                return VerifyResult(False, 0.0, ver, {"n_pass": i, "n_tests": len(tests), "error": "timeout"})
            if r.returncode != 0 or not _same_output(r.stdout, t["output"]):
                return VerifyResult(False, 0.0, ver, {"n_pass": i, "n_tests": len(tests),
                                                      "got": r.stdout[-200:], "stderr": r.stderr[-300:]})
    return VerifyResult(True, 1.0, ver, {"n_pass": len(tests), "n_tests": len(tests)})


def _same_output(got: str, want: str) -> bool:
    a, b = got.split(), str(want).split()
    if a == b:
        return True
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        try:
            if abs(float(x) - float(y)) > 1e-6 * max(1.0, abs(float(y))):
                return False
        except ValueError:
            return False
    return True


def verify_command(task: Task, solution: Optional[Solution], sandbox: Sandbox,
                   container: Optional[Container] = None) -> VerifyResult:
    ver = verifier_version(task)
    own = container is None
    c = sandbox.create(task) if own else container
    try:
        if own:
            c.apply(solution)
        code, out, err = c.run(task.env.test_cmd, task.env.timeout_sec)
    finally:
        if own:
            c.destroy()
    passed = code == 0
    return VerifyResult(passed, 1.0 if passed else 0.0, ver,
                        {"exit_code": code, "stdout_tail": out[-800:], "stderr_tail": err[-800:]})


# ---------------------------------------------------------------- answer match
_BOXED = re.compile(r"\\boxed\{")
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def _last_boxed(text: str) -> Optional[str]:
    i = None
    for m in _BOXED.finditer(text):
        i = m.end()
    if i is None:
        return None
    depth, j = 1, i
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i:j - 1] if depth == 0 else None


def extract_answer(text: str) -> Optional[str]:
    """\\boxed{...} if present, else text after '####' (GSM8K), else the last number."""
    if not text:
        return None
    b = _last_boxed(text)
    if b is not None:
        return b
    if "####" in text:
        return text.rsplit("####", 1)[1].strip().splitlines()[0] if text.rsplit("####", 1)[1].strip() else None
    nums = _NUM.findall(text)
    return nums[-1] if nums else None


def normalize_answer(a: str) -> str:
    a = a.strip().strip("$").replace(" ", "").replace("\\!", "").replace("\\,", "")
    a = re.sub(r"\\(?:text|mathrm)\{([^}]*)\}", r"\1", a)
    a = a.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").rstrip(".")
    if re.fullmatch(r"-?[\d,]+(\.\d+)?", a):
        a = a.replace(",", "")
        try:
            f = float(a)
            return str(int(f)) if f == int(f) else str(f)
        except ValueError:
            pass
    return a


# ---------------------------------------------------------------- registry
# name -> (function, needs_sandbox). Add a verifier with @register_verifier.
VERIFIERS: dict = {}


def register_verifier(name: str, needs_sandbox: bool = False):
    def deco(fn):
        VERIFIERS[name] = (fn, needs_sandbox)
        return fn
    return deco


@register_verifier("python_function")
@register_verifier("python_asserts")
def _v_python(task, solution, sandbox=None, container=None):
    return verify_python_function(task, solution)


@register_verifier("stdio")
def _v_stdio(task, solution, sandbox=None, container=None):
    return verify_stdio(task, solution)


@register_verifier("command", needs_sandbox=True)
def _v_command(task, solution, sandbox=None, container=None):
    return verify_command(task, solution, sandbox, container)


@register_verifier("answer_match")
def _v_answer(task, solution, sandbox=None, container=None):
    """Math-style tasks: compare the final answer in solution.text with
    verifier.answer (or reference.text)."""
    ver = verifier_version(task)
    gold = task.verifier.answer if task.verifier.answer is not None else (task.reference.text if task.reference else None)
    pred = extract_answer((solution.text or solution.code or "") if solution else "")
    gold_n = None
    if gold is not None:
        g = str(gold)
        g = _last_boxed(g) or (g.rsplit("####", 1)[1].strip() if "####" in g else g)
        gold_n = normalize_answer(g)
    ok = pred is not None and gold_n is not None and normalize_answer(pred) == gold_n
    return VerifyResult(ok, 1.0 if ok else 0.0, ver, {"pred": pred, "gold": gold_n})


def verify(task: Task, solution: Optional[Solution], sandbox: Optional[Sandbox] = None,
           container: Optional[Container] = None) -> VerifyResult:
    try:
        fn, needs_sandbox = VERIFIERS[task.verifier.type]
    except KeyError:
        raise ValueError(f"unknown verifier type {task.verifier.type!r}; known: {sorted(VERIFIERS)}")
    if needs_sandbox and sandbox is None:
        raise ValueError(f"{task.verifier.type} verifier needs a sandbox")
    return fn(task, solution, sandbox, container)
