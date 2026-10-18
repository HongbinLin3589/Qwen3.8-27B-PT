"""Mappers: turn one raw row into a Task dict (see schema.Task).

  fields     generic, configured entirely in YAML (covers most datasets)
  mbpp       MBPP full / sanitized (code + assert tests)
  humaneval  HumanEval (prompt + canonical solution + check())
  rllm_code  rLLM / DeepCoder code rows {problem, tests, solutions?}: stdin tests,
             APPS call-based tests, or LeetCode check(candidate) code

Options of the `fields` mapper (domain and verifier are literal strings; the
rest are field references): id, prompt, domain, verifier, answer,
answer_after, reference_text (full solution, used for SFT export), system,
reference_code, tests, function, created_at, repo,
prompt_template ("...{prompt}..."), prompt_strip (regex removed from the
prompt, e.g. a shared "think step by step" suffix), prompt_replace ({literal:
replacement}, e.g. a stray "<image>" whose picture the dataset dropped),
meta ({name: field}).
A list-valued answer (verl rows often store ["2"]) uses its first element.

Field references in the `fields` mapper:
  "question"            a column
  "prompt.0.content"    a nested path (dict keys / list indexes)
  "{source}-{index}"    a template; {index} is the row number, other names are columns
  {const: "math"}       a constant

Add a mapper with @register_mapper("name"); it gets (row, index, **options)
and returns a Task dict, or None to skip the row.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Optional

MAPPERS: dict = {}


def register_mapper(name: str):
    def deco(fn):
        MAPPERS[name] = fn
        return fn
    return deco


def get_path(row: dict, path: str) -> Any:
    cur: Any = row
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, (list, tuple)):
            cur = cur[int(part)] if part.lstrip("-").isdigit() and -len(cur) <= int(part) < len(cur) else None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


class _Fmt(dict):
    def __missing__(self, k):
        return "{" + k + "}"


def resolve(row: dict, index: int, ref: Any) -> Any:
    if ref is None:
        return None
    if isinstance(ref, dict) and "const" in ref:
        return ref["const"]
    if isinstance(ref, str) and "{" in ref:
        flat = {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
        return ref.format_map(_Fmt(flat, index=index))
    if isinstance(ref, str):
        return get_path(row, ref)
    return ref


@register_mapper("fields")
def map_fields(row: dict, index: int, *, source: str, id: Any = "{index}", prompt: Any = "prompt",
               domain: Any = "math", verifier: str = "answer_match", answer: Any = None,
               reference_code: Any = None, tests: Any = None, function: Any = None,
               created_at: Any = None, repo: Any = None, prompt_template: Optional[str] = None,
               answer_after: Optional[str] = None, reference_text: Any = None,
               system: Optional[str] = None, prompt_strip: Optional[str] = None,
               prompt_replace: Optional[dict] = None, meta: Optional[dict] = None, **_) -> Optional[dict]:
    p = resolve(row, index, prompt)
    if p is None or (isinstance(p, str) and not p.strip()):
        return None
    if prompt_strip:
        p = re.sub(prompt_strip, "", str(p)).strip()
    p = _replace(str(p), prompt_replace)
    if prompt_template:
        p = prompt_template.format(prompt=p)
    ans = resolve(row, index, answer)
    if isinstance(ans, (list, tuple)):
        ans = ans[0] if ans else None
    if isinstance(ans, str):
        ans = ans.strip()
    if ans is not None and answer_after and answer_after in str(ans):  # e.g. GSM8K "... #### 72"
        ans = str(ans).rsplit(answer_after, 1)[1].strip()
    task = {
        "id": f"{source}-{resolve(row, index, id)}", "source": source,
        "domain": domain if isinstance(domain, str) and "{" not in domain else resolve(row, index, domain),
        "prompt": str(p), "created_at": _iso(resolve(row, index, created_at)), "repo": resolve(row, index, repo),
        "verifier": {"type": verifier},
        "meta": {k: resolve(row, index, v) for k, v in (meta or {}).items()},
    }
    if system:
        task["meta"]["system"] = system
    if verifier == "answer_match":
        if ans is None:
            return None
        task["verifier"]["answer"] = str(ans)
        ref = resolve(row, index, reference_text)  # full worked solution, if the dataset has one
        # Without a worked solution the reference is just the boxed answer: enough
        # for env_check, marked answer_only so it is not exported as SFT.
        task["reference"] = {"text": str(ref)} if ref else {"text": f"\\boxed{{{ans}}}",
                                                            "meta": {"answer_only": True}}
    else:
        code = resolve(row, index, reference_code)
        if code:
            task["reference"] = {"code": code}
        t = resolve(row, index, tests)
        if t is not None:
            task["verifier"]["tests"] = t if isinstance(t, list) else [t]
        task["verifier"]["function"] = resolve(row, index, function)
    return task


def _replace(text: str, pairs: Optional[dict]) -> str:
    for old, new in (pairs or {}).items():
        text = text.replace(old, new)
    return text


def _iso(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v)
    return s[:10] if re.match(r"\d{4}-\d{2}-\d{2}", s) else s


_FN = re.compile(r"assert\s+(?:not\s+)?\(?\s*(?:set\(|math\.isclose\()?\s*([A-Za-z_]\w*)\s*\(")


@register_mapper("mbpp")
def map_mbpp(row: dict, index: int, *, source: str = "mbpp", tests_in_prompt: bool = True, **_) -> Optional[dict]:
    """tests_in_prompt: append the asserts to the prompt, as the MBPP paper does,
    so the model knows the function name and signature."""
    prompt = row.get("text") or row.get("prompt")
    tests = list(row.get("test_list") or [])
    if not prompt or not tests:
        return None
    if tests_in_prompt:
        prompt = prompt + " Your code should pass these tests:\n\n" + "\n".join(tests)
    setup = row.get("test_setup_code") or ""
    imports = row.get("test_imports") or []
    if imports:
        setup = "\n".join(imports) + ("\n" + setup if setup else "")
    m = _FN.search(tests[0])
    return {
        "id": f"{source}-{row.get('task_id', index)}", "source": source, "domain": "code", "prompt": prompt,
        "verifier": {"type": "python_asserts", "function": m.group(1) if m else None, "tests": tests,
                     "time_limit_sec": 5.0},
        "reference": {"code": row.get("code")},
        "meta": {"test_setup_code": setup, "orig_id": row.get("task_id")},
    }


@register_mapper("humaneval")
def map_humaneval(row: dict, index: int, *, source: str = "humaneval", **_) -> Optional[dict]:
    entry = row["entry_point"]
    return {
        "id": f"{source}-{str(row.get('task_id', index)).replace('/', '_')}", "source": source, "domain": "code",
        "prompt": row["prompt"],
        "verifier": {"type": "python_asserts", "function": entry,
                     "tests": [row["test"] + f"\ncheck({entry})\n"], "time_limit_sec": 10.0},
        "reference": {"code": row["prompt"] + row["canonical_solution"]},
        "meta": {"orig_id": row.get("task_id")},
    }


# Imports and LeetCode helper types the reference solutions and check() code assume.
LEETCODE_PRELUDE = """from typing import *
from collections import *
from functools import *
from itertools import *
from heapq import *
from bisect import *
from math import *
from operator import *
from string import *
import math, string, re, random, operator, datetime
inf = float("inf")


class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right


def list_node(values):
    head = cur = ListNode()
    for v in values:
        cur.next = ListNode(v)
        cur = cur.next
    return head.next


def is_same_list(a, b):
    while a and b:
        if a.val != b.val:
            return False
        a, b = a.next, b.next
    return a is None and b is None


def tree_node(values):
    if not values:
        return None
    root = TreeNode(values[0])
    q, i = deque([root]), 1
    while q and i < len(values):
        node = q.popleft()
        for side in ("left", "right"):
            if i < len(values) and values[i] is not None:
                child = TreeNode(values[i])
                setattr(node, side, child)
                q.append(child)
            i += 1
    return root


def is_same_tree(a, b):
    if not a or not b:
        return a is b
    return a.val == b.val and is_same_tree(a.left, b.left) and is_same_tree(a.right, b.right)
"""


def _literal(v):
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except ValueError:
        pass
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return {"functional": v}  # plain test code, e.g. HumanEval+ check()


@register_mapper("rllm_code")
def map_rllm_code(row: dict, index: int, *, source: str, max_tests: Optional[int] = None,
                  time_limit_sec: float = 6.0, prompt_strip: Optional[str] = None,
                  prompt_replace: Optional[dict] = None, **_) -> Optional[dict]:
    """Three test layouts, one verifier each:
      [{"input", "output"}] or {"inputs", "outputs"}  -> stdio (whole program, stdin/stdout)
      {"fn_name", "inputs", "outputs"}                -> python_function (APPS call-based)
      {"functional": "def check(candidate): ..."}     -> python_asserts (LeetCode)
    max_tests keeps the first N tests (None = all)."""
    prompt = row.get("problem")
    if not prompt or not row.get("tests"):
        return None
    if prompt_strip:
        prompt = re.sub(prompt_strip, "", prompt).strip()
    prompt = _replace(prompt, prompt_replace)
    t = _literal(row["tests"])
    sol = row.get("solutions")
    if isinstance(sol, list):
        sol = sol[0] if sol else None
    meta = {}
    if isinstance(t, dict) and "functional" in t:
        vtype, fn, tests = "python_asserts", None, [t["functional"]]
        calls = re.findall(r"^check\((?:Solution\(\)\.)?(\w+)\)", t["functional"], re.M)
        fn = calls[-1] if calls else None
        meta["code_prelude"] = LEETCODE_PRELUDE
    elif isinstance(t, dict) and t.get("fn_name"):
        vtype, fn = "python_function", t["fn_name"]
        tests = []
        for i, o in zip(t["inputs"], t["outputs"]):
            # APPS wraps the return value in a one-element list; accept either form
            case = {"input": list(i) if isinstance(i, (list, tuple)) else [i], "output": o}
            if isinstance(o, list) and len(o) == 1:
                case = {"input": case["input"], "output": o[0], "alt_output": o}
            tests.append(case)
        meta["code_prelude"] = LEETCODE_PRELUDE
    else:
        vtype, fn = "stdio", None
        pairs = zip(t["inputs"], t["outputs"]) if isinstance(t, dict) else ((c["input"], c["output"]) for c in t)
        tests = [{"input": _as_text(i), "output": _as_text(o)} for i, o in pairs]
    if max_tests:
        tests = tests[:max_tests]
    if not tests:
        return None
    task = {"id": f"{source}-{index}", "source": source, "domain": "code", "prompt": prompt,
            "verifier": {"type": vtype, "function": fn, "tests": tests, "time_limit_sec": time_limit_sec},
            "meta": meta}
    if sol:
        task["reference"] = {"code": sol}
    return task


def _as_text(v) -> str:
    if isinstance(v, list):
        return "\n".join(map(str, v))
    return str(v)
