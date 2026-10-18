"""Convert other task formats into Task dicts."""
from __future__ import annotations


def math2code_to_task(d: dict) -> dict:
    """composite-data/math2code-pilot/problems.jsonl -> Task. These rows carry
    tests with expected outputs but no reference code string."""
    return {
        "id": d["id"], "source": d.get("source", "math2code-pilot"), "domain": "code",
        "prompt": d["statement"], "created_at": d.get("created_at"),
        "verifier": {"type": "python_function", "function": d.get("function", "solve"),
                     "tests": d["tests"], "time_limit_sec": d.get("time_limit_sec", 2.0)},
        "meta": {"family": d.get("family")},
    }
