"""verl custom reward function that grades with datapipe's verifiers.

In the verl config:
    custom_reward_function:
      path: <repo>/src/qwenpt/data/export/verl_reward.py
      name: compute_score

`ground_truth` is what export.verl.rl_row wrote: the gold answer string for
answer_match tasks, or a JSON verifier spec for code tasks. Repo tasks
(type=command) need a sandbox and are not graded here; route them to your
agent-RL reward instead.
"""
from __future__ import annotations

import json
import re

from qwenpt.data.schema import Solution, Task, VerifierSpec  # noqa: E402
from qwenpt.data.verifiers import verify  # noqa: E402

_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **_):
    spec = None
    if isinstance(ground_truth, str) and ground_truth.startswith("{"):
        try:
            spec = json.loads(ground_truth)
        except ValueError:
            spec = None
    if spec is None:  # plain gold answer
        task = Task(id="r", source=str(data_source), domain="math", prompt="",
                    verifier=VerifierSpec(type="answer_match", answer=str(ground_truth)))
        return verify(task, Solution(text=solution_str)).reward
    if spec.get("type") == "command":
        raise ValueError("repo tasks need a sandbox reward; not graded by compute_score")
    task = Task(id="r", source=str(data_source), domain="code", prompt="",
                verifier=VerifierSpec(type=spec["type"], function=spec.get("function"), tests=spec.get("tests") or [],
                                      time_limit_sec=spec.get("time_limit_sec") or 5.0),
                meta={"test_setup_code": spec.get("test_setup_code"), "code_prelude": spec.get("code_prelude")})
    codes = _CODE.findall(solution_str or "")
    return verify(task, Solution(code=codes[-1] if codes else solution_str)).reward
