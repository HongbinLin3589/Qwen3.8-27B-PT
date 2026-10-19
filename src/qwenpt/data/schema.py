"""Records that flow through the pipeline.

Task        one problem / environment, before and after filtering
Solution    a candidate answer (code string, or file edits for repo tasks)
Rollout     one model attempt at a task, as returned by a policy backend
Trajectory  what step 9 stores: a rollout plus every field needed to reuse it
            for SFT, RFT, offline RL and OPD
"""
from __future__ import annotations

import dataclasses as dc
import json
import time
from typing import Any, Optional


def _drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


@dc.dataclass
class EnvSpec:
    """How to build the environment a task runs in.

    For repo tasks (SWE / terminal) `repo_dir` is a directory copied into every
    fresh container, `image` is the container image, and `test_cmd` decides
    pass/fail by exit code. Function tasks (math x code) leave these empty.
    """
    image: Optional[str] = None
    repo_dir: Optional[str] = None
    workdir: str = "/workspace"
    setup_cmd: Optional[str] = None
    test_cmd: Optional[str] = None
    timeout_sec: float = 60.0
    # Files the solver is supposed to change (used by the leak check to know
    # what is "the answer" and what is the surrounding repo).
    target_files: list[str] = dc.field(default_factory=list)


@dc.dataclass
class VerifierSpec:
    """`python_function`: call `function(*input)` and compare to `output`.
    `python_asserts`: run assert statements (MBPP / HumanEval style).
    `answer_match`: compare the final answer in the model output with `answer`.
    `command`: run env.test_cmd inside the container, exit code 0 = pass.
    New types: see verifiers.register_verifier."""
    type: str = "python_function"
    function: Optional[str] = None
    tests: list[dict] = dc.field(default_factory=list)
    time_limit_sec: float = 2.0
    answer: Optional[str] = None   # answer_match: gold final answer
    version: Optional[str] = None  # filled in by verifiers.verifier_version


@dc.dataclass
class Solution:
    code: Optional[str] = None                 # python_function tasks
    file_edits: Optional[dict[str, str]] = None  # repo tasks: path -> new content
    text: Optional[str] = None                 # full model output, if any
    meta: dict = dc.field(default_factory=dict)

    @staticmethod
    def from_dict(d: Optional[dict]) -> Optional["Solution"]:
        if d is None:
            return None
        return Solution(**{k: d.get(k) for k in ("code", "file_edits", "text")},
                        meta=d.get("meta") or {})


@dc.dataclass
class Task:
    id: str
    source: str
    domain: str                     # math | code | swe | terminal | ...
    prompt: str
    created_at: Optional[str] = None  # ISO date, used for the time split
    repo: Optional[str] = None        # "owner/name", used for the repo split
    env: EnvSpec = dc.field(default_factory=EnvSpec)
    verifier: VerifierSpec = dc.field(default_factory=VerifierSpec)
    reference: Optional[Solution] = None
    initial: Optional[Solution] = None  # the "unsolved" state; None = no edits
    meta: dict = dc.field(default_factory=dict)
    # Filled in by stages
    difficulty: Optional[dict] = None
    history: list[dict] = dc.field(default_factory=list)

    def note(self, stage: str, **info):
        self.history.append({"stage": stage, **info})

    def to_dict(self) -> dict:
        return dc.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Task":
        d = dict(d)
        d["env"] = EnvSpec(**(d.get("env") or {}))
        d["verifier"] = VerifierSpec(**(d.get("verifier") or {}))
        d["reference"] = Solution.from_dict(d.get("reference"))
        d["initial"] = Solution.from_dict(d.get("initial"))
        known = {f.name for f in dc.fields(Task)}
        extra = {k: d.pop(k) for k in list(d) if k not in known}
        t = Task(**d)
        t.meta.update(extra)
        return t


@dc.dataclass
class Rollout:
    """One attempt, as a policy / teacher backend returns it."""
    text: str
    solution: Optional[Solution]
    tokens: list[Any] = dc.field(default_factory=list)       # token ids or strings
    logprobs: Optional[list[float]] = None                   # of the model that sampled it
    tool_calls: list[dict] = dc.field(default_factory=list)
    env_outputs: list[dict] = dc.field(default_factory=list)
    messages: list[dict] = dc.field(default_factory=list)  # OpenAI format, incl. tool_calls / role=tool
    tools: list[dict] = dc.field(default_factory=list)     # OpenAI function schemas the harness exposed
    truncated: bool = False
    finish_reason: Optional[str] = None
    model: Optional[str] = None
    harness: Optional[str] = None
    sampling: dict = dc.field(default_factory=dict)
    reasoning_level: Optional[str] = None


@dc.dataclass
class VerifyResult:
    passed: bool
    reward: float
    verifier_version: str
    details: dict = dc.field(default_factory=dict)


@dc.dataclass
class Trajectory:
    """Step 9 storage record. One per stored rollout."""
    traj_id: str
    task_id: str
    source: str                       # task source dataset
    domain: str
    generation_method: str            # student_rollout | teacher_distill | teacher_demo
    model: str                        # model that produced the tokens
    harness: Optional[str]
    sampling: dict                    # temperature, top_p, max_tokens, seed, n ...
    reasoning_level: Optional[str]    # 推理档位
    length_tokens: int
    truncated: bool
    reward: float
    passed: bool
    verifier_version: str
    student_model: Optional[str]
    teacher_model: Optional[str]
    student_logprobs: Optional[list[float]]  # per token
    teacher_logprobs: Optional[list[float]]  # per token, same tokens
    tokens: list[Any]
    tool_calls: list[dict]
    env_outputs: list[dict]
    messages: list[dict]
    text: str
    tools: list[dict] = dc.field(default_factory=list)
    difficulty_tier: Optional[str] = None
    uses: list[str] = dc.field(default_factory=list)  # sft | rft | offline_rl | opd
    pipeline_run: Optional[str] = None
    created_at: str = dc.field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    extra: dict = dc.field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(dc.asdict(self), ensure_ascii=False)


REQUIRED_TRAJECTORY_FIELDS = [
    "source", "generation_method", "sampling", "reasoning_level", "length_tokens",
    "truncated", "reward", "verifier_version", "student_logprobs", "teacher_logprobs",
    "tool_calls", "env_outputs", "harness",
]
