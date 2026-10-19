"""Step 9: build and validate stored trajectory records.

Student and teacher per-token log-probs are both stored on every record:
  student rollout -> student_logprobs from sampling, teacher_logprobs scored
  teacher rollout -> teacher_logprobs from sampling, student_logprobs scored
so the same batch can be reused for RFT, offline RL and OPD.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from .backends.base import LogprobScorer
from .schema import REQUIRED_TRAJECTORY_FIELDS, Rollout, Task, Trajectory, VerifyResult


def make_trajectory(task: Task, ro: Rollout, vr: VerifyResult, method: str, *, role: str,
                    student_model: Optional[str], teacher_model: Optional[str],
                    other_scorer: Optional[LogprobScorer], uses: list[str], run_id: str,
                    extra: Optional[dict] = None) -> Trajectory:
    """role: 'student' if the student sampled `ro`, 'teacher' if the teacher did."""
    other = other_scorer.score(task, ro) if other_scorer is not None else None
    s_lp, t_lp = (ro.logprobs, other) if role == "student" else (other, ro.logprobs)
    tid = hashlib.sha256(f"{run_id}|{task.id}|{method}|{ro.model}|{ro.sampling}|{ro.text[:500]}".encode()).hexdigest()[:20]
    return Trajectory(
        traj_id=tid, task_id=task.id, source=task.source, domain=task.domain,
        generation_method=method, model=ro.model or "", harness=ro.harness, sampling=ro.sampling,
        reasoning_level=ro.reasoning_level, length_tokens=len(ro.tokens), truncated=ro.truncated,
        reward=vr.reward, passed=vr.passed, verifier_version=vr.verifier_version,
        student_model=student_model, teacher_model=teacher_model,
        student_logprobs=s_lp, teacher_logprobs=t_lp, tokens=ro.tokens,
        tool_calls=ro.tool_calls, env_outputs=ro.env_outputs, messages=ro.messages, text=ro.text,
        tools=ro.tools,
        difficulty_tier=(task.difficulty or {}).get("tier"), uses=uses, pipeline_run=run_id,
        extra={"verify": vr.details, **(extra or {})})


def validate(tr: Trajectory) -> list[str]:
    """Missing step-9 fields. `None` log-probs count as missing."""
    d = tr.__dict__
    return [f for f in REQUIRED_TRAJECTORY_FIELDS if f not in d or d[f] is None
            and f not in ("reasoning_level", "harness")]
