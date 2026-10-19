"""Local stand-ins so the whole pipeline runs without GPUs or a cluster.

MockPolicy        pretends to be Qwen3.8-27B (or a teacher). Per task it has a
                  hidden success rate; a correct sample returns a known-good
                  solution, a wrong one returns a known-bad one. Solutions are
                  real and get executed by the real verifiers.
MockScorer        deterministic fake per-token log-probs.
MockReviewAgent   reads the ground-truth label the mock policy attached.
GrepShortcutAgent a real (if simple) baseline, not a mock: searches the
                  environment for files that already contain the reference
                  answer and submits them.
"""
from __future__ import annotations

import hashlib
import os
import random
from typing import Optional

from ..schema import Rollout, Solution, Task
from .base import Container, LogprobScorer, Policy, ReviewAgent, Sandbox, ShortcutAgent


def _rng(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16))


def _solutions(task: Task, key: str) -> list[Solution]:
    return [Solution.from_dict(d) for d in task.meta.get(key) or []]


class MockPolicy(Policy):
    def __init__(self, model: str = "mock-Qwen3.8-27B", harness: str = "mock_harness", skill_boost: float = 0.0,
                 reasoning_level: Optional[str] = "high", seed: int = 0):
        self.model, self.harness, self.skill_boost = model, harness, skill_boost
        self.reasoning_level, self.seed = reasoning_level, seed
        self._calls: dict[str, int] = {}

    def p_correct(self, task: Task) -> float:
        p = task.meta.get("mock_p_correct")
        if p is None:
            p = _rng("p", task.id).choice([0.0, 0.25, 0.5, 0.75, 1.0])
        return min(1.0, p + self.skill_boost)

    def generate(self, task: Task, n: int, sampling: dict, sandbox: Optional[Sandbox] = None) -> list[Rollout]:
        start = self._calls.get(task.id, 0)
        self._calls[task.id] = start + n
        good = _solutions(task, "mock_correct_solutions") or ([task.reference] if task.reference else [])
        bad = _solutions(task, "mock_wrong_solutions") or [self._default_wrong(task)]
        p = self.p_correct(task)
        out = []
        for i in range(start, start + n):
            r = _rng(self.model, self.seed, task.id, i)
            ok = bool(good) and r.random() < p
            sol = r.choice(good if ok else bad)
            sol = Solution(code=sol.code, file_edits=sol.file_edits, text=sol.text,
                           meta={**sol.meta, "mock_truth": ok})
            body = sol.code or sol.text or "\n".join(f"--- {k}\n{v}" for k, v in (sol.file_edits or {}).items())
            pad = r.randint(50, 400)
            text = f"<think>{' step' * pad}</think>\n```python\n{body}\n```"
            toks = text.split()
            max_tokens = sampling.get("max_tokens")
            truncated = bool(max_tokens and len(toks) > max_tokens)
            if truncated:
                toks = toks[:max_tokens]
            tool_calls, env_outputs = [], []
            if task.verifier.type == "command":
                for f in task.env.target_files[:2]:
                    tool_calls.append({"name": "bash", "arguments": {"cmd": f"cat {f}"}})
                    env_outputs.append({"tool": "bash", "output": f"<contents of {f}>"})
                tool_calls.append({"name": "edit", "arguments": {"files": list((sol.file_edits or {}).keys())}})
                env_outputs.append({"tool": "edit", "output": "ok"})
            out.append(Rollout(
                text=" ".join(toks), solution=sol, tokens=toks,
                logprobs=[round(-abs(r.gauss(0.3, 0.4)), 4) for _ in toks],
                tool_calls=tool_calls, env_outputs=env_outputs,
                messages=[{"role": "user", "content": task.prompt}, {"role": "assistant", "content": " ".join(toks)}],
                truncated=truncated, finish_reason="length" if truncated else "stop",
                model=self.model, harness=self.harness, sampling=dict(sampling, n=n, sample_index=i),
                reasoning_level=self.reasoning_level))
        return out

    @staticmethod
    def _default_wrong(task: Task) -> Solution:
        if task.verifier.type in ("python_function", "python_asserts"):
            return Solution(code=f"def {task.verifier.function or 'solve'}(*args):\n    return None\n")
        if task.verifier.type == "answer_match":
            return Solution(text="\\boxed{mock-wrong-answer}")
        return task.initial or Solution(file_edits={})


class MockScorer(LogprobScorer):
    def __init__(self, model: str = "mock-scorer"):
        self.model = model

    def score(self, task: Task, rollout: Rollout) -> Optional[list[float]]:
        r = _rng(self.model, task.id, rollout.text[:200])
        return [round(-abs(r.gauss(0.5, 0.5)), 4) for _ in rollout.tokens]


class MockReviewAgent(ReviewAgent):
    name = "mock_review_agent"

    def judge(self, task: Task, solution: Solution) -> dict:
        truth = solution.meta.get("mock_truth")
        if truth is None:
            return {"correct": None, "confidence": 0.0, "rationale": "no label"}
        return {"correct": bool(truth), "confidence": 1.0, "rationale": "mock: ground-truth label"}


class GrepShortcutAgent(ShortcutAgent):
    """Look for the answer lying around: any non-target file (or git history)
    that contains most of the reference's added lines."""
    name = "grep_shortcut"

    def __init__(self, min_line_len: int = 12, min_hit_fraction: float = 0.8):
        self.min_line_len, self.min_hit_fraction = min_line_len, min_hit_fraction

    def attempt(self, task: Task, container: Container) -> Optional[Solution]:
        if not task.reference or not task.reference.file_edits:
            return None
        targets = set(task.reference.file_edits)
        init = (task.initial.file_edits if task.initial and task.initial.file_edits else {})
        files = [f for f in container.list_files() if f not in targets and not f.startswith(".git/")]
        for target, ref in task.reference.file_edits.items():
            before = set((init.get(target) or container.read_file(target) or "").splitlines())
            needles = [l.strip() for l in (ref or "").splitlines()
                       if len(l.strip()) >= self.min_line_len and l not in before]
            if not needles:
                continue
            for f in files:
                content = container.read_file(f) or ""
                hits = sum(n in content for n in needles)
                if hits / len(needles) >= self.min_hit_fraction:
                    edits = {target: content} if os.path.basename(f) == os.path.basename(target) else {target: ref}
                    return Solution(file_edits=edits, meta={"found_in": f, "hit_fraction": hits / len(needles)})
            code, log, _ = container.run("git log --all -p 2>/dev/null | head -c 2000000", 60)
            if code == 0 and log and sum(n in log for n in needles) / len(needles) >= self.min_hit_fraction:
                return Solution(file_edits={target: ref}, meta={"found_in": "git history"})
        return None
