"""Step 6: difficulty filter. A plugin, off by default (`difficulty.enabled`).

Sample the student (Qwen3.8-27B base) `initial_samples` times per task:
  all correct  -> keep `anchor_fraction` of them as anchors; the rest go to
                  compose_pool.jsonl (raw material for composed tasks, 4.3)
                  or are dropped
  some correct -> main pool, top up to `mid_total` samples
  none correct -> escalate through `zero_escalation` (e.g. 16 then 32 total);
                  solved at some point -> main pool, tier "hard";
                  never solved -> teacher demo (kept, flagged) or dropped
Every sample is stored as a step-9 trajectory in rollouts.jsonl, so this batch
is also round-1 data for RFT, APEX and offline RL.
"""
from __future__ import annotations

import random

from ..schema import Task
from ..storage import make_trajectory, validate
from ..verifiers import verify
from .base import Stage, register_stage


@register_stage("difficulty")
class DifficultyFilter(Stage):
    name = "difficulty"

    def run(self, tasks):
        sandbox = self.b("sandbox", required=False)
        student = self.b("student")
        teacher_scorer = self.b("teacher_scorer", required=False)
        teacher = self.b("teacher", required=False)
        n0 = int(self.cfg.get("initial_samples", 4))
        mid_total = int(self.cfg.get("mid_total", 8))
        escalation = [int(x) for x in self.cfg.get("zero_escalation", [16, 32])]
        sampling = self.cfg.get("sampling", {"temperature": 1.0, "top_p": 1.0, "max_tokens": 32768})
        store = self.cfg.get("store_rollouts", True)
        uses = self.cfg.get("rollout_uses", ["rft", "apex", "offline_rl", "opd"])

        def grade(t: Task):
            got = []  # (rollout, verify_result, round)

            def sample(k: int, rnd: int) -> list[bool]:
                for ro in student.generate(t, k, sampling, sandbox):
                    got.append((ro, verify(t, ro.solution, sandbox), rnd))
                return [vr.passed for _, vr, _ in got]

            res = sample(n0, 0)
            if all(res):
                d = {"tier": "all_correct"}
            elif any(res):
                if mid_total > n0:
                    res = sample(mid_total - n0, 1)
                d = {"tier": "main"}
            else:
                d = {"tier": "unsolved"}
                for i, total in enumerate(escalation):
                    if total > len(res):
                        res = sample(total - len(res), 2 + i)
                    if any(res):
                        d = {"tier": "hard"}
                        break
            d.update(n_samples=len(res), n_pass=sum(res))
            if store:
                for ro, vr, rnd in got:
                    tr = make_trajectory(t, ro, vr, "student_rollout", role="student",
                                         student_model=student.model,
                                         teacher_model=getattr(teacher_scorer, "model", None),
                                         other_scorer=teacher_scorer, uses=uses, run_id=self.ctx.run_id,
                                         extra={"difficulty_round": rnd})
                    tr.difficulty_tier = d["tier"]
                    missing = validate(tr)
                    if missing:
                        self.ctx.count(self.name, "rollouts_missing_fields")
                        tr.extra["missing_fields"] = missing
                    self.ctx.writer("rollouts.jsonl").write(tr.to_json())
            return d

        for t, d in zip(tasks, self.pmap(grade, tasks)):
            d["pass_rate"] = round(d["n_pass"] / d["n_samples"], 4)
            d["model"] = student.model
            t.difficulty = d
            self.ctx.count(self.name, f"tier:{d['tier']}")

        # anchors among all-correct tasks
        rnd = random.Random(self.cfg.get("seed", 0))
        full = [t for t in tasks if t.difficulty["tier"] == "all_correct"]
        n_anchor = round(len(full) * float(self.cfg.get("anchor_fraction", 0.125)))
        anchors = {t.id for t in rnd.sample(full, n_anchor)}
        full_action = self.cfg.get("all_correct_action", "compose_pool")  # compose_pool | drop
        unsolved_action = self.cfg.get("unsolved_action", "teacher_demo")  # teacher_demo | drop
        if unsolved_action == "teacher_demo" and teacher is None:
            unsolved_action = "drop"
            self.ctx.count(self.name, "no_teacher_unsolved_dropped")

        out = []
        for t in tasks:
            tier = t.difficulty["tier"]
            if tier == "all_correct":
                if t.id in anchors:
                    t.difficulty["anchor"] = True
                    out.append(t)
                elif full_action == "compose_pool":
                    self.ctx.divert(t, self.name, "compose_pool.jsonl", "all_correct_not_anchor")
                else:
                    self.ctx.drop(t, self.name, "all_correct_not_anchor")
            elif tier == "unsolved":
                if unsolved_action == "teacher_demo":
                    t.difficulty["needs_teacher_demo"] = True
                    out.append(t)
                else:
                    self.ctx.drop(t, self.name, "unsolved", n_samples=t.difficulty["n_samples"])
            else:
                out.append(t)
        self.ctx.count(self.name, "anchors", len(anchors))
        self.ctx.count(self.name, "kept", len(out))
        return out
