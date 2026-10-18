"""Step 8: teacher trajectories for SFT / distillation.

The teacher runs each task; a trajectory is kept only if
  - it is not truncated and its length is within [min_tokens, max_tokens], and
  - an independent re-verification in a fresh container passes (the result the
    harness saw during the run is not trusted).
Kept trajectories go to trajectories.jsonl with all step-9 fields; the student
log-probs on the teacher tokens are scored so OPD / offline RL can reuse them.
Tasks flagged `needs_teacher_demo` by the difficulty plugin always get a run.
"""
from __future__ import annotations

from ..schema import Task
from ..storage import make_trajectory, validate
from ..verifiers import verify
from .base import Stage, register_stage


@register_stage("trajectory")
class TrajectoryFilter(Stage):
    name = "trajectory"

    def run(self, tasks):
        sandbox = self.b("sandbox", required=False)
        teacher = self.b("teacher")
        student = self.b("student", required=False)
        student_scorer = self.b("student_scorer", required=False)
        n = int(self.cfg.get("n_per_task", 4))
        keep = int(self.cfg.get("keep_per_task", 1))
        lo, hi = int(self.cfg.get("min_tokens", 16)), int(self.cfg.get("max_tokens", 65536))
        tiers = self.cfg.get("tiers")  # None = all tasks
        sampling = self.cfg.get("sampling", {"temperature": 0.7, "top_p": 0.95, "max_tokens": hi})
        drop_unsolved = self.cfg.get("drop_tasks_without_trajectory", False)

        def wanted(t: Task) -> bool:
            if (t.difficulty or {}).get("needs_teacher_demo"):
                return True
            return tiers is None or (t.difficulty or {}).get("tier") in tiers

        def run_one(t: Task):
            if not wanted(t):
                return {"skipped": True}
            stats = {"sampled": 0, "truncated": 0, "length": 0, "verify_fail": 0, "kept": 0}
            for ro in teacher.generate(t, n, sampling, sandbox):
                stats["sampled"] += 1
                if ro.truncated:
                    stats["truncated"] += 1
                    continue
                if not lo <= len(ro.tokens) <= hi:
                    stats["length"] += 1
                    continue
                vr = verify(t, ro.solution, sandbox)  # fresh container, independent of the run
                if not vr.passed:
                    stats["verify_fail"] += 1
                    continue
                if stats["kept"] >= keep:
                    continue
                method = "teacher_demo" if (t.difficulty or {}).get("needs_teacher_demo") else "teacher_distill"
                tr = make_trajectory(t, ro, vr, method, role="teacher",
                                     student_model=getattr(student, "model", None), teacher_model=teacher.model,
                                     other_scorer=student_scorer, uses=self.cfg.get("uses", ["sft", "offline_rl", "opd"]),
                                     run_id=self.ctx.run_id)
                missing = validate(tr)
                if missing:
                    tr.extra["missing_fields"] = missing
                    self.ctx.count(self.name, "trajectories_missing_fields")
                self.ctx.writer("trajectories.jsonl").write(tr.to_json())
                stats["kept"] += 1
            return stats

        out = []
        for t, s in zip(tasks, self.pmap(run_one, tasks)):
            for k, v in s.items():
                if k != "skipped":
                    self.ctx.count(self.name, f"traj_{k}", v)
            t.note(self.name, **s)
            if drop_unsolved and not s.get("skipped") and s["kept"] == 0:
                self.ctx.drop(t, self.name, "no_valid_teacher_trajectory", **s)
                continue
            out.append(t)
        self.ctx.count(self.name, "kept", len(out))
        return out
