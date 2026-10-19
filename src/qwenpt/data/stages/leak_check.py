"""Step 5: leak check (repo tasks).

1. Cleanup: remove build artifacts, caches and installed copies of the target
   project. The cleanup command is appended to the task's env.setup_cmd, so
   every later container (difficulty sampling, teacher trajectories, RL) gets
   the cleaned environment too.
2. Re-check that the reference still passes after cleanup.
3. Let a shortcut agent try to get the answer without doing the work. If its
   submission passes the verifier, the task is dropped.
"""
from __future__ import annotations

from ..schema import Task
from ..verifiers import verify
from .base import Stage, register_stage

DEFAULT_CLEANUP = [
    "rm -rf build dist .eggs *.egg-info .pytest_cache .mypy_cache .tox node_modules/.cache target/debug target/release",
    "find . -name '__pycache__' -type d -prune -exec rm -rf {} +",
    "find . \\( -name '*.pyc' -o -name '*.pyo' -o -name '*.so' -o -name '*.o' -o -name '*.class' \\) -delete",
]


@register_stage("leak_check")
class LeakCheck(Stage):
    name = "leak_check"

    def run(self, tasks):
        sandbox = self.b("sandbox")
        agent = self.b("shortcut_agent", required=False)
        domains = set(self.cfg.get("domains", ["swe", "terminal"]))
        cleanup = list(self.cfg.get("cleanup_cmds", DEFAULT_CLEANUP))
        attempts = int(self.cfg.get("shortcut_attempts", 1))

        def check(t: Task):
            if t.verifier.type != "command" or t.domain not in domains:
                return ("keep", None, {"skipped": True})
            cmds = cleanup[:]
            pkg = t.meta.get("installed_package")
            if pkg:  # installed copy of the target project in site-packages
                cmds.append(f"pip uninstall -y {pkg} >/dev/null 2>&1 || true")
            clean = " ; ".join(f"({c}) 2>/dev/null" for c in cmds) + " ; true"
            if clean not in (t.env.setup_cmd or ""):
                t.env.setup_cmd = f"{t.env.setup_cmd} && {clean}" if t.env.setup_cmd else clean
            try:
                ref = verify(t, t.reference, sandbox)
            except Exception as e:
                return ("drop", "env_build_failed_after_cleanup", {"error": str(e)[:300]})
            if not ref.passed:
                return ("drop", "reference_fails_after_cleanup", {"detail": ref.details})
            if agent is None:
                return ("keep", None, {"shortcut": "not_run"})
            for k in range(attempts):
                with sandbox.create(t) as c:
                    c.apply(t.initial)
                    sol = agent.attempt(t, c)
                if sol is None:
                    continue
                vr = verify(t, sol, sandbox)
                if vr.passed:
                    return ("drop", "shortcut_found", {"attempt": k, "found": sol.meta})
            return ("keep", None, {"shortcut": "none_found"})

        out = []
        for t, (dec, reason, info) in zip(tasks, self.pmap(check, tasks)):
            if dec == "drop":
                self.ctx.drop(t, self.name, reason, **info)
            else:
                t.note(self.name, decision="keep", **info)
                out.append(t)
        self.ctx.count(self.name, "kept", len(out))
        return out
