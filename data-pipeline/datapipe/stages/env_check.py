"""Step 3: environment build + execution consistency.

For each repo task: build `n_initial + n_reference` brand-new containers, one
run each. Every initial-state run must fail, every reference run must pass.
Repeating each side several times is what catches flaky environments.
Defaults follow CodeMidas: 6 containers, 2 initial, 4 reference.
"""
from __future__ import annotations

from ..schema import Task
from ..verifiers import verify
from .base import Stage, register_stage


@register_stage("env_check")
class EnvCheck(Stage):
    name = "env_check"

    def run(self, tasks):
        sandbox = self.b("sandbox")
        domains = set(self.cfg.get("domains", ["swe", "terminal"]))
        n_init, n_ref = int(self.cfg.get("n_initial", 2)), int(self.cfg.get("n_reference", 4))
        check_functions = self.cfg.get("check_function_tasks", True)

        def check(t: Task):
            if t.verifier.type == "command" and t.domain in domains:
                if t.reference is None:
                    return ("drop", "no_reference", {})
                runs = []
                try:
                    for kind, sol, n in (("initial", t.initial, n_init), ("reference", t.reference, n_ref)):
                        for _ in range(n):
                            r = verify(t, sol, sandbox)  # own fresh container per call
                            runs.append({"kind": kind, "passed": r.passed, "exit": r.details.get("exit_code")})
                except Exception as e:  # build / setup failure
                    return ("drop", "env_build_failed", {"error": str(e)[:300]})
                init_pass = sum(r["passed"] for r in runs if r["kind"] == "initial")
                ref_pass = sum(r["passed"] for r in runs if r["kind"] == "reference")
                info = {"initial_pass": f"{init_pass}/{n_init}", "reference_pass": f"{ref_pass}/{n_ref}"}
                if init_pass == n_init and n_init:
                    return ("drop", "initial_state_passes", info)
                if ref_pass == 0:
                    return ("drop", "reference_fails", info)
                if init_pass or ref_pass < n_ref:
                    return ("drop", "inconsistent_runs", info)
                return ("keep", None, info)
            if t.verifier.type != "command" and check_functions:
                # Lighter contract: reference passes twice (if we have one), empty fails.
                if t.reference is not None:
                    a, b = verify(t, t.reference), verify(t, t.reference)
                    if not (a.passed and b.passed):
                        return ("drop", "reference_fails" if not (a.passed or b.passed) else "inconsistent_runs",
                                {"runs": [a.passed, b.passed], "detail": a.details})
                if verify(t, t.initial).passed:
                    return ("drop", "initial_state_passes", {})
                return ("keep", None, {})
            return ("keep", None, {"skipped": True})

        out = []
        for t, (dec, reason, info) in zip(tasks, self.pmap(check, tasks)):
            if dec == "drop":
                self.ctx.drop(t, self.name, reason, **info)
            else:
                t.note(self.name, decision="keep", **info)
                out.append(t)
        self.ctx.count(self.name, "kept", len(out))
        return out
