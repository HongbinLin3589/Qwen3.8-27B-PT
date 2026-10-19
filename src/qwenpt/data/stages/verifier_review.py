"""Step 4: verifier review.

a) Static: flag tests that check things the statement does not ask for
   (exact error text, exact stdout, private internals). Regex heuristics plus
   the review agent's own `review_tests`.
b) Dynamic: sample several solutions per task, grade each with the verifier,
   and have the review agent judge each against the statement.
     verifier pass + agent says wrong  -> false positive
     verifier fail + agent says right  -> false negative
   Tasks with defects are dropped.
c) Human audit: write a random 100-200 graded samples per source to
   audit/<source>.jsonl with an empty `human_label` column.
Every task gets its verifier version stamped here.
"""
from __future__ import annotations

import os
import random
import re
from collections import defaultdict

from ..schema import Task
from ..verifiers import verifier_version, verify
from .base import Stage, register_stage

DEFAULT_LINT = {
    "exact_error_message": [r"pytest\.raises\([^)]*match\s*=", r"assert\s+str\(\s*\w+(\.value)?\s*\)\s*==",
                            r"assertEqual\(\s*str\(", r"assertRaisesRegex"],
    "exact_stdout": [r"(capsys|capfd)\.readouterr\(\)\.(out|err)\s*==", r"assert\s+\w*(out|stdout)\w*\s*==\s*['\"]"],
    "private_internals": [r"\b\w+\._[a-z]\w*", r"patch\(\s*['\"][\w.]+\._[a-z]"],
}


def lint_tests(task: Task, patterns: dict) -> list[dict]:
    issues = []
    root = task.env.repo_dir
    if task.verifier.type != "command" or not root or not os.path.isdir(root):
        return issues
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            rel = os.path.relpath(p, root)
            if "test" not in rel.lower() or not f.endswith((".py", ".js", ".ts", ".go", ".rs", ".java", ".sh")):
                continue
            with open(p, errors="replace") as fh:
                for ln, line in enumerate(fh, 1):
                    for kind, pats in patterns.items():
                        if any(re.search(x, line) for x in pats):
                            issues.append({"kind": kind, "file": rel, "line": ln, "text": line.strip()[:160]})
    return issues


@register_stage("verifier_review")
class VerifierReview(Stage):
    name = "verifier_review"

    def run(self, tasks):
        sandbox = self.b("sandbox", required=False)
        agent = self.b("review_agent")
        policy = self.b(self.cfg.get("policy", "student"))
        n = int(self.cfg.get("n_solutions", 8))
        sampling = self.cfg.get("sampling", {"temperature": 1.0, "top_p": 1.0, "max_tokens": 32768})
        min_conf = float(self.cfg.get("min_judge_confidence", 0.7))
        max_fp, max_fn = int(self.cfg.get("max_false_positive", 1)), int(self.cfg.get("max_false_negative", 1))
        lint_cfg = self.cfg.get("lint", {})
        patterns = lint_cfg.get("patterns", DEFAULT_LINT)
        lint_action = lint_cfg.get("action", "flag")  # flag | drop

        def review(t: Task):
            verifier_version(t)
            issues = lint_tests(t, patterns) if lint_cfg.get("enabled", True) else []
            issues += [{"kind": "agent", "text": x} for x in agent.review_tests(t)]
            graded, fp, fn = [], 0, 0
            for ro in policy.generate(t, n, sampling, sandbox):
                vr = verify(t, ro.solution, sandbox)
                j = agent.judge(t, ro.solution)
                label = None
                if j.get("correct") is not None and j.get("confidence", 0) >= min_conf:
                    if vr.passed and not j["correct"]:
                        label, fp = "false_positive", fp + 1
                    elif not vr.passed and j["correct"]:
                        label, fn = "false_negative", fn + 1
                graded.append({"task_id": t.id, "source": t.source, "prompt": t.prompt[:4000],
                               "solution": (ro.solution.code if ro.solution else None) or
                                           (ro.solution.file_edits if ro.solution else None),
                               "verifier_passed": vr.passed, "verifier_version": vr.verifier_version,
                               "judge": j, "auto_label": label, "human_label": None})
            return issues, graded, fp, fn

        results = self.pmap(review, tasks)
        out, by_source = [], defaultdict(list)
        for t, (issues, graded, fp, fn) in zip(tasks, results):
            by_source[t.source].extend(graded)
            info = {"verifier_version": t.verifier.version, "false_positive": fp, "false_negative": fn,
                    "n_graded": len(graded), "lint_issues": issues[:20]}
            if fp >= max_fp and max_fp > 0:
                self.ctx.drop(t, self.name, "verifier_false_positive", **info)
            elif fn >= max_fn and max_fn > 0:
                self.ctx.drop(t, self.name, "verifier_false_negative", **info)
            elif issues and lint_action == "drop":
                self.ctx.drop(t, self.name, "overspecified_tests", **info)
            else:
                if issues:
                    self.ctx.count(self.name, "flagged_lint")
                t.note(self.name, decision="keep", **info)
                out.append(t)
        k = int(self.cfg.get("audit_per_source", 150))
        rnd = random.Random(self.cfg.get("seed", 0))
        for src, rows in by_source.items():
            for row in rnd.sample(rows, min(k, len(rows))):
                self.ctx.writer(f"audit/{re.sub(r'[^\w.-]', '_', src)}.jsonl").write(row)
            self.ctx.count(self.name, f"audit_rows:{src}", min(k, len(rows)))
        self.ctx.count(self.name, "kept", len(out))
        return out
