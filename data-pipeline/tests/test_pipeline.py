"""python -m unittest discover -s tests -v"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

import build_sample  # noqa: E402
from datapipe import backends as B  # noqa: E402
from datapipe.config import load_config  # noqa: E402
from datapipe.pipeline import load_tasks, run_pipeline  # noqa: E402
from datapipe.schema import REQUIRED_TRAJECTORY_FIELDS, Solution, Task  # noqa: E402
from datapipe.stages import Context  # noqa: E402
from datapipe.stages.dedup import DecontamSplit, Dedup  # noqa: E402
from datapipe.stages.difficulty import DifficultyFilter  # noqa: E402
from datapipe.stages.env_check import EnvCheck  # noqa: E402
from datapipe.stages.leak_check import LeakCheck  # noqa: E402
from datapipe.stages.verifier_review import VerifierReview  # noqa: E402
from datapipe.textutil import MinHasher, exact_key  # noqa: E402

SAMPLE_CFG = os.path.join(ROOT, "examples", "sample.yaml")
SAMPLE_TASKS = os.path.join(ROOT, "examples", "sample", "tasks.jsonl")


def setUpModule():
    build_sample.main()


def by_id(tasks):
    return {t.id: t for t in tasks}


class Base(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp()
        self.cfg = load_config(SAMPLE_CFG)
        self.tasks = by_id(load_tasks(SAMPLE_TASKS))

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def ctx(self, **backend_over):
        specs = dict(self.cfg["backends"], **backend_over)
        return Context(self.cfg, self.out, {k: B.build(v) for k, v in specs.items()}, "test")

    def dropped(self):
        p = os.path.join(self.out, "dropped.jsonl")
        if not os.path.exists(p):
            return {}
        with open(p) as f:
            return {d["id"]: d["reason"] for d in map(json.loads, f)}

    def pick(self, *ids):
        return [self.tasks[i] for i in ids]


class TestText(unittest.TestCase):
    def test_exact_key_normalizes(self):
        self.assertEqual(exact_key("Ａ  b\n"), exact_key("a b"))

    def test_minhash_orders_similarity(self):
        m = MinHasher(128, 3)
        a, b, c = "the quick brown fox jumps", "the quick brown fox jumped", "完全不同的句子"
        self.assertGreater(m.jaccard(m.signature(a), m.signature(b)), m.jaccard(m.signature(a), m.signature(c)))


class TestDedupDecontam(Base):
    def test_dedup(self):
        ctx = self.ctx()
        out = Dedup(self.cfg["dedup"], ctx).run(self.pick("phi", "phi_copy", "phi_reworded", "sigma"))
        ctx.close()
        self.assertEqual([t.id for t in out], ["phi", "sigma"])
        self.assertEqual(self.dropped(), {"phi_copy": "exact_dup", "phi_reworded": "near_dup"})

    def test_answer_guard(self):
        a = self.tasks["easy_sum"]
        b = Task.from_dict(a.to_dict())
        b.id, b.prompt = "easy_sum_sq", a.prompt.replace("之和", "之和的平方")
        b.verifier.tests = [{"input": [3], "output": 36}]
        ctx = self.ctx()
        out = Dedup(dict(self.cfg["dedup"], embedding={"enabled": False}, minhash=dict(self.cfg["dedup"]["minhash"], threshold=0.3)), ctx).run([a, b])
        ctx.close()
        self.assertEqual(len(out), 2)

    def test_prefer_sources(self):
        cfg = dict(self.cfg["dedup"], prefer_sources=["other-dataset"])
        ctx = self.ctx()
        out = Dedup(cfg, ctx).run(self.pick("phi", "phi_copy"))
        ctx.close()
        self.assertEqual([t.id for t in out], ["phi_copy"])

    def test_decontam_and_splits(self):
        ctx = self.ctx()
        out = DecontamSplit(self.cfg["decontam_split"], ctx).run(
            self.pick("contaminated", "after_cutoff", "swe_eval_repo", "sigma", "swe_good"))
        ctx.close()
        self.assertEqual({t.id for t in out}, {"sigma", "swe_good"})
        self.assertEqual(self.dropped(), {"contaminated": "contaminated_ngram"})
        held = {d["id"] for d in map(json.loads, open(os.path.join(self.out, "eval_holdout.jsonl")))}
        self.assertEqual(held, {"after_cutoff", "swe_eval_repo"})

    def test_boilerplate_ngrams_ignored(self):
        boiler = "The first line of the input contains two integers n and m separated by one space. "
        evals = [{"id": f"e{i}", "prompt": boiler + f"Eval story number {i} about apples."} for i in range(4)]
        ep = os.path.join(self.out, "boiler_eval.jsonl")
        with open(ep, "w") as f:
            f.writelines(json.dumps(e) + "\n" for e in evals)
        train = Task.from_dict({"id": "t", "source": "s", "domain": "code", "prompt": boiler + "A different story.",
                                "verifier": {"type": "stdio", "tests": [{"input": "1", "output": "1"}]}})
        cfg = dict(self.cfg["decontam_split"], eval_sets=[ep], embedding={"enabled": False})
        for ngram, kept in (({"n": 13}, 0), ({"n": 13, "max_eval_df": 3}, 1)):
            ctx = self.ctx()
            out = DecontamSplit(dict(cfg, ngram=ngram), ctx).run([Task.from_dict(train.to_dict())])
            ctx.close()
            self.assertEqual(len(out), kept, ngram)

    def test_cutoff_is_config(self):
        cfg = json.loads(json.dumps(self.cfg["decontam_split"]))
        cfg["time_split"]["cutoff"] = "2026-12-01"
        ctx = self.ctx()
        out = DecontamSplit(cfg, ctx).run(self.pick("after_cutoff"))
        ctx.close()
        self.assertEqual(len(out), 1)


class TestEnvChecks(Base):
    def test_env_contract(self):
        ctx = self.ctx()
        out = EnvCheck(self.cfg["env_check"], ctx).run(self.pick("swe_good", "swe_already_solved", "bad_reference", "phi"))
        ctx.close()
        self.assertEqual({t.id for t in out}, {"swe_good", "phi"})
        d = self.dropped()
        self.assertEqual(d["swe_already_solved"], "initial_state_passes")
        self.assertEqual(d["bad_reference"], "reference_fails")
        hist = by_id(out)["swe_good"].history[-1]
        self.assertEqual((hist["initial_pass"], hist["reference_pass"]), ("0/2", "4/4"))

    def test_container_counts_are_config(self):
        cfg = dict(self.cfg["env_check"], n_initial=1, n_reference=3)
        ctx = self.ctx()
        out = EnvCheck(cfg, ctx).run(self.pick("swe_good"))
        ctx.close()
        self.assertEqual(out[0].history[-1]["reference_pass"], "3/3")

    def test_flaky_env_dropped(self):
        # Reference passes on every other container -> must be dropped.
        from datapipe.backends.sandboxes import LocalDirSandbox

        class Alternating(LocalDirSandbox):
            n = 0

            def create(self, task):
                c = super().create(task)
                Alternating.n += 1
                if Alternating.n % 2 == 0:
                    c.apply(Solution(file_edits={"tests/test_zz.py": "import unittest\nclass Z(unittest.TestCase):\n    def test(self):\n        self.fail()\n"}))
                return c

        ctx = Context(self.cfg, self.out, {"sandbox": Alternating()}, "t")
        out = EnvCheck(self.cfg["env_check"], ctx).run(self.pick("swe_good"))
        ctx.close()
        self.assertEqual(out, [])
        self.assertEqual(self.dropped()["swe_good"], "inconsistent_runs")

    def test_verifier_review(self):
        ctx = self.ctx()
        out = VerifierReview(self.cfg["verifier_review"], ctx).run(self.pick("weak_tests", "strict_order", "sigma", "swe_lint_flag"))
        ctx.close()
        self.assertEqual({t.id for t in out}, {"sigma", "swe_lint_flag"})
        d = self.dropped()
        self.assertEqual(d["weak_tests"], "verifier_false_positive")
        self.assertEqual(d["strict_order"], "verifier_false_negative")
        lint = [h for h in by_id(out)["swe_lint_flag"].history if h["stage"] == "verifier_review"][0]["lint_issues"]
        self.assertTrue(any(i["kind"] == "exact_error_message" for i in lint))
        self.assertTrue(all(t.verifier.version for t in out))
        self.assertTrue(os.path.exists(os.path.join(self.out, "audit", "synthetic-math.jsonl")))

    def test_leak_check(self):
        ctx = self.ctx()
        out = LeakCheck(self.cfg["leak_check"], ctx).run(self.pick("swe_leaky", "swe_build_copy_only", "swe_good"))
        ctx.close()
        # build/lib copy is removed by cleanup; docs/ copy is a real leak
        self.assertEqual({t.id for t in out}, {"swe_build_copy_only", "swe_good"})
        self.assertEqual(self.dropped()["swe_leaky"], "shortcut_found")
        self.assertIn("rm -rf build", by_id(out)["swe_build_copy_only"].env.setup_cmd)


class TestDifficulty(Base):
    def test_tiers(self):
        cfg = dict(self.cfg["difficulty"], enabled=True)
        ctx = self.ctx()
        easy = [t for i, t in self.tasks.items() if i.startswith("easy_")]
        out = DifficultyFilter(cfg, ctx).run(self.pick("phi", "very_hard", "fib_mod") + easy)
        ctx.close()
        tiers = {t.id: t.difficulty["tier"] for t in out}
        self.assertEqual(tiers["very_hard"], "unsolved")
        self.assertTrue(by_id(out)["very_hard"].difficulty["needs_teacher_demo"])
        self.assertEqual(by_id(out)["very_hard"].difficulty["n_samples"], 32)
        self.assertIn(by_id(out)["phi"].difficulty["n_samples"], (4, 8))
        anchors = [t for t in out if t.difficulty.get("anchor")]
        self.assertEqual(len(anchors), round(len(easy) * cfg["anchor_fraction"]))
        compose = list(open(os.path.join(self.out, "compose_pool.jsonl")))
        self.assertEqual(len(compose), len(easy) - len(anchors))
        rows = [json.loads(l) for l in open(os.path.join(self.out, "rollouts.jsonl"))]
        self.assertTrue(all(r["difficulty_tier"] for r in rows))

    def test_sample_counts_are_config(self):
        cfg = dict(self.cfg["difficulty"], enabled=True, initial_samples=2, zero_escalation=[6])
        ctx = self.ctx()
        out = DifficultyFilter(cfg, ctx).run(self.pick("very_hard"))
        ctx.close()
        self.assertEqual(out[0].difficulty["n_samples"], 6)


class TestEndToEnd(Base):
    def test_default_run(self):
        rep = run_pipeline(list(self.tasks.values()), self.cfg, self.out)
        flow = {s["stage"]: s for s in rep["flow"]}
        self.assertTrue(flow["difficulty"]["skipped"])  # off by default
        kept = {json.loads(l)["id"] for l in open(os.path.join(self.out, "tasks.jsonl"))}
        for bad in ("phi_copy", "phi_reworded", "contaminated", "after_cutoff", "swe_eval_repo", "bad_reference",
                    "swe_already_solved", "weak_tests", "strict_order", "swe_leaky"):
            self.assertNotIn(bad, kept)
        self.assertIn("swe_good", kept)
        with open(os.path.join(self.out, "trajectories.jsonl")) as f:
            trajs = [json.loads(l) for l in f]
        self.assertTrue(trajs)
        for tr in trajs:
            for f in REQUIRED_TRAJECTORY_FIELDS:
                self.assertIn(f, tr)
            self.assertIsNotNone(tr["student_logprobs"])
            self.assertEqual(len(tr["student_logprobs"]), len(tr["teacher_logprobs"]))
            self.assertTrue(tr["passed"])

    def test_cli_with_difficulty(self):
        r = subprocess.run([sys.executable, "-m", "datapipe", "run", "--config", SAMPLE_CFG, "--input", SAMPLE_TASKS,
                            "--out", self.out, "--set", "difficulty.enabled=true"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        rep = json.load(open(os.path.join(self.out, "report.json")))
        self.assertIn("tier:all_correct", rep["stats"]["difficulty"])


@unittest.skipUnless(shutil.which("docker") and subprocess.run(["docker", "info"], capture_output=True).returncode == 0
                     and subprocess.run(["docker", "image", "inspect", "python:3.12-slim"], capture_output=True).returncode == 0,
                     "docker or python:3.12-slim not available")
class TestDocker(Base):
    def test_env_check_in_docker(self):
        ctx = self.ctx(sandbox={"type": "docker"})
        out = EnvCheck(self.cfg["env_check"], ctx).run(self.pick("swe_good", "swe_already_solved"))
        ctx.close()
        self.assertEqual([t.id for t in out], ["swe_good"])


if __name__ == "__main__":
    unittest.main()
