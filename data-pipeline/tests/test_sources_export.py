"""Download (sources), new verifiers, registries and verl export."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datapipe.export import export_verl, read_parquet, drop_nulls  # noqa: E402
from datapipe.export.verl import validate_sft_row, write_parquet  # noqa: E402
from datapipe.export.verl_reward import compute_score  # noqa: E402
from datapipe.schema import Solution, Task  # noqa: E402
from datapipe.sources import download_sources, register_loader, register_mapper  # noqa: E402
from datapipe.sources.mappers import get_path, map_fields  # noqa: E402
from datapipe.verifiers import extract_answer, verify  # noqa: E402

try:
    import pyarrow  # noqa: F401
    HAVE_PA = True
except ImportError:
    HAVE_PA = False

GSM_ROWS = [
    {"question": "Tom has 3 apples and buys 2 more. How many apples?", "answer": "3+2=<<3+2=5>>5\n#### 5"},
    {"question": "Tom has 3 apples and buys 2 more. How many apples?", "answer": "3+2=<<3+2=5>>5\n#### 5"},
    {"question": "A box holds 12 eggs. How many eggs are in 4 boxes?", "answer": "12*4=48\n#### 48"},
    {"question": "", "answer": "#### 1"},
]
EVAL_ROWS = [{"question": "A box holds 12 eggs. How many eggs are in 4 boxes?", "answer": "#### 48"}]
MBPP_ROWS = [{"task_id": 7, "text": "Write a function to double a number.", "code": "def dbl(x):\n    return 2*x",
              "test_list": ["assert dbl(2) == 4", "assert dbl(0) == 0"], "test_setup_code": ""}]


class Tmp(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def write_jsonl(self, name, rows):
        p = os.path.join(self.d, name)
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return p


class TestMappersVerifiers(unittest.TestCase):
    def test_get_path(self):
        row = {"prompt": [{"content": "hi"}], "reward_model": {"ground_truth": "3"}}
        self.assertEqual(get_path(row, "prompt.0.content"), "hi")
        self.assertEqual(get_path(row, "reward_model.ground_truth"), "3")
        self.assertIsNone(get_path(row, "prompt.5.content"))

    def test_fields_mapper(self):
        t = map_fields(GSM_ROWS[0], 0, source="gsm8k", prompt="question", answer="answer", answer_after="####",
                       reference_text="answer", domain="math")
        self.assertEqual((t["id"], t["domain"], t["verifier"]["answer"]), ("gsm8k-0", "math", "5"))
        self.assertIsNone(map_fields(GSM_ROWS[3], 3, source="g", prompt="question", answer="answer"))

    def test_answer_match(self):
        t = Task.from_dict({"id": "a", "source": "s", "domain": "math", "prompt": "",
                            "verifier": {"type": "answer_match", "answer": "\\frac{1}{2}"}})
        self.assertTrue(verify(t, Solution(text="so \\boxed{\\dfrac{1}{2}}")).passed)
        self.assertFalse(verify(t, Solution(text="\\boxed{2}")).passed)
        self.assertEqual(extract_answer("a\n#### 1,234"), "1,234")


class TestCodeMappers(unittest.TestCase):
    def test_rllm_code_layouts(self):
        from datapipe.sources.mappers import map_rllm_code
        stdin = map_rllm_code({"problem": "add <image>", "tests": "[{'input': '1 2', 'output': '3'}]"}, 0, source="cf",
                              prompt_replace={"<image>": "[image]"})
        self.assertEqual((stdin["verifier"]["type"], stdin["prompt"]), ("stdio", "add [image]"))
        t = Task.from_dict(stdin)
        self.assertTrue(verify(t, Solution(code="print(sum(map(int, input().split())))")).passed)
        self.assertFalse(verify(t, Solution(code="print(4)")).passed)
        call = map_rllm_code({"problem": "p", "tests": {"fn_name": "f", "inputs": [[1, 2]], "outputs": [[[1, 2]]]}},
                             0, source="apps")
        t = Task.from_dict(call)
        self.assertTrue(verify(t, Solution(code="def f(a, b):\n    return (a, b)")).passed)  # tuple == list
        lc = map_rllm_code({"problem": "p", "solutions": "class Solution:\n    def f(self, x: List[int]) -> int:\n"
                            "        return len(x)\n",
                            "tests": {"functional": "def check(candidate):\n    assert candidate(x=[1]) == 1\n"
                                                    "check(Solution().f)"}}, 0, source="lc")
        t = Task.from_dict(lc)
        self.assertEqual(t.verifier.function, "f")
        self.assertTrue(verify(t, t.reference).passed)  # needs the typing prelude

    def test_ignore_patterns(self):
        from datapipe.stages.dedup import strip_patterns
        self.assertEqual(strip_patterns("Intro. Solve x.", ("Intro\\.",)).strip(), "Solve x.")


class TestDownload(Tmp):
    def make_data(self):
        cfg = {"sources": [
            {"name": "toy-gsm", "role": "train",
             "loader": {"type": "file", "path": self.write_jsonl("g.jsonl", GSM_ROWS)},
             "mapper": {"type": "fields", "prompt": "question", "answer": "answer", "answer_after": "####",
                        "reference_text": "answer", "domain": "math"}},
            {"name": "toy-mbpp", "role": "train",
             "loader": {"type": "file", "path": self.write_jsonl("m.jsonl", MBPP_ROWS)}, "mapper": {"type": "mbpp"}},
            {"name": "toy-eval", "role": "eval",
             "loader": {"type": "file", "path": self.write_jsonl("e.jsonl", EVAL_ROWS)},
             "mapper": {"type": "fields", "prompt": "question", "answer": "answer"}},
            {"name": "broken", "role": "train", "loader": {"type": "file", "path": "/nonexistent.jsonl"}},
        ]}
        out = os.path.join(self.d, "data")
        return out, download_sources(cfg, out)

    def test_file_sources(self):
        out, m = self.make_data()
        st = {s["name"]: s for s in m["sources"]}
        self.assertEqual((st["toy-gsm"]["rows"], st["toy-gsm"]["written"]), (4, 3))
        self.assertEqual(st["broken"]["status"], "error")
        with open(os.path.join(out, "tasks.jsonl")) as f:
            tasks = [json.loads(l) for l in f]
        self.assertEqual(len(tasks), 4)
        self.assertIn("assert dbl(2) == 4", tasks[-1]["prompt"])  # MBPP tests go into the prompt
        self.assertTrue(os.path.exists(os.path.join(out, "eval", "toy-eval.jsonl")))

    def test_where_filter_and_custom_registration(self):
        @register_loader("toy_inline")
        def _load(rows, cache_dir=None, **_):
            yield from rows

        @register_mapper("toy_upper")
        def _map(row, i, source, **_):
            return {"id": f"{source}-{i}", "source": source, "domain": "math", "prompt": row["q"].upper(),
                    "verifier": {"type": "answer_match", "answer": row["a"]}}

        cfg = {"sources": [{"name": "t", "loader": {"type": "toy_inline", "rows": [{"q": "x", "a": "1", "lvl": 1},
                                                                                    {"q": "y", "a": "2", "lvl": 5}]},
                            "where": [{"field": "lvl", "min": 3}], "mapper": {"type": "toy_upper"}}]}
        download_sources(cfg, self.d)
        with open(os.path.join(self.d, "tasks.jsonl")) as f:
            rows = [json.loads(l) for l in f]
        self.assertEqual([r["prompt"] for r in rows], ["Y"])

    def test_git_loader(self):
        repo = os.path.join(self.d, "repo")
        os.makedirs(os.path.join(repo, "data"))
        with open(os.path.join(repo, "data", "a.jsonl"), "w") as f:
            f.write(json.dumps(GSM_ROWS[2]) + "\n")
        git = ["git", "-C", repo, "-c", "user.email=t@t", "-c", "user.name=t"]
        subprocess.run(["git", "init", "-q", repo], check=True)
        subprocess.run(git + ["add", "."], check=True)
        subprocess.run(git + ["commit", "-qm", "x"], check=True)
        from datapipe.sources import load_rows
        rows = list(load_rows({"type": "git", "repo": "file://" + repo, "path": "data/*.jsonl"},
                              os.path.join(self.d, "cache")))
        self.assertEqual(rows, [GSM_ROWS[2]])

    def test_hf_loader_reports_failure(self):
        cfg = {"sources": [{"name": "nohf", "loader": {"type": "hf", "repo": "x/y", "backend": "rows_api"}}]}
        m = download_sources(cfg, self.d)
        self.assertEqual(m["sources"][0]["status"], "error")

    @unittest.skipUnless(HAVE_PA, "pyarrow not installed")
    def test_download_run_export(self):
        out, _ = self.make_data()
        run, exp = os.path.join(self.d, "run"), os.path.join(self.d, "exp")
        cfg = os.path.join(ROOT, "configs", "github_demo.yaml")
        for args in (["run", "--data", out, "--out", run, "--config", cfg],
                     ["export", "--run", run, "--out", exp, "--config", cfg]):
            r = subprocess.run([sys.executable, "-m", "datapipe", *args], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(run, "dropped.jsonl")) as f:
            reasons = sorted(json.loads(l)["reason"] for l in f)
        self.assertEqual(reasons, ["contaminated_ngram", "exact_dup"])
        rl = read_parquet(os.path.join(exp, "rl", "train.parquet"))
        self.assertEqual(len(rl), 2)
        gsm = next(r for r in rl if r["data_source"] == "toy-gsm")
        self.assertEqual(gsm["reward_model"], {"style": "rule", "ground_truth": "5"})
        self.assertEqual(gsm["prompt"][0], {"role": "system", "content": "你是数学助手。"})
        sft = read_parquet(os.path.join(exp, "sft", "reference.parquet"))
        self.assertEqual(sft[0]["messages"][-1]["content"], "3+2=5\n#### 5")  # <<...>> stripped


class TestVerlFormat(unittest.TestCase):
    EX = os.path.join(ROOT, "examples", "verl_format", "computer_use_sft.json")

    def test_user_example_is_valid(self):
        with open(self.EX) as f:
            self.assertEqual(validate_sft_row(json.load(f)), [])

    def test_validation_catches_errors(self):
        with open(self.EX) as f:
            row = json.load(f)
        row["images"] = row["images"][:3]
        row["messages"][3]["tool_call_id"] = "nope"
        errs = " | ".join(validate_sft_row(row))
        self.assertIn("placeholders", errs)
        self.assertIn("no earlier matching tool_call", errs)
        self.assertIn("never answered", errs)

    @unittest.skipUnless(HAVE_PA, "pyarrow not installed")
    def test_parquet_roundtrip(self):
        with open(self.EX) as f:
            row = json.load(f)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.parquet")
            write_parquet([row], p)
            self.assertEqual(drop_nulls(read_parquet(p)[0]), row)

    def test_reward_function(self):
        self.assertEqual(compute_score("gsm8k", "so \\boxed{72}", "72"), 1.0)
        self.assertEqual(compute_score("gsm8k", "71", "72"), 0.0)
        gt = json.dumps({"type": "python_asserts", "function": "dbl", "tests": ["assert dbl(2) == 4"]})
        self.assertEqual(compute_score("mbpp", "```python\ndef dbl(x):\n    return 2*x\n```", gt), 1.0)
        self.assertEqual(compute_score("mbpp", "```python\ndef dbl(x):\n    return x\n```", gt), 0.0)


@unittest.skipUnless(HAVE_PA, "pyarrow not installed")
class TestVerlExportConfig(Tmp):
    """The user's computer-use sample through export with different format settings."""

    def setUp(self):
        super().setUp()
        with open(TestVerlFormat.EX) as f:
            self.ex = json.load(f)
        os.makedirs(os.path.join(self.d, "media", "images"))
        for im in self.ex["images"]:
            with open(os.path.join(self.d, "media", im["image"]), "wb") as f:
                f.write(b"\xff\xd8fake-jpeg-" + im["image"].encode())
        self.run_dir = os.path.join(self.d, "run")
        os.makedirs(self.run_dir)
        text = {"messages": [{"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "2"}]}
        self.write_jsonl("run/trajectories.jsonl", [
            {"messages": self.ex["messages"], "tools": self.ex["tools"], "domain": "gui",
             "reasoning_level": "off", "extra": {"images": [i["image"] for i in self.ex["images"]]}},
            dict(text, domain="math")])

    def export(self, **sft):
        from datapipe.export import export
        out = os.path.join(self.d, "out")
        shutil.rmtree(out, ignore_errors=True)
        rep = export(self.run_dir, out, {"rl": {"enabled": False}, "sft": dict({"media_root": os.path.join(self.d, "media")}, **sft)})
        return out, rep

    def test_default_split_by_modality(self):
        out, rep = self.export()
        self.assertEqual((rep["sft/trajectories.multimodal.parquet"], rep["sft/trajectories.parquet"]), (1, 1))
        from datapipe.export.verl import verl_view
        mm = verl_view(os.path.join(out, "sft", "trajectories.multimodal.parquet"))[0]
        expected = {k: v for k, v in self.ex.items() if k not in ("videos", "enable_thinking")}
        self.assertEqual(drop_nulls(mm), dict(expected, enable_thinking=False))  # reasoning_level "off"
        txt = verl_view(os.path.join(out, "sft", "trajectories.parquet"))[0]
        self.assertNotIn("images", txt)  # verl would demand a processor otherwise

    def test_bytes_and_user_layout(self):
        out, _ = self.export(media="bytes", empty_media_columns=True, enable_thinking=False)
        from datapipe.export import validate_file
        p = os.path.join(out, "sft", "trajectories.multimodal.parquet")
        row = read_parquet(p)[0]
        self.assertEqual(row["images"][0], {"bytes": b"\xff\xd8fake-jpeg-images/screen_0.jpg"})
        self.assertEqual((row["videos"], row["enable_thinking"]), ([], False))
        self.assertEqual(validate_file(p)["invalid"], 0)

    def test_renamed_keys(self):
        out, _ = self.export(keys={"messages": "conversations", "images": "imgs"})
        row = read_parquet(os.path.join(out, "sft", "trajectories.multimodal.parquet"))[0]
        self.assertIn("conversations", row)
        self.assertIn("imgs", row)

    def test_missing_media_goes_to_invalid(self):
        out, rep = self.export(media="bytes", media_root=os.path.join(self.d, "nowhere"))
        self.assertEqual(rep["sft/invalid.jsonl"], 1)


class TestCli(unittest.TestCase):
    def test_list(self):
        r = subprocess.run([sys.executable, "-m", "datapipe", "list"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        for word in ("dedup", "answer_match", "hf", "mbpp", "docker"):
            self.assertIn(word, r.stdout)


if __name__ == "__main__":
    unittest.main()
