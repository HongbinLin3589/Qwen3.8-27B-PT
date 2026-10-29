"""LCB data privacy and isolated official grader boundary contracts."""
from copy import deepcopy
import base64
import json
import os
from pathlib import Path
import pickle
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

from qwenpt.eval.livecodebench import (
    _tests, normalize_livecodebench, preflight_livecodebench, score_livecodebench, validate_checkout,
)


REVISION = "a" * 40
IMAGE = "sha256:" + "b" * 64
PICKLE_PROBE_CALLED = False


def _pickle_probe():
    global PICKLE_PROBE_CALLED
    PICKLE_PROBE_CALLED = True
    return "[]"


class UnsafePickle:
    def __reduce__(self):
        return _pickle_probe, ()


def compressed_tests(value):
    return base64.b64encode(zlib.compress(pickle.dumps(value))).decode("ascii")


def raw_row():
    return {"question_id": "sample_1", "question_content": "Print the input number.",
            "public_test_cases": json.dumps([{"input": "1", "output": "1", "testtype": "stdin"}]),
            "private_test_cases": compressed_tests(json.dumps([
                {"input": "PRIVATE_TEST_INPUT", "output": "PRIVATE_TEST_ANSWER", "testtype": "stdin"}])),
            "metadata": "{}", "starter_code": "", "platform": "atcoder", "difficulty": "easy",
            "contest_date": "2025-01-04T00:00:00"}


class TestLiveCodeBenchNormalization(unittest.TestCase):
    def test_private_tests_never_enter_prompt_and_official_sample_is_preserved(self):
        question = normalize_livecodebench([raw_row()])[0]
        self.assertNotIn("PRIVATE_TEST", question["prompt"])
        self.assertNotIn("answer", question)
        sample = json.loads(question["metadata"]["lcb_sample"]["input_output"])
        self.assertEqual(sample, {"inputs": ["1", "PRIVATE_TEST_INPUT"],
                                  "outputs": ["1", "PRIVATE_TEST_ANSWER"], "fn_name": None})
        self.assertEqual(question["metadata"]["stratum"], "easy")

    def test_functional_starter_and_function_name_match_official_sample(self):
        row = raw_row()
        row.update(starter_code="class Solution:\n    def solve(self, x):", metadata='{"func_name":"solve"}',
                   public_test_cases='[{"input":"1","output":"2","testtype":"functional"}]', private_test_cases="[]")
        question = normalize_livecodebench([row])[0]
        self.assertIn(row["starter_code"], question["prompt"])
        self.assertEqual(json.loads(question["metadata"]["lcb_sample"]["input_output"])["fn_name"], "solve")

    def test_missing_test_keys_and_invalid_source_types_are_rejected(self):
        invalid = [{**raw_row(), "public_test_cases": [{"input": "1", "testtype": "stdin"}]},
                   {**raw_row(), "public_test_cases": [{"input": 1, "output": "1"}]},
                   {**raw_row(), "metadata": "[]"}, {**raw_row(), "metadata": '{"func_name":42}'},
                   {**raw_row(), "question_content": ""}, {**raw_row(), "starter_code": None},
                   {**raw_row(), "question_id": True}, {**raw_row(), "metadata": '{"func_name":"solve"}'}, {}]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                normalize_livecodebench([row])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            normalize_livecodebench([raw_row(), raw_row()])
        with self.assertRaisesRegex(ValueError, "no tasks"):
            normalize_livecodebench([])

    def test_pickle_globals_never_execute_and_bad_payloads_fail_safely(self):
        global PICKLE_PROBE_CALLED
        PICKLE_PROBE_CALLED = False
        with self.assertRaisesRegex(ValueError, "unsafe"):
            _tests(compressed_tests(UnsafePickle()))
        self.assertFalse(PICKLE_PROBE_CALLED)
        for value in [None, {}, "invalid base64?", compressed_tests(42),
                      base64.b64encode(zlib.compress(b"not a pickle")).decode(),
                      base64.b64encode(zlib.compress(pickle.dumps("[]") + b"trailing")).decode()]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                _tests(value)


class TestLiveCodeBenchDocker(unittest.TestCase):
    def setUp(self):
        self.question = normalize_livecodebench([raw_row()])[0]
        self.identity = {"checkout": str(Path("official-lcb").resolve()), "revision": REVISION, "image": IMAGE}
        self.scorer = {**self.identity, "timeout": 6}

    def verdict(self, **changes):
        result = {"correct": True, "test_results": [True, True], "metadata": {},
                  "code": "print(input())", "infra_error": False}
        result.update(changes)
        return SimpleNamespace(returncode=0, stdout="QWENPT_LCB_RESULT=" + json.dumps(result) + "\n", stderr="")

    def test_runs_only_docker_with_readonly_mounts_no_network_and_no_keys(self):
        def docker(command, **kwargs):
            self.assertEqual(command[:2], ["docker", "run"])
            self.assertNotIn("shell", kwargs)
            self.assertIn("--read-only", command)
            self.assertEqual(command[command.index("--network") + 1], "none")
            self.assertEqual(command[command.index("--user") + 1], "65534:65534")
            self.assertEqual(command[command.index("--cap-drop") + 1], "ALL")
            self.assertIn("no-new-privileges", command)
            mounts = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
            self.assertEqual(len(mounts), 3)
            self.assertTrue(all(mount.endswith(",readonly") for mount in mounts))
            payload_mount = next(mount for mount in mounts if "dst=/input.json" in mount)
            path = payload_mount.split("src=", 1)[1].split(",dst=", 1)[0]
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertNotIn("SECRET_TOKEN_VALUE", json.dumps(payload))
            self.assertNotIn("SECRET_TOKEN_VALUE", " ".join(command))
            self.assertNotIn("API_KEY", " ".join(command))
            self.assertEqual(payload["text"], "```python\nprint(input())\n```")
            self.assertEqual(command[-3:], [IMAGE, "python", "/worker.py"])
            return self.verdict()

        with patch("qwenpt.eval.livecodebench.preflight_livecodebench", return_value=self.identity), \
                patch("qwenpt.eval.livecodebench.subprocess.run", side_effect=docker) as runner, \
                patch.dict(os.environ, {"BENCHMARK_API_KEY": "SECRET_TOKEN_VALUE", "JUDGE_API_KEY": "SECRET_TOKEN_VALUE"}):
            result = score_livecodebench(self.question, "<think>reason</think>```python\nprint(input())\n```", self.scorer)
        self.assertEqual(runner.call_count, 1)
        self.assertTrue(result["correct"])

    def test_model_failure_and_infrastructure_error_are_distinct(self):
        with patch("qwenpt.eval.livecodebench.preflight_livecodebench", return_value=self.identity), \
                patch("qwenpt.eval.livecodebench.subprocess.run", return_value=self.verdict(
                    correct=False, test_results=[-3], metadata={"error_code": -3})):
            self.assertFalse(score_livecodebench(self.question, "code", self.scorer)["correct"])
        with patch("qwenpt.eval.livecodebench.preflight_livecodebench", return_value=self.identity), \
                patch("qwenpt.eval.livecodebench.subprocess.run", return_value=self.verdict(infra_error=True)), \
                self.assertRaisesRegex(RuntimeError, "TestRunnerError"):
            score_livecodebench(self.question, "code", self.scorer)

    def test_partial_or_inconsistent_success_never_becomes_pass(self):
        invalid = [self.verdict(test_results=[True]), self.verdict(correct=True, test_results=[True, False]),
                   self.verdict(test_results=[True, 0.5]), self.verdict(test_results=[]),
                   self.verdict(correct="true"), self.verdict(metadata=None),
                   SimpleNamespace(returncode=0, stdout="QWENPT_LCB_RESULT={}\n", stderr=""),
                   SimpleNamespace(returncode=0, stdout="no verdict\n", stderr=""),
                   SimpleNamespace(returncode=1, stdout="", stderr="")]
        for result in invalid:
            with self.subTest(result=result), \
                    patch("qwenpt.eval.livecodebench.preflight_livecodebench", return_value=self.identity), \
                    patch("qwenpt.eval.livecodebench.subprocess.run", return_value=result), self.assertRaises(RuntimeError):
                score_livecodebench(self.question, "code", self.scorer)

    def test_outer_timeout_removes_container_and_reports_infrastructure_failure(self):
        calls = []
        def docker(command, **kwargs):
            calls.append(command)
            if command[1] == "run":
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            return SimpleNamespace(returncode=0)
        with patch("qwenpt.eval.livecodebench.preflight_livecodebench", return_value=self.identity), \
                patch("qwenpt.eval.livecodebench.subprocess.run", side_effect=docker), \
                self.assertRaisesRegex(RuntimeError, "outer timeout"):
            score_livecodebench(self.question, "code", self.scorer)
        self.assertEqual(calls[1][:3], ["docker", "rm", "-f"])
        self.assertEqual(calls[1][3], calls[0][calls[0].index("--name") + 1])

    def test_invalid_sample_or_fractional_alarm_timeout_fails_before_docker(self):
        for sample in [{"inputs": [], "outputs": [], "fn_name": None},
                       {"inputs": ["1"], "outputs": [], "fn_name": None},
                       {"inputs": [1], "outputs": ["1"], "fn_name": None},
                       {"inputs": ["1"], "outputs": ["1"], "fn_name": 42}]:
            question = deepcopy(self.question)
            question["metadata"]["lcb_sample"]["input_output"] = json.dumps(sample)
            with self.subTest(sample=sample), patch("qwenpt.eval.livecodebench.subprocess.run") as runner, self.assertRaises(ValueError):
                score_livecodebench(question, "code", self.scorer)
            runner.assert_not_called()
        with self.assertRaisesRegex(ValueError, "integer"):
            score_livecodebench(self.question, "code", {**self.scorer, "timeout": 0.5})

    def test_preflight_rejects_mutable_image_tags_and_changed_identity(self):
        with patch("qwenpt.eval.livecodebench.validate_checkout", return_value=Path("official-lcb")), \
                self.assertRaisesRegex(ValueError, "immutable"):
            preflight_livecodebench({**self.scorer, "image": "python:latest"})
        with patch("qwenpt.eval.livecodebench.validate_checkout", return_value=Path("official-lcb")), \
                patch("qwenpt.eval.livecodebench.subprocess.check_output", return_value="sha256:" + "c" * 64), \
                self.assertRaisesRegex(ValueError, "identity mismatch"):
            preflight_livecodebench(self.scorer)

    def test_checkout_must_be_repository_root_and_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve()
            for replies, message in [([str(path.parent), REVISION, ""], "root"),
                                     ([str(path), REVISION, " M evaluator.py"], "differs")]:
                with patch("qwenpt.eval.livecodebench.subprocess.check_output", side_effect=replies), \
                        self.assertRaisesRegex(ValueError, message):
                    validate_checkout(str(path), REVISION)


if __name__ == "__main__":
    unittest.main()
