"""IFBench normalization and upstream evaluator boundary contracts."""
from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qwenpt.eval.ifbench import (
    IFBENCH_REVISION, normalize_ifbench, prepare_ifbench, preflight_ifbench, score_ifbench,
)


def raw_row():
    return {"key": "17", "prompt": "  Include cat and dog.  ",
            "instruction_id_list": ["keywords:existence", "count:letters"],
            "kwargs": [{"keywords": ["cat", "dog"], "unused": None}, {"letter": "a", "frequency": 1}]}


@dataclass
class StubInput:
    key: str
    instruction_id_list: list
    prompt: str
    kwargs: list


class TestIFBench(unittest.TestCase):
    def test_normalization_retains_exact_prompts_ids_constraints_and_no_answer(self):
        original = raw_row()
        question = normalize_ifbench([original])[0]
        self.assertEqual(question["prompt"], original["prompt"])
        self.assertEqual(question["question_id"], "17")
        self.assertEqual(question["metadata"]["ifbench"]["kwargs"], original["kwargs"])
        self.assertEqual(question["metadata"]["stratum"], "count+keywords")
        self.assertNotIn("answer", question)
        question["metadata"]["ifbench"]["kwargs"][0]["keywords"].append("owl")
        self.assertEqual(original["kwargs"][0]["keywords"], ["cat", "dog"])

    def test_rejects_duplicate_missing_or_misaligned_constraints(self):
        invalid = [[], [raw_row(), raw_row()], [{**raw_row(), "key": True}],
                   [{**raw_row(), "instruction_id_list": []}],
                   [{**raw_row(), "kwargs": [{}]}], [{**raw_row(), "prompt": ""}],
                   [{**raw_row(), "kwargs": [{"value": float("nan")}, {}]}]]
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                normalize_ifbench(rows)

    def evaluator(self):
        calls = []

        def strict(example, responses):
            calls.append(("strict", example, responses))
            # This mirrors the upstream side effect the adapter must isolate.
            example.kwargs[0].pop("unused", None)
            return SimpleNamespace(follow_all_instructions=False, follow_instruction_list=[True, False])

        def loose(example, responses):
            self.assertNotIn("unused", example.kwargs[0])
            calls.append(("loose", example, responses))
            return SimpleNamespace(follow_all_instructions=True, follow_instruction_list=[True, True])

        module = SimpleNamespace(
            InputExample=StubInput,
            instructions_registry=SimpleNamespace(INSTRUCTION_DICT={key: object for key in raw_row()["instruction_id_list"]}),
            test_instruction_following_strict=strict,
            test_instruction_following_loose=loose,
        )
        return module, calls

    def test_calls_both_official_graders_with_all_constraints_and_preserves_metadata(self):
        question = normalize_ifbench([raw_row()])[0]
        before = deepcopy(question)
        module, calls = self.evaluator()
        scorer = {"checkout": ".", "revision": IFBENCH_REVISION}
        with patch("qwenpt.eval.ifbench._load_evaluator", return_value=module):
            result = score_ifbench(question, "cat and dog", scorer)
            strict = score_ifbench(question, "cat and dog", {**scorer, "mode": "strict"})
        self.assertEqual(question, before)
        self.assertTrue(result["correct"])
        self.assertFalse(strict["correct"])
        self.assertEqual(result["ifbench"]["strict"]["instruction_correct"], 1)
        self.assertEqual(result["ifbench"]["strict"]["instruction_total"], 2)
        self.assertEqual(result["ifbench"]["loose"]["instruction_correct"], 2)
        self.assertEqual([call[0] for call in calls], ["strict", "loose", "strict", "loose"])
        self.assertEqual(calls[0][2], {question["prompt"]: "cat and dog"})

    def test_unknown_constraint_and_malformed_official_result_fail_loudly(self):
        question = normalize_ifbench([raw_row()])[0]
        module, _ = self.evaluator()
        scorer = {"checkout": ".", "revision": IFBENCH_REVISION}
        with patch("qwenpt.eval.ifbench._load_evaluator", return_value=module):
            wrong = deepcopy(question)
            wrong["metadata"]["ifbench"]["instruction_id_list"][0] = "unsupported:new"
            with self.assertRaisesRegex(ValueError, "registry"):
                score_ifbench(wrong, "text", scorer)
            module.test_instruction_following_loose = lambda *_: SimpleNamespace(
                follow_all_instructions=True, follow_instruction_list=[True])
            with self.assertRaisesRegex(ValueError, "malformed"):
                score_ifbench(question, "text", scorer)

    def test_checkout_and_revision_are_required_not_moving_aliases(self):
        question = normalize_ifbench([raw_row()])[0]
        for scorer in [{}, {"checkout": ".", "revision": "main"},
                       {"checkout": ".", "revision": IFBENCH_REVISION, "mode": "judge"}]:
            with self.subTest(scorer=scorer), self.assertRaises(ValueError):
                score_ifbench(question, "text", scorer)


@unittest.skipUnless(os.environ.get("IFBENCH_CHECKOUT"), "Set IFBENCH_CHECKOUT to run official dependency integration")
class TestOfficialIFBench(unittest.TestCase):
    def test_official_300_tasks_and_real_constraint_failures(self):
        root = os.environ["IFBENCH_CHECKOUT"]
        rows, manifest = prepare_ifbench(root)
        self.assertEqual(manifest["count"], 300)
        self.assertEqual(len(rows), 300)
        info = preflight_ifbench(root)
        self.assertGreaterEqual(info["checker_count"], 58)
        self.assertFalse(info["llm_judge"])
        question = normalize_ifbench([{"key": "synthetic-contract", "prompt": "Include cat and dog.",
            "instruction_id_list": ["keywords:existence"], "kwargs": [{"keywords": ["cat", "dog"]}]}])[0]
        scorer = {"checkout": root, "revision": IFBENCH_REVISION}
        self.assertTrue(score_ifbench(question, "A cat and a dog.", scorer)["correct"])
        self.assertFalse(score_ifbench(question, "A cat.", scorer)["correct"])
        self.assertFalse(score_ifbench(question, "", scorer)["correct"])

    def test_changed_checkout_identity_is_refused(self):
        with self.assertRaisesRegex(ValueError, "HEAD"):
            prepare_ifbench(Path(os.environ["IFBENCH_CHECKOUT"]), "0" * 40)


if __name__ == "__main__":
    unittest.main()
