"""Opt-in integration with the real official grader, inside a Docker container."""
import os

import pytest

from qwenpt.eval.livecodebench import normalize_livecodebench, score_livecodebench


@pytest.mark.skipif(not os.environ.get("LCB_CHECKOUT") or not os.environ.get("LCB_SANDBOX_IMAGE"),
                    reason="Requires pinned official LCB checkout and built Docker grading image")
@pytest.mark.parametrize("functional", [False, True])
def test_official_execution_grader_distinguishes_correct_code_from_wrong_code(functional):
    testtype = "functional" if functional else "stdin"
    question = normalize_livecodebench([{
        "question_id": "synthetic-sum-contract", "question_content": "Print the sum of two integers.",
        "starter_code": "class Solution: pass" if functional else "",
        "metadata": {"func_name": "add"} if functional else {}, "difficulty": "easy", "platform": "synthetic",
        "public_test_cases": [{"input": "2\n3" if functional else "2 3\n", "output": "5\n", "testtype": testtype}],
        "private_test_cases": [{"input": "8\n-2" if functional else "8 -2\n", "output": "6\n", "testtype": testtype}],
    }])[0]
    scorer = {"checkout": os.environ["LCB_CHECKOUT"], "revision": "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"}
    good_code = "class Solution:\n    def add(self,a,b): return a+b" if functional else "print(sum(map(int,input().split())))"
    bad_code = "class Solution:\n    def add(self,a,b): return 0" if functional else "print(0)"
    correct = score_livecodebench(question, "```python\n" + good_code + "\n```", scorer)
    wrong = score_livecodebench(question, "```python\n" + bad_code + "\n```", scorer)
    assert correct["correct"] is True, correct["execution"]
    assert wrong["correct"] is False, wrong["execution"]
    assert correct["metric"] == wrong["metric"] == "pass@1"
