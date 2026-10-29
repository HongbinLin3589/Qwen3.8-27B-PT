"""Opt-in integration with the real official grader, inside a Docker container."""
import os

import pytest

from qwenpt.eval.livecodebench import normalize_livecodebench, score_livecodebench


@pytest.mark.skipif(not os.environ.get("LCB_CHECKOUT") or not os.environ.get("LCB_SANDBOX_IMAGE"),
                    reason="Requires pinned official LCB checkout and built Docker grading image")
def test_official_execution_grader_distinguishes_correct_code_from_wrong_code():
    question = normalize_livecodebench([{
        "question_id": "synthetic-sum-contract", "question_content": "Print the sum of two integers.",
        "starter_code": "", "metadata": {}, "difficulty": "easy", "platform": "synthetic",
        "public_test_cases": [{"input": "2 3\n", "output": "5\n", "testtype": "stdin"}],
        "private_test_cases": [{"input": "8 -2\n", "output": "6\n", "testtype": "stdin"}],
    }])[0]
    scorer = {"checkout": os.environ["LCB_CHECKOUT"], "revision": "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"}
    correct = score_livecodebench(question, "```python\nprint(sum(map(int,input().split())))\n```", scorer)
    wrong = score_livecodebench(question, "```python\nprint(0)\n```", scorer)
    assert correct["correct"] is True
    assert wrong["correct"] is False
    assert correct["metric"] == wrong["metric"] == "pass@1"
