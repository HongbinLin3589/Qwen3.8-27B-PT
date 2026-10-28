"""Check answer isolation and avg@32 coverage independently of GPU generation."""

import pytest

from legacy.benchmark.scoring import last_boxed, score_completion
from legacy.benchmark.summarize import summarize_task


def test_reasoning_is_never_graded():
    assert not score_completion("Answer: B", "gpqa_diamond", "B")["correct"]
    assert not score_completion(r"\boxed{42}", "aime2024", "42")["correct"]
    result = score_completion("Answer: B</think>Answer: C", "gpqa_diamond", "B")
    assert result["predicted"] == "C" and not result["correct"]


def test_gpqa_follows_explicit_answer_format():
    assert score_completion("thought</think>Answer: A", "gpqa_diamond", "A")["correct"]
    assert score_completion("</think>Answer : $A$", "gpqa_diamond", "A")["correct"]
    assert score_completion("</think>Answer: (A)", "gpqa_diamond", "A")["format_error"]
    assert score_completion("</think>Answer: AB", "gpqa_diamond", "A")["format_error"]


def test_aime_last_boxed_math_equivalence():
    assert last_boxed(r"\boxed{7} then \boxed{\frac{84}{2}}") == r"\frac{84}{2}"
    assert score_completion(r"</think>\boxed{042}", "aime2025", "42")["correct"]
    assert score_completion(r"</think>\boxed{\frac{84}{2}}", "aime2026", "42")["correct"]
    assert not score_completion(r"</think>\boxed{42} finally \boxed{43}", "aime2024", "42")["correct"]
    assert score_completion(r"</think>\boxed{42", "aime2024", "42")["format_error"]


def test_avg32_counts_samples_not_any_success_or_majority():
    bank = [{"benchmark": "aime2024", "question_id": "q1", "answer": "42", "samples": 32}]
    candidates = [{"sample_index": i, "correct": i == 0, "generated_tokens": 10,
                   "finish_reason": "stop", "format_error": False, "thinking_complete": True}
                  for i in range(32)]
    group = {"benchmark": "aime2024", "question_id": "q1", "answer": "42",
             "variant": "original", "run_fingerprint": "same", "samples": candidates}
    result = summarize_task(bank, [group], "original", "aime2024")
    assert result["complete"] and result["accuracy"] == 1 / 32
    group["samples"] = candidates[:-1]
    with pytest.raises(ValueError, match="Missing or duplicate"):
        summarize_task(bank, [group], "original", "aime2024")


def test_duplicate_questions_and_missing_coverage():
    bank = [{"benchmark": "gpqa_diamond", "question_id": "q1", "answer": "A", "samples": 1}]
    assert not summarize_task(bank, [], "original", "gpqa_diamond")["complete"]
    group = {"benchmark": "gpqa_diamond", "question_id": "q1", "answer": "A", "variant": "original",
             "run_fingerprint": "same", "samples": [{"sample_index": 0, "correct": True,
             "generated_tokens": 10, "finish_reason": "stop", "format_error": False, "thinking_complete": True}]}
    with pytest.raises(ValueError, match="duplicate"):
        summarize_task(bank, [group, group], "original", "gpqa_diamond")
