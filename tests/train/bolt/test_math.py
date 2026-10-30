"""Check offline math labels and preservation of generated Qwen reasoning."""

import importlib.util

import pytest

from qwenpt.train.bolt.math import parse_reference, score_math, split_completion


requires_verifier = pytest.mark.skipif(
    importlib.util.find_spec("math_verify") is None,
    reason="math_verify is available in the offline generation environment",
)


def test_split_raw_thinking_continuation_preserves_whitespace():
    assert split_completion("  Work\n\n</think>\n\nAnswer  ") == {
        "reasoning_content": "  Work\n\n", "content": "\n\nAnswer  ",
    }


def test_split_explicit_thinking_opening():
    assert split_completion("<think>\nWork</think>\nAnswer") == {
        "reasoning_content": "\nWork", "content": "\nAnswer",
    }


def test_truncated_thinking_has_no_fabricated_final_answer():
    assert split_completion(r"Still reasoning: \boxed{42}") == {
        "reasoning_content": r"Still reasoning: \boxed{42}", "content": "",
    }


def test_non_thinking_content_is_unchanged():
    text = "  Final answer\n"
    assert split_completion(text, enable_thinking=False) == {"content": text, "reasoning_content": ""}


@requires_verifier
@pytest.mark.parametrize("content,reference,reward", [
    (r"The answer is \boxed{42}.", "42", 1.0),
    (r"The answer is \boxed{41}.", "42", 0.0),
    (r"\boxed{\frac{1}{2}}", "0.5", 1.0),
    (r"<answer>\frac{1}{2}</answer>", "0.5", 1.0),
    (r"\boxed{\frac{1}{2}}", r"<answer>\frac{1}{2}</answer>", 1.0),
    (r"\boxed{f(x)=\sqrt[3]{x}}", r"f(x) = \sqrt[3]{x}", 1.0),
    ("No final answer.", "42", 0.0),
    ("", "42", 0.0),
])
def test_math_accuracy_matches_apex(content, reference, reward):
    assert score_math(content, reference) == reward


@requires_verifier
def test_boxed_reasoning_is_not_used_as_the_answer():
    completion = split_completion(r"An intermediate result is \boxed{42}.</think>\boxed{41}")
    assert score_math(completion["content"], "42") == 0.0
    truncated = split_completion(r"An intermediate result is \boxed{42}.")
    assert score_math(truncated["content"], "42") == 0.0


@requires_verifier
def test_invalid_reference_raises_even_for_missing_final_answer():
    with pytest.raises(ValueError, match="could not parse ground_truth"):
        score_math("", r"<answer></answer>")


@requires_verifier
def test_reference_cache_results_are_independent_copies():
    first = parse_reference("42")
    second = parse_reference("42")
    assert first == second
    assert first is not second
    first.clear()
    assert parse_reference("42") == second


@pytest.mark.parametrize("reference", ["", "  ", None, 42])
def test_empty_or_non_string_references_raise(reference):
    with pytest.raises(ValueError, match="ground_truth must be a nonempty string"):
        parse_reference(reference)
