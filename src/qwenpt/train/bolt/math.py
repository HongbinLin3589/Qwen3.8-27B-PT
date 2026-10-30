"""Split Qwen reasoning and grade offline math answers with APEX's verifier."""

from copy import deepcopy
from functools import lru_cache
import re
from typing import Any


def math_answer_environment(answer: str) -> str:
    """Return an answer with explicit math delimiters, preserving existing ones."""
    answer = re.sub(r"\$\s+\$", "", answer.strip())
    if not answer or re.search(r"\\(?:boxed|fbox)\s*\{|(?<!\\)\$|\\\(|\\\[", answer):
        return answer
    return r"\boxed{" + answer + "}"


@lru_cache(maxsize=8192)
def cached_reference(ground_truth: str) -> tuple[Any, ...]:
    """Cache the parsed reference tuple, raising when no answer can be extracted."""
    from math_verify import parse

    match = re.search(r"<answer>(.*?)</answer>", ground_truth, re.DOTALL)
    answer = match.group(1) if match else ground_truth
    parsed = parse(math_answer_environment(answer), extraction_mode="first_match", parsing_timeout=5)
    if not parsed:
        raise ValueError(f"Math verifier could not parse ground_truth: {ground_truth!r}")
    return tuple(parsed)


def parse_reference(ground_truth: str) -> list[Any]:
    """Parse a nonempty reference answer, returning a copy of the cached result.

    Invalid references raise instead of silently becoming negative labels.
    Verifier dependencies are imported only when grading is requested.
    """
    if not isinstance(ground_truth, str) or not ground_truth.strip():
        raise ValueError("ground_truth must be a nonempty string")
    return deepcopy(list(cached_reference(ground_truth)))


def split_completion(text: str, enable_thinking: bool = True) -> dict[str, str]:
    """Return assistant content and reasoning from a Qwen generation continuation.

    The thinking generation prompt already contains the opening tag; a missing
    closing tag therefore leaves all generated text in reasoning_content.
    """
    if not enable_thinking:
        return {"content": text, "reasoning_content": ""}
    if text.startswith("<think>"):
        text = text[len("<think>"):]
    reasoning, closing, content = text.partition("</think>")
    return {"content": content if closing else "", "reasoning_content": reasoning}


def score_math(content: str, ground_truth: str) -> float:
    """Return APEX's binary math accuracy for final answer text and a reference.

    Pass only final content, excluding reasoning; an empty final answer scores
    zero. Reference parse failures raise, while verification failures score zero.
    """
    gold = parse_reference(ground_truth)
    if not content.strip():
        return 0.0
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify

    match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    if match:
        prediction = parse(
            math_answer_environment(match.group(1)), extraction_mode="first_match", parsing_timeout=5,
        )
    else:
        prediction = parse(
            content,
            extraction_config=[LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False, malformed_operators=False, basic_latex=True, boxed=True, units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )],
            extraction_mode="first_match",
            parsing_timeout=5,
        )
    try:
        return float(verify(gold, prediction, timeout_seconds=10))
    except Exception:
        return 0.0
