"""Grade GPQA letters and AIME boxed answers outside Qwen's thinking text."""

import re


def last_boxed(text: str) -> str | None:
    """Return the last complete boxed expression, including nested LaTeX braces."""
    result = None
    for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        depth = 1
        start = match.end()
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    result = text[start:index].strip()
                    break
    return result


def score_completion(text: str, benchmark: str, answer: str) -> dict:
    """Grade a generated continuation against its gold answer without a model judge.

    The generation prompt already opens thinking, so a missing closing tag
    leaves no final answer. AIME mathematical equivalence uses the existing
    verified math grader, restricted to the last boxed expression.
    """
    _, closing, final = text.rpartition("</think>")
    result = {"predicted": None, "correct": False,
              "thinking_complete": bool(closing), "format_error": True}
    if not closing:
        return result
    if benchmark == "gpqa_diamond":
        if answer not in "ABCD" or len(answer) != 1:
            raise ValueError("GPQA gold answer must be A, B, C or D")
        matches = re.findall(r"\bAnswer[ \t]*:[ \t]*\$?([A-D])\b\$?", final, flags=re.IGNORECASE)
        if matches:
            result.update(predicted=matches[-1].upper(), format_error=False)
            result["correct"] = result["predicted"] == answer
    elif benchmark in ("aime2024", "aime2025", "aime2026"):
        if not answer.isdigit() or not 0 <= int(answer) <= 999:
            raise ValueError("AIME gold answer must be an integer from 0 to 999")
        expression = last_boxed(final)
        if expression:
            from qwenpt.train.utils.bolt_math import score_math

            prediction = str(int(expression)) if expression.isdigit() else expression
            result.update(predicted=prediction, format_error=False)
            result["correct"] = bool(score_math(r"\boxed{" + expression + "}", answer))
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return result
