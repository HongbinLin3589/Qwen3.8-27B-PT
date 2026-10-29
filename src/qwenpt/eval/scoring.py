"""Shared final-answer extraction helpers for API benchmark scorers."""

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
