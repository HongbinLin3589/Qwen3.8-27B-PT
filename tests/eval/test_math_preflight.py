"""Verify the math scorer's actual supported host boundary before API work."""
import os

import pytest

from qwenpt.eval.suite import preflight_scorers


def test_math_preflight_exercises_linux_runtime_or_blocks_windows_explicitly():
    config = {"benchmarks": {"math": {"scorer": {"type": "math_equivalence"}}}}
    if os.name == "nt":
        with pytest.raises(RuntimeError, match="requires the supported Linux runtime"):
            preflight_scorers(config)
    else:
        result = preflight_scorers(config)
        assert result["math"]["runtime_probe"] == "passed"
        assert set(result["math"]["dependencies"]) == {"math-verify", "latex2sympy2-extended"}
