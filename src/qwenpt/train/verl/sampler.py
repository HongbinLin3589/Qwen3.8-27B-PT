"""Compatibility import for qwenpt.train.sft.sampler."""

import sys
from qwenpt.train.sft import sampler as _implementation

sys.modules[__name__] = _implementation
