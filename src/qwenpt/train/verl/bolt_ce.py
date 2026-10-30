"""Compatibility import for qwenpt.train.bolt.ce."""

import sys
from qwenpt.train.bolt import ce as _implementation

sys.modules[__name__] = _implementation
