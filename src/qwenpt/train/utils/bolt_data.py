"""Compatibility import for qwenpt.train.bolt.data."""

import sys
from qwenpt.train.bolt import data as _implementation

sys.modules[__name__] = _implementation
