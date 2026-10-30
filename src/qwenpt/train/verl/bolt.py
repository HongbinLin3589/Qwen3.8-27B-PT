"""Compatibility import for qwenpt.train.bolt.trainer."""

import sys
from qwenpt.train.bolt import trainer as _implementation

sys.modules[__name__] = _implementation
