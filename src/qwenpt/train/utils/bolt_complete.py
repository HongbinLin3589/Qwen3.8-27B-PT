"""Compatibility import for qwenpt.train.bolt.complete."""

import sys
from qwenpt.train.bolt import complete as _implementation

sys.modules[__name__] = _implementation
