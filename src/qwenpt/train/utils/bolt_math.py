"""Compatibility import for qwenpt.train.bolt.math."""

import sys
from qwenpt.train.bolt import math as _implementation

sys.modules[__name__] = _implementation
