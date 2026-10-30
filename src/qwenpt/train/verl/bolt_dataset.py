"""Compatibility import for qwenpt.train.bolt.dataset."""

import sys
from qwenpt.train.bolt import dataset as _implementation

sys.modules[__name__] = _implementation
