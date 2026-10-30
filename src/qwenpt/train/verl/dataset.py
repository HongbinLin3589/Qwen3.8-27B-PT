"""Compatibility import for qwenpt.train.common.dataset."""

import sys
from qwenpt.train.common import dataset as _implementation

sys.modules[__name__] = _implementation
