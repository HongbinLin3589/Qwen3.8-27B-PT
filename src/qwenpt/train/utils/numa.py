"""Compatibility import for qwenpt.train.common.numa."""

import sys
from qwenpt.train.common import numa as _implementation

sys.modules[__name__] = _implementation
