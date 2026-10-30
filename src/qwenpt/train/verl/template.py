"""Compatibility import for qwenpt.train.common.template."""

import sys
from qwenpt.train.common import template as _implementation

sys.modules[__name__] = _implementation
