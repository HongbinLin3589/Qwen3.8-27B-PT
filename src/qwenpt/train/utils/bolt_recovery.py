"""Compatibility import and CLI for qwenpt.train.bolt.recovery."""

import sys
from qwenpt.train.bolt import recovery as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
