"""Compatibility import and CLI for qwenpt.train.bolt.train_verify."""

import sys
from qwenpt.train.bolt import train_verify as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
