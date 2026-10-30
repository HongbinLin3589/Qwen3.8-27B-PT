"""Compatibility import and CLI for qwenpt.train.sft.split_data."""

import sys
from qwenpt.train.sft import split_data as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
