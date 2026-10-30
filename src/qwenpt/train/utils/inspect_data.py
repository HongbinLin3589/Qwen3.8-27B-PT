"""Compatibility import and CLI for qwenpt.train.sft.inspect_data."""

import sys
from qwenpt.train.sft import inspect_data as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
