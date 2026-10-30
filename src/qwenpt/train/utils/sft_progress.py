"""Compatibility import and CLI for qwenpt.train.sft.progress."""

import sys
from qwenpt.train.sft import progress as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
