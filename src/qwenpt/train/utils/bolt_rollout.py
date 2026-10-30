"""Compatibility import and CLI for qwenpt.train.bolt.rollout."""

import sys
from qwenpt.train.bolt import rollout as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
