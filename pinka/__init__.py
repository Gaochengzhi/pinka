"""pinka: pack GPU jobs onto shared cards by measured memory. A single-node job pool with VRAM, CPU and RAM booking,
dependencies and retries. See README.md."""
from .jobs import Spec, cancel, fanout, retarget, submit, wait  # noqa: F401

__version__ = "0.1.0"
