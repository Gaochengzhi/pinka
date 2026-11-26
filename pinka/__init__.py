"""pinka: pack GPU jobs onto shared cards by measured memory. A single-node job pool with VRAM, CPU and RAM booking,
dependencies and retries. See AGENTS.md."""
from .batch import submit_list  # noqa: F401
from .jobs import Spec, cancel, fanout, retarget, submit, wait  # noqa: F401

__version__ = "0.2.0"
