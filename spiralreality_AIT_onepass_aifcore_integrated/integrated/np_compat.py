"""Compat layer that provides a `numpy`-like module.

The real project prefers to use NumPy for convenience, but the evaluation
environment used for these kata style exercises does not always ship the
`numpy` dependency.  Importing it at module import time would therefore raise a
`ModuleNotFoundError` and prevent the demo from running at all.

To keep the rest of the codebase unchanged we try to import real NumPy first
and, if that fails, fall back to a very small, pure Python substitute that
implements only the handful of operations the project relies on.  The stub is
*not* a drop-in replacement for the full library, yet it mirrors the
functionality we exercise in the demo and unit tests which keeps the example
self-contained and easy to run.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised implicitly when numpy is available.
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - deterministic in kata env.
    from . import np_stub as np  # noqa: F401

__all__ = ["np"]
