"""High-level numeric helpers with optional pure Python fall-back.

The helpers prefer to use the real :mod:`numpy` implementation when it is
available while keeping a compatibility layer that mirrors the legacy pure
Python stub.  Environments that cannot import NumPy—or that explicitly request
the lightweight shim via ``SPIRAL_NUMERIC_FORCE_STUB``—automatically fall back
to the Python implementation.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any

SAFE_EXP_CLIP = float(os.getenv("SPIRAL_SAFE_EXP_CLIP", "700.0"))

_FORCE_PURE = os.getenv("SPIRAL_NUMERIC_FORCE_STUB", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "python",
}

try:  # pragma: no cover - the import path is exercised in numpy environments
    import numpy as _np
    from numpy.typing import ArrayLike, NDArray
except Exception:  # pragma: no cover - exercised in numpy-less environments
    _np = None
    ArrayLike = Any  # type: ignore[misc,assignment]
    NDArray = Any  # type: ignore[misc,assignment]

if _np is None or _FORCE_PURE:  # pragma: no cover - covered in dedicated tests
    from . import _np_stub_purepy as _backend_module  # type: ignore
else:  # pragma: no cover - primary branch exercised via unit tests
    from . import _np_stub_numpy as _backend_module  # type: ignore

for _name in dir(_backend_module):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_backend_module, _name)

SAFE_EXP_CLIP = float(os.getenv("SPIRAL_SAFE_EXP_CLIP", "700.0"))

_current_module = sys.modules[__name__]


class _StubProxy(types.ModuleType):
    def __setattr__(self, name, value):  # pragma: no cover - exercised in tests
        super().__setattr__(name, value)
        if hasattr(_backend_module, name):
            setattr(_backend_module, name, value)

    def __getattr__(self, name):  # pragma: no cover - passthrough helper
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(_backend_module, name)


_proxy = _StubProxy(__name__)
_proxy.__dict__.update(_current_module.__dict__)
sys.modules[__name__] = _proxy
