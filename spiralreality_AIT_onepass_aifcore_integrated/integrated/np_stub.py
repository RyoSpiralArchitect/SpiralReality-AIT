"""High-level numeric helpers with optional pure Python fall-back.

This shim selects between the NumPy-backed implementation and the legacy
pure-Python stub at import time.  The NumPy path exposes optional Julia and
C++ accelerators while the pure-Python path keeps a dependency-free fallback
for constrained environments.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from typing import Iterable

_PACKAGE = __name__.rsplit(".", 1)[0]


def _truthy(value: str) -> bool:
    value = value.strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def _select_stub_module() -> str:
    """Return the module suffix for the active backend."""

    force_stub = os.getenv("SPIRAL_NUMERIC_FORCE_STUB", "")
    backend_pref = os.getenv("SPIRAL_NUMERIC_BACKEND", "auto").strip().lower()

    if force_stub and _truthy(force_stub):
        return "_np_stub_purepy"
    if backend_pref in {"python", "purepy", "stub"}:
        return "_np_stub_purepy"
    return "_np_stub_numpy"


_backend_module: types.ModuleType
_backend_error: Exception | None = None

_backend_name = _select_stub_module()
try:
    _backend_module = importlib.import_module(f".{_backend_name}", _PACKAGE)
except Exception as exc:  # pragma: no cover - defensive fallback when NumPy import fails
    _backend_error = exc
    _backend_module = importlib.import_module("._np_stub_purepy", _PACKAGE)
    _backend_name = "_np_stub_purepy"


class _StubProxy(types.ModuleType):
    """Proxy that forwards attribute access to the selected backend module."""

    __slots__: Iterable[str] = ()

    def __getattr__(self, name: str):  # pragma: no cover - trivial delegation
        return getattr(_backend_module, name)

    def __setattr__(self, name: str, value) -> None:  # pragma: no cover - exercised in tests
        if name.startswith("_PROXY_"):
            super().__setattr__(name, value)
            return
        setattr(_backend_module, name, value)
        super().__setattr__(name, value)

    def __dir__(self) -> list[str]:  # pragma: no cover - trivial delegation
        return sorted(set(super().__dir__()) | set(dir(_backend_module)))


_proxy = _StubProxy(__name__)
_proxy.__dict__.update(_backend_module.__dict__)
object.__setattr__(_proxy, "__doc__", __doc__)
object.__setattr__(_proxy, "_PROXY_BACKEND_NAME", _backend_name)
object.__setattr__(_proxy, "_PROXY_BACKEND_MODULE", _backend_module)
object.__setattr__(
    _proxy,
    "_PROXY_BACKEND_ERROR",
    _backend_error,
)
object.__setattr__(
    _proxy,
    "NUMERIC_BACKEND",
    getattr(_backend_module, "NUMERIC_BACKEND", "numpy"),
)
object.__setattr__(
    _proxy,
    "IS_PURE_PY",
    getattr(_backend_module, "IS_PURE_PY", False),
)
if hasattr(_backend_module, "STRICT_BACKEND"):
    object.__setattr__(_proxy, "STRICT_BACKEND", getattr(_backend_module, "STRICT_BACKEND"))
if hasattr(_backend_module, "_STRICT_BACKEND"):
    object.__setattr__(_proxy, "_STRICT_BACKEND", getattr(_backend_module, "_STRICT_BACKEND"))

sys.modules[__name__] = _proxy
