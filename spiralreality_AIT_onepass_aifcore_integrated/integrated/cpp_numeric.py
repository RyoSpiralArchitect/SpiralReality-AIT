"""Placeholder for optional C++ numeric accelerators."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional compiled module
    from spiral_numeric_cpp import api as _cpp_api  # type: ignore
except Exception:  # pragma: no cover - extension missing
    _cpp_api = None


def is_available() -> bool:
    """Return ``True`` when a compiled numeric backend is present."""

    return _cpp_api is not None


# When a compiled backend becomes available it should expose operations mirroring
# the ones used in :mod:`np_stub`.  The Python wrapper intentionally keeps the
# interface small; we only delegate calls when the extension implements the
# corresponding attribute.  This keeps the pure Python fallback resilient when
# the compiled module is absent.


def _delegate(name: str, *args: Any, **kwargs: Any):
    if _cpp_api is None:
        raise RuntimeError("C++ numeric backend is not available")
    func = getattr(_cpp_api, name, None)
    if func is None:
        raise AttributeError(f"C++ backend missing {name}")
    return func(*args, **kwargs)


def matmul(a, b):
    return _delegate("matmul", a, b)


def dot(a, b):
    return _delegate("dot", a, b)


def mean(data, axis):
    return _delegate("mean", data, axis)


def std(data, axis):
    return _delegate("std", data, axis)


def sum(data, axis, keepdims):
    return _delegate("sum", data, axis, keepdims)


def tanh(data):
    return _delegate("tanh", data)


def exp(data):
    return _delegate("exp", data)


def log(data):
    return _delegate("log", data)


def logaddexp(a, b):
    return _delegate("logaddexp", a, b)


def median(data, _unused=None):
    return _delegate("median", data)


def abs(data):
    return _delegate("abs", data)


def clip(data, lo, hi):
    return _delegate("clip", data, lo, hi)


def sqrt(data):
    return _delegate("sqrt", data)


def diff(data):
    return _delegate("diff", data)


def argsort(data):
    return _delegate("argsort", data)


def argmax(data):
    return _delegate("argmax", data)


def trace(data):
    return _delegate("trace", data)


def linalg_norm(data):
    return _delegate("linalg_norm", data)


def linalg_inv(data):
    return _delegate("linalg_inv", data)


def linalg_slogdet(data):
    return _delegate("linalg_slogdet", data)

