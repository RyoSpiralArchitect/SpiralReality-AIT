"""Bridge to the compiled C numeric helpers."""

from __future__ import annotations

from typing import Any, Callable

try:
    import spiral_numeric_cpp as _cpp_api
except ImportError as exc:  # pragma: no cover - import-time failure should surface early
    raise RuntimeError(
        "The spiral_numeric_cpp extension is missing. Build it via "
        "`python native/cpp/setup_spiral_numeric_cpp.py build_ext --inplace` before importing."
    ) from exc


def is_available() -> bool:
    return True


def _delegate(name: str, *args: Any) -> Any:
    func: Callable[..., Any] = getattr(_cpp_api, name)
    return func(*args)


def matmul(a, b):
    return _delegate("matmul", a, b)


def dot(a, b):
    return _delegate("dot", a, b)


def mean(data, axis, keepdims):
    return _delegate("mean", data, axis, keepdims)


def std(data, axis, ddof, keepdims):
    return _delegate("std", data, axis, ddof, keepdims)


def var(data, axis, ddof, keepdims):
    return _delegate("var", data, axis, ddof, keepdims)


def sum(data, axis, keepdims):
    return _delegate("sum", data, axis, keepdims)


def min(data, axis, keepdims):
    return _delegate("min", data, axis, keepdims)


def max(data, axis, keepdims):
    return _delegate("max", data, axis, keepdims)


def maximum(a, b):
    return _delegate("maximum", a, b)


def minimum(a, b):
    return _delegate("minimum", a, b)


def tanh(data):
    return _delegate("tanh", data)


def exp(data):
    return _delegate("exp", data)


def log(data):
    return _delegate("log", data)


def logaddexp(a, b):
    return _delegate("logaddexp", a, b)


def median(data, axis=None):
    return _delegate("median", data, axis)


def abs(data):
    return _delegate("abs", data)


def clip(data, lo, hi):
    return _delegate("clip", data, lo, hi)


def sqrt(data):
    return _delegate("sqrt", data)


def diff(data, order: int = 1):
    return _delegate("diff", data, order)


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


def linalg_solve(coeffs, rhs):
    return _delegate("linalg_solve", coeffs, rhs)


def linalg_cholesky(data):
    return _delegate("linalg_cholesky", data)

