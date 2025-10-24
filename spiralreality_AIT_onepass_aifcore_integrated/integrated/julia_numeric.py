"""Julia-backed helpers for accelerating the ``np_stub`` operations."""

from __future__ import annotations

import pathlib
from numbers import Integral
from typing import Any

try:  # pragma: no cover - optional dependency
    from juliacall import Main as jl
except Exception:  # pragma: no cover - optional dependency missing
    jl = None  # type: ignore[assignment]

_MODULE = None
if jl is not None:  # pragma: no cover - optional runtime path
    module_path = pathlib.Path(__file__).resolve().parents[2] / "native" / "julia" / "SpiralNumericJulia.jl"
    if module_path.exists():
        try:
            jl.include(str(module_path))
            _MODULE = jl.SpiralNumericJulia
        except Exception:
            _MODULE = None


def is_available() -> bool:
    """Return ``True`` when the Julia numeric module can be used."""

    return _MODULE is not None


def _to_python(value: Any):
    """Convert Julia values to plain Python containers."""

    if isinstance(value, float):
        return float(value)
    if isinstance(value, complex):
        return complex(value)
    if isinstance(value, Integral):
        return int(value)
    if hasattr(value, "tolist"):
        try:
            return _to_python(value.tolist())
        except Exception:
            pass
    try:
        return float(value)
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        try:
            return [_to_python(v) for v in value]
        except Exception:
            pass
    return value


def _axis_arg(axis: int | None):
    if jl is None:
        return axis
    if axis is None:
        return jl.nothing
    return int(axis)


def _maybe_nothing(value):
    if jl is None:
        return value
    if value is None:
        return jl.nothing
    return value


def matmul(a, b):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    result = _MODULE.matmul(a, b)
    return _to_python(result)


def dot(a, b):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    return _to_python(_MODULE.dot(a, b))


def flash_attention(q, k, v, scale, bias, block_size, return_weights):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    result = _MODULE.flash_attention(q, k, v, float(scale), _maybe_nothing(bias), int(block_size), bool(return_weights))
    if return_weights:
        context, weights = result
        return _to_python(context), _to_python(weights)
    return _to_python(result)


def mean(data, axis, keepdims):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    result = _MODULE.mean_reduce(data, _axis_arg(axis), bool(keepdims))
    return _to_python(result)


def std(data, axis, ddof, keepdims):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    result = _MODULE.std_reduce(data, _axis_arg(axis), int(ddof), bool(keepdims))
    return _to_python(result)


def var(data, axis, ddof, keepdims):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    result = _MODULE.var_reduce(data, _axis_arg(axis), int(ddof), bool(keepdims))
    return _to_python(result)


def sum(data, axis, keepdims):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    result = _MODULE.sum_reduce(data, _axis_arg(axis), bool(keepdims))
    return _to_python(result)


def min(data, axis, keepdims):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    result = _MODULE.min_reduce(data, _axis_arg(axis), bool(keepdims))
    return _to_python(result)


def max(data, axis, keepdims):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    result = _MODULE.max_reduce(data, _axis_arg(axis), bool(keepdims))
    return _to_python(result)


def maximum(a, b):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    return _to_python(_MODULE.maximum_map(a, b))


def minimum(a, b):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    return _to_python(_MODULE.minimum_map(a, b))


def tanh(data):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    return _to_python(_MODULE.tanh_map(data))


def exp(data):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    return _to_python(_MODULE.exp_map(data))


def log(data):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    return _to_python(_MODULE.log_map(data))


def logaddexp(a, b):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    return _to_python(_MODULE.logaddexp_map(a, b))


def median(data, axis=None):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    return _to_python(_MODULE.median_all(data, _axis_arg(axis)))


def abs(data):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    return _to_python(_MODULE.abs_map(data))


def clip(data, lo, hi):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    return _to_python(_MODULE.clip_map(data, float(lo), float(hi)))


def sqrt(data):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    return _to_python(_MODULE.sqrt_map(data))


def diff(data, order: int = 1):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    return _to_python(_MODULE.diff_vec(data, int(order)))


def argsort(data):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    return _to_python(_MODULE.argsort_indices(data))


def argmax(data):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    return _to_python(_MODULE.argmax_index(data))


def trace(data):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    return _to_python(_MODULE.trace_value(data))


def linalg_norm(data):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    return _to_python(_MODULE.norm_value(data))


def linalg_inv(data):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    return _to_python(_MODULE.inv_matrix(data))


def linalg_slogdet(data):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    result = _MODULE.slogdet_pair(data)
    converted = _to_python(result)
    if isinstance(converted, (list, tuple)) and len(converted) == 2:
        return converted[0], converted[1]
    return converted


def linalg_solve(coeffs, rhs):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    return _to_python(_MODULE.solve_matrix(coeffs, rhs))


def linalg_cholesky(data):
    if not is_available():
        raise RuntimeError("Julia numeric backend unavailable")
    return _to_python(_MODULE.cholesky_lower(data))

