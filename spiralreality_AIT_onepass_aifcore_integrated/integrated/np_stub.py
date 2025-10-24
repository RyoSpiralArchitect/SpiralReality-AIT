"""High-level numeric helpers with optional pure Python fall-back.

The helpers prefer to use the real :mod:`numpy` implementation when it is
available while keeping a compatibility layer that mirrors the legacy pure
Python stub.  Environments that cannot import NumPy—or that explicitly request
the lightweight shim via ``SPIRAL_NUMERIC_FORCE_STUB``—automatically fall back
to the Python implementation.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, Sequence, Tuple

import numpy as _np
from numpy.typing import ArrayLike, NDArray

try:  # pragma: no cover - optional Julia acceleration hook
    from . import julia_numeric as _julia_numeric  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    _julia_numeric = None

try:  # pragma: no cover - optional C++ acceleration hook
    from . import cpp_numeric as _cpp_numeric  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    _cpp_numeric = None


_Array = NDArray[Any]


def _resolve_dtype(dtype: Any) -> _np.dtype[Any] | None:
    if dtype in {None, "infer"}:
        return None
    if dtype in {float, "float", "float32", "float64"}:
        return _np.dtype(_np.float64)
    if dtype in {int, "int", "int32", "int64"}:
        return _np.dtype(_np.int64)
    if dtype in {bool, "bool"}:
        return _np.dtype(_np.bool_)
    return _np.dtype(dtype)


class _StubProxy(types.ModuleType):
    def __setattr__(self, name, value):  # pragma: no cover - exercised in tests
        super().__setattr__(name, value)
        if hasattr(_backend_module, name):
            setattr(_backend_module, name, value)

    for name in order:
        if name == "julia" and _julia_numeric is not None:
            try:
                if _julia_numeric.is_available():
                    return "julia", _julia_numeric
            except Exception:
                continue
        if name == "cpp" and _cpp_numeric is not None:
            try:
                if _cpp_numeric.is_available():
                    return "cpp", _cpp_numeric
            except Exception:
                continue
        if name == "numpy":
            return "numpy", None
    return "numpy", None


_BACKEND_PREF = os.getenv("SPIRAL_NUMERIC_BACKEND", "auto").strip().lower()
_BACKEND_NAME, _ACCEL_BACKEND = _select_backend(_BACKEND_PREF)
NUMERIC_BACKEND = _BACKEND_NAME
IS_PURE_PY = False

SAFE_EXP_MAX = 700.0
SAFE_EXP_MIN = -700.0
_INV_ABS_TOL = 1e-12
_INV_REL_TOL = 1e-9


def _to_backend_arg(value: Any) -> Any:
    if isinstance(value, ndarray):
        return value.to_list()
    if isinstance(value, _np.ndarray):
        return value.tolist()
    return value


def _backend_call(name: str, *args, **kwargs):
    if _ACCEL_BACKEND is None:
        return None
    func = getattr(_ACCEL_BACKEND, name, None)
    if func is None:
        return None
    converted_args = [_to_backend_arg(arg) for arg in args]
    converted_kwargs = {key: _to_backend_arg(val) for key, val in kwargs.items()}
    try:
        return func(*converted_args, **converted_kwargs)
    except Exception:
        return None


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, _np.generic):
        return value.item()
    return value


def _array_from_backend(value: Any):
    if value is None:
        return None
    if isinstance(value, ndarray):
        return value
    if isinstance(value, _np.ndarray):
        return ndarray(value)
    if isinstance(value, (list, tuple)):
        return ndarray(value)
    if hasattr(value, "tolist"):
        try:
            return ndarray(value.tolist())
        except Exception:
            pass
    if _np.isscalar(value):
        return _to_python_scalar(value)
    return None


def _wrap_stat_result(value: Any):
    if _np.isscalar(value):
        return _to_python_scalar(value)
    return ndarray(value)


def _coerce_operand(value: Any) -> Any:
    if isinstance(value, ndarray):
        return value._array
    return value


def _shape_to_text(shape: Sequence[int]) -> str:
    return "×".join(str(dim) for dim in shape)


def _gauss_jordan_inverse(matrix: _Array) -> _np.ndarray:
    arr = _np.asarray(matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("inv expects a square 2D array; got shape %s" % (_shape_to_text(arr.shape),))
    n = arr.shape[0]
    if n == 0:
        return _np.empty((0, 0), dtype=arr.dtype if arr.dtype.kind in {"f", "c"} else _np.float64)
    work_dtype = (
        _np.result_type(arr.dtype, _np.complex128)
        if arr.dtype.kind == "c"
        else _np.result_type(arr.dtype, _np.float64)
    )
    working = arr.astype(work_dtype, copy=True)
    # Scale rows by their maximum to improve conditioning before pivoting.
    scale = _np.max(_np.abs(working), axis=1)
    scale[scale == 0.0] = 1.0
    aug = _np.hstack([working / scale[:, None], _np.eye(n, dtype=work_dtype)])
    for col in range(n):
        pivot_idx = col + int(_np.argmax(_np.abs(aug[col:, col])))
        pivot_val = aug[pivot_idx, col]
        norm = float(_np.max(_np.abs(aug[:, col]))) if aug.size else 0.0
        tol = _INV_ABS_TOL + _INV_REL_TOL * max(1.0, norm)
        if abs(pivot_val) <= tol:
            raise _np.linalg.LinAlgError("Singular matrix: pivot %.3e at column %d" % (pivot_val, col))
        if pivot_idx != col:
            aug[[col, pivot_idx]] = aug[[pivot_idx, col]]
        aug[col] = aug[col] / pivot_val
        for row in range(n):
            if row == col:
                continue
            factor = aug[row, col]
            if factor != 0.0:
                aug[row] -= factor * aug[col]
    inv = aug[:, n:]
    inv = inv / scale[None, :]
    if arr.dtype.kind in {"f", "c"}:
        target_dtype = arr.dtype
    else:
        target_dtype = work_dtype
    return inv.astype(target_dtype, copy=False)


def _dot_validate(a_arr: _Array, b_arr: _Array) -> None:
    if a_arr.ndim not in (1, 2) or b_arr.ndim not in (1, 2):
        raise ValueError(
            "dot supports 1D or 2D operands; got %s and %s"
            % (_shape_to_text(a_arr.shape), _shape_to_text(b_arr.shape))
        )
    if a_arr.ndim == 1 and b_arr.ndim == 1 and a_arr.shape[0] != b_arr.shape[0]:
        raise ValueError(
            "dot expects aligned vectors; got lengths %d and %d" % (a_arr.shape[0], b_arr.shape[0])
        )
    if a_arr.ndim == 2 and a_arr.shape[1] != b_arr.shape[0]:
        raise ValueError(
            "dot expects shapes %s and %s to align" % (_shape_to_text(a_arr.shape), _shape_to_text(b_arr.shape))
        )
    if a_arr.ndim == 1 and b_arr.ndim == 2 and a_arr.shape[0] != b_arr.shape[0]:
        raise ValueError(
            "dot expects shapes %s and %s to align" % (_shape_to_text(a_arr.shape), _shape_to_text(b_arr.shape))
        )


def _matmul_dispatch(a_arr: _Array, b_arr: _Array):
    _dot_validate(a_arr, b_arr)
    result = _np.matmul(a_arr, b_arr)
    if _np.isscalar(result):
        return _to_python_scalar(result)
    return ndarray(result)


class ndarray:
    """Lightweight proxy that keeps NumPy arrays in line with the old stub API."""

    __slots__ = ("_array",)

    def __init__(self, data: ArrayLike, dtype: Any | None = None, copy: bool = False):
        resolved = _resolve_dtype(dtype)
        if copy:
            if resolved is None:
                self._array = _np.array(data, copy=True)
            else:
                self._array = _np.array(data, dtype=resolved, copy=True)
        else:
            if resolved is None:
                self._array = _np.asarray(data)
            else:
                self._array = _np.asarray(data, dtype=resolved)

    # ------------------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(int(dim) for dim in self._array.shape)

    @property
    def ndim(self) -> int:
        return int(self._array.ndim)

    @property
    def dtype(self) -> _np.dtype[Any]:
        return self._array.dtype

    def copy(self) -> "ndarray":
        return ndarray(self._array.copy())

    def astype(self, dtype: Any) -> "ndarray":
        resolved = _resolve_dtype(dtype)
        if resolved is None:
            return ndarray(self._array.copy())
        return ndarray(self._array.astype(resolved))

    def to_list(self) -> list[Any]:
        return self._array.tolist()

    def tolist(self) -> list[Any]:
        return self.to_list()

    def mean(self, axis: int | None = None, keepdims: bool = False):
        return mean(self, axis=axis, keepdims=keepdims)

    def std(self, axis: int | None = None, ddof: int = 0, keepdims: bool = False):
        return std(self, axis=axis, ddof=ddof, keepdims=keepdims)

    def sum(self, axis: int | None = None, keepdims: bool = False):
        return sum(self, axis=axis, keepdims=keepdims)

    def __len__(self) -> int:
        return int(self._array.shape[0])

    def __iter__(self):
        for item in self._array:
            if _np.isscalar(item):
                yield _to_python_scalar(item)
            else:
                yield ndarray(item)

    def __getitem__(self, idx):
        result = self._array[idx]
        if _np.isscalar(result):
            return _to_python_scalar(result)
        return ndarray(result)

    def __setitem__(self, idx, value) -> None:
        if isinstance(value, ndarray):
            self._array[idx] = value._array
        else:
            self._array[idx] = value

    # arithmetic ------------------------------------------------------
    def _binary_op(self, other: Any, op) -> Any:
        result = op(self._array, _coerce_operand(other))
        if _np.isscalar(result):
            return _to_python_scalar(result)
        return ndarray(result)

    def __add__(self, other: Any):
        return self._binary_op(other, _np.add)

    def __radd__(self, other: Any):
        return self._binary_op(other, _np.add)

    def __sub__(self, other: Any):
        return self._binary_op(other, _np.subtract)

    def __rsub__(self, other: Any):
        return ndarray(_np.subtract(_coerce_operand(other), self._array))

    def __mul__(self, other: Any):
        return self._binary_op(other, _np.multiply)

    def __rmul__(self, other: Any):
        return self._binary_op(other, _np.multiply)

    def __truediv__(self, other: Any):
        return self._binary_op(other, _np.divide)

    def __rtruediv__(self, other: Any):
        return ndarray(_np.divide(_coerce_operand(other), self._array))

    def __neg__(self):
        return ndarray(_np.negative(self._array))

    def _compare(self, other: Any, op) -> "ndarray":
        result = op(self._array, _coerce_operand(other))
        return ndarray(result.astype(_np.float64, copy=False))

    def __ge__(self, other: Any):
        return self._compare(other, _np.greater_equal)

    def __le__(self, other: Any):
        return self._compare(other, _np.less_equal)

    def __gt__(self, other: Any):
        return self._compare(other, _np.greater)

    def __lt__(self, other: Any):
        return self._compare(other, _np.less)

    def __eq__(self, other: Any):  # type: ignore[override]
        return self._compare(other, _np.equal)

    # linear algebra --------------------------------------------------
    def __matmul__(self, other: Any):
        other_arr = _ensure_ndarray(other)
        backend = _backend_call("matmul", self, other_arr)
        if backend is not None:
            converted = _array_from_backend(backend)
            if isinstance(converted, ndarray):
                return converted
            return converted
        return _matmul_dispatch(self._array, other_arr._array)

    @property
    def T(self) -> "ndarray":
        return ndarray(self._array.T)

    def max(self, axis: int | None = None, keepdims: bool = False):
        return max(self, axis=axis, keepdims=keepdims)

    def min(self, axis: int | None = None, keepdims: bool = False):
        return min(self, axis=axis, keepdims=keepdims)

    def var(self, axis: int | None = None, ddof: int = 0, keepdims: bool = False):
        return _wrap_stat_result(self._array.var(axis=axis, ddof=ddof, keepdims=keepdims))


def _ensure_ndarray(obj: Any) -> ndarray:
    if isinstance(obj, ndarray):
        return obj
    return ndarray(obj)


def _as_numpy(obj: Any) -> _Array:
    if isinstance(obj, ndarray):
        return obj._array
    return _np.asarray(obj)


def _from_numpy(arr: ArrayLike) -> ndarray:
    return ndarray(arr)


# public array constructors ---------------------------------------------------

def array(obj: ArrayLike, dtype: Any | None = None) -> ndarray:
    return ndarray(obj, dtype=dtype)


def asarray(obj: ArrayLike, dtype: Any | None = None) -> ndarray:
    return array(obj, dtype=dtype)


def zeros(shape, dtype: Any = float) -> ndarray:
    resolved = _resolve_dtype(dtype)
    if resolved is None:
        return ndarray(_np.zeros(shape))
    return ndarray(_np.zeros(shape, dtype=resolved))


def zeros_like(arr, dtype: Any = float) -> ndarray:
    resolved = _resolve_dtype(dtype)
    kwargs: dict[str, Any] = {}
    if resolved is not None:
        kwargs["dtype"] = resolved
    return ndarray(_np.zeros_like(_as_numpy(arr), **kwargs))


def ones(shape, dtype: Any = float) -> ndarray:
    resolved = _resolve_dtype(dtype)
    if resolved is None:
        return ndarray(_np.ones(shape))
    return ndarray(_np.ones(shape, dtype=resolved))


def ones_like(arr, dtype: Any = float) -> ndarray:
    resolved = _resolve_dtype(dtype)
    kwargs: dict[str, Any] = {}
    if resolved is not None:
        kwargs["dtype"] = resolved
    return ndarray(_np.ones_like(_as_numpy(arr), **kwargs))


def eye(n: int, dtype: Any = float) -> ndarray:
    resolved = _resolve_dtype(dtype)
    if resolved is None:
        return ndarray(_np.eye(n))
    return ndarray(_np.eye(n, dtype=resolved))


def full(shape, value: Any, dtype: Any = float) -> ndarray:
    resolved = _resolve_dtype(dtype)
    if resolved is None:
        return ndarray(_np.full(shape, value))
    return ndarray(_np.full(shape, value, dtype=resolved))


def full_like(arr, value: Any, dtype: Any = float) -> ndarray:
    resolved = _resolve_dtype(dtype)
    kwargs: dict[str, Any] = {}
    if resolved is not None:
        kwargs["dtype"] = resolved
    return ndarray(_np.full_like(_as_numpy(arr), value, **kwargs))


def reshape(arr, shape: Tuple[int, ...]) -> ndarray:
    return ndarray(_ensure_ndarray(arr)._array.reshape(shape))


def stack(arrs: Sequence[ndarray], axis: int = 0) -> ndarray:
    arrays = [_ensure_ndarray(arr)._array for arr in arrs]
    return ndarray(_np.stack(arrays, axis=axis))


def arange(n: int) -> ndarray:
    return ndarray(_np.arange(n, dtype=_np.int64))


# reductions ------------------------------------------------------------------

def mean(arr, axis: int | None = None, keepdims: bool = False):
    arr = _ensure_ndarray(arr)
    backend = _backend_call("mean", arr, axis, keepdims)
    if backend is not None:
        converted = _array_from_backend(backend)
        if isinstance(converted, ndarray):
            return converted
        return converted
    return _wrap_stat_result(arr._array.mean(axis=axis, keepdims=keepdims))


def std(arr, axis: int | None = None, ddof: int = 0, keepdims: bool = False):
    arr = _ensure_ndarray(arr)
    backend = _backend_call("std", arr, axis, ddof, keepdims)
    if backend is not None:
        converted = _array_from_backend(backend)
        if isinstance(converted, ndarray):
            return converted
        return converted
    return _wrap_stat_result(arr._array.std(axis=axis, ddof=ddof, keepdims=keepdims))


def var(arr, axis: int | None = None, ddof: int = 0, keepdims: bool = False):
    arr = _ensure_ndarray(arr)
    backend = _backend_call("var", arr, axis, ddof, keepdims)
    if backend is not None:
        converted = _array_from_backend(backend)
        if isinstance(converted, ndarray):
            return converted
        return converted
    return _wrap_stat_result(arr._array.var(axis=axis, ddof=ddof, keepdims=keepdims))


def sum(arr, axis: int | None = None, keepdims: bool = False):
    arr = _ensure_ndarray(arr)
    backend = _backend_call("sum", arr, axis, keepdims)
    if backend is not None:
        converted = _array_from_backend(backend)
        if isinstance(converted, ndarray):
            return converted
        return converted
    return _wrap_stat_result(arr._array.sum(axis=axis, keepdims=keepdims))


def min(arr, axis: int | None = None, keepdims: bool = False):
    arr = _ensure_ndarray(arr)
    backend = _backend_call("min", arr, axis, keepdims)
    if backend is not None:
        converted = _array_from_backend(backend)
        if isinstance(converted, ndarray):
            return converted
        return converted
    return _wrap_stat_result(arr._array.min(axis=axis, keepdims=keepdims))


def max(arr, axis: int | None = None, keepdims: bool = False):
    arr = _ensure_ndarray(arr)
    backend = _backend_call("max", arr, axis, keepdims)
    if backend is not None:
        converted = _array_from_backend(backend)
        if isinstance(converted, ndarray):
            return converted
        return converted
    return _wrap_stat_result(arr._array.max(axis=axis, keepdims=keepdims))


def maximum(a, b):
    backend = _backend_call("maximum", _ensure_ndarray(a), _ensure_ndarray(b))
    if backend is not None:
        converted = _array_from_backend(backend)
        if isinstance(converted, ndarray):
            return converted
        return ndarray(converted)
    return ndarray(_np.maximum(_coerce_operand(a), _coerce_operand(b)))


def minimum(a, b):
    backend = _backend_call("minimum", _ensure_ndarray(a), _ensure_ndarray(b))
    if backend is not None:
        converted = _array_from_backend(backend)
        if isinstance(converted, ndarray):
            return converted
        return ndarray(converted)
    return ndarray(_np.minimum(_coerce_operand(a), _coerce_operand(b)))


def median(arr, axis: int | None = None):
    arr = _ensure_ndarray(arr)
    backend = _backend_call("median", arr, axis)
    if backend is not None:
        converted = _array_from_backend(backend)
        if isinstance(converted, ndarray):
            return converted
        return converted
    return _wrap_stat_result(_np.median(arr._array, axis=axis))


# element-wise operations -----------------------------------------------------

def tanh(arr):
    arr = _ensure_ndarray(arr)
    backend = _backend_call("tanh", arr)
    if backend is not None:
        converted = _array_from_backend(backend)
        if isinstance(converted, ndarray):
            return converted
        return ndarray(converted)
    return ndarray(_np.tanh(arr._array))


def exp(arr):
    arr = _ensure_ndarray(arr)
    backend = _backend_call("exp", arr)
    if backend is not None:
        converted = _array_from_backend(backend)
        if isinstance(converted, ndarray):
            return converted
        return ndarray(converted)
    clipped = _np.clip(arr._array, SAFE_EXP_MIN, SAFE_EXP_MAX)
    return ndarray(_np.exp(clipped))


def safe_exp(arr, clip: float = SAFE_EXP_MAX):
    arr = _ensure_ndarray(arr)
    limit = float(abs(clip))
    backend = _backend_call("exp", arr)
    if backend is not None:
        converted = _array_from_backend(backend)
        if isinstance(converted, ndarray):
            return converted
        return ndarray(converted)
    clipped = _np.clip(arr._array, -limit, limit)
    return ndarray(_np.exp(clipped))


def log(arr):
    arr = _ensure_ndarray(arr)
    backend = _backend_call("log", arr)
    if backend is not None:
        converted = _array_from_backend(backend)
        if isinstance(converted, ndarray):
            return converted
        return ndarray(converted)
    return ndarray(_np.log(arr._array))


def logaddexp(a, b):
    backend = _backend_call("logaddexp", _ensure_ndarray(a), _ensure_ndarray(b))
    if backend is not None:
        converted = _array_from_backend(backend)
        if isinstance(converted, ndarray):
            return converted
        return ndarray(converted)
    return ndarray(_np.logaddexp(_coerce_operand(a), _coerce_operand(b)))


def abs(arr):
    arr = _ensure_ndarray(arr)
    backend = _backend_call("abs", arr)
    if backend is not None:
        converted = _array_from_backend(backend)
        if isinstance(converted, ndarray):
            return converted
        return ndarray(converted)
    return ndarray(_np.abs(arr._array))


def clip(arr, min_val, max_val):
    arr = _ensure_ndarray(arr)
    backend = _backend_call("clip", arr, float(min_val), float(max_val))
    if backend is not None:
        converted = _array_from_backend(backend)
        if isinstance(converted, ndarray):
            return converted
        return ndarray(converted)
    return ndarray(_np.clip(arr._array, float(min_val), float(max_val)))


def sqrt(arr):
    arr = _ensure_ndarray(arr)
    backend = _backend_call("sqrt", arr)
    if backend is not None:
        converted = _array_from_backend(backend)
        if isinstance(converted, ndarray):
            return converted
        return ndarray(converted)
    return ndarray(_np.sqrt(arr._array))


def diff(arr, n: int = 1):
    arr = _ensure_ndarray(arr)
    backend = _backend_call("diff", arr, n)
    if backend is not None:
        converted = _array_from_backend(backend)
        if isinstance(converted, ndarray):
            return converted
        return ndarray(converted)
    return ndarray(_np.diff(arr._array, n=n))


def dot(a, b):
    left = _ensure_ndarray(a)
    right = _ensure_ndarray(b)
    backend = _backend_call("dot", left, right)
    if backend is not None:
        converted = _array_from_backend(backend)
        if isinstance(converted, ndarray):
            return converted
        return converted
    a_arr = _as_numpy(left)
    b_arr = _as_numpy(right)
    _dot_validate(a_arr, b_arr)
    result = _np.dot(a_arr, b_arr)
    if _np.isscalar(result):
        return float(result)
    return ndarray(result)


def matmul(a, b):
    left = _ensure_ndarray(a)
    right = _ensure_ndarray(b)
    backend = _backend_call("matmul", left, right)
    if backend is not None:
        converted = _array_from_backend(backend)
        if isinstance(converted, ndarray):
            return converted
        return converted
    return _matmul_dispatch(_as_numpy(left), _as_numpy(right))


def outer(a, b):
    return ndarray(_np.outer(_coerce_operand(a), _coerce_operand(b)))


def argsort(arr):
    arr = _ensure_ndarray(arr)
    backend = _backend_call("argsort", arr)
    if backend is not None:
        converted = _array_from_backend(backend)
        if isinstance(converted, ndarray):
            return converted.astype(int)
        return ndarray(converted).astype(int)
    return ndarray(_np.argsort(arr._array))


def argmax(arr):
    arr = _ensure_ndarray(arr)
    backend = _backend_call("argmax", arr)
    if backend is not None:
        if isinstance(backend, Mapping):  # defensive: unexpected type
            backend = list(backend)[0]
        return int(backend)
    return int(_np.argmax(arr._array))


def trace(arr):
    arr = _ensure_ndarray(arr)
    backend = _backend_call("trace", arr)
    if backend is not None:
        if isinstance(backend, (list, tuple)):
            return float(backend[0])
        return float(backend)
    return float(_np.trace(arr._array))


class _Linalg:
    @staticmethod
    def norm(vec):
        vec = _ensure_ndarray(vec)
        backend = _backend_call("linalg_norm", vec)
        if backend is not None:
            return float(backend)
        return float(_np.linalg.norm(vec._array))

    @staticmethod
    def inv(mat):
        mat = _ensure_ndarray(mat)
        backend = _backend_call("linalg_inv", mat)
        if backend is not None:
            converted = _array_from_backend(backend)
            if isinstance(converted, ndarray):
                return converted
            return ndarray(converted)
        try:
            inv = _gauss_jordan_inverse(mat._array)
        except _np.linalg.LinAlgError as exc:
            raise _np.linalg.LinAlgError(str(exc))
        return ndarray(inv)

    @staticmethod
    def slogdet(mat):
        mat = _ensure_ndarray(mat)
        backend = _backend_call("linalg_slogdet", mat)
        if backend is not None:
            sign, logdet = backend
            return float(sign), float(logdet)
        sign, logdet = _np.linalg.slogdet(mat._array)
        return float(sign), float(logdet)

    @staticmethod
    def solve(a, b):
        a_arr = _ensure_ndarray(a)
        b_arr = _ensure_ndarray(b)
        backend = _backend_call("linalg_solve", a_arr, b_arr)
        if backend is not None:
            converted = _array_from_backend(backend)
            if isinstance(converted, ndarray):
                return converted
            if isinstance(converted, list):
                return ndarray(converted)
            return converted
        result = _np.linalg.solve(a_arr._array, b_arr._array)
        if _np.isscalar(result):
            return _to_python_scalar(result)
        return ndarray(result)

    @staticmethod
    def cholesky(mat):
        mat_arr = _ensure_ndarray(mat)
        backend = _backend_call("linalg_cholesky", mat_arr)
        if backend is not None:
            converted = _array_from_backend(backend)
            if isinstance(converted, ndarray):
                return converted
            if isinstance(converted, list):
                return ndarray(converted)
            return converted
        factor = _np.linalg.cholesky(mat_arr._array)
        return ndarray(factor)


linalg = _Linalg()


class RandomGenerator:
    def __init__(self, seed: int | None = None):
        self._rng = _np.random.default_rng(seed)

    def uniform(self, low: float, high: float, size: Tuple[int, ...]):
        return ndarray(self._rng.uniform(low, high, size=size))

    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Tuple[int, ...] | int = (1,),
    ):
        return ndarray(self._rng.normal(loc=loc, scale=scale, size=size))

    def permutation(self, n: int) -> ndarray:
        return ndarray(self._rng.permutation(n))

    def shuffle(self, arr):
        if isinstance(arr, ndarray):
            self._rng.shuffle(arr._array)
        else:
            self._rng.shuffle(arr)

    def choice(self, n: int, p: Sequence[float]):
        return int(self._rng.choice(n, p=p))


class _RandomModule:
    @staticmethod
    def default_rng(seed: int | None = None) -> RandomGenerator:
        return RandomGenerator(seed)


random = _RandomModule()


_proxy = _StubProxy(__name__)
_proxy.__dict__.update(_current_module.__dict__)
sys.modules[__name__] = _proxy
