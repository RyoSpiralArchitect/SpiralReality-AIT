"""Very small subset of NumPy implemented in pure Python.

The goal is not numerical performance; we merely provide enough surface area
for the rest of the demo to run in environments where the real `numpy` package
is unavailable.  Only the operations exercised by the project are supported and
the implementation intentionally favours clarity over speed.
"""

from __future__ import annotations

import builtins
import copy
import math
import os
import random as _py_random
from typing import Iterable, List, Sequence, Tuple

try:  # pragma: no cover - optional acceleration path
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - numpy not present in stub mode
    _np = None

_BACKEND_PREF = os.getenv("SPIRAL_NUMERIC_BACKEND", "auto").strip().lower()
_USE_NUMPY = _np is not None and _BACKEND_PREF in {"auto", "numpy", "accelerated", "native"}
NUMERIC_BACKEND = "numpy" if _USE_NUMPY else "python"

IS_PURE_PY = True

Number = float


def _to_nested(obj) -> List:
    if isinstance(obj, ndarray):
        return copy.deepcopy(obj._data)
    if isinstance(obj, (list, tuple)):
        return [_to_nested(x) for x in obj]
    return float(obj)


def _shape(data) -> Tuple[int, ...]:
    if not isinstance(data, list):
        return ()
    if not data:
        return (0,)
    first = data[0]
    tail = _shape(first)
    return (len(data),) + tail


def _ensure_ndarray(obj) -> "ndarray":
    if isinstance(obj, ndarray):
        return obj
    return ndarray(obj)


def _as_numpy(obj):  # pragma: no cover - optional fast-path helper
    if not _USE_NUMPY:
        raise RuntimeError("NumPy backend not available")
    if isinstance(obj, ndarray):
        return _np.asarray(obj.to_list(), dtype=_np.float64)
    return _np.asarray(obj, dtype=_np.float64)


def _from_numpy(arr):  # pragma: no cover - optional fast-path helper
    if not _USE_NUMPY:
        raise RuntimeError("NumPy backend not available")
    return ndarray(arr.tolist())


def _flatten(data) -> List[Number]:
    if isinstance(data, list):
        out: List[Number] = []
        for item in data:
            out.extend(_flatten(item))
        return out
    return [float(data)]


def _elementwise(op, a, b):
    if isinstance(a, list) and isinstance(b, list):
        return [_elementwise(op, x, y) for x, y in zip(a, b)]
    if isinstance(a, list):
        return [_elementwise(op, x, b) for x in a]
    if isinstance(b, list):
        return [_elementwise(op, a, y) for y in b]
    return float(op(float(a), float(b)))


class ndarray:
    def __init__(self, data):
        self._data = _to_nested(data)

    # basic protocol -------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return _shape(self._data)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def copy(self) -> "ndarray":
        return ndarray(copy.deepcopy(self._data))

    def astype(self, dtype) -> "ndarray":
        def cast(v):
            if dtype in (float, "float", "float32", "float64"):
                return float(v)
            if dtype in (int, "int", "int32", "int64"):
                return int(v)
            if dtype in (bool, "bool"):
                return bool(v)
            return dtype(v)
        return ndarray(_elementwise(lambda x, _: cast(x), self._data, 0.0))

    def to_list(self):
        return copy.deepcopy(self._data)

    def tolist(self):  # numpy compatibility name
        return self.to_list()

    def mean(self, axis=None):
        return mean(self, axis=axis)

    def std(self, axis=None):
        return std(self, axis=axis)

    def sum(self, axis=None, keepdims=False):
        return sum(self, axis=axis, keepdims=keepdims)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for item in self._data:
            if isinstance(item, list):
                yield ndarray(item)
            else:
                yield item

    def __getitem__(self, idx):
        res = self._data[idx]
        if isinstance(res, list):
            return ndarray(res)
        return res

    def __setitem__(self, idx, value):
        if isinstance(value, ndarray):
            self._data[idx] = value.to_list()
        else:
            self._data[idx] = value

    # arithmetic -----------------------------------------------------
    def _apply(self, other, op):
        other_data = other._data if isinstance(other, ndarray) else other
        return ndarray(_elementwise(op, self._data, other_data))

    def __add__(self, other):
        return self._apply(other, lambda x, y: x + y)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._apply(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return ndarray(_elementwise(lambda x, y: x - y, other, self._data))

    def __mul__(self, other):
        return self._apply(other, lambda x, y: x * y)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._apply(other, lambda x, y: x / y)

    def __rtruediv__(self, other):
        return ndarray(_elementwise(lambda x, y: x / y, other, self._data))

    def __neg__(self):
        return ndarray(_elementwise(lambda x, _: -x, self._data, 0.0))

    def _compare(self, other, op):
        other_data = other._data if isinstance(other, ndarray) else other
        return ndarray(_elementwise(lambda x, y: 1.0 if op(x, y) else 0.0, self._data, other_data))

    def __ge__(self, other):
        return self._compare(other, lambda x, y: x >= y)

    def __le__(self, other):
        return self._compare(other, lambda x, y: x <= y)

    def __gt__(self, other):
        return self._compare(other, lambda x, y: x > y)

    def __lt__(self, other):
        return self._compare(other, lambda x, y: x < y)

    def __eq__(self, other):  # type: ignore[override]
        return self._compare(other, lambda x, y: x == y)

    # linear algebra -------------------------------------------------
    def __matmul__(self, other):
        other = _ensure_ndarray(other)
        if _USE_NUMPY:
            result = _np.matmul(_as_numpy(self), _as_numpy(other))
            if result.ndim == 0:
                return float(result)
            return _from_numpy(result)
        if self.ndim == 1 and other.ndim == 1:
            return float(builtins.sum(x * y for x, y in zip(self._data, other._data)))
        if self.ndim == 2 and other.ndim == 1:
            out = []
            for row in self._data:
                out.append(float(builtins.sum(x * y for x, y in zip(row, other._data))))
            return ndarray(out)
        if self.ndim == 2 and other.ndim == 2:
            out = []
            other_cols = list(zip(*other._data))
            for row in self._data:
                out_row = []
                for col in other_cols:
                    out_row.append(float(builtins.sum(x * y for x, y in zip(row, col))))
                out.append(out_row)
            return ndarray(out)
        if self.ndim == 1 and other.ndim == 2:
            return ndarray([
                float(builtins.sum(x * y for x, y in zip(self._data, col)))
                for col in zip(*other._data)
            ])
        raise ValueError("Unsupported shapes for matmul")

    @property
    def T(self):
        if self.ndim == 1:
            return ndarray([[x] for x in self._data])
        if self.ndim == 2:
            return ndarray([list(col) for col in zip(*self._data)])
        raise ValueError("Transpose only implemented for up to 2D arrays")

    def max(self, axis=None, keepdims=False):
        if axis is None:
            return max(_flatten(self._data))
        if self.ndim != 2:
            raise ValueError("axis operations only supported for 2D arrays")
        if axis == -1:
            axis = 1
        if axis == 0:
            cols = zip(*self._data)
            vals = [max(col) for col in cols]
        elif axis == 1:
            vals = [max(row) for row in self._data]
        else:
            raise ValueError("Unsupported axis")
        if keepdims:
            if axis == 0:
                return ndarray([vals])
            return ndarray([[v] for v in vals])
        return ndarray(vals)


def array(obj, dtype=float):
    return ndarray(_to_nested(obj))


def asarray(obj, dtype=float):
    return array(obj, dtype=dtype)


def zeros(shape, dtype=float):
    if isinstance(shape, int):
        data = [0.0 for _ in range(shape)]
    else:
        if len(shape) == 1:
            data = [0.0 for _ in range(shape[0])]
        else:
            rows = shape[0]
            cols = shape[1]
            data = [[0.0 for _ in range(cols)] for _ in range(rows)]
    return ndarray(data)


def zeros_like(arr, dtype=float):
    arr = _ensure_ndarray(arr)
    return zeros(arr.shape, dtype=dtype)


def ones(shape, dtype=float):
    if isinstance(shape, int):
        data = [1.0 for _ in range(shape)]
    else:
        if len(shape) == 1:
            data = [1.0 for _ in range(shape[0])]
        else:
            rows = shape[0]
            cols = shape[1]
            data = [[1.0 for _ in range(cols)] for _ in range(rows)]
    return ndarray(data)


def ones_like(arr, dtype=float):
    arr = _ensure_ndarray(arr)
    return ones(arr.shape, dtype=dtype)


def eye(n: int, dtype=float):
    data = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        data[i][i] = 1.0
    return ndarray(data)


def stack(arrs: Sequence[ndarray], axis: int = 0):
    arrs = [_ensure_ndarray(a).to_list() for a in arrs]
    if axis == 0:
        return ndarray(arrs)
    raise ValueError("Only axis=0 supported in stub")


def mean(arr, axis=None):
    arr = _ensure_ndarray(arr)
    if axis is None:
        values = _flatten(arr._data)
        return float(builtins.sum(values) / max(1, len(values)))
    if axis == 0:
        cols = list(zip(*arr._data))
        return ndarray([float(builtins.sum(col) / len(col)) for col in cols])
    if axis == 1:
        return ndarray([float(builtins.sum(row) / len(row)) for row in arr._data])
    raise ValueError("Unsupported axis")


def std(arr, axis=None):
    arr = _ensure_ndarray(arr)
    if axis is None:
        values = _flatten(arr._data)
        if not values:
            return 0.0
        m = builtins.sum(values) / len(values)
        var = builtins.sum((x - m) ** 2 for x in values) / len(values)
        return math.sqrt(var)
    if axis == 0:
        cols = list(zip(*arr._data))
        out = []
        for col in cols:
            m = builtins.sum(col) / len(col)
            var = builtins.sum((x - m) ** 2 for x in col) / len(col)
            out.append(math.sqrt(var))
        return ndarray(out)
    if axis == 1:
        out = []
        for row in arr._data:
            m = builtins.sum(row) / len(row)
            var = builtins.sum((x - m) ** 2 for x in row) / len(row)
            out.append(math.sqrt(var))
        return ndarray(out)
    raise ValueError("Unsupported axis")


def maximum(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return max(float(a), float(b))
    a_data = _ensure_ndarray(a)._data if not isinstance(a, (int, float)) else a
    b_data = _ensure_ndarray(b)._data if not isinstance(b, (int, float)) else b
    return ndarray(_elementwise(max, a_data, b_data))


def arange(n: int):
    return ndarray([int(i) for i in range(n)])


def tanh(x):
    if isinstance(x, ndarray):
        if x.ndim == 2:
            return ndarray([[math.tanh(v) for v in row] for row in x._data])
        return ndarray([math.tanh(v) for v in x._data])
    return math.tanh(float(x))


def dot(a, b):
    a_arr = _ensure_ndarray(a)
    b_arr = _ensure_ndarray(b)
    if _USE_NUMPY:
        return float(_np.dot(_as_numpy(a_arr), _as_numpy(b_arr)))
    if a_arr.ndim == 1 and b_arr.ndim == 1:
        return float(builtins.sum(x * y for x, y in zip(a_arr._data, b_arr._data)))
    raise ValueError("dot only implemented for 1D vectors in stub")


def exp(x):
    def _safe_exp(v):
        if v > 700:
            return math.exp(700)
        if v < -700:
            return math.exp(-700)
        return math.exp(v)
    if isinstance(x, ndarray):
        if x.ndim == 2:
            return ndarray([[_safe_exp(v) for v in row] for row in x._data])
        return ndarray([_safe_exp(v) for v in x._data])
    return _safe_exp(float(x))


def log(x):
    if isinstance(x, ndarray):
        if x.ndim == 2:
            return ndarray([[math.log(v) for v in row] for row in x._data])
        return ndarray([math.log(v) for v in x._data])
    return math.log(float(x))


def logaddexp(a, b):
    if isinstance(a, ndarray) or isinstance(b, ndarray):
        a = _ensure_ndarray(a)
        b = _ensure_ndarray(b)
        def _logadd(x, y):
            m = max(x, y)
            return m + math.log(math.exp(x - m) + math.exp(y - m))
        data = _elementwise(_logadd, a._data, b._data)
        return ndarray(data)
    a_val = float(a)
    b_val = float(b)
    m = max(a_val, b_val)
    return m + math.log(math.exp(a_val - m) + math.exp(b_val - m))


def median(arr):
    arr = _ensure_ndarray(arr)
    values = sorted(_flatten(arr._data))
    n = len(values)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def abs(arr):
    arr = _ensure_ndarray(arr)
    return ndarray(_elementwise(lambda x, _: math.fabs(x), arr._data, 0.0))


def clip(arr, min_val, max_val):
    arr = _ensure_ndarray(arr)
    return ndarray(_elementwise(lambda x, _: max(min_val, min(max_val, x)), arr._data, 0.0))


def sqrt(arr):
    if isinstance(arr, ndarray):
        return ndarray(_elementwise(lambda x, _: math.sqrt(x), arr._data, 0.0))
    return math.sqrt(float(arr))


def diff(arr):
    arr = _ensure_ndarray(arr)
    data = arr._data
    if arr.ndim != 1:
        raise ValueError("diff only implemented for 1D arrays")
    return ndarray([float(data[i + 1] - data[i]) for i in range(len(data) - 1)])


def sum(arr, axis=None, keepdims=False):
    arr = _ensure_ndarray(arr)
    if axis is None:
        return float(builtins.sum(_flatten(arr._data)))
    if axis == 0:
        cols = list(zip(*arr._data))
        vals = [float(builtins.sum(col)) for col in cols]
        if keepdims:
            return ndarray([vals])
        return ndarray(vals)
    if axis == 1:
        vals = [float(builtins.sum(row)) for row in arr._data]
        if keepdims:
            return ndarray([[v] for v in vals])
        return ndarray(vals)
    raise ValueError("Unsupported axis")


def argsort(arr):
    arr = _ensure_ndarray(arr)
    flat = list(enumerate(arr._data))
    flat.sort(key=lambda x: x[1])
    return ndarray([idx for idx, _ in flat])


def argmax(arr):
    arr = _ensure_ndarray(arr)
    if arr.ndim != 1:
        raise ValueError("argmax only implemented for 1D arrays")
    data = arr._data
    best_idx = max(range(len(data)), key=lambda i: data[i])
    return int(best_idx)


def trace(arr):
    arr = _ensure_ndarray(arr)
    if arr.ndim != 2:
        raise ValueError("trace expects a matrix")
    return float(builtins.sum(arr._data[i][i] for i in range(len(arr._data))))


class _Linalg:
    @staticmethod
    def norm(vec):
        vec = _ensure_ndarray(vec)
        return math.sqrt(builtins.sum(float(x) ** 2 for x in _flatten(vec._data)))

    @staticmethod
    def inv(mat):
        mat = _ensure_ndarray(mat)
        n, m = mat.shape
        if n != m:
            raise ValueError("Only square matrices can be inverted")
        if _USE_NUMPY:
            return _from_numpy(_np.linalg.inv(_as_numpy(mat)))
        base = mat.to_list()
        a = [row[:] for row in base]
        inv = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for col in range(n):
            pivot = col
            while pivot < n and builtins.abs(a[pivot][col]) < 1e-12:
                pivot += 1
            if pivot == n:
                raise ValueError("Matrix is singular")
            if pivot != col:
                a[col], a[pivot] = a[pivot], a[col]
                inv[col], inv[pivot] = inv[pivot], inv[col]
            factor = a[col][col]
            inv_factor = 1.0 / factor
            for j in range(n):
                a[col][j] *= inv_factor
                inv[col][j] *= inv_factor
            for i in range(n):
                if i == col:
                    continue
                factor = a[i][col]
                for j in range(n):
                    a[i][j] -= factor * a[col][j]
                    inv[i][j] -= factor * inv[col][j]
        return ndarray(inv)

    @staticmethod
    def slogdet(mat):
        mat = _ensure_ndarray(mat)
        n, m = mat.shape
        if n != m:
            raise ValueError("slogdet expects a square matrix")
        a = [row[:] for row in mat.to_list()]
        sign = 1.0
        logdet = 0.0
        for i in range(n):
            pivot = i
            while pivot < n and builtins.abs(a[pivot][i]) < 1e-12:
                pivot += 1
            if pivot == n:
                return 0.0, float("-inf")
            if pivot != i:
                a[i], a[pivot] = a[pivot], a[i]
                sign *= -1.0
            pivot_val = a[i][i]
            sign *= math.copysign(1.0, pivot_val)
            logdet += math.log(builtins.abs(pivot_val))
            for j in range(i + 1, n):
                factor = a[j][i] / pivot_val
                for k in range(i, n):
                    a[j][k] -= factor * a[i][k]
        return float(sign), float(logdet)


linalg = _Linalg()


class RandomGenerator:
    def __init__(self, seed: int | None = None):
        self._rng = _py_random.Random(seed)

    def uniform(self, low: float, high: float, size: Tuple[int, ...]):
        if len(size) == 1:
            return ndarray([self._rng.uniform(low, high) for _ in range(size[0])])
        rows, cols = size
        return ndarray([[self._rng.uniform(low, high) for _ in range(cols)] for _ in range(rows)])

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Tuple[int, ...] | int = (1,)):
        if isinstance(size, int):
            size = (size,)
        def sample():
            u1 = self._rng.random()
            u2 = self._rng.random()
            z0 = math.sqrt(-2.0 * math.log(u1 + 1e-12)) * math.cos(2 * math.pi * u2)
            return loc + scale * z0

        if len(size) == 1:
            return ndarray([sample() for _ in range(size[0])])
        rows, cols = size
        return ndarray([[sample() for _ in range(cols)] for _ in range(rows)])

    def permutation(self, n: int) -> ndarray:
        items = list(range(n))
        self._rng.shuffle(items)
        return ndarray(items)

    def shuffle(self, arr):
        if isinstance(arr, ndarray):
            data = arr._data
            self._rng.shuffle(data)
        else:
            self._rng.shuffle(arr)

    def choice(self, n: int, p: Sequence[float]):
        r = self._rng.random()
        total = 0.0
        for idx, prob in enumerate(p):
            total += prob
            if r <= total:
                return idx
        return n - 1


class _RandomModule:
    @staticmethod
    def default_rng(seed: int | None = None) -> RandomGenerator:
        return RandomGenerator(seed)


random = _RandomModule()


def full(shape, value, dtype=float):
    if isinstance(shape, tuple):
        if len(shape) == 1:
            return ndarray([float(value) for _ in range(shape[0])])
        if len(shape) == 2:
            return ndarray([[float(value) for _ in range(shape[1])] for _ in range(shape[0])])
    else:
        return ndarray([float(value) for _ in range(shape)])
    raise ValueError("Unsupported shape for full")


def full_like(arr, value, dtype=float):
    arr = _ensure_ndarray(arr)
    return full(arr.shape, value, dtype=dtype)


def reshape(arr, shape):
    arr = _ensure_ndarray(arr)
    flat = _flatten(arr._data)
    if len(shape) == 1:
        return ndarray(flat[:shape[0]])
    rows, cols = shape
    data = []
    idx = 0
    for _ in range(rows):
        row = []
        for _ in range(cols):
            row.append(flat[idx])
            idx += 1
        data.append(row)
    return ndarray(data)


float32 = float
float64 = float


pi = math.pi

