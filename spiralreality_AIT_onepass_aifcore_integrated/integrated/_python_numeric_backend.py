"""Pure-Python numeric helpers shared by accelerated backends.

These functions operate on plain Python containers (lists/tuples) holding
float-compatible values.  They intentionally avoid depending on ``np_stub`` so
that native extensions can call into them without creating import cycles.
"""

from __future__ import annotations

import math
from typing import List

Number = float


def _to_float(value) -> float:
    return float(value)


def _to_list(obj) -> List:
    if isinstance(obj, list):
        return [(_to_list(x) if isinstance(x, (list, tuple)) else _to_float(x)) for x in obj]
    if isinstance(obj, tuple):
        return [(_to_list(x) if isinstance(x, (list, tuple)) else _to_float(x)) for x in obj]
    if hasattr(obj, "tolist"):
        try:
            return _to_list(obj.tolist())
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        try:
            return [_to_list(x) if isinstance(x, (list, tuple)) else _to_float(x) for x in obj]
        except Exception:  # pragma: no cover - defensive
            pass
    return [_to_float(obj)]


def _flatten(data) -> List[float]:
    if isinstance(data, (list, tuple)):
        out: List[float] = []
        for item in data:
            out.extend(_flatten(item))
        return out
    return [float(data)]


def _ensure_matrix(data) -> List[List[float]]:
    matrix = _to_list(data)
    if not matrix:
        return []
    if isinstance(matrix[0], list):
        return [list(map(float, row)) for row in matrix]
    return [list(map(float, matrix))]


def _ensure_vector(data) -> List[float]:
    values = _flatten(data)
    return [float(v) for v in values]


def matmul(a, b):
    A = _to_list(a)
    B = _to_list(b)
    if not A or not B:
        return []
    a_is_vec = not isinstance(A[0], list)
    b_is_vec = not isinstance(B[0], list)
    if a_is_vec and b_is_vec:
        return sum(float(x) * float(y) for x, y in zip(A, B))
    if not a_is_vec and b_is_vec:
        result = []
        for row in A:
            result.append(sum(float(x) * float(y) for x, y in zip(row, B)))
        return result
    if not a_is_vec and not b_is_vec:
        cols = list(zip(*B))
        result = []
        for row in A:
            out_row = []
            for col in cols:
                out_row.append(sum(float(x) * float(y) for x, y in zip(row, col)))
            result.append(out_row)
        return result
    if a_is_vec and not b_is_vec:
        cols = list(zip(*B))
        return [sum(float(x) * float(y) for x, y in zip(A, col)) for col in cols]
    raise ValueError("Unsupported operands for matmul")


def dot(a, b):
    vec_a = _ensure_vector(a)
    vec_b = _ensure_vector(b)
    return sum(x * y for x, y in zip(vec_a, vec_b))


def mean(data, axis=None):
    arr = _to_list(data)
    if axis is None:
        values = _flatten(arr)
        if not values:
            return 0.0
        return sum(values) / len(values)
    if not arr:
        return []
    if axis == 0:
        if not isinstance(arr[0], list):
            values = [float(v) for v in arr]
            if not values:
                return [0.0]
            return [sum(values) / len(values)]
        cols = list(zip(*arr))
        return [sum(col) / len(col) if col else 0.0 for col in cols]
    if axis == 1:
        if not isinstance(arr[0], list):
            raise ValueError("axis=1 requires a 2D input")
        return [sum(row) / len(row) if row else 0.0 for row in arr]
    raise ValueError("Unsupported axis")


def std(data, axis=None):
    arr = _to_list(data)
    if axis is None:
        values = _flatten(arr)
        if not values:
            return 0.0
        mean_val = sum(values) / len(values)
        return math.sqrt(sum((v - mean_val) ** 2 for v in values) / len(values))
    if axis == 0:
        if not isinstance(arr[0], list):
            values = [float(v) for v in arr]
            if not values:
                return [0.0]
            mean_val = sum(values) / len(values)
            return [math.sqrt(sum((v - mean_val) ** 2 for v in values) / len(values))]
        cols = list(zip(*arr))
        result = []
        for col in cols:
            if not col:
                result.append(0.0)
                continue
            mean_val = sum(col) / len(col)
            result.append(math.sqrt(sum((v - mean_val) ** 2 for v in col) / len(col)))
        return result
    if axis == 1:
        if not isinstance(arr[0], list):
            raise ValueError("axis=1 requires a 2D input")
        result = []
        for row in arr:
            if not row:
                result.append(0.0)
                continue
            mean_val = sum(row) / len(row)
            result.append(math.sqrt(sum((v - mean_val) ** 2 for v in row) / len(row)))
        return result
    raise ValueError("Unsupported axis")


def sum_reduce(data, axis=None, keepdims=False):
    arr = _to_list(data)
    if axis is None:
        return sum(_flatten(arr))
    if axis == 0:
        if not arr:
            return []
        if not isinstance(arr[0], list):
            total = sum(float(v) for v in arr)
            if keepdims:
                return [[total]]
            return [total]
        cols = list(zip(*arr))
        values = [sum(col) for col in cols]
        if keepdims:
            return [values]
        return values
    if axis == 1:
        if not arr:
            return []
        if not isinstance(arr[0], list):
            raise ValueError("axis=1 requires a 2D input")
        values = [sum(row) for row in arr]
        if keepdims:
            return [[v] for v in values]
        return values
    raise ValueError("Unsupported axis")


def _elementwise(data, func):
    if isinstance(data, (list, tuple)):
        return [_elementwise(item, func) for item in data]
    return func(float(data))


def tanh_map(data):
    return _elementwise(data, math.tanh)


def exp_map(data):
    return _elementwise(data, math.exp)


def log_map(data):
    return _elementwise(data, math.log)


def logaddexp_map(a, b):
    arr_a = _to_list(a)
    arr_b = _to_list(b)

    def _combine(x, y):
        m = max(x, y)
        return m + math.log(math.exp(x - m) + math.exp(y - m))

    if isinstance(arr_a, list) and isinstance(arr_a[0], list):
        return [[_combine(float(x), float(y)) for x, y in zip(row_a, row_b)] for row_a, row_b in zip(arr_a, arr_b)]
    if isinstance(arr_a, list) and not isinstance(arr_a[0], list):
        return [_combine(float(x), float(y)) for x, y in zip(arr_a, arr_b)]
    return _combine(float(arr_a), float(arr_b))


def median_all(data):
    values = sorted(_flatten(data))
    if not values:
        return 0.0
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def abs_map(data):
    return _elementwise(data, abs)


def clip_map(data, lo, hi):
    return _elementwise(data, lambda v: max(lo, min(hi, v)))


def sqrt_map(data):
    return _elementwise(data, math.sqrt)


def diff_vec(data):
    vec = _ensure_vector(data)
    if len(vec) <= 1:
        return []
    return [vec[i + 1] - vec[i] for i in range(len(vec) - 1)]


def argsort_indices(data):
    vec = _ensure_vector(data)
    return [idx for idx, _ in sorted(enumerate(vec), key=lambda item: item[1])]


def argmax_index(data):
    vec = _ensure_vector(data)
    if not vec:
        return 0
    max_idx = 0
    max_val = vec[0]
    for idx, value in enumerate(vec):
        if value > max_val:
            max_val = value
            max_idx = idx
    return max_idx


def trace_value(data):
    mat = _ensure_matrix(data)
    if not mat:
        return 0.0
    size = min(len(mat), len(mat[0]))
    return sum(mat[i][i] for i in range(size))


def norm_value(data):
    vec = _ensure_vector(data)
    return math.sqrt(sum(v * v for v in vec))


def _identity_matrix(n: int) -> List[List[float]]:
    ident = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        ident[i][i] = 1.0
    return ident


def inv_matrix(data):
    mat = _ensure_matrix(data)
    if not mat:
        return []
    n = len(mat)
    for row in mat:
        if len(row) != n:
            raise ValueError("Matrix must be square")
    augmented = [row[:] + ident_row for row, ident_row in zip(mat, _identity_matrix(n))]
    for i in range(n):
        pivot = augmented[i][i]
        if abs(pivot) < 1e-12:
            swap_row = None
            for j in range(i + 1, n):
                if abs(augmented[j][i]) > 1e-12:
                    swap_row = j
                    break
            if swap_row is None:
                raise ValueError("Matrix is singular")
            augmented[i], augmented[swap_row] = augmented[swap_row], augmented[i]
            pivot = augmented[i][i]
        pivot_inv = 1.0 / pivot
        augmented[i] = [value * pivot_inv for value in augmented[i]]
        for j in range(n):
            if j == i:
                continue
            factor = augmented[j][i]
            if factor == 0.0:
                continue
            augmented[j] = [val - factor * aug for val, aug in zip(augmented[j], augmented[i])]
    inverse = [row[n:] for row in augmented]
    return inverse


def slogdet_pair(data):
    mat = _ensure_matrix(data)
    if not mat:
        return (1.0, -math.inf)
    n = len(mat)
    for row in mat:
        if len(row) != n:
            raise ValueError("Matrix must be square")
    sign = 1.0
    log_abs_det = 0.0
    matrix = [row[:] for row in mat]
    for i in range(n):
        pivot_row = max(range(i, n), key=lambda r: abs(matrix[r][i]))
        pivot_val = matrix[pivot_row][i]
        if abs(pivot_val) < 1e-12:
            return 0.0, -math.inf
        if pivot_row != i:
            matrix[i], matrix[pivot_row] = matrix[pivot_row], matrix[i]
            sign *= -1.0
        pivot_val = matrix[i][i]
        sign *= 1.0 if pivot_val > 0 else -1.0
        log_abs_det += math.log(abs(pivot_val))
        for j in range(i + 1, n):
            factor = matrix[j][i] / pivot_val
            for k in range(i, n):
                matrix[j][k] -= factor * matrix[i][k]
    return sign, log_abs_det
