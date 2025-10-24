from __future__ import annotations

from typing import Any

import math

import numpy as real_numpy
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated import (
    _python_numeric_backend,
    np_stub,
)


class _FakeArray:
    def __init__(self, data: Any):
        self._data = data

    def tolist(self):  # pragma: no cover - trivial passthrough
        return self._data


def test_array_from_backend_uses_tolist():
    backend_value = _FakeArray([[1, 2], [3, 4]])
    result = np_stub._array_from_backend(backend_value)
    assert isinstance(result, np_stub.ndarray)
    assert result.to_list() == [[1.0, 2.0], [3.0, 4.0]]


def test_matmul_accepts_backend_arrays(monkeypatch: pytest.MonkeyPatch):
    def fake_backend(name: str, *_args, **_kwargs):
        if name == "matmul":
            return _FakeArray([[5, 6], [7, 8]])
        return None

    monkeypatch.setattr(np_stub, "_backend_call", fake_backend)

    left = np_stub.array([[1.0, 0.0], [0.0, 1.0]])
    right = np_stub.array([[5.0, 6.0], [7.0, 8.0]])
    result = left @ right
    assert isinstance(result, np_stub.ndarray)
    assert result.to_list() == [[5.0, 6.0], [7.0, 8.0]]


def test_mean_axis_backend_array(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}

    def fake_backend(name: str, *args, **kwargs):
        if name == "mean":
            captured["mean"] = (args, kwargs)
            return _FakeArray([1.0, 2.0, 3.0])
        return None

    monkeypatch.setattr(np_stub, "_backend_call", fake_backend)

    arr = np_stub.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    result = np_stub.mean(arr, axis=0)
    assert isinstance(result, np_stub.ndarray)
    assert result.to_list() == [1.0, 2.0, 3.0]
    mean_args, mean_kwargs = captured["mean"]
    assert mean_args[2] is False
    assert mean_kwargs == {}


def test_sum_axis_backend_array(monkeypatch: pytest.MonkeyPatch):
    def fake_backend(name: str, *_args, **_kwargs):
        if name == "sum":
            keepdims = bool(_args[-1]) if _args else False
            if keepdims:
                return _FakeArray([[3.0, 7.0]])
            return _FakeArray([3.0, 7.0])
        return None

    monkeypatch.setattr(np_stub, "_backend_call", fake_backend)

    arr = np_stub.array([[1.0, 2.0], [2.0, 5.0]])
    result = np_stub.sum(arr, axis=0)
    assert isinstance(result, np_stub.ndarray)
    assert result.to_list() == [3.0, 7.0]

    result_keepdims = np_stub.sum(arr, axis=0, keepdims=True)
    assert isinstance(result_keepdims, np_stub.ndarray)
    assert result_keepdims.to_list() == [[3.0, 7.0]]


def test_array_preserves_integer_dtype_and_scalars():
    arr = np_stub.array([1, 2, 3])
    assert arr.dtype == real_numpy.dtype("int64")
    assert arr.to_list() == [1, 2, 3]
    assert isinstance(arr[0], int)


def test_array_preserves_boolean_dtype():
    arr = np_stub.array([[True, False], [False, True]], dtype=bool)
    assert arr.dtype == real_numpy.dtype(bool)
    assert arr.to_list() == [[True, False], [False, True]]


def test_std_backend_receives_ddof_and_keepdims(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}

    def fake_backend(name: str, *args, **kwargs):
        if name == "std":
            captured["std"] = (args, kwargs)
            return _FakeArray([[0.5], [1.5]])
        return None

    monkeypatch.setattr(np_stub, "_backend_call", fake_backend)

    arr = np_stub.array([[1.0, 2.0], [3.0, 6.0]])
    result = np_stub.std(arr, axis=1, ddof=1, keepdims=True)
    assert isinstance(result, np_stub.ndarray)
    assert result.to_list() == [[0.5], [1.5]]
    std_args, std_kwargs = captured["std"]
    assert std_args[2] == 1
    assert std_args[3] is True
    assert std_kwargs == {}


def test_linalg_inv_preserves_float32_dtype():
    mat = np_stub.array([[4.0, 1.0], [2.0, 3.0]], dtype=real_numpy.float32)
    inv_stub = np_stub.linalg.inv(mat)
    assert isinstance(inv_stub, np_stub.ndarray)
    assert inv_stub.dtype == real_numpy.dtype(real_numpy.float32)
    expected = real_numpy.linalg.inv(mat._array)
    real_numpy.testing.assert_allclose(inv_stub._array, expected, rtol=1e-5, atol=1e-6)


def test_linalg_inv_handles_complex64():
    mat = np_stub.array(
        [[1.0 + 1.0j, 2.0 - 1.0j], [0.5 + 2.5j, 3.0 + 0.5j]],
        dtype=real_numpy.complex64,
    )
    inv_stub = np_stub.linalg.inv(mat)
    assert isinstance(inv_stub, np_stub.ndarray)
    assert inv_stub.dtype == real_numpy.dtype(real_numpy.complex64)
    expected = real_numpy.linalg.inv(mat._array)
    real_numpy.testing.assert_allclose(inv_stub._array, expected, rtol=1e-5, atol=1e-5)


def test_linalg_inv_skips_backend_for_float32(monkeypatch: pytest.MonkeyPatch):
    def fail_backend(*_args, **_kwargs):
        raise AssertionError("accelerated backend should not run for float32 inputs")

    monkeypatch.setattr(np_stub, "_backend_call", fail_backend)

    mat = np_stub.array([[4.0, 1.0], [2.0, 3.0]], dtype=real_numpy.float32)
    result = np_stub.linalg.inv(mat)
    expected = real_numpy.linalg.inv(mat._array)
    assert isinstance(result, np_stub.ndarray)
    real_numpy.testing.assert_allclose(result._array, expected, rtol=1e-5, atol=1e-6)


def test_linalg_inv_skips_backend_for_complex(monkeypatch: pytest.MonkeyPatch):
    def fail_backend(*_args, **_kwargs):
        raise AssertionError("accelerated backend should not run for complex inputs")

    monkeypatch.setattr(np_stub, "_backend_call", fail_backend)

    mat = np_stub.array(
        [[1.0 + 0.5j, 2.0 - 1.0j], [3.0 + 0.5j, 4.0 - 2.0j]],
        dtype=real_numpy.complex128,
    )
    result = np_stub.linalg.inv(mat)
    expected = real_numpy.linalg.inv(mat._array)
    assert isinstance(result, np_stub.ndarray)
    real_numpy.testing.assert_allclose(result._array, expected, rtol=1e-6, atol=1e-6)


def test_var_backend_receives_ddof_and_keepdims(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}

    def fake_backend(name: str, *args, **kwargs):
        if name == "var":
            captured["var"] = (args, kwargs)
            return _FakeArray([[0.5], [2.0]])
        return None

    monkeypatch.setattr(np_stub, "_backend_call", fake_backend)

    arr = np_stub.array([[1.0, 2.0], [3.0, 5.0]])
    result = np_stub.var(arr, axis=1, ddof=1, keepdims=True)
    assert isinstance(result, np_stub.ndarray)
    assert result.to_list() == [[0.5], [2.0]]
    var_args, var_kwargs = captured["var"]
    assert var_args[2] == 1
    assert var_args[3] is True
    assert var_kwargs == {}


def test_sum_keepdims_matches_numpy_shape():
    arr = np_stub.array([[1.0, 2.0], [3.0, 4.0]])
    keepdims_sum = np_stub.sum(arr, keepdims=True)
    assert isinstance(keepdims_sum, np_stub.ndarray)
    assert keepdims_sum.shape == (1, 1)
    assert keepdims_sum.to_list() == [[10.0]]

    vec = np_stub.array([1.0, 2.0, 3.0])
    keepdims_vec = np_stub.sum(vec, keepdims=True)
    assert isinstance(keepdims_vec, np_stub.ndarray)
    assert keepdims_vec.shape == (1,)
    assert keepdims_vec.to_list() == [6.0]


def test_mean_keepdims_matches_numpy():
    arr = np_stub.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    stub = np_stub.mean(arr, axis=1, keepdims=True)
    assert isinstance(stub, np_stub.ndarray)
    assert stub.shape == (2, 1)
    expected = real_numpy.mean(real_numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), axis=1, keepdims=True)
    assert stub.to_list() == pytest.approx(expected.tolist())

    stub_method = arr.mean(axis=0, keepdims=True)
    expected_method = real_numpy.mean(real_numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), axis=0, keepdims=True)
    assert isinstance(stub_method, np_stub.ndarray)
    assert stub_method.shape == (1, 3)
    assert stub_method.to_list() == pytest.approx(expected_method.tolist())


def test_std_supports_ddof_and_keepdims():
    values = [[1.0, 2.5, 4.5], [2.0, 5.0, 8.0]]
    arr = np_stub.array(values)
    stub = np_stub.std(arr, axis=1, ddof=1, keepdims=True)
    assert isinstance(stub, np_stub.ndarray)
    expected = real_numpy.std(real_numpy.array(values), axis=1, ddof=1, keepdims=True)
    assert stub.to_list() == pytest.approx(expected.tolist())

    stub_method = arr.std(axis=0, ddof=1, keepdims=True)
    expected_method = real_numpy.std(real_numpy.array(values), axis=0, ddof=1, keepdims=True)
    assert isinstance(stub_method, np_stub.ndarray)
    assert stub_method.to_list() == pytest.approx(expected_method.tolist())


def test_var_supports_ddof_and_keepdims():
    values = [[1.0, 3.0, 5.0], [2.0, 4.0, 8.0]]
    arr = np_stub.array(values)
    stub = np_stub.var(arr, axis=1, ddof=1, keepdims=True)
    expected = real_numpy.var(real_numpy.array(values), axis=1, ddof=1, keepdims=True)
    assert isinstance(stub, np_stub.ndarray)
    assert stub.to_list() == pytest.approx(expected.tolist())

    method_stub = arr.var(axis=0, ddof=1, keepdims=True)
    expected_method = real_numpy.var(real_numpy.array(values), axis=0, ddof=1, keepdims=True)
    assert isinstance(method_stub, np_stub.ndarray)
    assert method_stub.to_list() == pytest.approx(expected_method.tolist())


def test_dot_and_matmul_match_numpy_for_vector_matrix_cases():
    vector = np_stub.array([1.0, 2.0, 3.0])
    matrix = np_stub.array([[1.0, 0.0, 2.0], [0.5, 1.5, -1.0], [3.0, 1.0, 0.0]])
    left_np = real_numpy.array(vector.to_list())
    right_np = real_numpy.array(matrix.to_list())

    vec_dot = np_stub.dot(vector, vector)
    assert vec_dot == pytest.approx(float(real_numpy.dot(left_np, left_np)))

    mat_vec = np_stub.dot(matrix, vector)
    expected = real_numpy.dot(right_np, left_np)
    assert isinstance(mat_vec, np_stub.ndarray)
    assert mat_vec.to_list() == pytest.approx(expected.tolist())

    vec_mat = np_stub.dot(vector, matrix)
    expected_vec_mat = real_numpy.dot(left_np, right_np)
    assert isinstance(vec_mat, np_stub.ndarray)
    assert vec_mat.to_list() == pytest.approx(expected_vec_mat.tolist())

    mm = np_stub.matmul(matrix, matrix)
    expected_mm = real_numpy.matmul(right_np, right_np)
    assert isinstance(mm, np_stub.ndarray)
    assert mm.to_list() == pytest.approx(expected_mm.tolist())


def test_dot_raises_for_misaligned_shapes():
    with pytest.raises(ValueError):
        np_stub.dot(np_stub.array([1.0, 2.0]), np_stub.array([[1.0, 2.0]]))


def test_safe_exp_clips_extremes():
    arr = np_stub.array([0.0, 1000.0, -1000.0])
    result = np_stub.safe_exp(arr)
    assert isinstance(result, np_stub.ndarray)
    # Values should be clipped near exp(700)
    assert max(result.to_list()) == pytest.approx(math.exp(np_stub.SAFE_EXP_MAX), rel=1e-5)
    assert min(result.to_list()) == pytest.approx(math.exp(-np_stub.SAFE_EXP_MAX), rel=1e-5)


def test_linalg_inv_matches_numpy():
    mat = np_stub.array([[4.0, 7.0, 2.0], [3.0, 6.0, 1.0], [2.0, 5.0, 3.0]])
    inv = np_stub.linalg.inv(mat)
    expected = real_numpy.linalg.inv(real_numpy.array(mat.to_list()))
    assert isinstance(inv, np_stub.ndarray)
    assert inv.to_list() == pytest.approx(expected.tolist(), rel=1e-9, abs=1e-9)


def test_linalg_inv_prefers_backend_for_float64(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def fake_backend(name: str, *_args, **_kwargs):
        calls.append(name)
        return _FakeArray([[1.0, 0.0], [0.0, 1.0]])

    monkeypatch.setattr(np_stub, "_backend_call", fake_backend)

    mat = np_stub.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    result = np_stub.linalg.inv(mat)
    assert isinstance(result, np_stub.ndarray)
    assert result.to_list() == [[1.0, 0.0], [0.0, 1.0]]
    assert calls == ["linalg_inv"]


def test_linalg_inv_detects_singular_matrix():
    singular = np_stub.array([[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(real_numpy.linalg.LinAlgError):
        np_stub.linalg.inv(singular)


def test_slogdet_matches_numpy():
    mat = np_stub.array([[2.0, 0.0, 1.0], [0.0, 3.0, -1.0], [1.0, -1.0, 1.0]])
    sign, logdet = np_stub.linalg.slogdet(mat)
    expected_sign, expected_logdet = real_numpy.linalg.slogdet(real_numpy.array(mat.to_list()))
    assert sign == pytest.approx(expected_sign)
    assert logdet == pytest.approx(expected_logdet)


def test_linalg_solve_matches_numpy():
    coeffs = np_stub.array([[3.0, 1.0], [1.0, 2.0]])
    rhs_vec = np_stub.array([9.0, 8.0])
    stub_vec = np_stub.linalg.solve(coeffs, rhs_vec)
    expected_vec = real_numpy.linalg.solve(real_numpy.array(coeffs.to_list()), real_numpy.array(rhs_vec.to_list()))
    assert isinstance(stub_vec, np_stub.ndarray)
    assert stub_vec.to_list() == pytest.approx(expected_vec.tolist(), rel=1e-9, abs=1e-9)

    rhs_mat = np_stub.array([[9.0, 1.0], [8.0, 2.0]])
    stub_mat = np_stub.linalg.solve(coeffs, rhs_mat)
    expected_mat = real_numpy.linalg.solve(
        real_numpy.array(coeffs.to_list()), real_numpy.array(rhs_mat.to_list())
    )
    assert isinstance(stub_mat, np_stub.ndarray)
    assert stub_mat.to_list() == pytest.approx(expected_mat.tolist(), rel=1e-9, abs=1e-9)


def test_linalg_solve_skips_backend_for_float32(monkeypatch: pytest.MonkeyPatch):
    def fail_backend(*_args, **_kwargs):
        raise AssertionError("accelerated solve should not run for float32 inputs")

    monkeypatch.setattr(np_stub, "_backend_call", fail_backend)

    coeffs = np_stub.array([[3.0, 1.0], [1.0, 2.0]], dtype=real_numpy.float32)
    rhs_vec = np_stub.array([9.0, 8.0], dtype=real_numpy.float32)
    result = np_stub.linalg.solve(coeffs, rhs_vec)
    expected = real_numpy.linalg.solve(coeffs._array, rhs_vec._array)
    assert isinstance(result, np_stub.ndarray)
    real_numpy.testing.assert_allclose(result._array, expected, rtol=1e-5, atol=1e-6)


def test_linalg_cholesky_skips_backend_for_float32(monkeypatch: pytest.MonkeyPatch):
    def fail_backend(*_args, **_kwargs):
        raise AssertionError("accelerated cholesky should not run for float32 inputs")

    monkeypatch.setattr(np_stub, "_backend_call", fail_backend)

    mat = np_stub.array([[6.0, 3.0], [3.0, 6.0]], dtype=real_numpy.float32)
    result = np_stub.linalg.cholesky(mat)
    expected = real_numpy.linalg.cholesky(mat._array)
    assert isinstance(result, np_stub.ndarray)
    real_numpy.testing.assert_allclose(result._array, expected, rtol=1e-5, atol=1e-6)


def test_linalg_cholesky_matches_numpy():
    mat = np_stub.array([[6.0, 3.0, 4.0], [3.0, 6.0, 5.0], [4.0, 5.0, 10.0]])
    stub_factor = np_stub.linalg.cholesky(mat)
    expected_factor = real_numpy.linalg.cholesky(real_numpy.array(mat.to_list()))
    assert isinstance(stub_factor, np_stub.ndarray)
    assert stub_factor.to_list() == pytest.approx(expected_factor.tolist(), rel=1e-9, abs=1e-9)


def test_random_normal_matches_numpy_shape_and_stats():
    seed = 1234
    stub_rng = np_stub.random.default_rng(seed)
    np_rng = real_numpy.random.default_rng(seed)
    stub_samples = stub_rng.normal(loc=1.5, scale=2.0, size=(2000,))
    np_samples = np_rng.normal(loc=1.5, scale=2.0, size=(2000,))
    assert isinstance(stub_samples, np_stub.ndarray)
    assert stub_samples.shape == (2000,)
    stub_mean = sum(stub_samples.to_list()) / len(stub_samples)
    np_mean = float(real_numpy.mean(np_samples))
    assert stub_mean == pytest.approx(np_mean, rel=5e-2)


def test_min_backend_receives_keepdims(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}

    def fake_backend(name: str, *args, **kwargs):
        if name == "min":
            captured["min"] = (args, kwargs)
            return _FakeArray([[1.0, -2.0]])
        return None

    monkeypatch.setattr(np_stub, "_backend_call", fake_backend)

    arr = np_stub.array([[1.0, -2.0], [3.0, 0.0]])
    result = np_stub.min(arr, axis=0, keepdims=True)
    assert isinstance(result, np_stub.ndarray)
    assert result.to_list() == [[1.0, -2.0]]
    min_args, min_kwargs = captured["min"]
    assert min_args[2] is True
    assert min_kwargs == {}


def test_max_backend_receives_keepdims(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}

    def fake_backend(name: str, *args, **kwargs):
        if name == "max":
            captured["max"] = (args, kwargs)
            return _FakeArray([[3.0, 9.0]])
        return None

    monkeypatch.setattr(np_stub, "_backend_call", fake_backend)

    arr = np_stub.array([[1.0, 3.0], [3.0, 9.0]])
    result = np_stub.max(arr, axis=0, keepdims=True)
    assert isinstance(result, np_stub.ndarray)
    assert result.to_list() == [[3.0, 9.0]]
    max_args, max_kwargs = captured["max"]
    assert max_args[2] is True
    assert max_kwargs == {}


def test_min_supports_keepdims():
    values = [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]]
    arr = np_stub.array(values)

    stub = np_stub.min(arr, axis=1, keepdims=True)
    expected = real_numpy.min(real_numpy.array(values), axis=1, keepdims=True)
    assert isinstance(stub, np_stub.ndarray)
    assert stub.to_list() == pytest.approx(expected.tolist())

    method_stub = arr.min(axis=0, keepdims=True)
    expected_method = real_numpy.min(real_numpy.array(values), axis=0, keepdims=True)
    assert isinstance(method_stub, np_stub.ndarray)
    assert method_stub.to_list() == pytest.approx(expected_method.tolist())


def test_max_supports_keepdims():
    values = [[1.0, 4.0, -2.0], [7.0, 5.0, 3.0]]
    arr = np_stub.array(values)

    stub = np_stub.max(arr, axis=0, keepdims=True)
    expected = real_numpy.max(real_numpy.array(values), axis=0, keepdims=True)
    assert isinstance(stub, np_stub.ndarray)
    assert stub.to_list() == pytest.approx(expected.tolist())

    method_stub = arr.max(axis=1, keepdims=True)
    expected_method = real_numpy.max(real_numpy.array(values), axis=1, keepdims=True)
    assert isinstance(method_stub, np_stub.ndarray)
    assert method_stub.to_list() == pytest.approx(expected_method.tolist())


def test_python_backend_median_axis():
    data = [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
    assert _python_numeric_backend.median_all(data, axis=0) == [1.5, 3.5, 5.5]
    assert _python_numeric_backend.median_all(data, axis=1) == [3.0, 4.0]


def test_python_backend_diff_higher_order():
    data = [0.0, 1.0, 4.0, 9.0]
    assert _python_numeric_backend.diff_vec(data, 1) == [1.0, 3.0, 5.0]
    assert _python_numeric_backend.diff_vec(data, 2) == [2.0, 2.0]
    matrix = [[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]]
    assert _python_numeric_backend.diff_vec(matrix, 1) == [[1.0, 3.0], [7.0, 9.0]]


def test_python_backend_mean_keepdims_and_axis_handling():
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    expected_all = real_numpy.mean(real_numpy.array(data), keepdims=True)
    assert _python_numeric_backend.mean(data, keepdims=True) == pytest.approx(expected_all.tolist())

    expected_axis0 = real_numpy.mean(real_numpy.array(data), axis=0, keepdims=True)
    assert _python_numeric_backend.mean(data, axis=0, keepdims=True) == pytest.approx(expected_axis0.tolist())

    expected_axis1 = real_numpy.mean(real_numpy.array(data), axis=1, keepdims=True)
    assert _python_numeric_backend.mean(data, axis=1, keepdims=True) == pytest.approx(expected_axis1.tolist())

    vector = [2.0, 4.0, 6.0]
    assert _python_numeric_backend.mean(vector, axis=0) == pytest.approx(real_numpy.mean(real_numpy.array(vector), axis=0))
    assert _python_numeric_backend.mean(vector, axis=0, keepdims=True) == pytest.approx(
        real_numpy.mean(real_numpy.array(vector), axis=0, keepdims=True).tolist()
    )


def test_minimum_and_min_match_numpy():
    data = [[5.0, 1.0], [3.0, 4.0]]
    arr = np_stub.array(data)
    stub_reduce = np_stub.min(arr, axis=0, keepdims=True)
    expected_reduce = real_numpy.min(real_numpy.array(data), axis=0, keepdims=True)
    assert isinstance(stub_reduce, np_stub.ndarray)
    assert stub_reduce.to_list() == pytest.approx(expected_reduce.tolist())

    lhs = np_stub.array([1.0, 4.0, -2.0])
    rhs = np_stub.array([0.5, 5.0, -3.0])
    paired = np_stub.minimum(lhs, rhs)
    expected_pair = real_numpy.minimum(real_numpy.array([1.0, 4.0, -2.0]), real_numpy.array([0.5, 5.0, -3.0]))
    assert paired.to_list() == pytest.approx(expected_pair.tolist())


def test_maximum_backend_prefers_accelerator(monkeypatch: pytest.MonkeyPatch):
    def fake_backend(name: str, *args, **_kwargs):
        if name == "maximum":
            return _FakeArray([5.0, 7.0, 9.0])
        return None

    monkeypatch.setattr(np_stub, "_backend_call", fake_backend)

    lhs = np_stub.array([1.0, 5.0, 9.0])
    rhs = np_stub.array([5.0, 7.0, 3.0])
    result = np_stub.maximum(lhs, rhs)
    assert isinstance(result, np_stub.ndarray)
    assert result.to_list() == [5.0, 7.0, 9.0]


def test_minimum_backend_prefers_accelerator(monkeypatch: pytest.MonkeyPatch):
    def fake_backend(name: str, *args, **_kwargs):
        if name == "minimum":
            return _FakeArray([-2.0, 3.0, 4.0])
        return None

    monkeypatch.setattr(np_stub, "_backend_call", fake_backend)

    lhs = np_stub.array([1.0, -2.0, 6.0])
    rhs = np_stub.array([3.0, 4.0, 4.0])
    result = np_stub.minimum(lhs, rhs)
    assert isinstance(result, np_stub.ndarray)
    assert result.to_list() == [-2.0, 3.0, 4.0]


def test_linalg_solve_backend_conversion(monkeypatch: pytest.MonkeyPatch):
    def fake_backend(name: str, *args, **_kwargs):
        if name == "linalg_solve":
            return [0.5, 1.5]
        return None

    monkeypatch.setattr(np_stub, "_backend_call", fake_backend)

    coeffs = np_stub.array([[2.0, 0.0], [0.0, 4.0]])
    rhs = np_stub.array([1.0, 6.0])
    result = np_stub.linalg.solve(coeffs, rhs)
    assert isinstance(result, np_stub.ndarray)
    assert result.to_list() == pytest.approx([0.5, 1.5])


def test_linalg_cholesky_backend_conversion(monkeypatch: pytest.MonkeyPatch):
    def fake_backend(name: str, *args, **_kwargs):
        if name == "linalg_cholesky":
            return [[2.0, 0.0], [1.0, 3.0]]
        return None

    monkeypatch.setattr(np_stub, "_backend_call", fake_backend)

    mat = np_stub.array([[4.0, 2.0], [2.0, 10.0]])
    result = np_stub.linalg.cholesky(mat)
    assert isinstance(result, np_stub.ndarray)
    assert result.to_list() == pytest.approx([[2.0, 0.0], [1.0, 3.0]])


def test_python_backend_std_supports_ddof_and_keepdims():
    data = [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
    expected_axis0 = real_numpy.std(real_numpy.array(data), axis=0, ddof=1, keepdims=True)
    assert _python_numeric_backend.std(data, axis=0, ddof=1, keepdims=True) == pytest.approx(expected_axis0.tolist())

    expected_axis1 = real_numpy.std(real_numpy.array(data), axis=1, ddof=1, keepdims=True)
    assert _python_numeric_backend.std(data, axis=1, ddof=1, keepdims=True) == pytest.approx(expected_axis1.tolist())

    vector = [7.0, 9.0, 11.0]
    expected_vector = real_numpy.std(real_numpy.array(vector), axis=0, ddof=1, keepdims=True)
    assert _python_numeric_backend.std(vector, axis=0, ddof=1, keepdims=True) == pytest.approx(expected_vector.tolist())

    nan_result = _python_numeric_backend.std([5.0], axis=0, ddof=2)
    assert math.isnan(nan_result)


def test_python_backend_sum_reduce_keepdims_matches_numpy():
    data = [[1.0, 2.0], [3.0, 4.0]]
    expected_all = real_numpy.sum(real_numpy.array(data), keepdims=True)
    assert _python_numeric_backend.sum_reduce(data, keepdims=True) == pytest.approx(expected_all.tolist())

    expected_axis0 = real_numpy.sum(real_numpy.array(data), axis=0, keepdims=True)
    assert _python_numeric_backend.sum_reduce(data, axis=0, keepdims=True) == pytest.approx(expected_axis0.tolist())

    expected_axis1 = real_numpy.sum(real_numpy.array(data), axis=1, keepdims=True)
    assert _python_numeric_backend.sum_reduce(data, axis=1, keepdims=True) == pytest.approx(expected_axis1.tolist())

    vector = [1.0, 2.0, 3.0]
    expected_vector = real_numpy.sum(real_numpy.array(vector), axis=0, keepdims=True)
    assert _python_numeric_backend.sum_reduce(vector, axis=0, keepdims=True) == pytest.approx(expected_vector.tolist())
