from __future__ import annotations

from typing import Any

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
    def fake_backend(name: str, *_args, **_kwargs):
        if name == "mean":
            return _FakeArray([1.0, 2.0, 3.0])
        return None

    monkeypatch.setattr(np_stub, "_backend_call", fake_backend)

    arr = np_stub.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    result = np_stub.mean(arr, axis=0)
    assert isinstance(result, np_stub.ndarray)
    assert result.to_list() == [1.0, 2.0, 3.0]


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


def test_tuple_axis_operations_match_numpy():
    values = [
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
    ]
    arr = np_stub.array(values)
    np_values = real_numpy.array(values)

    mean_stub = np_stub.mean(arr, axis=(0, 2))
    mean_expected = real_numpy.mean(np_values, axis=(0, 2))
    assert isinstance(mean_stub, np_stub.ndarray)
    assert mean_stub.to_list() == pytest.approx(mean_expected.tolist())

    mean_method = arr.mean(axis=[0, 2], keepdims=True)
    mean_method_expected = real_numpy.mean(np_values, axis=(0, 2), keepdims=True)
    assert isinstance(mean_method, np_stub.ndarray)
    assert mean_method.shape == mean_method_expected.shape
    assert mean_method.to_list() == pytest.approx(mean_method_expected.tolist())

    sum_stub = np_stub.sum(arr, axis=(0, 2), keepdims=True)
    sum_expected = real_numpy.sum(np_values, axis=(0, 2), keepdims=True)
    assert isinstance(sum_stub, np_stub.ndarray)
    assert sum_stub.shape == sum_expected.shape
    assert sum_stub.to_list() == pytest.approx(sum_expected.tolist())

    std_stub = arr.std(axis=(0, 2), ddof=1, keepdims=True)
    std_expected = real_numpy.std(np_values, axis=(0, 2), ddof=1, keepdims=True)
    assert isinstance(std_stub, np_stub.ndarray)
    assert std_stub.to_list() == pytest.approx(std_expected.tolist())

    median_stub = np_stub.median(arr, axis=(0, 2))
    median_expected = real_numpy.median(np_values, axis=(0, 2))
    assert isinstance(median_stub, np_stub.ndarray)
    assert median_stub.to_list() == pytest.approx(median_expected.tolist())


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
