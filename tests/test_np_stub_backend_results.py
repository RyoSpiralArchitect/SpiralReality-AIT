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


def test_mean_dtype_matches_numpy(monkeypatch: pytest.MonkeyPatch):
    def fail_backend(*_args, **_kwargs):
        raise AssertionError("backend should be skipped when dtype is provided")

    monkeypatch.setattr(np_stub, "_backend_call", fail_backend)

    values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    arr = np_stub.array(values)
    stub = np_stub.mean(arr, axis=0, dtype=real_numpy.float32)
    assert isinstance(stub, np_stub.ndarray)
    expected = real_numpy.mean(real_numpy.array(values), axis=0, dtype=real_numpy.float32)
    assert stub.to_list() == pytest.approx(expected.tolist())


def test_std_dtype_matches_numpy(monkeypatch: pytest.MonkeyPatch):
    def fail_backend(*_args, **_kwargs):
        raise AssertionError("backend should be skipped when dtype is provided")

    monkeypatch.setattr(np_stub, "_backend_call", fail_backend)

    values = [1.0, 3.0, 5.0, 7.0]
    arr = np_stub.array(values)
    stub = np_stub.std(arr, dtype=real_numpy.float32, ddof=1)
    expected = real_numpy.std(real_numpy.array(values), dtype=real_numpy.float32, ddof=1)
    assert stub == pytest.approx(float(expected))


def test_sum_dtype_matches_numpy(monkeypatch: pytest.MonkeyPatch):
    def fail_backend(*_args, **_kwargs):
        raise AssertionError("backend should be skipped when dtype is provided")

    monkeypatch.setattr(np_stub, "_backend_call", fail_backend)

    values = [[1, 2, 3], [4, 5, 6]]
    arr = np_stub.array(values)
    stub = np_stub.sum(arr, axis=1, dtype=real_numpy.float32)
    assert isinstance(stub, np_stub.ndarray)
    expected = real_numpy.sum(real_numpy.array(values), axis=1, dtype=real_numpy.float32)
    assert stub.to_list() == pytest.approx(expected.tolist())


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
