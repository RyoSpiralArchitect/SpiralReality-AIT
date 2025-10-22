from __future__ import annotations

from typing import Any

import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated import np_stub


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
