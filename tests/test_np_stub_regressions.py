from __future__ import annotations

from types import SimpleNamespace

import numpy as real_numpy
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated import (
    _np_stub_numpy as numpy_backend,
    np_stub,
)


def _to_numpy(value):
    if isinstance(value, np_stub.ndarray):
        return real_numpy.asarray(value.to_list(), dtype=real_numpy.float64)
    if isinstance(value, (list, tuple)):
        return real_numpy.asarray(value, dtype=real_numpy.float64)
    return value


def test_np_stub_matches_numpy_statistics():
    data = real_numpy.array([[1.0, 2.0, 3.0], [4.0, 5.5, 6.5]], dtype=real_numpy.float64)
    stub_arr = np_stub.array(data)

    overall = np_stub.mean(stub_arr)
    assert overall == pytest.approx(real_numpy.mean(data))

    axis0 = _to_numpy(np_stub.mean(stub_arr, axis=0))
    assert real_numpy.allclose(axis0, real_numpy.mean(data, axis=0))

    axis1_std = _to_numpy(np_stub.std(stub_arr, axis=1, ddof=1))
    assert real_numpy.allclose(axis1_std, real_numpy.std(data, axis=1, ddof=1))

    summed = _to_numpy(np_stub.sum(stub_arr, axis=1, keepdims=True))
    assert real_numpy.allclose(summed, real_numpy.sum(data, axis=1, keepdims=True))


def test_np_stub_elementwise_and_order_ops():
    vec = np_stub.array([3.0, -1.0, 4.5, 2.5])
    other = np_stub.array([-2.0, 5.0, 1.5, -3.0])

    maximum = _to_numpy(np_stub.maximum(vec, other))
    assert real_numpy.allclose(maximum, real_numpy.maximum(vec._array, other._array))

    diffed = _to_numpy(np_stub.diff(vec, order=2))
    assert real_numpy.allclose(diffed, real_numpy.diff(vec._array, n=2))

    clipped = _to_numpy(np_stub.clip(vec, -1.5, 2.0))
    assert real_numpy.allclose(clipped, real_numpy.clip(vec._array, -1.5, 2.0))

    logadd = _to_numpy(np_stub.logaddexp(vec, other))
    assert real_numpy.allclose(logadd, real_numpy.logaddexp(vec._array, other._array))


def test_np_stub_linalg_helpers_match_numpy():
    matrix = np_stub.array([[2.0, 1.0], [1.0, 3.0]])
    rhs = np_stub.array([1.0, 2.0])

    norm = np_stub.linalg_norm(matrix)
    assert norm == pytest.approx(real_numpy.linalg.norm(matrix._array))

    solved = _to_numpy(np_stub.linalg_solve(matrix, rhs))
    assert real_numpy.allclose(solved, real_numpy.linalg.solve(matrix._array, rhs._array))

    inv = _to_numpy(np_stub.linalg_inv(matrix))
    assert real_numpy.allclose(inv, real_numpy.linalg.inv(matrix._array))


def test_select_backend_requires_available_strict_backend(monkeypatch: pytest.MonkeyPatch):
    backend = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setattr(numpy_backend, "_cpp_numeric", backend, raising=False)
    name, module, strict = numpy_backend._select_backend("cpp-strict")
    assert name == "cpp"
    assert module is backend
    assert strict is True

    monkeypatch.setattr(numpy_backend, "_cpp_numeric", None, raising=False)
    with pytest.raises(RuntimeError):
        numpy_backend._select_backend("cpp-strict")

    monkeypatch.setattr(numpy_backend, "_cpp_numeric", None, raising=False)
    monkeypatch.setattr(numpy_backend, "_julia_numeric", None, raising=False)
    with pytest.raises(RuntimeError):
        numpy_backend._select_backend("auto-strict")


def test_backend_call_strict_enforces_failures(monkeypatch: pytest.MonkeyPatch):
    def _failing(*_args, **_kwargs):
        raise ValueError("backend failure")

    backend = SimpleNamespace(matmul=_failing)
    monkeypatch.setattr(np_stub, "_ACCEL_BACKEND", backend, raising=False)
    monkeypatch.setattr(np_stub, "_BACKEND_NAME", "cpp", raising=False)
    monkeypatch.setattr(np_stub, "_STRICT_BACKEND", True, raising=False)
    monkeypatch.setattr(np_stub, "STRICT_BACKEND", True, raising=False)

    with pytest.raises(RuntimeError):
        np_stub._backend_call("matmul", 1, 2)

    monkeypatch.setattr(np_stub, "_ACCEL_BACKEND", None, raising=False)
    with pytest.raises(RuntimeError):
        np_stub._backend_call("matmul", 1, 2)
