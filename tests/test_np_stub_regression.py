from __future__ import annotations

import numpy as real_numpy
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated import np_stub


def _to_numpy(array: np_stub.ndarray) -> real_numpy.ndarray:
    return real_numpy.array(array.to_list(), dtype=array.dtype)


def _assert_close(stub_result, numpy_result, *, abs_tol=1e-10, rel_tol=1e-10):
    if isinstance(stub_result, np_stub.ndarray):
        real_numpy.testing.assert_allclose(
            stub_result._array, real_numpy.asarray(numpy_result), atol=abs_tol, rtol=rel_tol
        )
    else:
        assert stub_result == pytest.approx(numpy_result, abs=abs_tol, rel=rel_tol)


def test_mean_matches_numpy_default_axes():
    arr = np_stub.array([[1.5, 2.5, 3.5], [4.0, 5.0, 6.0]])
    stub = np_stub.mean(arr, axis=0)
    expected = real_numpy.mean(_to_numpy(arr), axis=0)
    _assert_close(stub, expected)


def test_std_and_var_match_numpy_axis_one():
    arr = np_stub.array([[0.5, 2.5, 4.5], [1.0, 3.0, 5.0], [1.5, 3.5, 5.5]])
    stub_std = np_stub.std(arr, axis=1, ddof=1, keepdims=True)
    stub_var = np_stub.var(arr, axis=1, ddof=1, keepdims=True)
    numpy_arr = _to_numpy(arr)
    expected_std = real_numpy.std(numpy_arr, axis=1, ddof=1, keepdims=True)
    expected_var = real_numpy.var(numpy_arr, axis=1, ddof=1, keepdims=True)
    _assert_close(stub_std, expected_std)
    _assert_close(stub_var, expected_var)


def test_logaddexp_matches_numpy():
    a = np_stub.array([-4.0, -2.0, 0.0, 2.0])
    b = np_stub.array([-3.0, -1.0, 1.0, 3.0])
    stub = np_stub.logaddexp(a, b)
    expected = real_numpy.logaddexp(_to_numpy(a), _to_numpy(b))
    _assert_close(stub, expected)


def test_clip_matches_numpy_scalar_bounds():
    arr = np_stub.array(real_numpy.linspace(-5.0, 5.0, num=11))
    stub = np_stub.clip(arr, -1.5, 2.5)
    expected = real_numpy.clip(_to_numpy(arr), -1.5, 2.5)
    _assert_close(stub, expected)

