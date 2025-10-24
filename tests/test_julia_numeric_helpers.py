from __future__ import annotations

import types

import numpy as np
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated import julia_numeric


class _FakeJuliaArray:
    def __init__(self, payload):
        self._payload = payload

    def tolist(self):
        return self._payload


def test_to_python_converts_numeric_scalars():
    assert julia_numeric._to_python(np.float32(3.25)) == pytest.approx(3.25)
    assert julia_numeric._to_python(np.complex64(1.0 - 2.0j)) == pytest.approx(1.0 - 2.0j)
    assert julia_numeric._to_python(np.int64(7)) == 7


def test_to_python_handles_arrays_and_nested_sequences():
    nested = _FakeJuliaArray([[np.float64(1.5), np.int32(2)], [3.25, 4.75]])
    result = julia_numeric._to_python(nested)
    expected = [[1.5, 2], [3.25, 4.75]]
    assert len(result) == len(expected)
    for row, expected_row in zip(result, expected):
        assert len(row) == len(expected_row)
        for value, expected_value in zip(row, expected_row):
            assert value == pytest.approx(expected_value)

    empty = _FakeJuliaArray([])
    assert julia_numeric._to_python(empty) == []

    tuple_result = julia_numeric._to_python((np.float32(5.5), np.float64(6.5)))
    assert len(tuple_result) == 2
    for value, expected_value in zip(tuple_result, (5.5, 6.5)):
        assert value == pytest.approx(expected_value)


def test_to_python_iterates_over_generic_iterables():
    class IterableWrapper:
        def __iter__(self):
            yield from (np.float64(1.0), np.float64(2.0), np.float64(3.0))

    converted = julia_numeric._to_python(IterableWrapper())
    assert len(converted) == 3
    for value, expected in zip(converted, (1.0, 2.0, 3.0)):
        assert value == pytest.approx(expected)


def test_axis_arg_uses_julia_nothing(monkeypatch: pytest.MonkeyPatch):
    sentinel = object()
    monkeypatch.setattr(julia_numeric, "jl", types.SimpleNamespace(nothing=sentinel))

    assert julia_numeric._axis_arg(None) is sentinel
    assert julia_numeric._axis_arg(2) == 2

