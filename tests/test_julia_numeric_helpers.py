from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated import julia_numeric


class _IterableOnly:
    def __iter__(self):
        yield 3.0
        yield 4.5


class _Opaque:
    pass


class _JuliaArray:
    def __init__(self, payload):
        self._payload = payload

    def tolist(self):
        return self._payload


@pytest.mark.parametrize(
    "value, expected",
    [
        (3.14, 3.14),
        (complex(1, -1), complex(1, -1)),
        (np.int64(7), 7),
        (_JuliaArray([[1, 2], [3, 4]]), [[1, 2], [3, 4]]),
        (_IterableOnly(), [3.0, 4.5]),
    ],
)
def test_to_python_normalises_known_types(value, expected):
    assert julia_numeric._to_python(value) == expected


def test_to_python_returns_original_for_opaque_values():
    sentinel = _Opaque()
    assert julia_numeric._to_python(sentinel) is sentinel


def test_axis_arg_without_julia(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(julia_numeric, "jl", None, raising=False)
    assert julia_numeric._axis_arg(None) is None
    assert julia_numeric._axis_arg(2) == 2


def test_axis_arg_with_julia(monkeypatch: pytest.MonkeyPatch):
    sentinel = object()
    monkeypatch.setattr(julia_numeric, "jl", SimpleNamespace(nothing=sentinel))
    assert julia_numeric._axis_arg(None) is sentinel
    assert julia_numeric._axis_arg(5) == 5
