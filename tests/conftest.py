import numpy as _np
import pytest

_original_approx = pytest.approx


def _coerce_nested(expected):
    if isinstance(expected, (list, tuple)):
        if expected and any(isinstance(item, (list, tuple)) for item in expected):
            try:
                return _np.asarray(expected, dtype=float)
            except Exception:
                return [_coerce_nested(item) for item in expected]
    return expected


def approx(expected, *args, **kwargs):
    return _original_approx(_coerce_nested(expected), *args, **kwargs)


pytest.approx = approx
