import builtins

import numpy as np
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.utils import (
    seeded_vector,
)


def test_seeded_vector_is_deterministic() -> None:
    seeded_vector.cache_clear()
    reference = seeded_vector("unit-test", dim=8)
    seeded_vector.cache_clear()
    repeat = seeded_vector("unit-test", dim=8)
    np.testing.assert_allclose(reference, repeat)


def test_seeded_vector_does_not_depend_on_python_hash(monkeypatch) -> None:
    seeded_vector.cache_clear()

    def _failing_hash(_: object) -> int:
        raise AssertionError("unexpected hash call")

    monkeypatch.setattr(builtins, "hash", _failing_hash)
    vector = seeded_vector("unit-test", dim=4)
    assert vector.shape == (4,)


def test_seeded_vector_rejects_non_positive_dim() -> None:
    seeded_vector.cache_clear()

    with pytest.raises(ValueError, match="dim must be positive"):
        seeded_vector("unit-test", dim=0)
