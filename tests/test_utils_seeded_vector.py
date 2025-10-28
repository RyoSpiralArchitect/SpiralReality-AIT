"""Tests for the deterministic text feature utilities."""

from __future__ import annotations

import numpy as np

from spiralreality_AIT_onepass_aifcore_integrated.integrated.utils import seeded_vector


def test_seeded_vector_is_stable() -> None:
    seeded_vector.cache_clear()
    first = seeded_vector("spiral", dim=4)
    second = seeded_vector("spiral", dim=4)
    expected = np.array([-0.13348736, -0.29732992, 0.53546241, 0.96762868])
    assert np.allclose(first, expected)
    assert np.allclose(second, expected)


def test_seeded_vector_changes_with_name() -> None:
    seeded_vector.cache_clear()
    base = seeded_vector("spiral", dim=4)
    variant = seeded_vector("spiral-ai", dim=4)
    assert not np.allclose(base, variant)


def test_seeded_vector_respects_dimension() -> None:
    seeded_vector.cache_clear()
    vec = seeded_vector("spiral", dim=7)
    assert vec.shape == (7,)
