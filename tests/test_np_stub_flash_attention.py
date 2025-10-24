import math

import numpy as np

from spiralreality_AIT_onepass_aifcore_integrated.integrated import np_stub


def _naive_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float, bias: np.ndarray | None = None):
    scores = q @ k.T * scale
    if bias is not None:
        scores = scores + bias
    row_max = scores.max(axis=1, keepdims=True)
    weights = np.exp(scores - row_max)
    denom = weights.sum(axis=1, keepdims=True)
    weights = weights / np.maximum(denom, 1e-12)
    context = weights @ v
    return context, weights


def test_flash_attention_matches_numpy_with_bias():
    rng = np.random.default_rng(42)
    q = rng.normal(size=(4, 6))
    k = rng.normal(size=(4, 6))
    v = rng.normal(size=(4, 6))
    bias = rng.normal(size=(4, 4)) * 0.1
    scale = 0.25

    context, weights = np_stub.flash_attention(
        q, k, v, scale=scale, bias=bias, block_size=2, return_weights=True
    )
    expected_context, expected_weights = _naive_attention(q, k, v, scale, bias)

    np.testing.assert_allclose(np.asarray(context), expected_context, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(weights), expected_weights, rtol=1e-10, atol=1e-10)


def test_flash_attention_default_scale_matches_manual():
    rng = np.random.default_rng(7)
    q = rng.normal(size=(3, 8))
    k = rng.normal(size=(3, 8))
    v = rng.normal(size=(3, 8))
    default_scale = 1.0 / math.sqrt(q.shape[1])

    context = np_stub.flash_attention(q, k, v)
    expected_context, _ = _naive_attention(q, k, v, default_scale)

    np.testing.assert_allclose(np.asarray(context), expected_context, rtol=1e-10, atol=1e-10)


def test_flash_attention_mismatched_key_value_length_raises():
    q = np.ones((2, 4))
    k = np.ones((3, 4))
    v = np.ones((2, 4))

    with np.testing.assert_raises(ValueError):
        np_stub.flash_attention(q, k, v)
