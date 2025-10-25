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


def test_flash_attention_backend_tuple_results_are_wrapped():
    class DummyBackend:
        def __init__(self):
            self.calls = []

        def flash_attention(self, q, k, v, scale, bias, block_size, return_weights):
            self.calls.append((q, k, v, scale, bias, block_size, return_weights))
            context = [[1.0, 2.0], [3.0, 4.0]]
            weights = [[0.6, 0.4], [0.25, 0.75]]
            if return_weights:
                return context, weights
            return context

    original_backend = np_stub._ACCEL_BACKEND
    try:
        backend = DummyBackend()
        np_stub._ACCEL_BACKEND = backend
        q = np.array([[0.1, 0.2], [0.3, 0.4]])
        context, weights = np_stub.flash_attention(q, q, q, return_weights=True)

        assert backend.calls, "backend should receive the delegated call"
        call = backend.calls[0]
        np.testing.assert_allclose(np.asarray(context), [[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(np.asarray(weights), [[0.6, 0.4], [0.25, 0.75]])
        expected_scale = 1.0 / math.sqrt(q.shape[1])
        np.testing.assert_allclose(call[3], expected_scale)
        assert call[4] is None
        assert call[5] == 64
        assert call[6] is True
    finally:
        np_stub._ACCEL_BACKEND = original_backend


def test_flash_attention_preserves_value_dtype():
    rng = np.random.default_rng(11)
    q = rng.normal(size=(3, 5)).astype(np.float32)
    k = rng.normal(size=(3, 5)).astype(np.float32)
    v = rng.normal(size=(3, 7)).astype(np.float32)

    context = np_stub.flash_attention(q, k, v)

    assert context.dtype == np.float32


def test_flash_attention_backend_respects_value_dtype():
    class DummyBackend:
        def flash_attention(self, q, k, v, scale, bias, block_size, return_weights):
            context = np.ones_like(np.asarray(v, dtype=np.float64)) * 2
            if return_weights:
                weights = np.full((v.shape[0], k.shape[0]), 1.0 / k.shape[0], dtype=np.float64)
                return context, weights
            return context

    original_backend = np_stub._ACCEL_BACKEND
    try:
        np_stub._ACCEL_BACKEND = DummyBackend()
        q = np.ones((2, 3), dtype=np.float32)
        v = np.ones((2, 4), dtype=np.float32)
        context = np_stub.flash_attention(q, q, v)
        ctx, weights = np_stub.flash_attention(q, q, v, return_weights=True)

        assert context.dtype == np.float32
        assert ctx.dtype == np.float32
        assert weights.dtype == np.float64
    finally:
        np_stub._ACCEL_BACKEND = original_backend
