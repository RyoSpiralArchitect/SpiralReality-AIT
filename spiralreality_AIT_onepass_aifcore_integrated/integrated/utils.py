from __future__ import annotations

import hashlib
import math
import unicodedata
from functools import lru_cache
from typing import Iterable, Union

from numpy.typing import DTypeLike

from .np_compat import np


def _normalize_name(name: Union[str, bytes, bytearray]) -> bytes:
    """Convert ``name`` into a canonical byte representation."""

    if isinstance(name, str):
        return name.encode("utf-8")
    return bytes(name)


def _stable_seed(name: bytes) -> int:
    """Return a stable 64-bit seed derived from ``name``.

    Python's built-in :func:`hash` is salted per-process which breaks the
    determinism guarantees of :func:`seeded_vector`.  Instead we derive a seed
    from the Blake2b digest which is stable across interpreters and platforms.
    """

    digest = hashlib.blake2b(name, digest_size=16).digest()
    return int.from_bytes(digest[:8], "little")


@lru_cache(maxsize=16384)
def _seeded_vector_cached(name: bytes, dim: int, dtype_str: str) -> np.ndarray:
    rng = np.random.default_rng(_stable_seed(name))
    vector = rng.uniform(-1.0, 1.0, size=(dim,))
    vector = vector.astype(np.dtype(dtype_str), copy=False)
    vector.setflags(write=False)
    return vector


def seeded_vector(
    name: Union[str, bytes, bytearray],
    dim: int = 64,
    dtype: DTypeLike = np.float32,
) -> np.ndarray:
    """Deterministic pseudo-random vector for a given name.

    The returned vector is stable across interpreter restarts regardless of the
    ``PYTHONHASHSEED`` configuration.  The optional ``dtype`` argument allows
    callers to control the floating point precision of the generated vector.
    Returned arrays are read-only to prevent accidental cache corruption.
    """

    if dim <= 0:
        raise ValueError("dim must be positive")

    dtype_str = np.dtype(dtype).str
    return _seeded_vector_cached(_normalize_name(name), dim, dtype_str)


def _seeded_vector_cache_clear() -> None:
    _seeded_vector_cached.cache_clear()


seeded_vector.cache_clear = _seeded_vector_cache_clear  # type: ignore[attr-defined]
seeded_vector.cache_info = _seeded_vector_cached.cache_info  # type: ignore[attr-defined]


def unit(v: Iterable[float]) -> np.ndarray:
    arr = np.array(list(v), dtype=float)
    n = np.linalg.norm(arr)
    return arr / (n + 1e-8)


def is_space(ch: str) -> bool:
    return ch.isspace()


def is_punct(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P")


def is_latin(ch: str) -> bool:
    return "LATIN" in unicodedata.name(ch, "")


def is_kana(ch: str) -> bool:
    name = unicodedata.name(ch, "")
    return "KATAKANA" in name or "HIRAGANA" in name


def is_cjk(ch: str) -> bool:
    name = unicodedata.name(ch, "")
    return "CJK" in name or "IDEOGRAPH" in name


def sigmoid(x: float) -> float:
    x = max(min(x, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-x))


def softplus(x: float) -> float:
    if x > 20:
        return x
    return math.log1p(math.exp(x))


def clipped_log(x: float) -> float:
    return math.log(max(x, 1e-12))

