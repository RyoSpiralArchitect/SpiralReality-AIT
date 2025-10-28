from __future__ import annotations

import hashlib
import struct
import math
import unicodedata
from functools import lru_cache
from typing import Iterable

from .np_compat import np


def _stable_seed(name: str) -> np.random.SeedSequence:
    """Return a stable seed sequence derived from ``name``.

    Python's built-in :func:`hash` is salted per-process which breaks the
    determinism guarantees of :func:`seeded_vector`.  Instead we derive entropy
    from a Blake2b digest which is stable across interpreters and platforms and
    feed it into :class:`numpy.random.SeedSequence` for high-quality mixing.
    """

    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=32).digest()
    entropy = struct.unpack("<8I", digest)
    return np.random.SeedSequence(entropy)


@lru_cache(maxsize=16384)
def seeded_vector(name: str, dim: int = 64) -> np.ndarray:
    """Deterministic pseudo-random vector for a given name.

    The returned vector is stable across interpreter restarts regardless of the
    ``PYTHONHASHSEED`` configuration.
    """

    if dim <= 0:
        msg = "dim must be positive"
        raise ValueError(msg)
    rng = np.random.default_rng(_stable_seed(name))
    return rng.uniform(-1.0, 1.0, size=(dim,))


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

