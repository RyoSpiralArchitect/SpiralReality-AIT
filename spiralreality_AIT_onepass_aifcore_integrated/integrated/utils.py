from __future__ import annotations

import hashlib
import math
import unicodedata
from functools import lru_cache
from typing import Iterable

from .np_compat import np


def _stable_seed(name: str) -> int:
    """Derive a stable 64-bit seed from the provided name."""

    digest = hashlib.blake2s(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


@lru_cache(maxsize=16384)
def seeded_vector(name: str, dim: int = 64) -> np.ndarray:
    """Deterministic pseudo-random vector for a given name.

    The seed is derived using a stable cryptographic hash so vectors remain
    consistent across interpreter runs regardless of Python's randomized hash
    salts.
    """

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

