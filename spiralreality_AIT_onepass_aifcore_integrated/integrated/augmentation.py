"""Text perturbation utilities for robustness benchmarking."""

from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class AugmentedSample:
    """A perturbed sample paired with updated segmentation labels."""

    text: str
    segments: List[str]
    tag: str


_DEFAULT_DIALECT_MAP: Dict[str, Mapping[str, str]] = {
    "en": {
        "analysis": "analyse",
        "calibrating": "tuning",
        "sustained": "steady",
        "diagnostics": "telemetry",
        "reflective": "meta",
        "segmentation": "chunking",
        "planner": "schemer",
    },
    "es": {
        "modelos": "modelitos",
        "entrenamiento": "adiestramiento",
        "estable": "firme",
        "señales": "señalitas",
    },
    "fr": {
        "frontières": "limites",
        "planification": "planif",
        "indices": "indices locaux",
    },
    "de": {
        "plan": "fahrplan",
        "übergänge": "wechsel",
        "grenzsignale": "signalschwellen",
    },
    "ja": {
        "境界": "ボーダー",
        "感度": "センシティビティ",
        "仮説": "仮定",
        "更新": "アップデート",
    },
    "zh": {
        "分段": "切分",
        "注意力": "关注",
        "预测": "推断",
    },
}


_VOWELS = set("aeiou" "áéíóú" "äëïöü" "àèìòù" "âêîôû")


class PerturbationGenerator:
    """Create noisy, dialectal, and tempo-altered variants of segmented text."""

    def __init__(
        self,
        *,
        dialect_map: Mapping[str, Mapping[str, str]] | None = None,
        seed: int | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.dialect_map: Dict[str, Mapping[str, str]] = {
            lang: dict(mapping) for lang, mapping in (dialect_map or _DEFAULT_DIALECT_MAP).items()
        }

    def noise_segments(self, segments: Sequence[str], noise_level: float = 0.08) -> List[str]:
        """Inject lightweight character noise into segments."""

        out: List[str] = []
        for token in segments:
            if not token.strip():
                out.append(token)
                continue
            if self.rng.random() >= noise_level:
                out.append(token)
                continue
            out.append(self._perturb_token(token))
        return out

    def dialect_segments(self, language: str, segments: Sequence[str]) -> List[str]:
        """Apply per-language lexical substitutions."""

        mapping = self.dialect_map.get(language.lower(), {})
        out: List[str] = []
        for token in segments:
            stripped = token.strip()
            if not stripped:
                out.append(token)
                continue
            lower = stripped.lower()
            replacement = mapping.get(lower)
            if replacement is None:
                out.append(token)
                continue
            if stripped[0].isupper():
                replacement = replacement.capitalize()
            # Preserve surrounding whitespace if the token carried it explicitly
            prefix = token[: len(token) - len(token.lstrip())]
            suffix = token[len(token.rstrip()):]
            out.append(f"{prefix}{replacement}{suffix}")
        return out

    def tempo_segments(self, segments: Sequence[str], factor: float) -> List[str]:
        """Simulate speech tempo changes by elongating or compressing tokens."""

        slower = factor < 1.0
        out: List[str] = []
        for token in segments:
            if not token.strip():
                out.append(token)
                continue
            if slower:
                out.append(self._elongate(token))
            else:
                out.append(self._compress(token))
        return out

    def generate_variants(
        self,
        text: str,
        segments: Sequence[str],
        *,
        language: str,
        noise_level: float = 0.08,
        tempo_factors: Tuple[float, float] = (0.85, 1.15),
    ) -> List[AugmentedSample]:
        """Produce noisy, dialectal, and tempo-altered samples."""

        segments = list(segments)
        variants: List[AugmentedSample] = []

        noisy_segments = self.noise_segments(segments, noise_level=noise_level)
        variants.append(
            AugmentedSample(
                text=self._rebuild_text(text, segments, noisy_segments),
                segments=list(noisy_segments),
                tag="noise",
            )
        )

        dialect_segments = self.dialect_segments(language, segments)
        variants.append(
            AugmentedSample(
                text=self._rebuild_text(text, segments, dialect_segments),
                segments=list(dialect_segments),
                tag="dialect",
            )
        )

        for tempo, label in zip(tempo_factors, ("tempo_slow", "tempo_fast")):
            tempo_segments = self.tempo_segments(segments, tempo)
            variants.append(
                AugmentedSample(
                    text=self._rebuild_text(text, segments, tempo_segments),
                    segments=list(tempo_segments),
                    tag=label,
                )
            )

        return variants

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _perturb_token(self, token: str) -> str:
        if len(token) <= 1:
            return token
        op = self.rng.choice(["drop", "repeat", "accent"])
        idx = self.rng.randrange(len(token))
        if op == "drop" and len(token) > 2:
            return token[:idx] + token[idx + 1 :]
        if op == "repeat":
            return token[: idx + 1] + token[idx] + token[idx + 1 :]
        return token[:idx] + self._accent_char(token[idx]) + token[idx + 1 :]

    def _accent_char(self, ch: str) -> str:
        accents: Dict[str, str] = {
            "a": "á",
            "e": "é",
            "i": "í",
            "o": "ó",
            "u": "ú",
            "s": "š",
            "n": "ñ",
        }
        lower = ch.lower()
        if lower in accents:
            accented = accents[lower]
            return accented.upper() if ch.isupper() else accented
        return ch.upper() if ch.islower() else ch.lower()

    def _elongate(self, token: str) -> str:
        if len(token) <= 2:
            return token
        last = token[-1]
        if last.isalpha():
            return token + last.lower()
        if unicodedata.category(last).startswith("P"):
            return token + "…"
        return token + last

    def _compress(self, token: str) -> str:
        if len(token) <= 3:
            return token
        if any(ch.lower() in _VOWELS for ch in token):
            head = token[0]
            tail = [ch for ch in token[1:] if ch.lower() not in _VOWELS]
            collapsed = head + "".join(tail)
            return collapsed if len(collapsed) >= 2 else token
        if token.endswith("ー"):
            return token[:-1]
        return token.rstrip("。") or token

    def _rebuild_text(
        self,
        original_text: str,
        original_segments: Sequence[str],
        new_segments: Sequence[str],
    ) -> str:
        if len(original_segments) != len(new_segments):
            raise ValueError("Segment counts must match to rebuild text")
        pieces: List[str] = []
        cursor = 0
        for orig, new in zip(original_segments, new_segments):
            idx = original_text.find(orig, cursor)
            if idx < 0:
                idx = cursor
            pieces.append(original_text[cursor:idx])
            pieces.append(new)
            cursor = idx + len(orig)
        pieces.append(original_text[cursor:])
        return "".join(pieces)


__all__ = ["AugmentedSample", "PerturbationGenerator"]
