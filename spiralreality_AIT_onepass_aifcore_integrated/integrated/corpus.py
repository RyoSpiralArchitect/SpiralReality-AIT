"""Shared training corpus and segmentation helpers for the demo/tests."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from .datasets import (
    CORPUS_LICENSE,
    REFLECTIVE_LANGUAGES,
    export_catalog,
    reflective_samples,
    iter_samples,
)

_BOUNDARY_PUNCT = ",.;。、「」…！？!?:;—‑-"

_REFLECTIVE_SAMPLES = reflective_samples()

TRAIN_TEXTS: tuple[str, ...] = tuple(sample.text for sample in _REFLECTIVE_SAMPLES)
TRAIN_SEGMENTS: tuple[List[str], ...] = tuple([list(sample.segments) for sample in _REFLECTIVE_SAMPLES])

_SEGMENT_MAP: Dict[str, List[str]] = {
    sample.text: list(sample.segments) for sample in iter_samples()
}


def naive_segments(text: str) -> List[str]:
    """Simple whitespace/punctuation split used across the demo and tests."""

    segments: List[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch.isspace():
            stripped = buf.strip()
            if stripped:
                segments.append(stripped)
            buf = ""
        elif ch in _BOUNDARY_PUNCT:
            stripped = buf[:-1].strip()
            if stripped:
                segments.append(stripped)
            segments.append(ch)
            buf = ""
    stripped = buf.strip()
    if stripped:
        segments.append(stripped)
    return segments


def teacher_segments(texts: Iterable[str] = TRAIN_TEXTS) -> List[List[str]]:
    """Return curated teacher segments, falling back to :func:`naive_segments` for unknown text."""

    out: List[List[str]] = []
    for text in texts:
        seg = _SEGMENT_MAP.get(text)
        if seg is not None:
            out.append(list(seg))
        else:
            out.append(naive_segments(text))
    return out


def register_teacher_segment(text: str, segments: Sequence[str]) -> None:
    """Register curated segments for ``text`` so :func:`teacher_segments` can reuse them.

    Parameters
    ----------
    text:
        The full input string to associate with ``segments``.
    segments:
        The ordered segmentation of ``text`` that should be treated as teacher
        supervision.
    """

    _SEGMENT_MAP[text] = list(segments)


def corpus_license() -> Dict[str, str]:
    """Expose licensing metadata for downstream publishing."""

    return dict(CORPUS_LICENSE)


def corpus_catalog(languages: Sequence[str] | None = None) -> Dict[str, object]:
    """Return a serialisable representation of the curated corpora."""

    return export_catalog(languages)


__all__ = [
    "TRAIN_TEXTS",
    "TRAIN_SEGMENTS",
    "REFLECTIVE_LANGUAGES",
    "corpus_catalog",
    "corpus_license",
    "naive_segments",
    "teacher_segments",
    "register_teacher_segment",
]
