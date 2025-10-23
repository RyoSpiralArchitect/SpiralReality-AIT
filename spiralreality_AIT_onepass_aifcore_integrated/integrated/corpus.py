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

# Keep dash punctuation that should terminate tokens alongside common ASCII/JP punctuation.
_BOUNDARY_PUNCT = ",.;。、「」…！？!?:;—–‒―‑-"
_INTRAWORD_HYPHENS = {"-", "‑"}


def _is_boundary_punct(ch: str, prev_ch: str, next_ch: str) -> bool:
    """Return ``True`` when ``ch`` should terminate the current segment."""

    if ch not in _BOUNDARY_PUNCT:
        return False

    if ch in _INTRAWORD_HYPHENS:
        if prev_ch.isalnum() and next_ch.isalnum():
            # Treat intra-word hyphen/dash as part of the token instead of
            # emitting it as its own boundary.
            return False
    return True

_REFLECTIVE_SAMPLES = reflective_samples()

TRAIN_TEXTS: tuple[str, ...] = tuple(sample.text for sample in _REFLECTIVE_SAMPLES)
TRAIN_SEGMENTS: tuple[List[str], ...] = tuple([list(sample.segments) for sample in _REFLECTIVE_SAMPLES])

_SEGMENT_MAP: Dict[str, List[str]] = {
    sample.text: list(sample.segments) for sample in iter_samples()
}


def naive_segments(text: str) -> List[str]:
    """Simple whitespace/punctuation split used across the demo and tests."""

    segments: List[str] = []
    token_buf: List[str] = []
    punct_buf: List[str] = []

    def flush_token() -> None:
        if token_buf:
            token = "".join(token_buf).strip()
            if token:
                segments.append(token)
            token_buf.clear()

    def flush_punct() -> None:
        if punct_buf:
            segments.append("".join(punct_buf))
            punct_buf.clear()

    last_index = len(text) - 1
    for idx, ch in enumerate(text):
        prev_ch = text[idx - 1] if idx > 0 else ""
        next_ch = text[idx + 1] if idx < last_index else ""

        if ch.isspace():
            flush_token()
            flush_punct()
            continue

        if _is_boundary_punct(ch, prev_ch, next_ch):
            flush_token()
            punct_buf.append(ch)

            is_next_boundary = False
            if idx < last_index:
                next_next = text[idx + 2] if idx + 1 < last_index else ""
                is_next_boundary = _is_boundary_punct(text[idx + 1], ch, next_next)
            if not is_next_boundary:
                flush_punct()
            continue

        flush_punct()
        token_buf.append(ch)

    flush_token()
    flush_punct()
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
