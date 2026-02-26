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
_BOUNDARY_PUNCT = frozenset(
    {
        ",",
        ".",
        ";",
        ":",
        "!",
        "?",
        "…",
        "—",
        "–",
        "‒",
        "―",
        "‑",
        "-",
        "‐",
        "。",
        "、",
        "！",
        "？",
        "「",
        "」",
        "『",
        "』",
        "《",
        "》",
        "〈",
        "〉",
        "・",
        "：",
        "；",
        "，",
        "．",
        "｡",
        "؟",
        "،",
        "؛",
        "۔",
        "।",
        "॥",
        "ฯ",
        "๚",
        "።",
        "፣",
    }
)
_INTRAWORD_HYPHENS = frozenset({"-", "‑", "‐"})

# Python does not treat zero-width spaces or half-spaces as whitespace, so capture them explicitly.
_EXPLICIT_WHITESPACE = {"\u200b", "\u200c", "\ufeff"}


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


def _materialize_segments(text: str, segments: Sequence[str]) -> List[str]:
    """Convert a token sequence into text-covering segments.

    Curated segments are stored in a token form that may omit whitespace. For the
    boundary learner, however, we need segments that partition the original text
    so boundary indices line up with character positions.

    The strategy below aligns each token sequentially in ``text`` and attaches
    any following whitespace to the token segment. If alignment fails we fall
    back to returning the entire text as a single segment.
    """

    if not text:
        return []
    if not segments:
        return [text]

    out: List[str] = []
    pos = 0
    length = len(text)

    def is_ws(ch: str) -> bool:
        return ch.isspace() or ch in _EXPLICIT_WHITESPACE

    while pos < length and is_ws(text[pos]):
        start = pos
        while pos < length and is_ws(text[pos]):
            pos += 1
        out.append(text[start:pos])

    for tok in segments:
        tok = "" if tok is None else str(tok)
        if not tok:
            continue

        if not text.startswith(tok, pos):
            found = text.find(tok, pos)
            if found < 0:
                return [text]
            if found > pos:
                out.append(text[pos:found])
            pos = found
            if not text.startswith(tok, pos):
                return [text]

        end = pos + len(tok)
        seg = text[pos:end]
        pos = end

        ws_start = pos
        while pos < length and is_ws(text[pos]):
            pos += 1
        seg += text[ws_start:pos]
        out.append(seg)

    if pos < length:
        out.append(text[pos:])

    if "".join(out) != text:
        return [text]
    return [seg for seg in out if seg]

_REFLECTIVE_SAMPLES = reflective_samples()

TRAIN_TEXTS: tuple[str, ...] = tuple(sample.text for sample in _REFLECTIVE_SAMPLES)
TRAIN_SEGMENTS: tuple[List[str], ...] = tuple(
    _materialize_segments(sample.text, list(sample.segments))
    for sample in _REFLECTIVE_SAMPLES
)

_SEGMENT_MAP: Dict[str, List[str]] = {
    sample.text: list(sample.segments) for sample in iter_samples()
}


def _naive_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    token_buf: List[str] = []
    punct_buf: List[str] = []

    def flush_token() -> None:
        if token_buf:
            token = "".join(token_buf)
            if token:
                tokens.append(token)
            token_buf.clear()

    def flush_punct() -> None:
        if punct_buf:
            tokens.append("".join(punct_buf))
            punct_buf.clear()

    last_index = len(text) - 1
    for idx, ch in enumerate(text):
        prev_ch = text[idx - 1] if idx > 0 else ""
        next_ch = text[idx + 1] if idx < last_index else ""

        if ch.isspace() or ch in _EXPLICIT_WHITESPACE:
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
    return tokens


def naive_segments(text: str) -> List[str]:
    """Simple whitespace/punctuation split used across the demo and tests."""

    return _naive_tokens(text)


def teacher_segments(texts: Iterable[str] = TRAIN_TEXTS) -> List[List[str]]:
    """Return curated teacher segments as text-covering supervision.

    Teacher segments are stored in a tokenised form that may omit whitespace.
    This helper materialises them into segments that partition the original
    input string so boundary indices line up with character positions.
    """

    out: List[List[str]] = []
    for text in texts:
        seg = _SEGMENT_MAP.get(text)
        tokens = seg if seg is not None else naive_segments(text)
        out.append(_materialize_segments(text, tokens))
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
