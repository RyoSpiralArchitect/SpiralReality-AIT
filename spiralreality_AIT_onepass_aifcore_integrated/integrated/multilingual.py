"""Utilities and curated corpora for multilingual boundary training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .corpus import (
    TRAIN_TEXTS,
    register_teacher_segment,
    teacher_segments,
)
from .datasets import CorpusSample, languages as dataset_languages, samples_for_language


@dataclass(frozen=True)
class LanguageCorpus:
    code: str
    texts: Tuple[str, ...]
    segments: Tuple[Tuple[str, ...], ...]
    samples: Tuple[CorpusSample, ...]


def _make_language_corpus(code: str) -> LanguageCorpus:
    samples = tuple(
        sample for sample in samples_for_language(code) if sample.group == "multilingual"
    )
    texts = tuple(sample.text for sample in samples)
    segs = tuple(sample.segments for sample in samples)
    return LanguageCorpus(code=code, texts=texts, segments=segs, samples=samples)


LANGUAGE_CORPORA: Dict[str, LanguageCorpus] = {}
for code in dataset_languages():
    corpus = _make_language_corpus(code)
    if corpus.texts:
        LANGUAGE_CORPORA[code] = corpus


AVAILABLE_LANGUAGES: Tuple[str, ...] = tuple(sorted(LANGUAGE_CORPORA.keys()))


def build_multilingual_corpus(
    languages: Sequence[str] | None = None,
    *,
    include_reflective: bool = True,
    shuffle: bool = True,
    seed: int | None = None,
) -> Tuple[List[str], List[List[str]], List[str]]:
    """Assemble training texts, segments, and language tags.

    Parameters
    ----------
    languages:
        Language codes to include. ``None`` selects all curated multilingual corpora.
    include_reflective:
        If ``True`` the default reflective English/Japanese corpus is included.
    shuffle:
        When ``True`` the combined dataset is shuffled deterministically using ``seed``.
    seed:
        Seed forwarded to :mod:`random` for dataset shuffling. ``None`` keeps the original
        ordering when ``shuffle`` is disabled.

    Returns
    -------
    texts, segments, tags:
        Lists of texts, segmentation labels, and per-text language tags. The tags match the
        ``languages`` argument with ``"reflective"`` reserved for the base corpus.
    """

    if languages is None:
        languages = AVAILABLE_LANGUAGES

    combined: List[Tuple[str, str, List[str]]] = []

    if include_reflective:
        base_texts = list(TRAIN_TEXTS)
        base_segments = teacher_segments(base_texts)
        for text, seg in zip(base_texts, base_segments):
            combined.append(("reflective", text, list(seg)))

    for code in languages:
        corpus = LANGUAGE_CORPORA.get(code)
        if corpus is None:
            raise ValueError(f"Unknown language code: {code}")
        for text, seg in zip(corpus.texts, corpus.segments):
            seg_list = list(seg)
            register_teacher_segment(text, seg_list)
            combined.append((code, text, seg_list))

    if shuffle and len(combined) > 1:
        rng = random.Random(seed)
        rng.shuffle(combined)

    texts = [text for _, text, _ in combined]
    segments = [seg for _, _, seg in combined]
    tags = [lang for lang, _, _ in combined]
    return texts, segments, tags


def language_histogram(tags: Sequence[str]) -> Dict[str, int]:
    """Count occurrences of each language tag."""

    hist: Dict[str, int] = {}
    for tag in tags:
        hist[tag] = hist.get(tag, 0) + 1
    return dict(sorted(hist.items(), key=lambda item: item[0]))


def language_statistics(
    texts: Sequence[str],
    segments: Sequence[Sequence[str]],
    tags: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Compute aggregate statistics for the multilingual dataset.

    Parameters
    ----------
    texts, segments, tags:
        Parallel sequences describing the dataset produced by
        :func:`build_multilingual_corpus`.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Per-language statistics including the sample ``count`` plus mean character
        lengths and segmentation granularity.
    """

    if not (len(texts) == len(segments) == len(tags)):
        raise ValueError("texts, segments, and tags must share the same length")

    aggregates: Dict[str, Dict[str, float]] = {}
    for text, seg, tag in zip(texts, segments, tags):
        seg_list = list(seg)
        char_count = float(len(text))
        token_count = float(len(seg_list))
        per_token = char_count if token_count == 0 else char_count / token_count
        if tag not in aggregates:
            aggregates[tag] = {
                "count": 0,
                "sum_chars": 0.0,
                "sum_tokens": 0.0,
                "sum_chars_per_token": 0.0,
            }
        agg = aggregates[tag]
        agg["count"] += 1
        agg["sum_chars"] += char_count
        agg["sum_tokens"] += token_count
        agg["sum_chars_per_token"] += per_token

    stats: Dict[str, Dict[str, float]] = {}
    for tag, agg in aggregates.items():
        count = int(agg["count"]) if agg["count"] else 0
        if count == 0:
            stats[tag] = {
                "count": 0.0,
                "mean_chars": 0.0,
                "mean_tokens": 0.0,
                "mean_chars_per_token": 0.0,
            }
            continue
        stats[tag] = {
            "count": float(count),
            "mean_chars": agg["sum_chars"] / count,
            "mean_tokens": agg["sum_tokens"] / count,
            "mean_chars_per_token": agg["sum_chars_per_token"] / count,
        }
    return dict(sorted(stats.items(), key=lambda item: item[0]))


__all__ = [
    "AVAILABLE_LANGUAGES",
    "LANGUAGE_CORPORA",
    "LanguageCorpus",
    "build_multilingual_corpus",
    "language_histogram",
    "language_statistics",
]

