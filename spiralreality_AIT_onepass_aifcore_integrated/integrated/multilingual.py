"""Utilities and curated corpora for multilingual boundary training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .corpus import (
    TRAIN_TEXTS,
    register_teacher_segment,
    teacher_segments,
)


@dataclass(frozen=True)
class LanguageCorpus:
    code: str
    texts: Tuple[str, ...]
    segments: Tuple[Tuple[str, ...], ...]


def _make_language_corpus(code: str, pairs: Iterable[Tuple[str, Sequence[str]]]) -> LanguageCorpus:
    texts: List[str] = []
    segs: List[Tuple[str, ...]] = []
    for text, seg in pairs:
        texts.append(text)
        segs.append(tuple(seg))
    return LanguageCorpus(code=code, texts=tuple(texts), segments=tuple(segs))


LANGUAGE_CORPORA: Dict[str, LanguageCorpus] = {
    "ja": _make_language_corpus(
        "ja",
        [
            (
                "境界判定モデルは多言語入力でも一貫して学習される。",
                ["境界", "判定", "モデル", "は", "多言語", "入力", "でも", "一貫", "して", "学習", "される", "。"],
            ),
            (
                "位相基底を共有すると変化点の感度が高まる。",
                ["位相", "基底", "を", "共有", "すると", "変化点", "の", "感度", "が", "高まる", "。"],
            ),
        ],
    ),
    "es": _make_language_corpus(
        "es",
        [
            (
                "La analista rastrea señales multilingües para ajustar los límites narrativos.",
                [
                    "La",
                    "analista",
                    "rastrea",
                    "señales",
                    "multilingües",
                    "para",
                    "ajustar",
                    "los",
                    "límites",
                    "narrativos",
                    ".",
                ],
            ),
            (
                "Modelos ligeros permiten entrenamiento estable sin depender de GPU.",
                [
                    "Modelos",
                    "ligeros",
                    "permiten",
                    "entrenamiento",
                    "estable",
                    "sin",
                    "depender",
                    "de",
                    "GPU",
                    ".",
                ],
            ),
        ],
    ),
    "fr": _make_language_corpus(
        "fr",
        [
            (
                "Les frontières apprises restent fiables même lorsque la langue change.",
                [
                    "Les",
                    "frontières",
                    "apprises",
                    "restent",
                    "fiables",
                    "même",
                    "lorsque",
                    "la",
                    "langue",
                    "change",
                    ".",
                ],
            ),
            (
                "Un encodeur de phase affine les indices locaux pour guider la planification.",
                [
                    "Un",
                    "encodeur",
                    "de",
                    "phase",
                    "affine",
                    "les",
                    "indices",
                    "locaux",
                    "pour",
                    "guider",
                    "la",
                    "planification",
                    ".",
                ],
            ),
        ],
    ),
    "de": _make_language_corpus(
        "de",
        [
            (
                "Grenzsignale stabilisieren den Plan auch bei mehrsprachigen Dialogen.",
                [
                    "Grenzsignale",
                    "stabilisieren",
                    "den",
                    "Plan",
                    "auch",
                    "bei",
                    "mehrsprachigen",
                    "Dialogen",
                    ".",
                ],
            ),
            (
                "Ein lernbarer Phasenfilter verstärkt Übergänge mit geringer Latenz.",
                [
                    "Ein",
                    "lernbarer",
                    "Phasenfilter",
                    "verstärkt",
                    "Übergänge",
                    "mit",
                    "geringer",
                    "Latenz",
                    ".",
                ],
            ),
        ],
    ),
    "zh": _make_language_corpus(
        "zh",
        [
            (
                "边界学生在多语言案例中保持高精度的分段预测。",
                ["边界", "学生", "在", "多语言", "案例", "中", "保持", "高精度", "的", "分段", "预测", "。"],
            ),
            (
                "相位编码帮助注意力在跨域文本上快速适应。",
                ["相位", "编码", "帮助", "注意力", "在", "跨域", "文本", "上", "快速", "适应", "。"],
            ),
        ],
    ),
}


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

