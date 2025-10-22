"""Curated multilingual text corpora with licensing metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple


@dataclass(frozen=True)
class CorpusSample:
    """A single text sample with segmentation metadata."""

    language: str
    text: str
    segments: Tuple[str, ...]
    group: str


# Creative Commons Attribution 4.0 International metadata for all corpora.
CORPUS_LICENSE: Dict[str, str] = {
    "id": "CC-BY-4.0",
    "name": "Creative Commons Attribution 4.0 International",
    "url": "https://creativecommons.org/licenses/by/4.0/",
    "attribution": "SpiralReality AIT authors",
    "notes": (
        "Synthetic reflective narratives curated for the SpiralReality one-pass AIT demo. "
        "The texts were authored for this repository and may be redistributed under CC-BY-4.0."
    ),
}


CORPUS_SAMPLES: Tuple[CorpusSample, ...] = (
    # Reflective English anchors
    CorpusSample(
        language="en",
        group="reflective",
        text="Bob re-examined Carol's motives and updated his provisional evaluation.",
        segments=(
            "Bob",
            "re",
            "-",
            "examined",
            "Carol's",
            "motives",
            "and",
            "updated",
            "his",
            "provisional",
            "evaluation",
            ".",
        ),
    ),
    CorpusSample(
        language="en",
        group="reflective",
        text="Avoid premature closure; maintain hypotheses and update them with evidence.",
        segments=(
            "Avoid",
            "premature",
            "closure",
            ";",
            "maintain",
            "hypotheses",
            "and",
            "update",
            "them",
            "with",
            "evidence",
            ".",
        ),
    ),
    CorpusSample(
        language="en",
        group="reflective",
        text="Meta-analysis of sensor feeds prevents overfitting to the first plausible story.",
        segments=(
            "Meta",
            "-",
            "analysis",
            "of",
            "sensor",
            "feeds",
            "prevents",
            "overfitting",
            "to",
            "the",
            "first",
            "plausible",
            "story",
            ".",
        ),
    ),
    CorpusSample(
        language="en",
        group="reflective",
        text="Iterative reframing of witness notes keeps latent clusters responsive to nuance.",
        segments=(
            "Iterative",
            "reframing",
            "of",
            "witness",
            "notes",
            "keeps",
            "latent",
            "clusters",
            "responsive",
            "to",
            "nuance",
            ".",
        ),
    ),
    CorpusSample(
        language="en",
        group="reflective",
        text="Systematic journaling of counterfactuals stabilizes the agent's decision frontier.",
        segments=(
            "Systematic",
            "journaling",
            "of",
            "counterfactuals",
            "stabilizes",
            "the",
            "agent's",
            "decision",
            "frontier",
            ".",
        ),
    ),
    CorpusSample(
        language="en",
        group="reflective",
        text="Layered attention summaries discourage impulsive segmentation during cold starts.",
        segments=(
            "Layered",
            "attention",
            "summaries",
            "discourage",
            "impulsive",
            "segmentation",
            "during",
            "cold",
            "starts",
            ".",
        ),
    ),
    CorpusSample(
        language="en",
        group="reflective",
        text="Sustained reflective questioning keeps the analyst from freezing the latent plan too soon.",
        segments=(
            "Sustained",
            "reflective",
            "questioning",
            "keeps",
            "the",
            "analyst",
            "from",
            "freezing",
            "the",
            "latent",
            "plan",
            "too",
            "soon",
            ".",
        ),
    ),
    CorpusSample(
        language="en",
        group="reflective",
        text="Calibrating priors against delayed confirmations avoids premature pruning of actions.",
        segments=(
            "Calibrating",
            "priors",
            "against",
            "delayed",
            "confirmations",
            "avoids",
            "premature",
            "pruning",
            "of",
            "actions",
            ".",
        ),
    ),
    CorpusSample(
        language="en",
        group="reflective",
        text="Contingency-aware sampling lets the bridge respect rare but critical micro-boundaries.",
        segments=(
            "Contingency",
            "-",
            "aware",
            "sampling",
            "lets",
            "the",
            "bridge",
            "respect",
            "rare",
            "but",
            "critical",
            "micro",
            "-",
            "boundaries",
            ".",
        ),
    ),
    CorpusSample(
        language="en",
        group="reflective",
        text="Slow breathing, careful parsing, then align evidence with provisional policies.",
        segments=(
            "Slow",
            "breathing",
            ",",
            "careful",
            "parsing",
            ",",
            "then",
            "align",
            "evidence",
            "with",
            "provisional",
            "policies",
            ".",
        ),
    ),
    CorpusSample(
        language="en",
        group="reflective",
        text="Maintain a rolling scratchpad; revise boundaries when fresh observations contradict assumptions.",
        segments=(
            "Maintain",
            "a",
            "rolling",
            "scratchpad",
            ";",
            "revise",
            "boundaries",
            "when",
            "fresh",
            "observations",
            "contradict",
            "assumptions",
            ".",
        ),
    ),
    CorpusSample(
        language="en",
        group="reflective",
        text="Socratic prompts elicit clarifications that re-open segments closed by cognitive fatigue.",
        segments=(
            "Socratic",
            "prompts",
            "elicit",
            "clarifications",
            "that",
            "re",
            "-",
            "open",
            "segments",
            "closed",
            "by",
            "cognitive",
            "fatigue",
            ".",
        ),
    ),
    CorpusSample(
        language="en",
        group="reflective",
        text="Cross-lingual glossaries ensure the encoder does not conflate idioms with hard stops.",
        segments=(
            "Cross",
            "-",
            "lingual",
            "glossaries",
            "ensure",
            "the",
            "encoder",
            "does",
            "not",
            "conflate",
            "idioms",
            "with",
            "hard",
            "stops",
            ".",
        ),
    ),
    CorpusSample(
        language="en",
        group="reflective",
        text="Goal-conditioned rehearsal simulates failure cases before committing to gating decisions.",
        segments=(
            "Goal",
            "-",
            "conditioned",
            "rehearsal",
            "simulates",
            "failure",
            "cases",
            "before",
            "committing",
            "to",
            "gating",
            "decisions",
            ".",
        ),
    ),
    CorpusSample(
        language="en",
        group="reflective",
        text="Streaming diagnostics reveal when the CRF head saturates and needs regularization.",
        segments=(
            "Streaming",
            "diagnostics",
            "reveal",
            "when",
            "the",
            "CRF",
            "head",
            "saturates",
            "and",
            "needs",
            "regularization",
            ".",
        ),
    ),
    CorpusSample(
        language="en",
        group="reflective",
        text="Noise-robust embeddings resist the temptation to snap to the majority class boundary.",
        segments=(
            "Noise",
            "-",
            "robust",
            "embeddings",
            "resist",
            "the",
            "temptation",
            "to",
            "snap",
            "to",
            "the",
            "majority",
            "class",
            "boundary",
            ".",
        ),
    ),
    CorpusSample(
        language="en",
        group="reflective",
        text="Phase-aware masking prevents the planner from extrapolating on stale latent states.",
        segments=(
            "Phase",
            "-",
            "aware",
            "masking",
            "prevents",
            "the",
            "planner",
            "from",
            "extrapolating",
            "on",
            "stale",
            "latent",
            "states",
            ".",
        ),
    ),
    CorpusSample(
        language="en",
        group="reflective",
        text="Manual audits of auto-segmented transcripts surface drift before metrics collapse.",
        segments=(
            "Manual",
            "audits",
            "of",
            "auto",
            "-",
            "segmented",
            "transcripts",
            "surface",
            "drift",
            "before",
            "metrics",
            "collapse",
            ".",
        ),
    ),
    CorpusSample(
        language="en",
        group="reflective",
        text="Align observations, plans, and critiques; triangulated narratives yield resilient cuts.",
        segments=(
            "Align",
            "observations",
            ",",
            "plans",
            ",",
            "and",
            "critiques",
            ";",
            "triangulated",
            "narratives",
            "yield",
            "resilient",
            "cuts",
            ".",
        ),
    ),
    CorpusSample(
        language="en",
        group="reflective",
        text="Async peer review catches overconfident merges in multilingual investigative reports.",
        segments=(
            "Async",
            "peer",
            "review",
            "catches",
            "overconfident",
            "merges",
            "in",
            "multilingual",
            "investigative",
            "reports",
            ".",
        ),
    ),
    CorpusSample(
        language="en",
        group="reflective",
        text="Temporal coherence checks highlight when the boundary signal lags the discourse flow.",
        segments=(
            "Temporal",
            "coherence",
            "checks",
            "highlight",
            "when",
            "the",
            "boundary",
            "signal",
            "lags",
            "the",
            "discourse",
            "flow",
            ".",
        ),
    ),
    CorpusSample(
        language="ja",
        group="reflective",
        text="ボブはキャロルの動機を再検討し、第三者の証拠で暫定評価を更新した。",
        segments=(
            "ボブ",
            "は",
            "キャロル",
            "の",
            "動機",
            "を",
            "再検討",
            "し",
            "、",
            "第三者",
            "の",
            "証拠",
            "で",
            "暫定",
            "評価",
            "を",
            "更新",
            "した",
            "。",
        ),
    ),
    CorpusSample(
        language="ja",
        group="reflective",
        text="結論を急ぎ過ぎないこと。内部対話で仮説を維持し、証拠で更新する。",
        segments=(
            "結論",
            "を",
            "急ぎ過ぎ",
            "ない",
            "こと",
            "。",
            "内部",
            "対話",
            "で",
            "仮説",
            "を",
            "維持",
            "し",
            "、",
            "証拠",
            "で",
            "更新",
            "する",
            "。",
        ),
    ),
    # Multilingual enrichment corpora
    CorpusSample(
        language="ja",
        group="multilingual",
        text="境界判定モデルは多言語入力でも一貫して学習される。",
        segments=("境界", "判定", "モデル", "は", "多言語", "入力", "でも", "一貫", "して", "学習", "される", "。"),
    ),
    CorpusSample(
        language="ja",
        group="multilingual",
        text="位相基底を共有すると変化点の感度が高まる。",
        segments=("位相", "基底", "を", "共有", "すると", "変化点", "の", "感度", "が", "高まる", "。"),
    ),
    CorpusSample(
        language="es",
        group="multilingual",
        text="La analista rastrea señales multilingües para ajustar los límites narrativos.",
        segments=(
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
        ),
    ),
    CorpusSample(
        language="es",
        group="multilingual",
        text="Modelos ligeros permiten entrenamiento estable sin depender de GPU.",
        segments=(
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
        ),
    ),
    CorpusSample(
        language="fr",
        group="multilingual",
        text="Les frontières apprises restent fiables même lorsque la langue change.",
        segments=(
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
        ),
    ),
    CorpusSample(
        language="fr",
        group="multilingual",
        text="Un encodeur de phase affine les indices locaux pour guider la planification.",
        segments=(
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
        ),
    ),
    CorpusSample(
        language="de",
        group="multilingual",
        text="Grenzsignale stabilisieren den Plan auch bei mehrsprachigen Dialogen.",
        segments=(
            "Grenzsignale",
            "stabilisieren",
            "den",
            "Plan",
            "auch",
            "bei",
            "mehrsprachigen",
            "Dialogen",
            ".",
        ),
    ),
    CorpusSample(
        language="de",
        group="multilingual",
        text="Ein lernbarer Phasenfilter verstärkt Übergänge mit geringer Latenz.",
        segments=(
            "Ein",
            "lernbarer",
            "Phasenfilter",
            "verstärkt",
            "Übergänge",
            "mit",
            "geringer",
            "Latenz",
            ".",
        ),
    ),
    CorpusSample(
        language="zh",
        group="multilingual",
        text="边界学生在多语言案例中保持高精度的分段预测。",
        segments=("边界", "学生", "在", "多语言", "案例", "中", "保持", "高精度", "的", "分段", "预测", "。"),
    ),
    CorpusSample(
        language="zh",
        group="multilingual",
        text="相位编码帮助注意力在跨域文本上快速适应。",
        segments=("相位", "编码", "帮助", "注意力", "在", "跨域", "文本", "上", "快速", "适应", "。"),
    ),
)


REFLECTIVE_LANGUAGES: Tuple[str, ...] = ("en", "ja")


def _group_by_language(samples: Sequence[CorpusSample]) -> Dict[str, Tuple[CorpusSample, ...]]:
    grouped: Dict[str, List[CorpusSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.language, []).append(sample)
    return {lang: tuple(items) for lang, items in grouped.items()}


LANGUAGE_SAMPLES: Dict[str, Tuple[CorpusSample, ...]] = _group_by_language(CORPUS_SAMPLES)


def languages() -> Tuple[str, ...]:
    """Return the sorted list of available language codes."""

    return tuple(sorted(LANGUAGE_SAMPLES.keys()))


def samples_for_language(language: str) -> Tuple[CorpusSample, ...]:
    """Return immutable samples for the requested ``language``."""

    if language not in LANGUAGE_SAMPLES:
        raise KeyError(f"Unknown language: {language}")
    return LANGUAGE_SAMPLES[language]


def iter_samples(languages_filter: Sequence[str] | None = None) -> Iterator[CorpusSample]:
    """Iterate over samples filtered by language codes."""

    if languages_filter is None:
        languages_filter = languages()
    for code in languages_filter:
        for sample in samples_for_language(code):
            yield sample


def reflective_samples() -> Tuple[CorpusSample, ...]:
    """Samples forming the default reflective training corpus."""

    items: List[CorpusSample] = []
    for sample in CORPUS_SAMPLES:
        if sample.group == "reflective":
            items.append(sample)
    return tuple(items)


def export_catalog(languages_filter: Sequence[str] | None = None) -> Dict[str, object]:
    """Export the dataset catalog in a JSON serialisable structure."""

    languages_payload: Dict[str, Dict[str, object]] = {}
    for sample in iter_samples(languages_filter):
        entry = languages_payload.setdefault(
            sample.language,
            {"groups": [], "samples": []},
        )
        groups = entry["groups"]
        if sample.group not in groups:
            groups.append(sample.group)
        entry["samples"].append(
            {
                "text": sample.text,
                "segments": list(sample.segments),
            }
        )
    return {
        "license": CORPUS_LICENSE,
        "languages": languages_payload,
    }


__all__ = [
    "CORPUS_LICENSE",
    "CORPUS_SAMPLES",
    "CorpusSample",
    "LANGUAGE_SAMPLES",
    "REFLECTIVE_LANGUAGES",
    "export_catalog",
    "iter_samples",
    "languages",
    "reflective_samples",
    "samples_for_language",
]
