"""Shared training corpus and segmentation helpers for the demo/tests."""

from __future__ import annotations

from typing import Iterable, List, Sequence

_BOUNDARY_PUNCT = ",.;。、「」…！？!?:;—‑-"

TRAIN_TEXTS: tuple[str, ...] = (
    "Bob re-examined Carol's motives and updated his provisional evaluation.",
    "Avoid premature closure; maintain hypotheses and update them with evidence.",
    "Meta-analysis of sensor feeds prevents overfitting to the first plausible story.",
    "Iterative reframing of witness notes keeps latent clusters responsive to nuance.",
    "Systematic journaling of counterfactuals stabilizes the agent's decision frontier.",
    "Layered attention summaries discourage impulsive segmentation during cold starts.",
    "Sustained reflective questioning keeps the analyst from freezing the latent plan too soon.",
    "Calibrating priors against delayed confirmations avoids premature pruning of actions.",
    "Contingency-aware sampling lets the bridge respect rare but critical micro-boundaries.",
    "Slow breathing, careful parsing, then align evidence with provisional policies.",
    "Maintain a rolling scratchpad; revise boundaries when fresh observations contradict assumptions.",
    "Socratic prompts elicit clarifications that re-open segments closed by cognitive fatigue.",
    "Cross-lingual glossaries ensure the encoder does not conflate idioms with hard stops.",
    "Goal-conditioned rehearsal simulates failure cases before committing to gating decisions.",
    "Streaming diagnostics reveal when the CRF head saturates and needs regularization.",
    "Noise-robust embeddings resist the temptation to snap to the majority class boundary.",
    "Phase-aware masking prevents the planner from extrapolating on stale latent states.",
    "Manual audits of auto-segmented transcripts surface drift before metrics collapse.",
    "Align observations, plans, and critiques; triangulated narratives yield resilient cuts.",
    "Async peer review catches overconfident merges in multilingual investigative reports.",
    "Temporal coherence checks highlight when the boundary signal lags the discourse flow.",
    "ボブはキャロルの動機を再検討し、第三者の証拠で暫定評価を更新した。",
    "結論を急ぎ過ぎないこと。内部対話で仮説を維持し、証拠で更新する。",
)


TRAIN_SEGMENTS: tuple[List[str], ...] = (
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
    [
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
    ],
)


_SEGMENT_MAP = {text: seg for text, seg in zip(TRAIN_TEXTS, TRAIN_SEGMENTS)}


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


__all__ = [
    "TRAIN_TEXTS",
    "TRAIN_SEGMENTS",
    "naive_segments",
    "teacher_segments",
    "register_teacher_segment",
]
