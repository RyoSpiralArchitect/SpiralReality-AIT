"""Benchmarking utilities integrating perturbations and metrics reporting."""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from typing import Dict, List, Mapping, Sequence, Tuple

from .augmentation import PerturbationGenerator
from .corpus import (
    TRAIN_TEXTS,
    corpus_catalog,
    corpus_license,
    teacher_segments,
)
from .datasets import iter_samples
from .multilingual import build_multilingual_corpus, language_histogram
from .onepass_ait import OnePassAIT, StudentTrainingConfig

import numpy as real_numpy

from . import np_stub


def segmentation_f1(text: str, gold_segments: Sequence[str], predicted_segments: Sequence[str]) -> float:
    """Compute F1 between gold and predicted boundary positions."""

    def cuts(segments: Sequence[str]) -> List[int]:
        idx = 0
        out: List[int] = []
        for tok in segments:
            idx += len(tok)
            out.append(idx)
        if out and out[-1] == len(text):
            out.pop()
        return out

    gold = set(cuts(gold_segments))
    pred = set(cuts(predicted_segments))
    if not gold and not pred:
        return 1.0
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


_TEXT_LANG_MAP: Dict[str, str] = {sample.text: sample.language for sample in iter_samples()}


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def _language_for_text(text: str, fallback: str) -> str:
    return _TEXT_LANG_MAP.get(text, fallback)


def _compare_numpy_stub(seed: int = 0) -> Dict[str, float | bool | None]:
    rng = real_numpy.random.default_rng(seed)
    base = rng.standard_normal(64).reshape(8, 8)
    scaled = base * 1.5 + rng.standard_normal(base.shape) * 0.05
    stub_arr = np_stub.array(scaled.tolist())
    stub_result = np_stub.tanh(stub_arr).to_list()
    real_result = real_numpy.tanh(scaled)
    diff = real_result - real_numpy.array(stub_result)
    linf = float(real_numpy.max(real_numpy.abs(diff)))
    mse = float(real_numpy.mean(diff**2))
    return {"available": True, "linf": linf, "mse": mse}


def _encode_latencies(ait: OnePassAIT, texts: Sequence[str], runs: int = 3) -> List[float]:
    latencies: List[float] = []
    for _ in range(max(1, runs)):
        for text in texts:
            ait._encode_cache.pop(text, None)
            start = time.perf_counter()
            ait.encode(text)
            latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies


def _segment_latencies(segmenter, texts: Sequence[str], runs: int = 1) -> List[float]:
    latencies: List[float] = []
    for _ in range(max(1, runs)):
        for text in texts:
            start = time.perf_counter()
            segmenter(text)
            latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies


def run_benchmark(
    *,
    languages: Sequence[str] | None = None,
    include_reflective: bool = True,
    max_samples: int | None = 12,
    output_dir: str | None = "reports",
    seed: int = 5042,
) -> Dict[str, object]:
    """Train a student, apply perturbations, and collect metrics."""

    texts: List[str]
    segments: List[List[str]]
    tags: List[str]

    ml_texts, ml_segments, ml_tags = build_multilingual_corpus(
        languages=languages,
        include_reflective=include_reflective,
        shuffle=False,
        seed=seed,
    )
    if not ml_texts:
        texts = list(TRAIN_TEXTS)
        segments = teacher_segments(texts)
        tags = ["reflective" for _ in texts]
    else:
        texts = list(ml_texts)
        segments = [list(seg) for seg in ml_segments]
        tags = list(ml_tags)

    if max_samples is not None and len(texts) > max_samples:
        texts = texts[:max_samples]
        segments = segments[:max_samples]
        tags = tags[:max_samples]

    ait = OnePassAIT(latent_dim=32, seed=seed)
    cfg = StudentTrainingConfig(
        lr=0.05,
        epochs=8,
        batch_size=2,
        validation_split=0.25,
        patience=3,
        hidden_dim=20,
        emb_dim=14,
        window=2,
        phase_lr=0.3,
        cache_sequences=False,
        shuffle_train=False,
    )
    ait.train_student(texts, segments, cfg=cfg)

    baseline_scores: Dict[str, float] = {}
    generator = PerturbationGenerator(seed=seed)
    variant_scores: Dict[str, List[float]] = {}
    variant_drop: Dict[str, List[float]] = {}
    actual_languages: List[str] = []

    for text, seg, tag in zip(texts, segments, tags):
        lang = _language_for_text(text, tag)
        actual_languages.append(lang)
        predicted_result = ait.student.decode_with_logit_bias(text, 0.0)
        predicted = predicted_result["tokens"] if isinstance(predicted_result, dict) else predicted_result
        f1 = segmentation_f1(text, seg, predicted)
        baseline_scores[text] = f1
        for variant in generator.generate_variants(text, seg, language=lang):
            pred_result = ait.student.decode_with_logit_bias(variant.text, 0.0)
            pred_segments = pred_result["tokens"] if isinstance(pred_result, dict) else pred_result
            vf1 = segmentation_f1(variant.text, variant.segments, pred_segments)
            variant_scores.setdefault(variant.tag, []).append(vf1)
            baseline = f1
            drop = 0.0 if baseline <= 1e-8 else max(0.0, (baseline - vf1) / baseline)
            variant_drop.setdefault(variant.tag, []).append(drop)

    def evaluate_setting(*, context: bool, aif: bool) -> Dict[str, object]:
        original_context = bool(getattr(ait.student, "use_encoder_context", True))
        scores: Dict[str, float] = {}
        policy_hist: Dict[str, int] = {}
        seg_latencies: List[float] = []

        def segment_one(text: str) -> Sequence[str]:
            if aif:
                result = ait.segment_text(text, return_metadata=True, use_aif=True)
                if isinstance(result, dict):
                    policy = result.get("chosen_policy")
                    if isinstance(policy, str) and policy:
                        policy_hist[policy] = policy_hist.get(policy, 0) + 1
                    return result.get("tokens", [])
                return result  # type: ignore[return-value]
            decoded = ait.student.decode_with_logit_bias(text, 0.0)
            return decoded["tokens"] if isinstance(decoded, dict) else decoded  # type: ignore[return-value]

        try:
            ait.student.use_encoder_context = bool(context)
            for text, seg in zip(texts, segments):
                start = time.perf_counter()
                predicted = list(segment_one(text))
                seg_latencies.append((time.perf_counter() - start) * 1000.0)
                scores[text] = segmentation_f1(text, seg, predicted)
        finally:
            ait.student.use_encoder_context = original_context

        mean_latency = float(statistics.mean(seg_latencies)) if seg_latencies else 0.0
        p95_latency = _percentile(seg_latencies, 95.0)
        max_latency = max(seg_latencies) if seg_latencies else 0.0

        info: Dict[str, object] = {
            "mean_f1": float(statistics.mean(scores.values())) if scores else 0.0,
            "per_text": scores,
            "segment_latency_ms": {
                "samples": len(seg_latencies),
                "mean": mean_latency,
                "p95": p95_latency,
                "max": max_latency,
            },
        }
        if aif:
            info["policy_hist"] = policy_hist
        return info

    latencies = _encode_latencies(ait, texts)
    mean_latency = float(statistics.mean(latencies)) if latencies else 0.0
    p95_latency = _percentile(latencies, 95.0)
    max_latency = max(latencies) if latencies else 0.0

    dataset_hist = language_histogram(actual_languages)
    catalog_languages = sorted({lang for lang in actual_languages})
    dataset_info = {
        "size": len(texts),
        "languages": dataset_hist,
        "license": corpus_license(),
        "catalog": corpus_catalog(catalog_languages),
    }

    baseline_info = {
        "f1": float(statistics.mean(baseline_scores.values())) if baseline_scores else 0.0,
        "per_text": baseline_scores,
        "latency_ms": {
            "samples": len(latencies),
            "mean": mean_latency,
            "p95": p95_latency,
            "max": max_latency,
        },
    }

    ablations = {
        "context_on_aif_off": evaluate_setting(context=True, aif=False),
        "context_on_aif_on": evaluate_setting(context=True, aif=True),
        "context_off_aif_off": evaluate_setting(context=False, aif=False),
        "context_off_aif_on": evaluate_setting(context=False, aif=True),
    }

    variants_info: Dict[str, Dict[str, float]] = {}
    for tag, scores in variant_scores.items():
        mean_score = float(statistics.mean(scores)) if scores else 0.0
        drops = variant_drop.get(tag, [])
        mean_drop = float(statistics.mean(drops)) if drops else 0.0
        variants_info[tag] = {
            "mean_f1": mean_score,
            "mean_degradation": mean_drop,
            "samples": len(scores),
        }

    np_metrics = _compare_numpy_stub(seed=seed)

    metrics: Dict[str, object] = {
        "dataset": dataset_info,
        "baseline": baseline_info,
        "ablations": ablations,
        "variants": variants_info,
        "np_stub": np_metrics,
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "benchmark_report.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, ensure_ascii=False)
        _write_markdown_report(metrics, os.path.join(output_dir, "benchmark_report.md"))

    return metrics


def _write_markdown_report(metrics: Mapping[str, object], path: str) -> None:
    dataset = metrics.get("dataset", {})
    baseline = metrics.get("baseline", {})
    ablations = metrics.get("ablations", {})
    variants = metrics.get("variants", {})
    np_metrics = metrics.get("np_stub", {})

    lines = ["# SpiralReality AIT Benchmark", ""]
    license_info = dataset.get("license", {}) if isinstance(dataset, Mapping) else {}
    if license_info:
        lines.append("## Dataset License")
        lines.append("")
        for key in ("id", "name", "url", "attribution"):
            value = license_info.get(key)
            if value:
                lines.append(f"- **{key}**: {value}")
        notes = license_info.get("notes")
        if notes:
            lines.append(f"- **notes**: {notes}")
        lines.append("")

    lines.append("## Baseline Metrics")
    if isinstance(baseline, Mapping):
        f1 = baseline.get("f1", 0.0)
        latency = baseline.get("latency_ms", {})
        lines.append(f"- Mean F1: {f1:.4f}")
        if isinstance(latency, Mapping):
            lines.append(
                "- Encode latency (ms): mean={:.3f}, p95={:.3f}, max={:.3f}".format(
                    latency.get("mean", 0.0),
                    latency.get("p95", 0.0),
                    latency.get("max", 0.0),
                )
            )
    lines.append("")

    if isinstance(ablations, Mapping) and ablations:
        lines.append("## Ablations (Context / AIF)")
        lines.append("| Setting | Mean F1 | Segment latency mean (ms) | p95 (ms) | Policy hist |")
        lines.append("| --- | --- | --- | --- | --- |")
        for key, info in ablations.items():
            if not isinstance(info, Mapping):
                continue
            mean_f1 = float(info.get("mean_f1", 0.0) or 0.0)
            seg_latency = info.get("segment_latency_ms", {})
            seg_mean = seg_p95 = 0.0
            if isinstance(seg_latency, Mapping):
                seg_mean = float(seg_latency.get("mean", 0.0) or 0.0)
                seg_p95 = float(seg_latency.get("p95", 0.0) or 0.0)
            policy_hist = info.get("policy_hist")
            policy_str = ""
            if isinstance(policy_hist, Mapping) and policy_hist:
                items = sorted(policy_hist.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
                policy_str = ", ".join(f"{name}:{count}" for name, count in items)
            lines.append(f"| {key} | {mean_f1:.4f} | {seg_mean:.3f} | {seg_p95:.3f} | {policy_str} |")
        lines.append("")

    if variants:
        lines.append("## Perturbation Variants")
        lines.append("| Variant | Mean F1 | Degradation (%) | Samples |")
        lines.append("| --- | --- | --- | --- |")
        for tag, info in variants.items():
            if not isinstance(info, Mapping):
                continue
            mean_f1 = info.get("mean_f1", 0.0)
            mean_drop = info.get("mean_degradation", 0.0) * 100.0
            samples = info.get("samples", 0)
            lines.append(f"| {tag} | {mean_f1:.4f} | {mean_drop:.2f} | {samples} |")
        lines.append("")

    if isinstance(np_metrics, Mapping) and np_metrics.get("available"):
        lines.append("## NumPy vs np_stub")
        lines.append(
            "- L_inf error: {linf:.6f}\n- MSE: {mse:.6f}".format(
                linf=np_metrics.get("linf", 0.0) or 0.0,
                mse=np_metrics.get("mse", 0.0) or 0.0,
            )
        )
        lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


__all__ = [
    "run_benchmark",
    "segmentation_f1",
]
