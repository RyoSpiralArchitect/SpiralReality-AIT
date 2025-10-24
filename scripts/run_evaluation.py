#!/usr/bin/env python3
"""Run reproducible latency, F1, and robustness evaluations for SpiralReality AIT."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spiralreality_AIT_onepass_aifcore_integrated.integrated.corpus import (
    TRAIN_TEXTS,
    teacher_segments,
)
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import (
    GateDiagnostics,
    OnePassAIT,
    StudentTrainingConfig,
)


def segmentation_f1(text: str, gold_segments: Sequence[str], predicted_segments: Sequence[str]) -> float:
    """Compute character boundary F1 given gold and predicted segmentations."""

    def cuts(segments: Sequence[str]) -> set[int]:
        idx = 0
        out: set[int] = set()
        for tok in segments:
            idx += len(tok)
            out.add(idx)
        out.discard(len(text))
        return out

    gold = cuts(gold_segments)
    pred = cuts(predicted_segments)
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _segment_lengths(segments: Sequence[str]) -> List[int]:
    return [len(tok) for tok in segments]


def _segments_from_lengths(text: str, lengths: Sequence[int]) -> List[str]:
    segments: List[str] = []
    start = 0
    for length in lengths:
        end = start + length
        segments.append(text[start:end])
        start = end
    if start < len(text):
        segments.append(text[start:])
    return segments


def _perturb_text(text: str, noise_level: float, rng: random.Random) -> str:
    """Return a perturbed version of *text* without changing its length."""

    def flip_case(ch: str) -> str:
        if ch.islower():
            return ch.upper()
        if ch.isupper():
            return ch.lower()
        return ch

    glyphs = [
        "~",
        "?",
        "…",
        "○",
        "◇",
        "▪",
    ]

    out_chars: List[str] = []
    for ch in text:
        if ch.strip() == "":
            out_chars.append(ch)
            continue
        if rng.random() > noise_level:
            if rng.random() < 0.5:
                out_chars.append(flip_case(ch))
            else:
                out_chars.append(ch)
            continue
        replacement = rng.choice(glyphs)
        if len(replacement) != 1:
            replacement = replacement[0]
        out_chars.append(replacement)
    return "".join(out_chars)


def _collect_gate_diagnostics(ait: OnePassAIT, text: str) -> Dict[str, float]:
    diagnostics: GateDiagnostics = ait.gate_diagnostics()
    attention = diagnostics.attention_strength if diagnostics.attention_strength else []
    attn_mean = float(sum(attention) / len(attention)) if attention else 0.0
    attn_std = (
        float(math.sqrt(sum((v - attn_mean) ** 2 for v in attention) / len(attention)))
        if attention
        else 0.0
    )
    return {
        "mask_energy": float(diagnostics.mask_energy),
        "attention_mean": attn_mean,
        "attention_std": attn_std,
    }


def run_evaluation(
    output_dir: Path,
    latency_runs: int,
    robustness_trials: int,
    robustness_noise: float,
    seed: int,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    ait = OnePassAIT(latent_dim=32, seed=seed)
    cfg = StudentTrainingConfig(
        lr=0.05,
        epochs=16,
        batch_size=2,
        validation_split=0.4,
        patience=4,
        hidden_dim=20,
        emb_dim=14,
        window=2,
        phase_lr=0.4,
        cache_sequences=True,
        shuffle_train=True,
    )

    texts = TRAIN_TEXTS[:4]
    segments = teacher_segments(texts)
    summary = ait.train_student(texts, segments, cfg=cfg)

    per_sample: List[Dict[str, object]] = []
    lengths_cache = [_segment_lengths(seg) for seg in segments]
    f1_scores: List[float] = []

    for text, gold_segments, lengths in zip(texts, segments, lengths_cache):
        predicted_result = ait.student.decode(text)
        predicted = (
            predicted_result["tokens"]
            if isinstance(predicted_result, dict)
            else predicted_result
        )
        f1 = segmentation_f1(text, gold_segments, predicted)
        per_sample.append(
            {
                "text": text,
                "f1": f1,
                "gold_segments": gold_segments,
                "predicted_segments": predicted,
            }
        )
        f1_scores.append(f1)

    latency_prompt = " ".join(texts[:2])
    start = time.perf_counter()
    for _ in range(latency_runs):
        ait.encode(latency_prompt)
    latency = (time.perf_counter() - start) / max(1, latency_runs)

    rng = random.Random(seed + 42)
    robustness_records: List[Dict[str, object]] = []
    mean_robustness_per_text: List[Tuple[str, float]] = []

    for text, gold_segments, lengths in zip(texts, segments, lengths_cache):
        per_text_scores: List[float] = []
        for trial in range(robustness_trials):
            noisy_text = _perturb_text(text, robustness_noise, rng)
            projected_gold = _segments_from_lengths(noisy_text, lengths)
            predicted_result = ait.student.decode(noisy_text)
            predicted = (
                predicted_result["tokens"]
                if isinstance(predicted_result, dict)
                else predicted_result
            )
            score = segmentation_f1(noisy_text, projected_gold, predicted)
            robustness_records.append(
                {
                    "text": text,
                    "trial": trial,
                    "noisy_text": noisy_text,
                    "robustness_f1": score,
                }
            )
            per_text_scores.append(score)
        mean_robustness_per_text.append((text, float(statistics.mean(per_text_scores))))

    overall_robustness = [rec["robustness_f1"] for rec in robustness_records]

    gate_info = _collect_gate_diagnostics(ait, texts[0])

    results: Dict[str, object] = {
        "seed": seed,
        "latency_prompt": latency_prompt,
        "latency_runs": latency_runs,
        "latency_seconds": latency,
        "segmentation": {
            "per_sample": per_sample,
            "mean": float(statistics.mean(f1_scores)) if f1_scores else 0.0,
            "stdev": float(statistics.pstdev(f1_scores)) if len(f1_scores) > 1 else 0.0,
        },
        "robustness": {
            "noise_level": robustness_noise,
            "trials": robustness_trials,
            "records": robustness_records,
            "per_text_mean": [
                {"text": text, "mean_f1": score} for text, score in mean_robustness_per_text
            ],
            "mean": float(statistics.mean(overall_robustness)) if overall_robustness else 0.0,
            "stdev": float(statistics.pstdev(overall_robustness))
            if len(overall_robustness) > 1
            else 0.0,
        },
        "gate_diagnostics": gate_info,
        "train_summary": summary,
    }

    json_path = output_dir / "evaluation_metrics.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_lines = ["text,f1"]
    for entry in per_sample:
        snippet = entry["text"].replace("\n", " ")
        csv_lines.append(f'"{snippet}",{entry["f1"]:.6f}')
    (output_dir / "segmentation_f1.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    robustness_lines = ["text,trial,robustness_f1"]
    for entry in robustness_records:
        snippet = entry["text"].replace("\n", " ")
        robustness_lines.append(f'"{snippet}",{entry["trial"]},{entry["robustness_f1"]:.6f}')
    (output_dir / "robustness.csv").write_text("\n".join(robustness_lines), encoding="utf-8")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/whitepaper/data"),
        help="Directory where evaluation artefacts will be stored.",
    )
    parser.add_argument("--latency-runs", type=int, default=8, help="Number of encode passes for latency.")
    parser.add_argument(
        "--robustness-trials",
        type=int,
        default=5,
        help="Number of perturbation trials per sample when estimating robustness.",
    )
    parser.add_argument(
        "--robustness-noise",
        type=float,
        default=0.12,
        help="Probability of perturbing a character when measuring robustness.",
    )
    parser.add_argument("--seed", type=int, default=2024, help="Random seed used for evaluation runs.")
    args = parser.parse_args()

    results = run_evaluation(
        output_dir=args.output,
        latency_runs=args.latency_runs,
        robustness_trials=args.robustness_trials,
        robustness_noise=args.robustness_noise,
        seed=args.seed,
    )
    print(json.dumps({k: v for k, v in results.items() if k not in {"train_summary", "segmentation", "robustness"}}, indent=2))


if __name__ == "__main__":
    main()
