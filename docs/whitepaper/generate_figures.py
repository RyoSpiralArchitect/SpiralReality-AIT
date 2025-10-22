#!/usr/bin/env python3
"""Generate simple SVG figures for latency, segmentation F1, and robustness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

SVG_HEADER = "<?xml version='1.0' encoding='UTF-8'?>\n"


def _svg_bar_chart(
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    height: int = 320,
    width: int = 600,
    value_format: str = "{:.2f}",
) -> str:
    max_val = max(values) if values else 1.0
    chart_height = height - 100
    bar_width = (width - 160) / max(1, len(values))
    svg = [SVG_HEADER, f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'>"]
    svg.append("<style>text{font-family:Arial,Helvetica,sans-serif;}</style>")
    svg.append(f"<text x='{width/2}' y='30' text-anchor='middle' font-size='18'>{title}</text>")
    svg.append(f"<text x='40' y='{height - 40}' font-size='12' transform='rotate(-90 40,{height - 40})'>{ylabel}</text>")
    for idx, (label, val) in enumerate(zip(labels, values)):
        bar_height = 0 if max_val == 0 else (val / max_val) * chart_height
        x = 100 + idx * bar_width
        y = height - 60 - bar_height
        svg.append(
            f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width * 0.6:.1f}' height='{bar_height:.1f}' fill='#2563EB'/>"
        )
        svg.append(
            f"<text x='{x + bar_width * 0.3:.1f}' y='{height - 40}' text-anchor='middle' font-size='12'>{label}</text>"
        )
        svg.append(
            f"<text x='{x + bar_width * 0.3:.1f}' y='{y - 8:.1f}' text-anchor='middle' font-size='11'>{value_format.format(val)}</text>"
        )
    svg.append("</svg>")
    return "".join(svg)


def _svg_latency(latency_ms: float, runs: int) -> str:
    width, height = 480, 200
    svg = [SVG_HEADER, f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'>"]
    svg.append("<style>text{font-family:Arial,Helvetica,sans-serif;}</style>")
    svg.append("<rect x='60' y='80' width='340' height='40' fill='#E5E7EB' rx='6' />")
    bar_width = min(340, max(5, latency_ms))
    svg.append(f"<rect x='60' y='80' width='{bar_width:.1f}' height='40' fill='#059669' rx='6' />")
    svg.append(
        f"<text x='{width/2}' y='50' text-anchor='middle' font-size='16'>Encode latency over {runs} runs</text>"
    )
    svg.append(
        f"<text x='{width/2}' y='105' text-anchor='middle' font-size='14'>{latency_ms:.1f} ms</text>"
    )
    svg.append("</svg>")
    return "".join(svg)


def _svg_line_chart(labels: List[str], values: List[float], title: str, ylabel: str) -> str:
    width, height = 600, 320
    chart_width, chart_height = width - 160, height - 120
    max_val = max(values) if values else 1.0
    svg = [SVG_HEADER, f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'>"]
    svg.append("<style>text{font-family:Arial,Helvetica,sans-serif;}</style>")
    svg.append(f"<text x='{width/2}' y='30' text-anchor='middle' font-size='18'>{title}</text>")
    svg.append(f"<text x='60' y='{height-60}' font-size='12' transform='rotate(-90 60,{height-60})'>{ylabel}</text>")
    if len(values) > 1:
        step = chart_width / (len(values) - 1)
    else:
        step = chart_width
    points: List[str] = []
    for idx, val in enumerate(values):
        x = 100 + idx * step
        y = height - 80 - (0 if max_val == 0 else (val / max_val) * chart_height)
        points.append(f"{x:.1f},{y:.1f}")
        svg.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='#7C3AED'/>"
        )
        svg.append(
            f"<text x='{x:.1f}' y='{height-50}' text-anchor='middle' font-size='12'>{labels[idx]}</text>"
        )
        svg.append(
            f"<text x='{x:.1f}' y='{y-10:.1f}' text-anchor='middle' font-size='11'>{val:.2f}</text>"
        )
    if points:
        svg.append(
            f"<polyline points='{' '.join(points)}' fill='none' stroke='#7C3AED' stroke-width='2'/>"
        )
    svg.append("</svg>")
    return "".join(svg)


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/whitepaper/data/evaluation_metrics.json"),
        help="Evaluation JSON file produced by run_evaluation.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/whitepaper/figures"),
        help="Directory where SVG figures will be created.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Evaluation metrics not found: {args.input}")

    data = json.loads(args.input.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seg = data.get("segmentation", {})
    labels = [f"Sample {i+1}" for i in range(len(seg.get("per_sample", [])))]
    values = [float(entry.get("f1", 0.0)) for entry in seg.get("per_sample", [])]
    seg_svg = _svg_bar_chart(labels, values, "Segmentation F1 per sample", "F1 score")
    seg_path = args.output_dir / "segmentation_f1.svg"
    write_file(seg_path, seg_svg)

    latency_ms = float(data.get("latency_seconds", 0.0) * 1000.0)
    latency_svg = _svg_latency(latency_ms, int(data.get("latency_runs", 0)))
    latency_path = args.output_dir / "latency.svg"
    write_file(latency_path, latency_svg)

    robustness = data.get("robustness", {})
    r_labels = [f"Sample {i+1}" for i in range(len(robustness.get("per_text_mean", [])))]
    r_values = [float(entry.get("mean_f1", 0.0)) for entry in robustness.get("per_text_mean", [])]
    robustness_svg = _svg_line_chart(
        r_labels,
        r_values,
        "Robustness under character perturbations",
        "Mean robustness F1",
    )
    robustness_path = args.output_dir / "robustness.svg"
    write_file(robustness_path, robustness_svg)

    manifest = {
        "segmentation": str(seg_path.relative_to(args.output_dir.parent)),
        "latency": str(latency_path.relative_to(args.output_dir.parent)),
        "robustness": str(robustness_path.relative_to(args.output_dir.parent)),
    }
    write_file(args.output_dir / "figures.json", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
