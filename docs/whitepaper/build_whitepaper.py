#!/usr/bin/env python3
"""Convert the Markdown whitepaper draft into a PDF using Matplotlib."""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path
from typing import Iterable

plt = None  # type: ignore[assignment]
PdfPages = None  # type: ignore[assignment]


def _ensure_matplotlib() -> None:
    global plt, PdfPages
    if plt is not None and PdfPages is not None:
        return
    try:
        import matplotlib.pyplot as _plt  # type: ignore
        from matplotlib.backends.backend_pdf import PdfPages as _PdfPages  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "Matplotlib is required to build the whitepaper PDF. Install it via 'pip install matplotlib'."
        ) from exc
    plt = _plt
    PdfPages = _PdfPages


PAGE_SIZE = (8.27, 11.69)  # A4 portrait
MARGIN_Y = 0.92
LINE_SPACING = 0.035


def _new_page():
    _ensure_matplotlib()
    assert plt is not None
    fig, ax = plt.subplots(figsize=PAGE_SIZE)
    ax.axis("off")
    return fig, MARGIN_Y


def _render_text(fig, y: float, text: str, fontsize: float):
    if y < 0.1:
        return fig, y
    fig.text(0.08, y, text, ha="left", va="top", fontsize=fontsize, wrap=True)
    return fig, y - LINE_SPACING * (fontsize / 10.0)


def _render_wrapped(pdf, fig, y: float, text: str, width: int = 90, fontsize: float = 10.5):
    wrapper = textwrap.TextWrapper(width=width)
    for wrapped in wrapper.wrap(text):
        if y < 0.12:
            assert plt is not None
            pdf.savefig(fig)
            plt.close(fig)
            fig, y = _new_page()
        assert plt is not None
        fig.text(0.08, y, wrapped, ha="left", va="top", fontsize=fontsize)
        y -= LINE_SPACING
    return fig, y - LINE_SPACING * 0.3


def _render_image(pdf, image_path: Path, caption: str):
    _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=PAGE_SIZE)
    ax.axis("off")
    img = plt.imread(str(image_path))
    ax.imshow(img)
    ax.set_title(caption, fontsize=11)
    pdf.savefig(fig, dpi=300)
    plt.close(fig)
    return _new_page()


def _process_lines(pdf, lines: Iterable[str], base_dir: Path) -> None:
    fig, y = _new_page()
    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            y -= LINE_SPACING * 1.5
            continue
        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
        if image_match:
            caption, rel_path = image_match.groups()
            image_path = (base_dir / rel_path).resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"Image referenced in Markdown not found: {image_path}")
            assert plt is not None
            pdf.savefig(fig)
            plt.close(fig)
            fig, y = _render_image(pdf, image_path, caption or "Figure")
            continue
        if line.startswith("# "):
            text = line[2:].strip()
            fig, y = _render_text(fig, y, text, fontsize=18)
            y -= LINE_SPACING
        elif line.startswith("## "):
            text = line[3:].strip()
            fig, y = _render_text(fig, y, text, fontsize=14)
            y -= LINE_SPACING * 0.5
        elif line.startswith("### "):
            text = line[4:].strip()
            fig, y = _render_text(fig, y, text, fontsize=12)
        elif line.startswith("- "):
            bullet = line[2:].strip()
            wrapper = textwrap.TextWrapper(width=85)
            for idx, wrapped in enumerate(wrapper.wrap(bullet)):
                prefix = "• " if idx == 0 else "  "
                if y < 0.12:
                    pdf.savefig(fig)
                    plt.close(fig)
                    fig, y = _new_page()
                fig.text(0.08, y, prefix + wrapped, ha="left", va="top", fontsize=10.5)
                y -= LINE_SPACING
            y -= LINE_SPACING * 0.3
        else:
            fig, y = _render_wrapped(pdf, fig, y, line)
        if y < 0.12:
            assert plt is not None
            pdf.savefig(fig)
            plt.close(fig)
            fig, y = _new_page()
    assert plt is not None
    pdf.savefig(fig)
    plt.close(fig)


def build_whitepaper(markdown_path: Path, pdf_path: Path) -> None:
    _ensure_matplotlib()
    assert PdfPages is not None
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        lines = markdown_path.read_text(encoding="utf-8").splitlines()
        _process_lines(pdf, lines, markdown_path.parent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("docs/whitepaper/whitepaper.md"),
        help="Path to the Markdown source file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/whitepaper/whitepaper.pdf"),
        help="Destination PDF path.",
    )
    args = parser.parse_args()

    if not args.markdown.exists():
        raise FileNotFoundError(f"Markdown source not found: {args.markdown}")

    build_whitepaper(args.markdown, args.output)
    print(f"Whitepaper written to {args.output}")


if __name__ == "__main__":
    main()
