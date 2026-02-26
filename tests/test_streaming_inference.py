from __future__ import annotations

from typing import Sequence

from spiralreality_AIT_onepass_aifcore_integrated.integrated.corpus import TRAIN_TEXTS
from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import OnePassAIT
from spiralreality_AIT_onepass_aifcore_integrated.integrated.streaming_inference import (
    ChunkedStreamingSegmenter,
)


def test_chunked_streaming_segmenter_respects_max_window() -> None:
    calls: list[int] = []

    def segmenter(text: str) -> Sequence[str]:
        calls.append(len(text))
        return [text]

    streamer = ChunkedStreamingSegmenter(
        segmenter,
        max_window_chars=16,
        lookahead_chars=4,
        context_chars=6,
        hard_split=True,
    )
    text = "abcdefghijklmnopqrstuvwxyz"
    out: list[str] = []
    for chunk in ("abc", "defghijklmnop", "qrstuvwxyz"):
        out.extend(streamer.feed(chunk))
    out.extend(streamer.flush())
    assert "".join(out) == text
    assert calls and max(calls) <= 16


def test_onepassait_streaming_segmenter_roundtrip() -> None:
    ait = OnePassAIT(latent_dim=16, seed=123)
    streamer = ait.streaming_segmenter(
        max_window_chars=128,
        lookahead_chars=32,
        context_chars=64,
    )
    text = TRAIN_TEXTS[0]
    step = max(1, len(text) // 3)
    out: list[str] = []
    out.extend(streamer.feed(text[:step]))
    out.extend(streamer.feed(text[step : 2 * step]))
    out.extend(streamer.feed(text[2 * step :]))
    out.extend(streamer.flush())
    assert "".join(out) == text

