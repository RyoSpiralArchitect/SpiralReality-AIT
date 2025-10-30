import itertools
import threading
from typing import Sequence

import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.corpus import (
    TRAIN_TEXTS,
    teacher_segments,
)
from spiralreality_AIT_onepass_aifcore_integrated.integrated.multilingual import (
    AVAILABLE_LANGUAGES,
    build_multilingual_corpus,
    iter_multilingual_texts,
    multilingual_segment_stream,
)
from spiralreality_AIT_onepass_aifcore_integrated.integrated.streaming import (
    SegmentBatch,
    SegmentStream,
)


def _flatten_segments(batches: list[SegmentBatch]) -> list[list[str]]:
    return [list(segment) for batch in batches for segment in batch.segments]


def test_segment_stream_matches_teacher_segments() -> None:
    texts = TRAIN_TEXTS[:5]
    batches = list(SegmentStream(texts, chunk_size=2))
    expected = teacher_segments(texts)
    assert _flatten_segments(batches) == [list(seg) for seg in expected]


def test_segment_stream_prefetch_preserves_order() -> None:
    texts = TRAIN_TEXTS[:6]
    eager = list(SegmentStream(texts, chunk_size=3, max_prefetch=0))
    prefetched = list(SegmentStream(texts, chunk_size=3, max_prefetch=2))
    assert [_flatten_segments([batch]) for batch in eager] == [
        _flatten_segments([batch]) for batch in prefetched
    ]


def test_segment_stream_prefetch_allows_early_exit() -> None:
    texts = TRAIN_TEXTS[:8]
    iterator = iter(SegmentStream(texts, chunk_size=2, max_prefetch=3))
    first_batch = next(iterator)
    assert len(first_batch.texts) == 2

    closer = threading.Thread(target=iterator.close)
    closer.start()
    closer.join(timeout=1.0)
    assert not closer.is_alive()


def test_segment_stream_metadata_alignment() -> None:
    texts = TRAIN_TEXTS[:3]
    metadata = (f"meta-{i}" for i in range(len(texts)))
    batches = list(SegmentStream(texts, metadata=metadata, chunk_size=2))
    collected = [meta for batch in batches for meta in (batch.metadata or ())]
    assert collected == ["meta-0", "meta-1", "meta-2"]


def test_segment_stream_rejects_invalid_chunk_size() -> None:
    with pytest.raises(ValueError):
        SegmentStream(TRAIN_TEXTS[:2], chunk_size=0)


def test_segment_stream_detects_metadata_length_mismatch() -> None:
    texts = TRAIN_TEXTS[:2]
    with pytest.raises(ValueError):
        list(SegmentStream(texts, metadata=["tag"], chunk_size=2))

    with pytest.raises(ValueError):
        list(SegmentStream(texts, metadata=["tag", "extra", "overflow"], chunk_size=1))


def test_segment_stream_exposes_length_hints() -> None:
    texts = TRAIN_TEXTS[:5]
    stream = SegmentStream(texts, chunk_size=2)
    assert stream.text_length_hint == len(texts)
    assert stream.batch_length_hint == 3
    assert stream.metadata_length_hint is None


def test_segment_stream_length_hints_respect_drop_incomplete() -> None:
    texts = TRAIN_TEXTS[:5]
    metadata = [f"tag-{i}" for i in range(len(texts))]
    chunk_size = 2
    stream = SegmentStream(
        texts,
        metadata=metadata,
        chunk_size=chunk_size,
        drop_incomplete=True,
    )
    assert stream.batch_length_hint == len(texts) // chunk_size
    expected_metadata = len(metadata) - len(metadata) % chunk_size
    assert stream.metadata_length_hint == expected_metadata


def test_segment_stream_can_drop_incomplete_batches() -> None:
    texts = TRAIN_TEXTS[:5]
    metadata = (f"tag-{i}" for i in range(len(texts)))
    batches = list(
        SegmentStream(
            texts,
            metadata=metadata,
            chunk_size=2,
            drop_incomplete=True,
        )
    )
    assert batches
    assert all(len(batch.texts) == 2 for batch in batches)
    collected = [meta for batch in batches for meta in (batch.metadata or ())]
    assert len(collected) == 4


def test_segment_stream_drop_incomplete_with_prefetch() -> None:
    texts = TRAIN_TEXTS[:7]
    batches = list(
        SegmentStream(texts, chunk_size=3, max_prefetch=2, drop_incomplete=True)
    )
    assert all(len(batch.texts) == 3 for batch in batches)
    assert len(batches) == 2


def test_segment_stream_propagates_segmenter_errors() -> None:
    calls = 0

    def failing_segmenter(chunk: Sequence[str]) -> Sequence[Sequence[str]]:
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise RuntimeError("segmenter boom")
        return teacher_segments(chunk)

    stream = SegmentStream(
        TRAIN_TEXTS[:5],
        segmenter=failing_segmenter,
        chunk_size=2,
        max_prefetch=2,
    )
    iterator = iter(stream)
    first = next(iterator)
    assert len(first.texts) == 2
    with pytest.raises(RuntimeError):
        next(iterator)


def test_iter_multilingual_texts_registers_tags() -> None:
    subset = AVAILABLE_LANGUAGES[:2]
    records = list(itertools.islice(iter_multilingual_texts(subset, shuffle=False), 4))
    assert all(tag in set(subset) | {"reflective"} for tag, _ in records)


def test_multilingual_segment_stream_batches_include_metadata() -> None:
    stream = multilingual_segment_stream(
        languages=("es", "ja"),
        include_reflective=False,
        chunk_size=2,
        max_prefetch=1,
    )
    batches = list(stream)
    assert batches
    for batch in batches:
        assert batch.metadata is not None
        assert len(batch.metadata) == len(batch.texts) == len(batch.segments)
    tags = {tag for batch in batches for tag in (batch.metadata or ())}
    assert tags == {"es", "ja"}


def test_multilingual_segment_stream_drop_incomplete() -> None:
    texts, _, _ = build_multilingual_corpus(
        languages=("es",), include_reflective=False, shuffle=False
    )
    assert len(texts) >= 2
    chunk_size = max(1, len(texts) - 1)
    stream = multilingual_segment_stream(
        languages=("es",),
        include_reflective=False,
        chunk_size=chunk_size,
        drop_incomplete=True,
    )
    batches = list(stream)
    assert batches
    assert all(len(batch.texts) == chunk_size for batch in batches)
    consumed = sum(len(batch.texts) for batch in batches)
    assert consumed == len(texts) - (len(texts) % chunk_size)


def test_multilingual_segment_stream_shuffle_materialises_dataset() -> None:
    stream = multilingual_segment_stream(
        languages=("es",),
        include_reflective=False,
        shuffle=True,
        seed=123,
        chunk_size=1,
    )
    batches = list(stream)
    assert len(batches) == len(list(iter_multilingual_texts(("es",), include_reflective=False)))
