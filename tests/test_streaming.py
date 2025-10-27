import itertools

import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated.corpus import (
    TRAIN_TEXTS,
    teacher_segments,
)
from spiralreality_AIT_onepass_aifcore_integrated.integrated.multilingual import (
    AVAILABLE_LANGUAGES,
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
