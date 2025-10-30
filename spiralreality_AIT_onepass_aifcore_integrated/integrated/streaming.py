"""Streaming helpers for chunked boundary supervision ingestion.

These utilities expose a small ``SegmentStream`` abstraction that lazily wraps
``teacher_segments`` so training loops can consume curated supervision without
materialising the full corpus in memory.  The stream optionally prefetches
segments on a background thread which provides basic backpressure control when
downstream consumers process batches slowly.
"""

from __future__ import annotations

import itertools
import operator
import queue
import threading
from contextlib import suppress
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional, Sequence, Tuple, TypeVar

from .corpus import teacher_segments

T = TypeVar("T")


Segmenter = Callable[[Sequence[str]], Sequence[Sequence[str]]]


@dataclass(frozen=True)
class SegmentBatch:
    """Container bundling a batch of texts with their teacher segments.

    Parameters
    ----------
    texts:
        Ordered batch of raw input strings.
    segments:
        Teacher segmentations aligned with :attr:`texts`.
    metadata:
        Optional metadata (for example language tags) aligned with
        :attr:`texts`.  When provided it is stored as a tuple with the same
        cardinality as :attr:`texts`.
    """

    texts: Tuple[str, ...]
    segments: Tuple[Tuple[str, ...], ...]
    metadata: Optional[Tuple[T, ...]] = None


class SegmentStream(Iterable[SegmentBatch]):
    """Lazily batch texts and resolve their teacher segments.

    Parameters
    ----------
    texts:
        Iterable providing raw input strings.  The iterable is consumed lazily
        so it can be a generator.
    segmenter:
        Callable resolving teacher segmentations for a batch of texts.  Defaults
        to :func:`teacher_segments`.
    metadata:
        Optional iterable that yields metadata objects aligned with ``texts``.
    chunk_size:
        Number of examples per yielded batch.  Must be positive.
    max_prefetch:
        When greater than zero the stream resolves up to ``max_prefetch``
        batches on a background thread.  This keeps a small buffer of ready
        batches which provides basic backpressure when the consumer is slower
        than the segmenter.
    drop_incomplete:
        Drop the final batch when it is smaller than ``chunk_size``.  Useful
        for training loops that require fixed batch sizes.

    Notes
    -----
    Instances expose lightweight ``*_length_hint`` properties that report
    best-effort counts for the underlying texts, metadata, and produced batch
    count when such information can be derived without materialising the
    iterables.
    """

    def __init__(
        self,
        texts: Iterable[str],
        *,
        segmenter: Segmenter = teacher_segments,
        metadata: Optional[Iterable[T]] = None,
        chunk_size: int = 32,
        max_prefetch: int = 0,
        drop_incomplete: bool = False,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if max_prefetch < 0:
            raise ValueError("max_prefetch must be non-negative")
        self._texts = texts
        self._segmenter = segmenter
        self._metadata = metadata
        self._chunk_size = int(chunk_size)
        self._max_prefetch = int(max_prefetch)
        self._drop_incomplete = bool(drop_incomplete)
        self._text_length_hint = operator.length_hint(texts, -1)
        self._metadata_length_hint = (
            operator.length_hint(metadata, -1) if metadata is not None else -1
        )

        if (
            metadata is not None
            and isinstance(texts, Sequence)
            and isinstance(metadata, Sequence)
            and len(texts) != len(metadata)
        ):
            raise ValueError("metadata iterable must have the same length as texts")

    def __iter__(self) -> Iterator[SegmentBatch]:
        chunk_iter = self._iter_chunk_inputs()
        if self._max_prefetch == 0:
            for chunk_texts, metadata in chunk_iter:
                yield self._build_batch(chunk_texts, metadata)
            return

        sentinel = object()
        buffer: "queue.SimpleQueue[object]" = queue.SimpleQueue()
        slots = threading.BoundedSemaphore(self._max_prefetch)
        stop_event = threading.Event()
        errors: list[BaseException] = []

        def acquire_slot() -> bool:
            while True:
                if stop_event.is_set():
                    return False
                acquired = slots.acquire(timeout=0.05)
                if acquired:
                    return True

        def producer() -> None:
            try:
                for chunk_texts, metadata in chunk_iter:
                    if not acquire_slot():
                        break
                    if stop_event.is_set():
                        slots.release()
                        break
                    try:
                        batch = self._build_batch(chunk_texts, metadata)
                    except BaseException as exc:  # pragma: no cover - defensive path
                        slots.release()
                        errors.append(exc)
                        break
                    buffer.put(batch)
            except BaseException as exc:  # pragma: no cover - defensive path
                errors.append(exc)
            finally:
                buffer.put(sentinel)

        worker = threading.Thread(target=producer, daemon=True)
        worker.start()
        try:
            while True:
                item = buffer.get()
                if item is sentinel:
                    break
                yield item  # type: ignore[misc]
                slots.release()
        finally:
            stop_event.set()
            with suppress(queue.Empty):
                while True:
                    leftover = buffer.get_nowait()
                    if leftover is sentinel:
                        continue
                    slots.release()

            worker.join()

        if errors:
            raise errors[0]

    @property
    def text_length_hint(self) -> Optional[int]:
        """Return a best-effort estimate of the number of texts in the stream."""

        return None if self._text_length_hint < 0 else self._text_length_hint

    @property
    def batch_length_hint(self) -> Optional[int]:
        """Return an estimated number of batches produced by the stream."""

        hint = self.text_length_hint
        if hint is None:
            return None
        if self._drop_incomplete:
            return hint // self._chunk_size
        return (hint + self._chunk_size - 1) // self._chunk_size

    @property
    def metadata_length_hint(self) -> Optional[int]:
        """Return a best-effort estimate for the metadata length.

        When :attr:`drop_incomplete` is enabled the estimate is truncated to the
        nearest multiple of :attr:`chunk_size` to reflect the amount of
        metadata that will actually be emitted by the stream.
        """

        if self._metadata is None or self._metadata_length_hint < 0:
            return None
        hint = self._metadata_length_hint
        if self._drop_incomplete and hint > 0:
            remainder = hint % self._chunk_size
            if remainder:
                hint -= remainder
        return hint

    def _iter_chunk_inputs(self) -> Iterator[Tuple[Tuple[str, ...], Optional[Tuple[T, ...]]]]:
        text_iter = iter(self._texts)
        meta_iter = iter(self._metadata) if self._metadata is not None else None

        while True:
            chunk_texts: list[str] = []
            chunk_meta: Optional[list[T]] = [] if meta_iter is not None else None

            for _ in range(self._chunk_size):
                try:
                    text = next(text_iter)
                except StopIteration:
                    break
                chunk_texts.append(text)
                if chunk_meta is not None:
                    try:
                        chunk_meta.append(next(meta_iter))  # type: ignore[arg-type]
                    except StopIteration as exc:
                        raise ValueError(
                            "metadata iterable exhausted before texts"
                        ) from exc

            if not chunk_texts:
                if meta_iter is not None:
                    try:
                        next(meta_iter)
                    except StopIteration:
                        pass
                    else:  # pragma: no cover - defensive guard
                        raise ValueError("metadata iterable has extra items")
                break

            if self._drop_incomplete and len(chunk_texts) < self._chunk_size:
                if meta_iter is not None:
                    try:
                        next(meta_iter)
                    except StopIteration:
                        pass
                    else:
                        raise ValueError("metadata iterable has extra items")
                break

            texts_tuple = tuple(chunk_texts)
            if chunk_meta is not None:
                metadata_tuple: Optional[Tuple[T, ...]] = tuple(chunk_meta)
                if len(metadata_tuple) != len(texts_tuple):
                    raise ValueError("metadata length does not match batch size")
            else:
                metadata_tuple = None

            yield texts_tuple, metadata_tuple

    def _build_batch(
        self,
        chunk_texts: Tuple[str, ...],
        metadata: Optional[Tuple[T, ...]],
    ) -> SegmentBatch:
        segments = self._segmenter(chunk_texts)
        if len(segments) != len(chunk_texts):
            raise ValueError(
                "segmenter returned a different number of segments than texts"
            )

        segment_tuples = tuple(tuple(seg) for seg in segments)
        return SegmentBatch(
            texts=chunk_texts,
            segments=segment_tuples,
            metadata=metadata,
        )


def chunked(iterable: Iterable[T], size: int) -> Iterator[Tuple[T, ...]]:
    """Yield fixed-size chunks from ``iterable``.

    The final chunk may be smaller when the input is not perfectly divisible by
    ``size``.  ``size`` must be positive.
    """

    if size <= 0:
        raise ValueError("size must be positive")

    it = iter(iterable)
    while True:
        batch = tuple(itertools.islice(it, size))
        if not batch:
            break
        yield batch


__all__ = ["SegmentBatch", "SegmentStream", "chunked"]
