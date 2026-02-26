"""Chunked streaming inference helpers for text segmentation.

The main model APIs operate on full strings which can become expensive for long
streams (especially when contextual encoders scale quadratically in sequence
length). This module provides a small stateful wrapper that:

- Maintains a rolling context suffix for feature quality near chunk boundaries.
- Limits the maximum window length passed to the underlying segmenter.
- Commits tokens incrementally while keeping a configurable lookahead tail.

The implementation is intentionally lightweight and model-agnostic: callers
provide a ``segmenter(text) -> tokens`` callable (for example
``OnePassAIT.segment_text``).
"""

from __future__ import annotations

from typing import Callable, List, Sequence

SegmenterFn = Callable[[str], Sequence[str]]


class ChunkedStreamingSegmenter:
    """Stateful chunked text segmenter for streaming workloads."""

    def __init__(
        self,
        segmenter: SegmenterFn,
        *,
        max_window_chars: int = 512,
        lookahead_chars: int = 64,
        context_chars: int = 128,
        hard_split: bool = True,
    ) -> None:
        if max_window_chars <= 8:
            raise ValueError("max_window_chars must be > 8")
        if lookahead_chars < 0:
            raise ValueError("lookahead_chars must be >= 0")
        if context_chars < 0:
            raise ValueError("context_chars must be >= 0")
        self.segmenter = segmenter
        self.max_window_chars = int(max_window_chars)
        self.lookahead_chars = int(lookahead_chars)
        self.context_chars = int(context_chars)
        self.hard_split = bool(hard_split)
        self._context_tokens: List[str] = []
        self._pending: str = ""

    @property
    def pending_text(self) -> str:
        return self._pending

    def reset(self) -> None:
        self._context_tokens = []
        self._pending = ""

    def feed(self, chunk: str) -> List[str]:
        if not chunk:
            return []
        self._pending += str(chunk)
        return self._drain(final=False)

    def flush(self, *, reset: bool = True) -> List[str]:
        out = self._drain(final=True)
        if reset:
            self.reset()
        return out

    def _context_text(self) -> str:
        if not self._context_tokens:
            return ""
        return "".join(self._context_tokens)

    def _trim_context_tokens(self) -> None:
        if self.context_chars <= 0:
            self._context_tokens = []
            return
        total = sum(len(tok) for tok in self._context_tokens)
        while total > self.context_chars and len(self._context_tokens) > 1:
            dropped = self._context_tokens.pop(0)
            total -= len(dropped)

    def _drain(self, *, final: bool) -> List[str]:
        out: List[str] = []
        if final:
            while self._pending:
                committed = self._commit_once(final=True)
                if not committed:
                    # Defensive: avoid infinite loops by emitting the remaining buffer.
                    out.append(self._pending)
                    self._context_tokens.append(self._pending)
                    self._trim_context_tokens()
                    self._pending = ""
                    break
                out.extend(committed)
            return out

        while True:
            committed = self._commit_once(final=False)
            if not committed:
                break
            out.extend(committed)
            if len(self._pending) <= self.lookahead_chars:
                break
        return out

    def _commit_once(self, *, final: bool) -> List[str]:
        if not self._pending:
            return []
        context_text = self._context_text()
        if context_text and len(context_text) >= self.max_window_chars:
            # Keep the most recent context even if it exceeds the budget.
            keep = max(0, self.max_window_chars // 2)
            context_text = context_text[-keep:]
        context_len = len(context_text)
        budget = self.max_window_chars - context_len
        if budget <= 0:
            context_text = ""
            context_len = 0
            budget = self.max_window_chars
        pending_prefix = self._pending[:budget]
        if not pending_prefix:
            return []
        work_text = context_text + pending_prefix

        result = self.segmenter(work_text)
        if isinstance(result, dict) and "tokens" in result:
            tokens = list(result["tokens"])
        else:
            tokens = list(result)

        effective_lookahead = 0
        if not final:
            effective_lookahead = min(self.lookahead_chars, max(0, len(pending_prefix) - 1))
        commit_limit = len(work_text) - effective_lookahead
        commit_limit = max(context_len, commit_limit)

        committed: List[str] = []
        pos = 0
        for tok in tokens:
            start = pos
            end = pos + len(tok)
            pos = end
            if end <= context_len:
                continue
            if start < context_len < end:
                tok = tok[context_len - start :]
                start = context_len
            if end <= commit_limit:
                if tok:
                    committed.append(tok)
            else:
                break

        if not committed and self.hard_split and commit_limit > context_len:
            forced = work_text[context_len:commit_limit]
            if forced:
                committed = [forced]

        if not committed:
            return []

        committed_chars = sum(len(tok) for tok in committed)
        self._pending = self._pending[committed_chars:]
        self._context_tokens.extend(committed)
        self._trim_context_tokens()
        return committed


__all__ = ["ChunkedStreamingSegmenter"]

