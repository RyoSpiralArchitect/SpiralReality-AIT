from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

from .boundary_cpp import CompiledStudentHandle, load_compiled_student
from .boundary_julia import JuliaStudentHandle, load_julia_student
from .np_compat import HAS_NUMPY, np
from .phase import PhaseBasisLearner
from .utils import is_cjk, is_kana, is_latin, is_punct, is_space, sigmoid

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .encoder import ToyTransformerAdapter


_CHAR_CLASSES = ["space", "latin", "cjk", "kana", "punct", "digit", "other"]


def _char_category(ch: str) -> int:
    if is_space(ch):
        return 0
    if is_latin(ch):
        return 1
    if is_cjk(ch):
        return 2
    if is_kana(ch):
        return 3
    if is_punct(ch):
        return 4
    if ch.isdigit():
        return 5
    return 6


@dataclass
class BoundarySequence:
    text: str
    categories: np.ndarray
    labels: np.ndarray
    curvature: np.ndarray
    phases: np.ndarray


@dataclass
class StudentTrainingConfig:
    lr: float = 0.05
    epochs: int = 80
    batch_size: int = 2
    reg: float = 1e-4
    crf_lr: float = 0.02
    phase_lr: float = 1.0
    encoder_lr: float = 2e-3
    hidden_dim: int = 24
    emb_dim: int = 16
    window: int = 2
    validation_split: float = 0.15
    patience: int = 6
    dtype: str = "float32"
    max_grad_norm: Optional[float] = 10.0


class BoundaryStudent:
    """Trainable boundary detector with a shallow NN + CRF head."""

    def __init__(self, phase: PhaseBasisLearner, seed: int = 0):
        self.phase = phase
        self.rng = random.Random(seed)
        self.hidden_dim = 24
        self.emb_dim = 16
        self.window = 2
        self.window_dim = self.emb_dim * (self.window * 2)
        if HAS_NUMPY:
            self.dtype = np.float32 if hasattr(np, "float32") else float
        else:
            self.dtype = float
        self.max_grad_norm: Optional[float] = 10.0
        self._init_parameters()
        self.encoder_adapter: Optional["ToyTransformerAdapter"] = None
        self.history: List[Dict[str, float]] = []
        self.best_state: Optional[Dict[str, object]] = None
        self.julia_backend: Optional[JuliaStudentHandle] = load_julia_student()
        if self.julia_backend is not None:
            try:
                self.julia_backend.attach_phase(self.phase)
            except Exception:
                self.julia_backend = None
        self.compiled_backend: Optional[CompiledStudentHandle] = load_compiled_student()
        if self.compiled_backend is not None:
            try:
                self.compiled_backend.attach_phase(self.phase)
            except Exception:
                self.compiled_backend = None

    def configure(self, cfg: StudentTrainingConfig) -> None:
        self.hidden_dim = cfg.hidden_dim
        self.emb_dim = cfg.emb_dim
        self.window = cfg.window
        self.window_dim = self.emb_dim * (self.window * 2)
        if HAS_NUMPY:
            if cfg.dtype == "float64" and hasattr(np, "float64"):
                self.dtype = np.float64  # type: ignore[assignment]
            else:
                self.dtype = np.float32 if hasattr(np, "float32") else float
        else:
            self.dtype = float
        self.max_grad_norm = cfg.max_grad_norm
        self._init_parameters()
        if self.julia_backend is not None:
            try:
                self.julia_backend.configure(cfg.__dict__)
            except Exception:
                self.julia_backend = None
        if self.compiled_backend is not None:
            try:
                self.compiled_backend.configure(cfg.__dict__)
            except Exception:
                self.compiled_backend = None

    def bind_encoder(self, encoder: "ToyTransformerAdapter") -> None:
        self.encoder_adapter = encoder
        if self.julia_backend is not None:
            try:
                self.julia_backend.attach_encoder(encoder)
            except Exception:
                self.julia_backend = None
        if self.compiled_backend is not None:
            try:
                self.compiled_backend.attach_encoder(encoder)
            except Exception:
                self.compiled_backend = None

    def _init_parameters(self) -> None:
        def rand_vec(size: int, scale: float = 0.1):
            values = [self.rng.uniform(-scale, scale) for _ in range(size)]
            return np.array(values, dtype=self.dtype)

        num_classes = len(_CHAR_CLASSES)
        self.embeddings = np.stack([rand_vec(self.emb_dim, 0.2) for _ in range(num_classes)], axis=0)
        self.W_window = np.stack([rand_vec(self.window_dim, 0.1) for _ in range(self.hidden_dim)], axis=0)
        self.b_window = np.zeros(self.hidden_dim, dtype=self.dtype)
        self.W_out = rand_vec(self.hidden_dim, 0.1)
        self.b_out = 0.0
        self.gate_w = rand_vec(3, 0.05)
        self.gate_b = 0.0
        self.transitions = np.zeros((2, 2), dtype=self.dtype)

    # ------------------------------------------------------------------
    # Dataset construction helpers
    # ------------------------------------------------------------------
    def build_sequences(self, texts: Sequence[str], segments: Sequence[Sequence[str]]) -> List[BoundarySequence]:
        sequences: List[BoundarySequence] = []
        for text, seg in zip(texts, segments):
            categories = np.array([_char_category(ch) for ch in text], dtype=int)
            labels = np.array(self._segments_to_boundaries(text, seg), dtype=int)
            curvature_raw = self.phase.curvature(text)
            curvature = np.array(
                curvature_raw.to_list() if hasattr(curvature_raw, "to_list") else list(curvature_raw),
                dtype=self.dtype,
            )
            phases = np.array([self.phase.phase_triplet(ch) for ch in text], dtype=self.dtype)
            sequences.append(
                BoundarySequence(
                    text=text,
                    categories=categories,
                    labels=labels,
                    curvature=curvature,
                    phases=phases,
                )
            )
        return sequences

    def _segments_to_boundaries(self, text: str, seg: Sequence[str]) -> List[int]:
        cuts = set()
        idx = 0
        for tok in seg:
            idx += len(tok)
            if idx < len(text):
                cuts.add(idx)
        return [1 if (i + 1) in cuts else 0 for i in range(len(text) - 1)]

    def _labels_to_int(self, labels: Sequence[int]) -> List[int]:
        if hasattr(labels, "tolist"):
            raw = labels.tolist()
        else:
            raw = list(labels)
        return [int(round(float(v))) for v in raw]

    # ------------------------------------------------------------------
    # Forward utilities
    # ------------------------------------------------------------------
    def _window_indices(self, idx: int, length: int) -> List[int]:
        indices: List[int] = []
        for offset in range(-self.window + 1, 1):
            pos = idx + offset
            if pos < 0:
                indices.append(-1)
            else:
                indices.append(min(pos, length - 1))
        for offset in range(1, self.window + 1):
            pos = idx + offset
            if pos >= length:
                indices.append(length - 1)
            else:
                indices.append(pos)
        return indices

    def _window_vector(self, embeddings: np.ndarray, indices: List[int]) -> np.ndarray:
        if len(indices) == 0:
            return np.zeros(0, dtype=self.dtype)
        if hasattr(embeddings, "shape"):
            window = np.zeros((len(indices), self.emb_dim), dtype=self.dtype)
            for pos, idx in enumerate(indices):
                if 0 <= idx < embeddings.shape[0]:
                    window[pos] = embeddings[int(idx)]
            return np.reshape(window, (len(indices) * self.emb_dim,))
        vec: List[float] = []
        for idx in indices:
            if idx < 0 or idx >= len(embeddings):
                vec.extend([0.0] * self.emb_dim)
            else:
                vec.extend(embeddings[idx])
        return np.array(vec, dtype=self.dtype)

    def _gate_features(self, seq: BoundarySequence, idx: int) -> np.ndarray:
        curv = seq.curvature
        phases = seq.phases
        left_phase = float(phases[idx][0]) if idx < len(phases) else 0.0
        right_phase = float(phases[idx + 1][1]) if idx + 1 < len(phases) else float(phases[-1][1])
        curv_left = float(curv[idx]) if idx < len(curv) else 0.0
        curv_right = float(curv[idx + 1]) if idx + 1 < len(curv) else float(curv[-1])
        phase_feature = math.sin(left_phase - right_phase)
        curv_feature = math.tanh(0.5 * (curv_left + curv_right))
        return np.array([phase_feature, curv_feature, 1.0], dtype=self.dtype)

    def _linear_forward(self, window_vec: np.ndarray) -> np.ndarray:
        vec = window_vec if hasattr(window_vec, "shape") else np.array(window_vec, dtype=self.dtype)
        return self.W_window @ vec + self.b_window

    def _forward_sequence(self, seq: BoundarySequence) -> Tuple[List[float], List[Dict[str, object]]]:
        length = len(seq.categories)
        embeddings = np.zeros((length, self.emb_dim), dtype=self.dtype)
        for i, cat in enumerate(seq.categories):
            embeddings[i] = self.embeddings[int(cat)]
        caches: List[Dict[str, object]] = []
        logits: List[float] = []
        for idx in range(len(seq.labels)):
            indices = self._window_indices(idx, len(embeddings))
            window_vec = self._window_vector(embeddings, indices)
            pre = self._linear_forward(window_vec)
            hidden = np.tanh(pre) if hasattr(np, "tanh") else np.array([math.tanh(float(v)) for v in pre], dtype=self.dtype)
            gate_feats = self._gate_features(seq, idx)
            core = float(np.dot(self.W_out, hidden)) + self.b_out
            gate_score = float(np.dot(self.gate_w, gate_feats)) + self.gate_b
            logits.append(core + gate_score)
            caches.append(
                {
                    "indices": indices,
                    "window": window_vec,
                    "pre": pre,
                    "hidden": hidden,
                    "gate_feats": gate_feats,
                }
            )
        return logits, caches

    # ------------------------------------------------------------------
    # CRF helpers
    # ------------------------------------------------------------------
    def _logsumexp(self, values: Iterable[float]) -> float:
        vals = list(values)
        m = max(vals)
        if m == float("-inf"):
            return m
        return m + math.log(sum(math.exp(v - m) for v in vals))

    def _crf_loss(self, logits: List[float], labels: List[int]) -> Tuple[float, List[float], List[List[float]], List[List[float]]]:
        length = len(logits)
        if length == 0:
            return 0.0, [], [[0.0, 0.0], [0.0, 0.0]], []
        emit = [[0.0 for _ in range(length)], [logits[i] for i in range(length)]]
        trans = self.transitions
        alpha = [[0.0, 0.0] for _ in range(length)]
        alpha[0][0] = emit[0][0]
        alpha[0][1] = emit[1][0]
        for i in range(1, length):
            for state in (0, 1):
                scores = [alpha[i - 1][prev] + trans[prev][state] for prev in (0, 1)]
                alpha[i][state] = emit[state][i] + self._logsumexp(scores)
        log_z = self._logsumexp(alpha[-1])

        score = emit[labels[0]][0]
        for i in range(1, length):
            score += emit[labels[i]][i] + trans[labels[i - 1]][labels[i]]
        nll = log_z - score

        beta = [[0.0, 0.0] for _ in range(length)]
        beta[-1][0] = 0.0
        beta[-1][1] = 0.0
        for i in range(length - 2, -1, -1):
            for state in (0, 1):
                scores = [
                    trans[state][next_state]
                    + emit[next_state][i + 1]
                    + beta[i + 1][next_state]
                    for next_state in (0, 1)
                ]
                beta[i][state] = self._logsumexp(scores)

        grad_logits = [0.0 for _ in range(length)]
        marginals: List[List[float]] = []
        for i in range(length):
            gamma0 = alpha[i][0] + beta[i][0] - log_z
            gamma1 = alpha[i][1] + beta[i][1] - log_z
            p0 = math.exp(gamma0)
            p1 = math.exp(gamma1)
            grad_logits[i] = p1 - labels[i]
            marginals.append([p0, p1])

        grad_trans = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(1, length):
            for prev in (0, 1):
                for state in (0, 1):
                    xi = (
                        alpha[i - 1][prev]
                        + trans[prev][state]
                        + emit[state][i]
                        + beta[i][state]
                        - log_z
                    )
                    grad_trans[prev][state] += math.exp(xi)
        for i in range(1, length):
            grad_trans[labels[i - 1]][labels[i]] -= 1.0
        grad_logits_arr = np.array(grad_logits, dtype=self.dtype)
        grad_trans_arr = np.array(grad_trans, dtype=self.dtype)
        marginals_arr = [np.array(m, dtype=self.dtype) for m in marginals]
        return nll, grad_logits_arr, grad_trans_arr, marginals_arr

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def split_sequences(
        self, sequences: List[BoundarySequence], validation_split: float
    ) -> Tuple[List[BoundarySequence], List[BoundarySequence]]:
        if validation_split <= 0.0 or len(sequences) < 2:
            return sequences, []
        indices = list(range(len(sequences)))
        self.rng.shuffle(indices)
        cut = max(1, int(len(sequences) * (1.0 - validation_split)))
        train_idx, val_idx = indices[:cut], indices[cut:]
        train = [sequences[i] for i in train_idx]
        val = [sequences[i] for i in val_idx]
        return train, val

    def train(
        self,
        texts: Sequence[str],
        segments: Sequence[Sequence[str]],
        cfg: Optional[StudentTrainingConfig] = None,
    ) -> Dict[str, object]:
        cfg = cfg or StudentTrainingConfig()
        if self.julia_backend is not None:
            cfg_dict = dict(cfg.__dict__)
            try:
                summary = self.julia_backend.train(texts, segments, cfg_dict)
                if isinstance(summary, dict):
                    self.history = list(summary.get("history", []))
                return summary
            except Exception:
                self.julia_backend = None
        if self.compiled_backend is not None:
            cfg_dict = dict(cfg.__dict__)
            try:
                summary = self.compiled_backend.train(texts, segments, cfg_dict)
                if isinstance(summary, dict):
                    self.history = list(summary.get("history", []))
                    summary.setdefault("backend", f"compiled:{self.compiled_backend.device}")
                return summary
            except Exception:
                # Fallback to pure NumPy implementation if compiled backend fails.
                self.compiled_backend = None
        self.configure(cfg)
        sequences = self.build_sequences(texts, segments)
        train_seqs, val_seqs = self.split_sequences(sequences, cfg.validation_split)
        best_val = float("inf")
        patience = 0
        history: List[Dict[str, float]] = []
        start_time = time.perf_counter()
        train_tokens = sum(int(len(seq.labels)) + 1 for seq in train_seqs)
        for epoch in range(cfg.epochs):
            self.rng.shuffle(train_seqs)
            accum = self._zero_grad()
            batch_count = 0
            total_loss = 0.0
            for seq in train_seqs:
                loss, grads, marginals = self._sequence_gradients(seq, cfg)
                total_loss += loss
                self._accumulate(accum, grads)
                batch_count += 1
                if batch_count % cfg.batch_size == 0:
                    self._apply_gradients(accum, cfg, cfg.batch_size)
                    accum = self._zero_grad()
            if batch_count % cfg.batch_size != 0:
                self._apply_gradients(accum, cfg, batch_count % cfg.batch_size)
            metrics = {"epoch": float(epoch + 1), "train_loss": float(total_loss / max(1, len(train_seqs)))}
            if val_seqs:
                val_loss, val_f1 = self.evaluate(val_seqs)
                metrics.update({"val_loss": val_loss, "val_f1": val_f1})
                if val_loss + 1e-6 < best_val:
                    best_val = val_loss
                    patience = 0
                    self.best_state = self._capture_state()
                else:
                    patience += 1
                    if patience >= cfg.patience:
                        history.append(metrics)
                        break
            history.append(metrics)
        if self.best_state is not None:
            self._restore_state(self.best_state)
        self.history = history
        elapsed = max(1e-9, time.perf_counter() - start_time)
        summary: Dict[str, object] = {
            "train_sequences": len(train_seqs),
            "history": history,
            "train_tokens": train_tokens,
            "train_seconds": elapsed,
            "tokens_per_second": train_tokens / elapsed if train_tokens else 0.0,
            "backend": (
                f"julia:{self.julia_backend.device}"
                if self.julia_backend is not None
                else (f"compiled:{self.compiled_backend.device}" if self.compiled_backend is not None else ("numpy" if HAS_NUMPY else "stub"))
            ),
            "dtype": getattr(self.dtype, "name", getattr(self.dtype, "__name__", str(self.dtype))),
        }
        if val_seqs:
            summary["val_sequences"] = len(val_seqs)
            summary["val_tokens"] = sum(int(len(seq.labels)) + 1 for seq in val_seqs)
            last = history[-1]
            if "val_loss" in last:
                summary["val_loss"] = last["val_loss"]
                summary["val_f1"] = last["val_f1"]
        return summary

    def _zero_grad(self) -> Dict[str, object]:
        num_classes = len(self.embeddings)
        return {
            "embeddings": np.zeros((num_classes, self.emb_dim), dtype=self.dtype),
            "W_window": np.zeros((self.hidden_dim, self.window_dim), dtype=self.dtype),
            "b_window": np.zeros(self.hidden_dim, dtype=self.dtype),
            "W_out": np.zeros(self.hidden_dim, dtype=self.dtype),
            "b_out": 0.0,
            "gate_w": np.zeros(len(self.gate_w), dtype=self.dtype),
            "gate_b": 0.0,
            "transitions": np.zeros((2, 2), dtype=self.dtype),
        }

    def _sequence_gradients(
        self, seq: BoundarySequence, cfg: StudentTrainingConfig
    ) -> Tuple[float, Dict[str, object], List[np.ndarray]]:
        logits, caches = self._forward_sequence(seq)
        label_list = self._labels_to_int(seq.labels)
        loss, grad_logits, grad_trans, marginals = self._crf_loss(logits, label_list)
        grads = self._zero_grad()
        embed_grads = np.zeros((len(seq.categories), self.emb_dim), dtype=self.dtype)
        for i, cache in enumerate(caches):
            grad_logit = float(grad_logits[i])
            hidden = cache["hidden"]
            pre = cache["pre"]
            window_vec = cache["window"]
            indices = cache["indices"]
            gate_feats = cache["gate_feats"]

            grads["gate_w"] += grad_logit * gate_feats
            grads["gate_b"] += grad_logit
            self.phase.apply_error(seq.text, i, grad_logit, scale=cfg.phase_lr)

            grads["W_out"] += grad_logit * hidden
            grads["b_out"] += grad_logit

            grad_hidden = grad_logit * self.W_out
            hidden_sq = hidden * hidden if hasattr(hidden, "__mul__") else np.array([float(h) ** 2 for h in hidden])
            grad_pre = grad_hidden * (1.0 - hidden_sq)
            grads["b_window"] += grad_pre
            if HAS_NUMPY:
                grads["W_window"] += np.outer(grad_pre, window_vec)
                grad_window = self.W_window.T @ grad_pre
                grad_window_matrix = np.reshape(grad_window, (len(indices), self.emb_dim))
                for pos, char_idx in enumerate(indices):
                    if 0 <= char_idx < embed_grads.shape[0]:
                        embed_grads[int(char_idx)] += grad_window_matrix[pos]
            else:
                grad_pre_vals = grad_pre.tolist() if hasattr(grad_pre, "tolist") else list(grad_pre)
                window_vals = window_vec.tolist() if hasattr(window_vec, "tolist") else list(window_vec)
                outer = [[gp * wv for wv in window_vals] for gp in grad_pre_vals]
                grads["W_window"] += np.array(outer, dtype=self.dtype)
                grad_window = self.W_window.T @ np.array(grad_pre_vals, dtype=self.dtype)
                grad_window_vals = grad_window.tolist() if hasattr(grad_window, "tolist") else list(grad_window)
                for pos, char_idx in enumerate(indices):
                    if 0 <= char_idx < embed_grads.shape[0]:
                        start = pos * self.emb_dim
                        end = start + self.emb_dim
                        slice_vals = grad_window_vals[start:end]
                        embed_grads[int(char_idx)] += np.array(slice_vals, dtype=self.dtype)

        for pos, cat in enumerate(seq.categories):
            grads["embeddings"][int(cat)] += embed_grads[pos]

        grads["transitions"] += grad_trans

        if self.encoder_adapter is not None:
            gate_targets = self._char_gate_targets(label_list, marginals)
            base_gate = [sigmoid(float(c)) for c in (seq.curvature.tolist() if hasattr(seq.curvature, "tolist") else seq.curvature)]
            self.encoder_adapter.tune_from_boundary(base_gate, gate_targets, lr=cfg.encoder_lr)

        loss += 0.5 * cfg.reg * self._l2_norm()
        return float(loss), grads, marginals

    def _char_gate_targets(self, labels: List[int], marginals: List[List[float]]) -> List[float]:
        length = len(labels) + 1
        targets = [0.0 for _ in range(length)]
        preds = [m[1] for m in marginals]
        for i, label in enumerate(labels):
            if label >= 0.5:
                targets[i] = max(targets[i], 1.0)
                targets[i + 1] = max(targets[i + 1], 1.0)
            else:
                val = preds[i]
                targets[i] = max(targets[i], val)
                targets[i + 1] = max(targets[i + 1], val)
        return targets

    def _l2_norm(self) -> float:
        total = 0.0
        total += float(np.sum(self.W_window * self.W_window))
        total += float(np.sum(self.W_out * self.W_out))
        total += float(np.sum(self.gate_w * self.gate_w))
        total += float(np.sum(self.embeddings * self.embeddings))
        return total

    def _accumulate(self, accum: Dict[str, object], grads: Dict[str, object]) -> None:
        accum["embeddings"] += grads["embeddings"]
        accum["W_window"] += grads["W_window"]
        accum["b_window"] += grads["b_window"]
        accum["W_out"] += grads["W_out"]
        accum["b_out"] += grads["b_out"]
        accum["gate_w"] += grads["gate_w"]
        accum["gate_b"] += grads["gate_b"]
        accum["transitions"] += grads["transitions"]

    def _apply_gradients(self, grads: Dict[str, object], cfg: StudentTrainingConfig, batch_size: int) -> None:
        base_scale = cfg.lr / max(1, batch_size)
        grad_scale = 1.0
        if cfg.max_grad_norm:
            norm = self._grad_norm(grads)
            if norm > cfg.max_grad_norm:
                grad_scale = cfg.max_grad_norm / (norm + 1e-9)
        scale = base_scale * grad_scale
        crf_scale = cfg.crf_lr * grad_scale

        self.embeddings -= scale * (grads["embeddings"] + cfg.reg * self.embeddings)
        self.W_window -= scale * (grads["W_window"] + cfg.reg * self.W_window)
        self.b_window -= scale * grads["b_window"]
        self.W_out -= scale * (grads["W_out"] + cfg.reg * self.W_out)
        self.b_out -= scale * grads["b_out"]
        self.gate_w -= scale * (grads["gate_w"] + cfg.reg * self.gate_w)
        self.gate_b -= scale * grads["gate_b"]
        self.transitions -= crf_scale * (grads["transitions"] + cfg.reg * self.transitions)

    def _grad_norm(self, grads: Dict[str, object]) -> float:
        total = 0.0
        total += float(np.sum(grads["embeddings"] * grads["embeddings"]))
        total += float(np.sum(grads["W_window"] * grads["W_window"]))
        total += float(np.sum(grads["b_window"] * grads["b_window"]))
        total += float(np.sum(grads["W_out"] * grads["W_out"]))
        total += float(np.sum(grads["gate_w"] * grads["gate_w"]))
        total += float(np.sum(grads["transitions"] * grads["transitions"]))
        total += float(grads["b_out"] ** 2)
        total += float(grads["gate_b"] ** 2)
        return math.sqrt(total)

    def _capture_state(self) -> Dict[str, object]:
        return {
            "embeddings": self.embeddings.tolist() if hasattr(self.embeddings, "tolist") else [row[:] for row in self.embeddings],
            "W_window": self.W_window.tolist() if hasattr(self.W_window, "tolist") else [row[:] for row in self.W_window],
            "b_window": self.b_window.tolist() if hasattr(self.b_window, "tolist") else self.b_window[:],
            "W_out": self.W_out.tolist() if hasattr(self.W_out, "tolist") else self.W_out[:],
            "b_out": self.b_out,
            "gate_w": self.gate_w.tolist() if hasattr(self.gate_w, "tolist") else self.gate_w[:],
            "gate_b": self.gate_b,
            "transitions": self.transitions.tolist() if hasattr(self.transitions, "tolist") else [row[:] for row in self.transitions],
        }

    def _restore_state(self, state: Dict[str, object]) -> None:
        self.embeddings = np.array(state["embeddings"], dtype=self.dtype)
        self.W_window = np.array(state["W_window"], dtype=self.dtype)
        self.b_window = np.array(state["b_window"], dtype=self.dtype)
        self.W_out = np.array(state["W_out"], dtype=self.dtype)
        self.b_out = float(state["b_out"])
        self.gate_w = np.array(state["gate_w"], dtype=self.dtype)
        self.gate_b = float(state["gate_b"])
        self.transitions = np.array(state["transitions"], dtype=self.dtype)

    def export_state(self) -> Dict[str, object]:
        state = self._capture_state()
        if self.julia_backend is not None:
            try:
                state["_julia"] = {
                    "backend": self.julia_backend.backend,
                    "device": self.julia_backend.device,
                    "state": self.julia_backend.export_state(),
                }
            except Exception:
                pass
        if self.compiled_backend is not None:
            try:
                state["_compiled"] = {
                    "device": self.compiled_backend.device,
                    "state": self.compiled_backend.export_state(),
                }
            except Exception:
                pass
        return state

    def load_state(self, state: Dict[str, object]) -> None:
        compiled_state = state.get("_compiled") if isinstance(state, dict) else None
        julia_state = state.get("_julia") if isinstance(state, dict) else None
        base = dict(state) if isinstance(state, dict) else state
        if isinstance(base, dict) and "_compiled" in base:
            base = dict(base)
            base.pop("_compiled", None)
        if isinstance(base, dict) and "_julia" in base:
            base = dict(base)
            base.pop("_julia", None)
        self._restore_state(base)
        if julia_state and self.julia_backend is not None:
            try:
                self.julia_backend.load_state(julia_state.get("state", {}))
            except Exception:
                self.julia_backend = None
        if compiled_state and self.compiled_backend is not None:
            try:
                self.compiled_backend.load_state(compiled_state.get("state", {}))
            except Exception:
                self.compiled_backend = None

    # ------------------------------------------------------------------
    # Evaluation and inference
    # ------------------------------------------------------------------
    def evaluate(self, sequences: Sequence[BoundarySequence]) -> Tuple[float, float]:
        total_loss = 0.0
        total_tp = total_fp = total_fn = 0
        for seq in sequences:
            logits, _ = self._forward_sequence(seq)
            labels = self._labels_to_int(seq.labels)
            loss, _, _, marginals = self._crf_loss(logits, labels)
            total_loss += loss
            preds = self._viterbi(logits)
            tp, fp, fn = self._boundary_confusion(preds, labels)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return total_loss / max(1, len(sequences)), f1

    def _boundary_confusion(self, preds: List[int], labels: List[int]) -> Tuple[int, int, int]:
        tp = fp = fn = 0
        for p, y in zip(preds, labels):
            if p == 1 and y == 1:
                tp += 1
            elif p == 1 and y == 0:
                fp += 1
            elif p == 0 and y == 1:
                fn += 1
        return tp, fp, fn

    def _viterbi(self, logits: List[float]) -> List[int]:
        length = len(logits)
        if length == 0:
            return []
        trans = self.transitions
        emit = [[0.0 for _ in range(length)], [logits[i] for i in range(length)]]
        dp = [[float("-inf"), float("-inf")] for _ in range(length)]
        back: List[List[int]] = [[0, 0] for _ in range(length)]
        dp[0][0] = emit[0][0]
        dp[0][1] = emit[1][0]
        for i in range(1, length):
            for state in (0, 1):
                best_score = float("-inf")
                best_prev = 0
                for prev in (0, 1):
                    trans_val = trans[prev][state]
                    score = dp[i - 1][prev] + (float(trans_val) if hasattr(trans_val, "__float") else trans_val)
                    if score > best_score:
                        best_score = score
                        best_prev = prev
                dp[i][state] = best_score + emit[state][i]
                back[i][state] = best_prev
        last_state = 1 if dp[-1][1] > dp[-1][0] else 0
        out = [0 for _ in range(length)]
        out[-1] = last_state
        for i in range(length - 1, 0, -1):
            out[i - 1] = back[i][out[i]]
        return out

    def boundary_probs(self, text: str) -> np.ndarray:
        if self.julia_backend is not None:
            try:
                return self.julia_backend.boundary_probs(text)
            except Exception:
                self.julia_backend = None
        if self.compiled_backend is not None:
            try:
                return self.compiled_backend.boundary_probs(text)
            except Exception:
                self.compiled_backend = None
        if len(text) <= 1:
            return np.zeros(0, dtype=float)
        seq = self.build_sequences([text], [[text]])[0]
        logits, _ = self._forward_sequence(seq)
        labels = self._labels_to_int(seq.labels)
        _, _, _, marginals = self._crf_loss(logits, labels)
        probs = [m[1] for m in marginals]
        return np.array(probs, dtype=float)

    def decode(self, text: str) -> List[str]:
        if self.julia_backend is not None:
            try:
                return list(self.julia_backend.decode(text))
            except Exception:
                self.julia_backend = None
        if self.compiled_backend is not None:
            try:
                return list(self.compiled_backend.decode(text))
            except Exception:
                self.compiled_backend = None
        seq = self.build_sequences([text], [[text]])[0]
        logits, _ = self._forward_sequence(seq)
        labels = self._viterbi(logits)
        tokens: List[str] = []
        start = 0
        for i, label in enumerate(labels):
            if label == 1:
                tokens.append(text[start : i + 1])
                start = i + 1
        tokens.append(text[start:])
        return tokens

