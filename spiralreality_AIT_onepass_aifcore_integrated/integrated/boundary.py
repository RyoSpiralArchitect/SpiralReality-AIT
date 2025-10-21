from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

from .np_compat import np
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
    categories: List[int]
    labels: List[int]
    curvature: List[float]
    phases: List[Tuple[float, float, float]]


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


class BoundaryStudent:
    """Trainable boundary detector with a shallow NN + CRF head."""

    def __init__(self, phase: PhaseBasisLearner, seed: int = 0):
        self.phase = phase
        self.rng = random.Random(seed)
        self.hidden_dim = 24
        self.emb_dim = 16
        self.window = 2
        self.window_dim = self.emb_dim * (self.window * 2)
        self._init_parameters()
        self.encoder_adapter: Optional["ToyTransformerAdapter"] = None
        self.history: List[Dict[str, float]] = []
        self.best_state: Optional[Dict[str, object]] = None

    def configure(self, cfg: StudentTrainingConfig) -> None:
        self.hidden_dim = cfg.hidden_dim
        self.emb_dim = cfg.emb_dim
        self.window = cfg.window
        self.window_dim = self.emb_dim * (self.window * 2)
        self._init_parameters()

    def bind_encoder(self, encoder: "ToyTransformerAdapter") -> None:
        self.encoder_adapter = encoder

    def _init_parameters(self) -> None:
        def rand_vec(size: int, scale: float = 0.1) -> List[float]:
            return [self.rng.uniform(-scale, scale) for _ in range(size)]

        num_classes = len(_CHAR_CLASSES)
        self.embeddings: List[List[float]] = [rand_vec(self.emb_dim, 0.2) for _ in range(num_classes)]
        self.W_window: List[List[float]] = [rand_vec(self.window_dim, 0.1) for _ in range(self.hidden_dim)]
        self.b_window: List[float] = [0.0 for _ in range(self.hidden_dim)]
        self.W_out: List[float] = rand_vec(self.hidden_dim, 0.1)
        self.b_out: float = 0.0
        self.gate_w: List[float] = rand_vec(3, 0.05)
        self.gate_b: float = 0.0
        self.transitions: List[List[float]] = [[0.0, 0.0], [0.0, 0.0]]

    # ------------------------------------------------------------------
    # Dataset construction helpers
    # ------------------------------------------------------------------
    def build_sequences(self, texts: Sequence[str], segments: Sequence[Sequence[str]]) -> List[BoundarySequence]:
        sequences: List[BoundarySequence] = []
        for text, seg in zip(texts, segments):
            categories = [_char_category(ch) for ch in text]
            labels = self._segments_to_boundaries(text, seg)
            curvature = self.phase.curvature(text)
            phases = [self.phase.phase_triplet(ch) for ch in text]
            sequences.append(
                BoundarySequence(
                    text=text,
                    categories=categories,
                    labels=labels,
                    curvature=curvature.to_list() if hasattr(curvature, "to_list") else list(curvature),
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

    def _window_vector(self, embeddings: List[List[float]], indices: List[int]) -> List[float]:
        vec: List[float] = []
        for idx in indices:
            if idx < 0 or idx >= len(embeddings):
                vec.extend([0.0] * self.emb_dim)
            else:
                vec.extend(embeddings[idx])
        return vec

    def _gate_features(self, seq: BoundarySequence, idx: int) -> List[float]:
        curv = seq.curvature
        phases = seq.phases
        left_phase = phases[idx][0] if idx < len(phases) else 0.0
        right_phase = phases[idx + 1][1] if idx + 1 < len(phases) else phases[-1][1]
        curv_left = curv[idx] if idx < len(curv) else 0.0
        curv_right = curv[idx + 1] if idx + 1 < len(curv) else curv[-1]
        phase_feature = math.sin(left_phase - right_phase)
        curv_feature = math.tanh(0.5 * (curv_left + curv_right))
        return [phase_feature, curv_feature, 1.0]

    def _linear_forward(self, window_vec: List[float]) -> List[float]:
        out: List[float] = []
        for j in range(self.hidden_dim):
            total = self.b_window[j]
            weights = self.W_window[j]
            total += sum(weights[k] * window_vec[k] for k in range(self.window_dim))
            out.append(total)
        return out

    def _forward_sequence(self, seq: BoundarySequence) -> Tuple[List[float], List[Dict[str, object]]]:
        embeddings = [self.embeddings[cat][:] for cat in seq.categories]
        caches: List[Dict[str, object]] = []
        logits: List[float] = []
        for idx in range(len(seq.labels)):
            indices = self._window_indices(idx, len(embeddings))
            window_vec = self._window_vector(embeddings, indices)
            pre = self._linear_forward(window_vec)
            hidden = [math.tanh(v) for v in pre]
            gate_feats = self._gate_features(seq, idx)
            core = sum(self.W_out[j] * hidden[j] for j in range(self.hidden_dim)) + self.b_out
            gate_score = sum(self.gate_w[k] * gate_feats[k] for k in range(len(self.gate_w))) + self.gate_b
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
        return nll, grad_logits, grad_trans, marginals

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
        self.configure(cfg)
        sequences = self.build_sequences(texts, segments)
        train_seqs, val_seqs = self.split_sequences(sequences, cfg.validation_split)
        best_val = float("inf")
        patience = 0
        history: List[Dict[str, float]] = []
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
            metrics = {"epoch": epoch + 1, "train_loss": total_loss / max(1, len(train_seqs))}
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
        summary: Dict[str, object] = {
            "train_sequences": len(train_seqs),
            "history": history,
        }
        if val_seqs:
            summary["val_sequences"] = len(val_seqs)
            last = history[-1]
            if "val_loss" in last:
                summary["val_loss"] = last["val_loss"]
                summary["val_f1"] = last["val_f1"]
        return summary

    def _zero_grad(self) -> Dict[str, object]:
        num_classes = len(self.embeddings)
        return {
            "embeddings": [[0.0 for _ in range(self.emb_dim)] for _ in range(num_classes)],
            "W_window": [[0.0 for _ in range(self.window_dim)] for _ in range(self.hidden_dim)],
            "b_window": [0.0 for _ in range(self.hidden_dim)],
            "W_out": [0.0 for _ in range(self.hidden_dim)],
            "b_out": 0.0,
            "gate_w": [0.0 for _ in range(len(self.gate_w))],
            "gate_b": 0.0,
            "transitions": [[0.0, 0.0], [0.0, 0.0]],
        }

    def _sequence_gradients(
        self, seq: BoundarySequence, cfg: StudentTrainingConfig
    ) -> Tuple[float, Dict[str, object], List[List[float]]]:
        logits, caches = self._forward_sequence(seq)
        loss, grad_logits, grad_trans, marginals = self._crf_loss(logits, seq.labels)
        grads = self._zero_grad()
        embed_grads = [[0.0 for _ in range(self.emb_dim)] for _ in range(len(seq.categories))]
        for i, cache in enumerate(caches):
            grad_logit = grad_logits[i]
            hidden = cache["hidden"]
            pre = cache["pre"]
            window_vec = cache["window"]
            indices = cache["indices"]
            gate_feats = cache["gate_feats"]

            for k in range(len(self.gate_w)):
                grads["gate_w"][k] += grad_logit * gate_feats[k]
            grads["gate_b"] += grad_logit
            self.phase.apply_error(seq.text, i, grad_logit, scale=cfg.phase_lr)

            for j in range(self.hidden_dim):
                grads["W_out"][j] += grad_logit * hidden[j]
            grads["b_out"] += grad_logit

            grad_hidden = [grad_logit * self.W_out[j] for j in range(self.hidden_dim)]
            grad_pre = [grad_hidden[j] * (1.0 - math.tanh(pre[j]) ** 2) for j in range(self.hidden_dim)]
            for j in range(self.hidden_dim):
                grads["b_window"][j] += grad_pre[j]
            grad_window = [0.0 for _ in range(self.window_dim)]
            for j in range(self.hidden_dim):
                weights = self.W_window[j]
                for k in range(self.window_dim):
                    grads["W_window"][j][k] += grad_pre[j] * window_vec[k]
                    grad_window[k] += grad_pre[j] * weights[k]
            for pos, char_idx in enumerate(indices):
                base = pos * self.emb_dim
                for d in range(self.emb_dim):
                    grad = grad_window[base + d]
                    if 0 <= char_idx < len(embed_grads):
                        embed_grads[char_idx][d] += grad
        for pos, cat in enumerate(seq.categories):
            grad_vec = embed_grads[pos]
            target = grads["embeddings"][cat]
            for d in range(self.emb_dim):
                target[d] += grad_vec[d]
        for prev in (0, 1):
            for state in (0, 1):
                grads["transitions"][prev][state] += grad_trans[prev][state]
        if self.encoder_adapter is not None:
            gate_targets = self._char_gate_targets(seq.labels, marginals)
            base_gate = [sigmoid(c) for c in seq.curvature]
            self.encoder_adapter.tune_from_boundary(base_gate, gate_targets, lr=cfg.encoder_lr)
        loss += 0.5 * cfg.reg * self._l2_norm()
        return loss, grads, marginals

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
        for row in self.W_window:
            total += sum(w * w for w in row)
        total += sum(w * w for w in self.W_out)
        total += sum(w * w for w in self.gate_w)
        for row in self.embeddings:
            total += sum(v * v for v in row)
        return total

    def _accumulate(self, accum: Dict[str, object], grads: Dict[str, object]) -> None:
        for key in ("W_window", "embeddings"):
            for i in range(len(accum[key])):
                for j in range(len(accum[key][i])):
                    accum[key][i][j] += grads[key][i][j]
        for key in ("b_window", "W_out"):
            for i in range(len(accum[key])):
                accum[key][i] += grads[key][i]
        accum["b_out"] += grads["b_out"]
        for i in range(len(self.gate_w)):
            accum["gate_w"][i] += grads["gate_w"][i]
        accum["gate_b"] += grads["gate_b"]
        for i in range(2):
            for j in range(2):
                accum["transitions"][i][j] += grads["transitions"][i][j]

    def _apply_gradients(self, grads: Dict[str, object], cfg: StudentTrainingConfig, batch_size: int) -> None:
        scale = cfg.lr / max(1, batch_size)
        for c in range(len(self.embeddings)):
            for d in range(self.emb_dim):
                self.embeddings[c][d] -= scale * (grads["embeddings"][c][d] + cfg.reg * self.embeddings[c][d])
        for j in range(self.hidden_dim):
            for k in range(self.window_dim):
                self.W_window[j][k] -= scale * (grads["W_window"][j][k] + cfg.reg * self.W_window[j][k])
            self.b_window[j] -= scale * grads["b_window"][j]
            self.W_out[j] -= scale * (grads["W_out"][j] + cfg.reg * self.W_out[j])
        self.b_out -= scale * grads["b_out"]
        for k in range(len(self.gate_w)):
            self.gate_w[k] -= scale * (grads["gate_w"][k] + cfg.reg * self.gate_w[k])
        self.gate_b -= scale * grads["gate_b"]
        for i in range(2):
            for j in range(2):
                self.transitions[i][j] -= cfg.crf_lr * (grads["transitions"][i][j] + cfg.reg * self.transitions[i][j])

    def _capture_state(self) -> Dict[str, object]:
        return {
            "embeddings": [row[:] for row in self.embeddings],
            "W_window": [row[:] for row in self.W_window],
            "b_window": self.b_window[:],
            "W_out": self.W_out[:],
            "b_out": self.b_out,
            "gate_w": self.gate_w[:],
            "gate_b": self.gate_b,
            "transitions": [row[:] for row in self.transitions],
        }

    def _restore_state(self, state: Dict[str, object]) -> None:
        self.embeddings = [row[:] for row in state["embeddings"]]
        self.W_window = [row[:] for row in state["W_window"]]
        self.b_window = state["b_window"][:]
        self.W_out = state["W_out"][:]
        self.b_out = float(state["b_out"])
        self.gate_w = state["gate_w"][:]
        self.gate_b = float(state["gate_b"])
        self.transitions = [row[:] for row in state["transitions"]]

    def export_state(self) -> Dict[str, object]:
        return self._capture_state()

    def load_state(self, state: Dict[str, object]) -> None:
        self._restore_state(state)

    # ------------------------------------------------------------------
    # Evaluation and inference
    # ------------------------------------------------------------------
    def evaluate(self, sequences: Sequence[BoundarySequence]) -> Tuple[float, float]:
        total_loss = 0.0
        total_tp = total_fp = total_fn = 0
        for seq in sequences:
            logits, _ = self._forward_sequence(seq)
            loss, _, _, marginals = self._crf_loss(logits, seq.labels)
            total_loss += loss
            preds = self._viterbi(logits)
            tp, fp, fn = self._boundary_confusion(preds, seq.labels)
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
                    score = dp[i - 1][prev] + trans[prev][state]
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
        if len(text) <= 1:
            return np.zeros(0, dtype=float)
        seq = self.build_sequences([text], [[text]])[0]
        logits, _ = self._forward_sequence(seq)
        _, _, _, marginals = self._crf_loss(logits, seq.labels)
        probs = [m[1] for m in marginals]
        return np.array(probs, dtype=float)

    def decode(self, text: str) -> List[str]:
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

