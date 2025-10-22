from __future__ import annotations

import math
from typing import Iterable, List, Optional

from .np_compat import np


class SpectralTransformerAdapter:
    """A multi-head NumPy transformer tuned for streaming gate signals."""

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_multiplier: float = 4.0,
        seed: int = 2025,
    ) -> None:
        if n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        ff_dim = max(self.head_dim * n_heads, int(d_model * ff_multiplier))
        self.ff_dim = ff_dim
        scale = 1.0 / math.sqrt(d_model)
        self.Wq = [rng.normal(scale=scale, size=(d_model, d_model)) for _ in range(n_layers)]
        self.Wk = [rng.normal(scale=scale, size=(d_model, d_model)) for _ in range(n_layers)]
        self.Wv = [rng.normal(scale=scale, size=(d_model, d_model)) for _ in range(n_layers)]
        self.Wo = [rng.normal(scale=scale, size=(d_model, d_model)) for _ in range(n_layers)]
        self.Wff1 = [rng.normal(scale=scale, size=(d_model, ff_dim)) for _ in range(n_layers)]
        self.bff1 = [rng.normal(scale=scale, size=(ff_dim,)) for _ in range(n_layers)]
        self.Wff2 = [rng.normal(scale=scale, size=(ff_dim, d_model)) for _ in range(n_layers)]
        self.bff2 = [rng.normal(scale=scale, size=(d_model,)) for _ in range(n_layers)]
        self.ln1_gamma = [np.ones(d_model) for _ in range(n_layers)]
        self.ln1_beta = [np.zeros(d_model) for _ in range(n_layers)]
        self.ln2_gamma = [np.ones(d_model) for _ in range(n_layers)]
        self.ln2_beta = [np.zeros(d_model) for _ in range(n_layers)]
        self.gate_bias = [rng.normal(scale=0.1, size=(2,)) for _ in range(n_layers)]
        self.ff_gate = [rng.normal(scale=0.05, size=(2,)) for _ in range(n_layers)]
        self.last_attn: List[np.ndarray] = []
        self.last_gate_mask: Optional[np.ndarray] = None
        self.device = "cpu"
        self.backend = "spectral-numpy"

    # ------------------------------------------------------------------
    # Core layers
    # ------------------------------------------------------------------
    def _layer_norm(self, H: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        eps = 1e-5
        if hasattr(H, "to_list"):
            rows = H.to_list()
        elif hasattr(H, "tolist"):
            rows = H.tolist()
        else:
            rows = [list(row) for row in H]
        gamma_list = gamma.to_list() if hasattr(gamma, "to_list") else (
            gamma.tolist() if hasattr(gamma, "tolist") else list(gamma)
        )
        beta_list = beta.to_list() if hasattr(beta, "to_list") else (
            beta.tolist() if hasattr(beta, "tolist") else list(beta)
        )
        norm_rows = []
        for row in rows:
            row_vals = row.to_list() if hasattr(row, "to_list") else (
                row.tolist() if hasattr(row, "tolist") else list(row)
            )
            if not row_vals:
                norm_rows.append([0.0 for _ in gamma_list])
                continue
            mean_val = sum(float(v) for v in row_vals) / len(row_vals)
            var_val = sum((float(v) - mean_val) ** 2 for v in row_vals) / len(row_vals)
            denom = math.sqrt(var_val + eps)
            normalized = [(float(v) - mean_val) / denom for v in row_vals]
            modulated = []
            for idx, val in enumerate(normalized):
                gamma_val = gamma_list[idx] if idx < len(gamma_list) else gamma_list[-1]
                beta_val = beta_list[idx] if idx < len(beta_list) else beta_list[-1]
                modulated.append(gamma_val * val + beta_val)
            norm_rows.append(modulated)
        return np.array(norm_rows, dtype=float)

    def _prepare_mask(
        self, gate_pos: np.ndarray, gate_mask: Optional[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        gate_vals = gate_pos.tolist() if hasattr(gate_pos, "tolist") else list(gate_pos)
        if gate_mask is None:
            if gate_vals:
                gate_mask = np.array(
                    [[min(gi, gj) for gj in gate_vals] for gi in gate_vals], dtype=float
                )
            else:
                gate_mask = np.zeros((len(gate_vals), len(gate_vals)))
        elif not isinstance(gate_mask, np.ndarray):
            if hasattr(gate_mask, "tolist"):
                gate_mask = np.array(gate_mask.tolist(), dtype=float)
            else:
                gate_mask = np.array([[float(v) for v in row] for row in gate_mask], dtype=float)
        outer = np.zeros_like(gate_mask, dtype=float)
        if gate_vals:
            outer = np.array(
                [[float(gi) * float(gj) for gj in gate_vals] for gi in gate_vals],
                dtype=float,
            )
        return outer, np.array(gate_mask, dtype=float, copy=False)

    def _to_rows(self, matrix: np.ndarray) -> List[List[float]]:
        if hasattr(matrix, "to_list"):
            raw = matrix.to_list()
        elif hasattr(matrix, "tolist"):
            raw = matrix.tolist()
        else:
            raw = [list(row) for row in matrix]
        return [
            [float(val) for val in (row if isinstance(row, list) else [row])]
            for row in raw
        ]

    def _split_heads(self, matrix: np.ndarray) -> List[List[List[float]]]:
        rows = self._to_rows(matrix)
        split: List[List[List[float]]] = []
        for row in rows:
            head_chunks: List[List[float]] = []
            for h in range(self.n_heads):
                start = h * self.head_dim
                end = start + self.head_dim
                chunk = [float(v) for v in row[start:end]]
                if len(chunk) < self.head_dim:
                    chunk.extend([0.0] * (self.head_dim - len(chunk)))
                head_chunks.append(chunk)
            split.append(head_chunks)
        return split

    def device_inventory(self) -> List[str]:
        return [self.device]

    def forward(self, X: np.ndarray, gate_pos: np.ndarray, gate_mask: Optional[np.ndarray] = None) -> np.ndarray:
        if hasattr(X, "size"):
            total = X.size  # type: ignore[attr-defined]
        elif hasattr(X, "to_list"):
            total = len(X.to_list())
        else:
            total = len(X) if hasattr(X, "__len__") else 0
        if total == 0:
            self.last_attn = []
            self.last_gate_mask = np.zeros((0, 0))
            return np.zeros((0, self.d_model), dtype=float)
        H = np.array(X, dtype=float)
        gate_vals = gate_pos.tolist() if hasattr(gate_pos, "tolist") else list(gate_pos)
        gate_pos_arr = (
            np.array(gate_vals, dtype=float)
            if gate_vals
            else np.zeros((0,), dtype=float)
        )
        outer_mask, gate_mask_arr = self._prepare_mask(gate_pos_arr, gate_mask)
        if gate_vals:
            gate_mean = sum(gate_vals) / len(gate_vals)
            gate_var = sum((g - gate_mean) ** 2 for g in gate_vals) / len(gate_vals)
            gate_std = math.sqrt(max(gate_var, 0.0))
        else:
            gate_mean = 0.0
            gate_std = 0.0
        mask_energy = 0.0
        gate_mask_arr = np.array(gate_mask_arr, dtype=float, copy=False)
        if gate_mask_arr.size:
            mask_energy = float(np.mean(gate_mask_arr))
        seq_len = H.shape[0] if hasattr(H, "shape") and H.shape else len(getattr(H, "to_list", lambda: [])())
        self.last_attn = []
        self.last_gate_mask = gate_mask_arr
        for layer in range(self.n_layers):
            norm_in = self._layer_norm(H, self.ln1_gamma[layer], self.ln1_beta[layer])
            Q = norm_in @ self.Wq[layer]
            K = norm_in @ self.Wk[layer]
            V = norm_in @ self.Wv[layer]
            Q_heads = self._split_heads(Q)
            K_heads = self._split_heads(K)
            V_heads = self._split_heads(V)
            outer_list = self._to_rows(outer_mask)
            mask_list = self._to_rows(gate_mask_arr)
            seq_len = min(seq_len, len(Q_heads))
            key_len = len(K_heads)
            attn_heads: List[List[List[float]]] = []
            for head_idx in range(self.n_heads):
                head_scores: List[List[float]] = []
                for i in range(seq_len):
                    row_scores: List[float] = []
                    for j in range(key_len):
                        dot = 0.0
                        q_head = Q_heads[i][head_idx] if head_idx < len(Q_heads[i]) else [0.0] * self.head_dim
                        k_head = K_heads[j][head_idx] if head_idx < len(K_heads[j]) else [0.0] * self.head_dim
                        for d in range(self.head_dim):
                            dot += q_head[d] * k_head[d]
                        outer_val = (
                            outer_list[i][j]
                            if i < len(outer_list) and j < len(outer_list[i])
                            else 0.0
                        )
                        mask_val = (
                            mask_list[i][j]
                            if i < len(mask_list) and j < len(mask_list[i])
                            else 0.0
                        )
                        bias_val = self.gate_bias[layer][0] * outer_val + self.gate_bias[layer][1] * mask_val
                        row_scores.append(dot / math.sqrt(self.head_dim) + bias_val)
                    head_scores.append(row_scores)
                head_weights: List[List[float]] = []
                for row in head_scores:
                    row_max = max(row)
                    exps = [math.exp(val - row_max) for val in row]
                    denom = sum(exps) + 1e-12
                    head_weights.append([val / denom for val in exps])
                attn_heads.append(head_weights)
            context_rows: List[List[float]] = []
            for i in range(seq_len):
                merged: List[float] = []
                for head_idx in range(self.n_heads):
                    weighted = [0.0 for _ in range(self.head_dim)]
                    for j in range(key_len):
                        weight = attn_heads[head_idx][i][j]
                        head_vals = V_heads[j][head_idx] if head_idx < len(V_heads[j]) else [0.0] * self.head_dim
                        for d in range(self.head_dim):
                            weighted[d] += weight * head_vals[d]
                    merged.extend(weighted)
                context_rows.append(merged)
            context = np.array(context_rows, dtype=float)
            attn_out = context @ self.Wo[layer]
            H = H + attn_out
            ff_in = self._layer_norm(H, self.ln2_gamma[layer], self.ln2_beta[layer])
            ff_hidden = np.tanh(ff_in @ self.Wff1[layer] + self.bff1[layer])
            ff_out = ff_hidden @ self.Wff2[layer] + self.bff2[layer]
            modulation = 1.0 + self.ff_gate[layer][0] * gate_mean + self.ff_gate[layer][1] * (gate_std + mask_energy)
            H = H + modulation * ff_out
            attn_mean = []
            for i in range(seq_len):
                row_vals = []
                for j in range(key_len):
                    row_vals.append(
                        sum(attn_heads[h][i][j] for h in range(self.n_heads)) / max(1, self.n_heads)
                    )
                attn_mean.append(row_vals)
            self.last_attn.append(np.array(attn_mean, dtype=float))
        return H

    # ------------------------------------------------------------------
    # Adaptation helpers
    # ------------------------------------------------------------------
    def tune_from_boundary(self, base_gate: Iterable[float], targets: Iterable[float], lr: float = 1e-3) -> None:
        base_vals = [float(v) for v in base_gate]
        target_vals = [float(v) for v in targets]
        if not base_vals or len(base_vals) != len(target_vals):
            return
        diffs = [b - t for b, t in zip(base_vals, target_vals)]
        err_mean = sum(diffs) / len(diffs)
        err_var = sum((d - err_mean) ** 2 for d in diffs) / len(diffs)
        err_std = math.sqrt(max(err_var, 0.0))
        for layer in range(self.n_layers):
            self.gate_bias[layer][0] -= lr * err_mean
            self.gate_bias[layer][1] -= lr * err_std
            self.ff_gate[layer][0] -= lr * err_mean
            self.ff_gate[layer][1] -= lr * err_std

    # ------------------------------------------------------------------
    # State I/O
    # ------------------------------------------------------------------
    def export_state(self) -> dict:
        return {
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "Wq": [mat.tolist() for mat in self.Wq],
            "Wk": [mat.tolist() for mat in self.Wk],
            "Wv": [mat.tolist() for mat in self.Wv],
            "Wo": [mat.tolist() for mat in self.Wo],
            "Wff1": [mat.tolist() for mat in self.Wff1],
            "bff1": [vec.tolist() for vec in self.bff1],
            "Wff2": [mat.tolist() for mat in self.Wff2],
            "bff2": [vec.tolist() for vec in self.bff2],
            "ln1_gamma": [vec.tolist() for vec in self.ln1_gamma],
            "ln1_beta": [vec.tolist() for vec in self.ln1_beta],
            "ln2_gamma": [vec.tolist() for vec in self.ln2_gamma],
            "ln2_beta": [vec.tolist() for vec in self.ln2_beta],
            "gate_bias": [vec.tolist() for vec in self.gate_bias],
            "ff_gate": [vec.tolist() for vec in self.ff_gate],
        }

    def load_state(self, state: dict) -> None:
        self.d_model = int(state.get("d_model", self.d_model))
        self.n_layers = int(state.get("n_layers", self.n_layers))
        self.n_heads = int(state.get("n_heads", self.n_heads))
        self.head_dim = self.d_model // max(1, self.n_heads)
        def to_array_list(key: str) -> List[np.ndarray]:
            return [np.array(mat, dtype=float) for mat in state[key]]

        self.Wq = to_array_list("Wq")
        self.Wk = to_array_list("Wk")
        self.Wv = to_array_list("Wv")
        self.Wo = to_array_list("Wo")
        self.Wff1 = to_array_list("Wff1")
        self.bff1 = [np.array(vec, dtype=float) for vec in state["bff1"]]
        self.Wff2 = to_array_list("Wff2")
        self.bff2 = [np.array(vec, dtype=float) for vec in state["bff2"]]
        self.ln1_gamma = [np.array(vec, dtype=float) for vec in state["ln1_gamma"]]
        self.ln1_beta = [np.array(vec, dtype=float) for vec in state["ln1_beta"]]
        self.ln2_gamma = [np.array(vec, dtype=float) for vec in state["ln2_gamma"]]
        self.ln2_beta = [np.array(vec, dtype=float) for vec in state["ln2_beta"]]
        self.gate_bias = [np.array(vec, dtype=float) for vec in state["gate_bias"]]
        self.ff_gate = [np.array(vec, dtype=float) for vec in state["ff_gate"]]
        self.last_attn = []
        self.last_gate_mask = None


# Backwards compatibility alias for previous imports
ToyTransformerAdapter = SpectralTransformerAdapter
