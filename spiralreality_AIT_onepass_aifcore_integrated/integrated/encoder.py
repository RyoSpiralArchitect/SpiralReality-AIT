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
        arr = np.asarray(H, dtype=float)
        if arr.size == 0:
            return arr
        mean = arr.mean(axis=-1, keepdims=True)
        var = arr.var(axis=-1, keepdims=True)
        normalized = (arr - mean) / np.sqrt(var + eps)
        gamma_arr = np.asarray(gamma, dtype=float)
        beta_arr = np.asarray(beta, dtype=float)
        return normalized * gamma_arr + beta_arr

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
        return outer, gate_mask

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
        H = np.asarray(X, dtype=float)
        if H.size == 0:
            self.last_attn = []
            self.last_gate_mask = np.zeros((0, 0), dtype=float)
            return np.zeros((0, self.d_model), dtype=float)

        if H.ndim != 2 or H.shape[1] != self.d_model:
            raise ValueError(f"Expected X with shape (seq_len, {self.d_model}); got {H.shape}")

        seq_len = int(H.shape[0])

        gate_pos_arr = np.asarray(gate_pos, dtype=float).reshape(-1)
        if gate_pos_arr.size == 0:
            gate_pos_arr = np.zeros(seq_len, dtype=float)
        elif gate_pos_arr.size < seq_len:
            gate_pos_arr = np.pad(gate_pos_arr, (0, seq_len - gate_pos_arr.size), mode="edge")
        elif gate_pos_arr.size > seq_len:
            gate_pos_arr = gate_pos_arr[:seq_len]

        if gate_mask is None:
            gate_mask_arr = np.minimum.outer(gate_pos_arr, gate_pos_arr).astype(float, copy=False)
        else:
            gate_mask_arr = np.asarray(gate_mask, dtype=float)
            if gate_mask_arr.shape != (seq_len, seq_len):
                gate_mask_arr = np.minimum.outer(gate_pos_arr, gate_pos_arr).astype(float, copy=False)

        outer_mask = np.outer(gate_pos_arr, gate_pos_arr).astype(float, copy=False)

        gate_mean = float(gate_pos_arr.mean()) if gate_pos_arr.size else 0.0
        gate_std = float(gate_pos_arr.std()) if gate_pos_arr.size else 0.0
        mask_energy = float(gate_mask_arr.mean()) if gate_mask_arr.size else 0.0

        self.last_attn = []
        self.last_gate_mask = gate_mask_arr

        scale = 1.0 / math.sqrt(float(self.head_dim)) if self.head_dim > 0 else 1.0

        for layer in range(self.n_layers):
            norm_in = self._layer_norm(H, self.ln1_gamma[layer], self.ln1_beta[layer])
            Q = norm_in @ self.Wq[layer]
            K = norm_in @ self.Wk[layer]
            V = norm_in @ self.Wv[layer]

            Qh = Q.reshape(seq_len, self.n_heads, self.head_dim)
            Kh = K.reshape(seq_len, self.n_heads, self.head_dim)
            Vh = V.reshape(seq_len, self.n_heads, self.head_dim)

            bias = self.gate_bias[layer][0] * outer_mask + self.gate_bias[layer][1] * gate_mask_arr
            scores = np.einsum("lhd,mhd->hlm", Qh, Kh) * scale
            scores = scores + bias[None, :, :]

            scores = scores - scores.max(axis=-1, keepdims=True)
            weights = np.exp(scores)
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-12)

            # (heads, seq, seq) @ (heads, seq, head_dim) -> (heads, seq, head_dim)
            context = weights @ Vh.transpose(1, 0, 2)
            context = context.transpose(1, 0, 2).reshape(seq_len, self.d_model)

            H = H + (context @ self.Wo[layer])

            ff_in = self._layer_norm(H, self.ln2_gamma[layer], self.ln2_beta[layer])
            ff_hidden = np.tanh(ff_in @ self.Wff1[layer] + self.bff1[layer])
            ff_out = ff_hidden @ self.Wff2[layer] + self.bff2[layer]
            modulation = 1.0 + self.ff_gate[layer][0] * gate_mean + self.ff_gate[layer][1] * (gate_std + mask_energy)
            H = H + modulation * ff_out

            self.last_attn.append(weights.mean(axis=0).astype(float, copy=False))

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
