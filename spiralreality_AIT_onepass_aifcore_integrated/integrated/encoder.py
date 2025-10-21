from __future__ import annotations

import math
from typing import Iterable, List

from .np_compat import np


class ToyTransformerAdapter:
    def __init__(self, d_model: int = 64, n_layers: int = 2, seed: int = 2025):
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.n_layers = n_layers
        self.Wq = [rng.normal(scale=0.2, size=(d_model, d_model)) for _ in range(n_layers)]
        self.Wk = [rng.normal(scale=0.2, size=(d_model, d_model)) for _ in range(n_layers)]
        self.Wv = [rng.normal(scale=0.2, size=(d_model, d_model)) for _ in range(n_layers)]
        self.Wo = [rng.normal(scale=0.2, size=(d_model, d_model)) for _ in range(n_layers)]
        self.film_W = [rng.normal(scale=0.2, size=(2, 2)) for _ in range(n_layers)]
        self.last_attn: List[np.ndarray] = []

    def forward(self, X: np.ndarray, gate_pos: np.ndarray) -> np.ndarray:
        H = X.copy()
        gate_values = gate_pos.to_list() if hasattr(gate_pos, "to_list") else list(gate_pos)
        gm = float(np.mean(gate_values)) if gate_values else 0.0
        gs = float(np.std(gate_values)) if gate_values else 0.0
        attn_per_layer: List[np.ndarray] = []
        for layer in range(self.n_layers):
            Q = H @ self.Wq[layer]
            K = H @ self.Wk[layer]
            V = H @ self.Wv[layer]
            scores = (Q @ K.T) / math.sqrt(self.d_model)
            softmax_rows = []
            for row, g in zip(scores, gate_values):
                if hasattr(row, "tolist"):
                    row_list = row.tolist()
                elif hasattr(row, "to_list"):
                    row_list = row.to_list()
                else:
                    row_list = list(row)
                adjusted = [val + 0.5 * g for val in row_list]
                m = max(adjusted)
                exps = [math.exp(v - m) for v in adjusted]
                denom = sum(exps) + 1e-12
                softmax_rows.append([v / denom for v in exps])
            A = np.array(softmax_rows)
            attn_per_layer.append(A)
            H2 = A @ V @ self.Wo[layer]
            gamma_beta = self.film_W[layer] @ np.array([gm, gs])
            gamma = 1.0 + float(gamma_beta[0])
            beta = float(gamma_beta[1])
            H = gamma * H2 + beta
        self.last_attn = attn_per_layer
        return H

    def tune_from_boundary(self, base_gate: Iterable[float], targets: Iterable[float], lr: float = 1e-3) -> None:
        base_vals = list(base_gate)
        target_vals = list(targets)
        if not base_vals or len(base_vals) != len(target_vals):
            return
        gm = float(np.mean(base_vals))
        gs = float(np.std(base_vals) + 1e-6)
        diffs = [b - t for b, t in zip(base_vals, target_vals)]
        err_mean = sum(diffs) / len(diffs)
        centered_targets = [t - (sum(target_vals) / len(target_vals)) for t in target_vals]
        err_std = 0.0
        for diff, centered in zip(diffs, centered_targets):
            err_std += diff * centered
        err_std /= max(1, len(diffs))
        for layer in range(self.n_layers):
            mat = self.film_W[layer]
            mat[0][0] -= lr * err_mean * gm
            mat[0][1] -= lr * err_mean * gs
            mat[1][0] -= lr * err_std * gm
            mat[1][1] -= lr * err_std * gs
            self.film_W[layer] = mat

    def export_state(self) -> dict:
        return {
            "Wq": [mat.tolist() for mat in self.Wq],
            "Wk": [mat.tolist() for mat in self.Wk],
            "Wv": [mat.tolist() for mat in self.Wv],
            "Wo": [mat.tolist() for mat in self.Wo],
            "film_W": [mat.tolist() for mat in self.film_W],
        }

    def load_state(self, state: dict) -> None:
        self.Wq = [np.array(mat, dtype=float) for mat in state["Wq"]]
        self.Wk = [np.array(mat, dtype=float) for mat in state["Wk"]]
        self.Wv = [np.array(mat, dtype=float) for mat in state["Wv"]]
        self.Wo = [np.array(mat, dtype=float) for mat in state["Wo"]]
        self.film_W = [np.array(mat, dtype=float) for mat in state["film_W"]]
        self.last_attn = []

