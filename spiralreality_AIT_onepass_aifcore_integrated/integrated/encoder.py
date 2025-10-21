from __future__ import annotations

import math
from typing import Iterable, List, Optional

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
        self.last_gate_mask: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray, gate_pos: np.ndarray, gate_mask: Optional[np.ndarray] = None) -> np.ndarray:
        H = X.copy()
        gate_values = gate_pos.to_list() if hasattr(gate_pos, "to_list") else list(gate_pos)
        gate_mask_list: Optional[List[List[float]]] = None
        if gate_mask is None and gate_values:
            gate_mask_list = [[min(gi, gj) for gj in gate_values] for gi in gate_values]
        elif gate_mask is not None:
            if hasattr(gate_mask, "tolist"):
                gate_mask_list = gate_mask.tolist()
            elif hasattr(gate_mask, "to_list"):
                gate_mask_list = gate_mask.to_list()
            else:
                gate_mask_list = [list(row) for row in gate_mask]
        if gate_mask_list is not None:
            gate_mask = np.array(gate_mask_list, dtype=float)
        else:
            gate_mask = None
        if gate_values:
            gm = sum(gate_values) / len(gate_values)
            if len(gate_values) > 1:
                var = sum((g - gm) ** 2 for g in gate_values) / len(gate_values)
            else:
                var = 0.0
            gs = math.sqrt(var)
        else:
            gm = 0.0
            gs = 0.0
        attn_per_layer: List[np.ndarray] = []
        self.last_gate_mask = gate_mask
        for layer in range(self.n_layers):
            Q = H @ self.Wq[layer]
            K = H @ self.Wk[layer]
            V = H @ self.Wv[layer]
            scores = (Q @ K.T) / math.sqrt(self.d_model)
            softmax_rows = []
            for row_idx, (row, g) in enumerate(zip(scores, gate_values)):
                if hasattr(row, "tolist"):
                    row_list = row.tolist()
                elif hasattr(row, "to_list"):
                    row_list = row.to_list()
                else:
                    row_list = list(row)
                adjusted = []
                for col_idx, val in enumerate(row_list):
                    bias = 0.0
                    if gate_mask_list is not None:
                        if row_idx < len(gate_mask_list) and col_idx < len(gate_mask_list[row_idx]):
                            bias = gate_mask_list[row_idx][col_idx]
                    adjusted.append(val + 0.5 * g + 0.75 * bias)
                m = max(adjusted)
                exps = [math.exp(v - m) for v in adjusted]
                denom = sum(exps) + 1e-12
                softmax_rows.append([v / denom for v in exps])
            A = np.array(softmax_rows)
            attn_per_layer.append(A)
            H2 = A @ V @ self.Wo[layer]
            if gate_mask_list is not None:
                if gate_mask_list and hasattr(gate_mask_list[0], "__iter__") and not isinstance(gate_mask_list[0], (int, float)):
                    flat = [float(val) for row in gate_mask_list for val in row]
                else:
                    flat = [float(val) for val in gate_mask_list]
                bias_mean = sum(flat) / max(1, len(flat))
            else:
                bias_mean = 0.0
            gamma_beta = self.film_W[layer] @ np.array([gm + bias_mean, gs + abs(bias_mean)])
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

