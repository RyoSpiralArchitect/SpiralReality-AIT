from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from .np_compat import HAS_NUMPY, np
from .utils import sigmoid


@dataclass
class DynamicsConfig:
    hidden_dim: int = 128
    lr: float = 0.01
    epochs: int = 8
    batch_size: int = 16


class LatentDynamicsModel:
    """Simple MLP-based dynamics approximator trained from teacher signals."""

    def __init__(self, latent_dim: int, context_dim: int, seed: int = 0, cfg: DynamicsConfig | None = None):
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.cfg = cfg or DynamicsConfig()
        rng = np.random.default_rng(seed)
        input_dim = latent_dim * 2 + context_dim
        self.W1 = rng.normal(scale=0.2, size=(self.cfg.hidden_dim, input_dim))
        self.b1 = np.zeros(self.cfg.hidden_dim)
        self.W_mu = rng.normal(scale=0.1, size=(latent_dim, self.cfg.hidden_dim))
        self.b_mu = np.zeros(latent_dim)
        self.W_sigma = rng.normal(scale=0.05, size=(latent_dim, self.cfg.hidden_dim))
        self.b_sigma = np.zeros(latent_dim)
        self.buffer: List[Tuple[np.ndarray, np.ndarray]] = []
        self.trained_steps = 0

    def _to_list(self, arr) -> list[float]:
        if hasattr(arr, "tolist"):
            return arr.tolist()
        if hasattr(arr, "to_list"):
            return arr.to_list()
        return list(arr)

    def _concat_inputs(self, *arrays) -> np.ndarray:
        if HAS_NUMPY:
            return np.concatenate(arrays)
        data: list[float] = []
        for arr in arrays:
            data.extend(self._to_list(arr))
        return np.array(data, dtype=float)

    def _outer(self, a, b) -> np.ndarray:
        if HAS_NUMPY:
            return np.outer(a, b)
        a_vals = self._to_list(a)
        b_vals = self._to_list(b)
        return np.array([[ai * bj for bj in b_vals] for ai in a_vals], dtype=float)

    def _zeros_like(self, arr) -> np.ndarray:
        if HAS_NUMPY:
            return np.zeros_like(arr)
        shape = getattr(arr, "shape", None)
        if shape is None:
            return np.array([0.0 for _ in self._to_list(arr)], dtype=float)
        if len(shape) == 2:
            return np.array([[0.0 for _ in range(shape[1])] for _ in range(shape[0])], dtype=float)
        return np.array([0.0 for _ in range(shape[0])], dtype=float)

    def _tanh(self, arr) -> np.ndarray:
        if HAS_NUMPY:
            return np.tanh(arr)
        return np.array([math.tanh(v) for v in self._to_list(arr)], dtype=float)

    def _diag(self, values) -> np.ndarray:
        if HAS_NUMPY:
            return np.diag(values)
        vals = self._to_list(values)
        return np.array([[vals[i] if i == j else 0.0 for j in range(len(vals))] for i in range(len(vals))], dtype=float)

    def _forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = self._tanh(self.W1 @ inputs + self.b1)
        delta = self.W_mu @ h + self.b_mu
        sigma_logits = self.W_sigma @ h + self.b_sigma
        sigma_diag = np.array([sigmoid(v) + 1e-3 for v in sigma_logits])
        return delta, sigma_diag

    def predict(self, mu: np.ndarray, policy_vec: np.ndarray, ctx_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        inputs = self._concat_inputs(mu, policy_vec, ctx_vec)
        delta, sigma_diag = self._forward(inputs)
        mu_next = mu + delta
        Sigma_next = self._diag(sigma_diag)
        return mu_next, Sigma_next

    def record(self, mu: np.ndarray, policy_vec: np.ndarray, ctx_vec: np.ndarray, target_mu: np.ndarray) -> None:
        inputs = self._concat_inputs(mu, policy_vec, ctx_vec)
        self.buffer.append((inputs, target_mu - mu))
        if len(self.buffer) > 2048:
            self.buffer = self.buffer[-2048:]

    def train(self) -> None:
        if not self.buffer:
            return
        cfg = self.cfg
        rng = np.random.default_rng(self.trained_steps + 1)
        for _ in range(cfg.epochs):
            rng.shuffle(self.buffer)
            for start in range(0, len(self.buffer), cfg.batch_size):
                end = min(start + cfg.batch_size, len(self.buffer))
                batch = self.buffer[start:end]
                grad_W1 = self._zeros_like(self.W1)
                grad_b1 = self._zeros_like(self.b1)
                grad_W_mu = self._zeros_like(self.W_mu)
                grad_b_mu = self._zeros_like(self.b_mu)
                grad_W_sigma = self._zeros_like(self.W_sigma)
                grad_b_sigma = self._zeros_like(self.b_sigma)
                batch_size = max(1, len(batch))
                for inputs, target_delta in batch:
                    h = self._tanh(self.W1 @ inputs + self.b1)
                    delta = self.W_mu @ h + self.b_mu
                    sigma_logits = self.W_sigma @ h + self.b_sigma
                    sigma_diag = np.array([sigmoid(v) + 1e-3 for v in sigma_logits])
                    diff = delta - target_delta
                    grad_mu = diff / batch_size
                    grad_W_mu += self._outer(grad_mu, h)
                    grad_b_mu += grad_mu
                    abs_target = np.array([abs(v) for v in self._to_list(target_delta)], dtype=float)
                    grad_sigma_logits = (sigma_diag - abs_target) / batch_size
                    grad_W_sigma += self._outer(grad_sigma_logits, h)
                    grad_b_sigma += grad_sigma_logits
                    grad_h_raw = self.W_mu.T @ grad_mu + self.W_sigma.T @ grad_sigma_logits
                    scale = np.array([1.0 - (v * v) for v in self._to_list(h)], dtype=float)
                    if HAS_NUMPY:
                        grad_h = grad_h_raw * scale
                    else:
                        grad_h = np.array(
                            [g * s for g, s in zip(self._to_list(grad_h_raw), scale)],
                            dtype=float,
                        )
                    grad_W1 += self._outer(grad_h, inputs)
                    grad_b1 += grad_h
                scale = cfg.lr / batch_size
                self.W1 -= scale * grad_W1
                self.b1 -= scale * grad_b1
                self.W_mu -= scale * grad_W_mu
                self.b_mu -= scale * grad_b_mu
                self.W_sigma -= scale * grad_W_sigma
                self.b_sigma -= scale * grad_b_sigma
        self.trained_steps += 1

    def is_ready(self) -> bool:
        return self.trained_steps > 0

    def export_state(self) -> dict:
        return {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W_mu": self.W_mu.tolist(),
            "b_mu": self.b_mu.tolist(),
            "W_sigma": self.W_sigma.tolist(),
            "b_sigma": self.b_sigma.tolist(),
            "trained_steps": self.trained_steps,
        }

    def load_state(self, state: dict) -> None:
        self.W1 = np.array(state["W1"], dtype=float)
        self.b1 = np.array(state["b1"], dtype=float)
        self.W_mu = np.array(state["W_mu"], dtype=float)
        self.b_mu = np.array(state["b_mu"], dtype=float)
        self.W_sigma = np.array(state["W_sigma"], dtype=float)
        self.b_sigma = np.array(state["b_sigma"], dtype=float)
        self.trained_steps = int(state.get("trained_steps", 0))

