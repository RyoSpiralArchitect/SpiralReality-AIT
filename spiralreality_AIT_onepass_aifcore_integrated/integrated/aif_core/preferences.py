from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class QuadraticPreference:
    o_star: np.ndarray
    W: np.ndarray

    def log_prob(self, o_mu: np.ndarray, o_Sigma: np.ndarray | None = None) -> float:
        d = o_mu - self.o_star
        term = float(d.T @ self.W @ d)
        if o_Sigma is not None:
            term += float(np.trace(self.W @ o_Sigma))
        return -0.5 * term
