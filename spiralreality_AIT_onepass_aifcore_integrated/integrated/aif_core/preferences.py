from __future__ import annotations
from dataclasses import dataclass

from ..np_compat import np

@dataclass
class QuadraticPreference:
    o_star: np.ndarray
    W: np.ndarray

    def log_prob(self, o_mu: np.ndarray, o_Sigma: np.ndarray | None = None) -> float:
        def _to_list(arr):
            if isinstance(arr, np.ndarray):
                if hasattr(arr, "to_list"):
                    return arr.to_list()
                return arr.tolist()
            return list(arr)

        diff = _to_list(o_mu - self.o_star)
        W = _to_list(self.W)
        term = 0.0
        for i, di in enumerate(diff):
            for j, dj in enumerate(diff):
                term += di * W[i][j] * dj
        if o_Sigma is not None:
            Sigma = _to_list(self.W @ o_Sigma)
            trace = 0.0
            for idx, row in enumerate(Sigma):
                trace += row[idx]
            term += trace
        return -0.5 * term
