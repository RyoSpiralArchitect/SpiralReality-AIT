from __future__ import annotations
from dataclasses import dataclass

from ..np_compat import np

@dataclass
class GaussianBelief:
    mu: np.ndarray
    Sigma: np.ndarray

    def copy(self) -> "GaussianBelief":
        return GaussianBelief(self.mu.copy(), self.Sigma.copy())

    @property
    def dim(self) -> int:
        return int(self.mu.shape[0])

    def entropy(self) -> float:
        sign, logdet = np.linalg.slogdet(self.Sigma + 1e-12*np.eye(self.dim))
        return 0.5*(self.dim*(1.0 + np.log(2.0*np.pi)) + logdet)

    def merge_with_observation(self, obs_mu: np.ndarray, obs_Sigma: np.ndarray) -> "GaussianBelief":
        H = np.eye(self.dim)
        mu, Sigma = self.mu, self.Sigma
        S = H @ Sigma @ H.T + obs_Sigma
        K = Sigma @ H.T @ np.linalg.inv(S)
        mu_post = mu + K @ (obs_mu - H @ mu)
        Sigma_post = (np.eye(self.dim) - K @ H) @ Sigma
        return GaussianBelief(mu_post, Sigma_post)
