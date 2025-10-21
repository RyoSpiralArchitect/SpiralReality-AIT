from __future__ import annotations
import numpy as np, math
from .aif_core import GaussianBelief, ActionSpace, ActiveInferenceAgent, AgentConfig, QuadraticPreference, EFEEngine
from .onepass_ait import OnePassAIT, unit, seeded_vector

class AITGWMBridge:
    """Wraps One‑Pass AIT as the dynamics used in AIF planning.
    - ctx_provider: policy‑specific local gates over H to produce a context vector.
    - step_fn: uses AIT.predict_next(mu,Sigma,policy,ctx) and returns (belief_next, o_mu, o_Sigma).
    - r3_provider: returns AIT's current R3_mix as a function of prefix length (approx. with internal EWMA).
    """
    def __init__(self, ait: OnePassAIT, enc: dict, obs_sigma: float = 0.6):
        self.ait = ait
        self.enc = enc  # {"H","r2_local","ps","gate_pos"}
        self.obs_sigma = obs_sigma
        self.dim = ait.latent_dim

    def ctx_from_prefix(self, prefix: list[str]) -> np.ndarray:
        if not prefix:
            return np.mean(self.enc["H"], axis=0)
        last = prefix[-1]
        g = self.ait.policy_local_gates(self.enc["H"], self.enc["r2_local"], last)
        ctx = np.sum(g[:,None]*self.enc["H"], axis=0) / (np.sum(g)+1e-6)
        return ctx

    def r3_from_prefix(self, prefix: list[str]) -> float:
        # Approximate schedule by tanh over prefix length, nudged by current ait.R3_mix
        t = max(1, len(prefix))
        base = math.tanh(0.35*t) * 2.0
        return 0.7*base + 0.3*float(self.ait.R3_mix)

    def step_fn(self, belief: GaussianBelief, action: str, ctx_vec: np.ndarray):
        # Predict next latent with AIT dynamics and synthesize observation
        mu_next, Sigma_next = self.ait.predict_next(belief.mu, belief.Sigma, action, ctx_vec)
        # Update AIT's internal R3 tracker with the predicted μ
        self.ait.update_global_phases(mu_next)
        o_mu = mu_next.copy()
        o_Sigma = (self.obs_sigma**2)*np.eye(self.dim)
        return GaussianBelief(mu_next, Sigma_next), o_mu, o_Sigma
