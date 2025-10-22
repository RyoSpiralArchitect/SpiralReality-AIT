from __future__ import annotations
import math
from .np_compat import np
from .aif_core import GaussianBelief, ActionSpace, ActiveInferenceAgent, AgentConfig, QuadraticPreference, EFEEngine
from .onepass_ait import OnePassAIT
from .utils import unit, seeded_vector

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
        H = self.enc["H"]
        weights = g.to_list() if isinstance(g, np.ndarray) else list(g)
        H_rows = H.to_list() if isinstance(H, np.ndarray) else list(H)
        if not weights:
            return np.mean(H, axis=0)
        dim = len(H_rows[0])
        accum = [0.0 for _ in range(dim)]
        total = 0.0
        for w, row in zip(weights, H_rows):
            total += w
            for j in range(dim):
                accum[j] += w * row[j]
        if total <= 0.0:
            total = 1.0
        ctx = [val / total for val in accum]
        return np.array(ctx)

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
