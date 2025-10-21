from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .beliefs import GaussianBelief
from .preferences import QuadraticPreference

@dataclass
class EFEConfig:
    epistemic_weight_base: float = 1.0
    epistemic_weight_gain: float = 0.6
    epistemic_thermo: float = 1.0

class EFEEngine:
    def __init__(self, pref: QuadraticPreference, cfg: EFEConfig | None = None):
        self.pref = pref
        self.cfg = cfg or EFEConfig()

    def decompose_step(self, prior: GaussianBelief, o_mu, o_Sigma, post_if_observed: GaussianBelief, r3_mix: float=0.0) -> dict:
        risk = -self.pref.log_prob(o_mu, o_Sigma)
        H_prior = prior.entropy()
        H_post = post_if_observed.entropy()
        epistemic = max(0.0, (H_prior - H_post) * self.cfg.epistemic_thermo)
        w = self.cfg.epistemic_weight_base + self.cfg.epistemic_weight_gain * (1.0/(1.0 + np.exp(-r3_mix)))
        total = risk - w * epistemic
        return {"risk": float(risk), "epistemic": float(epistemic), "total": float(total), "w": float(w)}

    def total_over_rollout(self, terms: list[dict]) -> dict:
        keys = ["risk","epistemic","total"]
        agg = {k: float(sum(t[k] for t in terms)) for k in keys}
        agg["w_mean"] = float(np.mean([t["w"] for t in terms])) if terms else 0.0
        return agg
