from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Callable
from .beliefs import GaussianBelief
from .preferences import QuadraticPreference
from .efe import EFEEngine, EFEConfig
from .world_model import ActionSpace
from .planning import CEMPlanner, SimpleTreePlanner, RolloutResult

@dataclass
class AgentConfig:
    horizon: int = 3
    planner: str = "cem"
    cem_K: int = 48
    cem_elite_frac: float = 0.25
    cem_iters: int = 2
    tree_branch_limit: int = 4
    efe_epistemic_w0: float = 1.0
    efe_epistemic_w1: float = 0.6
    obs_sigma: float = 0.6

class ActiveInferenceAgent:
    def __init__(self, dim: int, action_space: ActionSpace, goal_vec: np.ndarray,
                 step_fn: Callable, ctx_provider: Callable, r3_provider: Callable,
                 cfg: AgentConfig | None = None):
        self.dim = dim
        self.belief = GaussianBelief(mu=np.zeros(dim), Sigma=np.eye(dim))
        self.action_space = action_space
        self.pref = QuadraticPreference(o_star=goal_vec, W=np.eye(dim))
        self.cfg = cfg or AgentConfig()
        efe_cfg = EFEConfig(epistemic_weight_base=self.cfg.efe_epistemic_w0,
                            epistemic_weight_gain=self.cfg.efe_epistemic_w1)
        self.efe = EFEEngine(self.pref, cfg=efe_cfg)
        if self.cfg.planner == "cem":
            self.planner = CEMPlanner(self.efe, action_space, horizon=self.cfg.horizon,
                                      K=self.cfg.cem_K, elite_frac=self.cfg.cem_elite_frac,
                                      iters=self.cfg.cem_iters)
        else:
            self.planner = SimpleTreePlanner(self.efe, action_space, horizon=self.cfg.horizon, branch_limit=self.cfg.tree_branch_limit)
        self.step_fn = step_fn
        self.ctx_provider = ctx_provider
        self.r3_provider = r3_provider
        self.obs_sigma = self.cfg.obs_sigma

    def plan(self) -> RolloutResult:
        return self.planner.plan(self.belief, self.step_fn, self.ctx_provider, self.r3_provider)

    def act_and_update(self, action: str) -> dict:
        ctx = self.ctx_provider([action])
        b_next, o_mu, o_Sigma = self.step_fn(self.belief, action, ctx)
        post = self.belief.merge_with_observation(o_mu, o_Sigma)
        self.belief = b_next
        return {"obs_mu_norm": float(np.linalg.norm(o_mu)),
                "prior_H": float(self.belief.entropy()),
                "post_H": float(post.entropy())}
