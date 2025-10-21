from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Callable
from .beliefs import GaussianBelief
from .efe import EFEEngine
from .world_model import ActionSpace

@dataclass
class RolloutResult:
    actions: list[str]
    terms: list[dict]
    aggregate: dict

class CEMPlanner:
    def __init__(self, efe: EFEEngine, action_space: ActionSpace, horizon: int=3, K: int=48, elite_frac: float=0.25, iters: int=2):
        self.efe = efe; self.action_space = action_space
        self.horizon = horizon; self.K = K; self.elite_frac = elite_frac; self.iters = iters

    def plan(self, belief: GaussianBelief,
             step_fn: Callable[[GaussianBelief, str, np.ndarray], tuple[GaussianBelief, np.ndarray, np.ndarray]],
             ctx_provider: Callable[[list[str]], np.ndarray],
             r3_provider: Callable[[list[str]], float]) -> RolloutResult:
        actions = self.action_space.all_actions()
        A = len(actions)
        probs = np.full((self.horizon, A), 1.0/A, dtype=float)
        rng = np.random.default_rng(42)

        def sample_seq():
            seq=[]
            for t in range(self.horizon):
                a_idx = rng.choice(A, p=probs[t])
                seq.append(actions[a_idx])
            return seq

        def eval_seq(seq):
            b = belief.copy()
            terms = []
            for t, a in enumerate(seq):
                ctx = ctx_provider(seq[:t+1])
                b_next, o_mu, o_Sigma = step_fn(b, a, ctx)
                b_post = b.merge_with_observation(o_mu, o_Sigma)
                r3 = r3_provider(seq[:t+1])
                term = self.efe.decompose_step(b, o_mu, o_Sigma, b_post, r3_mix=r3)
                terms.append(term)
                b = b_next
            agg = self.efe.total_over_rollout(terms)
            return agg["total"]

        for _ in range(self.iters):
            samples = [sample_seq() for _ in range(self.K)]
            scores = np.array([eval_seq(s) for s in samples], dtype=float)
            elite_idx = np.argsort(scores)[:max(1, int(self.K*self.elite_frac))]
            elite = [samples[i] for i in elite_idx]
            probs = np.full_like(probs, 1e-9, dtype=float)
            for seq in elite:
                for t, a in enumerate(seq):
                    probs[t, actions.index(a)] += 1.0
            probs = probs / (probs.sum(axis=1, keepdims=True)+1e-12)

        best_seq = [actions[int(np.argmax(probs[t]))] for t in range(self.horizon)]
        b = belief.copy(); terms=[]
        for t,a in enumerate(best_seq):
            ctx = ctx_provider(best_seq[:t+1])
            b_next, o_mu, o_Sigma = step_fn(b, a, ctx)
            b_post = b.merge_with_observation(o_mu, o_Sigma)
            r3 = r3_provider(best_seq[:t+1])
            term = self.efe.decompose_step(b, o_mu, o_Sigma, b_post, r3_mix=r3)
            terms.append(term)
            b = b_next
        return RolloutResult(best_seq, terms, self.efe.total_over_rollout(terms))

class SimpleTreePlanner:
    def __init__(self, efe: EFEEngine, action_space: ActionSpace, horizon: int=3, branch_limit: int=4):
        self.efe = efe; self.action_space = action_space
        self.horizon = horizon; self.branch_limit = branch_limit

    def plan(self, belief: GaussianBelief,
             step_fn, ctx_provider, r3_provider) -> RolloutResult:
        actions = self.action_space.all_actions()
        best = (None, float("+inf"), None)
        def dfs(prefix, b, d, acc):
            nonlocal best
            if d==self.horizon:
                agg = self.efe.total_over_rollout(acc)
                if agg["total"] < best[1]: best = (prefix.copy(), agg["total"], acc.copy())
                return
            for a in actions[:self.branch_limit]:
                ctx = ctx_provider(prefix+[a])
                b_next, o_mu, o_Sigma = step_fn(b, a, ctx)
                b_post = b.merge_with_observation(o_mu, o_Sigma)
                r3 = r3_provider(prefix+[a])
                term = self.efe.decompose_step(b, o_mu, o_Sigma, b_post, r3_mix=r3)
                dfs(prefix+[a], b_next, d+1, acc+[term])
        dfs([], belief.copy(), 0, [])
        return RolloutResult(best[0], best[2], self.efe.total_over_rollout(best[2]))
