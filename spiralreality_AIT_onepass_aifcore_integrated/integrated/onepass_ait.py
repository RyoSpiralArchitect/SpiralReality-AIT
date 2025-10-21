from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np, math, unicodedata

def seeded_vector(name: str, dim: int=64) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    return rng.uniform(-1.0, 1.0, size=(dim,))

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-8)

def build_phase_basis(dim: int=64):
    def plane(seed):
        a = seeded_vector(seed+"_a", dim); b = seeded_vector(seed+"_b", dim)
        e1 = unit(a)
        b = b - np.dot(b, e1) * e1
        e2 = unit(b)
        return np.stack([e1, e2], axis=0)
    return {"AB": plane("AB"), "BC": plane("BC"), "CA": plane("CA")}

PHASE_BASIS = build_phase_basis(64)

def token_phase_triplet(tok: str) -> tuple[float,float,float]:
    emb = seeded_vector("tok::"+tok, 64)
    def phase(E):
        x, y = float(np.dot(E[0], emb)), float(np.dot(E[1], emb))
        return math.atan2(y, x)
    return (phase(PHASE_BASIS["AB"]), phase(PHASE_BASIS["BC"]), phase(PHASE_BASIS["CA"]))

def is_space(ch: str) -> bool: return ch.isspace()
def is_punct(ch: str) -> bool: return unicodedata.category(ch).startswith("P")
def is_latin(ch: str) -> bool: return "LATIN" in unicodedata.name(ch, "")
def is_kana(ch: str) -> bool: return "KATAKANA" in unicodedata.name(ch, "") or "HIRAGANA" in unicodedata.name(ch, "")
def is_cjk(ch: str) -> bool: return "CJK" in unicodedata.name(ch, "") or "IDEOGRAPH" in unicodedata.name(ch, "")

@dataclass
class StudentTrainingConfig:
    lr: float = 0.15
    epochs: int = 320
    reg: float = 1e-3
    batch_size: int = 4096
    validation_split: float = 0.12
    shuffle: bool = True
    early_stopping_patience: int = 12


class BoundaryDataset:
    """Pre-computed boundary classification dataset for the Student head."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)

    def __len__(self) -> int:
        return self.y.shape[0]

    def split(self, val_frac: float, rng: np.random.Generator) -> tuple["BoundaryDataset", Optional["BoundaryDataset"]]:
        if val_frac <= 0.0 or len(self) < 2:
            return self, None
        idx = rng.permutation(len(self))
        cut = max(1, int(len(self) * (1.0 - val_frac)))
        train_idx, val_idx = idx[:cut], idx[cut:]
        if len(val_idx) == 0:
            return BoundaryDataset(self.X[train_idx], self.y[train_idx]), None
        return BoundaryDataset(self.X[train_idx], self.y[train_idx]), BoundaryDataset(self.X[val_idx], self.y[val_idx])

    def iter_batches(self, batch_size: int):
        if batch_size <= 0:
            batch_size = len(self)
        for start in range(0, len(self), batch_size):
            end = min(start + batch_size, len(self))
            yield self.X[start:end], self.y[start:end]


class BoundaryStudent:
    def __init__(self):
        self.W=None; self.b=0.0; self.mu=None; self.std=None

    def _feat(self, left: str, right: str) -> np.ndarray:
        def cls(ch):
            if ch == " ": return 0
            if is_latin(ch): return 1
            if is_cjk(ch): return 2
            if is_kana(ch): return 3
            if is_punct(ch): return 4
            return 5
        f = np.zeros(18, dtype=float)
        L = cls(left); R = cls(right)
        f[L] += 1.0; f[6+R] += 1.0
        f[12] = 1.0 if L==R else 0.0
        f[13] = 1.0 if left==" " or right==" " else 0.0
        f[14] = 1.0 if is_punct(left) or is_punct(right) else 0.0
        f[15] = 1.0 if (left in "'-" or right in "'-") and (is_latin(left) or is_latin(right)) else 0.0
        f[16] = 1.0 if (is_latin(left) and is_latin(right)) else 0.0
        f[17] = 1.0 if ((is_cjk(left) or is_kana(left)) and (is_cjk(right) or is_kana(right))) else 0.0
        return f

    def build_dataset(self, texts: Iterable[str], segments: Iterable[Iterable[str]]) -> BoundaryDataset:
        X=[]; y=[]
        for text, seg in zip(texts, segments):
            idx=0; cuts=set()
            for tok in seg:
                idx += len(tok); cuts.add(idx)
            for i in range(len(text)-1):
                left, right = text[i], text[i+1]
                f = self._feat(left, right)
                X.append(f); y.append(1.0 if (i+1) in cuts else 0.0)
        return BoundaryDataset(np.array(X, dtype=float), np.array(y, dtype=float))

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def _bce_loss(self, logits: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(np.logaddexp(0.0, logits) - y * logits))

    def train(self, texts: list[str], segments: list[list[str]], cfg: StudentTrainingConfig | None = None,
              rng: Optional[np.random.Generator] = None) -> dict:
        cfg = cfg or StudentTrainingConfig()
        rng = rng or np.random.default_rng(0)
        dataset = self.build_dataset(texts, segments)
        train_ds, val_ds = dataset.split(cfg.validation_split, rng)
        X_train = train_ds.X
        y_train = train_ds.y
        mu = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std = np.maximum(std, 1e-6)
        X_train = (X_train - mu) / std
        if val_ds is not None:
            X_val = (val_ds.X - mu) / std
            y_val = val_ds.y
        else:
            X_val = None; y_val = None
        N, D = X_train.shape
        W = np.zeros(D); b = 0.0
        best = {"loss": float("inf"), "W": W.copy(), "b": b, "acc": None}
        patience = 0
        indices = np.arange(N)
        stop_training = False
        history = []
        for ep in range(cfg.epochs):
            if cfg.shuffle:
                rng.shuffle(indices)
            else:
                indices = np.arange(N)
            for start in range(0, N, max(1, cfg.batch_size)):
                end = min(start + max(1, cfg.batch_size), N)
                batch_idx = indices[start:end]
                xb = X_train[batch_idx]
                yb = y_train[batch_idx]
                zb = xb @ W + b
                pb = self._sigmoid(zb)
                gradW = (xb.T @ (pb - yb)) / yb.shape[0] + cfg.reg * W
                gradb = float(np.mean(pb - yb))
                W -= cfg.lr * gradW
                b -= cfg.lr * gradb
            train_logits = X_train @ W + b
            train_loss = self._bce_loss(train_logits, y_train) + 0.5 * cfg.reg * float(np.dot(W, W))
            metrics = {"epoch": ep + 1, "train_loss": train_loss}
            if X_val is not None:
                val_logits = X_val @ W + b
                val_loss = self._bce_loss(val_logits, y_val) + 0.5 * cfg.reg * float(np.dot(W, W))
                preds = (self._sigmoid(val_logits) >= 0.5).astype(float)
                acc = float(np.mean(preds == y_val))
                metrics.update({"val_loss": val_loss, "val_acc": acc})
                if val_loss + 1e-6 < best["loss"]:
                    best = {"loss": val_loss, "W": W.copy(), "b": b, "acc": acc}
                    patience = 0
                else:
                    patience += 1
                    if cfg.early_stopping_patience and patience >= cfg.early_stopping_patience:
                        stop_training = True
            else:
                if train_loss + 1e-6 < best["loss"]:
                    best = {"loss": train_loss, "W": W.copy(), "b": b, "acc": None}
            history.append(metrics)
            if stop_training:
                break
        self.W = best["W"]
        self.b = best["b"]
        self.mu = mu
        self.std = std
        summary = {"train_samples": int(N), "features": int(D), "best_loss": best["loss"],
                   "epochs_trained": len(history), "history": history}
        if X_val is not None:
            summary["val_size"] = int(len(y_val))
            if best["acc"] is not None:
                summary["best_val_acc"] = best["acc"]
        return summary

    def boundary_probs(self, text: str) -> np.ndarray:
        if len(text) <= 1: return np.zeros(0, dtype=float)
        if self.W is None or self.mu is None or self.std is None:
            raise RuntimeError("BoundaryStudent must be trained before calling boundary_probs")
        ps=[]
        for i in range(len(text)-1):
            left, right = text[i], text[i+1]
            x = (self._feat(left, right)-self.mu)/self.std
            z = float(x @ self.W + self.b)
            p = 1.0/(1.0 + math.exp(-z))
            ps.append(p)
        return np.array(ps, dtype=float)

class ToyTransformerAdapter:
    def __init__(self, d_model=64, n_layers=2, seed=2025):
        rng = np.random.default_rng(seed)
        self.d_model=d_model; self.n_layers=n_layers
        self.Wq=[rng.normal(scale=0.2, size=(d_model,d_model)) for _ in range(n_layers)]
        self.Wk=[rng.normal(scale=0.2, size=(d_model,d_model)) for _ in range(n_layers)]
        self.Wv=[rng.normal(scale=0.2, size=(d_model,d_model)) for _ in range(n_layers)]
        self.Wo=[rng.normal(scale=0.2, size=(d_model,d_model)) for _ in range(n_layers)]
        self.film_W=[rng.normal(scale=0.2, size=(2,2)) for _ in range(n_layers)]

    def forward(self, X: np.ndarray, gate_pos: np.ndarray) -> np.ndarray:
        H = X.copy()
        gm = float(np.mean(gate_pos)); gs = float(np.std(gate_pos))
        for L in range(self.n_layers):
            Q = H @ self.Wq[L]; K = H @ self.Wk[L]; V = H @ self.Wv[L]
            scores = (Q @ K.T) / math.sqrt(self.d_model)
            scores += gate_pos[None,:] * 0.5
            A = np.exp(scores - scores.max(axis=-1, keepdims=True)); A = A / (A.sum(axis=-1, keepdims=True)+1e-12)
            H2 = A @ V @ self.Wo[L]
            gamma_beta = self.film_W[L] @ np.array([gm, gs])
            gamma = 1.0 + float(gamma_beta[0]); beta = float(gamma_beta[1])
            H = gamma*H2 + beta
        return H

class OnePassAIT:
    def __init__(self, latent_dim=64, seed=4242):
        self.latent_dim=latent_dim
        self.rng = np.random.default_rng(seed)
        self.policies = ["ProbeMotivation","ProbeReliability","SeekEvidence","DecideNow"]
        self.policy_vecs = {p: seeded_vector(p, latent_dim) for p in self.policies}
        self.goal_vec = unit(self.rng.normal(size=latent_dim))
        self.encoder = ToyTransformerAdapter(d_model=latent_dim, n_layers=2, seed=seed)
        self.student = BoundaryStudent()
        self.beta_ewma = 0.2
        self.gate_a0, self.gate_a1 = 1.0, 1.2
        self.mu = np.zeros(latent_dim); self.Sigma = np.eye(latent_dim)
        self.R3_mix = 0.0; self.R2_time = 0.0
        self._phi_hist = {"AB": [], "BC": [], "CA": []}

    def train_student(self, texts: list[str], segments: list[list[str]],
                      cfg: StudentTrainingConfig | None = None,
                      rng: Optional[np.random.Generator] = None) -> dict:
        return self.student.train(texts, segments, cfg=cfg, rng=rng)

    def _char_embs(self, text: str) -> np.ndarray:
        return np.stack([seeded_vector("char::"+c, self.latent_dim) for c in text], axis=0)

    def _unwrap(self, last, now):
        d = now - last; d = d - 2*math.pi*round(d/(2*math.pi))
        return last + d, d

    def _phase_curvature(self, text: str) -> np.ndarray:
        N = len(text); curv = np.zeros(N, dtype=float)
        seq={"AB": [],"BC": [],"CA": []}
        for c in text:
            ph = token_phase_triplet(c)
            seq["AB"].append(ph[0]); seq["BC"].append(ph[1]); seq["CA"].append(ph[2])
        def unwrap(xs):
            if not xs: return []
            out=[xs[0]]
            for i in range(1,len(xs)):
                d = xs[i]-xs[i-1]; d = d - 2*math.pi*round(d/(2*math.pi)); out.append(out[-1]+d)
            return out
        uw={k: unwrap(seq[k]) for k in seq}
        for i in range(N):
            v=0.0
            for k in ["AB","BC","CA"]:
                h=uw[k]; 
                if len(h)>=3 and i>=2:
                    dd = h[i]-2*h[i-1]+h[i-2]; v += abs(dd)/3.0
            curv[i]=v
        med=np.median(curv); mad=np.median(np.abs(curv-med))+1e-6
        return (curv - med)/mad

    def encode(self, text: str):
        # Single pass over text to produce H, r2_local, ps, gate_pos
        chars=list(text)
        X = self._char_embs(text)
        ps = self.student.boundary_probs(text)  # len = N-1
        r2_local = self._phase_curvature(text)  # len = N
        g_base = np.zeros(len(chars))
        for i in range(len(chars)):
            left = ps[i-1] if i-1 >= 0 else 0.0
            right = ps[i] if i < len(ps) else 0.0
            g_base[i] = max(left, right)
        gate_pos = 0.6 * g_base + 0.4 * (1.0 / (1.0 + np.exp(-r2_local)))
        H = self.encoder.forward(X, gate_pos)  # (N,d)
        return {"H": H, "r2_local": r2_local, "ps": ps, "gate_pos": gate_pos}

    def policy_local_gates(self, H: np.ndarray, r2_local: np.ndarray, policy: str) -> np.ndarray:
        pv = unit(self.policy_vecs[policy])
        aligns = np.array([float(np.dot(unit(h), pv)) for h in H])
        alpha_loc = self.gate_a0 + self.gate_a1 * np.clip(r2_local, -5.0, 5.0)
        g = 1.0 / (1.0 + np.exp(-alpha_loc * aligns))
        return g

    def predict_next(self, mu: np.ndarray, Sigma: np.ndarray, policy: str, ctx_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        alpha, beta, gamma = 0.22, 0.12, 0.15
        mu_next = mu + alpha*unit(self.policy_vecs[policy]) + beta*self.goal_vec*(1.0/(1.0+np.linalg.norm(mu))) + gamma*unit(ctx_vec)
        Sigma_next = np.maximum(0.25, 0.9) * Sigma + 1e-6*np.eye(mu.shape[0])
        return mu_next, Sigma_next

    def update_global_phases(self, mu: np.ndarray):
        def proj(pair):
            E = PHASE_BASIS[pair]
            x, y = float(np.dot(E[0], mu)), float(np.dot(E[1], mu))
            return math.atan2(y, x)
        # R3 and R2 (time) update from latent μ
        for key in ["AB","BC","CA"]:
            p = proj(key)
            hist = self._phi_hist[key]
            if not hist: self._phi_hist[key]=[p]; continue
            new,_ = self._unwrap(hist[-1], p); hist.append(new); self._phi_hist[key]=hist[-6:]
        v={}
        for k in ["AB","BC","CA"]:
            h=self._phi_hist[k]; dv=0.0 if len(h)<2 else h[-1]-h[-2]
            sc = np.median(np.abs(np.diff(h)-np.median(np.diff(h)))) + 1e-6 if len(h)>=3 else 1.0
            v[k]=dv/max(sc,1e-6)
        chi = math.copysign(1.0, v["AB"])*math.copysign(1.0, v["BC"])*math.copysign(1.0, v["CA"])
        chi = 1.0 if chi>0 else -1.0
        r = math.sqrt((v["AB"]**2+v["BC"]**2+v["CA"]**2)/3.0) * max(0.0, chi)
        self.R3_mix = (1-self.beta_ewma)*self.R3_mix + self.beta_ewma*min(5.0, r)
        acc=0.0
        for k in ["AB","BC","CA"]:
            h=self._phi_hist[k]; dd=0.0 if len(h)<3 else (h[-1]-2*h[-2]+h[-3])
            acc += abs(dd)
        curv=acc/3.0
        self.R2_time = (1-self.beta_ewma)*self.R2_time + self.beta_ewma*min(5.0, curv)
