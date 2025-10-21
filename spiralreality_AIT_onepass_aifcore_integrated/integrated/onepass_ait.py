from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional
import math, unicodedata

from .np_compat import np, HAS_NUMPY

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

    def __init__(self, X, y):
        if HAS_NUMPY:
            self.X = np.asarray(X, dtype=float)
            self.y = np.asarray(y, dtype=float)
        else:
            self.X = [list(map(float, row)) for row in X]
            self.y = [float(v) for v in y]

    def __len__(self) -> int:
        return len(self.y)

    def _slice(self, indices: list[int]):
        if HAS_NUMPY:
            return self.X[indices], self.y[indices]
        X_rows = [self.X[i][:] for i in indices]
        y_rows = [self.y[i] for i in indices]
        return X_rows, y_rows

    def split(self, val_frac: float, rng: np.random.Generator) -> tuple["BoundaryDataset", Optional["BoundaryDataset"]]:
        if val_frac <= 0.0 or len(self) < 2:
            return self, None
        perm = rng.permutation(len(self))
        if isinstance(perm, np.ndarray):
            if hasattr(perm, "to_list"):
                indices = [int(i) for i in perm.to_list()]
            else:
                indices = [int(i) for i in perm.tolist()]
        else:
            indices = [int(i) for i in perm]
        cut = max(1, int(len(self) * (1.0 - val_frac)))
        train_idx, val_idx = indices[:cut], indices[cut:]
        train_X, train_y = self._slice(train_idx)
        if not val_idx:
            return BoundaryDataset(train_X, train_y), None
        val_X, val_y = self._slice(val_idx)
        return BoundaryDataset(train_X, train_y), BoundaryDataset(val_X, val_y)

    def iter_batches(self, batch_size: int):
        if batch_size <= 0:
            batch_size = len(self)
        for start in range(0, len(self), batch_size):
            end = min(start + batch_size, len(self))
            if HAS_NUMPY:
                yield self.X[start:end], self.y[start:end]
            else:
                batch_X = [self.X[i] for i in range(start, end)]
                batch_y = [self.y[i] for i in range(start, end)]
                yield batch_X, batch_y


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
        if HAS_NUMPY:
            X_data = np.array(X, dtype=float)
            y_data = np.array(y, dtype=float)
        else:
            X_data = [list(map(float, row)) for row in X]
            y_data = [float(v) for v in y]
        return BoundaryDataset(X_data, y_data)

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
        if HAS_NUMPY:
            return self._train_with_numpy(train_ds, val_ds, cfg, rng)
        return self._train_pure_python(train_ds, val_ds, cfg, rng)

    def _train_with_numpy(self, train_ds: BoundaryDataset, val_ds: Optional[BoundaryDataset],
                          cfg: StudentTrainingConfig, rng: np.random.Generator) -> dict:
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
            X_val = None
            y_val = None
        N, D = X_train.shape
        def _rows(arr):
            if isinstance(arr, np.ndarray):
                if hasattr(arr, "to_list"):
                    return arr.to_list()
                return arr.tolist()
            return list(arr)
        X_rows = _rows(X_train)
        y_rows = _rows(y_train)
        W = np.zeros(D)
        b = 0.0
        best = {"loss": float("inf"), "W": W.copy(), "b": b, "acc": None}
        patience = 0
        base_indices = list(range(N))
        stop_training = False
        history = []
        batch_size = max(1, cfg.batch_size)
        for ep in range(cfg.epochs):
            if cfg.shuffle:
                rng.shuffle(base_indices)
            else:
                base_indices = list(range(N))
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch_idx = base_indices[start:end]
                xb = np.array([X_rows[i] for i in batch_idx])
                yb = np.array([y_rows[i] for i in batch_idx])
                zb = xb @ W + b
                pb = self._sigmoid(zb)
                gradW = (xb.T @ (pb - yb)) / yb.shape[0] + cfg.reg * W
                gradb = float(np.mean(pb - yb))
                W -= cfg.lr * gradW
                b -= cfg.lr * gradb
            train_logits = X_train @ W + b
            train_loss = self._bce_loss(train_logits, y_train) + 0.5 * cfg.reg * float(np.dot(W, W))
            metrics = {"epoch": ep + 1, "train_loss": float(train_loss)}
            if X_val is not None:
                val_logits = X_val @ W + b
                val_loss = self._bce_loss(val_logits, y_val) + 0.5 * cfg.reg * float(np.dot(W, W))
                preds = (self._sigmoid(val_logits) >= 0.5).astype(float)
                acc = float(np.mean(preds == y_val))
                metrics.update({"val_loss": float(val_loss), "val_acc": acc})
                if val_loss + 1e-6 < best["loss"]:
                    best = {"loss": float(val_loss), "W": W.copy(), "b": b, "acc": acc}
                    patience = 0
                else:
                    patience += 1
                    if cfg.early_stopping_patience and patience >= cfg.early_stopping_patience:
                        stop_training = True
            else:
                if train_loss + 1e-6 < best["loss"]:
                    best = {"loss": float(train_loss), "W": W.copy(), "b": b, "acc": None}
            history.append(metrics)
            if stop_training:
                break
        self.W = best["W"]
        self.b = best["b"]
        self.mu = mu
        self.std = std
        summary = {"train_samples": int(N), "features": int(D), "best_loss": float(best["loss"]),
                   "epochs_trained": len(history), "history": history}
        if X_val is not None and y_val is not None:
            summary["val_size"] = int(len(y_val))
            if best["acc"] is not None:
                summary["best_val_acc"] = float(best["acc"])
        return summary

    def _train_pure_python(self, train_ds: BoundaryDataset, val_ds: Optional[BoundaryDataset],
                           cfg: StudentTrainingConfig, rng: np.random.Generator) -> dict:
        X_train = train_ds.X
        y_train = train_ds.y
        N = len(X_train)
        if N == 0:
            raise RuntimeError("Training dataset is empty")
        D = len(X_train[0]) if X_train else 0
        mu = [0.0 for _ in range(D)]
        for row in X_train:
            for j, val in enumerate(row):
                mu[j] += val
        mu = [m / N for m in mu]
        var = [0.0 for _ in range(D)]
        for row in X_train:
            for j, val in enumerate(row):
                diff = val - mu[j]
                var[j] += diff * diff
        std = [max(math.sqrt(v / N), 1e-6) for v in var]
        for row in X_train:
            for j in range(D):
                row[j] = (row[j] - mu[j]) / std[j]
        if val_ds is not None:
            X_val = [[(val - mu[j]) / std[j] for j, val in enumerate(row)] for row in val_ds.X]
            y_val = [float(v) for v in val_ds.y]
        else:
            X_val = None
            y_val = None
        W = [0.0 for _ in range(D)]
        b = 0.0
        best_loss = float("inf")
        best_W = W[:]
        best_b = b
        best_acc: Optional[float] = None
        patience = 0
        base_indices = list(range(N))
        history = []
        stop_training = False
        batch_size = max(1, cfg.batch_size)
        for ep in range(cfg.epochs):
            if cfg.shuffle:
                rng.shuffle(base_indices)
            else:
                base_indices = list(range(N))
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                gradW = [0.0 for _ in range(D)]
                gradb = 0.0
                count = end - start
                if count == 0:
                    continue
                for idx in base_indices[start:end]:
                    row = X_train[idx]
                    z = sum(row[j] * W[j] for j in range(D)) + b
                    z = max(min(z, 60.0), -60.0)
                    p = 1.0 / (1.0 + math.exp(-z))
                    diff = p - y_train[idx]
                    for j in range(D):
                        gradW[j] += row[j] * diff
                    gradb += diff
                scale = 1.0 / count
                for j in range(D):
                    gradW[j] = gradW[j] * scale + cfg.reg * W[j]
                    W[j] -= cfg.lr * gradW[j]
                gradb = gradb * scale
                b -= cfg.lr * gradb
            train_loss = self._loss_pure_python(X_train, y_train, W, b, cfg.reg)
            metrics = {"epoch": ep + 1, "train_loss": train_loss}
            if X_val is not None and y_val is not None and len(X_val) > 0:
                val_loss = self._loss_pure_python(X_val, y_val, W, b, cfg.reg)
                acc = self._accuracy_pure_python(X_val, y_val, W, b)
                metrics.update({"val_loss": val_loss, "val_acc": acc})
                if val_loss + 1e-6 < best_loss:
                    best_loss = val_loss
                    best_W = W[:]
                    best_b = b
                    best_acc = acc
                    patience = 0
                else:
                    patience += 1
                    if cfg.early_stopping_patience and patience >= cfg.early_stopping_patience:
                        stop_training = True
            else:
                if train_loss + 1e-6 < best_loss:
                    best_loss = train_loss
                    best_W = W[:]
                    best_b = b
            history.append(metrics)
            if stop_training:
                break
        self.W = np.array(best_W, dtype=float)
        self.b = best_b
        self.mu = np.array(mu, dtype=float)
        self.std = np.array(std, dtype=float)
        summary = {"train_samples": int(N), "features": int(D), "best_loss": float(best_loss),
                   "epochs_trained": len(history), "history": history}
        if X_val is not None and y_val is not None:
            summary["val_size"] = int(len(y_val))
            if best_acc is not None:
                summary["best_val_acc"] = float(best_acc)
        return summary

    @staticmethod
    def _loss_pure_python(X: list[list[float]], y: list[float], W: list[float], b: float, reg: float) -> float:
        total = 0.0
        for row, label in zip(X, y):
            z = sum(row[j] * W[j] for j in range(len(W))) + b
            if z >= 0:
                total += z + math.log1p(math.exp(-z)) - label * z
            else:
                total += math.log1p(math.exp(z)) - label * z
        if not X:
            return 0.5 * reg * sum(w * w for w in W)
        avg = total / len(X)
        return avg + 0.5 * reg * sum(w * w for w in W)

    @staticmethod
    def _accuracy_pure_python(X: list[list[float]], y: list[float], W: list[float], b: float) -> float:
        if not X:
            return 0.0
        correct = 0
        for row, label in zip(X, y):
            z = sum(row[j] * W[j] for j in range(len(W))) + b
            z = max(min(z, 60.0), -60.0)
            p = 1.0 / (1.0 + math.exp(-z))
            pred = 1.0 if p >= 0.5 else 0.0
            if pred == float(label):
                correct += 1
        return correct / len(X)

    def boundary_probs(self, text: str) -> np.ndarray:
        if len(text) <= 1: return np.zeros(0, dtype=float)
        if self.W is None or self.mu is None or self.std is None:
            raise RuntimeError("BoundaryStudent must be trained before calling boundary_probs")
        ps=[]
        for i in range(len(text)-1):
            left, right = text[i], text[i+1]
            x = (self._feat(left, right)-self.mu)/self.std
            z = float(x @ self.W + self.b)
            z = max(min(z, 60.0), -60.0)
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
            score_rows = scores.to_list() if isinstance(scores, np.ndarray) else list(scores)
            gate_vals = gate_pos.to_list() if isinstance(gate_pos, np.ndarray) else list(gate_pos)
            softmax_rows = []
            for row in score_rows:
                adjusted = [row[j] + 0.5 * gate_vals[j] for j in range(len(row))]
                m = max(adjusted)
                exps = [math.exp(v - m) for v in adjusted]
                denom = sum(exps) + 1e-12
                softmax_rows.append([v / denom for v in exps])
            A = np.array(softmax_rows)
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
