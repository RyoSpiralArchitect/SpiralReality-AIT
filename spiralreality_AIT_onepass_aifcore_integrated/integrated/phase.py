from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

from .np_compat import HAS_NUMPY, np
from .utils import seeded_vector


@dataclass
class PhaseBasisState:
    basis: Dict[str, List[List[float]]]

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(data: str) -> "PhaseBasisState":
        raw = json.loads(data)
        return PhaseBasisState(basis=raw["basis"])


class PhaseBasisLearner:
    """Learnable set of phase planes used to generate curvature/gating signals."""

    def __init__(self, dim: int = 64, lr: float = 5e-3):
        self.dim = dim
        self.lr = lr
        self.basis: Dict[str, np.ndarray] = {
            key: self._init_plane(key) for key in ("AB", "BC", "CA")
        }

    def _init_plane(self, seed: str) -> np.ndarray:
        a = seeded_vector(f"{seed}_a", self.dim)
        b = seeded_vector(f"{seed}_b", self.dim)
        e1 = self._unit(a)
        b = b - float(np.dot(b, e1)) * e1
        e2 = self._unit(b)
        return np.stack([e1, e2], axis=0)

    def _unit(self, vec: Iterable[float]) -> np.ndarray:
        arr = np.array(list(vec), dtype=float)
        n = np.linalg.norm(arr)
        return arr / (n + 1e-8)

    def export_state(self) -> PhaseBasisState:
        basis = {
            key: plane.to_list() if hasattr(plane, "to_list") else plane.tolist()
            for key, plane in self.basis.items()
        }
        return PhaseBasisState(basis=basis)

    def load_state(self, state: PhaseBasisState) -> None:
        self.basis = {
            key: np.array(vals, dtype=float) for key, vals in state.basis.items()
        }

    def phase_triplet(self, tok: str) -> Tuple[float, float, float]:
        emb = seeded_vector(f"tok::{tok}", self.dim)
        out: List[float] = []
        for key in ("AB", "BC", "CA"):
            plane = self.basis[key]
            x = float(np.dot(plane[0], emb))
            y = float(np.dot(plane[1], emb))
            out.append(math.atan2(y, x))
        return out[0], out[1], out[2]

    def _unwrap_seq(self, seq: List[float]) -> List[float]:
        if not seq:
            return []
        unwrapped = [seq[0]]
        for i in range(1, len(seq)):
            d = seq[i] - seq[i - 1]
            d -= 2 * math.pi * round(d / (2 * math.pi))
            unwrapped.append(unwrapped[-1] + d)
        return unwrapped

    def curvature(self, text: str) -> np.ndarray:
        seqs = {"AB": [], "BC": [], "CA": []}
        for ch in text:
            ph = self.phase_triplet(ch)
            seqs["AB"].append(ph[0])
            seqs["BC"].append(ph[1])
            seqs["CA"].append(ph[2])
        unwrapped = {k: self._unwrap_seq(v) for k, v in seqs.items()}
        curv = np.zeros(len(text), dtype=float)
        for i in range(len(text)):
            acc = 0.0
            for key in ("AB", "BC", "CA"):
                seq = unwrapped[key]
                if len(seq) >= 3 and i >= 2:
                    dd = seq[i] - 2 * seq[i - 1] + seq[i - 2]
                    acc += abs(dd) / 3.0
            curv[i] = acc
        med = float(np.median(curv))
        mad = float(np.median(np.abs(curv - med)) + 1e-6)
        return (curv - med) / mad

    def apply_error(self, text: str, boundary_index: int, error: float, scale: float = 1.0) -> None:
        """Heuristically update basis vectors near a boundary error."""
        span = range(max(0, boundary_index - 2), min(len(text), boundary_index + 3))
        for pos in span:
            ch = text[pos]
            emb = seeded_vector(f"tok::{ch}", self.dim)
            emb_vals = emb.to_list() if hasattr(emb, "to_list") else list(emb)
            rev = list(reversed(emb_vals))
            if HAS_NUMPY:
                emb_arr = np.array(emb_vals, dtype=float)
                rev_arr = np.array(rev, dtype=float)
            else:
                emb_arr = emb_vals
                rev_arr = rev
            for key in ("AB", "BC", "CA"):
                plane = self.basis[key]
                if HAS_NUMPY:
                    plane[0] -= self.lr * scale * error * emb_arr
                    plane[1] -= self.lr * scale * error * rev_arr
                else:
                    p0 = plane[0].to_list() if hasattr(plane[0], "to_list") else list(plane[0])
                    p1 = plane[1].to_list() if hasattr(plane[1], "to_list") else list(plane[1])
                    p0 = [v - self.lr * scale * error * g for v, g in zip(p0, emb_arr)]
                    p1 = [v - self.lr * scale * error * g for v, g in zip(p1, rev_arr)]
                    plane = np.array([p0, p1], dtype=float)
                plane[0] = self._unit(plane[0])
                plane[1] -= float(np.dot(plane[1], plane[0])) * plane[0]
                plane[1] = self._unit(plane[1])
                self.basis[key] = plane

    def local_features(self, text: str, window: int = 3) -> np.ndarray:
        """Return curvature-derived local features for positional encoding."""

        if not text:
            return np.zeros((0, 3), dtype=float)
        curvature = self.curvature(text)
        curv_vals = curvature.to_list() if hasattr(curvature, "to_list") else list(curvature)
        feats_list = [[0.0, 0.0, 0.0] for _ in range(len(text))]
        for i, val in enumerate(curv_vals):
            start = max(0, i - window)
            end = min(len(curv_vals), i + window + 1)
            local = curv_vals[start:end]
            mean_val = sum(local) / max(1, len(local))
            if len(local) > 1:
                var = sum((x - mean_val) ** 2 for x in local) / len(local)
            else:
                var = 0.0
            feats_list[i][0] = float(val)
            feats_list[i][1] = float(mean_val)
            feats_list[i][2] = float(math.sqrt(var))
        return np.array(feats_list, dtype=float)

