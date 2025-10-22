from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .np_compat import np
from .boundary import BoundaryStudent, StudentTrainingConfig
from .dynamics import LatentDynamicsModel
from .encoder import ToyTransformerAdapter
from .multilingual import build_multilingual_corpus, language_histogram
from .phase import PhaseBasisLearner
from .utils import seeded_vector, sigmoid, unit


@dataclass
class GateDiagnostics:
    gate_trace: List[float]
    attention_strength: List[float]
    mask_energy: float = 0.0


class OnePassAIT:
    def __init__(self, latent_dim: int = 64, seed: int = 4242):
        self.latent_dim = latent_dim
        self.rng = np.random.default_rng(seed)
        self.policies = ["ProbeMotivation", "ProbeReliability", "SeekEvidence", "DecideNow"]
        self.policy_vecs = {p: seeded_vector(p, latent_dim) for p in self.policies}
        self.goal_vec = unit(self.rng.normal(size=latent_dim))
        self.phase = PhaseBasisLearner(dim=latent_dim)
        self.student = BoundaryStudent(self.phase, seed=seed)
        self.encoder = ToyTransformerAdapter(d_model=latent_dim, n_layers=3, seed=seed)
        self.student.bind_encoder(self.encoder)
        self.dynamics = LatentDynamicsModel(latent_dim, latent_dim, seed=seed)
        self.beta_ewma = 0.2
        self.gate_a0, self.gate_a1 = 1.0, 1.3
        self.mu = np.zeros(latent_dim)
        self.Sigma = np.eye(latent_dim)
        self.R3_mix = 0.0
        self.R2_time = 0.0
        self._phi_hist: Dict[str, List[float]] = {"AB": [], "BC": [], "CA": []}
        self.last_gate_trace: List[float] = []
        self.last_attention: List[np.ndarray] = []
        self.last_gate_mask: Optional[np.ndarray] = None

    def train_student(
        self,
        texts: Optional[Sequence[str]] = None,
        segments: Optional[Sequence[Sequence[str]]] = None,
        cfg: Optional[StudentTrainingConfig] = None,
        languages: Optional[Sequence[str]] = None,
        include_reflective: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, object]:
        """Train the boundary student on the provided or assembled corpus."""

        dataset_texts: List[str] = []
        dataset_segments: List[List[str]] = []
        dataset_tags: List[str] = []

        if texts is not None:
            dataset_texts = list(texts)
            if segments is not None:
                dataset_segments = [list(seg) for seg in segments]
            else:
                from .corpus import teacher_segments

                dataset_segments = teacher_segments(dataset_texts)
            dataset_tags.extend(["custom"] * len(dataset_texts))
        elif segments is not None:
            raise ValueError("segments were provided without matching texts")

        if languages is not None:
            ml_seed = seed if seed is not None else self._next_seed()
            ml_texts, ml_segments, ml_tags = build_multilingual_corpus(
                languages=languages,
                include_reflective=include_reflective,
                shuffle=shuffle,
                seed=ml_seed,
            )
            if dataset_texts:
                dataset_texts.extend(ml_texts)
                dataset_segments.extend(ml_segments)
                dataset_tags.extend(ml_tags)
            else:
                dataset_texts = ml_texts
                dataset_segments = ml_segments
                dataset_tags = ml_tags

        if not dataset_texts:
            from .corpus import TRAIN_TEXTS, teacher_segments

            dataset_texts = list(TRAIN_TEXTS)
            dataset_segments = teacher_segments(dataset_texts)
            dataset_tags = ["reflective"] * len(dataset_texts)

        summary = self.student.train(dataset_texts, dataset_segments, cfg=cfg)
        if isinstance(summary, dict):
            summary = dict(summary)
            summary.setdefault("dataset_size", len(dataset_texts))
            summary.setdefault("dataset_tags", dataset_tags)
            summary.setdefault("dataset_languages", language_histogram(dataset_tags))
            summary.setdefault("dataset_texts", dataset_texts)
            summary.setdefault("dataset_segments", dataset_segments)
        else:
            summary = {
                "result": summary,
                "dataset_size": len(dataset_texts),
                "dataset_tags": dataset_tags,
                "dataset_languages": language_histogram(dataset_tags),
                "dataset_texts": dataset_texts,
                "dataset_segments": dataset_segments,
            }
        return summary

    def _next_seed(self) -> int:
        if hasattr(self.rng, "integers"):
            return int(self.rng.integers(2**31 - 1))
        if hasattr(self.rng, "randint"):
            return int(self.rng.randint(0, 2**31 - 1))
        if hasattr(self.rng, "random"):
            return int(self.rng.random() * (2**31 - 1))
        return 4242

    def _char_embs(self, text: str) -> np.ndarray:
        embs = [seeded_vector(f"char::{c}", self.latent_dim) for c in text]
        return np.stack(embs, axis=0)

    def encode(self, text: str) -> Dict[str, np.ndarray]:
        if not text:
            return {
                "H": np.zeros((0, self.latent_dim)),
                "r2_local": np.zeros(0),
                "ps": np.zeros(0),
                "gate_pos": np.zeros(0),
                "phase_local": np.zeros((0, 3)),
                "gate_mask": np.zeros((0, 0)),
            }
        chars = list(text)
        X = self._char_embs(text)
        ps = self.student.boundary_probs(text)
        curvature = self.phase.curvature(text)
        curv_list = curvature.to_list() if hasattr(curvature, "to_list") else list(curvature)
        char_signal = [0.0 for _ in chars]
        for i in range(len(chars) - 1):
            val = float(ps[i]) if i < len(ps) else 0.0
            char_signal[i] = max(char_signal[i], val)
            char_signal[i + 1] = max(char_signal[i + 1], val)
        gate_from_curv = [sigmoid(v) for v in curv_list]
        gate_pos = np.array([
            0.55 * char_signal[i] + 0.45 * gate_from_curv[i] for i in range(len(chars))
        ], dtype=float)
        phase_local = self.phase.local_features(text)
        phase_bias = self._phase_positional(phase_local)
        X_phase = X + phase_bias
        if len(gate_pos):
            gate_list = gate_pos.tolist() if hasattr(gate_pos, "tolist") else list(gate_pos)
            gate_mask = np.array(
                [[min(gi, gj) for gj in gate_list] for gi in gate_list], dtype=float
            )
        else:
            gate_mask = np.zeros((0, 0))
        H = self.encoder.forward(X_phase, gate_pos, gate_mask=gate_mask)
        self.last_gate_trace = gate_pos.tolist()
        self.last_attention = self.encoder.last_attn
        self.last_gate_mask = gate_mask
        return {
            "H": H,
            "r2_local": curvature,
            "ps": ps,
            "gate_pos": gate_pos,
            "phase_local": phase_local,
            "gate_mask": gate_mask,
        }

    def _phase_positional(self, phase_local: np.ndarray) -> np.ndarray:
        if hasattr(phase_local, "tolist"):
            phase_rows = phase_local.tolist()
        elif hasattr(phase_local, "to_list"):
            phase_rows = phase_local.to_list()
        else:
            phase_rows = list(phase_local)
        if not phase_rows:
            return np.zeros((0, self.latent_dim), dtype=float)
        seq_len = len(phase_rows)
        feat_dim = len(phase_rows[0]) if phase_rows and hasattr(phase_rows[0], "__len__") else 0
        proj_list = [[0.0 for _ in range(self.latent_dim)] for _ in range(seq_len)]
        freqs = [0.75 + 0.5 * idx for idx in range(max(1, feat_dim))]
        for i in range(seq_len):
            offset = 0
            row = phase_rows[i]
            if hasattr(row, "__iter__") and not isinstance(row, (int, float)):
                iterable_row = list(row)
            else:
                iterable_row = [float(row)]
            for feat_idx, val in enumerate(iterable_row):
                freq = freqs[feat_idx]
                if offset < self.latent_dim:
                    proj_list[i][offset] = math.sin(freq * val)
                if offset + 1 < self.latent_dim:
                    proj_list[i][offset + 1] = math.cos(freq * val)
                offset += 2
                if offset >= self.latent_dim:
                    break
        proj_arr = np.array(proj_list, dtype=float)
        return proj_arr * 0.1

    def policy_local_gates(self, H: np.ndarray, r2_local: np.ndarray, policy: str) -> np.ndarray:
        pv = unit(self.policy_vecs[policy])
        rows = H.to_list() if hasattr(H, "to_list") else H.tolist()
        aligns = np.array([float(np.dot(unit(row), pv)) for row in rows])
        r2_vals = r2_local.to_list() if hasattr(r2_local, "to_list") else list(r2_local)
        alpha_loc = self.gate_a0 + self.gate_a1 * np.clip(r2_vals, -5.0, 5.0)
        g = 1.0 / (1.0 + np.exp(-alpha_loc * aligns))
        return g

    def predict_next(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        policy: str,
        ctx_vec: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        policy_vec = unit(self.policy_vecs[policy])
        teacher_delta = (
            0.22 * policy_vec
            + 0.12 * self.goal_vec * (1.0 / (1.0 + np.linalg.norm(mu)))
            + 0.15 * unit(ctx_vec)
        )
        teacher_mu = mu + teacher_delta
        teacher_Sigma = 0.9 * Sigma + 0.05 * np.eye(self.latent_dim)
        self.dynamics.record(mu, policy_vec, ctx_vec, teacher_mu)
        if len(self.dynamics.buffer) >= 32:
            self.dynamics.train()
        if self.dynamics.is_ready():
            return self.dynamics.predict(mu, policy_vec, ctx_vec)
        return teacher_mu, teacher_Sigma

    def update_global_phases(self, mu: np.ndarray) -> None:
        for key in ("AB", "BC", "CA"):
            plane = self.phase.basis[key]
            x = float(np.dot(plane[0], mu))
            y = float(np.dot(plane[1], mu))
            p = math.atan2(y, x)
            hist = self._phi_hist[key]
            if not hist:
                hist.append(p)
            else:
                last = hist[-1]
                d = p - last
                d -= 2 * math.pi * round(d / (2 * math.pi))
                hist.append(last + d)
            self._phi_hist[key] = hist[-6:]
        dv = {}
        for key in ("AB", "BC", "CA"):
            hist = self._phi_hist[key]
            if len(hist) < 2:
                dv[key] = 0.0
            else:
                dv[key] = hist[-1] - hist[-2]
        chi = math.copysign(1.0, dv.get("AB", 1.0)) * math.copysign(1.0, dv.get("BC", 1.0)) * math.copysign(1.0, dv.get("CA", 1.0))
        chi = 1.0 if chi > 0 else -1.0
        r = math.sqrt(sum(v * v for v in dv.values()) / max(1, len(dv))) * max(0.0, chi)
        self.R3_mix = (1 - self.beta_ewma) * self.R3_mix + self.beta_ewma * min(5.0, r)
        acc = 0.0
        for key in ("AB", "BC", "CA"):
            hist = self._phi_hist[key]
            if len(hist) >= 3:
                acc += abs(hist[-1] - 2 * hist[-2] + hist[-3])
        curv = acc / 3.0 if acc else 0.0
        self.R2_time = (1 - self.beta_ewma) * self.R2_time + self.beta_ewma * min(5.0, curv)

    def segment_text(self, text: str) -> List[str]:
        return self.student.decode(text)

    def gate_diagnostics(self) -> GateDiagnostics:
        strengths = []
        for attn in self.last_attention:
            if hasattr(attn, "tolist"):
                matrix = attn.tolist()
            elif hasattr(attn, "to_list"):
                matrix = attn.to_list()
            else:
                matrix = list(attn)
            strengths.append(float(max(sum(row) for row in matrix)))
        mask_energy = 0.0
        if self.last_gate_mask is not None:
            if hasattr(self.last_gate_mask, "tolist"):
                matrix = self.last_gate_mask.tolist()
            elif hasattr(self.last_gate_mask, "to_list"):
                matrix = self.last_gate_mask.to_list()
            else:
                matrix = self.last_gate_mask
            if matrix and hasattr(matrix, "__iter__") and not isinstance(matrix[0], (int, float)):
                flat = [float(val) for row in matrix for val in row]
            else:
                flat = [float(val) for val in matrix]
            if flat:
                mask_energy = sum(flat) / len(flat)
        return GateDiagnostics(
            gate_trace=self.last_gate_trace[:],
            attention_strength=strengths,
            mask_energy=mask_energy,
        )

    def state_dict(self) -> Dict[str, object]:
        return {
            "latent_dim": self.latent_dim,
            "goal_vec": self.goal_vec.tolist(),
            "policy_vecs": {k: v.tolist() for k, v in self.policy_vecs.items()},
            "phase": self.phase.export_state().basis,
            "student": self.student.export_state(),
            "encoder": self.encoder.export_state(),
            "dynamics": self.dynamics.export_state(),
            "phi_hist": self._phi_hist,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if state.get("latent_dim") != self.latent_dim:
            raise ValueError("Latent dimension mismatch in checkpoint")
        self.goal_vec = np.array(state["goal_vec"], dtype=float)
        self.policy_vecs = {k: np.array(v, dtype=float) for k, v in state["policy_vecs"].items()}
        phase_state = self.phase.export_state()
        phase_state.basis = state["phase"]
        self.phase.load_state(phase_state)
        self.student.load_state(state["student"])
        self.encoder.load_state(state["encoder"])
        self.dynamics.load_state(state["dynamics"])
        self._phi_hist = {k: list(v) for k, v in state.get("phi_hist", {}).items()}

