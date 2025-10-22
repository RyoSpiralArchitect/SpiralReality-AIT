from __future__ import annotations

from typing import Optional

try:  # pragma: no cover - optional dependency
    from fastapi import FastAPI
    from pydantic import BaseModel
except Exception:  # pragma: no cover - FastAPI not installed
    FastAPI = None
    BaseModel = object  # type: ignore

from .checkpoint import load_checkpoint
from .onepass_ait import OnePassAIT, StudentTrainingConfig


class TrainRequest(BaseModel):  # type: ignore[misc]
    texts: list[str]
    segments: list[list[str]]
    lr: float = 0.05
    epochs: int = 40


class TextRequest(BaseModel):  # type: ignore[misc]
    text: str


def create_app(model: Optional[OnePassAIT] = None) -> "FastAPI":
    if FastAPI is None:  # pragma: no cover - guard
        raise RuntimeError("FastAPI is not available in this environment")
    ait = model or OnePassAIT()
    app = FastAPI(title="One-Pass AIT API")

    @app.get("/health")
    def health():
        return {"status": "ok", "latent_dim": ait.latent_dim}

    @app.post("/train")
    def train(req: TrainRequest):
        cfg = StudentTrainingConfig(lr=req.lr, epochs=req.epochs)
        summary = ait.train_student(req.texts, req.segments, cfg=cfg)
        return summary

    @app.post("/segment")
    def segment(req: TextRequest):
        tokens = ait.segment_text(req.text)
        return {"tokens": tokens}

    @app.post("/encode")
    def encode(req: TextRequest):
        enc = ait.encode(req.text)
        H = enc["H"].tolist() if hasattr(enc["H"], "tolist") else enc["H"].to_list()
        return {
            "H": H,
            "ps": enc["ps"].tolist() if hasattr(enc["ps"], "tolist") else enc["ps"].to_list(),
            "gate": enc["gate_pos"].tolist() if hasattr(enc["gate_pos"], "tolist") else enc["gate_pos"].to_list(),
        }

    @app.post("/load")
    def load(path: TextRequest):
        state = load_checkpoint(path.text)
        ait.load_state_dict(state)
        return {"status": "loaded"}

    return app


__all__ = ["create_app"]

