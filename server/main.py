"""FastAPI server exposing One-Pass AIT inference over HTTP and WebSocket."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from spiralreality_AIT_onepass_aifcore_integrated.integrated.onepass_ait import (
    GateDiagnostics,
    OnePassAIT,
)

logger = logging.getLogger(__name__)


def _serialise_gate(diag: GateDiagnostics) -> Dict[str, Any]:
    return {
        "gate_trace": list(diag.gate_trace),
        "attention_strength": list(diag.attention_strength),
        "mask_energy": float(diag.mask_energy),
    }


def _serialise_encode(result: Dict[str, Any]) -> Dict[str, Any]:
    serialised: Dict[str, Any] = {
        "boundary_probabilities": list(map(float, result.get("ps", []))),
        "gate_trace": list(map(float, result.get("gate_pos", []))),
    }
    if "phase_local" in result:
        serialised["phase_local"] = [list(map(float, row)) for row in result["phase_local"]]
    if "gate_mask" in result:
        serialised["gate_mask"] = [list(map(float, row)) for row in result["gate_mask"]]
    if "H" in result:
        serialised["embedding"] = [list(map(float, row)) for row in result["H"]]
    return serialised


ait = OnePassAIT()
app = FastAPI(title="SpiralReality One-Pass AIT Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "latent_dim": ait.latent_dim})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json({"type": "ready", "message": "Send a JSON payload with a 'text' field."})
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
                logger.warning("Received invalid JSON payload: %s", exc)
                await websocket.send_json({"type": "error", "message": "Invalid JSON payload."})
                continue
            text = payload.get("text", "")
            if not isinstance(text, str):
                await websocket.send_json({"type": "error", "message": "'text' must be a string."})
                continue
            await websocket.send_json({"type": "processing", "length": len(text)})
            result = ait.encode(text)
            segments = ait.segment_text(text)
            gate_diag = ait.gate_diagnostics()
            message = {
                "type": "result",
                "text": text,
                "segments": segments,
                "diagnostics": _serialise_gate(gate_diag),
                "encoding": _serialise_encode(result),
            }
            await websocket.send_json(message)
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unhandled server error: %%s", exc)
        await websocket.close(code=1011, reason="Internal server error")


__all__ = ["app"]
