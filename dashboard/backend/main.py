"""
Seyir Dashboard Backend — FastAPI + WebSocket.

REST endpoints for run history.
WebSocket /ws/live streams live simulation state at ~10 Hz.
"""
from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "dashboard_backend.log",
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("dashboard.backend")

METRICS_DIR = Path(__file__).parent.parent.parent / "logs"

app = FastAPI(title="Seyir Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for live simulation state (written by run_simulation.py)
_live_state: dict[str, Any] = {}
_ws_clients: list[WebSocket] = []


# ── REST ──────────────────────────────────────────────────────────────────── #

@app.get("/api/runs")
async def list_runs() -> list[dict]:
    """Return metadata for all saved evaluation runs."""
    runs: list[dict] = []
    for path in sorted(METRICS_DIR.glob("metrics_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            runs.append({
                "id":       path.stem.replace("metrics_", ""),
                "filename": path.name,
                "summary":  data.get("summary", {}),
            })
        except Exception as exc:
            logger.warning("Could not parse %s: %s", path, exc)
    return runs


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    """Return full metrics for a specific run."""
    path = METRICS_DIR / f"metrics_{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    with open(path) as f:
        return json.load(f)


@app.get("/api/runs/compare")
async def compare_runs(run_a: str, run_b: str) -> dict:
    """Return side-by-side summaries for two runs."""
    result = {}
    for rid in (run_a, run_b):
        path = METRICS_DIR / f"metrics_{rid}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Run {rid!r} not found")
        with open(path) as f:
            data = json.load(f)
        result[rid] = data.get("summary", {})
    return result


# ── WebSocket live stream ─────────────────────────────────────────────────── #

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    """Stream live simulation state to connected frontend clients."""
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info("WebSocket client connected (total=%d)", len(_ws_clients))
    try:
        while True:
            # Send the latest state at ~10 Hz
            if _live_state:
                await websocket.send_json(_live_state)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


# ── Internal endpoint called by run_simulation.py ────────────────────────── #

@app.post("/internal/update")
async def update_state(payload: dict) -> dict:
    """
    Called by the simulation loop (localhost only) to push the latest frame.
    Not exposed to the public frontend.
    """
    global _live_state
    _live_state = payload
    return {"ok": True}


async def broadcast(state: dict) -> None:
    """Broadcast state to all connected WebSocket clients (called from sim loop)."""
    dead: list[WebSocket] = []
    for ws in _ws_clients:
        try:
            await ws.send_json(state)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


def set_live_state(state: dict) -> None:
    """Thread-safe setter called from the synchronous simulation loop."""
    global _live_state
    _live_state = state


if __name__ == "__main__":
    uvicorn.run(
        "dashboard.backend.main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
    )
