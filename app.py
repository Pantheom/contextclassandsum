"""
app.py
------
FastAPI backend for the Summarizer/Classifier Debug UI.

Run with:
    uvicorn app:app --reload

Design decisions:
- POST /api/init is BLOCKING. Model loading (both Phi-4-mini + Gemma 3) may
  take 30-90 seconds. For a single-developer debug tool this is acceptable
  and simpler than async polling. The frontend shows a spinner and waits.
- No CORS middleware needed: index.html is served by this same process via
  StaticFiles, so all fetch() calls are same-origin.
- No authentication: local single-user tool assumption.
- All state lives in the existing SQLite DB. This app adds no new persistence.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()  # Load .env before any os.environ reads

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Package imports — direct Python calls, no HTTP between services
# ---------------------------------------------------------------------------
import summarizer
from summarizer import (
    write_turn,
    summarize_on_demand,
    open_connection,
    get_all_turns,
    get_all_session_ids,
    cfg as summarizer_cfg,
)
from summarizer.db import get_session     # read-only helper, already exists

import classifier
from classifier import get_response_context
from classifier.service import classify_debug   # debug variant (returns raw output)

logger = logging.getLogger("app")

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Summarizer / Classifier Debug UI",
    description="Internal testing tool — not for production.",
    docs_url="/docs",        # Swagger available at /docs
    redoc_url=None,
)

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    """Serve the single-page debug UI."""
    return FileResponse(_STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# Service readiness state — module-level dict (single process, single user)
# ---------------------------------------------------------------------------
_state: Dict[str, Any] = {
    "summarizer_ready": False,
    "classifier_ready": False,
    "init_error": None,
}


# ---------------------------------------------------------------------------
# Request body models
# ---------------------------------------------------------------------------
class TurnBody(BaseModel):
    role: str   # 'user' | 'assistant'
    text: str

class PromptBody(BaseModel):
    prompt: str


# ---------------------------------------------------------------------------
# /api/init  — block until both models are loaded
# ---------------------------------------------------------------------------
@app.post("/api/init")
def api_init() -> Dict[str, str]:
    """Initialise both services and force-load both model weights into RAM.

    BLOCKING — may take 30-90 s on first call.
    Safe to call again: services and models are idempotent on re-init.
    """
    _state["init_error"] = None
    try:
        # 1. Init summarizer (logging + DB schema)
        summarizer.init_service()
        _state["summarizer_ready"] = True

        # 2. Init classifier (logging only — no schema ownership)
        classifier.init_classifier()
        _state["classifier_ready"] = True

        # 3. Force-load models now rather than on first inference call.
        #    This way /api/status reflects "ready" only when both are in RAM.
        from summarizer.model import get_model as _get_sum_model
        _get_sum_model()

        from classifier.model import get_model as _get_cls_model
        _get_cls_model()

        return {"status": "ready", "detail": "Both models loaded successfully."}

    except Exception as exc:
        _state["init_error"] = str(exc)
        # Partial-ready flags: keep whichever service did succeed.
        logger.exception("Init failed")
        return {"status": "error", "detail": str(exc)}


# ---------------------------------------------------------------------------
# /api/status
# ---------------------------------------------------------------------------
@app.get("/api/status")
def api_status() -> Dict[str, Any]:
    return {
        "summarizer_ready": _state["summarizer_ready"],
        "classifier_ready": _state["classifier_ready"],
        "ready": _state["summarizer_ready"] and _state["classifier_ready"],
        "error": _state["init_error"],
    }


# ---------------------------------------------------------------------------
# /api/sessions
# ---------------------------------------------------------------------------
@app.get("/api/sessions")
def api_list_sessions() -> Dict[str, List[str]]:
    """Return all known session IDs, newest first."""
    import sqlite3 as _sqlite3
    try:
        conn = open_connection(summarizer_cfg.db_path)
        try:
            session_ids = get_all_session_ids(conn)
        finally:
            conn.close()
        return {"sessions": session_ids}
    except (_sqlite3.OperationalError, Exception):
        return {"sessions": []}


# ---------------------------------------------------------------------------
# /api/session/{session_id}/turn
# ---------------------------------------------------------------------------
@app.post("/api/session/{session_id}/turn")
def api_add_turn(session_id: str, body: TurnBody) -> Dict[str, int]:
    if body.role not in ("user", "assistant"):
        raise HTTPException(
            status_code=422,
            detail="role must be 'user' or 'assistant'",
        )
    turn_index = write_turn(session_id, body.role, body.text)
    return {"turn_index": turn_index}


# ---------------------------------------------------------------------------
# /api/session/{session_id}/turns
# ---------------------------------------------------------------------------
@app.get("/api/session/{session_id}/turns")
def api_get_turns(session_id: str) -> Dict[str, List[Dict]]:
    """Return all turns for the session in chronological order."""
    import sqlite3 as _sqlite3
    try:
        conn = open_connection(summarizer_cfg.db_path)
        try:
            turns = get_all_turns(conn, session_id)
        finally:
            conn.close()
    except (_sqlite3.OperationalError, Exception):
        return {"turns": []}
    return {
        "turns": [
            {
                "turn_id":    t.turn_id,
                "turn_index": t.turn_index,
                "role":       t.role,
                "text":       t.text,
                "timestamp":  t.timestamp,
            }
            for t in turns
        ]
    }


# ---------------------------------------------------------------------------
# /api/session/{session_id}/summary
# ---------------------------------------------------------------------------
@app.get("/api/session/{session_id}/summary")
def api_get_summary(session_id: str) -> Dict[str, Any]:
    """Return current_summary and last_summarized_turn_index directly from DB."""
    import sqlite3 as _sqlite3
    _empty = {"current_summary": None, "last_summarized_turn_index": 0, "updated_at": None}
    try:
        conn = open_connection(summarizer_cfg.db_path)
        try:
            session = get_session(conn, session_id)
        finally:
            conn.close()
    except (_sqlite3.OperationalError, Exception):
        return _empty
    if session is None:
        return _empty
    return {
        "current_summary":            session.current_summary,
        "last_summarized_turn_index": session.last_summarized_turn_index,
        "updated_at":                 session.updated_at,
    }


# ---------------------------------------------------------------------------
# /api/session/{session_id}/summarize  (manual trigger)
# ---------------------------------------------------------------------------
@app.post("/api/session/{session_id}/summarize")
def api_force_summarize(session_id: str) -> Dict[str, str]:
    """Directly call summarize_on_demand — bypasses the classifier entirely."""
    summary = summarize_on_demand(session_id)
    return {"summary": summary}


# ---------------------------------------------------------------------------
# /api/session/{session_id}/classify  (debug: returns raw model output too)
# ---------------------------------------------------------------------------
@app.post("/api/session/{session_id}/classify")
def api_classify(session_id: str, body: PromptBody) -> Dict[str, Any]:
    """Run the classifier and return both the verdict and the raw model output."""
    result = classify_debug(session_id, body.prompt)
    return {
        "needs_context": result["needs_context"],
        "raw_output":    result["raw_output"],
    }


# ---------------------------------------------------------------------------
# /api/session/{session_id}/get_context  (full pipeline)
# ---------------------------------------------------------------------------
@app.post("/api/session/{session_id}/get_context")
def api_get_context(session_id: str, body: PromptBody) -> Dict[str, Any]:
    """Full pipeline: classify -> optionally summarize_on_demand -> return context."""
    result = get_response_context(session_id, body.prompt)
    return result
