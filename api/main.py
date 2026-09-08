"""
api/main.py
-----------
Production FastAPI application for the Context Service.

This module is the entry point that gets deployed on AWS.
Run with: uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}

Design:
- Both GGUF models are loaded automatically on startup in a background thread.
  Endpoints return 503 while loading is in progress.
- All state lives in the existing SQLite DB owned by summarizer/db.py.
- No authentication is currently configured — see api/auth.py for the stub.
  To enable auth: uncomment `Depends(require_auth)` in every route definition
  and implement the function body in api/auth.py.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # Load .env before any os.environ reads (config singletons read at import time)

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Package imports — direct Python calls, same packages as the debug app,
# no HTTP between services.
# ---------------------------------------------------------------------------
import summarizer
from summarizer import write_turn, init_service
import classifier
from classifier import init_classifier, get_response_context

# ---------------------------------------------------------------------------
# API-layer imports
# ---------------------------------------------------------------------------
from api.auth import require_auth  # noqa: F401 — stub, ready to Depends() on
from api.config import cfg
from api.errors import register_handlers
from api.models import (
    ProcessRequest,
    ProcessResponse,
    ReplyRequest,
    ReplyResponse,
    HealthResponse,
    build_combined_prompt,
)

logger = logging.getLogger("api.main")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

_DESCRIPTION = """
## Overview

This API handles conversation context management for a multi-session chat system.
It classifies incoming user prompts, manages rolling summarization, and returns
a ready-to-use context-enriched prompt for your downstream LLM — so you never
have to manage conversation history yourself.

---

## Integration Pattern — 2 calls per turn

### Step 1 — Process the user's message
**`POST /v1/process`**

Send the user's raw prompt. The API will:
1. Determine whether prior conversation context is needed (using this session's history)
2. If yes — fetch a rolling summary + recent turns and bundle them into `combined_prompt`
3. Log the user's turn to the session DB automatically
4. Return a response containing `combined_prompt`

**Take `combined_prompt` from the response and send it directly to your own LLM.
No further assembly is needed on your end.**

### Step 2 — Log the assistant's reply
**`POST /v1/sessions/{session_id}/reply`**

Once your LLM generates a response, call this endpoint with the raw reply text.
This keeps the conversation history complete so the **next** `/v1/process` call
has accurate context to work with.

---

## Authentication

**No authentication is currently required.** Auth will be added in a future
revision — see `api/auth.py` for the implementation stub.
"""

app = FastAPI(
    title="Context Service — Production API",
    description=_DESCRIPTION,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    # Debug app references are intentionally absent from this app's /docs.
)

register_handlers(app)


# ---------------------------------------------------------------------------
# Model-loading state — thread-safe event flag
# ---------------------------------------------------------------------------

_models_ready = threading.Event()
_init_error: Optional[str] = None


def _load_models_bg() -> None:
    """Load both GGUF models in a background thread.

    Sets _models_ready when complete. Sets _init_error on failure.
    Never raises — errors are captured and surfaced via /v1/health.
    """
    global _init_error
    try:
        logger.info("Starting model loading…")

        # 1. Init summarizer (logging + DB schema)
        init_service()

        # 2. Init classifier (logging only — no DB schema)
        init_classifier()

        # 3. Force-load summarizer weights into RAM now.
        #    The Llama singleton is lazy by default; we eagerly load here so
        #    /v1/health accurately reflects "ready" only when the model is
        #    actually in RAM, not just on the next inference call.
        from summarizer.model import get_model as _get_sum_model
        _get_sum_model()
        logger.info("Summarizer model loaded.")

        # 4. Force-load classifier weights.
        from classifier.model import get_model as _get_cls_model
        _get_cls_model()
        logger.info("Classifier model loaded.")

        _models_ready.set()
        logger.info("Both models ready — API accepting requests.")

    except Exception as exc:
        _init_error = str(exc)
        logger.exception("Model loading failed: %s", exc)


@app.on_event("startup")
async def startup_event() -> None:
    """Kick off model loading in a thread-pool thread.

    The event loop remains unblocked during model loading (which can take
    30–90 s). Requests that arrive during this window receive a 503 response
    until _models_ready is set.

    Note: @app.on_event("startup") is deprecated in favour of the lifespan
    context manager in FastAPI ≥ 0.93. Migrating is straightforward — this
    usage works correctly for now and can be updated when the project upgrades.
    """
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _load_models_bg)


# ---------------------------------------------------------------------------
# Readiness dependency — injected into every inference endpoint
# ---------------------------------------------------------------------------

def _require_models_ready() -> None:
    """FastAPI dependency that rejects requests while models are loading.

    Raises HTTP 503 if _models_ready is not yet set.
    Add Depends(_require_models_ready) to any endpoint that needs the models.

    # AUTH SLOT: add Depends(require_auth) alongside this dependency when
    # auth is implemented in api/auth.py.
    """
    if not _models_ready.is_set():
        raise HTTPException(
            status_code=503,
            detail="Models are still loading, retry shortly.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/v1/health",
    response_model=HealthResponse,
    summary="Service health check",
    description=(
        "Returns whether the API process is reachable (`status: ok`) and "
        "whether both GGUF models have finished loading (`models_loaded`). "
        "Poll this after deployment to know when the service is ready to "
        "accept `/v1/process` requests."
    ),
    tags=["Operations"],
)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded=_models_ready.is_set(),
    )


@app.post(
    "/v1/process",
    response_model=ProcessResponse,
    summary="Process a user prompt",
    description=(
        "**Main integration endpoint — call this once per incoming user message.**\n\n"
        "The API classifies whether prior conversation context is required, "
        "optionally fetches it, and returns `combined_prompt` which is ready "
        "to send directly to your answering LLM.\n\n"
        "The user's turn is automatically logged to the session history — "
        "you do **not** need a separate 'log user turn' call.\n\n"
        "After your LLM generates its reply, call "
        "`POST /v1/sessions/{session_id}/reply` to keep history complete."
    ),
    tags=["Context"],
    dependencies=[Depends(_require_models_ready), Depends(require_auth)],
)
def process_prompt(body: ProcessRequest) -> ProcessResponse:
    session_id = body.session_id
    prompt     = body.prompt

    # Step 1+2: Classify and conditionally fetch context.
    # get_response_context handles: classify → (if YES) summarize_on_demand
    # → get_last_n_turns → format context block. Returns:
    #   {"needs_context": bool, "context": str | None}
    ctx_result = get_response_context(session_id, prompt)
    needs_context: bool         = ctx_result["needs_context"]
    context:       Optional[str] = ctx_result.get("context")

    # Step 3: Build the combined prompt.
    if needs_context and context:
        combined = build_combined_prompt(context, prompt)
    else:
        # Self-contained prompt — pass through unchanged.
        combined = prompt

    # Step 4: Log the user's turn. This may trigger periodic background
    # summarization if the session gap reaches the threshold (15 turns).
    turn_index: int = write_turn(session_id, "user", prompt)

    logger.info(
        "PROCESSED session=%s turn=%d needs_context=%s",
        session_id,
        turn_index,
        needs_context,
    )

    return ProcessResponse(
        needs_context  = needs_context,
        context        = context,
        combined_prompt= combined,
        turn_index     = turn_index,
    )


@app.post(
    "/v1/sessions/{session_id}/reply",
    response_model=ReplyResponse,
    summary="Log the assistant's reply",
    description=(
        "Call this once your own LLM has generated its response to the "
        "user's message. Stores the reply in the session history so that "
        "future `/v1/process` calls in this session have accurate context "
        "to work with.\n\n"
        "**Always call this after every `/v1/process` call** — skipping it "
        "means the next turn's context will be incomplete (the assistant "
        "side of the conversation will be missing from the history)."
    ),
    tags=["Context"],
    dependencies=[Depends(_require_models_ready), Depends(require_auth)],
)
def log_reply(session_id: str, body: ReplyRequest) -> ReplyResponse:
    turn_index = write_turn(session_id, "assistant", body.text)

    logger.info(
        "REPLY_LOGGED session=%s turn=%d",
        session_id,
        turn_index,
    )

    return ReplyResponse(turn_index=turn_index)
