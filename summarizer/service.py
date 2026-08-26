"""
summarizer/service.py
---------------------
Core summarizer logic and both entry points.

Entry Point 1 — Periodic (automatic, non-blocking):
    write_turn(session_id, role, text) -> int
    After every new turn is written, checks whether
    (latest_turn_index - last_summarized_turn_index) >= PERIODIC_THRESHOLD.
    If so, submits regenerate_summary() to a module-level ThreadPoolExecutor
    and returns immediately without blocking the caller.

Entry Point 2 — On-demand (synchronous, external caller):
    summarize_on_demand(session_id) -> str
    Calls regenerate_summary() directly and blocks until the summary is ready.

Both entry points funnel through the single source of truth:
    regenerate_summary(session_id) -> str

Design invariants:
- last_summarized_turn_index is ONLY written by db.update_summary(), which is
  ONLY called by regenerate_summary(). No other code path may touch it.
- A per-session threading.Lock prevents two concurrent background jobs from
  summarizing the same session simultaneously. The second job is silently
  skipped (the first will have advanced the pointer; the next write_turn()
  will trigger again if the gap grows back to threshold).
- Each DB operation opens its own connection and closes it on exit, because
  sqlite3 connections must not be shared across threads.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Optional

from .config import cfg
from .db import (
    get_or_create_session,
    get_turns_after,
    get_latest_turn_index,
    insert_turn,
    open_connection,
    update_summary,
)
from .logging_cfg import get_logger, log_summarization_event
from .model import run_inference
from .prompt import build_prompt

logger = get_logger("summarizer.service")

# ---------------------------------------------------------------------------
# Module-level shared state
# ---------------------------------------------------------------------------

# Single-worker pool: background summarisation jobs run sequentially, which is
# correct because the single Llama instance serialises inference anyway.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="summarizer-bg")

# Per-session locks to prevent two concurrent background jobs for the same
# session.  Access to this dict is protected by _session_locks_lock.
_session_locks: Dict[str, threading.Lock] = {}
_session_locks_lock = threading.Lock()


def _get_session_lock(session_id: str) -> threading.Lock:
    """Return (creating if necessary) the per-session Lock."""
    with _session_locks_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = threading.Lock()
        return _session_locks[session_id]


# ---------------------------------------------------------------------------
# Core summarisation function — single source of truth
# ---------------------------------------------------------------------------

def regenerate_summary(session_id: str, trigger: str = "on-demand") -> str:
    """Fetch new turns, call the model, and atomically update the DB.

    This is the ONLY function that calls db.update_summary(), and therefore
    the ONLY place that advances last_summarized_turn_index.

    Args:
        session_id: The session to summarize.
        trigger:    'periodic' or 'on-demand' — used only for logging.

    Returns:
        The new consolidated summary string.

    Raises:
        ValueError: If session_id does not exist in the database.
        RuntimeError: If the model is not loaded (propagated from model.py).
    """
    conn = open_connection(cfg.db_path)
    try:
        # ------------------------------------------------------------------ #
        # 1. Read current state                                                #
        # ------------------------------------------------------------------ #
        session = get_or_create_session(conn, session_id)
        previous_summary: Optional[str] = session.current_summary
        since_index: int = session.last_summarized_turn_index

        # ------------------------------------------------------------------ #
        # 2. Fetch turns to include                                            #
        # ------------------------------------------------------------------ #
        # Overlap-safe: fetch strictly after since_index.
        # If since_index is 0 and there's a non-empty previous_summary (edge
        # case: externally written), we still fetch all turns — the model will
        # reconcile them with the provided previous_summary.
        new_turns = get_turns_after(conn, session_id, since_index)

        if not new_turns:
            logger.info(
                "regenerate_summary called for session=%s but no new turns "
                "(last_summarized_turn_index=%d). Returning existing summary.",
                session_id,
                since_index,
            )
            return previous_summary or ""

        first_idx = new_turns[0].turn_index
        last_idx = new_turns[-1].turn_index

        # ------------------------------------------------------------------ #
        # 3. Build prompt and call model                                       #
        # ------------------------------------------------------------------ #
        prompt = build_prompt(
            previous_summary=previous_summary,
            turns=new_turns,
            first_idx=first_idx,
            last_idx=last_idx,
        )

        logger.debug(
            "Calling model for session=%s, turns %d-%d (trigger=%s)",
            session_id, first_idx, last_idx, trigger,
        )

        new_summary = run_inference(prompt)

        # Guard against a degenerate empty response.
        if not new_summary:
            logger.warning(
                "Model returned an empty summary for session=%s. "
                "Retaining previous summary.",
                session_id,
            )
            return previous_summary or ""

        # ------------------------------------------------------------------ #
        # 4. Atomically persist — only write path for last_summarized_turn_index
        # ------------------------------------------------------------------ #
        update_summary(conn, session_id, new_summary, last_idx)

        # ------------------------------------------------------------------ #
        # 5. Structured log                                                    #
        # ------------------------------------------------------------------ #
        log_summarization_event(
            session_id=session_id,
            trigger=trigger,
            turn_range=(first_idx, last_idx),
            summary_length=len(new_summary),
        )

        return new_summary

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Entry Point 1 — Periodic / automatic
# ---------------------------------------------------------------------------

def write_turn(session_id: str, role: str, text: str) -> int:
    """Write a conversation turn and fire background summarisation if needed.

    This is the primary write path.  It:
      1. Ensures the session row exists.
      2. Inserts the turn (atomically computing the next turn_index).
      3. Checks whether the unsummarised gap has reached the threshold.
      4. If so, submits a background summarisation job — non-blocking.

    Args:
        session_id: Target session.
        role:       'user' or 'assistant'.
        text:       Turn content.

    Returns:
        The turn_index assigned to this turn (monotonically increasing per session).
    """
    conn = open_connection(cfg.db_path)
    try:
        # Ensure session exists before inserting the turn.
        get_or_create_session(conn, session_id)

        new_turn_index = insert_turn(conn, session_id, role, text)
        logger.debug(
            "Inserted turn session=%s role=%s turn_index=%d",
            session_id, role, new_turn_index,
        )

        # Reload session to get the current last_summarized_turn_index.
        # (We need a fresh read because a concurrent bg job may have advanced
        # it between the session creation above and now.)
        from .db import get_session  # local import to avoid circular at module level
        session = get_session(conn, session_id)
        last_summarized = session.last_summarized_turn_index  # type: ignore[union-attr]

        gap = new_turn_index - last_summarized
        logger.debug(
            "session=%s gap=%d threshold=%d",
            session_id, gap, cfg.periodic_threshold,
        )

    finally:
        conn.close()

    # ------------------------------------------------------------------ #
    # Fire background job if threshold reached                             #
    # ------------------------------------------------------------------ #
    if gap >= cfg.periodic_threshold:
        _submit_background_job(session_id)

    return new_turn_index


def _submit_background_job(session_id: str) -> Optional[Future]:  # type: ignore[type-arg]
    """Submit a background regenerate_summary() job for session_id.

    Uses a non-blocking tryacquire on the per-session lock.  If the lock is
    already held (a previous job for this session is still running), the new
    job is silently skipped — the running job will advance the pointer; the
    gap will shrink below threshold; the next write_turn() will re-trigger
    when the gap grows again.
    """
    session_lock = _get_session_lock(session_id)

    if not session_lock.acquire(blocking=False):
        logger.debug(
            "Background job for session=%s already in progress; skipping.",
            session_id,
        )
        return None

    def _job() -> str:
        try:
            return regenerate_summary(session_id, trigger="periodic")
        except Exception:
            logger.exception(
                "Background summarization failed for session=%s", session_id
            )
            return ""
        finally:
            session_lock.release()

    future: Future = _executor.submit(_job)  # type: ignore[type-arg]
    logger.info(
        "Submitted background summarization job for session=%s", session_id
    )
    return future


# ---------------------------------------------------------------------------
# Entry Point 2 — On-demand (synchronous, callable by external systems)
# ---------------------------------------------------------------------------

def summarize_on_demand(session_id: str) -> str:
    """Synchronously regenerate the summary for session_id.

    Called by external systems (e.g. the classifier component).  Blocks until
    the summary is ready and returns it.  Uses the identical regenerate_summary()
    path as Entry Point 1 — same semantics, same DB updates.

    Calling this naturally resets the unsummarised-turns counter that Entry
    Point 1 checks (last_summarized_turn_index advances), so the periodic
    threshold is effectively reset without any separate counter.

    Args:
        session_id: The session to summarize.

    Returns:
        The new consolidated summary string.
    """
    logger.info("On-demand summarization requested for session=%s", session_id)
    return regenerate_summary(session_id, trigger="on-demand")
