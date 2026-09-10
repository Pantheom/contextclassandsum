"""
summarizer/service.py
---------------------
Core summarizer logic and both entry points.

Entry Point 1 — Periodic (automatic, non-blocking):
    write_turn(uid, role, text) -> int
    After every new turn is written, checks whether the count of unsummarized
    messages for this uid has reached PERIODIC_THRESHOLD.
    If so, submits regenerate_summary() to a module-level ThreadPoolExecutor
    and returns immediately without blocking the caller.

Entry Point 2 — On-demand (synchronous, external caller):
    summarize_on_demand(uid) -> str
    Calls regenerate_summary() directly and blocks until the summary is ready.

Both entry points funnel through the single source of truth:
    regenerate_summary(uid) -> str

Design invariants:
- last_summarized_message_id is ONLY written by db.update_summary(), which is
  ONLY called by regenerate_summary(). No other code path may touch it.
- A per-uid threading.Lock prevents two concurrent background jobs from
  summarizing the same user simultaneously. The second job is silently
  skipped (the first will have advanced the pointer; the next write_turn()
  will trigger again if the count grows back to threshold).
- A single Supabase client is shared across threads — the supabase-py client
  is thread-safe for concurrent reads and writes.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Optional

from supabase import Client

from .config import cfg
from .db import (
    get_supabase_client,
    get_or_create_session,
    get_turns_after,
    count_unsummarized_turns,
    insert_turn,
    update_summary,
)
from .logging_cfg import get_logger, log_summarization_event
from .model import run_inference
from .prompt import build_prompt

logger = get_logger("summarizer.service")

# ---------------------------------------------------------------------------
# Module-level shared state
# ---------------------------------------------------------------------------

# Single Supabase client — created once at init_service() time.
_supabase_client: Optional[Client] = None
_client_lock = threading.Lock()

# Single-worker pool: background summarisation jobs run sequentially, which is
# correct because the single Llama instance serialises inference anyway.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="summarizer-bg")

# Per-uid locks to prevent two concurrent background jobs for the same user.
# Access to this dict is protected by _uid_locks_lock.
_uid_locks: Dict[str, threading.Lock] = {}
_uid_locks_lock = threading.Lock()


def _get_client() -> Client:
    """Return the module-level Supabase client, creating it if necessary."""
    global _supabase_client
    with _client_lock:
        if _supabase_client is None:
            _supabase_client = get_supabase_client()
    return _supabase_client


def _get_uid_lock(uid: str) -> threading.Lock:
    """Return (creating if necessary) the per-uid Lock."""
    with _uid_locks_lock:
        if uid not in _uid_locks:
            _uid_locks[uid] = threading.Lock()
        return _uid_locks[uid]


# ---------------------------------------------------------------------------
# Core summarisation function — single source of truth
# ---------------------------------------------------------------------------

def regenerate_summary(uid: str, trigger: str = "on-demand") -> str:
    """Fetch new turns, call the model, and atomically update the DB.

    This is the ONLY function that calls db.update_summary(), and therefore
    the ONLY place that advances last_summarized_message_id.

    Args:
        uid:     The user UUID to summarize.
        trigger: 'periodic' or 'on-demand' — used only for logging.

    Returns:
        The new consolidated summary string.

    Raises:
        RuntimeError: If the Supabase client is not configured, or if the
                      model is not loaded (propagated from model.py).
    """
    client = _get_client()

    # ------------------------------------------------------------------ #
    # 1. Read current state                                                #
    # ------------------------------------------------------------------ #
    session = get_or_create_session(client, uid)
    previous_summary: Optional[str] = session.current_summary
    since_id: int = session.last_summarized_message_id

    # ------------------------------------------------------------------ #
    # 2. Fetch turns to include                                            #
    # ------------------------------------------------------------------ #
    # Fetch all messages strictly after the last summarized message id.
    new_turns = get_turns_after(client, uid, since_id)

    if not new_turns:
        logger.info(
            "regenerate_summary called for uid=%s but no new turns "
            "(last_summarized_message_id=%d). Returning existing summary.",
            uid,
            since_id,
        )
        return previous_summary or ""

    first_idx = new_turns[0].turn_index
    last_idx = new_turns[-1].turn_index
    last_message_id = new_turns[-1].turn_id

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
        "Calling model for uid=%s, turns %d-%d (trigger=%s)",
        uid, first_idx, last_idx, trigger,
    )

    new_summary = run_inference(prompt)

    # Guard against a degenerate empty response.
    if not new_summary:
        logger.warning(
            "Model returned an empty summary for uid=%s. "
            "Retaining previous summary.",
            uid,
        )
        return previous_summary or ""

    # ------------------------------------------------------------------ #
    # 4. Atomically persist — only write path for last_summarized_message_id
    # ------------------------------------------------------------------ #
    update_summary(client, uid, new_summary, last_message_id)

    # ------------------------------------------------------------------ #
    # 5. Structured log                                                    #
    # ------------------------------------------------------------------ #
    log_summarization_event(
        session_id=uid,
        trigger=trigger,
        turn_range=(first_idx, last_idx),
        summary_length=len(new_summary),
    )

    return new_summary


# ---------------------------------------------------------------------------
# Entry Point 1 — Periodic / automatic
# ---------------------------------------------------------------------------

def write_turn(uid: str, role: str, text: str) -> int:
    """Write a conversation turn and fire background summarisation if needed.

    This is the primary write path.  It:
      1. Ensures the context_classifier row exists for this uid.
      2. Inserts the message into chat_history, receiving its DB-generated id.
      3. Counts the number of unsummarized messages for this uid.
      4. If the count >= periodic_threshold, submits a background job.

    Args:
        uid:  User UUID.
        role: 'user' or 'assistant'.
        text: Turn content.

    Returns:
        The chat_history.id assigned to this message (auto-generated bigint).
    """
    client = _get_client()

    # Ensure context_classifier row exists before inserting the turn.
    get_or_create_session(client, uid)

    new_message_id = insert_turn(client, uid, role, text)
    logger.debug(
        "Inserted turn uid=%s role=%s message_id=%d",
        uid, role, new_message_id,
    )

    # Read the current last_summarized_message_id (a concurrent bg job may
    # have advanced it between the session creation above and now).
    from .db import get_session  # local import to avoid circular at module level
    session = get_session(client, uid)
    last_summarized = session.last_summarized_message_id  # type: ignore[union-attr]

    # Count actual unsummarized rows for this uid to avoid false positives
    # from non-contiguous global identity values.
    unsummarized_count = count_unsummarized_turns(client, uid, last_summarized)
    logger.debug(
        "uid=%s unsummarized_count=%d threshold=%d",
        uid, unsummarized_count, cfg.periodic_threshold,
    )

    # ------------------------------------------------------------------ #
    # Fire background job if threshold reached                             #
    # ------------------------------------------------------------------ #
    if unsummarized_count >= cfg.periodic_threshold:
        _submit_background_job(uid)

    return new_message_id


def _submit_background_job(uid: str) -> Optional[Future]:  # type: ignore[type-arg]
    """Submit a background regenerate_summary() job for uid.

    Uses a non-blocking tryacquire on the per-uid lock.  If the lock is
    already held (a previous job for this uid is still running), the new
    job is silently skipped — the running job will advance the pointer; the
    count will drop below threshold; the next write_turn() will re-trigger
    when the count grows again.
    """
    uid_lock = _get_uid_lock(uid)

    if not uid_lock.acquire(blocking=False):
        logger.debug(
            "Background job for uid=%s already in progress; skipping.",
            uid,
        )
        return None

    def _job() -> str:
        try:
            return regenerate_summary(uid, trigger="periodic")
        except Exception:
            logger.exception(
                "Background summarization failed for uid=%s", uid
            )
            return ""
        finally:
            uid_lock.release()

    future: Future = _executor.submit(_job)  # type: ignore[type-arg]
    logger.info(
        "Submitted background summarization job for uid=%s", uid
    )
    return future


# ---------------------------------------------------------------------------
# Entry Point 2 — On-demand (synchronous, callable by external systems)
# ---------------------------------------------------------------------------

def summarize_on_demand(uid: str) -> str:
    """Synchronously regenerate the summary for uid.

    Called by external systems (e.g. the classifier component).  Blocks until
    the summary is ready and returns it.  Uses the identical regenerate_summary()
    path as Entry Point 1 — same semantics, same DB updates.

    Calling this naturally resets the unsummarised-turns counter that Entry
    Point 1 checks (last_summarized_message_id advances), so the periodic
    threshold is effectively reset without any separate counter.

    Args:
        uid: The user UUID to summarize.

    Returns:
        The new consolidated summary string.
    """
    logger.info("On-demand summarization requested for uid=%s", uid)
    return regenerate_summary(uid, trigger="on-demand")
