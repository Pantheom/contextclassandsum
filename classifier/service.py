"""
classifier/service.py
---------------------
Core classifier logic: two public functions.

classify(uid, current_prompt) -> bool
    - Fetches the user's running summary from context_classifier (Supabase).
    - Fetches last N turns from chat_history (Supabase, read-only).
    - Builds a compact Gemma 3 prompt and calls the model.
    - Parses the YES/NO response with a fail-safe bias toward True:
        raw.strip().upper() == "NO"  -> False  (unambiguous NO only)
        anything else                -> True   (YES, empty, garbage = needs context)
    - Logs the classification event.

get_response_context(uid, current_prompt) -> dict
    - Calls classify(). If False: returns {"needs_context": False, "context": None}.
    - If True: calls summarize_on_demand() (synchronous), fetches last
      context_turns turns, formats a context block, returns it.
    - Does NOT call write_turn. Does NOT generate a user-facing answer.
      Responsibility ends at returning the context dict.

DB access: read-only via Supabase, using the same client as the summarizer.
All writes remain the summarizer's exclusive responsibility.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Summarizer imports — public API only.
# ---------------------------------------------------------------------------
from summarizer import summarize_on_demand          # patchable in tests
from summarizer.db import (
    get_supabase_client,
    get_session,
    get_last_n_turns,
    TurnRow,
)

from .config import cfg
from .logging_cfg import get_logger, log_classification_event
from .model import run_inference                    # patchable in tests
from .prompt import build_classifier_prompt, format_history

logger = get_logger("classifier.service")


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def classify(uid: str, current_prompt: str) -> bool:
    """Classify whether current_prompt requires prior context.

    Returns:
        True  — prior context should be fetched (YES or ambiguous output).
        False — prompt is self-contained (unambiguous "NO" only).
    """
    client = get_supabase_client()

    # ------------------------------------------------------------------ #
    # 1. Fetch running summary from context_classifier                    #
    # ------------------------------------------------------------------ #
    session = get_session(client, uid)
    summary: Optional[str] = session.current_summary if session else None

    # ------------------------------------------------------------------ #
    # 2. Fetch recent history turns from chat_history (read-only)         #
    # ------------------------------------------------------------------ #
    history_turns: List[TurnRow] = get_last_n_turns(client, uid, cfg.history_turns)

    # ------------------------------------------------------------------ #
    # 3. Build compact prompt and call model                              #
    # ------------------------------------------------------------------ #
    prompt = build_classifier_prompt(
        history_turns=history_turns,
        current_prompt=current_prompt,
        summary=summary,
    )

    logger.debug(
        "Classifying uid=%s history_turns=%d has_summary=%s",
        uid,
        len(history_turns),
        summary is not None,
    )

    raw_output = run_inference(prompt)

    # ------------------------------------------------------------------ #
    # 4. Parse — fail-safe: only unambiguous "NO" suppresses context      #
    # ------------------------------------------------------------------ #
    # With GBNF grammar active, raw_output is guaranteed to be "YES" or "NO".
    # In the grammar-unavailable fallback, raw_output may be anything — the
    # strip().upper() normalisation followed by the != "NO" test still gives
    # the correct fail-safe behaviour: wrong-case, whitespace, or garbage all
    # resolve to True (fetch context).
    verdict: bool = raw_output.strip().upper() != "NO"

    # ------------------------------------------------------------------ #
    # 5. Log                                                              #
    # ------------------------------------------------------------------ #
    log_classification_event(
        session_id=uid,
        verdict=verdict,
        raw_output=raw_output,
        prompt_preview=current_prompt,
    )

    return verdict


# ---------------------------------------------------------------------------
# Orchestration — main entry point
# ---------------------------------------------------------------------------

def get_response_context(
    uid: str,
    current_prompt: str,
) -> Dict[str, object]:
    """Classify the prompt and, if context is needed, fetch and format it.

    Returns a dict with two keys:
        "needs_context" : bool
        "context"       : str | None
            None when needs_context is False.
            A formatted string block when needs_context is True.

    This function:
    - Does NOT call write_turn.
    - Does NOT generate a user-facing answer.
    - Responsibility ends at returning the context dict — wiring the context
      into an actual LLM call is the answering pipeline's job.

    Args:
        uid:            The user to classify and optionally fetch context for.
        current_prompt: The new user message to classify.
    """
    needs_context = classify(uid, current_prompt)

    if not needs_context:
        return {"needs_context": False, "context": None}

    # ------------------------------------------------------------------ #
    # Context needed — fetch summary + recent turns                       #
    # ------------------------------------------------------------------ #

    # summarize_on_demand is synchronous and may trigger the model.
    # It updates last_summarized_message_id in the DB, which is intentional:
    # we want the freshest possible summary before injecting context.
    summary: str = summarize_on_demand(uid)

    client = get_supabase_client()
    context_turns: List[TurnRow] = get_last_n_turns(client, uid, cfg.context_turns)

    context_block = _format_context_block(summary, context_turns)

    logger.debug(
        "Context block assembled for uid=%s: summary_chars=%d, recent_turns=%d",
        uid,
        len(summary),
        len(context_turns),
    )

    return {"needs_context": True, "context": context_block}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_context_block(summary: str, turns: List[TurnRow]) -> str:
    """Format summary + recent turns into a single injectable context string.

    The format is intentionally minimal: plain Markdown headings so any
    downstream model can parse it reliably without specialised handling.

    Example output:
        ## Prior Context
        The user asked about the project deadline. The assistant confirmed
        it is Friday the 22nd.

        ## Recent Turns
        [U] And what about the budget?
        [A] The budget is $50k.
    """
    summary_text = (
        summary.strip()
        if summary and summary.strip()
        else "(no summary available yet)"
    )
    turns_text = (
        format_history(turns)
        if turns
        else "(no recent turns)"
    )
    return (
        f"## Prior Context\n{summary_text}\n\n"
        f"## Recent Turns\n{turns_text}"
    )


# ---------------------------------------------------------------------------
# Debug variant — for observability tooling only
# ---------------------------------------------------------------------------

def classify_debug(uid: str, current_prompt: str) -> dict:
    """Debug-only variant of classify() that also surfaces the raw model output.

    Identical logic to classify() — does NOT change classify()'s signature or
    contract.  Use only from debug/testing tooling (e.g. the debug UI or test
    scripts); production code should call classify() directly.

    Returns:
        {"needs_context": bool, "raw_output": str}
    """
    client = get_supabase_client()

    session = get_session(client, uid)
    summary: Optional[str] = session.current_summary if session else None

    history_turns: List[TurnRow] = get_last_n_turns(client, uid, cfg.history_turns)

    prompt = build_classifier_prompt(
        history_turns=history_turns,
        current_prompt=current_prompt,
        summary=summary,
    )
    raw_output = run_inference(prompt)
    verdict: bool = raw_output.strip().upper() != "NO"

    log_classification_event(
        session_id=uid,
        verdict=verdict,
        raw_output=raw_output,
        prompt_preview=current_prompt,
    )

    return {"needs_context": verdict, "raw_output": raw_output}
