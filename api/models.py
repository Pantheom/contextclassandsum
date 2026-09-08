"""
api/models.py
-------------
Pydantic request/response models for the production API, plus the
build_combined_prompt helper.

Every field carries a `description=` so the /docs page is self-explanatory
to a backend engineer who has never seen this codebase.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Session-ID and text field constraints (mirrored in validators below)
# ---------------------------------------------------------------------------

_SESSION_ID_PATTERN = r"^[a-zA-Z0-9_-]+$"
_MAX_TEXT_LEN       = 8_000
_MAX_SESSION_ID_LEN = 200


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ProcessRequest(BaseModel):
    """Payload for POST /v1/process — one call per incoming user message."""

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_SESSION_ID_LEN,
        pattern=_SESSION_ID_PATTERN,
        description=(
            "A stable, caller-assigned identifier for this conversation. "
            "Must be 1–200 characters, using only letters, digits, hyphens, "
            "and underscores. Use the same value across all turns of the same "
            "conversation — history accumulates per session_id."
        ),
        examples=["user-42-conv-7", "session_abc123"],
    )

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_TEXT_LEN,
        description=(
            "The user's latest message, exactly as received from your frontend. "
            "Maximum 8 000 characters. The API will determine whether prior "
            "conversation context is needed and return a `combined_prompt` that "
            "is ready to send to your answering LLM."
        ),
        examples=["What was the budget we agreed on last week?"],
    )


class ReplyRequest(BaseModel):
    """Payload for POST /v1/sessions/{session_id}/reply."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_TEXT_LEN,
        description=(
            "The assistant's reply, verbatim. Call this endpoint once your own "
            "LLM has generated its response so that the conversation history "
            "stays complete for future /v1/process calls in this session."
        ),
        examples=["The budget was $50 000, as agreed in the kickoff meeting."],
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ProcessResponse(BaseModel):
    """Response from POST /v1/process."""

    needs_context: bool = Field(
        description=(
            "True if prior conversation context was injected into "
            "`combined_prompt`. False if the prompt was self-contained and "
            "no context was needed."
        )
    )

    context: Optional[str] = Field(
        description=(
            "The raw context block that was prepended to the prompt (a rolling "
            "summary of prior turns + the N most recent turns), or null if "
            "`needs_context` is False. Provided for debugging and logging; "
            "the ready-to-use value is `combined_prompt`."
        )
    )

    combined_prompt: str = Field(
        description=(
            "Send this directly to your answering LLM as-is. "
            "If context was needed this is: "
            "{context_block}\\n\\n---\\n\\nUser's current message: {prompt}. "
            "If no context was needed this equals the original `prompt` unchanged."
        )
    )

    turn_index: int = Field(
        description=(
            "The sequential turn index the user's message was stored as in the "
            "session history. Useful for debugging ordering and gaps."
        )
    )


class ReplyResponse(BaseModel):
    """Response from POST /v1/sessions/{session_id}/reply."""

    turn_index: int = Field(
        description="The sequential turn index the assistant reply was stored as."
    )


class HealthResponse(BaseModel):
    """Response from GET /v1/health."""

    status: str = Field(
        description='Always "ok" when the process is reachable.'
    )

    models_loaded: bool = Field(
        description=(
            "True when both the summarizer and classifier models have finished "
            "loading and the API is ready to serve /v1/process requests. "
            "False during the initial model-loading window after startup — "
            "retry /v1/process after a few seconds."
        )
    )


# ---------------------------------------------------------------------------
# Prompt builder — single source of truth for the combined format
# ---------------------------------------------------------------------------

def build_combined_prompt(context_block: str, prompt: str) -> str:
    """Merge a context block and the current user prompt into one LLM-ready string.

    The separator (---) is intentionally unambiguous so the downstream LLM
    can parse or ignore the context section reliably.

    Format:
        {context_block}

        ---

        User's current message: {prompt}

    Args:
        context_block: The formatted context string produced by
                       classifier.get_response_context (contains rolling
                       summary + recent turns in Markdown).
        prompt:        The user's raw incoming message.

    Returns:
        A single string ready to send to the answering LLM. Never empty.
    """
    return f"{context_block}\n\n---\n\nUser's current message: {prompt}"
