"""
summarizer/db.py
----------------
All Supabase (PostgreSQL) access for the summarizer service.

Design rules:
- Every public function accepts a supabase.Client instance; callers are
  responsible for creating the client (typically once at startup).  This keeps
  connection control explicit and avoids hidden state.
- All queries are keyed on uid (uuid) — session_id is present in the schema
  but is treated as redundant and is never used for reads or writes.
- The 10-message threshold is enforced by counting rows in chat_history for
  the given uid where id > last_summarized_message_id.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from supabase import Client, create_client

from .config import cfg


# ---------------------------------------------------------------------------
# Row types (lightweight value objects — no ORM)
# ---------------------------------------------------------------------------

@dataclass
class SessionRow:
    uid: str
    current_summary: Optional[str]
    last_summarized_message_id: int
    updated_at: Optional[str]


@dataclass
class TurnRow:
    turn_id: int          # maps to chat_history.id
    uid: str
    role: str             # 'user' | 'assistant'
    text: str             # maps to chat_history.message
    timestamp: str        # maps to chat_history.created_at

    # Shim attributes kept for backwards-compatibility with prompt.py /
    # classifier/service.py that still reference turn_index / session_id.
    # They are derived fields, not stored in Supabase.
    turn_index: int = 0
    session_id: str = ""


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def get_supabase_client() -> Client:
    """Create and return a Supabase client using values from cfg.

    Raises:
        RuntimeError: If SUPABASE_URL or SUPABASE_KEY are not configured.
    """
    if not cfg.supabase_url or not cfg.supabase_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY must be set in the environment "
            "before the summarizer service can connect to Supabase."
        )
    return create_client(cfg.supabase_url, cfg.supabase_key)


# ---------------------------------------------------------------------------
# Session helpers  (maps to public.context_classifier)
# ---------------------------------------------------------------------------

def get_session(client: Client, uid: str) -> Optional[SessionRow]:
    """Return the context_classifier row for uid, or None if it doesn't exist."""
    resp = (
        client.table("context_classifier")
        .select("uid, chat_summary, last_summarized_message_id, created_at")
        .eq("uid", uid)
        .maybe_single()
        .execute()
    )
    if resp is None:
        return None
    row = resp.data
    return SessionRow(
        uid=row["uid"],
        current_summary=row["chat_summary"],
        last_summarized_message_id=row["last_summarized_message_id"],
        updated_at=row["created_at"],
    )


def get_or_create_session(client: Client, uid: str) -> SessionRow:
    """Return existing context_classifier row or upsert a fresh one."""
    session = get_session(client, uid)
    if session is not None:
        return session

    client.table("context_classifier").upsert(
        {
            "uid": uid,
            "chat_summary": None,
            "last_summarized_message_id": 0,
        },
        on_conflict="uid",
    ).execute()

    return get_session(client, uid)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Turn helpers  (maps to public.chat_history)
# ---------------------------------------------------------------------------

# Dummy session_id constant used to satisfy the NOT NULL constraint on the
# chat_history.session_id column.  The value is ignored by all query paths.
_DUMMY_SESSION_ID = "00000000-0000-0000-0000-000000000000"


def insert_turn(client: Client, uid: str, role: str, text: str) -> int:
    """Insert a new message into chat_history and return its auto-generated id.

    Args:
        client: Supabase client.
        uid:    User UUID.
        role:   'user' or 'assistant'.
        text:   Message content.

    Returns:
        The bigint identity id assigned by the database.
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")

    resp = (
        client.table("chat_history")
        .insert(
            {
                "uid": uid,
                "session_id": _DUMMY_SESSION_ID,
                "role": role,
                "message": text,
            }
        )
        .execute()
    )
    return int(resp.data[0]["id"])


def get_latest_message_id(client: Client, uid: str) -> int:
    """Return the highest chat_history.id for uid, or 0 if none."""
    resp = (
        client.table("chat_history")
        .select("id")
        .eq("uid", uid)
        .order("id", desc=True)
        .limit(1)
        .execute()
    )
    if not resp.data:
        return 0
    return int(resp.data[0]["id"])


def count_unsummarized_turns(
    client: Client,
    uid: str,
    last_summarized_message_id: int,
) -> int:
    """Count chat_history rows for uid where id > last_summarized_message_id.

    This is the safe way to check the threshold: it counts actual rows for
    this specific user, so a global identity jump caused by other users never
    triggers a false positive.
    """
    resp = (
        client.table("chat_history")
        .select("id", count="exact")
        .eq("uid", uid)
        .gt("id", last_summarized_message_id)
        .execute()
    )
    return resp.count or 0


def get_turns_after(
    client: Client,
    uid: str,
    after_id: int,
) -> List[TurnRow]:
    """Return all turns where id > after_id for uid, ordered by id ascending."""
    resp = (
        client.table("chat_history")
        .select("id, uid, role, message, created_at")
        .eq("uid", uid)
        .gt("id", after_id)
        .order("id", desc=False)
        .execute()
    )
    return [_row_to_turn(r, idx + 1) for idx, r in enumerate(resp.data)]


def get_last_n_turns(client: Client, uid: str, n: int) -> List[TurnRow]:
    """Return the most recent n turns for uid in ascending (chronological) order.

    If the user has fewer than n turns, all available turns are returned.
    """
    if n <= 0:
        return []
    resp = (
        client.table("chat_history")
        .select("id, uid, role, message, created_at")
        .eq("uid", uid)
        .order("id", desc=True)
        .limit(n)
        .execute()
    )
    # Reverse to restore chronological ASC order.
    rows = list(reversed(resp.data))
    return [_row_to_turn(r, idx + 1) for idx, r in enumerate(rows)]


def get_all_turns(client: Client, uid: str) -> List[TurnRow]:
    """Return every turn for uid in ascending (chronological) order.

    Convenience wrapper around get_turns_after with after_id=0.
    Used by the debug UI to display the full conversation history.
    """
    return get_turns_after(client, uid, after_id=0)


def get_all_uids(client: Client) -> List[str]:
    """Return all distinct uids that have context_classifier rows.

    Used by the debug UI to populate the session/user picker dropdown.
    """
    resp = (
        client.table("context_classifier")
        .select("uid")
        .order("created_at", desc=True)
        .execute()
    )
    return [r["uid"] for r in resp.data]


# ---------------------------------------------------------------------------
# Summary writer  (maps to public.context_classifier)
# ---------------------------------------------------------------------------

def update_summary(
    client: Client,
    uid: str,
    new_summary: str,
    last_message_id: int,
) -> None:
    """Atomically replace the user's summary and advance the message pointer.

    Uses upsert on uid (the UNIQUE constraint column) so it works whether
    a row already exists or not.

    This is the ONLY place in the codebase that writes
    last_summarized_message_id.  All summarization paths funnel through here.
    """
    client.table("context_classifier").upsert(
        {
            "uid": uid,
            "chat_summary": new_summary,
            "last_summarized_message_id": last_message_id,
        },
        on_conflict="uid",
    ).execute()


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _row_to_turn(row: dict, turn_index: int) -> TurnRow:
    """Convert a chat_history dict from Supabase into a TurnRow."""
    return TurnRow(
        turn_id=int(row["id"]),
        uid=row["uid"],
        role=row["role"],
        text=row["message"],
        timestamp=row.get("created_at", ""),
        turn_index=turn_index,
        session_id="",
    )
