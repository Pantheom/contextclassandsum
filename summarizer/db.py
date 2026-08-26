"""
summarizer/db.py
----------------
All SQLite access for the summarizer service.

Design rules:
- Every public function accepts an open sqlite3.Connection; callers are
  responsible for opening/closing connections.  This keeps transaction
  control explicit and avoids hidden connection state.
- WAL mode is enabled on init so concurrent readers don't block the writer.
- All writes use parameterised queries — no string interpolation.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional


# ---------------------------------------------------------------------------
# Row types (lightweight value objects — no ORM)
# ---------------------------------------------------------------------------

@dataclass
class SessionRow:
    session_id: str
    current_summary: Optional[str]
    last_summarized_turn_index: int
    updated_at: Optional[str]


@dataclass
class TurnRow:
    turn_id: int
    session_id: str
    turn_index: int
    role: str           # 'user' | 'assistant'
    text: str
    timestamp: str


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sessions (
    session_id                  TEXT    PRIMARY KEY,
    current_summary             TEXT,
    last_summarized_turn_index  INTEGER NOT NULL DEFAULT 0,
    updated_at                  TEXT
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL REFERENCES sessions(session_id),
    turn_index  INTEGER NOT NULL,
    role        TEXT    NOT NULL CHECK(role IN ('user', 'assistant')),
    text        TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(session_id, turn_index)
);

CREATE INDEX IF NOT EXISTS idx_turns_session_index
    ON turns(session_id, turn_index);
"""


def init_db(conn: sqlite3.Connection) -> None:
    """Create tables and indexes if they do not already exist.

    Safe to call on every startup — all statements are idempotent.
    """
    conn.executescript(_SCHEMA_SQL)
    conn.commit()


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

def open_connection(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with sensible defaults.

    - `check_same_thread=False` because the ThreadPoolExecutor background
      worker runs on a different thread than the writer.  Each code path
      that calls the DB must use its own connection (or hold the service-level
      lock) — we do not share a single connection across threads.
    - Row factory set to sqlite3.Row for named-column access.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def get_session(conn: sqlite3.Connection, session_id: str) -> Optional[SessionRow]:
    """Return the session row, or None if it does not exist."""
    row = conn.execute(
        "SELECT session_id, current_summary, last_summarized_turn_index, updated_at "
        "FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return SessionRow(
        session_id=row["session_id"],
        current_summary=row["current_summary"],
        last_summarized_turn_index=row["last_summarized_turn_index"],
        updated_at=row["updated_at"],
    )


def get_or_create_session(conn: sqlite3.Connection, session_id: str) -> SessionRow:
    """Return existing session or insert a fresh one with defaults."""
    session = get_session(conn, session_id)
    if session is not None:
        return session

    conn.execute(
        "INSERT OR IGNORE INTO sessions "
        "(session_id, current_summary, last_summarized_turn_index, updated_at) "
        "VALUES (?, NULL, 0, ?)",
        (session_id, _utcnow()),
    )
    conn.commit()
    return get_session(conn, session_id)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Turn helpers
# ---------------------------------------------------------------------------

def insert_turn(
    conn: sqlite3.Connection,
    session_id: str,
    role: str,
    text: str,
) -> int:
    """Insert a new turn and return its monotonically increasing turn_index.

    turn_index is computed as MAX(turn_index) + 1 for the session, atomically
    within a transaction so concurrent inserts cannot collide.
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")

    # Ensure session row exists before inserting the turn.
    get_or_create_session(conn, session_id)

    with conn:  # BEGIN / COMMIT / ROLLBACK on exit
        row = conn.execute(
            "SELECT COALESCE(MAX(turn_index), 0) AS max_idx "
            "FROM turns WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        next_index: int = row["max_idx"] + 1

        conn.execute(
            "INSERT INTO turns (session_id, turn_index, role, text, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, next_index, role, text, _utcnow()),
        )

    return next_index


def get_latest_turn_index(conn: sqlite3.Connection, session_id: str) -> int:
    """Return the highest turn_index for the session, or 0 if none."""
    row = conn.execute(
        "SELECT COALESCE(MAX(turn_index), 0) AS max_idx "
        "FROM turns WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return int(row["max_idx"])


def get_turns_after(
    conn: sqlite3.Connection,
    session_id: str,
    after_index: int,
) -> List[TurnRow]:
    """Return all turns where turn_index > after_index, ordered ascending.

    Overlap-safe: the boundary is strictly greater than, so if the caller
    passes `last_summarized_turn_index` it naturally includes any turn that
    lands exactly at the boundary in the *next* window (never skips).
    """
    rows = conn.execute(
        "SELECT turn_id, session_id, turn_index, role, text, timestamp "
        "FROM turns "
        "WHERE session_id = ? AND turn_index > ? "
        "ORDER BY turn_index ASC",
        (session_id, after_index),
    ).fetchall()
    return [
        TurnRow(
            turn_id=r["turn_id"],
            session_id=r["session_id"],
            turn_index=r["turn_index"],
            role=r["role"],
            text=r["text"],
            timestamp=r["timestamp"],
        )
        for r in rows
    ]


def get_last_n_turns(
    conn: sqlite3.Connection,
    session_id: str,
    n: int,
) -> List[TurnRow]:
    """Return the most recent n turns for session_id, in ascending (chronological) order.

    If the session has fewer than n turns, all available turns are returned.
    If the session has no turns at all, an empty list is returned — never raises
    IndexError or any other error due to an under-populated session.

    Uses ORDER BY turn_index DESC LIMIT n to efficiently fetch only the tail
    of the turns table, then reverses to restore ASC (chronological) order
    suitable for display and prompt construction.
    """
    if n <= 0:
        return []
    rows = conn.execute(
        "SELECT turn_id, session_id, turn_index, role, text, timestamp "
        "FROM turns "
        "WHERE session_id = ? "
        "ORDER BY turn_index DESC "
        "LIMIT ?",
        (session_id, n),
    ).fetchall()
    # reversed() restores chronological ASC order without a second sort pass.
    return [
        TurnRow(
            turn_id=r["turn_id"],
            session_id=r["session_id"],
            turn_index=r["turn_index"],
            role=r["role"],
            text=r["text"],
            timestamp=r["timestamp"],
        )
        for r in reversed(rows)
    ]


def get_all_turns(
    conn: sqlite3.Connection,
    session_id: str,
) -> List[TurnRow]:
    """Return every turn for session_id in ascending (chronological) order.

    Convenience wrapper around get_turns_after with after_index=0.
    Used by the debug UI to display the full conversation history.
    """
    return get_turns_after(conn, session_id, after_index=0)


def get_all_session_ids(conn: sqlite3.Connection) -> List[str]:
    """Return all distinct session_ids, ordered by most recently updated first.

    Used by the debug UI to populate the session picker dropdown.
    Returns an empty list if no sessions exist yet.
    """
    rows = conn.execute(
        "SELECT session_id FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    return [r["session_id"] for r in rows]


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def update_summary(
    conn: sqlite3.Connection,
    session_id: str,
    new_summary: str,
    last_turn_index: int,
) -> None:
    """Atomically replace the session's summary and advance the turn pointer.

    This is the ONLY place in the codebase that writes
    last_summarized_turn_index.  All summarization paths funnel through here.
    """
    with conn:
        conn.execute(
            "UPDATE sessions "
            "SET current_summary = ?, "
            "    last_summarized_turn_index = ?, "
            "    updated_at = ? "
            "WHERE session_id = ?",
            (new_summary, last_turn_index, _utcnow(), session_id),
        )


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
