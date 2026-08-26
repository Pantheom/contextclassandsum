"""
summarizer/__init__.py
-----------------------
Public API surface for the summarizer package.

External callers (e.g. the classifier component) should import from here,
not from individual sub-modules, so internal refactors don't break callers.

Usage:
    from summarizer import write_turn, summarize_on_demand, init_service
"""

from .logging_cfg import configure_logging
from .db import init_db, open_connection, get_last_n_turns, TurnRow, get_all_turns, get_all_session_ids
from .config import cfg
from .service import write_turn, summarize_on_demand, regenerate_summary


def init_service(db_path: str | None = None) -> None:
    """Initialise logging and the database schema.

    Call this once at application startup before any write_turn() or
    summarize_on_demand() calls.

    Args:
        db_path: Override the database path (defaults to cfg.db_path).
                 Useful in tests to point at a temp file.
    """
    configure_logging()
    path = db_path or cfg.db_path
    conn = open_connection(path)
    try:
        init_db(conn)
    finally:
        conn.close()


__all__ = [
    "init_service",
    "write_turn",
    "summarize_on_demand",
    "regenerate_summary",
    "cfg",
    # DB helpers exposed for read-only consumers (e.g. the classifier, debug UI)
    "open_connection",
    "get_last_n_turns",
    "get_all_turns",
    "get_all_session_ids",
    "TurnRow",
]
