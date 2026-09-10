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
from .db import (
    get_supabase_client,
    get_session,
    get_last_n_turns,
    get_all_turns,
    get_all_uids,
    TurnRow,
    SessionRow,
)
from .config import cfg
from .service import write_turn, summarize_on_demand, regenerate_summary


def init_service() -> None:
    """Initialise logging and verify the Supabase connection.

    Call this once at application startup before any write_turn() or
    summarize_on_demand() calls.

    Raises:
        RuntimeError: If SUPABASE_URL or SUPABASE_KEY are not configured.
    """
    configure_logging()
    # Eagerly create the client to surface misconfiguration at startup time
    # rather than on the first DB call.
    from .service import _get_client
    _get_client()


# ---------------------------------------------------------------------------
# Backward-compatibility shims
# ---------------------------------------------------------------------------
# The classifier package imports these names.  Rather than breaking it (which
# is out-of-scope for this migration), we expose shims that forward to the
# Supabase equivalents.  The classifier passes the "conn" object as the first
# positional arg to get_last_n_turns / get_all_turns; our shims accept it but
# ignore it, using the shared Supabase client from the service layer instead.

def open_connection(_db_path: str = "") -> object:  # type: ignore[return]
    """Shim: returns the Supabase client so legacy callers compile without changes.

    The returned object is a supabase.Client, not a sqlite3.Connection.
    It is passed as the first arg to shim versions of get_last_n_turns /
    get_all_turns which ignore it and use the service-level client.
    """
    from .service import _get_client
    return _get_client()


def _shim_get_last_n_turns(_ignored_conn: object, uid: str, n: int):
    """Shim wrapping get_last_n_turns; ignores the legacy conn arg."""
    from .service import _get_client
    return get_last_n_turns(_get_client(), uid, n)


def _shim_get_all_turns(_ignored_conn: object, uid: str):
    """Shim wrapping get_all_turns; ignores the legacy conn arg."""
    from .service import _get_client
    return get_all_turns(_get_client(), uid)


def get_all_session_ids(_ignored_conn: object = None):
    """Shim: returns all uids (replaces the old session_id list)."""
    from .service import _get_client
    return get_all_uids(_get_client())


# Replace the real functions with shims in the module namespace so that
# `from summarizer import get_last_n_turns` picks up the shim.
import sys as _sys
_mod = _sys.modules[__name__]
_mod.get_last_n_turns = _shim_get_last_n_turns  # type: ignore[attr-defined]
_mod.get_all_turns = _shim_get_all_turns         # type: ignore[attr-defined]


__all__ = [
    "init_service",
    "write_turn",
    "summarize_on_demand",
    "regenerate_summary",
    "cfg",
    # Primary Supabase helpers
    "get_supabase_client",
    "get_session",
    "get_last_n_turns",
    "get_all_turns",
    "get_all_uids",
    "TurnRow",
    "SessionRow",
    # Backward-compat shims (for the classifier package)
    "open_connection",
    "get_all_session_ids",
]
