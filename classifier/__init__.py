"""
classifier/__init__.py
-----------------------
Public API surface for the classifier package.

External callers should import from here, not from individual sub-modules,
so internal refactors don't break callers.

Usage:
    from classifier import init_classifier, classify, get_response_context
"""

from __future__ import annotations

from .logging_cfg import configure_logging
from .service import classify, get_response_context


def init_classifier() -> None:
    """Initialise the classifier service.

    Call this once at application startup, alongside summarizer.init_service().

    The classifier does not own any DB schema — it reads from the summarizer's
    DB using the summarizer's own helpers. init_classifier() therefore only
    sets up logging; no init_db() call is made here.

    The Gemma 3 model itself is loaded lazily on the first classify() call
    (same singleton pattern as the summarizer) — call init_classifier() early
    to get the startup WARNING logged if grammar-constrained decoding is
    unavailable, without triggering an actual model load.
    """
    configure_logging()


__all__ = [
    "init_classifier",
    "classify",
    "get_response_context",
]
