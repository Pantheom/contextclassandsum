"""
summarizer/logging_cfg.py
-------------------------
Structured logging setup and summarization-event helper.

Import `get_logger` to get a named logger with the shared format, or call
`log_summarization_event` directly after each successful regenerate_summary().
"""

from __future__ import annotations

import logging
import sys
from typing import Tuple


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Set up root logger with the shared format.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid adding duplicate handlers if the caller already configured logging.
    if not root.handlers:
        root.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring the root logger on first call."""
    configure_logging()
    return logging.getLogger(name)


def log_summarization_event(
    *,
    session_id: str,
    trigger: str,                   # 'periodic' | 'on-demand'
    turn_range: Tuple[int, int],    # (first_turn_index, last_turn_index)
    summary_length: int,            # character length of the produced summary
) -> None:
    """Emit a single structured INFO line for every summarization event.

    Example output:
        2026-08-15T23:05:00 | INFO     | summarizer.service |
        SUMMARIZED session=abc123 trigger=periodic turns=1-15 summary_chars=284
    """
    logger = get_logger("summarizer.service")
    first, last = turn_range
    logger.info(
        "SUMMARIZED session=%s trigger=%s turns=%d-%d summary_chars=%d",
        session_id,
        trigger,
        first,
        last,
        summary_length,
    )
