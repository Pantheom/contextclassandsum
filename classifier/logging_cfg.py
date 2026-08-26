"""
classifier/logging_cfg.py
-------------------------
Logging setup for the classifier service.

Re-uses the root logger that summarizer/logging_cfg.py configures (same
format, same handler) — both services share a process so their log lines
interleave correctly on the same stdout stream.

Adds log_classification_event() — one INFO line per classify() call.
"""

from __future__ import annotations

# Delegate root-logger configuration to the summarizer's logging_cfg so both
# services always share the same format and we never double-configure.
from summarizer.logging_cfg import configure_logging, get_logger


def log_classification_event(
    *,
    session_id: str,
    verdict: bool,              # True = YES (needs context), False = NO
    raw_output: str,            # Exact string returned by the model
    prompt_preview: str,        # First 60 chars of current_prompt, for tracing
) -> None:
    """Emit a single structured INFO line for every classify() call.

    Example output:
        2026-08-22T21:05:00 | INFO     | classifier.service |
        CLASSIFIED session=abc123 verdict=YES raw="YES" prompt="Can you elabo..."
    """
    logger = get_logger("classifier.service")
    verdict_str = "YES" if verdict else "NO"
    # Truncate and sanitise for single-line log safety.
    preview = prompt_preview[:60].replace("\n", " ").replace("\r", "")
    logger.info(
        'CLASSIFIED session=%s verdict=%s raw=%r prompt="%s..."',
        session_id,
        verdict_str,
        raw_output,
        preview,
    )


__all__ = ["configure_logging", "get_logger", "log_classification_event"]
