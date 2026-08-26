"""
summarizer/prompt.py
--------------------
Builds the exact prompt sent to Phi-4-mini-instruct for each summarization run.

The template is defined here as a module-level constant so it can be reviewed
and tested independently of service logic.
"""

from __future__ import annotations

from typing import List

from .db import TurnRow


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
# Uses Phi-4-mini-instruct's native chat format:
#   <|system|>...<|end|><|user|>...<|end|><|assistant|>
#
# The <|assistant|> token at the end primes the model to begin generating
# the summary immediately, without any preamble.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a conversation summarizer. Your job is to produce ONE concise, \
self-contained summary that will be used as context for future AI responses. \
This summary REPLACES the previous one — it must stand alone, not append to it.

Rules:
- Write in third-person prose (e.g. "The user asked...", "The assistant explained...").
- Preserve: decisions made, named entities (people, products, places), key facts \
stated, and any open or unresolved threads.
- Omit: pleasantries, filler phrases, and redundant restatements.
- Keep the summary under 300 words unless the complexity of the conversation \
genuinely demands more.
- Do NOT copy-paste raw dialogue. Synthesise it into cohesive prose.
- Do NOT start with phrases like "Here is a summary" or "This is a summary". \
Begin immediately with substantive content.\
"""

_USER_TEMPLATE = """\
## Previous Summary
{previous_summary}

## New Conversation Turns (turns {first_new_turn}–{last_new_turn})
{formatted_turns}

Produce a single, consolidated summary that covers everything above — both the \
previous summary's content and the new turns. Replace the old summary entirely.\
"""

_EMPTY_SUMMARY_PLACEHOLDER = "(none — this is the first summary for this session)"


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def format_turns(turns: List[TurnRow]) -> str:
    """Render a list of TurnRow objects as numbered dialogue lines.

    Example:
        [Turn 1] USER: Hello, what's the project deadline?
        [Turn 2] ASSISTANT: The deadline is Friday 22nd.
    """
    lines: List[str] = []
    for t in turns:
        role_label = t.role.upper()
        lines.append(f"[Turn {t.turn_index}] {role_label}: {t.text}")
    return "\n".join(lines)


def build_prompt(
    previous_summary: str | None,
    turns: List[TurnRow],
    first_idx: int,
    last_idx: int,
) -> str:
    """Assemble the full Phi-4-mini-instruct prompt for one summarization call.

    Args:
        previous_summary: The session's current_summary from the DB.
                          Pass None or empty string if this is the first run.
        turns:            The new TurnRow objects to be folded into the summary.
        first_idx:        turn_index of the first turn in `turns`.
        last_idx:         turn_index of the last turn in `turns`.

    Returns:
        A complete prompt string ending with ``<|assistant|>`` so llama.cpp
        begins generating the summary immediately.
    """
    summary_text = (
        previous_summary.strip()
        if previous_summary and previous_summary.strip()
        else _EMPTY_SUMMARY_PLACEHOLDER
    )

    user_content = _USER_TEMPLATE.format(
        previous_summary=summary_text,
        first_new_turn=first_idx,
        last_new_turn=last_idx,
        formatted_turns=format_turns(turns),
    )

    return (
        f"<|system|>\n{_SYSTEM_PROMPT}\n<|end|>"
        f"<|user|>\n{user_content}\n<|end|>"
        f"<|assistant|>\n"
    )
