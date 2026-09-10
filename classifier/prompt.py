"""
classifier/prompt.py
--------------------
Builds the exact prompt sent to Gemma 3 2B for each YES/NO classification.

The template is intentionally compact — the classifier only needs YES or NO,
so we minimise tokens while still giving the model the three signals it needs:
  1. A running summary of the conversation so far (from context_classifier).
  2. The last few raw turns (recent query/answer pairs from chat_history).
  3. The current user prompt.

Gemma 3 chat format:
    <start_of_turn>user
    {content}
    <end_of_turn>
    <start_of_turn>model
    {generation begins here}

Note: <bos> is added automatically by llama.cpp — do NOT include it in the
raw string or it will be doubled and may degrade output quality.
"""

from __future__ import annotations

from typing import List, Optional

from summarizer.db import TurnRow


# ---------------------------------------------------------------------------
# Prompt template constants
# ---------------------------------------------------------------------------

_SYSTEM_BLOCK = (
    "You are a context classifier. "
    "Decide if the CURRENT PROMPT requires prior conversation context "
    "to be understood or answered correctly.\n"
    "Answer ONLY with YES (context needed) or NO (self-contained).\n"
    "When in doubt, answer YES."
)

_TEMPLATE = """\
{system_block}

[PRIOR SUMMARY]
{summary}

[RECENT CONVERSATION]
{formatted_history}

[CURRENT PROMPT]
{current_prompt}

Requires prior context? (YES/NO):\
"""

_NO_SUMMARY_PLACEHOLDER = "(none)"
_NO_HISTORY_PLACEHOLDER = "(none — start of conversation)"


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def format_history(turns: List[TurnRow]) -> str:
    """Render turns as compact labelled dialogue lines for prompt inclusion.

    Example:
        [U] What was the deadline?
        [A] The deadline is Friday.
    """
    if not turns:
        return _NO_HISTORY_PLACEHOLDER
    role_label = {"user": "[U]", "assistant": "[A]"}
    return "\n".join(
        f"{role_label.get(t.role, '[?]')} {t.text}"
        for t in turns
    )


def build_classifier_prompt(
    history_turns: List[TurnRow],
    current_prompt: str,
    summary: Optional[str] = None,
) -> str:
    """Assemble the compact Gemma 3 prompt for one YES/NO classification call.

    Args:
        history_turns:  Recent turns (last N) from chat_history.
                        May be empty for a brand-new session.
        current_prompt: The new user message being classified.
        summary:        Optional running summary from context_classifier.
                        Pass None or empty string when no summary exists yet.

    Returns:
        A complete prompt string ending with ``<start_of_turn>model\\n`` so
        llama.cpp begins generating the YES/NO answer immediately.
    """
    summary_text = (
        summary.strip()
        if summary and summary.strip()
        else _NO_SUMMARY_PLACEHOLDER
    )

    user_content = _TEMPLATE.format(
        system_block=_SYSTEM_BLOCK,
        summary=summary_text,
        formatted_history=format_history(history_turns),
        current_prompt=current_prompt.strip(),
    )

    return (
        f"<start_of_turn>user\n"
        f"{user_content}\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
