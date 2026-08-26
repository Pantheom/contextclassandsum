"""
classifier/prompt.py
--------------------
Builds the exact prompt sent to Gemma 3 2B for each YES/NO classification.

The template constants are module-level so they can be reviewed, tested, and
modified independently of the service logic — same convention as
summarizer/prompt.py.

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

from typing import List

from summarizer.db import TurnRow


# ---------------------------------------------------------------------------
# Prompt template constants
# ---------------------------------------------------------------------------

_SYSTEM_BLOCK = """\
You are a context-need classifier for a conversational AI system.
Your sole task is to decide whether the current user prompt can be answered \
correctly WITHOUT knowing what was said earlier in this conversation.

Answer YES if prior context is needed to give a correct answer.
Answer NO if the prompt is fully self-contained and needs no prior context.

Rules:
- Output ONLY the single word YES or NO. No punctuation, no explanation, \
no other words.
- When in doubt, output YES. Fetching context unnecessarily is far less harmful \
than missing context that is actually required.

Examples of NO (self-contained — no prior context needed):
  - "What is the capital of France?"
  - "Write me a haiku about autumn."
  - "How do I reverse a list in Python?"
  - "Summarise the French Revolution in three sentences."

Examples of YES (back-reference — prior context required):
  - "Can you elaborate on that?"            (no antecedent for 'that')
  - "What did you mean by the last point?"  ('last point' refers to prior turn)
  - "Make it shorter."                      ('it' refers to earlier content)
  - "What was the deadline you mentioned?"  (fact stated in a prior turn)
  - "Can we go back to what we were discussing?" (implicit back-reference)\
"""

_USER_TEMPLATE = """\
{system_block}

RECENT CONVERSATION HISTORY ({n_turns} turn(s)):
{formatted_history}

CURRENT USER PROMPT:
{current_prompt}

Does the current prompt require knowledge of prior conversation context \
to answer correctly?

Respond with ONLY the single word YES or NO.\
"""

_NO_HISTORY_PLACEHOLDER = "(none — this is the beginning of the conversation)"


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def format_history(turns: List[TurnRow]) -> str:
    """Render turns as labelled dialogue lines for prompt inclusion.

    Example:
        [Turn 3] USER: What was the deadline?
        [Turn 4] ASSISTANT: The deadline is Friday.
    """
    if not turns:
        return _NO_HISTORY_PLACEHOLDER
    return "\n".join(
        f"[Turn {t.turn_index}] {t.role.upper()}: {t.text}"
        for t in turns
    )


def build_classifier_prompt(
    history_turns: List[TurnRow],
    current_prompt: str,
) -> str:
    """Assemble the full Gemma 3 prompt for one YES/NO classification call.

    Args:
        history_turns:  Recent turns (last N) to show as conversation history.
                        May be empty for a brand-new session; the model should
                        still produce a valid NO in that case.
        current_prompt: The new user message being classified.

    Returns:
        A complete prompt string ending with ``<start_of_turn>model\\n`` so
        llama.cpp begins generating the YES/NO answer immediately.
    """
    n_turns = len(history_turns)
    history_label = n_turns if n_turns > 0 else "no"

    user_content = _USER_TEMPLATE.format(
        system_block=_SYSTEM_BLOCK,
        n_turns=history_label,
        formatted_history=format_history(history_turns),
        current_prompt=current_prompt.strip(),
    )

    return (
        f"<start_of_turn>user\n"
        f"{user_content}\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
