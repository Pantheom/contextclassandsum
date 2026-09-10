"""
classifier/config.py
--------------------
All tunable knobs for the classifier service are read from environment
variables. Import `cfg` from this module everywhere inside the package.

Follows the identical frozen-dataclass pattern as summarizer/config.py so
both services are configured the same way and neither surprises the other.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClassifierConfig:
    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    model_path: str = field(
        default_factory=lambda: os.environ.get("CLASSIFIER_MODEL_PATH", "")
    )
    """Absolute path to the Gemma 3 2B Q4 GGUF file.
    Required — the service hard-fails at load time if this is absent or the
    file does not exist. No auto-download."""

    n_ctx: int = field(
        default_factory=lambda: int(os.environ.get("CLASSIFIER_N_CTX", "2048"))
    )
    """LLM context window in tokens.
    2048 is sufficient: input is at most 2 prior turns + the current prompt.
    Keeps the KV-cache small so both models coexist comfortably in 8 GB RAM."""

    n_threads: int = field(
        default_factory=lambda: int(os.environ.get("CLASSIFIER_N_THREADS", "1"))
    )
    """CPU threads for llama.cpp inference.
    Defaults to 1 to leave headroom for the summarizer's own inference calls
    on the shared 2-vCPU machine. Tune with CLASSIFIER_N_THREADS."""

    max_tokens: int = field(
        default_factory=lambda: int(os.environ.get("CLASSIFIER_MAX_TOKENS", "5"))
    )
    """Maximum tokens the model may emit.
    Output is constrained to YES or NO (1 token each) via GBNF grammar;
    this is a safety ceiling for the grammar-unavailable fallback path."""

    # ------------------------------------------------------------------ #
    # Behaviour                                                            #
    # ------------------------------------------------------------------ #
    history_turns: int = field(
        default_factory=lambda: int(
            os.environ.get("CLASSIFIER_HISTORY_TURNS", "6")
        )
    )
    """Number of prior turns fed to the classifier when building its prompt.
    Default 6 covers the last 2-3 query/answer pairs, giving the model enough
    recency signal for pronoun/reference resolution without inflating the KV cache.
    Override with CLASSIFIER_HISTORY_TURNS env var."""

    context_turns: int = field(
        default_factory=lambda: int(
            os.environ.get("CLASSIFIER_CONTEXT_TURNS", "3")
        )
    )
    """Number of recent turns included in the context block returned on a YES
    classification. Slightly wider window than history_turns so the answering
    model has enough recent dialogue to work with."""


# Module-level singleton — import this everywhere.
cfg = ClassifierConfig()
