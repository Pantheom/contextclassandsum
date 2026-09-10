"""
summarizer/config.py
--------------------
All tunable knobs are read from environment variables so nothing is hard-coded.
Import `cfg` from this module to access settings anywhere in the package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    model_path: str = field(
        default_factory=lambda: os.environ.get("SUMMARIZER_MODEL_PATH", "")
    )
    """Absolute path to the Phi-4-mini-instruct Q4_K_M GGUF file.
    Required — the service will hard-fail at load time if this is absent."""

    n_ctx: int = field(
        default_factory=lambda: int(os.environ.get("SUMMARIZER_N_CTX", "4096"))
    )
    """LLM context window in tokens.
    4096 keeps KV-cache well under 200 MB, leaving headroom for the ~2 GB
    classifier model that will share the same machine."""

    n_threads: int = field(
        default_factory=lambda: int(os.environ.get("SUMMARIZER_N_THREADS", "2"))
    )
    """CPU threads for llama.cpp inference. Matches target 2-vCPU deployment."""

    max_tokens: int = field(
        default_factory=lambda: int(os.environ.get("SUMMARIZER_MAX_TOKENS", "512"))
    )
    """Maximum tokens the model may emit per summary."""

    n_gpu_layers: int = field(
        default_factory=lambda: int(os.environ.get("SUMMARIZER_N_GPU_LAYERS", "-1"))
    )
    """Number of layers to offload to GPU VRAM (-1 for all layers, 0 for CPU only)."""

    # ------------------------------------------------------------------ #
    # Supabase                                                             #
    # ------------------------------------------------------------------ #
    supabase_url: str = field(
        default_factory=lambda: os.environ.get("SUPABASE_URL", "")
    )
    """Supabase project URL (e.g. https://xyz.supabase.co).
    Required — the service will fail at startup if absent."""

    supabase_key: str = field(
        default_factory=lambda: os.environ.get("SUPABASE_KEY", "")
    )
    """Supabase anon or service-role key.
    Required — the service will fail at startup if absent."""

    # ------------------------------------------------------------------ #
    # Trigger                                                              #
    # ------------------------------------------------------------------ #
    periodic_threshold: int = field(
        default_factory=lambda: int(
            os.environ.get("SUMMARIZER_PERIODIC_THRESHOLD", "10")
        )
    )
    """Number of new messages (for a given uid) since last summary that triggers
    automatic periodic summarization (Entry Point 1)."""

    # ------------------------------------------------------------------ #
    # Backward-compatibility shim                                          #
    # ------------------------------------------------------------------ #
    db_path: str = field(default_factory=lambda: "")
    """Deprecated — kept as an empty-string shim so legacy callers that read
    summarizer_cfg.db_path (e.g. classifier/service.py) don't raise
    AttributeError.  The value is passed to open_connection() which ignores it."""


# Module-level singleton — import this everywhere.
cfg = Config()
