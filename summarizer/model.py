"""
summarizer/model.py
-------------------
Phi-4-mini-instruct singleton loader and inference wrapper.

Design constraints:
- Weights are loaded exactly once per process (ModelSingleton pattern with a
  class-level threading.Lock protecting the load path).
- A separate threading.Lock serialises inference calls so concurrent
  ThreadPoolExecutor jobs cannot race on the same Llama instance.
- n_gpu_layers=0 -> CPU-only, predictable RAM footprint on a 2-vCPU / 8 GB
  machine that will also host a ~2 GB classifier model.
- Hard-fails immediately if SUMMARIZER_MODEL_PATH is missing or the file does
  not exist — no silent fallback, no auto-download.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

from .config import cfg
from .logging_cfg import get_logger

logger = get_logger("summarizer.model")


class _ModelSingleton:
    """Thread-safe, load-once wrapper around llama_cpp.Llama."""

    _instance: Optional["_ModelSingleton"] = None
    _class_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        # Validate path eagerly — fail loud and fast.
        model_path = cfg.model_path
        if not model_path:
            raise RuntimeError(
                "SUMMARIZER_MODEL_PATH environment variable is not set. "
                "Set it to the absolute path of the Phi-4-mini-instruct "
                "Q4_K_M GGUF file before starting the service."
            )
        if not os.path.isfile(model_path):
            raise RuntimeError(
                f"Model file not found: {model_path!r}\n"
                "Ensure the GGUF file is downloaded and the path is correct. "
                "The service will not auto-download model weights."
            )

        logger.info(
            "Loading model from %s  (n_ctx=%d, n_threads=%d, n_gpu_layers=0)",
            model_path,
            cfg.n_ctx,
            cfg.n_threads,
        )

        # Import here so that tests can patch before the import resolves.
        from llama_cpp import Llama  # noqa: PLC0415

        self._llm = Llama(
            model_path=model_path,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_gpu_layers=0,     # CPU-only — predictable RAM
            verbose=False,      # Suppress llama.cpp startup chatter
        )
        self._inference_lock = threading.Lock()
        logger.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_inference(self, prompt: str, max_tokens: int) -> str:
        """Run a raw completion against the loaded model.

        Thread-safe: serialises concurrent calls via _inference_lock so the
        ThreadPoolExecutor background jobs cannot race on the same instance.

        Stop tokens are the Phi-4-mini special tokens that would follow an
        assistant turn, preventing the model from generating phantom user/system
        turns after finishing its summary.
        """
        with self._inference_lock:
            output = self._llm(
                prompt,
                max_tokens=max_tokens,
                stop=["<|end|>", "<|user|>", "<|system|>"],
                echo=False,
            )
        text: str = output["choices"][0]["text"]
        return text.strip()

    # ------------------------------------------------------------------
    # Singleton constructor
    # ------------------------------------------------------------------

    @classmethod
    def get(cls) -> "_ModelSingleton":
        """Return the shared instance, loading the model on first call."""
        if cls._instance is None:
            with cls._class_lock:
                # Double-checked locking — re-test inside the lock.
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Discard the singleton — used by tests to inject a mock instance."""
        with cls._class_lock:
            cls._instance = None


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def get_model() -> _ModelSingleton:
    """Return the shared ModelSingleton, loading the model if needed."""
    return _ModelSingleton.get()


def run_inference(prompt: str, max_tokens: int | None = None) -> str:
    """Convenience wrapper used by service.py and tests."""
    tokens = max_tokens if max_tokens is not None else cfg.max_tokens
    return get_model().run_inference(prompt, tokens)
