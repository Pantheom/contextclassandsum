"""
classifier/model.py
-------------------
Gemma 3 2B Q4 GGUF singleton loader and inference wrapper for YES/NO
classification.

Design:
- Same double-checked locking singleton pattern as summarizer/model.py.
- Grammar-constrained decoding via LlamaGrammar (GBNF): the model can only
  emit the exact string "YES" or "NO", eliminating the need to handle
  ambiguous output in the normal path.
- Graceful fallback: if LlamaGrammar cannot be imported (older
  llama-cpp-python build), a WARNING is logged at startup and inference falls
  back to max_tokens=5 with a strict parse step in service.py.
- n_threads=1 default — respects shared 2-vCPU budget alongside the
  summarizer's Phi-4-mini instance.
- Hard-fails if CLASSIFIER_MODEL_PATH is absent or the file does not exist.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

from .config import cfg
from .logging_cfg import get_logger

logger = get_logger("classifier.model")

# ---------------------------------------------------------------------------
# GBNF grammar for YES/NO constrained decoding
# ---------------------------------------------------------------------------

_YESNO_GRAMMAR_TEXT = 'root ::= "YES" | "NO"'

# Try to import LlamaGrammar at module level so the availability check
# happens once, not on every inference call.
try:
    from llama_cpp import LlamaGrammar as _LlamaGrammar
    _GRAMMAR_AVAILABLE = True
except ImportError:
    _LlamaGrammar = None  # type: ignore[assignment,misc]
    _GRAMMAR_AVAILABLE = False


class _ModelSingleton:
    """Thread-safe, load-once wrapper around llama_cpp.Llama for classification."""

    _instance: Optional["_ModelSingleton"] = None
    _class_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        # ------------------------------------------------------------------ #
        # Validate model path — fail loud and fast                            #
        # ------------------------------------------------------------------ #
        model_path = cfg.model_path
        if not model_path:
            raise RuntimeError(
                "CLASSIFIER_MODEL_PATH environment variable is not set. "
                "Set it to the absolute path of the Gemma 3 2B Q4 GGUF file "
                "before starting the classifier service."
            )
        if not os.path.isfile(model_path):
            raise RuntimeError(
                f"Classifier model file not found: {model_path!r}\n"
                "Ensure the GGUF file is downloaded and the path is correct. "
                "The service will not auto-download model weights."
            )

        logger.info(
            "Loading classifier model from %s  "
            "(n_ctx=%d, n_threads=%d, n_gpu_layers=0)",
            model_path,
            cfg.n_ctx,
            cfg.n_threads,
        )

        # Import here so tests can patch before the import resolves.
        from llama_cpp import Llama  # noqa: PLC0415

        self._llm = Llama(
            model_path=model_path,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_gpu_layers=0,     # CPU-only — predictable RAM alongside summarizer
            verbose=False,
        )
        self._inference_lock = threading.Lock()

        # ------------------------------------------------------------------ #
        # Build GBNF grammar (once, at load time)                             #
        # ------------------------------------------------------------------ #
        if _GRAMMAR_AVAILABLE:
            self._grammar = _LlamaGrammar.from_string(_YESNO_GRAMMAR_TEXT)
            logger.info(
                "Classifier model loaded. Grammar-constrained YES/NO decoding active."
            )
        else:
            self._grammar = None
            logger.warning(
                "LlamaGrammar is not available in this llama-cpp-python build. "
                "Classifier will fall back to max_tokens=%d + parse-based validation. "
                "Upgrade llama-cpp-python to enable grammar-constrained decoding.",
                cfg.max_tokens,
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_inference(self, prompt: str) -> str:
        """Run a constrained YES/NO completion.

        Thread-safe: serialises calls via _inference_lock.

        With grammar active:   model can only emit "YES" or "NO" (exact).
        Without grammar:       model emits up to max_tokens tokens; service.py
                               applies the fail-safe parse rule.

        Stop tokens cover Gemma 3's end-of-turn marker to prevent the model
        from generating a phantom next turn.
        """
        kwargs: dict = {}
        if self._grammar is not None:
            kwargs["grammar"] = self._grammar

        with self._inference_lock:
            output = self._llm(
                prompt,
                max_tokens=cfg.max_tokens,
                stop=["<end_of_turn>"],
                echo=False,
                **kwargs,
            )

        text: str = output["choices"][0]["text"]
        return text.strip()

    # ------------------------------------------------------------------
    # Singleton lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def get(cls) -> "_ModelSingleton":
        """Return the shared instance, loading the model on first call."""
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Discard the singleton. Used by tests to inject a mock."""
        with cls._class_lock:
            cls._instance = None


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------

def get_model() -> _ModelSingleton:
    """Return the shared ModelSingleton, loading on first call."""
    return _ModelSingleton.get()


def run_inference(prompt: str) -> str:
    """Convenience wrapper called by service.py and patchable in tests."""
    return get_model().run_inference(prompt)
