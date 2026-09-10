"""
tests/test_production_api.py
----------------------------
TestClient-based test suite for the production API (api/main.py).

All GGUF inference is mocked — no real models are needed.
The model-loading background thread is patched out in every test.

Run with:
    python -m pytest tests/test_production_api.py -v
    # or just: python tests/test_production_api.py
"""
from __future__ import annotations

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `api`, `summarizer`, etc. resolve.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient

import api.main as main_module
from api.main import app

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

VALID_SESSION = "test-session-1"
VALID_PROMPT  = "What was the deadline we agreed on?"
SHORT_CONTEXT = (
    "## Prior Context\n"
    "User asked about the project deadline. Assistant said it is Friday.\n\n"
    "## Recent Turns\n"
    "[Turn 1] USER: When is the deadline?\n"
    "[Turn 2] ASSISTANT: It's this Friday."
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_client(models_ready: bool = True) -> TestClient:
    """Return a TestClient with model loading patched out.

    If models_ready=True, manually sets _models_ready so endpoints accept
    requests. If False, leaves the event unset so endpoints return 503.
    """
    if models_ready:
        main_module._models_ready.set()
    else:
        main_module._models_ready.clear()
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestHealthEndpoint(unittest.TestCase):

    def test_health_returns_ok_when_models_loaded(self):
        with patch("api.main._load_models_bg"):
            client = _make_client(models_ready=True)
            resp = client.get("/v1/health")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["models_loaded"])

    def test_health_returns_ok_but_not_loaded_before_init(self):
        with patch("api.main._load_models_bg"):
            client = _make_client(models_ready=False)
            resp = client.get("/v1/health")

        # Health check itself never 503s — it just reports the state
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertFalse(data["models_loaded"])


class TestProcessEndpoint503(unittest.TestCase):
    """Endpoint called before models finish loading must return 503."""

    def test_process_returns_503_when_models_not_ready(self):
        with patch("api.main._load_models_bg"):
            client = _make_client(models_ready=False)
            resp = client.post(
                "/v1/process",
                json={"session_id": VALID_SESSION, "prompt": VALID_PROMPT},
            )

        self.assertEqual(resp.status_code, 503)
        data = resp.json()
        self.assertEqual(data["error"], "service_unavailable")
        self.assertIn("detail", data)


class TestProcessValidation(unittest.TestCase):
    """Input validation — 422 with consistent error shape."""

    def setUp(self):
        with patch("api.main._load_models_bg"):
            self.client = _make_client(models_ready=True)

    def test_oversized_prompt_returns_422(self):
        resp = self.client.post(
            "/v1/process",
            json={"session_id": VALID_SESSION, "prompt": "x" * 8_001},
        )
        self.assertEqual(resp.status_code, 422)
        data = resp.json()
        self.assertEqual(data["error"], "validation_error")
        self.assertIn("detail", data)

    def test_empty_prompt_returns_422(self):
        resp = self.client.post(
            "/v1/process",
            json={"session_id": VALID_SESSION, "prompt": ""},
        )
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"], "validation_error")

    def test_malformed_session_id_special_chars_returns_422(self):
        resp = self.client.post(
            "/v1/process",
            json={"session_id": "invalid session!", "prompt": VALID_PROMPT},
        )
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"], "validation_error")

    def test_empty_session_id_returns_422(self):
        resp = self.client.post(
            "/v1/process",
            json={"session_id": "", "prompt": VALID_PROMPT},
        )
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"], "validation_error")

    def test_session_id_too_long_returns_422(self):
        resp = self.client.post(
            "/v1/process",
            json={"session_id": "a" * 201, "prompt": VALID_PROMPT},
        )
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"], "validation_error")


class TestProcessNoContext(unittest.TestCase):
    """classify → NO: combined_prompt equals the original prompt, context is null."""

    def test_no_context_path(self):
        no_ctx_result = {"needs_context": False, "context": None}

        with patch("api.main._load_models_bg"), \
             patch("api.main.get_response_context", return_value=no_ctx_result) as mock_ctx, \
             patch("api.main.write_turn", return_value=3) as mock_wt:

            client = _make_client(models_ready=True)
            resp = client.post(
                "/v1/process",
                json={"session_id": VALID_SESSION, "prompt": VALID_PROMPT},
            )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # All four fields present
        self.assertIn("needs_context",   data)
        self.assertIn("context",         data)
        self.assertIn("combined_prompt", data)
        self.assertIn("turn_index",      data)

        # No context → combined_prompt must equal original prompt unchanged
        self.assertFalse(data["needs_context"])
        self.assertIsNone(data["context"])
        self.assertEqual(data["combined_prompt"], VALID_PROMPT)
        self.assertEqual(data["turn_index"], 3)

        # Verify underlying service calls
        mock_ctx.assert_called_once_with(VALID_SESSION, VALID_PROMPT)
        mock_wt.assert_called_once_with(VALID_SESSION, "user", VALID_PROMPT)


class TestProcessWithContext(unittest.TestCase):
    """classify → YES: combined_prompt contains both context block and prompt."""

    def test_yes_context_path(self):
        yes_ctx_result = {"needs_context": True, "context": SHORT_CONTEXT}

        with patch("api.main._load_models_bg"), \
             patch("api.main.get_response_context", return_value=yes_ctx_result) as mock_ctx, \
             patch("api.main.write_turn", return_value=7) as mock_wt:

            client = _make_client(models_ready=True)
            resp = client.post(
                "/v1/process",
                json={"session_id": VALID_SESSION, "prompt": VALID_PROMPT},
            )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertTrue(data["needs_context"])
        self.assertEqual(data["context"], SHORT_CONTEXT)
        self.assertEqual(data["turn_index"], 7)

        combined = data["combined_prompt"]

        # combined_prompt must contain the context block
        self.assertIn(SHORT_CONTEXT, combined)

        # combined_prompt must contain the original prompt
        self.assertIn(VALID_PROMPT, combined)

        # The separator must be present and unambiguous
        self.assertIn("---", combined)

        # The prompt section must be labelled
        self.assertIn("User's current message:", combined)

        # combined_prompt must NOT simply equal the original prompt
        self.assertNotEqual(combined, VALID_PROMPT)


class TestReplyEndpoint(unittest.TestCase):
    """POST /v1/sessions/{id}/reply — logs assistant turn, returns turn_index."""

    def test_valid_reply_returns_turn_index(self):
        with patch("api.main._load_models_bg"), \
             patch("api.main.write_turn", return_value=8) as mock_wt:

            client = _make_client(models_ready=True)
            resp = client.post(
                f"/v1/sessions/{VALID_SESSION}/reply",
                json={"text": "The deadline is this Friday."},
            )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["turn_index"], 8)
        mock_wt.assert_called_once_with(VALID_SESSION, "assistant", "The deadline is this Friday.")

    def test_reply_oversized_text_returns_422(self):
        with patch("api.main._load_models_bg"):
            client = _make_client(models_ready=True)
            resp = client.post(
                f"/v1/sessions/{VALID_SESSION}/reply",
                json={"text": "y" * 8_001},
            )
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["error"], "validation_error")

    def test_reply_503_when_models_not_ready(self):
        with patch("api.main._load_models_bg"):
            client = _make_client(models_ready=False)
            resp = client.post(
                f"/v1/sessions/{VALID_SESSION}/reply",
                json={"text": "Some reply."},
            )
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.json()["error"], "service_unavailable")


class TestDocsAvailable(unittest.TestCase):
    """/docs and /redoc must render without errors."""

    def setUp(self):
        with patch("api.main._load_models_bg"):
            self.client = _make_client(models_ready=True)

    def test_swagger_docs_available(self):
        resp = self.client.get("/docs")
        self.assertEqual(resp.status_code, 200)

    def test_redoc_available(self):
        resp = self.client.get("/redoc")
        self.assertEqual(resp.status_code, 200)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Production API Test Suite")
    print("Models: STUBBED (no real GGUF required)")
    print("=" * 70)
    unittest.main(verbosity=2)
