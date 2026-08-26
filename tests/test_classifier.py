"""
tests/test_classifier.py
------------------------
Standalone test script for the classifier service.

Can be run directly:
    python -X utf8 tests/test_classifier.py

Or via pytest:
    python -m pytest tests/test_classifier.py -v

Tests cover (matching the specification):
    1  classify() returns True on unambiguous "YES" model output
    2  classify() returns False ONLY on unambiguous "NO"
    3  classify() returns True (fail-safe) on ambiguous/unparseable model output
    4  get_response_context() returns {"needs_context": False, "context": None}
       when classify() is False, WITHOUT calling summarize_on_demand
    5  get_response_context() calls summarize_on_demand exactly once and returns
       a non-empty context string when classify() is True
    6  get_last_n_turns returns turns in correct chronological (ASC) order
    7  get_last_n_turns handles a session with 0 turns (empty list, no crash)
    8  get_last_n_turns handles a session with fewer than N turns (returns all)

Plus unit tests for the prompt builder (token presence, history rendering,
empty-history placeholder).

No real GGUF is required — classifier.service.run_inference is mocked,
and summarizer.service.run_inference is also mocked (for any summarize_on_demand
calls that go through).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Environment setup — MUST happen before any package imports so the
# frozen-dataclass Config() singletons read the right values.
# ---------------------------------------------------------------------------
_tmp_dir = tempfile.mkdtemp(prefix="classifier_test_")
_TEST_DB = os.path.join(_tmp_dir, "test_classifier.db")

os.environ["SUMMARIZER_DB_PATH"] = _TEST_DB
os.environ["SUMMARIZER_MODEL_PATH"] = "/stub/phi4mini.gguf"   # never loaded
os.environ["CLASSIFIER_MODEL_PATH"] = "/stub/gemma3-2b.gguf"  # never loaded
os.environ.setdefault("CLASSIFIER_HISTORY_TURNS", "2")
os.environ.setdefault("CLASSIFIER_CONTEXT_TURNS", "3")

# ---------------------------------------------------------------------------
# Path setup — allow running from the repo root without installing packages.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Now safe to import — env vars are in place.
# ---------------------------------------------------------------------------
import summarizer                          # noqa: E402
from summarizer import db as _db           # noqa: E402
import classifier                          # noqa: E402
from classifier import service as _svc    # noqa: E402


# ---------------------------------------------------------------------------
# Test-wide mock for the summarizer's inference (needed if any path triggers
# summarize_on_demand -> regenerate_summary -> model).  Applied globally via
# a patcher started in __main__ and also via setUp where needed.
# ---------------------------------------------------------------------------

def _fake_summary_inference(prompt: str, max_tokens=None) -> str:
    return "Fake summary from summarizer model."


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _seed_turns(session_id: str, n: int, start: int = 1) -> None:
    """Write n alternating turns directly via summarizer.write_turn."""
    for i in range(start, start + n):
        role = "user" if i % 2 == 1 else "assistant"
        summarizer.write_turn(session_id, role, f"Turn {i} content for '{session_id}'")


def _insert_turns_directly(session_id: str, turns: list) -> None:
    """Insert (role, text) pairs directly into the DB for lower-level tests."""
    conn = _db.open_connection(_TEST_DB)
    try:
        _db.get_or_create_session(conn, session_id)
        for role, text in turns:
            _db.insert_turn(conn, session_id, role, text)
    finally:
        conn.close()


def _fresh_conn():
    return _db.open_connection(_TEST_DB)


# ===========================================================================
# Test Group 1: classify() output parsing (Tests 1, 2, 3)
# ===========================================================================

class TestClassifyParsing(unittest.TestCase):
    """Tests 1-3: parse rule correctness and fail-safe bias."""

    SESSION_ID = "cls-parse-session"

    def setUp(self):
        summarizer.init_service()
        classifier.init_classifier()
        # Seed a couple of turns so get_last_n_turns has data.
        _seed_turns(self.SESSION_ID, 2)

    def _classify_with_raw(self, raw_output: str) -> bool:
        """Run classify() with a mocked inference that returns raw_output."""
        with patch("classifier.service.run_inference", return_value=raw_output):
            return _svc.classify(self.SESSION_ID, "Test prompt for parsing.")

    # --- Test 1 ---
    def test_yes_returns_true(self):
        """classify() MUST return True when model outputs 'YES'."""
        result = self._classify_with_raw("YES")
        self.assertTrue(result, "classify('YES') should return True")

    # --- Test 2 ---
    def test_no_returns_false(self):
        """classify() MUST return False when model outputs 'NO'."""
        result = self._classify_with_raw("NO")
        self.assertFalse(result, "classify('NO') should return False")

    # --- Test 3 ---
    def test_fail_safe_on_ambiguous_outputs(self):
        """classify() MUST return True (fail-safe) on any non-'NO' output."""
        ambiguous_cases = [
            "",             # empty string
            "Maybe",        # neither YES nor NO
            "YES AND NO",   # compound
            "yEs",          # wrong case alone — strip+upper is "YES" != "NO" -> True
            "NO.",          # trailing punctuation — strip+upper is "NO." != "NO" -> True
            " YES ",        # surrounding whitespace — strip is "YES", upper "YES" != "NO" -> True
            "YESNO",        # concatenated
            "I think YES",  # preamble
            "\n",           # only whitespace
        ]
        for raw in ambiguous_cases:
            with self.subTest(raw=repr(raw)):
                result = self._classify_with_raw(raw)
                self.assertTrue(
                    result,
                    f"classify({raw!r}) should return True (fail-safe), got False",
                )

    def test_no_with_surrounding_whitespace_returns_false(self):
        """' NO ' after strip+upper is 'NO' -> False (whitespace is stripped)."""
        result = self._classify_with_raw("  NO  ")
        self.assertFalse(
            result,
            "classify(' NO ') should return False — strip() removes whitespace before compare",
        )

    def test_lowercase_no_returns_false(self):
        """'no' after strip+upper is 'NO' -> False (case is normalised)."""
        result = self._classify_with_raw("no")
        self.assertFalse(
            result,
            "classify('no') should return False — upper() normalises case",
        )


# ===========================================================================
# Test Group 2: get_response_context() orchestration (Tests 4, 5)
# ===========================================================================

class TestGetResponseContext(unittest.TestCase):
    """Tests 4-5: orchestration, summarize_on_demand call discipline."""

    SESSION_ID = "ctx-session"

    def setUp(self):
        summarizer.init_service()
        classifier.init_classifier()
        _seed_turns(self.SESSION_ID, 4)

    def test_no_verdict_returns_correct_dict_and_skips_summarizer(self):
        """Test 4: False classify -> {needs_context: False, context: None},
        and summarize_on_demand must NOT be called."""
        with (
            patch("classifier.service.run_inference", return_value="NO"),
            patch(
                "classifier.service.summarize_on_demand",
                wraps=_svc.summarize_on_demand,
            ) as mock_summarize,
        ):
            result = _svc.get_response_context(self.SESSION_ID, "What is 2 + 2?")

        self.assertEqual(result, {"needs_context": False, "context": None})
        mock_summarize.assert_not_called()

    def test_yes_verdict_calls_summarizer_once_and_returns_context(self):
        """Test 5: True classify -> summarize_on_demand called exactly once,
        context is a non-empty string."""
        fake_summary = "The user discussed the project deadline. It is Friday."

        with (
            patch("classifier.service.run_inference", return_value="YES"),
            patch(
                "classifier.service.summarize_on_demand",
                return_value=fake_summary,
            ) as mock_summarize,
        ):
            result = _svc.get_response_context(
                self.SESSION_ID, "What was the deadline you mentioned?"
            )

        # summarize_on_demand called exactly once with the right session_id.
        mock_summarize.assert_called_once_with(self.SESSION_ID)

        self.assertTrue(result["needs_context"])
        self.assertIsNotNone(result["context"])
        self.assertIsInstance(result["context"], str)
        self.assertTrue(len(result["context"]) > 0, "context string must be non-empty")

    def test_yes_context_contains_summary_and_turns(self):
        """Context block must include both the summary text and recent turns."""
        fake_summary = "Project deadline is Friday. Budget is $50k."

        with (
            patch("classifier.service.run_inference", return_value="YES"),
            patch(
                "classifier.service.summarize_on_demand",
                return_value=fake_summary,
            ),
        ):
            result = _svc.get_response_context(self.SESSION_ID, "Remind me of the budget.")

        context = result["context"]
        self.assertIn("## Prior Context", context)
        self.assertIn("## Recent Turns", context)
        self.assertIn(fake_summary, context)

    def test_yes_context_no_turns_session(self):
        """get_response_context on a brand-new session (no turns) must not crash."""
        fresh_session = "ctx-fresh-session"
        summarizer.init_service()

        with (
            patch("classifier.service.run_inference", return_value="YES"),
            patch(
                "classifier.service.summarize_on_demand",
                return_value="",
            ),
        ):
            result = _svc.get_response_context(fresh_session, "Hello, who are you?")

        self.assertTrue(result["needs_context"])
        self.assertIsNotNone(result["context"])


# ===========================================================================
# Test Group 3: get_last_n_turns DB function (Tests 6, 7, 8)
# ===========================================================================

class TestGetLastNTurns(unittest.TestCase):
    """Tests 6-8: get_last_n_turns correctness, ordering, and edge cases."""

    def setUp(self):
        summarizer.init_service()

    def _conn(self):
        return _db.open_connection(_TEST_DB)

    # --- Test 6 ---
    def test_returns_turns_in_chronological_order(self):
        """Turns must be returned ascending by turn_index (chronological)."""
        session_id = "gln-order"
        _insert_turns_directly(session_id, [
            ("user",      "First message"),
            ("assistant", "First reply"),
            ("user",      "Second message"),
            ("assistant", "Second reply"),
            ("user",      "Third message"),
        ])
        conn = self._conn()
        try:
            turns = _db.get_last_n_turns(conn, session_id, 3)
        finally:
            conn.close()

        self.assertEqual(len(turns), 3)
        indices = [t.turn_index for t in turns]
        self.assertEqual(
            indices, sorted(indices),
            f"Turns must be in ASC (chronological) order; got {indices}",
        )
        # The last 3 of 5 should be turns 3, 4, 5.
        self.assertEqual(indices, [3, 4, 5])

    # --- Test 7 ---
    def test_empty_session_returns_empty_list(self):
        """A session with 0 turns must return [] without raising."""
        session_id = "gln-empty"
        conn = self._conn()
        try:
            _db.get_or_create_session(conn, session_id)
            conn.commit()
            turns = _db.get_last_n_turns(conn, session_id, 5)
        finally:
            conn.close()

        self.assertEqual(turns, [], "Empty session must return []")

    # --- Test 8 ---
    def test_fewer_turns_than_n_returns_all(self):
        """If the session has k < n turns, all k are returned (no crash)."""
        session_id = "gln-sparse"
        _insert_turns_directly(session_id, [
            ("user",      "Only turn 1"),
            ("assistant", "Only turn 2"),
        ])
        conn = self._conn()
        try:
            turns = _db.get_last_n_turns(conn, session_id, 10)
        finally:
            conn.close()

        self.assertEqual(
            len(turns), 2,
            "Should return all 2 available turns when n=10 but only 2 exist",
        )
        indices = [t.turn_index for t in turns]
        self.assertEqual(indices, sorted(indices), "Must still be ASC")

    def test_n_zero_returns_empty(self):
        """n=0 must return [] immediately without a DB query."""
        session_id = "gln-zero"
        _insert_turns_directly(session_id, [("user", "Something")])
        conn = self._conn()
        try:
            turns = _db.get_last_n_turns(conn, session_id, 0)
        finally:
            conn.close()
        self.assertEqual(turns, [])

    def test_returns_correct_texts(self):
        """The content of returned turns must match what was inserted."""
        session_id = "gln-content"
        _insert_turns_directly(session_id, [
            ("user", "Alpha"),
            ("assistant", "Beta"),
            ("user", "Gamma"),
        ])
        conn = self._conn()
        try:
            turns = _db.get_last_n_turns(conn, session_id, 2)
        finally:
            conn.close()

        texts = [t.text for t in turns]
        self.assertEqual(texts, ["Beta", "Gamma"])


# ===========================================================================
# Test Group 4: Prompt builder unit tests
# ===========================================================================

class TestPromptBuilder(unittest.TestCase):
    """Unit tests for classifier/prompt.py."""

    def _make_turn(self, idx, role, text) -> "summarizer.TurnRow":
        from summarizer.db import TurnRow
        return TurnRow(
            turn_id=idx,
            session_id="s",
            turn_index=idx,
            role=role,
            text=text,
            timestamp="2026-08-22T00:00:00Z",
        )

    def test_prompt_contains_gemma3_tokens(self):
        """Built prompt must include Gemma 3's chat-format control tokens."""
        from classifier.prompt import build_classifier_prompt
        turns = [self._make_turn(1, "user", "Hi")]
        prompt = build_classifier_prompt(turns, "What is the weather?")
        self.assertIn("<start_of_turn>user", prompt)
        self.assertIn("<end_of_turn>", prompt)
        self.assertIn("<start_of_turn>model", prompt)

    def test_prompt_ends_with_model_turn_token(self):
        """Prompt must end with <start_of_turn>model\\n to prime generation."""
        from classifier.prompt import build_classifier_prompt
        turns = [self._make_turn(1, "user", "Hi")]
        prompt = build_classifier_prompt(turns, "Something.")
        self.assertTrue(
            prompt.rstrip("\n").endswith("<start_of_turn>model") or
            "<start_of_turn>model" in prompt,
        )

    def test_no_bos_token_in_raw_string(self):
        """<bos> must NOT appear in the prompt string — llama.cpp adds it."""
        from classifier.prompt import build_classifier_prompt
        turns = [self._make_turn(1, "user", "Hello")]
        prompt = build_classifier_prompt(turns, "Test.")
        self.assertNotIn("<bos>", prompt, "<bos> must not be in the raw string")

    def test_empty_history_uses_placeholder(self):
        """No prior turns -> placeholder text appears, no crash."""
        from classifier.prompt import build_classifier_prompt, _NO_HISTORY_PLACEHOLDER
        prompt = build_classifier_prompt([], "First message ever.")
        self.assertIn(_NO_HISTORY_PLACEHOLDER, prompt)

    def test_history_turns_rendered_in_prompt(self):
        """Prior turns must appear in the prompt with correct role labels."""
        from classifier.prompt import build_classifier_prompt
        turns = [
            self._make_turn(3, "user",      "What is the deadline?"),
            self._make_turn(4, "assistant", "Friday the 22nd."),
        ]
        prompt = build_classifier_prompt(turns, "Can you confirm that date?")
        self.assertIn("[Turn 3] USER: What is the deadline?", prompt)
        self.assertIn("[Turn 4] ASSISTANT: Friday the 22nd.", prompt)
        self.assertIn("Can you confirm that date?", prompt)

    def test_current_prompt_in_output(self):
        """Current prompt text must appear verbatim in the built prompt."""
        from classifier.prompt import build_classifier_prompt
        current = "Remind me what the budget was."
        prompt = build_classifier_prompt([], current)
        self.assertIn(current.strip(), prompt)

    def test_yes_no_rule_in_system_block(self):
        """The system block must include both YES and NO instructions."""
        from classifier.prompt import _SYSTEM_BLOCK
        self.assertIn("YES", _SYSTEM_BLOCK)
        self.assertIn("NO", _SYSTEM_BLOCK)
        self.assertIn("doubt", _SYSTEM_BLOCK.lower(),
                      "Fail-safe 'when in doubt' instruction must be present")


# ===========================================================================
# Entry point
# ===========================================================================

def _patch_all_inference():
    """Patch both classifier and summarizer inference before any tests run."""
    p1 = patch(
        "classifier.service.run_inference",
        side_effect=lambda prompt: "YES",  # overridden per-test where needed
    )
    p2 = patch(
        "summarizer.service.run_inference",
        side_effect=_fake_summary_inference,
    )
    return p1, p2


if __name__ == "__main__":
    print("=" * 70)
    print("Classifier Test Suite")
    print(f"Database: {_TEST_DB}")
    print("Models:   STUBBED (no real GGUF required)")
    print("=" * 70)

    p1, p2 = _patch_all_inference()
    p1.start()
    p2.start()

    try:
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        suite.addTests(loader.loadTestsFromTestCase(TestGetLastNTurns))
        suite.addTests(loader.loadTestsFromTestCase(TestPromptBuilder))
        suite.addTests(loader.loadTestsFromTestCase(TestClassifyParsing))
        suite.addTests(loader.loadTestsFromTestCase(TestGetResponseContext))

        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
    finally:
        p1.stop()
        p2.stop()

    sys.exit(0 if result.wasSuccessful() else 1)
