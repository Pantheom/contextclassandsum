"""
tests/test_summarizer.py
------------------------
Standalone test script for the context summarizer service.

Can be run directly:
    python tests/test_summarizer.py

Or via pytest:
    python -m pytest tests/test_summarizer.py -v

Tests cover:
    A. Periodic threshold fires at exactly turn 15.
    B. On-demand works standalone (below threshold, no auto-fire).
    C. No gaps: on-demand after periodic keeps all turns covered.
    D. Double-trigger guard: second concurrent bg job is suppressed.

The real Phi-4-mini model is NOT required. The test patches
summarizer.model.run_inference with a fast stub before any test runs.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Path setup — allow running from the repo root without installing the package.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Point the service at a temporary database BEFORE importing the package,
# so the module-level cfg singleton reads the right value.
# ---------------------------------------------------------------------------
_tmp_dir = tempfile.mkdtemp(prefix="summarizer_test_")
_TEST_DB = os.path.join(_tmp_dir, "test_summarizer.db")
os.environ["SUMMARIZER_DB_PATH"] = _TEST_DB

# Stub out the model path so config validation passes (model.py validates on
# first get_model() call, not at import time — but set it anyway for clarity).
os.environ["SUMMARIZER_MODEL_PATH"] = "/stub/phi4mini.gguf"

# Keep periodic threshold at default (15) unless overridden.
os.environ.setdefault("SUMMARIZER_PERIODIC_THRESHOLD", "15")

# ---------------------------------------------------------------------------
# Now safe to import — env vars are in place.
# ---------------------------------------------------------------------------
import summarizer                          # noqa: E402
from summarizer import db as _db           # noqa: E402
from summarizer import service as _svc    # noqa: E402
from summarizer.model import _ModelSingleton  # noqa: E402


# ---------------------------------------------------------------------------
# Test utilities
# ---------------------------------------------------------------------------

_FAKE_SUMMARY_COUNTER = 0

def _fake_inference(prompt: str, max_tokens: int | None = None) -> str:
    """Mock inference: returns a deterministic, non-empty summary string."""
    global _FAKE_SUMMARY_COUNTER
    _FAKE_SUMMARY_COUNTER += 1
    return f"Fake summary #{_FAKE_SUMMARY_COUNTER}: {prompt[:80].replace(chr(10), ' ')}..."


def _fresh_db_conn():
    """Open a connection to the test database."""
    conn = _db.open_connection(_TEST_DB)
    return conn


def _get_session_state(session_id: str):
    """Return (current_summary, last_summarized_turn_index) from the DB."""
    conn = _fresh_db_conn()
    try:
        row = conn.execute(
            "SELECT current_summary, last_summarized_turn_index "
            "FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None, 0
        return row["current_summary"], row["last_summarized_turn_index"]
    finally:
        conn.close()


def _seed_turns(session_id: str, count: int, start_index: int = 1) -> None:
    """Write `count` alternating user/assistant turns starting at start_index.

    Uses write_turn() so the periodic trigger logic runs normally.
    Does NOT wait for any background jobs — callers do that explicitly.
    """
    for i in range(start_index, start_index + count):
        role = "user" if i % 2 == 1 else "assistant"
        _svc.write_turn(session_id, role, f"Turn content for turn {i}")


def _wait_for_summary(session_id: str, expected_last_index: int, timeout: float = 10.0) -> bool:
    """Poll until last_summarized_turn_index == expected_last_index or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _, last_idx = _get_session_state(session_id)
        if last_idx >= expected_last_index:
            return True
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestPeriodicThreshold(unittest.TestCase):
    """Test A: Periodic background summarisation fires at exactly turn 15."""

    SESSION_ID = "test-session-A"

    def setUp(self):
        summarizer.init_service()

    def test_periodic_threshold_flow(self):
        """Sequential: 14 turns must NOT fire; turn 15 MUST fire.

        Both checks are in one method to guarantee ordering independent of
        how the test runner sorts method names alphabetically.
        """
        # --- Phase 1: write 14 turns, confirm no auto-fire ---
        _seed_turns(self.SESSION_ID, 14)
        time.sleep(0.3)  # Give any (incorrect) background job time to run.
        summary, last_idx = _get_session_state(self.SESSION_ID)
        self.assertEqual(
            last_idx, 0,
            f"Expected last_summarized_turn_index=0 after 14 turns, got {last_idx}",
        )
        self.assertIn(
            summary, (None, ""),
            f"Expected no summary after 14 turns, got: {summary!r}",
        )

        # --- Phase 2: write turn 15, confirm periodic fires ---
        _svc.write_turn(self.SESSION_ID, "user", "Turn 15 content")
        fired = _wait_for_summary(self.SESSION_ID, expected_last_index=15)
        self.assertTrue(
            fired,
            "Periodic summarisation did not fire within timeout after turn 15",
        )
        summary, last_idx = _get_session_state(self.SESSION_ID)
        self.assertEqual(last_idx, 15)
        self.assertTrue(summary, "Summary should be non-empty after periodic fire")
        print(f"  [A] last_idx={last_idx}, summary_len={len(summary or '')}")


class TestOnDemandStandalone(unittest.TestCase):
    """Test B: on-demand works below threshold, updates DB correctly."""

    SESSION_ID = "test-session-B"

    def setUp(self):
        summarizer.init_service()

    def test_on_demand_below_threshold(self):
        """5 turns (below threshold), then on-demand -> summary produced."""
        _seed_turns(self.SESSION_ID, 5)
        # Confirm periodic did NOT fire.
        time.sleep(0.2)
        _, last_idx = _get_session_state(self.SESSION_ID)
        self.assertEqual(last_idx, 0, "Periodic should not have fired for 5 turns")

        result = summarizer.summarize_on_demand(self.SESSION_ID)

        self.assertTrue(result, "summarize_on_demand should return a non-empty string")
        _, last_idx = _get_session_state(self.SESSION_ID)
        self.assertEqual(
            last_idx, 5,
            f"last_summarized_turn_index should be 5 after on-demand, got {last_idx}",
        )
        print(f"  [B] last_idx={last_idx}, result_len={len(result)}")

    def test_on_demand_return_value_matches_db(self):
        """Return value of summarize_on_demand must equal what's in the DB."""
        result = summarizer.summarize_on_demand(self.SESSION_ID)
        db_summary, _ = _get_session_state(self.SESSION_ID)
        self.assertEqual(result, db_summary)


class TestNoGapsCounterReset(unittest.TestCase):
    """Test C: No gaps between periodic and on-demand, counter resets correctly."""

    SESSION_ID = "test-session-C"

    def setUp(self):
        summarizer.init_service()

    def test_no_gaps_after_combined_flow(self):
        """15 turns -> periodic fires -> 10 more turns -> on-demand covers all."""
        # Phase 1: write 15 turns, wait for periodic.
        _seed_turns(self.SESSION_ID, 15)
        fired = _wait_for_summary(self.SESSION_ID, expected_last_index=15)
        self.assertTrue(fired, "Periodic did not fire after 15 turns")
        _, last_after_periodic = _get_session_state(self.SESSION_ID)
        self.assertEqual(last_after_periodic, 15)

        # Phase 2: write 10 more turns (gap = 10, below threshold).
        _seed_turns(self.SESSION_ID, 10, start_index=16)
        time.sleep(0.2)
        _, last_after_more = _get_session_state(self.SESSION_ID)
        self.assertEqual(
            last_after_more, 15,
            "Periodic should NOT fire again for only 10 new turns",
        )

        # Phase 3: on-demand should cover turns 16–25 (no gap).
        result = summarizer.summarize_on_demand(self.SESSION_ID)
        _, last_after_ondemand = _get_session_state(self.SESSION_ID)
        self.assertEqual(
            last_after_ondemand, 25,
            f"last_summarized_turn_index should be 25 after on-demand, got {last_after_ondemand}",
        )
        self.assertTrue(result)

        # Verify no gap: total turns in DB should be 25.
        conn = _fresh_db_conn()
        try:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM turns WHERE session_id = ?",
                (self.SESSION_ID,),
            ).fetchone()["c"]
        finally:
            conn.close()
        self.assertEqual(count, 25, f"Expected 25 total turns, found {count}")
        print(
            f"  [C] total_turns={count}, last_idx={last_after_ondemand}, "
            f"result_len={len(result)}"
        )


class TestDoubleTriggerGuard(unittest.TestCase):
    """Test D: Second concurrent background job for the same session is suppressed."""  # noqa: E501

    SESSION_ID = "test-session-D"

    def setUp(self):
        summarizer.init_service()

    def test_second_concurrent_job_skipped(self):
        """If a bg job is running, a second trigger must not start another job."""
        # Use a list as a mutable, thread-shared counter.
        # threading.local() only stores per-thread values — it cannot be read
        # by the main thread for values written in a worker thread.
        call_count = [0]
        count_lock = threading.Lock()
        barrier = threading.Event()

        def _slow_inference(prompt: str, max_tokens=None) -> str:
            """Slow stub: records calls and blocks until barrier is set."""
            with count_lock:
                call_count[0] += 1
            # Block to simulate slow inference so the second trigger arrives
            # before the first job finishes.
            barrier.wait(timeout=5.0)
            return f"Slow summary call #{call_count[0]}"

        # Seed 14 turns first (no trigger yet).
        _seed_turns(self.SESSION_ID, 14)
        time.sleep(0.1)

        with patch("summarizer.service.run_inference", side_effect=_slow_inference):
            # Turn 15 -> fires first background job (which will block).
            _svc.write_turn(self.SESSION_ID, "user", "Turn 15")
            # Wait long enough for the bg thread to enter _slow_inference
            # and acquire the session lock before turn 16 is written.
            time.sleep(0.3)

            # Turn 16 -> gap is still >= 15 (last_idx still 0 since job is blocked).
            # But the session lock is held by the first job -> second job suppressed.
            _svc.write_turn(self.SESSION_ID, "user", "Turn 16")
            time.sleep(0.1)

            # Release the barrier so the first job can finish.
            barrier.set()
            time.sleep(1.0)  # Wait for first job to complete.

        # Only one inference call should have been made.
        total_calls = call_count[0]
        self.assertEqual(
            total_calls, 1,
            f"Expected exactly 1 inference call, got {total_calls} "
            "(second concurrent job was not suppressed)",
        )
        print(f"  [D] inference_calls={total_calls} (expected 1)")


# ---------------------------------------------------------------------------
# Additional unit tests for prompt and DB helpers
# ---------------------------------------------------------------------------

class TestPromptBuilder(unittest.TestCase):
    """Verify the prompt template produces the expected structure."""

    def test_prompt_contains_system_tokens(self):
        from summarizer.prompt import build_prompt
        from summarizer.db import TurnRow

        turns = [
            TurnRow(1, "s", 1, "user", "Hello", "2026-01-01T00:00:00Z"),
            TurnRow(2, "s", 2, "assistant", "Hi there", "2026-01-01T00:00:01Z"),
        ]
        prompt = build_prompt(None, turns, 1, 2)
        self.assertIn("<|system|>", prompt)
        self.assertIn("<|user|>", prompt)
        self.assertIn("<|assistant|>", prompt)
        self.assertIn("<|end|>", prompt)

    def test_prompt_ends_with_assistant_token(self):
        from summarizer.prompt import build_prompt
        from summarizer.db import TurnRow

        turns = [TurnRow(1, "s", 1, "user", "Test", "2026-01-01T00:00:00Z")]
        prompt = build_prompt("previous summary text", turns, 1, 1)
        self.assertTrue(
            prompt.rstrip().endswith("<|assistant|>") or "<|assistant|>" in prompt,
            "Prompt must end with <|assistant|> to prime generation",
        )

    def test_empty_previous_summary_uses_placeholder(self):
        from summarizer.prompt import build_prompt, _EMPTY_SUMMARY_PLACEHOLDER
        from summarizer.db import TurnRow

        turns = [TurnRow(1, "s", 1, "user", "Hi", "2026-01-01T00:00:00Z")]
        prompt = build_prompt(None, turns, 1, 1)
        self.assertIn(_EMPTY_SUMMARY_PLACEHOLDER, prompt)

    def test_previous_summary_included_when_present(self):
        from summarizer.prompt import build_prompt
        from summarizer.db import TurnRow

        turns = [TurnRow(1, "s", 1, "user", "Hi", "2026-01-01T00:00:00Z")]
        prompt = build_prompt("The user discussed Project Alpha.", turns, 1, 1)
        self.assertIn("Project Alpha", prompt)

    def test_turn_format(self):
        from summarizer.prompt import format_turns
        from summarizer.db import TurnRow

        turns = [
            TurnRow(1, "s", 3, "user", "What is the deadline?", "2026-01-01T00:00:00Z"),
            TurnRow(2, "s", 4, "assistant", "Friday the 22nd.", "2026-01-01T00:00:01Z"),
        ]
        formatted = format_turns(turns)
        self.assertIn("[Turn 3] USER: What is the deadline?", formatted)
        self.assertIn("[Turn 4] ASSISTANT: Friday the 22nd.", formatted)


class TestDBHelpers(unittest.TestCase):
    """Unit tests for the database helpers."""

    def _conn(self):
        conn = _db.open_connection(_TEST_DB)
        _db.init_db(conn)
        return conn

    def test_insert_turn_monotonic(self):
        session_id = "db-test-monotonic"
        conn = self._conn()
        try:
            idx1 = _db.insert_turn(conn, session_id, "user", "First")
            idx2 = _db.insert_turn(conn, session_id, "assistant", "Second")
            idx3 = _db.insert_turn(conn, session_id, "user", "Third")
        finally:
            conn.close()
        self.assertEqual(idx1, 1)
        self.assertEqual(idx2, 2)
        self.assertEqual(idx3, 3)

    def test_get_turns_after_boundary(self):
        """get_turns_after(after_index=2) must return turns 3, 4, 5 only."""
        session_id = "db-test-boundary"
        conn = self._conn()
        try:
            for i in range(5):
                role = "user" if i % 2 == 0 else "assistant"
                _db.insert_turn(conn, session_id, role, f"turn {i+1}")
            turns = _db.get_turns_after(conn, session_id, after_index=2)
        finally:
            conn.close()
        indices = [t.turn_index for t in turns]
        self.assertEqual(indices, [3, 4, 5])

    def test_update_summary_is_single_write_path(self):
        """update_summary must advance last_summarized_turn_index atomically."""
        session_id = "db-test-update"
        conn = self._conn()
        try:
            _db.get_or_create_session(conn, session_id)
            _db.insert_turn(conn, session_id, "user", "Hello")
            _db.insert_turn(conn, session_id, "assistant", "Hi")
            _db.update_summary(conn, session_id, "Test summary text", 2)
            row = conn.execute(
                "SELECT current_summary, last_summarized_turn_index "
                "FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        finally:
            conn.close()
        self.assertEqual(row["current_summary"], "Test summary text")
        self.assertEqual(row["last_summarized_turn_index"], 2)

    def test_invalid_role_raises(self):
        session_id = "db-test-role"
        conn = self._conn()
        try:
            with self.assertRaises(ValueError):
                _db.insert_turn(conn, session_id, "system", "Invalid role")
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _patch_model():
    """Apply the fake inference patch globally before all tests run."""
    patcher = patch("summarizer.service.run_inference", side_effect=_fake_inference)
    patcher.start()
    return patcher


if __name__ == "__main__":
    print("=" * 70)
    print("Context Summarizer Test Suite")
    print(f"Database: {_TEST_DB}")
    print("Model: STUBBED (no real GGUF required)")
    print("=" * 70)

    # Apply mock before any tests run.
    patcher = _patch_model()

    try:
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()

        # Ordered so that session A's two sub-tests share state correctly.
        suite.addTests(loader.loadTestsFromTestCase(TestDBHelpers))
        suite.addTests(loader.loadTestsFromTestCase(TestPromptBuilder))
        suite.addTests(loader.loadTestsFromTestCase(TestPeriodicThreshold))
        suite.addTests(loader.loadTestsFromTestCase(TestOnDemandStandalone))
        suite.addTests(loader.loadTestsFromTestCase(TestNoGapsCounterReset))
        suite.addTests(loader.loadTestsFromTestCase(TestDoubleTriggerGuard))

        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
    finally:
        patcher.stop()

    sys.exit(0 if result.wasSuccessful() else 1)
