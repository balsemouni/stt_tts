"""
test_models.py — Unit tests for gateway/models.py
  • State enum
  • RepetitionGuard: single-word repetition, n-gram patterns, stop words, reset
  • drain_q helper

Run:
    pytest tests/test_models.py -v
"""

import sys
import os
import asyncio
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gateway"))

from models import State, RepetitionGuard, drain_q  # noqa


# ═══════════════════════════════════════════════════════════════════════════════
#  State Enum Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestState:
    def test_idle_exists(self):
        assert State.IDLE is not None

    def test_thinking_exists(self):
        assert State.THINKING is not None

    def test_speaking_exists(self):
        assert State.SPEAKING is not None

    def test_states_are_distinct(self):
        assert State.IDLE != State.THINKING
        assert State.THINKING != State.SPEAKING
        assert State.IDLE != State.SPEAKING


# ═══════════════════════════════════════════════════════════════════════════════
#  RepetitionGuard Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRepetitionGuard:

    def test_no_repetition_returns_false(self):
        g = RepetitionGuard()
        assert g.feed("hello") is False
        assert g.feed("world") is False
        assert g.feed("how") is False

    def test_single_word_repetition_detected(self):
        """3 repetitions of same word → hallucination."""
        g = RepetitionGuard(threshold=3)
        g.feed("hello")
        g.feed("hello")
        result = g.feed("hello")
        assert result is True

    def test_stop_words_ignored(self):
        g = RepetitionGuard(threshold=3)
        # Stop words should not trigger
        assert g.feed("the") is False
        assert g.feed("the") is False
        assert g.feed("the") is False

    def test_stop_words_list(self):
        """Verify common stop words are in the set."""
        for w in ("a", "the", "is", "are", "to", "of", "and", "or", "but"):
            assert w in RepetitionGuard._STOP

    def test_word_normalization(self):
        g = RepetitionGuard(threshold=3)
        g.feed("Hello,")
        g.feed("HELLO!")
        result = g.feed("hello.")
        assert result is True

    def test_bigram_repetition(self):
        """Repeating bigram pattern ≥3 times triggers."""
        g = RepetitionGuard()
        words = ["ask", "nova"] * 4  # "ask nova" repeated 4 times
        results = [g.feed(w) for w in words]
        assert any(results), "Should detect repeating bigram"

    def test_trigram_repetition(self):
        """Repeating trigram pattern ≥3 times triggers."""
        g = RepetitionGuard()
        words = ["great", "business", "solution"] * 4
        results = [g.feed(w) for w in words]
        assert any(results), "Should detect repeating trigram"

    def test_mixed_content_no_false_positive(self):
        g = RepetitionGuard()
        for w in "Our platform offers comprehensive business solutions for enterprises".split():
            assert g.feed(w) is False

    def test_window_limits_history(self):
        g = RepetitionGuard(window=5, threshold=3)
        # Fill window with unique words
        for w in ["alpha", "beta", "gamma", "delta", "epsilon"]:
            g.feed(w)
        # Now the window is full of unique words; "alpha" from position 0 is gone
        assert len(g._history) <= 5

    def test_reset_clears_history(self):
        g = RepetitionGuard()
        g.feed("hello")
        g.feed("hello")
        g.reset()
        assert g._history == []
        assert g._phrase_history == []
        # Should not trigger after reset
        assert g.feed("hello") is False

    def test_threshold_minimum_is_3(self):
        g = RepetitionGuard(threshold=1)
        assert g._threshold == 3  # min enforced


# ═══════════════════════════════════════════════════════════════════════════════
#  drain_q Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDrainQ:
    def test_drain_empty_queue(self):
        q = asyncio.Queue()
        drain_q(q)
        assert q.empty()

    def test_drain_full_queue(self):
        q = asyncio.Queue()
        for i in range(5):
            q.put_nowait(i)
        drain_q(q)
        assert q.empty()

    def test_drain_returns_nothing(self):
        q = asyncio.Queue()
        q.put_nowait("item")
        result = drain_q(q)
        assert result is None
