"""
test_echo_gate.py — Unit tests for gateway/echo_gate.py
  • AITextEchoFilter: feed, is_echo_word, is_echo_segment
  • TimingEchoGate: feed_tts, tts_stopped, check, echo tail

Run:
    pytest tests/test_echo_gate.py -v
"""

import sys
import os
import time
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gateway"))

from echo_gate import AITextEchoFilter, TimingEchoGate, _ECHO_STOP  # noqa


# ═══════════════════════════════════════════════════════════════════════════════
#  AITextEchoFilter Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAITextEchoFilter:

    def test_init_empty(self):
        f = AITextEchoFilter()
        assert not f._ai_words

    def test_feed_ai_text_populates_words(self):
        f = AITextEchoFilter()
        f.feed_ai_text("Hello how are you")
        assert len(f._ai_words) == 4

    def test_is_echo_word_matches_ai_speech(self):
        f = AITextEchoFilter()
        f.feed_ai_text("Welcome to AskNova")
        assert f.is_echo_word("welcome") is True
        assert f.is_echo_word("asknova") is True

    def test_is_echo_word_ignores_stop_words(self):
        f = AITextEchoFilter()
        f.feed_ai_text("I am the best")
        # Only words in _ECHO_STOP are ignored; "am" is NOT in the stop set
        for stop in ("i", "the"):
            assert f.is_echo_word(stop) is False
        # "am" is not a stop word in the filter, so it matches as echo
        assert f.is_echo_word("am") is True

    def test_is_echo_word_negative(self):
        f = AITextEchoFilter()
        f.feed_ai_text("Hello world")
        assert f.is_echo_word("goodbye") is False

    def test_is_echo_word_empty_returns_false(self):
        f = AITextEchoFilter()
        assert f.is_echo_word("") is False

    def test_is_echo_word_strips_punctuation(self):
        f = AITextEchoFilter()
        f.feed_ai_text("Hello!")
        assert f.is_echo_word("hello.") is True
        assert f.is_echo_word("HELLO?") is True

    def test_is_any_ai_word_includes_stop_words(self):
        f = AITextEchoFilter()
        f.feed_ai_text("I am here")
        assert f.is_any_ai_word("am") is True
        assert f.is_any_ai_word("here") is True

    def test_is_echo_segment_high_overlap(self):
        f = AITextEchoFilter()
        f.feed_ai_text("Welcome to our platform for business solutions")
        # Repeating AI words should be detected as echo
        assert f.is_echo_segment("welcome platform business solutions") is True

    def test_is_echo_segment_low_overlap(self):
        f = AITextEchoFilter()
        f.feed_ai_text("Welcome to AskNova")
        assert f.is_echo_segment("I want to buy a car") is False

    def test_is_echo_segment_empty(self):
        f = AITextEchoFilter()
        assert f.is_echo_segment("") is False

    def test_is_echo_segment_repeated_ai_word(self):
        """Repeated single word that IS in AI speech → echo."""
        f = AITextEchoFilter()
        f.feed_ai_text("Hello there friend")
        assert f.is_echo_segment("hello hello hello") is True

    def test_is_echo_segment_repeated_non_ai_word(self):
        """Repeated word NOT in AI speech → not echo."""
        f = AITextEchoFilter()
        f.feed_ai_text("Welcome to AskNova")
        assert f.is_echo_segment("goodbye goodbye goodbye") is False

    def test_is_echo_segment_all_stop_words_in_ai(self):
        f = AITextEchoFilter()
        f.feed_ai_text("I am the one")
        # "I am the" are all stop words that appear in AI speech
        assert f.is_echo_segment("I am the") is True

    def test_is_echo_segment_all_stop_words_not_in_ai(self):
        f = AITextEchoFilter()
        # No AI speech fed → no matches
        assert f.is_echo_segment("I am the") is False

    def test_reset_clears_words(self):
        f = AITextEchoFilter()
        f.feed_ai_text("Hello world")
        f.reset()
        assert f.is_echo_word("hello") is False

    def test_expiry_removes_old_words(self):
        f = AITextEchoFilter()
        f.feed_ai_text("old words here")
        # Manually age the words beyond the window
        cutoff = time.monotonic() - 120  # way past 60s window
        f._ai_words = [(cutoff, w) for _, w in f._ai_words]
        assert f.is_echo_word("old") is False


# ═══════════════════════════════════════════════════════════════════════════════
#  TimingEchoGate Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimingEchoGate:

    def test_init_not_armed(self):
        g = TimingEchoGate()
        assert g._is_armed() is False

    def test_feed_tts_arms_gate(self):
        g = TimingEchoGate()
        g.feed_tts(b"\x00" * 1024)
        assert g._tts_active is True
        assert g._is_armed() is True

    def test_tts_stopped_starts_tail(self):
        g = TimingEchoGate()
        g.feed_tts(b"\x00" * 1024)
        g.tts_stopped()
        assert g._tts_active is False
        assert g._is_armed() is True  # still in echo tail

    def test_check_drops_frame_when_armed_not_speaking(self):
        """During echo tail (not ai_speaking), frames should be dropped."""
        g = TimingEchoGate()
        g.feed_tts(b"\x00" * 1024)
        g.tts_stopped()
        # Still in echo tail
        result = g.check(b"\x01" * 320, ai_speaking=False)
        assert result is True  # dropped

    def test_check_passes_frame_when_ai_speaking(self):
        """During active AI speaking, let audio through for barge-in detection."""
        g = TimingEchoGate()
        g.feed_tts(b"\x00" * 1024)
        result = g.check(b"\x01" * 320, ai_speaking=True)
        assert result is False  # passed through

    def test_check_passes_when_not_armed(self):
        g = TimingEchoGate()
        result = g.check(b"\x01" * 320, ai_speaking=False)
        assert result is False

    def test_check_empty_pcm(self):
        g = TimingEchoGate()
        g.feed_tts(b"\x00" * 1024)
        result = g.check(b"", ai_speaking=False)
        assert result is False  # empty → pass

    def test_echo_tail_expires(self):
        g = TimingEchoGate()
        g.feed_tts(b"\x00" * 1024)
        g.tts_stopped()
        # Manually set stopped time way in the past
        g._tts_stopped_at = time.monotonic() - 10.0
        assert g._is_armed() is False

    def test_reset_clears_state(self):
        g = TimingEchoGate()
        g.feed_tts(b"\x00" * 1024)
        g.reset()
        assert g._tts_active is False
        assert g.frames_checked == 0
        assert g.frames_dropped == 0

    def test_frame_counters(self):
        g = TimingEchoGate()
        g.feed_tts(b"\x00" * 1024)
        g.tts_stopped()
        g.check(b"\x01" * 320, ai_speaking=False)  # dropped
        g.check(b"\x01" * 320, ai_speaking=False)  # dropped
        assert g.frames_checked == 2
        assert g.frames_dropped == 2

    def test_is_enrolled_always_true(self):
        g = TimingEchoGate()
        assert g.is_enrolled is True

    def test_backward_compat_aliases(self):
        from echo_gate import GatewayEchoGate, TTSVoiceFingerprintGate
        assert GatewayEchoGate is TimingEchoGate
        assert TTSVoiceFingerprintGate is TimingEchoGate
