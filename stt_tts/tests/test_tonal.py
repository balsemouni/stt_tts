"""
test_tonal.py — Unit tests for gateway/tonal.py
  • ChunkTone classification
  • TonalAccumulator: feed, flush, sentence splitting, clause breaks, first-chunk

Run:
    pytest tests/test_tonal.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gateway"))

from tonal import (  # noqa
    ChunkTone, TonalChunk, TonalAccumulator,
    classify_tone, FIRST_CHUNK_CHARS, MIN_TTS_CHARS,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  classify_tone Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyTone:
    def test_question_mark(self):
        assert classify_tone("How are you?") == ChunkTone.TONE

    def test_exclamation_mark(self):
        assert classify_tone("That's great!") == ChunkTone.TONE

    def test_short_text_is_tone(self):
        assert classify_tone("Hello there") == ChunkTone.TONE

    def test_long_text_is_logic(self):
        text = "This is a very long statement that exceeds the tone maximum character limit for classification purposes"
        assert classify_tone(text) == ChunkTone.LOGIC

    def test_exact_boundary(self):
        text = "x" * 60
        assert classify_tone(text) == ChunkTone.TONE
        text = "x" * 61
        assert classify_tone(text) == ChunkTone.LOGIC


# ═══════════════════════════════════════════════════════════════════════════════
#  TonalAccumulator Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTonalAccumulator:

    def test_empty_token_returns_nothing(self):
        acc = TonalAccumulator()
        assert acc.feed("") == []

    def test_first_chunk_fires_at_threshold(self):
        """First chunk should fire once buffer has enough chars + word boundary."""
        acc = TonalAccumulator()
        # Feed tokens until we exceed FIRST_CHUNK_CHARS
        results = []
        for word in ["Hello", " how", " are", " you"]:
            results.extend(acc.feed(word))
        # Should have produced at least one chunk
        assert len(results) >= 1
        assert all(isinstance(r, TonalChunk) for r in results)

    def test_sentence_end_flushes(self):
        acc = TonalAccumulator()
        results = []
        for token in ["Hello", " there.", " How", " are"]:
            results.extend(acc.feed(token))
        # "Hello there." should produce a chunk
        sentence_chunks = [r for r in results if r.text.endswith(".")]
        assert len(sentence_chunks) >= 1

    def test_question_mark_flushes(self):
        acc = TonalAccumulator()
        results = []
        # First, fill past the first-chunk threshold
        for token in ["Tell", " me", " something.", " How", " are", " you?"]:
            results.extend(acc.feed(token))
        question_chunks = [r for r in results if "?" in r.text]
        assert len(question_chunks) >= 1

    def test_exclamation_mark_flushes(self):
        acc = TonalAccumulator()
        results = []
        for token in ["First", " sentence.", " That", " is", " great!"]:
            results.extend(acc.feed(token))
        excl_chunks = [r for r in results if "!" in r.text]
        assert len(excl_chunks) >= 1

    def test_clause_break_comma(self):
        acc = TonalAccumulator()
        acc._first_sent = False  # skip first-chunk logic
        results = []
        # Feed enough text with a comma clause break
        text = "When you consider the overall architecture of the system, the design patterns become clear"
        for word in text.split():
            results.extend(acc.feed(" " + word))
        # Should produce at least one chunk at the comma
        assert len(results) >= 1

    def test_flush_returns_remaining(self):
        acc = TonalAccumulator()
        acc.feed("Hello")
        chunk = acc.flush()
        assert chunk is not None
        assert chunk.text == "Hello"
        assert isinstance(chunk.tone, ChunkTone)

    def test_flush_empty_returns_none(self):
        acc = TonalAccumulator()
        assert acc.flush() is None

    def test_reset_clears_state(self):
        acc = TonalAccumulator()
        acc.feed("Some text here")
        acc.reset()
        assert acc._buf == ""
        assert acc._first_sent is True

    def test_no_double_spaces(self):
        acc = TonalAccumulator()
        acc.feed("Hello ")
        acc.feed(" world")
        chunk = acc.flush()
        assert "  " not in chunk.text

    def test_long_text_hard_cap(self):
        """Extremely long text without punctuation should still produce chunks."""
        acc = TonalAccumulator()
        acc._first_sent = False
        results = []
        for i in range(50):
            results.extend(acc.feed(f" word{i}"))
        # With 50 words (~300 chars), should hit hard cap
        assert len(results) >= 1

    def test_tone_classification_on_chunks(self):
        acc = TonalAccumulator()
        results = []
        for token in ["Is", " this", " working?"]:
            results.extend(acc.feed(token))
        remaining = acc.flush()
        if remaining:
            results.append(remaining)
        # At least one chunk should exist with TONE classification
        tones = [r.tone for r in results]
        assert ChunkTone.TONE in tones

    def test_multiple_sentences(self):
        acc = TonalAccumulator()
        results = []
        text = "First sentence. Second sentence. Third one."
        for word in text.split():
            results.extend(acc.feed(" " + word))
        remaining = acc.flush()
        if remaining:
            results.append(remaining)
        assert len(results) >= 2

    def test_punctuation_at_start_no_extra_space(self):
        acc = TonalAccumulator()
        acc.feed("Hello")
        acc.feed(",")
        chunk = acc.flush()
        # Comma should attach to Hello without space
        assert chunk.text == "Hello,"
