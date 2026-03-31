"""
test_azure_tts.py — Unit tests for tts/azure_tts.py
  • _detect_tone: classification logic
  • build_ssml: SSML output structure
  • (HTTP calls are NOT tested — those require Azure credentials)

Run:
    pytest tests/test_azure_tts.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tts"))

from azure_tts import _detect_tone, build_ssml, AZURE_TTS_VOICE  # noqa


# ═══════════════════════════════════════════════════════════════════════════════
#  _detect_tone Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectTone:

    def test_question_mark(self):
        assert _detect_tone("How are you?") == "question"

    def test_exclamation_mark(self):
        assert _detect_tone("That is great!") == "cheerful"

    def test_greeting_hello(self):
        assert _detect_tone("Hello, welcome aboard") == "cheerful"

    def test_greeting_hi(self):
        assert _detect_tone("Hi there") == "cheerful"

    def test_greeting_welcome(self):
        assert _detect_tone("Welcome to our platform") == "cheerful"

    def test_empathy_sorry(self):
        assert _detect_tone("I'm sorry to hear that") == "empathetic"

    def test_empathy_understand(self):
        assert _detect_tone("I understand your concern about the pricing") == "empathetic"

    def test_empathy_sounds_like(self):
        assert _detect_tone("That sounds like a difficult situation") == "empathetic"

    def test_calm_short(self):
        assert _detect_tone("Sure thing") == "calm"

    def test_calm_default(self):
        assert _detect_tone("Our platform offers comprehensive business solutions for enterprise clients") == "calm"

    def test_empty_string(self):
        result = _detect_tone("")
        assert result in ("calm", "cheerful", "question", "empathetic")


# ═══════════════════════════════════════════════════════════════════════════════
#  build_ssml Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildSSML:

    def test_returns_string(self):
        ssml = build_ssml("Hello there")
        assert isinstance(ssml, str)

    def test_contains_speak_tag(self):
        ssml = build_ssml("Test")
        assert "<speak" in ssml
        assert "</speak>" in ssml

    def test_contains_voice_name(self):
        ssml = build_ssml("Test")
        assert AZURE_TTS_VOICE in ssml

    def test_contains_mstts_express(self):
        ssml = build_ssml("Test")
        assert "mstts:express-as" in ssml

    def test_contains_prosody(self):
        ssml = build_ssml("Test")
        assert "<prosody" in ssml

    def test_question_tone_style(self):
        ssml = build_ssml("How are you?", tone="question")
        assert 'style="friendly"' in ssml
        assert 'pitch="+5%"' in ssml

    def test_cheerful_tone_style(self):
        ssml = build_ssml("Great news!", tone="cheerful")
        assert 'style="cheerful"' in ssml
        assert 'rate="+5%"' in ssml

    def test_empathetic_tone_style(self):
        ssml = build_ssml("I understand", tone="empathetic")
        assert 'style="empathetic"' in ssml
        assert 'rate="-5%"' in ssml

    def test_calm_tone_style(self):
        ssml = build_ssml("OK", tone="calm")
        assert 'style="chat"' in ssml

    def test_auto_detects_tone_if_none(self):
        ssml = build_ssml("How can I help you?")
        # Should auto-detect "question"
        assert 'style="friendly"' in ssml

    def test_escapes_special_characters(self):
        ssml = build_ssml('Price is < $100 & "free"')
        assert "&lt;" in ssml
        assert "&amp;" in ssml
        # xml.sax.saxutils.escape() does not escape double quotes by default
        assert "$100" in ssml

    def test_xml_lang_english(self):
        ssml = build_ssml("Test")
        assert 'xml:lang="en-US"' in ssml

    def test_xmlns_mstts(self):
        ssml = build_ssml("Test")
        assert "xmlns:mstts" in ssml
