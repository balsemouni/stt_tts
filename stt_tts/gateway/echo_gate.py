"""
echo_gate.py — Echo suppression (timing gate + text-layer filter)
"""
from __future__ import annotations

import logging
import os
import time

log = logging.getLogger("gateway")

# ─── Configuration ────────────────────────────────────────────────────────────

AI_TEXT_ECHO_WINDOW_S  = float(os.getenv("AI_TEXT_ECHO_WINDOW_S",  "3.0"))   # 3s — only catch immediate speaker echo
AI_TEXT_ECHO_RATIO     = float(os.getenv("AI_TEXT_ECHO_RATIO",     "0.65"))  # require high overlap to drop
AI_TEXT_ECHO_MIN_WORDS = int(os.getenv("AI_TEXT_ECHO_MIN_WORDS",   "3"))
ECHO_TAIL_S            = float(os.getenv("ECHO_TAIL_S",           "2.0"))   # echo timing gate tail

_ECHO_STOP = frozenset({
    "a", "an", "the", "i", "me", "my", "we", "our", "you", "your",
    "is", "are", "was", "were", "to", "of", "in", "on", "at", "by",
    "and", "or", "but", "not", "so", "it", "its", "be", "do", "did",
})


# ─── AI Text Echo Filter ─────────────────────────────────────────────────────

class AITextEchoFilter:
    """
    Text-level echo filter.  Records words the AI speaks; drops STT results
    that are predominantly drawn from recent AI speech.
    """

    def __init__(self):
        self._ai_words: list[tuple[float, str]] = []

    def feed_ai_text(self, text: str):
        """Call with every sentence/chunk the AI sends to TTS."""
        now = time.monotonic()
        words = [w.lower().strip(".,!?;:\"'") for w in text.split()]
        for w in words:
            if w:
                self._ai_words.append((now, w))
        self._expire()

    def _expire(self):
        cutoff = time.monotonic() - AI_TEXT_ECHO_WINDOW_S
        self._ai_words = [(t, w) for t, w in self._ai_words if t >= cutoff]

    def _recent_ai_set(self) -> frozenset:
        self._expire()
        return frozenset(w for _, w in self._ai_words)

    def is_echo_word(self, word: str) -> bool:
        """Return True if a single STT word came from recent AI speech."""
        w = word.lower().strip(".,!?;:\"'")
        if not w or w in _ECHO_STOP:
            return False
        return w in self._recent_ai_set()

    def is_any_ai_word(self, word: str) -> bool:
        """Return True if word (including stop words) appears in recent AI speech.
        Used during SPEAKING state for aggressive echo filtering."""
        w = word.lower().strip(".,!?;:\"'")
        if not w:
            return False
        return w in self._recent_ai_set()

    def is_echo_segment(self, text: str) -> bool:
        """Return True if an STT segment is predominantly composed of recent AI words."""
        if not text:
            return False
        words = [w.lower().strip(".,!?;:\"'") for w in text.split()]
        words = [w for w in words if w]
        if not words:
            return False

        # Repeated single-word pattern (e.g. "you you you") = echo/hallucination
        # Only treat as echo if the repeated word is in recent AI speech
        if len(words) >= 3 and len(set(words)) == 1:
            ai_set = self._recent_ai_set()
            if words[0] in ai_set:
                log.info(f"[AITextEchoFilter] ECHO segment (repeated AI word): {text!r}")
                return True

        ai_set = self._recent_ai_set()
        content = [w for w in words if w not in _ECHO_STOP]
        if not content:
            # ALL stop words — check if they all appear in recent AI speech
            if all(w in ai_set for w in words) and ai_set:
                log.info(f"[AITextEchoFilter] ECHO segment (all-stop-word, in AI): {text!r}")
                return True
            return False
        if len(content) < AI_TEXT_ECHO_MIN_WORDS:
            return False

        # Check ALL words (including stop words) against AI set
        total_overlap = sum(1 for w in words if w in ai_set)
        total_ratio   = total_overlap / len(words)
        # Check content words only
        content_overlap = sum(1 for w in content if w in ai_set)
        content_ratio   = content_overlap / len(content)
        # Drop if either ratio exceeds threshold
        ratio = max(total_ratio, content_ratio)
        if ratio >= AI_TEXT_ECHO_RATIO:
            log.info(
                f"[AITextEchoFilter] ECHO segment ({ratio:.0%} overlap, "
                f"content={content_ratio:.0%}, total={total_ratio:.0%}): {text!r}"
            )
            return True
        return False

    def reset(self):
        self._ai_words.clear()


# ─── Timing Echo Gate ─────────────────────────────────────────────────────────

class TimingEchoGate:
    """
    Pure timing-based mic suppression gate.
    Armed from first feed_tts() until ECHO_TAIL_S after tts_stopped().
    """

    def __init__(self):
        self._tts_active     = False
        self._tts_stopped_at = 0.0
        self.frames_checked  = 0
        self.frames_dropped  = 0

    def feed_tts(self, pcm_bytes: bytes):
        self._tts_active = True

    def tts_stopped(self):
        self._tts_active     = False
        self._tts_stopped_at = time.monotonic()

    def _is_armed(self) -> bool:
        if self._tts_active:
            return True
        return (time.monotonic() - self._tts_stopped_at) < ECHO_TAIL_S

    def check(self, mic_pcm_bytes: bytes, ai_speaking: bool = False) -> bool:
        """Returns True → DROP mic frame.
        When ai_speaking=True, frames pass through so STT can detect
        user speech for barge-in.  Only suppress during echo tail
        (after TTS stops, reverb lingers).
        """
        self.frames_checked += 1
        if not mic_pcm_bytes:
            return False
        # During active TTS — let audio through for barge-in detection
        if ai_speaking:
            return False
        # Echo tail period (TTS stopped, reverb still in room) — drop
        if self._is_armed():
            self.frames_dropped += 1
            return True
        return False

    def reset(self):
        self._tts_active     = False
        self._tts_stopped_at = 0.0
        self.frames_checked  = 0
        self.frames_dropped  = 0

    @property
    def is_enrolled(self) -> bool:
        return True


# Backwards-compat aliases
GatewayEchoGate = TimingEchoGate
TTSVoiceFingerprintGate = TimingEchoGate
