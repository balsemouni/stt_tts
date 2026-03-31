"""
models.py — Shared enums, guards, and helpers for the gateway
"""
from __future__ import annotations

import asyncio
import logging
import os
from enum import Enum, auto

import websockets

log = logging.getLogger("gateway")

# ─── Configuration ────────────────────────────────────────────────────────────

HALLUC_WINDOW    = int(os.getenv("HALLUC_WINDOW",    "10"))
HALLUC_THRESHOLD = int(os.getenv("HALLUC_THRESHOLD", "3"))


# ─── Enums ────────────────────────────────────────────────────────────────────

class State(Enum):
    IDLE     = auto()
    THINKING = auto()
    SPEAKING = auto()


# ─── Repetition guard ─────────────────────────────────────────────────────────

class RepetitionGuard:
    _STOP = frozenset({
        "a", "an", "the", "i", "me", "my", "we", "our", "you", "your",
        "he", "she", "it", "they", "his", "her", "its", "their",
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
        "to", "of", "in", "on", "at", "by", "for", "with",
        "and", "or", "but", "not", "no", "so", "if", "as",
        "that", "this", "these", "those", "what", "which", "who",
        "how", "when", "where", "why", "up", "out", "about", "into",
        "from", "than", "then", "can", "will", "would", "could",
        "should", "may", "might", "just", "also", "very", "more", "some", "any",
    })

    def __init__(self, window=HALLUC_WINDOW, threshold=HALLUC_THRESHOLD):
        self._threshold = max(threshold, 3)
        self._history: list[str] = []
        self._phrase_history: list[str] = []
        self._window = window

    def feed(self, word: str) -> bool:
        w = word.lower().strip().rstrip(".,!?;:")
        if not w or w in self._STOP:
            return False
        self._history.append(w)
        if len(self._history) > self._window:
            self._history.pop(0)
        # Single-word repetition check
        if len(self._history) >= self._threshold:
            tail = self._history[-self._threshold:]
            if len(set(tail)) == 1:
                return True
        # Phrase-level repetition: check bigrams/trigrams
        self._phrase_history.append(w)
        if len(self._phrase_history) > 30:
            self._phrase_history = self._phrase_history[-30:]
        if len(self._phrase_history) >= 6:
            # Check for repeating bigram pattern
            words = self._phrase_history
            for n in (2, 3):
                if len(words) >= n * 3:
                    last_ngram = tuple(words[-n:])
                    count = 0
                    for i in range(len(words) - n, -1, -n):
                        if tuple(words[i:i+n]) == last_ngram:
                            count += 1
                        else:
                            break
                    if count >= 3:
                        return True
        return False

    def reset(self):
        self._history.clear()
        self._phrase_history.clear()


# ─── Async helpers ────────────────────────────────────────────────────────────

def drain_q(q: asyncio.Queue):
    """Drain all items from an asyncio queue."""
    while not q.empty():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            break


async def ws_connect(url: str, max_retries: int, label: str, **kwargs):
    """Connect to a WebSocket with exponential backoff."""
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            return await websockets.connect(url, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                raise
            log.warning(f"{label} connect failed ({e}), retry {attempt}/{max_retries} in {delay:.1f}s")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
