"""
tonal.py — Sentence accumulator with tone classification for TTS chunking
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# ─── Configuration ────────────────────────────────────────────────────────────

MIN_TTS_CHARS   = int(os.getenv("MIN_TTS_CHARS",    "2"))
TONE_MAX_CHARS  = int(os.getenv("TONE_MAX_CHARS",   "60"))
LOGIC_MAX_CHARS = int(os.getenv("LOGIC_MAX_CHARS",  "160"))
FIRST_CHUNK_CHARS = int(os.getenv("FIRST_CHUNK_CHARS", "8"))

# ─── Enums ────────────────────────────────────────────────────────────────────

class ChunkTone(str, Enum):
    TONE  = "tone"
    LOGIC = "logic"

# ─── Regex ────────────────────────────────────────────────────────────────────

_RE_SENTENCE_END = re.compile(r'(?<=[^\d])([.!?]+["\']?)(?=\s|$)')
_RE_CLAUSE_BREAK = re.compile(r'([,;:—–])\s')
_RE_STARTS_PUNCT = re.compile(r'^[\s,\.!?;:\)\]\}\'\"\\u2019\\u2018\\u201c\\u201d\-]')


def classify_tone(text: str) -> ChunkTone:
    s = text.strip()
    if s.endswith("?") or s.endswith("!") or len(s) <= TONE_MAX_CHARS:
        return ChunkTone.TONE
    return ChunkTone.LOGIC


@dataclass
class TonalChunk:
    text: str
    tone: ChunkTone


class TonalAccumulator:
    def __init__(self):
        self._buf        = ""
        self._first_sent = True

    def reset(self):
        self._buf        = ""
        self._first_sent = True

    def feed(self, token: str) -> list[TonalChunk]:
        if not token:
            return []
        if token.startswith(" "):
            if self._buf and self._buf[-1] == " ":
                token = token.lstrip(" ")
        else:
            if self._buf and not self._buf[-1].isspace() and not _RE_STARTS_PUNCT.match(token):
                token = " " + token
        self._buf += token
        return self._try_flush()

    def flush(self) -> Optional[TonalChunk]:
        text = self._buf.strip()
        self._buf        = ""
        self._first_sent = True
        if len(text) >= 1:
            return TonalChunk(text=text, tone=classify_tone(text))
        return None

    def _try_flush(self) -> list[TonalChunk]:
        results: list[TonalChunk] = []
        while True:
            buf = self._buf

            if self._first_sent and len(buf) >= FIRST_CHUNK_CHARS:
                split = buf.rfind(" ")
                if split >= MIN_TTS_CHARS:
                    candidate = buf[:split].strip()
                    remainder = buf[split:].lstrip()
                    if candidate:
                        results.append(TonalChunk(text=candidate, tone=classify_tone(candidate)))
                        self._buf        = remainder
                        self._first_sent = False
                        continue
                elif len(buf) >= TONE_MAX_CHARS:
                    candidate = buf[:TONE_MAX_CHARS].strip()
                    remainder = buf[TONE_MAX_CHARS:].lstrip()
                    if candidate:
                        results.append(TonalChunk(text=candidate, tone=classify_tone(candidate)))
                        self._buf        = remainder
                        self._first_sent = False
                        continue
                break

            m = _RE_SENTENCE_END.search(buf)
            if m:
                candidate = buf[:m.end()].strip()
                remainder = buf[m.end():].lstrip()
                if candidate:
                    results.append(TonalChunk(text=candidate, tone=classify_tone(candidate)))
                    self._buf        = remainder
                    self._first_sent = False
                    continue
                break

            m = _RE_CLAUSE_BREAK.search(buf)
            if m:
                candidate = buf[:m.start() + 1].strip()
                remainder = buf[m.end():].lstrip()
                if len(candidate) >= MIN_TTS_CHARS and len(remainder) >= 3:
                    results.append(TonalChunk(text=candidate, tone=classify_tone(candidate)))
                    self._buf        = remainder
                    self._first_sent = False
                    continue
                break

            max_cap = LOGIC_MAX_CHARS if not self._first_sent else TONE_MAX_CHARS
            if len(buf) > max_cap:
                split = buf[:max_cap].rfind(" ")
                if split <= MIN_TTS_CHARS:
                    split = max_cap
                candidate = buf[:split].strip()
                remainder = buf[split:].lstrip()
                if candidate:
                    results.append(TonalChunk(text=candidate, tone=classify_tone(candidate)))
                    self._buf        = remainder
                    self._first_sent = False
                    continue
                break
            break
        return results
