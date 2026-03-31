"""
latency.py — Per-turn and session-level latency tracking
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

lat_log = logging.getLogger("gateway.latency")


def _r(v: Optional[float]) -> Optional[float]:
    return round(v, 1) if v is not None else None


def _latency_color(ms: float) -> str:
    if ms < 200: return "🟢"
    if ms < 400: return "🟡"
    if ms < 800: return "🟠"
    return "🔴"


def _bar(v: float, mx: float, w: int = 28) -> str:
    f = int((v / mx) * w) if mx else 0
    return "█" * f + "░" * (w - f)


@dataclass
class TurnLatency:
    turn_id:            str   = ""
    query_text:         str   = ""
    barge_in:           bool  = False

    stt_first_word_ts:  Optional[float] = None
    stt_segment_ts:     Optional[float] = None
    stt_latency_ms:     Optional[float] = None

    query_sent_ts:      Optional[float] = None
    first_token_ts:     Optional[float] = None
    cag_first_token_ms: Optional[float] = None
    total_tokens:       int             = 0

    first_tts_chunk_ts: Optional[float] = None
    cag_to_tts_ms:      Optional[float] = None

    tts_audio_start_ts: Optional[float] = None
    tts_synth_ms:       Optional[float] = None

    e2e_ms:             Optional[float] = None
    tts_chunks:         list = field(default_factory=list)

    def finalize(self):
        if self.stt_first_word_ts and self.stt_segment_ts:
            self.stt_latency_ms     = (self.stt_segment_ts     - self.stt_first_word_ts) * 1000
        if self.query_sent_ts and self.first_token_ts:
            self.cag_first_token_ms = (self.first_token_ts     - self.query_sent_ts)     * 1000
        if self.first_token_ts and self.first_tts_chunk_ts:
            self.cag_to_tts_ms      = (self.first_tts_chunk_ts - self.first_token_ts)    * 1000
        if self.first_tts_chunk_ts and self.tts_audio_start_ts:
            self.tts_synth_ms       = (self.tts_audio_start_ts - self.first_tts_chunk_ts) * 1000
        if self.stt_first_word_ts and self.tts_audio_start_ts:
            self.e2e_ms             = (self.tts_audio_start_ts - self.stt_first_word_ts) * 1000

    def to_report(self) -> dict:
        self.finalize()
        return {
            "turn_id":            self.turn_id,
            "query":              self.query_text[:80],
            "barge_in":           self.barge_in,
            "stt_latency_ms":     _r(self.stt_latency_ms),
            "cag_first_token_ms": _r(self.cag_first_token_ms),
            "cag_to_tts_ms":      _r(self.cag_to_tts_ms),
            "tts_synth_ms":       _r(self.tts_synth_ms),
            "e2e_ms":             _r(self.e2e_ms),
            "total_tokens":       self.total_tokens,
            "tts_chunks":         self.tts_chunks,
        }


class LatencyTracker:
    def __init__(self, sid: str):
        self.sid      = sid
        self.current: Optional[TurnLatency] = None
        self.history: list[TurnLatency]     = []
        self._tts_first_chunk_ts: Optional[float] = None
        self._tts_chunk_index: int = 0

    def new_turn(self, turn_id: str, query: str):
        self.current             = TurnLatency(turn_id=turn_id, query_text=query)
        self._tts_first_chunk_ts = None
        self._tts_chunk_index    = 0

    def on_stt_first_word(self):
        if self.current and not self.current.stt_first_word_ts:
            self.current.stt_first_word_ts = time.monotonic()

    def on_stt_segment(self):
        if self.current:
            self.current.stt_segment_ts = time.monotonic()

    def on_query_sent(self):
        if self.current:
            self.current.query_sent_ts = time.monotonic()

    def on_first_token(self):
        now = time.monotonic()
        if self.current and not self.current.first_token_ts:
            self.current.first_token_ts = now
            if self.current.query_sent_ts:
                ms = (now - self.current.query_sent_ts) * 1000
                self.current.cag_first_token_ms = ms
                lat_log.info(f"[{self.sid}] CAG first token: {ms:.0f}ms  {_latency_color(ms)}")

    def on_token(self):
        if self.current:
            self.current.total_tokens += 1

    def on_tts_chunk_sent(self, text: str):
        now = time.monotonic()
        if self.current and not self.current.first_tts_chunk_ts:
            self.current.first_tts_chunk_ts = now
            if self.current.first_token_ts:
                ms = (now - self.current.first_token_ts) * 1000
                self.current.cag_to_tts_ms = ms
                lat_log.info(f"[{self.sid}] CAG→TTS first chunk: {ms:.0f}ms")

    def on_tts_audio_start(self):
        now = time.monotonic()
        if self.current and not self.current.tts_audio_start_ts:
            self.current.tts_audio_start_ts = now
            if self.current.first_tts_chunk_ts:
                ms = (now - self.current.first_tts_chunk_ts) * 1000
                self.current.tts_synth_ms = ms
                lat_log.info(f"[{self.sid}] TTS synth latency: {ms:.0f}ms  {_latency_color(ms)}")

    def on_tts_chunk_complete(self, synthesis_latency_ms: float,
                              synth_duration_ms: float, duration_sec: float):
        if not self.current:
            return
        idx = self._tts_chunk_index
        now = time.monotonic()
        if self._tts_first_chunk_ts is None:
            self._tts_first_chunk_ts = now
            first_chunk_latency_ms   = 0.0
        else:
            first_chunk_latency_ms   = (now - self._tts_first_chunk_ts) * 1000
        self.current.tts_chunks.append({
            "chunk_index":            idx,
            "synthesis_latency_ms":   _r(synthesis_latency_ms),
            "synth_duration_ms":      _r(synth_duration_ms),
            "first_chunk_latency_ms": _r(first_chunk_latency_ms),
            "duration_sec":           _r(duration_sec),
        })
        mx = max((c["synthesis_latency_ms"] or 0) for c in self.current.tts_chunks) or 1
        lat_log.info(
            f"[{self.sid}] TTS chunk {idx}  "
            f"synth={synth_duration_ms:.0f}ms  lat={synthesis_latency_ms:.0f}ms "
            f"[{_bar(synthesis_latency_ms, mx)}]  dur={duration_sec:.2f}s"
        )
        self._tts_chunk_index += 1

    def complete_turn(self) -> Optional[dict]:
        if not self.current:
            return None
        self.current.finalize()
        report = self.current.to_report()
        self.history.append(self.current)
        e2e = self.current.e2e_ms
        if e2e is not None:
            lat_log.info(
                f"[{self.sid}] {_latency_color(e2e)} E2E {e2e:.0f}ms  "
                f"[STT {_r(self.current.stt_latency_ms)}ms | "
                f"CAG {_r(self.current.cag_first_token_ms)}ms | "
                f"TTS {_r(self.current.tts_synth_ms)}ms]"
            )
        self.current = None
        return report

    def session_summary(self) -> dict:
        turns = [t for t in self.history if t.e2e_ms is not None]
        if not turns:
            return {"sid": self.sid, "turns": 0}

        def _stats(vals: list) -> dict:
            if not vals:
                return {}
            sv  = sorted(vals)
            p95 = sv[max(0, int(len(sv) * 0.95) - 1)]
            return {"min": _r(min(sv)), "max": _r(max(sv)),
                    "avg": _r(sum(sv) / len(sv)), "p95": _r(p95)}

        stt_lats  = [t.stt_latency_ms     for t in turns if t.stt_latency_ms     is not None]
        cag_lats  = [t.cag_first_token_ms for t in turns if t.cag_first_token_ms is not None]
        tts_lats  = [t.tts_synth_ms       for t in turns if t.tts_synth_ms       is not None]
        e2e_lats  = [t.e2e_ms             for t in turns]
        barge_ins = sum(1 for t in turns if t.barge_in)

        lat_log.info(
            f"[{self.sid}] ══ SESSION SUMMARY ══  turns={len(turns)}  "
            f"barge_ins={barge_ins}  e2e avg={_r(sum(e2e_lats)/len(e2e_lats))}ms"
        )
        return {
            "sid": self.sid, "turns": len(turns), "barge_ins": barge_ins,
            "stt": _stats(stt_lats), "cag": _stats(cag_lats),
            "tts_synth": _stats(tts_lats), "e2e": _stats(e2e_lats),
        }

    def all_reports(self) -> list[dict]:
        return [t.to_report() for t in self.history]
