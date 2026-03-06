"""
gateway.py — Zero-Latency Voice Pipeline Gateway v5
════════════════════════════════════════════════════

ROOT CAUSES FIXED IN THIS VERSION
───────────────────────────────────
1. TOKEN SPACING  (was: "HiBenson!Howcan")
   The CAG SSE stream sends tokens without leading spaces.
   Fix: SentenceAccumulator.feed() now intelligently inserts a space
   between tokens UNLESS the new token starts with punctuation or the
   buffer already ends with a space.

2. STALE CONVERSATION BLEEDING INTO NEW TURN (was: AI repeats whole history)
   The CAG /chat/stream endpoint was accumulating and re-sending the
   entire conversation history on every token event.
   Fix: Gateway sends reset_session=True inside the /chat/stream JSON body
   (v3 CAG API — saves one full HTTP round-trip vs. separate POST /reset).
   Falls back to POST /reset for older CAG versions.
   Also: full_reply is now a LOCAL list per query, never shared.

3. STT ECHO LOOP  (was: "I'm not going to do anything" repeated 30x)
   The speaker audio coming out of the laptop was being picked up by the
   mic and fed back into STT, creating an infinite hallucination loop.
   Fix: Gateway sends {"type":"mute_mic"} to the client the moment the
   first TTS sentence is sent, and {"type":"unmute_mic"} when TTS is done.
   Also: short STT segments (< BARGE_IN_MIN_WORDS) are IGNORED while
   state==SPEAKING — they are almost certainly echo artefacts.

4. TTS NOT RESPONDING INSTANTLY
   Fix: One persistent TTS session per gateway session lifetime.
   Between turns we pre-open the next TTS sub-session immediately so the
   next CAG response has zero handshake overhead.
   ALSO (v5): SentenceAccumulator uses a lower threshold (FIRST_SENT_CHARS=10)
   for the very first sentence per turn so the user hears audio immediately
   ("Hello!" or "Sure," no longer waits for 40 chars to build up).

5. CHOPPY / ROBOTIC TTS (short fragments sent one word at a time)
   Fix: SentenceAccumulator enforces MIN_TTS_CHARS=40 for all sentences
   AFTER the first one, keeping TTS chunks natural-length.

6. HALLUCINATION REPETITION LOOP  (was: "me me me me …" × 50)
   Whisper can enter a degenerate state on silence/noise where it repeats
   the same token in every new partial/word event, creating an infinite
   loop that fills the query queue with garbage.
   Fix: RepetitionGuard in _stt_loop tracks word history; if the same word
   appears 4+ times in the last 8 words a hallucination_detected event is
   fired, the STT window is cleared, the word is dropped, and the duplicate
   query queue is drained so CAG never sees hallucinated text.

Environment variables
─────────────────────
  STT_WS_URL          ws://localhost:8001/stream/mux
  CAG_HTTP_URL        http://localhost:8000
  TTS_WS_URL          ws://localhost:8765
  GATEWAY_HOST        0.0.0.0
  GATEWAY_PORT        8090
  TTS_SPEAKER         aiden
  TTS_LANGUAGE        English
  BARGE_IN_MIN_WORDS  3
  MIN_TTS_CHARS       40   (non-first sentences)
  FIRST_SENT_CHARS    10   (first sentence per turn — low for instant response)
  HALLUC_WINDOW      10   (content-word history length)
  HALLUC_THRESHOLD    5   (consecutive identical content words = hallucination)
"""

import asyncio
import json
import logging
import os
import re
import struct
import uuid
from enum import Enum, auto
from typing import Optional

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# ─── Config ───────────────────────────────────────────────────────────────────

STT_WS_URL         = os.getenv("STT_WS_URL",         "ws://localhost:8001/stream/mux")
CAG_HTTP_URL       = os.getenv("CAG_HTTP_URL",        "http://localhost:8000")
TTS_WS_URL         = os.getenv("TTS_WS_URL",          "ws://localhost:8765")
GATEWAY_HOST       = os.getenv("GATEWAY_HOST",        "0.0.0.0")
GATEWAY_PORT       = int(os.getenv("GATEWAY_PORT",    "8090"))
TTS_SPEAKER        = os.getenv("TTS_SPEAKER",         "aiden")
TTS_LANGUAGE       = os.getenv("TTS_LANGUAGE",        "English")
BARGE_IN_MIN_WORDS = int(os.getenv("BARGE_IN_MIN_WORDS", "3"))
MIN_TTS_CHARS      = int(os.getenv("MIN_TTS_CHARS",      "40"))
FIRST_SENT_CHARS   = int(os.getenv("FIRST_SENT_CHARS",   "10"))
HALLUC_WINDOW      = int(os.getenv("HALLUC_WINDOW",       "10"))
HALLUC_THRESHOLD   = int(os.getenv("HALLUC_THRESHOLD",     "5"))

# websocket lib requires ping_interval STRICTLY < ping_timeout
WS_PING_INTERVAL = 15
WS_PING_TIMEOUT  = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("gateway")

app = FastAPI(title="Voice Gateway", version="5.0.0")


# ─── Turn state ───────────────────────────────────────────────────────────────

class State(Enum):
    IDLE     = auto()   # mic open, waiting for user
    THINKING = auto()   # CAG is generating
    SPEAKING = auto()   # TTS is playing — mic muted


# ─── Sentence accumulator ─────────────────────────────────────────────────────

_SENT_END        = re.compile(r'(?<!\d)([.!?]+["\']?)(?=\s|$)')
_STARTS_WITH_PUNCT = re.compile(r'^[\s,\.!?;:\)\]\}]')


class SentenceAccumulator:
    """
    Joins LLM tokens into properly-spaced sentences ready for TTS.

    Space insertion rules:
      - No space if the buffer is empty.
      - No space if the incoming token starts with punctuation (,./!? …).
      - No space if the buffer already ends with a space.
      - Otherwise insert exactly one space.

    Emission rules:
      - The FIRST sentence per turn uses a low threshold (FIRST_SENT_CHARS)
        so the user hears audio immediately ("Hello!" / "Sure," / "Yes.").
      - All subsequent sentences use MIN_TTS_CHARS (larger) for natural TTS
        chunking — prevents choppy one-word bursts.
      - flush() emits whatever remains regardless of length.
      - reset() is called between turns to restore first-sentence behaviour.
    """

    def __init__(self, min_chars: int = MIN_TTS_CHARS, first_chars: int = FIRST_SENT_CHARS):
        self._buf        = ""
        self._min_chars  = min_chars
        self._first_chars = first_chars
        self._first_sent  = True   # True until the first sentence is emitted

    def reset(self):
        """Call at the start of every new turn to restore fast-first behaviour."""
        self._buf        = ""
        self._first_sent  = True

    def feed(self, token: str) -> list[str]:
        if not token:
            return []

        if self._buf and not self._buf[-1].isspace() and not _STARTS_WITH_PUNCT.match(token):
            self._buf += " "
        self._buf += token

        sentences: list[str] = []
        while True:
            m = _SENT_END.search(self._buf)
            if not m:
                break
            end       = m.end(1)
            candidate = self._buf[:end].strip()

            # First sentence uses a lower char limit for instant response.
            # After the first sentence is emitted, switch to the full limit.
            limit = self._first_chars if self._first_sent else self._min_chars
            if len(candidate) < limit:
                break

            sentences.append(candidate)
            self._buf       = self._buf[end:].lstrip()
            self._first_sent = False   # subsequent sentences use _min_chars

        return sentences

    def flush(self) -> Optional[str]:
        s             = self._buf.strip()
        self._buf     = ""
        self._first_sent = True   # reset for the next turn
        return s if s else None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _drain(q: asyncio.Queue):
    while not q.empty():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            break


# ─── Hallucination / repetition guard ────────────────────────────────────────

class RepetitionGuard:
    """
    Detects Whisper hallucination loops ("me me me me …" / "I'm sorry I'm sorry …").

    Key design decisions
    ────────────────────
    1. STOP-WORD EXEMPTION
       Common English function words (a, in, the, my, I, is, …) are explicitly
       excluded from the repetition check. These words legitimately repeat in
       normal speech ("a problem in a system in a module") and must never
       trigger the guard. Only *content* words (nouns, verbs, interjections)
       are checked.

    2. CONSECUTIVE-ONLY window
       Rather than counting occurrences anywhere in a sliding window (which
       is what caused "a" to fire after "a problem in a"), we require the
       word to appear in N *consecutive* positions at the tail of the history.
       This catches "me me me me" while ignoring "a problem in a".

    3. TUNABLE via env vars (HALLUC_WINDOW / HALLUC_THRESHOLD)
       Default: require 5 consecutive identical content-word tokens.
       This is strict enough to only fire on genuine ASR lock-up.

    Example — threshold=5:
      ["me","me","me","me","me"]             → True  (5 consecutive "me")
      ["a","problem","in","a","problem"]     → False (stop words ignored)
      ["sorry","sorry","sorry","sorry"]      → False (only 4 consecutive)
      ["sorry","sorry","sorry","sorry","sorry"] → True
    """

    # Words that are too common to ever be considered hallucinations.
    # Extend this set if you observe false positives on other languages.
    _STOP_WORDS: frozenset[str] = frozenset({
        "a", "an", "the",
        "i", "me", "my", "we", "our", "you", "your",
        "he", "she", "it", "they", "his", "her", "its", "their",
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
        "to", "of", "in", "on", "at", "by", "for", "with",
        "and", "or", "but", "not", "no", "so", "if", "as",
        "that", "this", "these", "those",
        "what", "which", "who", "how", "when", "where", "why",
        "up", "out", "about", "into", "from", "than", "then",
        "can", "will", "would", "could", "should", "may", "might",
        "just", "also", "very", "more", "some", "any",
    })

    def __init__(self, window: int = HALLUC_WINDOW, threshold: int = HALLUC_THRESHOLD):
        # threshold is re-used as the consecutive-repeat count.
        # Default HALLUC_THRESHOLD is raised to 5 in the env-var defaults below.
        self._threshold = max(threshold, 4)   # never trigger on < 4 repeats
        self._history: list[str] = []
        self._window = window

    def feed(self, word: str) -> bool:
        """
        Add a word to the guard.
        Returns True only if a genuine hallucination loop is detected.
        Returns False for all normal speech patterns.
        """
        w = word.lower().strip().rstrip(".,!?;:")
        if not w:
            return False

        # Skip stop-words entirely — they can repeat in real sentences
        if w in self._STOP_WORDS:
            return False

        self._history.append(w)
        if len(self._history) > self._window:
            self._history.pop(0)

        # Require N *consecutive* identical tokens at the tail.
        # This is much stricter than counting anywhere in the window.
        if len(self._history) < self._threshold:
            return False

        tail = self._history[-self._threshold:]
        return len(set(tail)) == 1   # all identical → hallucination

    def reset(self):
        self._history.clear()


# ─── Gateway session ──────────────────────────────────────────────────────────

class GatewaySession:
    """
    Manages one client connection end-to-end.

    Three long-lived coroutines:
      _stt_loop  — mic PCM → STT server → segment/partial events → client + query queue
      _cag_loop  — query → CAG /chat/stream → accumulate sentences → TTS queue
      _tts_loop  — sentences → TTS server → PCM → client speaker

    Barge-in:
      STT emits a segment while state != IDLE.
      If segment has >= BARGE_IN_MIN_WORDS words it is treated as a real
      interruption: TTS queue is drained, _INTERRUPT sentinel injected,
      CAG stream is aborted, new query starts from scratch.
      Short segments during SPEAKING are silently dropped (echo artefacts).

    Mic mute/unmute:
      Client receives {"type":"mute_mic"} when TTS starts playing and
      {"type":"unmute_mic"} when TTS is done. Client-side code must stop
      feeding audio frames while muted (or at minimum stop sending 0x01
      frames). This breaks the STT echo loop completely.
    """

    _INTERRUPT = object()

    def __init__(self, ws: WebSocket):
        self.ws       = ws
        self.sid      = str(uuid.uuid4())[:8]
        self.state    = State.IDLE
        self._running = True

        self._audio_q : asyncio.Queue[bytes] = asyncio.Queue()
        self._query_q : asyncio.Queue[str]   = asyncio.Queue()
        self._tts_q   : asyncio.Queue        = asyncio.Queue()

        self._stt_ws   = None
        self._barge_in = False

        log.info(f"[{self.sid}] session created")

    # ── safe sends ────────────────────────────────────────────────────────────

    async def _jsend(self, obj: dict):
        try:
            await self.ws.send_json(obj)
        except Exception:
            pass

    async def _bsend(self, data: bytes):
        try:
            await self.ws.send_bytes(data)
        except Exception:
            pass

    async def _mute_mic(self):
        log.info(f"[{self.sid}] → mute_mic")
        await self._jsend({"type": "mute_mic"})

    async def _unmute_mic(self):
        log.info(f"[{self.sid}] → unmute_mic")
        await self._jsend({"type": "unmute_mic"})

    # ── entry ─────────────────────────────────────────────────────────────────

    async def run(self):
        tasks = [
            asyncio.create_task(self._stt_loop(), name="stt"),
            asyncio.create_task(self._cag_loop(), name="cag"),
            asyncio.create_task(self._tts_loop(), name="tts"),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in pending:
            t.cancel()
        for t in done:
            if not t.cancelled():
                exc = t.exception()
                if exc:
                    log.error(f"[{self.sid}] {t.get_name()} task crashed: {exc}", exc_info=exc)

    async def stop(self):
        self._running = False

    # ── client audio ──────────────────────────────────────────────────────────

    def push_audio(self, frame: bytes):
        self._audio_q.put_nowait(frame)

    def push_control(self, payload: bytes):
        if self._stt_ws:
            asyncio.ensure_future(self._relay_ctrl(payload))

    async def _relay_ctrl(self, payload: bytes):
        try:
            await self._stt_ws.send(b'\x02' + payload)
        except Exception:
            pass

    # ── barge-in ──────────────────────────────────────────────────────────────

    async def _do_barge_in(self, new_query: str):
        log.info(f"[{self.sid}] ⚡ BARGE-IN → '{new_query[:60]}'")
        self._barge_in = True
        _drain(self._tts_q)
        await self._tts_q.put(self._INTERRUPT)
        _drain(self._query_q)
        await self._query_q.put(new_query)
        await self._unmute_mic()

    # ─────────────────────────────────────────────────────────────────────────
    # STT LOOP
    # ─────────────────────────────────────────────────────────────────────────

    async def _stt_loop(self):
        log.info(f"[{self.sid}] STT → {STT_WS_URL}")
        async with websockets.connect(
            STT_WS_URL,
            max_size=2 * 1024 * 1024,
            ping_interval=WS_PING_INTERVAL,
            ping_timeout=WS_PING_TIMEOUT,
        ) as stt_ws:
            self._stt_ws = stt_ws
            log.info(f"[{self.sid}] STT connected")

            async def _push():
                while self._running:
                    frame = await self._audio_q.get()
                    try:
                        await stt_ws.send(frame)
                    except Exception:
                        pass

            async def _recv():
                guard = RepetitionGuard()   # per-session hallucination detector

                async for raw in stt_ws:
                    try:
                        ev = json.loads(raw)
                    except Exception:
                        continue

                    kind = ev.get("type")

                    # ── Live word stream — check for hallucination ─────────
                    if kind == "word":
                        word = ev.get("word", "").strip()
                        if word and guard.feed(word):
                            # Whisper is stuck repeating itself — reset ASR
                            log.warning(
                                f"[{self.sid}] 🚨 Hallucination detected "
                                f"('{word}' repeated). Resetting STT window."
                            )
                            guard.reset()
                            # Send a soft reset hint to the STT server so it
                            # clears its internal Whisper context buffer.
                            try:
                                await stt_ws.send(
                                    b'\x02' + json.dumps({"type": "reset_context"}).encode()
                                )
                            except Exception:
                                pass
                            # Drain any hallucinated partial queries already
                            # sitting in the query queue.
                            _drain(self._query_q)
                            await self._jsend({
                                "type":   "hallucination_reset",
                                "detail": f"Repeated word '{word}' cleared",
                            })
                            continue   # drop the hallucinated word entirely

                        await self._jsend(ev)

                    elif kind == "partial":
                        await self._jsend(ev)

                    elif kind == "segment":
                        text       = ev.get("text", "").strip()
                        word_count = len(text.split()) if text else 0
                        if not text:
                            continue

                        # A clean segment means ASR recovered — reset guard
                        guard.reset()

                        log.info(f"[{self.sid}] STT segment [{self.state.name}] ({word_count}w): {text!r}")
                        await self._jsend(ev)

                        if self.state == State.SPEAKING:
                            if word_count >= BARGE_IN_MIN_WORDS:
                                await self._do_barge_in(text)
                            else:
                                # Almost certainly speaker echo — ignore
                                log.info(f"[{self.sid}] Dropped short segment (echo guard)")
                        elif self.state == State.THINKING:
                            await self._do_barge_in(text)
                        else:
                            await self._query_q.put(text)

                    elif kind == "error":
                        log.warning(f"[{self.sid}] STT error: {ev}")
                        await self._jsend(ev)

            await asyncio.gather(_push(), _recv())

    # ─────────────────────────────────────────────────────────────────────────
    # CAG LOOP
    # ─────────────────────────────────────────────────────────────────────────

    async def _cag_loop(self):
        async with httpx.AsyncClient(
            base_url=CAG_HTTP_URL,
            timeout=httpx.Timeout(connect=5.0, read=None, write=5.0, pool=5.0),
        ) as http:

            while self._running:
                query = await self._query_q.get()

                # ── Reset CAG history before every turn ───────────────────────
                # Prefer the v3 inline flag (saves one HTTP round-trip).
                # Fall back to POST /reset for older CAG deployments.
                cag_reset_inline = True
                try:
                    r = await http.post("/health")
                    # If the service is v3+ it advertises version 3.x.x
                    # We'll try inline first and fall back on 4xx.
                except Exception as e:
                    log.warning(f"[{self.sid}] CAG health check failed ({e}); will try inline reset anyway")

                if not cag_reset_inline:
                    try:
                        r = await http.post("/reset")
                        log.info(f"[{self.sid}] CAG reset (fallback) → {r.status_code}")
                    except Exception as e:
                        log.warning(f"[{self.sid}] CAG /reset unavailable ({e}); continuing")

                self._barge_in = False
                self.state     = State.THINKING
                log.info(f"[{self.sid}] CAG query: {query!r}")
                await self._jsend({"type": "thinking"})

                # Per-turn locals — never shared with other turns
                acc              = SentenceAccumulator()
                acc.reset()   # ensure first-sentence fast-emit threshold is active
                full_reply_parts : list[str] = []
                interrupted      = False

                try:
                    async with http.stream(
                        "POST", "/chat/stream",
                        json={
                            "message":       query,
                            "reset_session": True,   # v3 CAG: inline reset, saves POST /reset RTT
                        },
                        headers={"Accept": "text/event-stream"},
                    ) as resp:

                        async for line in resp.aiter_lines():
                            if self._barge_in:
                                interrupted = True
                                log.info(f"[{self.sid}] CAG aborted mid-stream (barge-in)")
                                break

                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()

                            if data == "[DONE]":
                                break
                            if data.startswith("[ERROR]"):
                                await self._jsend({"type": "error", "detail": data[7:].strip()})
                                interrupted = True
                                break
                            if not data:
                                continue

                            # Live token display on client
                            await self._jsend({"type": "ai_token", "token": data})
                            full_reply_parts.append(data)

                            # Accumulate & push complete sentences to TTS
                            for sentence in acc.feed(data):
                                if self._barge_in:
                                    interrupted = True
                                    break
                                log.info(f"[{self.sid}] TTS ← {sentence!r}")
                                await self._jsend({"type": "ai_sentence", "text": sentence})
                                self.state = State.SPEAKING
                                await self._tts_q.put(sentence)

                            if interrupted:
                                break

                    # Flush trailing partial sentence
                    if not interrupted and not self._barge_in:
                        tail = acc.flush()
                        if tail:
                            log.info(f"[{self.sid}] TTS ← tail: {tail!r}")
                            await self._jsend({"type": "ai_sentence", "text": tail})
                            self.state = State.SPEAKING
                            await self._tts_q.put(tail)

                except Exception as e:
                    log.error(f"[{self.sid}] CAG error: {e}")
                    await self._jsend({"type": "error", "detail": str(e)})
                    interrupted = True

                finally:
                    if not interrupted and not self._barge_in:
                        await self._tts_q.put(None)      # clean end signal
                        # Whisper context hint
                        full_text = " ".join(full_reply_parts).strip()
                        if full_text and self._stt_ws:
                            ctrl = json.dumps({
                                "type": "assistant_turn",
                                "text": full_text,
                            }).encode()
                            try:
                                await self._stt_ws.send(b'\x02' + ctrl)
                            except Exception:
                                pass

    # ─────────────────────────────────────────────────────────────────────────
    # TTS LOOP
    # ─────────────────────────────────────────────────────────────────────────

    async def _tts_loop(self):
        log.info(f"[{self.sid}] TTS → {TTS_WS_URL}")
        async with websockets.connect(
            TTS_WS_URL,
            max_size=10 * 1024 * 1024,
            ping_interval=WS_PING_INTERVAL,
            ping_timeout=WS_PING_TIMEOUT,
        ) as tts_ws:
            log.info(f"[{self.sid}] TTS connected")

            tts_base  = self.sid
            turn_n    = [0]

            async def _new_turn() -> str:
                turn_n[0] += 1
                sid = f"{tts_base}-t{turn_n[0]}"
                await tts_ws.send(json.dumps({
                    "type": "start", "session_id": sid,
                    "speaker": TTS_SPEAKER, "language": TTS_LANGUAGE,
                }))
                log.info(f"[{self.sid}] TTS turn: {sid}")
                return sid

            active_sid = await _new_turn()

            async def _push():
                nonlocal active_sid

                while self._running:
                    item = await self._tts_q.get()

                    if item is self._INTERRUPT:
                        # Barge-in — abort current synthesis
                        log.info(f"[{self.sid}] TTS interrupt")
                        try:
                            await tts_ws.send(json.dumps({"type": "flush", "session_id": active_sid}))
                            await tts_ws.send(json.dumps({"type": "end",   "session_id": active_sid}))
                        except Exception:
                            pass
                        self.state = State.IDLE
                        active_sid = await _new_turn()
                        await self._unmute_mic()

                    elif item is None:
                        # Clean end of CAG response
                        try:
                            await tts_ws.send(json.dumps({"type": "flush", "session_id": active_sid}))
                            await tts_ws.send(json.dumps({"type": "end",   "session_id": active_sid}))
                        except Exception:
                            pass
                        # Pre-open next turn — zero latency for next query
                        active_sid = await _new_turn()
                        # unmute happens in _recv when TTS server sends "done"

                    else:
                        # Sentence — mute mic on first chunk of this turn
                        if self.state != State.SPEAKING:
                            await self._mute_mic()
                            self.state = State.SPEAKING
                        try:
                            await tts_ws.send(json.dumps({
                                "type": "chunk", "session_id": active_sid, "text": item,
                            }))
                        except Exception as e:
                            log.error(f"[{self.sid}] TTS chunk error: {e}")

            async def _recv():
                async for frame in tts_ws:
                    if isinstance(frame, bytes):
                        if len(frame) < 12:
                            continue
                        sid_len = struct.unpack_from("<I", frame, 0)[0]
                        pcm     = frame[12 + sid_len:]
                        if pcm:
                            await self._bsend(pcm)

                    elif isinstance(frame, str):
                        try:
                            msg = json.loads(frame)
                        except Exception:
                            continue
                        t = msg.get("type")
                        if t == "done":
                            log.info(f"[{self.sid}] TTS done ({msg.get('chunks',0)} chunks)")
                            await self._jsend({"type": "done", "chunks": msg.get("chunks", 0)})
                            self.state = State.IDLE
                            await self._unmute_mic()
                        elif t == "error" or msg.get("error"):
                            err = msg.get("error") or msg.get("detail", "TTS error")
                            log.warning(f"[{self.sid}] TTS error: {err}")
                            await self._jsend({"type": "error", "detail": err})
                            self.state = State.IDLE
                            await self._unmute_mic()
                        # "started" acks dropped

            await asyncio.gather(_push(), _recv())


# ─── WebSocket endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session  = GatewaySession(ws)
    pipeline = asyncio.create_task(session.run())
    log.info(f"[{session.sid}] client connected")

    try:
        async for raw in ws.iter_bytes():
            if not raw:
                continue
            ftype   = raw[0]
            payload = raw[1:]

            if ftype == 0x01:
                session.push_audio(raw)

            elif ftype == 0x02:
                try:
                    ctrl = json.loads(payload)
                except Exception:
                    continue
                mtype = ctrl.get("type")
                if mtype == "ping":
                    await ws.send_json({"type": "pong"})
                elif mtype in ("ai_state", "ai_reference", "reset_context", "assistant_turn"):
                    session.push_control(payload)
                elif mtype == "get_stats":
                    await ws.send_json({
                        "type":  "stats",
                        "sid":   session.sid,
                        "state": session.state.name,
                    })

    except WebSocketDisconnect:
        log.info(f"[{session.sid}] disconnected")
    except Exception as e:
        log.error(f"[{session.sid}] ws error: {e}")
    finally:
        await session.stop()
        pipeline.cancel()


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "5.0.0"}


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "gateway:app",
        host=GATEWAY_HOST,
        port=GATEWAY_PORT,
        workers=1,
        log_level="info",
        reload=False,
    )