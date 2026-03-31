"""
session.py — GatewaySession: full voice pipeline per WebSocket client
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Optional, Callable, Awaitable

import httpx

from tts.azure_tts import azure_tts_request, azure_tts_stream
from gateway.models import State, RepetitionGuard, drain_q, ws_connect
from gateway.echo_gate import TimingEchoGate, AITextEchoFilter
from gateway.latency import LatencyTracker
from gateway.tonal import TonalAccumulator, TonalChunk, classify_tone

log = logging.getLogger("gateway")

# ─── Prometheus counters (safe import — metrics registered in gateway.py) ────
def _safe_metric(name):
    """Retrieve an already-registered Prometheus metric by name, or return a no-op."""
    try:
        from prometheus_client import REGISTRY
        return REGISTRY._names_to_collectors.get(name)
    except Exception:
        return None

class _Noop:
    def inc(self, *a, **kw): pass
    def observe(self, *a, **kw): pass

def _get(name): return _safe_metric(name) or _Noop()


# ─── Configuration (read from env) ───────────────────────────────────────────

STT_WS_URL          = os.getenv("STT_WS_URL",          "ws://localhost:8001/stream/mux")
CAG_WS_URL          = os.getenv("CAG_WS_URL",          "ws://localhost:8000/chat/ws")
CAG_HTTP_URL        = os.getenv("CAG_HTTP_URL",         "http://localhost:8000")

BARGE_IN_MIN_WORDS  = int(os.getenv("BARGE_IN_MIN_WORDS",    "1"))
BARGE_IN_COOLDOWN_S = float(os.getenv("BARGE_IN_COOLDOWN_S", "0.2"))
ECHO_TAIL_GUARD_S   = float(os.getenv("ECHO_TAIL_GUARD_S",   "0.3"))
STT_SILENCE_MS      = float(os.getenv("STT_SILENCE_MS",      "350"))

TTS_MAX_PARALLEL    = int(os.getenv("TTS_MAX_PARALLEL",       "4"))
TTS_MAX_RETRIES     = int(os.getenv("TTS_MAX_RETRIES",        "3"))
STT_MAX_RETRIES     = int(os.getenv("STT_MAX_RETRIES",        "5"))
CAG_MAX_RETRIES     = int(os.getenv("CAG_MAX_RETRIES",        "3"))

STT_AUDIO_QUEUE_MAX = int(os.getenv("STT_AUDIO_QUEUE_MAX",   "200"))
IDLE_TIMEOUT_S      = float(os.getenv("IDLE_TIMEOUT_S",       "120.0"))
HEARTBEAT_S         = float(os.getenv("HEARTBEAT_S",          "25.0"))

TTS_GREETING        = os.getenv("TTS_GREETING", "")
TEST_MODE           = os.getenv("TEST_MODE", "0").strip() in ("1", "true", "yes")

WS_PING_INTERVAL = 15
WS_PING_TIMEOUT  = 20


class GatewaySession:
    """Full voice pipeline session — one per WebSocket client."""

    _INTERRUPT = object()
    _TURN_END  = object()

    def __init__(self, ws, chat_session_id: str | None = None,
                 save_message_fn: Callable[[str, str, str], Awaitable[None]] | None = None,
                 update_title_fn: Callable[[str, str], Awaitable[None]] | None = None,
                 load_history_fn: Callable[[str], Awaitable[list]] | None = None):
        self.ws       = ws
        self.sid      = str(uuid.uuid4())[:8]
        self.state    = State.IDLE
        self._running = True

        self._chat_session_id = chat_session_id
        self._save_message    = save_message_fn
        self._update_title    = update_title_fn
        self._load_history    = load_history_fn
        self._title_set       = False

        self._audio_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=STT_AUDIO_QUEUE_MAX)
        self._query_q: asyncio.Queue        = asyncio.Queue()
        self._tts_q:   asyncio.Queue        = asyncio.Queue()
        self._pcm_q:   asyncio.Queue        = asyncio.Queue()

        self._tts_sem = asyncio.Semaphore(TTS_MAX_PARALLEL)

        self._stt_ws: Optional[object]  = None
        self._cag_ws: Optional[object]  = None

        self._barge_in        = False
        self._barge_in_until  = 0.0
        self._tts_stopped_at  = 0.0
        self._last_pong_time  = time.monotonic()
        self._last_query_time = time.monotonic()

        self._stt_ready  = asyncio.Event()
        self._barge_in_event = asyncio.Event()
        self._lat        = LatencyTracker(self.sid)
        self._echo_gate        = TimingEchoGate()
        self._text_echo_filter = AITextEchoFilter()
        self._stt_notified_speaking = False

        log.info(f"[{self.sid}] session created")

    # ── Persist message (fire-and-forget) ─────────────────────────────────────

    async def _persist(self, role: str, content: str):
        if self._chat_session_id and self._save_message and content.strip():
            try:
                await self._save_message(self._chat_session_id, role, content)
            except Exception as e:
                log.warning(f"[{self.sid}] persist({role}) failed: {e}")

            # Auto-title the session from the first user message
            if role == "user" and not self._title_set and self._update_title:
                self._title_set = True
                try:
                    await self._update_title(self._chat_session_id, content)
                except Exception:
                    pass

    # ── Safe send ─────────────────────────────────────────────────────────────

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

    async def _notify_stt_speaking(self, speaking: bool):
        """Tell STT pipeline whether AI TTS is currently playing."""
        if speaking == self._stt_notified_speaking:
            return
        self._stt_notified_speaking = speaking
        if self._stt_ws:
            try:
                ctrl = json.dumps({"type": "ai_speaking", "speaking": speaking}).encode()
                await self._stt_ws.send(b'\x02' + ctrl)
            except Exception:
                pass

    # ── Entry / stop ──────────────────────────────────────────────────────────

    async def run(self):
        tasks = [
            asyncio.create_task(self._stt_loop(),     name="stt"),
            asyncio.create_task(self._cag_loop(),      name="cag"),
            asyncio.create_task(self._synth_worker(),  name="synth"),
            asyncio.create_task(self._play_worker(),   name="play"),
            asyncio.create_task(self._heartbeat(),     name="heartbeat"),
            asyncio.create_task(self._idle_watchdog(), name="idle"),
        ]
        if not TEST_MODE:
            asyncio.create_task(self._startup_sequence(), name="startup")

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in pending:
            t.cancel()
        for t in done:
            if not t.cancelled():
                exc = t.exception()
                if exc:
                    log.error(f"[{self.sid}] {t.get_name()} crashed: {exc}", exc_info=exc)

    async def stop(self, latency_store: dict):
        self._running = False
        self._echo_gate.reset()
        self._text_echo_filter.reset()
        summary = self._lat.session_summary()
        latency_store[self.sid] = {"summary": summary, "turns": self._lat.all_reports()}
        await self._jsend({"type": "session_summary", "latency": summary})
        if self._cag_ws:
            try:
                await self._cag_ws.close()
            except Exception:
                pass

    # ── Startup / greeting ────────────────────────────────────────────────────

    async def _startup_sequence(self):
        log.info(f"[{self.sid}] Waiting for STT ready…")
        try:
            await asyncio.wait_for(self._stt_ready.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            log.warning(f"[{self.sid}] STT not ready after 15s — continuing anyway")

        asyncio.create_task(self._prewarm_tts(), name=f"prewarm_{self.sid}")
        await self._jsend({"type": "ready", "message": "Pipeline ready — speak to begin"})

    async def _prewarm_tts(self):
        try:
            await azure_tts_request("Hello.")
            log.info(f"[{self.sid}] TTS pre-warm complete")
        except Exception as e:
            log.debug(f"[{self.sid}] TTS pre-warm skipped: {e}")

    async def _play_greeting(self):
        log.info(f"[{self.sid}] Playing greeting: {TTS_GREETING!r}")
        try:
            self.state = State.SPEAKING
            await self._notify_stt_speaking(True)
            await self._jsend({"type": "ai_sentence", "text": TTS_GREETING,
                                "tone": classify_tone(TTS_GREETING)})
            pcm_data = await azure_tts_request(TTS_GREETING)
            for i in range(0, len(pcm_data), 4096):
                frame = pcm_data[i:i + 4096]
                await self._bsend(frame)
                self._echo_gate.feed_tts(frame)
        except Exception as e:
            log.warning(f"[{self.sid}] Greeting error: {e}")
        self._tts_stopped_at = time.monotonic()
        self.state = State.IDLE
        await self._notify_stt_speaking(False)
        await self._jsend({"type": "done", "chunks": 1})
        self._echo_gate.tts_stopped()

    # ── Audio push ────────────────────────────────────────────────────────────

    def push_audio(self, frame: bytes):
        self._last_pong_time = time.monotonic()
        if self._audio_q.full():
            try:
                self._audio_q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._audio_q.put_nowait(frame)

    # ── Heartbeat / idle ──────────────────────────────────────────────────────

    async def _heartbeat(self):
        while self._running:
            await asyncio.sleep(HEARTBEAT_S)
            await self._jsend({"type": "ping"})
            if time.monotonic() - self._last_pong_time > HEARTBEAT_S * 2:
                log.warning(f"[{self.sid}] heartbeat timeout — closing")
                self._running = False
                break

    async def _idle_watchdog(self):
        while self._running:
            await asyncio.sleep(10)
            if (time.monotonic() - self._last_query_time > IDLE_TIMEOUT_S
                    and self.state == State.IDLE):
                log.info(f"[{self.sid}] idle reset")
                try:
                    async with httpx.AsyncClient(base_url=CAG_HTTP_URL, timeout=5.0) as http:
                        await http.post("/reset")
                except Exception as e:
                    log.debug(f"[{self.sid}] idle /reset: {e}")
                self._last_query_time = time.monotonic()
                await self._jsend({"type": "session_reset", "reason": "idle_timeout"})

    # ── Barge-in ──────────────────────────────────────────────────────────────

    async def _do_barge_in_immediate(self):
        now = time.monotonic()
        if now < self._barge_in_until:
            return
        log.info(f"[{self.sid}] ⚡ BARGE-IN")
        _get("gateway_barge_ins_total").inc()

        if self._lat.current:
            self._lat.current.barge_in = True

        self._barge_in       = True
        self._barge_in_until = now + BARGE_IN_COOLDOWN_S
        self._barge_in_event.set()  # wake up CAG stream loop immediately

        # Immediately disarm echo gates so user speech flows through after barge-in
        self._echo_gate.reset()
        self._text_echo_filter.reset()
        self._tts_stopped_at = 0.0   # clear echo-tail window

        # Tell STT pipeline AI is no longer speaking — critical for AEC gate
        # Force-send even if we think we already sent False
        self._stt_notified_speaking = True  # set to True so _notify will send False
        await self._notify_stt_speaking(False)

        drain_q(self._tts_q)
        drain_q(self._pcm_q)
        await self._tts_q.put(self._INTERRUPT)
        await self._pcm_q.put(self._INTERRUPT)

        # Send cancel frame to CAG so it kills generation instantly
        # (keeps the WS alive — no reconnection needed)
        if self._cag_ws:
            try:
                await self._cag_ws.send(json.dumps({"type": "cancel"}))
            except Exception:
                pass

        await self._jsend({"type": "barge_in"})

    # ─── STT loop ─────────────────────────────────────────────────────────────

    async def _stt_loop(self):
        log.info(f"[{self.sid}] STT → {STT_WS_URL}")
        retries = 0

        while self._running and retries < STT_MAX_RETRIES:
            try:
                stt_ws = await ws_connect(
                    f"{STT_WS_URL}?sid={self.sid}",
                    max_retries=STT_MAX_RETRIES,
                    label=f"[{self.sid}] STT",
                    max_size=2 * 1024 * 1024,
                    ping_interval=WS_PING_INTERVAL,
                    ping_timeout=WS_PING_TIMEOUT,
                )
                self._stt_ws = stt_ws
                retries      = 0
                log.info(f"[{self.sid}] STT connected ✓")
                self._stt_ready.set()

                async def _push():
                    while self._running:
                        frame = await self._audio_q.get()
                        try:
                            # ALWAYS send audio to STT — no gating.
                            # Echo is handled at the text level in _recv.
                            await stt_ws.send(frame if (frame and frame[0] == 0x01) else b'\x01' + frame)
                        except Exception:
                            pass

                async def _recv():
                    guard = RepetitionGuard()
                    word_buf: list[str]              = []
                    silence_task: Optional[asyncio.Task] = None
                    barge_triggered_this_turn: bool  = False

                    async def _fire_query():
                        nonlocal word_buf, silence_task, barge_triggered_this_turn
                        await asyncio.sleep(STT_SILENCE_MS / 1000.0)
                        if not word_buf:
                            return

                        text       = " ".join(word_buf).strip()
                        word_count = len(word_buf)
                        was_barge  = barge_triggered_this_turn
                        word_buf   = []
                        silence_task              = None
                        barge_triggered_this_turn = False

                        if not text:
                            return

                        # Skip echo-tail / text-echo drops for barge-in utterances
                        if not was_barge:
                            in_echo_tail = (time.monotonic() - self._tts_stopped_at) < ECHO_TAIL_GUARD_S
                            if word_count < 8 and self.state == State.IDLE and in_echo_tail:
                                log.info(f"[{self.sid}] echo-tail drop ({word_count}w, tail): {text!r}")
                                return

                            if self._text_echo_filter.is_echo_segment(text):
                                log.info(f"[{self.sid}] text-echo drop: {text!r}")
                                return

                            if time.monotonic() < self._barge_in_until and word_count < BARGE_IN_MIN_WORDS:
                                log.info(f"[{self.sid}] post-barge-in drop ({word_count}w): {text!r}")
                                return

                        guard.reset()
                        turn_id = str(uuid.uuid4())
                        self._lat.new_turn(turn_id, text)
                        self._lat.on_stt_segment()
                        log.info(f"[{self.sid}] STT silence [{self.state.name}] ({word_count}w){' [BARGE]' if was_barge else ''}: {text!r}")
                        await self._jsend({"type": "segment", "text": text, "barge_in": was_barge})
                        _get("gateway_stt_segments_total").inc()

                        if self.state in (State.SPEAKING, State.THINKING):
                            if not self._barge_in:
                                await self._do_barge_in_immediate()
                            drain_q(self._query_q)
                        await self._query_q.put((turn_id, text))

                    async for raw in stt_ws:
                        if isinstance(raw, bytes) and len(raw) > 1:
                            ftype = raw[0]
                            if ftype != 0x01:
                                continue
                            payload = raw[1:]
                        elif isinstance(raw, str):
                            payload = raw.encode()
                        else:
                            continue

                        try:
                            ev = json.loads(payload)
                        except Exception:
                            continue

                        kind = ev.get("type", "")
                        if kind in ("word", "segment", "barge_in", "vad"):
                            log.info(f"[{self.sid}] STT→ {kind}: {ev.get('word') or ev.get('text') or ev.get('is_voice', '')}")

                        if kind == "barge_in":
                            if not barge_triggered_this_turn:
                                barge_triggered_this_turn = True
                                await self._do_barge_in_immediate()
                                word_buf.clear()
                                if silence_task and not silence_task.done():
                                    silence_task.cancel()
                                silence_task = None
                            continue

                        elif kind == "word":
                            word = ev.get("word", "").strip().rstrip("?.!,;:")
                            if word:
                                # ── Hallucination guard ──
                                if guard.feed(word):
                                    log.warning(f"[{self.sid}] hallucination reset")
                                    word_buf.clear()
                                    guard.reset()
                                    if silence_task and not silence_task.done():
                                        silence_task.cancel()
                                    silence_task = None
                                    await self._jsend({"type": "hallucination_reset"})
                                    continue

                                # ── Accumulate ──
                                if not word_buf or word_buf[-1].lower() != word.lower():
                                    word_buf.append(word)
                                self._lat.on_stt_first_word()

                                # ── Instant barge-in on first real word during AI speech ──
                                if (self.state in (State.SPEAKING, State.THINKING)
                                        and not barge_triggered_this_turn
                                        and len(word_buf) >= 1):
                                    barge_triggered_this_turn = True
                                    await self._do_barge_in_immediate()

                                # Always use silence timer — collects full utterance
                                if silence_task and not silence_task.done():
                                    silence_task.cancel()
                                silence_task = asyncio.create_task(_fire_query())
                            await self._jsend(ev)

                        elif kind == "segment":
                            text       = ev.get("text", "").strip()
                            word_count = len(text.split()) if text else 0
                            if not text:
                                continue

                            if silence_task and not silence_task.done():
                                silence_task.cancel()
                                silence_task = None
                            word_buf.clear()
                            guard.reset()
                            barge_triggered_this_turn = False

                            # Only echo-filter segments in idle state (not during barge-in)
                            if self.state == State.IDLE:
                                in_echo_tail = (time.monotonic() - self._tts_stopped_at) < ECHO_TAIL_GUARD_S
                                if word_count < 6 and in_echo_tail:
                                    log.info(f"[{self.sid}] echo-tail drop (segment): {text!r}")
                                    continue

                                if self._text_echo_filter.is_echo_segment(text):
                                    log.info(f"[{self.sid}] text-echo drop segment: {text!r}")
                                    continue

                            seg_turn_id = str(uuid.uuid4())
                            self._lat.new_turn(seg_turn_id, text)
                            self._lat.on_stt_segment()
                            log.info(f"[{self.sid}] STT segment [{self.state.name}] ({word_count}w): {text!r}")
                            is_barge_seg = self.state in (State.SPEAKING, State.THINKING)
                            await self._jsend({"type": "segment", "text": text, "barge_in": is_barge_seg})

                            if self.state in (State.SPEAKING, State.THINKING):
                                if word_count >= BARGE_IN_MIN_WORDS:
                                    if self._barge_in:
                                        drain_q(self._query_q)
                                        await self._query_q.put((seg_turn_id, text))
                                    else:
                                        await self._do_barge_in_immediate()
                                        drain_q(self._query_q)
                                        await self._query_q.put((seg_turn_id, text))
                            else:
                                await self._query_q.put((seg_turn_id, text))

                        elif kind == "partial":
                            await self._jsend(ev)

                        elif kind == "pong":
                            self._last_pong_time = time.monotonic()

                        elif kind == "error":
                            log.warning(f"[{self.sid}] STT error: {ev}")
                            await self._jsend(ev)

                await asyncio.gather(_push(), _recv())

            except Exception as e:
                retries += 1
                log.warning(f"[{self.sid}] STT disconnected ({e}), retry {retries}/{STT_MAX_RETRIES}")
                self._stt_ws = None
                if retries < STT_MAX_RETRIES:
                    await asyncio.sleep(min(2 ** retries, 30))

    # ─── CAG loop ─────────────────────────────────────────────────────────────

    async def _cag_loop(self):
        self._cag_turn_count = 0
        retries = 0

        while self._running and retries < CAG_MAX_RETRIES:
            try:
                cag_ws = await ws_connect(
                    CAG_WS_URL, max_retries=CAG_MAX_RETRIES,
                    label=f"[{self.sid}] CAG",
                    max_size=4 * 1024 * 1024,
                    ping_interval=WS_PING_INTERVAL,
                    ping_timeout=WS_PING_TIMEOUT,
                )
                self._cag_ws = cag_ws
                retries      = 0
                log.info(f"[{self.sid}] CAG WebSocket connected ✓")

                # ── Load past conversation history on session resume ──
                if self._chat_session_id and self._load_history and self._cag_turn_count == 0:
                    try:
                        history = await self._load_history(self._chat_session_id)
                        if history:
                            self._title_set = True  # session already has messages → title set
                            for msg in history:
                                role = msg.get("role", "")
                                text = msg.get("content", "").strip()
                                if not text:
                                    continue
                                if role == "agent":
                                    # Feed past AI replies into CAG conversation memory
                                    await cag_ws.send(json.dumps({
                                        "type": "query", "turn_id": "history",
                                        "message": f"[HISTORY] AI said: {text}",
                                        "reset": False,
                                    }))
                                # Send to client so it can display chat history
                                await self._jsend({
                                    "type": "history",
                                    "role": role,
                                    "content": text,
                                    "created_at": msg.get("created_at", ""),
                                })
                            # Drain any history responses from CAG
                            try:
                                while True:
                                    raw = await asyncio.wait_for(cag_ws.recv(), timeout=0.5)
                                    frame = json.loads(raw) if isinstance(raw, (str, bytes)) else {}
                                    if frame.get("turn_id") != "history":
                                        break
                                    if frame.get("type") == "done":
                                        break
                            except (asyncio.TimeoutError, Exception):
                                pass
                            log.info(f"[{self.sid}] loaded {len(history)} history messages")
                    except Exception as e:
                        log.warning(f"[{self.sid}] history load failed: {e}")

                while self._running:
                    query = await self._query_q.get()
                    self._last_query_time = time.monotonic()

                    if isinstance(query, tuple):
                        turn_id, query_text = query
                    else:
                        turn_id    = str(uuid.uuid4())
                        query_text = query

                    self._cag_turn_count += 1
                    self._barge_in        = False
                    self._barge_in_event.clear()
                    self.state            = State.THINKING

                    self._lat.new_turn(turn_id, query_text)
                    self._lat.on_query_sent()

                    log.info(f"[{self.sid}] CAG query: {query_text!r}")
                    _get("gateway_cag_queries_total").inc()
                    await self._jsend({"type": "thinking", "turn_id": turn_id})
                    asyncio.create_task(self._persist("user", query_text))

                    try:
                        await cag_ws.send(json.dumps({
                            "type": "query", "turn_id": turn_id,
                            "message": query_text,
                            "reset": self._cag_turn_count == 1,
                        }))
                    except Exception as e:
                        log.error(f"[{self.sid}] CAG send error: {e}")
                        raise

                    await self._process_cag_stream_ws(cag_ws, turn_id)

            except asyncio.CancelledError:
                return
            except Exception as e:
                retries += 1
                self._cag_ws = None
                log.warning(f"[{self.sid}] CAG disconnected ({e}), retry {retries}/{CAG_MAX_RETRIES}")
                if retries < CAG_MAX_RETRIES:
                    await asyncio.sleep(min(2 ** retries, 10))
                    await self._cag_loop_http_fallback()

    async def _process_cag_stream_ws(self, cag_ws, turn_id: str):
        acc              = TonalAccumulator()
        acc.reset()
        full_reply_parts: list[str] = []
        interrupted      = False
        stream_confirmed = False

        try:
            while True:
                # Use event-aware recv so barge-in exits instantly
                recv_task = asyncio.ensure_future(cag_ws.recv())
                barge_task = asyncio.ensure_future(self._barge_in_event.wait())
                done, pending = await asyncio.wait(
                    {recv_task, barge_task}, return_when=asyncio.FIRST_COMPLETED
                )
                for p in pending:
                    p.cancel()

                if barge_task in done or self._barge_in:
                    interrupted = True
                    break

                raw_frame = recv_task.result()

                try:
                    frame = json.loads(raw_frame) if isinstance(raw_frame, (str, bytes)) else {}
                except Exception:
                    continue

                ftype = frame.get("type", "")

                if ftype == "turn_id":
                    if frame.get("turn_id", "") != turn_id:
                        # Stale frame from a cancelled turn — skip it
                        continue
                    stream_confirmed = True
                    continue

                # Skip frames from other turns (stale after barge-in cancel)
                frame_turn = frame.get("turn_id", "")
                if frame_turn and frame_turn != turn_id:
                    continue

                if ftype == "done":
                    break
                if ftype in ("error", "timeout"):
                    await self._jsend({"type": "error", "detail": frame.get("detail", "CAG error")})
                    interrupted = True
                    break
                if ftype != "token" or not stream_confirmed:
                    continue

                token = frame.get("token", "")
                if not token:
                    continue

                self._lat.on_first_token()
                self._lat.on_token()
                full_reply_parts.append(token)
                await self._jsend({"type": "ai_token", "token": token})

                for tc in acc.feed(token):
                    if self._barge_in:
                        interrupted = True
                        break
                    log.info(f"[{self.sid}] TTS← [{tc.tone}] {tc.text!r}")
                    await self._jsend({"type": "ai_sentence", "text": tc.text, "tone": tc.tone})
                    self._text_echo_filter.feed_ai_text(tc.text)
                    if self.state != State.SPEAKING:
                        self.state = State.SPEAKING
                        # Notify STT immediately so barge-in detection activates
                        await self._notify_stt_speaking(True)
                    self._lat.on_tts_chunk_sent(tc.text)
                    await self._tts_q.put(tc)

                if interrupted:
                    break

        except Exception as e:
            log.error(f"[{self.sid}] CAG stream error: {e}")
            interrupted = True
            await self._jsend({"type": "error", "detail": str(e)})
            raise

        finally:
            if not interrupted and not self._barge_in:
                tail = acc.flush()
                if tail:
                    await self._jsend({"type": "ai_sentence", "text": tail.text, "tone": tail.tone})
                    self._text_echo_filter.feed_ai_text(tail.text)
                    self.state = State.SPEAKING
                    self._lat.on_tts_chunk_sent(tail.text)
                    await self._tts_q.put(tail)

                await self._tts_q.put(self._TURN_END)

                full_text = " ".join(full_reply_parts).strip()
                if full_text:
                    asyncio.create_task(self._persist("agent", full_text))
                    if self._stt_ws:
                        ctrl = json.dumps({"type": "assistant_turn", "text": full_text}).encode()
                        try:
                            await self._stt_ws.send(b'\x02' + ctrl)
                        except Exception:
                            pass

    async def _cag_loop_http_fallback(self):
        try:
            query = self._query_q.get_nowait()
        except asyncio.QueueEmpty:
            return

        if isinstance(query, tuple):
            turn_id, query_text = query
        else:
            turn_id    = str(uuid.uuid4())
            query_text = query

        self._cag_turn_count += 1
        self._barge_in        = False
        self._barge_in_event.clear()
        self.state            = State.THINKING
        self._lat.new_turn(turn_id, query_text)
        self._lat.on_query_sent()

        await self._jsend({"type": "thinking", "turn_id": turn_id})
        asyncio.create_task(self._persist("user", query_text))

        acc              = TonalAccumulator()
        acc.reset()
        full_reply_parts: list[str] = []
        interrupted      = False
        stream_confirmed = False

        async with httpx.AsyncClient(
            base_url=CAG_HTTP_URL,
            timeout=httpx.Timeout(connect=5.0, read=None, write=5.0, pool=5.0),
        ) as http:
            try:
                async with http.stream(
                    "POST", "/chat/stream",
                    json={"message": query_text, "reset_session": self._cag_turn_count == 1,
                          "turn_id": turn_id},
                    headers={"Accept": "text/event-stream"},
                ) as resp:
                    async for line in resp.aiter_lines():
                        if self._barge_in:
                            interrupted = True
                            break
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()

                        if data.startswith("[TURN_ID]"):
                            if data[9:].strip() != turn_id:
                                interrupted = True
                                break
                            stream_confirmed = True
                            continue
                        if data == "[DONE]":
                            break
                        if data in ("[TIMEOUT]", "") or data.startswith("[ERROR]"):
                            interrupted = True
                            break
                        if not stream_confirmed:
                            continue

                        self._lat.on_first_token()
                        self._lat.on_token()
                        full_reply_parts.append(data)
                        await self._jsend({"type": "ai_token", "token": data})

                        for tc in acc.feed(data):
                            if self._barge_in:
                                interrupted = True
                                break
                            await self._jsend({"type": "ai_sentence", "text": tc.text, "tone": tc.tone})
                            self.state = State.SPEAKING
                            self._lat.on_tts_chunk_sent(tc.text)
                            await self._tts_q.put(tc)

                        if interrupted:
                            break

            except Exception as e:
                log.error(f"[{self.sid}] CAG HTTP fallback error: {e}")
                interrupted = True

            finally:
                if not interrupted and not self._barge_in:
                    tail = acc.flush()
                    if tail:
                        await self._jsend({"type": "ai_sentence", "text": tail.text, "tone": tail.tone})
                        self.state = State.SPEAKING
                        self._lat.on_tts_chunk_sent(tail.text)
                        await self._tts_q.put(tail)
                    await self._tts_q.put(self._TURN_END)

                    full_text = "".join(full_reply_parts).strip()
                    if full_text:
                        asyncio.create_task(self._persist("agent", full_text))

    # ─── Synth worker ─────────────────────────────────────────────────────────

    async def _synth_worker(self):
        order_index    = 0
        pending_tasks: list[asyncio.Task] = []

        while self._running:
            item = await self._tts_q.get()
            log.info(f"[{self.sid}] synth_worker got item: {type(item).__name__}")

            if item is self._INTERRUPT:
                self._barge_in = True
                for t in pending_tasks:
                    t.cancel()
                pending_tasks.clear()
                order_index = 0
                await self._pcm_q.put(self._INTERRUPT)
                continue

            if item is self._TURN_END:
                total = order_index
                await self._pcm_q.put(("TURN_END", total))
                pending_tasks = [t for t in pending_tasks if not t.done()]
                order_index   = 0
                continue

            idx         = order_index
            order_index += 1
            task = asyncio.create_task(self._synth_one(item, idx), name=f"synth_{self.sid}_{idx}")
            pending_tasks.append(task)
            pending_tasks = [t for t in pending_tasks if not t.done()]

    async def _synth_one(self, tc: TonalChunk, idx: int):
        text = tc.text.strip()
        if not text:
            await self._pcm_q.put((idx, []))
            return

        log.info(f"[{self.sid}] synth[{idx}] START: {text!r}")
        _get("gateway_tts_chunks_total").inc()

        async with self._tts_sem:
            if self._barge_in:
                await self._pcm_q.put((idx, []))
                return

            retries = 0
            t_start = time.monotonic()
            total_bytes = 0

            while retries < TTS_MAX_RETRIES:
                try:
                    first_audio_ok = False
                    async for chunk in azure_tts_stream(text):
                        if self._barge_in:
                            break
                        total_bytes += len(chunk)
                        if not first_audio_ok:
                            first_audio_ok = True
                            if idx == 0:
                                self._lat.on_tts_audio_start()
                            latency_ms = (time.monotonic() - t_start) * 1000
                            log.info(f"[{self.sid}] synth[{idx}] first byte: {latency_ms:.0f}ms")
                        await self._pcm_q.put(("PCM_FRAME", idx, chunk))

                    if not self._barge_in:
                        synth_ms = (time.monotonic() - t_start) * 1000
                        _get("gateway_tts_synth_seconds").observe(synth_ms / 1000.0)
                        duration_sec = total_bytes / (24000 * 2)
                        self._lat.on_tts_chunk_complete(
                            synthesis_latency_ms=synth_ms,
                            synth_duration_ms=synth_ms,
                            duration_sec=duration_sec,
                        )
                        log.info(f"[{self.sid}] synth[{idx}] DONE {total_bytes}B in {synth_ms:.0f}ms")
                    break

                except Exception as e:
                    retries += 1
                    log.warning(f"[{self.sid}] synth[{idx}] Azure TTS error ({e}), retry {retries}")
                    total_bytes = 0
                    if retries < TTS_MAX_RETRIES:
                        await asyncio.sleep(min(2 ** retries, 10))

        await self._pcm_q.put((idx, []))

    # ─── Play worker ──────────────────────────────────────────────────────────

    async def _play_worker(self):
        next_expected    = 0
        reorder_buf:     dict[int, list[bytes]] = {}
        chunk_count      = 0
        total_expected   = -1
        chunks_received  = 0

        async def _flush_ordered():
            nonlocal next_expected, chunk_count
            while next_expected in reorder_buf:
                frames = reorder_buf.pop(next_expected)
                for f in frames:
                    await self._bsend(f)
                    self._echo_gate.feed_tts(f)
                chunk_count   += 1
                next_expected += 1

        async def _finalize_turn():
            nonlocal reorder_buf, next_expected, chunk_count
            nonlocal total_expected, chunks_received
            await _flush_ordered()
            # Mark TTS stopped immediately — the ECHO_TAIL_GUARD_S window
            # in _fire_query and echo_gate.ECHO_TAIL_S handle the tail
            self._tts_stopped_at = time.monotonic()
            self._echo_gate.tts_stopped()
            await self._notify_stt_speaking(False)
            report = self._lat.complete_turn()
            if report:
                e2e = report.get("e2e_ms")
                if e2e:
                    _get("gateway_e2e_latency_seconds").observe(e2e / 1000.0)
                stt_ms = report.get("stt_latency_ms")
                if stt_ms:
                    _get("gateway_stt_latency_seconds").observe(stt_ms / 1000.0)
                cag_ms = report.get("cag_first_token_ms")
                if cag_ms:
                    _get("gateway_cag_first_token_seconds").observe(cag_ms / 1000.0)
                await self._jsend({"type": "latency", "stage": "turn_complete", **report})
            await self._jsend({"type": "done", "chunks": chunk_count})
            reorder_buf.clear()
            next_expected   = 0
            chunk_count     = 0
            total_expected  = -1
            chunks_received = 0
            self.state      = State.IDLE

        while self._running:
            item = await self._pcm_q.get()

            if item is self._INTERRUPT:
                reorder_buf.clear()
                next_expected   = 0
                chunk_count     = 0
                total_expected  = -1
                chunks_received = 0
                self.state      = State.IDLE
                # Don't set _tts_stopped_at here — _do_barge_in_immediate
                # already cleared it so user speech isn't blocked
                self._echo_gate.reset()
                await self._notify_stt_speaking(False)
                report = self._lat.complete_turn()
                if report:
                    report["barge_in"] = True
                    await self._jsend({"type": "latency", "stage": "turn_interrupted", **report})
                continue

            if isinstance(item, tuple) and item[0] == "TURN_END":
                total_expected = item[1]
                await _flush_ordered()
                if total_expected == 0 or chunks_received >= total_expected:
                    await _finalize_turn()
                continue

            if isinstance(item, tuple) and len(item) == 3 and item[0] == "PCM_FRAME":
                _, frame_idx, pcm_bytes = item
                if self._barge_in:
                    continue
                self.state = State.SPEAKING
                await self._notify_stt_speaking(True)   # deduped internally
                if frame_idx == next_expected:
                    await self._bsend(pcm_bytes)
                    self._echo_gate.feed_tts(pcm_bytes)
                else:
                    reorder_buf.setdefault(frame_idx, []).append(pcm_bytes)
                continue

            order_idx, frames = item

            if self._barge_in:
                chunks_received += 1
                continue

            chunks_received += 1
            self.state = State.SPEAKING
            await self._notify_stt_speaking(True)   # deduped internally

            if frames:
                reorder_buf[order_idx] = frames

            if order_idx == next_expected and order_idx not in reorder_buf:
                next_expected += 1
                chunk_count   += 1

            await _flush_ordered()

            if total_expected >= 0 and chunks_received >= total_expected:
                await _flush_ordered()
                if not reorder_buf:
                    await _finalize_turn()
