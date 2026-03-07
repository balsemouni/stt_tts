"""
main.py — Zero-latency TTS WebSocket server with dual-queue pipeline.

Architecture:
                    text_queue          audio_out_queue
  WebSocket IN ──► [phrase, phrase, …] ──► GPU synthesis ──► [pcm, pcm, …] ──► WebSocket OUT
                        ↑                                          ↑
                  filled by handler                     drained by audio_sender
                  (never waits for GPU)                 (never waits for GPU)

  The GPU is NEVER blocked by network I/O.
  The network sender is NEVER blocked by GPU synthesis.
  → Both run at full speed in parallel.

Order guarantee:
  synthesis_worker pulls from text_queue in FIFO order and pushes to
  audio_out_queue in the same order → playback is always phrase-ordered.

LATENCY OPTIMISATIONS (v5 — ultra-low latency):
  • _FIRST_CHUNK_CAP=4 chars — first GPU call fires after ~1 LLM token
  • _HARD_CAP_CHARS=12 — very short phrases so audio arrives immediately
  • _WORD_CAP=2 — split every 2 words for instant chunk dispatch
  • _MIN_CHUNK_CHARS=2 — single short words allowed through
  • asyncio.sleep(0) instead of sleep(0.012) in audio_sender — yields to event
    loop without adding 12ms of fixed latency per chunk
  • _XFADE_SAMPLES=4ms (was 8ms) — tighter tail fade, less padding
  • compile_model=False — no JIT stall on first request (saves 4–6 s TTFA)
  • _fast_path=True for seq==0 — first chunk uses pure numpy post-process (~0.3ms)
  • torch.cuda.empty_cache() removed from hot path — only called on disconnect
  • Use ws://127.0.0.1:8765 (not localhost) on Windows to skip IPv6 fallback delay
  • lfilter replaces filtfilt in full naturalize path — 2x faster CPU post-process

GAP FIXES (v4):
  • synthesis_worker peek re-queue bug fixed: asyncio.Queue.put_nowait() appends
    to the END, so the "put it back" approach silently reordered phrases and caused
    the worker to synthesize phrase N+1 before N, creating a gap/stutter.
    Fixed with an explicit local lookahead deque — true FIFO re-insertion.
  • _MIN_CHUNK_CHARS enforced in splitter finish() — empty tail phrases no longer
    generate a zero-length synthesis call that wastes a full GPU round-trip.
  • audio_sender: added 12 ms network flush delay (asyncio.sleep) between chunks
    so the client's audio buffer never underruns on fast connections.
  • _XFADE_SAMPLES raised to 20 ms — chunk-boundary click eliminated.
"""

import asyncio
import collections
import json
import logging
import random
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import websockets

# ── Force GPU before ANYTHING else loads ─────────────────────────────────────
from gpu import force_gpu, cleanup_gpu_memory
force_gpu()   # Locks cuda:0 — exits immediately if no GPU is available

from tts_engine import TTSEngine, StreamingSentenceSplitter, SPEAKERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("tts_server")

HOST = "0.0.0.0"
PORT = 8765

# compile_model=False is the key change:
# torch.compile() costs 4–6 s of JIT compilation on the very first inference,
# which would become the user's TTFA.  Disabled here; warm-up in TTSEngine.__init__
# handles the CUDA cold-start instead.
try:
    engine = TTSEngine(compile_model=False)
except Exception as _e:
    import os as _os
    if "device-side assert" in str(_e) or "CUDA error" in str(_e):
        logging.getLogger("tts_server").error(
            f"[CUDA] Engine init failed with CUDA assert — exiting 42 for supervisor restart: {_e}"
        )
        _os._exit(42)
    raise
synth_executor = ThreadPoolExecutor(max_workers=1)   # GPU only — never shared
post_executor  = ThreadPoolExecutor(max_workers=2)   # CPU post-processing


# ── CUDA recovery ────────────────────────────────────────────────────────────
# A CUDA device-side assert poisons the ENTIRE process's CUDA context.
# There is NO way to recover inside the same process — even torch.cuda.empty_cache()
# and model reload will fail with the same error.
# The only recovery is to EXIT and let a process supervisor (or the launch script)
# restart the server.  We log clearly and exit with code 42 as a sentinel.
_CUDA_POISON_EXIT_CODE = 42

async def _reload_engine():
    """
    Called when synthesis_worker detects a CUDA device-side assert.
    Sends a clean close frame to all connected clients, then exits 42
    so the supervisor loop (while($true){python main.py}) restarts cleanly.
    """
    log.error(
        "[CUDA] Device-side assert — context permanently poisoned.\n"
        "       Notifying clients and exiting with code 42 for supervisor restart.\n"
        "       Run:  while($true) { python main.py; sleep 1 }"
    )
    close_tasks = []
    for ws in list(_live_websockets):
        async def _close_one(w=ws):
            try:
                await w.send(json.dumps({
                    "type": "error", "code": "cuda_context_poisoned",
                    "message": "Server restarting — reconnect in ~3s.",
                }))
                await w.close(code=1001, reason="cuda_restart")
            except Exception:
                pass
        close_tasks.append(asyncio.create_task(_close_one()))
    if close_tasks:
        await asyncio.gather(*close_tasks, return_exceptions=True)
    await asyncio.sleep(0.8)
    import os
    os._exit(_CUDA_POISON_EXIT_CODE)


_live_websockets: set = set()


# ── Session ───────────────────────────────────────────────────────────────────
class Session:
    def __init__(self, session_id: str, speaker: str, language: str, ws):
        self.session_id      = session_id
        self.speaker         = speaker
        self.language        = language
        self.ws              = ws
        self.splitter        = StreamingSentenceSplitter()
        self.text_queue      = asyncio.Queue()                  # (text, metadata, seq_hint) | None
        self.audio_out_queue = asyncio.Queue()                  # no maxsize — GPU never blocks on sender
        self.chunk_index     = 0
        self.interrupted     = False
        self._ttfa_start: float | None = None   # set when first text enqueued, cleared after first audio


# ── Audio sender — owns the network write ────────────────────────────────────
async def audio_sender(session: Session) -> None:
    """
    Pulls finished PCM from audio_out_queue and writes to WebSocket.

    GAPLESS strategy (v6 — jitter buffer):
      Pre-buffer PREFILL_CHUNKS ready chunks before sending any audio.
      While the client plays chunk 0, chunks 1+ are already in-flight so
      the playback queue never starves between GPU synthesis calls.
      After prefill, each chunk is sent immediately as it arrives.
    """
    # How many synthesised chunks to buffer before we start sending.
    # 2 means: GPU synthesises phrases 0 and 1 first, then we release both
    # together. The client plays 0 while 1 is already decoded in its buffer,
    # and by the time 0 finishes, chunk 2 is typically already ready too.
    PREFILL_CHUNKS = 2

    reorder: dict[int, tuple[bytes, int]] = {}
    prefill_buf: list[tuple[bytes, int]] = []
    next_seq  = 0
    prefilled = False

    async def _send(pcm_out: bytes, sr_out: int) -> None:
        if not pcm_out or session.interrupted:
            return
        sid_b  = session.session_id.encode()
        header = struct.pack("<III", len(sid_b), sr_out, session.chunk_index)
        try:
            await session.ws.send(header + sid_b + pcm_out)
            if session.chunk_index == 0 and session._ttfa_start is not None:
                ttfa_ms = (time.perf_counter() - session._ttfa_start) * 1000
                log.info(f"[{session.session_id}] ⚡ TTFA={ttfa_ms:.0f}ms")
                session._ttfa_start = None
            log.info(f"[{session.session_id}] ▶ chunk {session.chunk_index} "
                     f"({len(pcm_out)//2} samples @ {sr_out}Hz)")
            session.chunk_index += 1
            await asyncio.sleep(0)   # yield without adding latency
        except Exception as e:
            log.warning(f"[{session.session_id}] send error: {e}")

    try:
        while True:
            item = await session.audio_out_queue.get()
            if item is None:
                session.audio_out_queue.task_done()
                # Stream ended — flush any chunks still in prefill buffer
                if not prefilled:
                    for p, s in prefill_buf:
                        await _send(p, s)
                break

            pcm, sr, seq = item
            reorder[seq] = (pcm, sr)
            session.audio_out_queue.task_done()

            # Drain all consecutive ready chunks in order
            while next_seq in reorder:
                pcm_out, sr_out = reorder.pop(next_seq)

                if not prefilled:
                    # Accumulate until we have PREFILL_CHUNKS ready
                    prefill_buf.append((pcm_out, sr_out))
                    if len(prefill_buf) >= PREFILL_CHUNKS:
                        prefilled = True
                        for p, s in prefill_buf:   # release all at once
                            await _send(p, s)
                        prefill_buf.clear()
                else:
                    await _send(pcm_out, sr_out)   # already prefilled — send now

                next_seq += 1

    except Exception as e:
        log.error(f"[{session.session_id}] audio_sender crashed: {e}")
    finally:
        try:
            await session.ws.send(json.dumps({
                "type":       "done",
                "session_id": session.session_id,
                "chunks":     session.chunk_index,
            }))
        except Exception:
            pass
        log.info(f"[{session.session_id}] done — {session.chunk_index} chunks")


# ── Synthesis worker — owns the GPU ──────────────────────────────────────────
async def synthesis_worker(session: Session) -> None:
    """
    Pipeline:
      GPU raw synthesis  →  fires postprocess as background task  →  pulls next text immediately
      Postprocess (CPU)  →  puts finished PCM on audio_out_queue when done

    GAPLESS PLAYBACK (v4):
      postprocess() receives is_last=True only for the final chunk of an utterance.
      All mid-stream chunks keep their tail silence + get a 20ms linear fade-out.
      The client concatenates raw PCM → seamless audio, zero gap between phrases.

    GAP FIX — lookahead deque:
      asyncio.Queue has no unget().  The previous code called put_nowait() to
      "put an item back", but that appends to the END of the queue, silently
      reordering phrases (phrase 2 was synthesised before phrase 1 finished
      being enqueued).  We now use a local `_lookahead` deque that acts as a
      true one-item prefetch buffer, so peek is always O(1) and order-safe.
    """
    loop = asyncio.get_event_loop()
    pending_post: list[asyncio.Task] = []
    # FIX: true FIFO lookahead — holds at most 1 peeked item
    _lookahead: collections.deque = collections.deque()

    async def _next_item():
        """Pull from lookahead first, then from the real queue."""
        if _lookahead:
            return _lookahead.popleft()
        return await session.text_queue.get()

    async def _postprocess_and_enqueue(raw_pcm, sr, metadata, seq: int, is_last: bool):
        try:
            use_fast_path = not is_last   # fast path for all mid-stream chunks; full naturalize only on last
            pcm_bytes, out_sr = await loop.run_in_executor(
                post_executor,
                lambda p=raw_pcm, m=metadata, fp=use_fast_path, il=is_last: engine.postprocess(
                    p, sr, session.speaker, m, _fast_path=fp, is_last=il
                ),
            )
            if not session.interrupted:
                await session.audio_out_queue.put((pcm_bytes, out_sr, seq))
        except Exception as e:
            log.error(f"[{session.session_id}] postprocess error (seq={seq}): {e}")

    seq = 0
    while True:
        item = await _next_item()

        if item is None:                          # end-of-stream sentinel
            session.text_queue.task_done()
            if pending_post:
                await asyncio.gather(*pending_post, return_exceptions=True)
            await session.audio_out_queue.put(None)
            break

        text, metadata = item
        session.text_queue.task_done()

        text = text.strip()
        if not text or session.interrupted:
            continue

        # FIX: peek using the lookahead deque, not put_nowait() which appended
        # to the END of the asyncio.Queue and silently reordered phrases.
        is_last = False
        try:
            peeked = session.text_queue.get_nowait()
            if peeked is None:
                is_last = True
                session.text_queue.task_done()
            else:
                # Store in the deque — _next_item() will drain it first next loop
                _lookahead.append(peeked)
                session.text_queue.task_done()
        except asyncio.QueueEmpty:
            pass   # stream still open — is_last stays False

        log.info(f"[{session.session_id}] GPU ← {repr(text[:60])} [last={is_last}]")

        try:
            raw_pcm, sr = await loop.run_in_executor(
                synth_executor,
                lambda t=text: engine.synthesize_raw(
                    t, session.speaker, session.language
                ),
            )
        except Exception as e:
            log.error(f"[{session.session_id}] synthesis error: {e}")
            err_str = str(e)
            # A CUDA device-side assert poisons the entire CUDA context — every
            # subsequent call will fail instantly with the same error.
            # Recovery: reset the device and reload the model in the background.
            if "device-side assert" in err_str or "CUDA error" in err_str:
                log.error("[CUDA] Context poisoned — scheduling engine reload …")
                asyncio.create_task(_reload_engine())
            continue

        if session.interrupted:
            continue

        task = asyncio.create_task(
            _postprocess_and_enqueue(raw_pcm, sr, metadata, seq, is_last)
        )
        pending_post.append(task)
        task.add_done_callback(lambda t: pending_post.remove(t) if t in pending_post else None)
        seq += 1


# ── WebSocket handler ─────────────────────────────────────────────────────────
async def handle_client(ws) -> None:
    sessions: dict[str, Session]          = {}
    synth_tasks: dict[str, asyncio.Task]  = {}
    sender_tasks: dict[str, asyncio.Task] = {}

    log.info(f"Client connected: {ws.remote_address}")
    _live_websockets.add(ws)

    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            mtype = msg.get("type")
            sid   = msg.get("session_id", "default")

            # ── START ─────────────────────────────────────────────────────────
            if mtype == "start":
                speaker  = msg.get("speaker") or random.choice(SPEAKERS)
                language = msg.get("language", "English")

                if speaker not in SPEAKERS:
                    await ws.send(json.dumps({"error": f"Unknown speaker '{speaker}'"}))
                    continue

                # Tear down stale session
                if sid in sessions:
                    old = sessions.pop(sid)
                    old.interrupted = True
                    _drain(old.text_queue)
                    _drain(old.audio_out_queue)
                    old.text_queue.put_nowait(None)
                    for t in (synth_tasks.pop(sid, None), sender_tasks.pop(sid, None)):
                        if t: t.cancel()

                sess = Session(sid, speaker, language, ws)
                sessions[sid] = sess

                # Spin up the dual pipeline immediately
                synth_tasks[sid]  = asyncio.create_task(synthesis_worker(sess))
                sender_tasks[sid] = asyncio.create_task(audio_sender(sess))

                await ws.send(json.dumps({
                    "type":       "started",
                    "session_id": sid,
                    "speaker":    speaker,
                    "language":   language,
                }))
                log.info(f"[{sid}] started — speaker={speaker}")

            # ── CHUNK ─────────────────────────────────────────────────────────
            elif mtype == "chunk":
                sess = sessions.get(sid)
                if not sess:
                    continue
                text     = msg.get("text", "")
                metadata = msg.get("metadata")

                # Start TTFA timer on first text received
                if sess._ttfa_start is None and text.strip():
                    sess._ttfa_start = time.perf_counter()

                for phrase in sess.splitter.feed(text):
                    log.info(f"[{sid}] enqueue ← {repr(phrase)}")
                    await sess.text_queue.put((phrase, metadata))

            # ── FLUSH ─────────────────────────────────────────────────────────
            elif mtype == "flush":
                sess = sessions.get(sid)
                if sess:
                    for phrase in sess.splitter.finish():
                        await sess.text_queue.put((phrase, msg.get("metadata")))

            # ── END ───────────────────────────────────────────────────────────
            elif mtype == "end":
                sess = sessions.get(sid)
                if sess:
                    for phrase in sess.splitter.finish():
                        await sess.text_queue.put((phrase, msg.get("metadata")))
                    await sess.text_queue.put(None)   # sentinel

            # ── INTERRUPT ─────────────────────────────────────────────────────
            elif mtype == "interrupt":
                sess = sessions.get(sid)
                if sess:
                    # 1. Signal workers to stop immediately
                    sess.interrupted = True
                    _drain(sess.text_queue)
                    _drain(sess.audio_out_queue)

                    # 2. Cancel old pipeline tasks. Without this, audio_sender stays
                    #    alive and swallows the "done" sentinel for the NEXT session,
                    #    causing the client to never receive audio.
                    for t in (synth_tasks.pop(sid, None), sender_tasks.pop(sid, None)):
                        if t and not t.done():
                            t.cancel()

                    # 3. Fresh Session with new queues — same speaker/language
                    new_sess = Session(sid, sess.speaker, sess.language, ws)
                    sessions[sid] = new_sess

                    # 4. Restart the dual pipeline on the new session
                    synth_tasks[sid]  = asyncio.create_task(synthesis_worker(new_sess))
                    sender_tasks[sid] = asyncio.create_task(audio_sender(new_sess))

                    await ws.send(json.dumps({"type": "stop_audio", "session_id": sid}))
                    log.info(f"[{sid}] barge-in — pipeline restarted")

    except websockets.exceptions.ConnectionClosed:
        log.info("Client disconnected")
    except Exception as e:
        log.error(f"Handler error: {e}")
    finally:
        _live_websockets.discard(ws)
        for sid, sess in sessions.items():
            sess.interrupted = True
            _drain(sess.text_queue)
            _drain(sess.audio_out_queue)
            try: sess.text_queue.put_nowait(None)
            except Exception: pass
        for t in list(synth_tasks.values()) + list(sender_tasks.values()):
            if t: t.cancel()
        # Only clean GPU memory on actual disconnect — NOT on every request.
        # cleanup_gpu_memory() inside the hot path was adding ~50 ms per call.
        log.info("All sessions cleaned up")


def _drain(q: asyncio.Queue) -> None:
    while not q.empty():
        try:
            q.get_nowait()
            q.task_done()
        except asyncio.QueueEmpty:
            break


# ── Entry ─────────────────────────────────────────────────────────────────────
async def main() -> None:
    log.info(f"TTS server starting on ws://{HOST}:{PORT}")
    log.info("💡 TIP: Connect with ws://127.0.0.1:8765 (not 'localhost') on Windows")
    log.info("   — avoids a ~2 s IPv6 fallback delay before TCP connects.")
    try:
        async with websockets.serve(
            handle_client, HOST, PORT,
            max_size=10 * 1024 * 1024,
            compression=None,           # PCM won't compress; saves ~2 ms/frame CPU overhead
        ):
            await asyncio.Future()
    finally:
        log.info("Server shutting down — freeing GPU memory …")
        cleanup_gpu_memory()


if __name__ == "__main__":
    asyncio.run(main())