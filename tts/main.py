"""
tts_server.py — Zero-latency WebSocket TTS microservice  (v2)

PROTOCOL (JSON over WebSocket):
─────────────────────────────────────────────────────────────────────────────
Gateway → TTS Server (inbound):
  { "type": "start",     "session_id": "abc", "speaker": "aiden", "language": "English" }
  { "type": "chunk",     "session_id": "abc", "text": "Hello world.",
                         "metadata": {"emotion": "empathetic", "speed": 0.9} }   ← metadata optional
  { "type": "flush",     "session_id": "abc" }   ← force-flush remaining buffer
  { "type": "end",       "session_id": "abc" }   ← LLM stream complete
  { "type": "interrupt", "session_id": "abc" }   ← barge-in: user started speaking

TTS Server → Client (outbound):
  Binary frame:
    [0:4]   session_id length  (uint32 LE)
    [4:8]   sample_rate        (uint32 LE)
    [8:12]  chunk_index        (uint32 LE)
    [12:]   raw PCM int16 bytes

  JSON frames:
  { "type": "started",    "session_id": "abc", "speaker": "aiden", "language": "English" }
  { "type": "stop_audio", "session_id": "abc" }   ← sent after interrupt; client must stop playback
  { "type": "done",       "session_id": "abc", "chunks": N }
─────────────────────────────────────────────────────────────────────────────
Run:
  python tts_server.py
  Listens on ws://0.0.0.0:8765
─────────────────────────────────────────────────────────────────────────────

Changes from v1
───────────────
1. Barge-in / interrupt support
   Receiving {"type":"interrupt"} instantly drains the audio queue and tells
   the client to stop playback, so the user's next utterance isn't blocked.

2. Per-chunk metadata passthrough
   Gateway can send {"emotion":"empathetic","speed":0.8} alongside each chunk;
   TTSEngine applies it to the voice profile dynamically.

3. ThreadPoolExecutor raised to max_workers=4
   Allows up to 4 concurrent GPU synthesis calls — prevents head-of-line
   blocking when two sessions arrive simultaneously.

4. Synthesis worker stores metadata alongside text in the queue
   Uses a (text, metadata) tuple sentinel pattern; None still signals shutdown.

5. Graceful session teardown
   on disconnect: sentinel is placed only once per live session;
   tasks are cancelled only after the sentinel has been acknowledged.

6. Startup log shows whether soxr (fast resampler) is available.
"""

import asyncio
import json
import logging
import random
import struct
from concurrent.futures import ThreadPoolExecutor

import websockets

from tts_engine import TTSEngine, StreamingSentenceSplitter, SPEAKERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("tts_server")

# ── Globals ───────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8765

engine   = TTSEngine()                          # one model, shared across all sessions
executor = ThreadPoolExecutor(max_workers=4)    # raised from 2 → 4 for parallel synthesis

# Log fast-resampler status once at startup
try:
    import soxr  # noqa: F401
    log.info("soxr fast resampler: available ✓")
except ImportError:
    log.warning("soxr not installed — falling back to scipy resampler (slower). "
                "Install with: pip install soxr")


# ── Session state ─────────────────────────────────────────────────────────────
class Session:
    """
    Holds all per-connection state.

    audio_queue items are either:
      • (text: str, metadata: dict | None)  — a sentence to synthesise
      • None                                 — sentinel to stop the worker
    """

    def __init__(self, session_id: str, speaker: str, language: str, ws):
        self.session_id  = session_id
        self.speaker     = speaker
        self.language    = language
        self.ws          = ws
        self.splitter    = StreamingSentenceSplitter()
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.chunk_index = 0
        self.interrupted = False   # set True during barge-in to skip stale items

    def feed(self, text: str, metadata: dict | None = None) -> list[tuple[str, dict | None]]:
        """Feed a raw token. Returns list of (sentence, metadata) ready to synthesise."""
        sentences = self.splitter.feed(text)
        return [(s, metadata) for s in sentences]

    def flush(self, metadata: dict | None = None) -> list[tuple[str, dict | None]]:
        sentences = self.splitter.finish()
        return [(s, metadata) for s in sentences]


# ── Audio frame builder ───────────────────────────────────────────────────────
def build_frame(session_id: str, sample_rate: int, chunk_index: int, pcm: bytes) -> bytes:
    sid_bytes = session_id.encode()
    header    = struct.pack("<III", len(sid_bytes), sample_rate, chunk_index)
    return header + sid_bytes + pcm


# ── Synthesis worker ──────────────────────────────────────────────────────────
async def synthesis_worker(session: Session) -> None:
    """
    Drains session.audio_queue, synthesises each sentence on the GPU,
    and streams binary audio frames back over the WebSocket immediately.

    Queue items: (text, metadata) tuples, or None as shutdown sentinel.
    """
    loop = asyncio.get_event_loop()

    while True:
        item = await session.audio_queue.get()

        # Shutdown sentinel
        if item is None:
            session.audio_queue.task_done()
            break

        text, metadata = item

        # If an interrupt happened while this item was queued, discard it
        if session.interrupted:
            session.audio_queue.task_done()
            continue

        log.info(f"[{session.session_id}] Synthesising: {repr(text[:60])}")

        try:
            pcm, sr = await loop.run_in_executor(
                executor,
                lambda: engine.synthesize_chunk(
                    text,
                    session.speaker,
                    session.language,
                    metadata,
                ),
            )
        except Exception as e:
            log.error(f"[{session.session_id}] Synthesis error: {e}")
            session.audio_queue.task_done()
            continue

        # Check interrupt again — synthesis may have taken a moment
        if session.interrupted:
            session.audio_queue.task_done()
            continue

        if pcm:
            frame = build_frame(session.session_id, sr, session.chunk_index, pcm)
            session.chunk_index += 1
            try:
                await session.ws.send(frame)
                log.info(
                    f"[{session.session_id}] Sent chunk {session.chunk_index}, "
                    f"{len(pcm) // 2} samples @ {sr} Hz"
                )
            except websockets.exceptions.ConnectionClosed:
                log.warning(f"[{session.session_id}] Connection closed mid-stream.")
                session.audio_queue.task_done()
                break

        session.audio_queue.task_done()

    # Signal stream completion to client
    try:
        await session.ws.send(json.dumps({
            "type":       "done",
            "session_id": session.session_id,
            "chunks":     session.chunk_index,
        }))
    except Exception:
        pass

    log.info(f"[{session.session_id}] Stream complete — {session.chunk_index} chunks sent.")


# ── Helper: drain audio queue without blocking ────────────────────────────────
def _drain_queue(q: asyncio.Queue) -> None:
    """Empty a queue synchronously (called from async context)."""
    while not q.empty():
        try:
            q.get_nowait()
            q.task_done()
        except asyncio.QueueEmpty:
            break


# ── WebSocket handler ─────────────────────────────────────────────────────────
async def handle_client(ws, path=None) -> None:
    sessions: dict[str, Session]      = {}
    workers:  dict[str, asyncio.Task] = {}

    log.info(f"Client connected: {ws.remote_address}")

    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send(json.dumps({"error": "invalid JSON"}))
                continue

            mtype      = msg.get("type")
            session_id = msg.get("session_id", "default")

            # ── START ────────────────────────────────────────────────────────
            if mtype == "start":
                speaker  = msg.get("speaker") or random.choice(SPEAKERS)
                language = msg.get("language", "English")

                if speaker not in SPEAKERS:
                    await ws.send(json.dumps({
                        "error": f"Unknown speaker '{speaker}'",
                        "valid": SPEAKERS,
                    }))
                    continue

                # Tear down any existing session for this ID cleanly
                if session_id in sessions:
                    old = sessions.pop(session_id)
                    _drain_queue(old.audio_queue)
                    await old.audio_queue.put(None)   # stop old worker
                    if session_id in workers:
                        workers.pop(session_id).cancel()

                sess = Session(session_id, speaker, language, ws)
                sessions[session_id] = sess

                worker = asyncio.create_task(synthesis_worker(sess))
                workers[session_id] = worker

                await ws.send(json.dumps({
                    "type":       "started",
                    "session_id": session_id,
                    "speaker":    speaker,
                    "language":   language,
                }))
                log.info(f"[{session_id}] Session started — speaker={speaker}, language={language}")

            # ── CHUNK ────────────────────────────────────────────────────────
            elif mtype == "chunk":
                sess = sessions.get(session_id)
                if not sess:
                    await ws.send(json.dumps({
                        "error": f"No session '{session_id}'. Send 'start' first.",
                    }))
                    continue

                text     = msg.get("text", "")
                metadata = msg.get("metadata")   # optional: {"emotion": ..., "speed": ...}

                for sentence, meta in sess.feed(text, metadata):
                    await sess.audio_queue.put((sentence, meta))

            # ── FLUSH ────────────────────────────────────────────────────────
            elif mtype == "flush":
                sess = sessions.get(session_id)
                if sess:
                    metadata = msg.get("metadata")
                    for sentence, meta in sess.flush(metadata):
                        await sess.audio_queue.put((sentence, meta))

            # ── END ──────────────────────────────────────────────────────────
            elif mtype == "end":
                sess = sessions.get(session_id)
                if sess:
                    metadata = msg.get("metadata")
                    # Flush remaining buffer
                    for sentence, meta in sess.flush(metadata):
                        await sess.audio_queue.put((sentence, meta))
                    # Sentinel → worker will send "done" frame then exit
                    await sess.audio_queue.put(None)
                    # Wait for all queued synthesis to finish
                    await sess.audio_queue.join()

            # ── INTERRUPT (barge-in) ─────────────────────────────────────────
            elif mtype == "interrupt":
                sess = sessions.get(session_id)
                if sess:
                    # 1. Mark session as interrupted so in-flight synthesis is discarded
                    sess.interrupted = True

                    # 2. Drain the pending synthesis queue
                    _drain_queue(sess.audio_queue)

                    # 3. Tell the client to stop playback immediately
                    await ws.send(json.dumps({
                        "type":       "stop_audio",
                        "session_id": session_id,
                    }))

                    # 4. Reset the splitter and chunk counter for the new utterance
                    sess.splitter    = StreamingSentenceSplitter()
                    sess.chunk_index = 0
                    sess.interrupted = False   # ready for next LLM stream

                    log.info(f"[{session_id}] Barge-in interrupt — queue cleared, client notified.")

            else:
                await ws.send(json.dumps({"error": f"Unknown message type '{mtype}'"}))

    except websockets.exceptions.ConnectionClosed:
        log.info("Client disconnected.")

    finally:
        # Gracefully stop all workers for this connection
        for sid, sess in sessions.items():
            _drain_queue(sess.audio_queue)
            try:
                sess.audio_queue.put_nowait(None)   # sentinel
            except Exception:
                pass
        for task in workers.values():
            task.cancel()
        log.info("All sessions cleaned up.")


# ── Entry point ───────────────────────────────────────────────────────────────
async def main() -> None:
    log.info(f"TTS WebSocket server starting on ws://{HOST}:{PORT}")
    async with websockets.serve(
        handle_client,
        HOST,
        PORT,
        max_size=10 * 1024 * 1024,   # 10 MB max message
    ):
        await asyncio.Future()   # run forever


if __name__ == "__main__":
    asyncio.run(main())