"""
test_tts_client.py — Gapless TTS WebSocket test client.

Connects to ws://127.0.0.1:8765, sends text as tiny token-stream chunks,
receives PCM audio chunks, and plays them GAPLESSLY via a continuous
sounddevice OutputStream fed by an async playback queue.

The key fix for gaps:
  • A single OutputStream stays open for the whole session (no open/close per chunk).
  • An asyncio.Queue feeds a background playback coroutine that keeps the
    stream's internal ring-buffer topped up — the OS audio callback never starves.
  • Server-side: jitter buffer (PREFILL_CHUNKS=2) ensures chunks 0 and 1
    arrive together so the client queue is never empty mid-playback.

Usage:
    python test_tts_client.py
    python test_tts_client.py --speaker vivian
    python test_tts_client.py --save output.wav
    python test_tts_client.py --text "Hello," " world" "!"

Requirements:
    pip install websockets sounddevice numpy
    sudo apt install libportaudio2   # Linux
    brew install portaudio           # macOS
"""

import argparse
import asyncio
import json
import struct
import sys
import time
import wave

import numpy as np
import websockets

# ── Config ────────────────────────────────────────────────────────────────────
SERVER_URI  = "ws://127.0.0.1:8765"
SESSION_ID  = "test-session-001"
SAMPLE_RATE = 24_000

# Tiny token-like chunks — simulates a real LLM streaming at ~30 tok/s
TEST_SENTENCES = [
    "Hey", ",", " this", " is", " a", " test", " of", " the",
    " text", "-", "to", "-", "speech", " micro", "service", ".",
    " I'm", " sending", " text", " in", " small", " chunks",
    ",", " just", " like", " a", " language", " model",
    " would", " stream", " tokens", ".",
    " The", " server", " should", " synth", "esize", " each",
    " phrase", " and", " stream", " audio", " back", " in",
    " real", " time", ".",
    " If", " you", " can", " hear", " this", " clearly",
    ",", " the", " pipeline", " is", " working", " correctly", ".",
    " Thanks", " for", " listening", "!",
]

SPEAKERS = ["serena", "vivian", "ono_anna", "sohee", "aiden", "ryan"]


# ── Audio helpers ─────────────────────────────────────────────────────────────
def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def _try_import_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        return None


# ── Gapless playback engine ───────────────────────────────────────────────────
class GaplessPlayer:
    """
    Wraps a single sounddevice OutputStream that stays open for the whole
    session. Audio chunks are pushed via put() and consumed by a background
    asyncio task that calls stream.write() continuously.

    Why this fixes gaps:
      sounddevice.OutputStream.write() blocks until the OS ring-buffer has
      space. If we call it once per received chunk (and chunks arrive slowly),
      the ring-buffer drains between calls → audible gap/click.

      Instead we keep a Python asyncio.Queue of float32 arrays. The _player
      coroutine drains that queue as fast as the stream can consume data,
      so the ring-buffer stays topped up regardless of network jitter.
    """

    BLOCK = 2048   # frames per write — ~85ms at 24kHz; trade-off: lower = less latency, higher = fewer underruns

    def __init__(self, sd, sample_rate: int):
        self._sd   = sd
        self._sr   = sample_rate
        self._q: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=32)
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.BLOCK,
            latency="low",
        )
        self._stream.start()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._player())

    async def put(self, pcm_bytes: bytes) -> None:
        """Enqueue a raw PCM-16 chunk for playback."""
        arr = pcm16_to_float32(pcm_bytes)
        await self._q.put(arr)

    async def stop(self) -> None:
        """Drain remaining audio then shut down cleanly."""
        await self._q.put(None)          # sentinel
        if self._task:
            await self._task
        await asyncio.sleep(0.1)         # let OS ring-buffer drain
        self._stream.stop()
        self._stream.close()

    async def _player(self) -> None:
        """Background coroutine: feeds the OutputStream in BLOCK-sized writes."""
        leftover = np.zeros(0, dtype=np.float32)

        while True:
            # Pull next chunk (or sentinel)
            chunk = await self._q.get()
            if chunk is None:
                # Flush leftover silence-padded to a full block
                if len(leftover) > 0:
                    pad = np.zeros(self.BLOCK - len(leftover) % self.BLOCK, dtype=np.float32)
                    self._stream.write(np.concatenate([leftover, pad]))
                break

            # Combine with any leftover from previous chunk
            data = np.concatenate([leftover, chunk]) if len(leftover) else chunk

            # Write in BLOCK-sized pieces so write() never blocks long
            i = 0
            while i + self.BLOCK <= len(data):
                self._stream.write(data[i : i + self.BLOCK])
                i += self.BLOCK
                await asyncio.sleep(0)   # yield to event loop between blocks

            leftover = data[i:]          # keep sub-block tail for next chunk


# ── Main client ───────────────────────────────────────────────────────────────
async def run_test(
    text_chunks: list[str],
    speaker: str,
    save_path: str | None,
    chunk_delay_ms: int,
) -> None:
    sd = _try_import_sounddevice()
    if sd is None and save_path is None:
        print("⚠️  sounddevice not installed — saving to tts_output.wav")
        save_path = "tts_output.wav"

    all_pcm: list[bytes] = []
    received_chunks      = 0
    t_first_audio: float | None = None

    print(f"\n{'='*60}")
    print(f"  TTS WebSocket Test Client  (gapless v2)")
    print(f"  Server  : {SERVER_URI}")
    print(f"  Speaker : {speaker}")
    print(f"  Tokens  : {len(text_chunks)}")
    print(f"  Output  : {'▶  live gapless playback' if sd and not save_path else save_path or 'tts_output.wav'}")
    print(f"{'='*60}\n")

    player: GaplessPlayer | None = None

    try:
        async with websockets.connect(
            SERVER_URI,
            max_size=10 * 1024 * 1024,
            open_timeout=5,
        ) as ws:
            print("✅ Connected\n")

            # ── START ─────────────────────────────────────────────────────────
            await ws.send(json.dumps({
                "type":       "start",
                "session_id": SESSION_ID,
                "speaker":    speaker,
                "language":   "English",
            }))
            started = json.loads(await ws.recv())
            if started.get("type") != "started":
                print(f"❌ {started}")
                return
            print(f"✅ Session started — speaker={started['speaker']}\n")

            # Spin up gapless player now so it's ready for first audio
            if sd and not save_path:
                player = GaplessPlayer(sd, SAMPLE_RATE)
                player.start()
                print("🔊 Gapless audio stream open\n")

            # ── SEND tokens ───────────────────────────────────────────────────
            t_send_start = time.perf_counter()

            async def send_tokens() -> None:
                for i, tok in enumerate(text_chunks):
                    print(f"  → tok {i+1:>2}/{len(text_chunks)}: {repr(tok)}")
                    await ws.send(json.dumps({
                        "type":       "chunk",
                        "session_id": SESSION_ID,
                        "text":       tok,
                    }))
                    await asyncio.sleep(chunk_delay_ms / 1000)
                await ws.send(json.dumps({"type": "flush", "session_id": SESSION_ID}))
                await ws.send(json.dumps({"type": "end",   "session_id": SESSION_ID}))
                print("\n  ✅ All tokens sent\n")

            send_task = asyncio.create_task(send_tokens())

            # ── RECEIVE audio ─────────────────────────────────────────────────
            async for message in ws:
                if isinstance(message, bytes):
                    if len(message) < 12:
                        continue
                    sid_len, sr, chunk_idx = struct.unpack_from("<III", message, 0)
                    pcm_bytes = message[12 + sid_len:]
                    if not pcm_bytes:
                        continue

                    if t_first_audio is None:
                        t_first_audio = time.perf_counter()
                        ttfa_ms = (t_first_audio - t_send_start) * 1000
                        print(f"  ⚡ TTFA = {ttfa_ms:.0f} ms\n")

                    samples     = len(pcm_bytes) // 2
                    duration_ms = samples / SAMPLE_RATE * 1000
                    gap_warn    = "  ⚠ possible gap" if received_chunks > 0 and chunk_idx != received_chunks else ""
                    print(f"  ◀ chunk {chunk_idx:>3}  |  {samples:>6} smp  |  {duration_ms:>6.1f} ms  |  {sr} Hz{gap_warn}")

                    all_pcm.append(pcm_bytes)
                    received_chunks += 1

                    if player:
                        await player.put(pcm_bytes)

                elif isinstance(message, str):
                    msg   = json.loads(message)
                    mtype = msg.get("type")
                    if mtype == "done":
                        print(f"\n✅ Done — {msg.get('chunks','?')} chunks from server")
                        break
                    elif mtype == "error":
                        print(f"❌ Server error: {msg}")
                        break

            await send_task
            if player:
                await player.stop()

    except ConnectionRefusedError:
        print(f"❌ Cannot connect to {SERVER_URI}")
        print("   Start the server first:  python main.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ {e}")
        raise

    # ── Summary + save ────────────────────────────────────────────────────────
    if not all_pcm:
        print("\n⚠️  No audio received.")
        return

    total_samples  = sum(len(b) // 2 for b in all_pcm)
    total_duration = total_samples / SAMPLE_RATE
    print(f"\n{'─'*60}")
    print(f"  Chunks received : {received_chunks}")
    print(f"  Total audio     : {total_duration:.2f} s  ({total_samples:,} samples)")
    print(f"{'─'*60}")

    out_path = save_path or ("tts_output.wav" if not sd else None)
    if out_path:
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(all_pcm))
        print(f"\n💾 Saved → {out_path}")
        print(f"   Play:  ffplay {out_path}   /   aplay {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    global SERVER_URI
    p = argparse.ArgumentParser(description="Gapless TTS WebSocket test client")
    p.add_argument("--text",    "-t", nargs="+",  help="Text tokens to send (each arg = one chunk)")
    p.add_argument("--speaker", "-s", default="aiden", choices=SPEAKERS)
    p.add_argument("--save",          default=None, metavar="FILE.wav")
    p.add_argument("--delay",   "-d", type=int, default=30, metavar="MS",
                   help="ms between token sends (default 30, mimics ~33 tok/s LLM)")
    p.add_argument("--server",        default=SERVER_URI)
    args = p.parse_args()

    SERVER_URI = args.server
    asyncio.run(run_test(
        text_chunks   = args.text or TEST_SENTENCES,
        speaker       = args.speaker,
        save_path     = args.save,
        chunk_delay_ms= args.delay,
    ))


if __name__ == "__main__":
    main()