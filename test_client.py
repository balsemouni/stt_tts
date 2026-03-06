#!/usr/bin/env python3
"""
test_client.py — test client for the STT microservice (main.py)
────────────────────────────────────────────────────────────────
Connects directly to the STT WebSocket at /stream/mux.

What it does
────────────
  • Opens mic → streams raw int16 PCM chunks to the server (0x01 frames)
  • Displays words appearing live on one line as they are confirmed
  • Shows the in-progress partial word in [brackets]
  • Prints the full sentence on its own line when a segment is finalised
  • Shows enrollment progress until the speaker is locked in
  • Handles barge-in, speaker-rejected, and error events

There is NO gateway here — this client speaks directly to main.py.
Echo cancellation is handled server-side by AECGate, not by muting the mic.

Install:  pip install sounddevice websocket-client numpy
Run:      python test_client.py

Options:
  --url     ws://localhost:8001/stream/mux   (STT microservice WebSocket)
  --chunk   320                              samples per mic frame (20ms)
"""

import argparse
import json
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import websocket

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE   = 16000
CHANNELS      = 1
DTYPE         = "float32"
DEFAULT_CHUNK = 320   # 20 ms

# ── Build mux audio frame (0x01 header + int16 PCM body) ─────────────────────

def _build_frame(pcm: np.ndarray) -> bytes:
    int16 = (pcm * 32768).clip(-32768, 32767).astype("int16")
    return bytes([0x01]) + int16.tobytes()


# ── Thread-safe console display ───────────────────────────────────────────────

_lock            = threading.Lock()
_words_on_line:  list[str] = []   # confirmed words for the current utterance
_last_enroll_pct = -1
_partial_active  = False           # True when a partial is showing on current line


def _clear_line():
    """Overwrite current line with spaces then return to column 0."""
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()


def _redraw_words(partial: str = ""):
    """
    Redraw the live word line.
    Shows the last 15 confirmed words + optional [partial] at the end.
    """
    display = _words_on_line[-15:]
    line    = "🎙  " + " ".join(display)
    if partial:
        line += f" [{partial}]"
    sys.stdout.write("\r" + line + "   ")
    sys.stdout.flush()


# ── WebSocket message handler ─────────────────────────────────────────────────

def on_message(ws_app, raw):
    global _last_enroll_pct, _words_on_line, _partial_active

    # Binary frames are not expected from this server — ignore silently
    if isinstance(raw, bytes):
        return

    try:
        ev = json.loads(raw)
        t  = ev.get("type")
    except Exception:
        return

    with _lock:

        # ── Live confirmed word ────────────────────────────────────────────
        if t == "word":
            word = ev.get("word", "").strip()
            if word:
                if len(_words_on_line) > 30:
                    # Safety: reset if we somehow accumulated too many words
                    # without a segment (hallucination guard)
                    _words_on_line = []
                _words_on_line.append(word)
                _partial_active = False
                _redraw_words()

        # ── In-progress partial word ───────────────────────────────────────
        elif t == "partial":
            partial = ev.get("word", "").strip()
            if partial:
                _partial_active = True
                _redraw_words(partial=partial)

        # ── Full utterance finalised ───────────────────────────────────────
        elif t == "segment":
            text = ev.get("text", "").strip()
            _clear_line()
            _words_on_line  = []
            _partial_active = False
            print(f"📝 TRANSCRIPT: {text}\n", flush=True)

        # ── Enrollment progress ────────────────────────────────────────────
        elif t == "enrollment":
            if ev.get("enrolled"):
                if _last_enroll_pct != 100:
                    _clear_line()
                    print("🎉 Speaker enrolled — transcription active.\n", flush=True)
                    _last_enroll_pct = 100
            else:
                pct = int(ev.get("progress", 0) * 100)
                if pct != _last_enroll_pct:
                    _last_enroll_pct = pct
                    # Show enrollment bar inline alongside live words so the
                    # user can see transcription AND enrollment progress at once.
                    bar_filled = pct // 5   # 20-char bar
                    bar = "█" * bar_filled + "░" * (20 - bar_filled)
                    sys.stdout.write(f"\r  [enroll {bar} {pct:3d}%]   ")
                    sys.stdout.flush()

        # ── Barge-in (user interrupted — utterance reset) ─────────────────
        elif t == "barge_in":
            _clear_line()
            _words_on_line  = []
            _partial_active = False
            print("⚡ BARGE-IN — utterance reset\n", flush=True)

        # ── Speaker mismatch ───────────────────────────────────────────────
        elif t == "segment_rejected":
            sim = ev.get("similarity")
            _clear_line()
            _words_on_line  = []
            _partial_active = False
            print(f"🚫 REJECTED — speaker mismatch (sim={sim})\n", flush=True)

        # ── VAD state change (voice ↔ silence transitions only) ───────────
        elif t == "vad":
            # Uncomment the line below for verbose VAD debug output:
            # print(f"  [vad] is_voice={ev.get('is_voice')} prob={ev.get('prob')}", flush=True)
            pass

        # ── AEC suppression (echo gate active) ────────────────────────────
        elif t == "aec":
            # Uncomment for debug:
            # print(f"  [aec] suppressed={ev.get('suppressed')}", flush=True)
            pass

        # ── Latency stats ──────────────────────────────────────────────────
        elif t == "latency":
            # Uncomment for debug:
            # asr_ms = ev.get("asr_inference_ms", 0)
            # rtf    = ev.get("realtime_factor", 0)
            # print(f"  [latency] asr={asr_ms:.1f}ms rtf={rtf:.2f}", flush=True)
            pass

        # ── Errors ────────────────────────────────────────────────────────
        elif t == "error":
            _clear_line()
            print(f"\n❌ ERROR: {ev.get('detail', 'unknown')}\n", flush=True)

        elif t == "context_reset":
            pass   # server cleared Whisper context — no action needed on client

        elif t == "pong":
            pass


def on_open(ws_app):
    print(
        "✅  Connected to STT microservice\n"
        "🎙   Speak — words appear live, full sentence prints on silence.\n"
        "     Press Ctrl+C to quit.\n",
        flush=True,
    )


def on_error(ws_app, err):
    print(f"\n❌  WebSocket error: {err}", flush=True)


def on_close(ws_app, *_):
    print("\n🔌  Disconnected.", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="STT microservice test client")
    ap.add_argument(
        "--url",
        default="ws://localhost:8001/stream/mux",
        help="STT WebSocket URL (default: ws://localhost:8001/stream/mux)",
    )
    ap.add_argument(
        "--chunk",
        type=int,
        default=DEFAULT_CHUNK,
        help="Mic chunk size in samples (default 320 = 20ms)",
    )
    args = ap.parse_args()

    chunk_ms = args.chunk / SAMPLE_RATE * 1000
    print(f"Connecting to {args.url} …", flush=True)

    ws = websocket.WebSocketApp(
        args.url,
        on_open    = on_open,
        on_message = on_message,
        on_error   = on_error,
        on_close   = on_close,
    )

    ws_thread = threading.Thread(
        target=lambda: ws.run_forever(ping_interval=30, ping_timeout=10),
        daemon=True,
        name="WSThread",
    )
    ws_thread.start()
    time.sleep(0.8)   # let the connection open before starting mic

    def mic_callback(indata, frames, ti, status):
        """Called by sounddevice for every mic chunk — send immediately."""
        if ws.sock and ws.sock.connected:
            ws.send_bytes(_build_frame(indata[:, 0]))

    print(f"🎤  Mic chunk: {args.chunk} samples ({chunk_ms:.0f} ms)\n", flush=True)

    try:
        with sd.InputStream(
            samplerate = SAMPLE_RATE,
            channels   = CHANNELS,
            dtype      = DTYPE,
            blocksize  = args.chunk,
            callback   = mic_callback,
        ):
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping…", flush=True)
        ws.close()


if __name__ == "__main__":
    main()