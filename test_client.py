#!/usr/bin/env python3
"""
test_gateway_client.py — Interactive test client for the Voice Gateway (gateway.py)
═══════════════════════════════════════════════════════════════════════════════════════
Connects to the Gateway WebSocket at ws://localhost:8090/ws

What it does
────────────
  • Opens mic → streams raw int16 PCM chunks to the gateway (0x01 frames)
  • Displays live STT words appearing word-by-word on one scrolling line
  • Shows [partial] token in brackets while Whisper is mid-word
  • Prints full transcript sentence when a segment is finalised
  • Displays AI tokens as they stream in (live typewriter effect)
  • Shows each AI sentence when it's dispatched to TTS ("▶ TTS: …")
  • Plays received PCM audio through local speakers in real time
  • Indicates mic mute/unmute events with clear status icons
  • Shows barge-in, hallucination-reset, idle-reset, and error events
  • Sends periodic pong responses to gateway heartbeat pings
  • Ctrl+C for clean shutdown

Install:
  pip install sounddevice websocket-client numpy

Run:
  python test_gateway_client.py [--url ws://localhost:8090/ws] [--chunk 320]
"""

import argparse
import json
import queue
import struct
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import websocket

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE    = 16000
CHANNELS       = 1
DTYPE          = "float32"
DEFAULT_CHUNK  = 320   # 20 ms at 16kHz

# ── State ─────────────────────────────────────────────────────────────────────

_lock              = threading.Lock()
_words_on_line     : list[str] = []   # confirmed STT words this utterance
_partial_active    = False
_mic_muted         = False
_ai_token_buf      = ""               # accumulating AI tokens for typewriter display
_last_state        = "IDLE"
_session_stats     = {
    "queries":         0,
    "barge_ins":       0,
    "hallucinations":  0,
    "tts_sentences":   0,
    "session_resets":  0,
}

# Audio playback queue — receives raw PCM bytes from gateway
_audio_q: queue.Queue = queue.Queue()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_frame(pcm: np.ndarray) -> bytes:
    """Pack a float32 mic frame into a 0x01 mux frame (int16 PCM body)."""
    int16 = (pcm * 32768).clip(-32768, 32767).astype("int16")
    return bytes([0x01]) + int16.tobytes()


def _clear_line():
    sys.stdout.write("\r" + " " * 100 + "\r")
    sys.stdout.flush()


def _redraw_stt(partial: str = ""):
    """Redraw the live STT word line with optional [partial] suffix."""
    display = _words_on_line[-20:]
    line    = "🎙  " + " ".join(display)
    if partial:
        line += f" [{partial}]"
    sys.stdout.write("\r" + line + "   ")
    sys.stdout.flush()


def _mic_icon() -> str:
    return "🔇 MIC MUTED" if _mic_muted else "🎤 MIC OPEN "


def _print_divider():
    print("─" * 60, flush=True)


def _print_stats():
    print("\n📊 SESSION STATS", flush=True)
    _print_divider()
    for k, v in _session_stats.items():
        print(f"   {k:<20} {v}", flush=True)
    _print_divider()


# ── Audio playback thread ─────────────────────────────────────────────────────

def _audio_player():
    """
    Drains _audio_q and plays PCM chunks through the default output device.
    The gateway sends raw 16-bit signed PCM at 16kHz mono.
    """
    with sd.RawOutputStream(
        samplerate = SAMPLE_RATE,
        channels   = CHANNELS,
        dtype      = "int16",
    ) as stream:
        while True:
            try:
                chunk = _audio_q.get(timeout=0.1)
                if chunk is None:
                    break
                stream.write(chunk)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n⚠️  Audio playback error: {e}", flush=True)


# ── WebSocket handlers ────────────────────────────────────────────────────────

def on_message(ws_app, raw):
    global _mic_muted, _partial_active, _ai_token_buf, _last_state

    # ── Binary frame: PCM audio from TTS ──────────────────────────────────
    if isinstance(raw, bytes):
        _audio_q.put_nowait(raw)
        return

    # ── JSON control frame ─────────────────────────────────────────────────
    try:
        ev = json.loads(raw)
        t  = ev.get("type")
    except Exception:
        return

    with _lock:

        # ── Live STT word ──────────────────────────────────────────────────
        if t == "word":
            word = ev.get("word", "").strip()
            if word:
                _words_on_line.append(word)
                _partial_active = False
                _redraw_stt()

        # ── STT partial ────────────────────────────────────────────────────
        elif t == "partial":
            partial = ev.get("word", "").strip()
            if partial:
                _partial_active = True
                _redraw_stt(partial=partial)

        # ── STT segment finalised ──────────────────────────────────────────
        elif t == "segment":
            text = ev.get("text", "").strip()
            _clear_line()
            _words_on_line[:] = []
            _partial_active   = False
            _ai_token_buf     = ""
            _session_stats["queries"] += 1
            print(f"\n📝 USER:  {text}", flush=True)

        # ── Gateway state: model thinking ──────────────────────────────────
        elif t == "thinking":
            turn_id = ev.get("turn_id", "")[:8]
            _ai_token_buf = ""
            _clear_line()
            sys.stdout.write(f"\n🧠 AI [{turn_id}]: ")
            sys.stdout.flush()
            _last_state = "THINKING"

        # ── Streaming AI token (typewriter) ────────────────────────────────
        elif t == "ai_token":
            token = ev.get("token", "")
            if token:
                _ai_token_buf += token
                sys.stdout.write(token)
                sys.stdout.flush()

        # ── Sentence dispatched to TTS ─────────────────────────────────────
        elif t == "ai_sentence":
            text = ev.get("text", "")
            _session_stats["tts_sentences"] += 1
            # Print on new line so TTS sentences are clearly visible
            sys.stdout.write(f"\n   ▶ TTS: {text}")
            sys.stdout.flush()

        # ── TTS turn complete ──────────────────────────────────────────────
        elif t == "done":
            chunks = ev.get("chunks", 0)
            print(f"\n✅ TTS done ({chunks} chunks)  {_mic_icon()}", flush=True)
            _last_state = "IDLE"

        # ── Mic mute ───────────────────────────────────────────────────────
        elif t == "mute_mic":
            _mic_muted = True
            print(f"\n🔇 [MIC MUTED — AI speaking]", flush=True)

        # ── Mic unmute ─────────────────────────────────────────────────────
        elif t == "unmute_mic":
            _mic_muted = False
            print(f"🎤 [MIC OPEN — listening]\n", flush=True)
            _last_state = "IDLE"

        # ── Barge-in ───────────────────────────────────────────────────────
        elif t == "segment":   # duplicate guard — handled above
            pass

        # ── Hallucination guard fired ──────────────────────────────────────
        elif t == "hallucination_reset":
            detail = ev.get("detail", "")
            _session_stats["hallucinations"] += 1
            _clear_line()
            _words_on_line[:] = []
            print(f"\n🚨 HALLUCINATION RESET: {detail}", flush=True)

        # ── Idle auto-reset ────────────────────────────────────────────────
        elif t == "session_reset":
            reason = ev.get("reason", "")
            _session_stats["session_resets"] += 1
            print(f"\n⏱  SESSION RESET ({reason})", flush=True)

        # ── Error ──────────────────────────────────────────────────────────
        elif t == "error":
            detail = ev.get("detail", "unknown")
            _clear_line()
            print(f"\n❌ ERROR: {detail}", flush=True)
            _last_state = "IDLE"

        # ── Gateway ping → reply pong ──────────────────────────────────────
        elif t == "ping":
            try:
                ws_app.send(json.dumps({"type": "pong"}))
            except Exception:
                pass

        # ── Stats response ─────────────────────────────────────────────────
        elif t == "stats":
            print(
                f"\n📡 GATEWAY STATS  sid={ev.get('sid','')}  "
                f"state={ev.get('state','?')}",
                flush=True,
            )


def on_open(ws_app):
    print(
        "\n╔══════════════════════════════════════════════════╗\n"
        "║    Voice Gateway Test Client  (gateway v6)       ║\n"
        "╠══════════════════════════════════════════════════╣\n"
        "║  Speak → words appear live → AI replies via TTS  ║\n"
        "║  Ctrl+C to quit  •  's' + Enter for stats        ║\n"
        "╚══════════════════════════════════════════════════╝\n",
        flush=True,
    )


def on_error(ws_app, err):
    print(f"\n❌  WebSocket error: {err}", flush=True)


def on_close(ws_app, *_):
    print("\n🔌  Disconnected from gateway.", flush=True)
    _audio_q.put(None)   # signal audio thread to stop


# ── Keyboard command thread ───────────────────────────────────────────────────

def _keyboard_loop(ws_app):
    """
    Allows typing simple commands while mic is streaming:
      s  → print session stats
      r  → send reset_context control message
      q  → quit
    """
    while True:
        try:
            cmd = input().strip().lower()
        except EOFError:
            break
        if cmd == "s":
            _print_stats()
        elif cmd == "r":
            try:
                ctrl = json.dumps({"type": "reset_context"})
                ws_app.send(b'\x02' + ctrl.encode())
                print("↺  Reset context sent to STT", flush=True)
            except Exception as e:
                print(f"Reset failed: {e}", flush=True)
        elif cmd == "g":
            try:
                ws_app.send(json.dumps({"type": "get_stats"}))
            except Exception:
                pass
        elif cmd in ("q", "quit", "exit"):
            print("Quitting…", flush=True)
            ws_app.close()
            break
        else:
            print("Commands: s=stats  r=reset  g=gateway-stats  q=quit", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Voice Gateway test client")
    ap.add_argument(
        "--url",
        default="ws://localhost:8090/ws",
        help="Gateway WebSocket URL (default: ws://localhost:8090/ws)",
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

    # Start WebSocket thread
    ws_thread = threading.Thread(
        target=lambda: ws.run_forever(ping_interval=30, ping_timeout=10),
        daemon=True,
        name="WSThread",
    )
    ws_thread.start()
    time.sleep(0.8)   # wait for connection before starting mic

    # Start audio playback thread
    audio_thread = threading.Thread(
        target=_audio_player,
        daemon=True,
        name="AudioThread",
    )
    audio_thread.start()

    # Start keyboard command thread
    kb_thread = threading.Thread(
        target=_keyboard_loop,
        args=(ws,),
        daemon=True,
        name="KeyboardThread",
    )
    kb_thread.start()

    # Mic → gateway
    def mic_callback(indata, frames, ti, status):
        if _mic_muted:
            return   # client-side gate: don't send audio while AI is speaking
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
            while ws_thread.is_alive():
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping…", flush=True)
        ws.close()
        _audio_q.put(None)
        _print_stats()


if __name__ == "__main__":
    main()