"""
TTS Client Test — with actual audio playback
=============================================
Sends text to the TTS service and PLAYS the audio as chunks arrive.

Usage:
    python test_client.py
    python test_client.py --text "Your custom text here."
    python test_client.py --host 192.168.1.10 --port 8765

Requires:
    pip install requests sounddevice numpy
"""

import argparse
import base64
import time
import wave
from io import BytesIO

import numpy as np
import requests
import sounddevice as sd

# ── Config ────────────────────────────────────────────────────────
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8765
DEFAULT_TEXT = (
    "Hello! How are you today? "
    "This is a test of the streaming TTS microservice. "
    "It splits text into chunks and synthesises them in order."
)
SAMPLE_RATE = 24000


# ── Helpers ───────────────────────────────────────────────────────

def decode_wav_to_numpy(audio_b64: str) -> np.ndarray:
    raw = base64.b64decode(audio_b64)
    with wave.open(BytesIO(raw), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        audio  = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    return audio


def play_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    sd.play(audio, samplerate=sample_rate)
    sd.wait()  # block until this chunk finishes before playing the next


def bar(value: float, max_val: float, width: int = 30) -> str:
    filled = int((value / max_val) * width) if max_val else 0
    return "█" * filled + "░" * (width - filled)


# ── Main ──────────────────────────────────────────────────────────

def run(base_url: str, text: str) -> None:

    # 1. Health check
    print(f"\n{'='*60}")
    print(f"  Server : {base_url}")
    print(f"  Text   : {text[:70]}{'…' if len(text) > 70 else ''}")
    print(f"{'='*60}")

    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        r.raise_for_status()
        d = r.json()
        print(f"\n  ✓ Server OK  |  CUDA: {d.get('cuda')}  |  Model: {d.get('model', 'n/a')}\n")
    except Exception as e:
        print(f"\n  ✗ Server not reachable: {e}")
        print("  Start with: python tts_microservice.py\n")
        return

    # 2. Preview split
    prev = requests.post(f"{base_url}/tts/preview-chunks", json={"text": text}, timeout=10)
    prev.raise_for_status()
    preview     = prev.json()
    chunk_texts = [c["text"] for c in preview["chunks"]]
    chunk_types = [c["type"] for c in preview["chunks"]]

    print(f"  Text split into {len(chunk_texts)} chunks:")
    for c in preview["chunks"]:
        print(f"    [{c['index']}] {c['type']:<6}  \"{c['text']}\"")

    # 3. Synthesize all chunks, then play each one
    print(f"\n  Synthesizing …\n")
    t0 = time.time()

    r = requests.post(
        f"{base_url}/tts/chunks",
        json={"chunks": chunk_texts, "chunk_types": chunk_types},
        timeout=120,
    )
    r.raise_for_status()

    chunks        = r.json()["chunks"]
    synth_done_ms = (time.time() - t0) * 1000
    max_lat       = max(c["latency"]["synthesis_latency_ms"] for c in chunks) or 1

    print(f"  All chunks synthesized in {synth_done_ms:.0f} ms\n")

    for c in chunks:
        lat   = c["latency"]
        audio = decode_wav_to_numpy(c["audio_b64"])

        print(f"  ▶ Chunk {c['chunk_index']}  [{c['chunk_type']:<6}]  \"{c['text']}\"")
        print(f"     duration    : {c['duration_sec']:.2f}s")
        print(f"     synth time  : {lat['synth_duration_ms']:.0f}ms")
        print(f"     from job    : {lat['synthesis_latency_ms']:.0f}ms  [{bar(lat['synthesis_latency_ms'], max_lat)}]")
        if c["chunk_index"] == 0:
            print(f"     ★ first chunk latency : {lat['synthesis_latency_ms']:.0f}ms")
        else:
            print(f"     from 1st chunk : {lat['first_chunk_latency_ms']:.0f}ms")

        for line in c.get("display_waveform", "").split("\n"):
            print(f"     {line}")

        print(f"     → playing now …")
        play_audio(audio)
        print(f"     ✓ done\n")

    total_ms = (time.time() - t0) * 1000
    print(f"{'='*60}")
    print(f"  Chunks played   : {len(chunks)}")
    print(f"  Total time      : {total_ms:.0f} ms  (synth + playback)")
    print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT, type=int)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    args = parser.parse_args()

    run(f"http://{args.host}:{args.port}", args.text)