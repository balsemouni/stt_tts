"""
benchmark.py — Measures raw GPU synthesis time per chunk size.
Run this directly to see your true hardware baseline.

python benchmark.py
"""
import time
import sys

print("Loading engine...", flush=True)
try:
    from tts_engine import TTSEngine
except Exception as e:
    print(f"[ERROR] Cannot import TTSEngine: {e}")
    sys.exit(1)

engine = TTSEngine()
speaker = "aiden"

tests = [
    "Hey!",
    "Just wanted to",
    "Hey! Just wanted to check in.",
    "How are you doing today?",
    "I've been thinking about our conversation.",
]

print("\n── Raw GPU synthesis benchmark ──────────────────────────")
print(f"{'Text':<45} {'Time':>8}  {'Audio':>8}  {'RTF':>6}")
print("─" * 75)

for text in tests:
    # Warmup run (first call includes CUDA kernel launch overhead)
    engine.synthesize_chunk(text, speaker)

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        pcm, sr = engine.synthesize_chunk(text, speaker)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        audio_ms = len(pcm) / 2 / sr * 1000

    avg = sum(times) / len(times)
    rtf = audio_ms / avg  # real-time factor: >1 = faster than real-time
    print(f"  {repr(text):<43} {avg:7.0f}ms  {audio_ms:7.0f}ms  {rtf:5.2f}x")

print("─" * 75)
print("\nIf synth time >> audio time, your GPU is the bottleneck.")
print("RTF < 1.0 means synthesis is SLOWER than real-time (bad).")
print("RTF > 2.0 means you have headroom for streaming.\n")
