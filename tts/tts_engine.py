"""
tts_engine.py — Core TTS engine with async chunk pipeline.

Improvements over v1:
  • Aggressive "first-chunk" splitter — sends audio after ~25 chars or first
    punctuation, dramatically cutting Time-to-First-Audio (TTFA).
  • Dynamic voice profiles via per-chunk metadata (emotion, speed overrides).
  • Server-side audio resampling (soxr → scipy fallback) — removes browser
    resampling latency.
  • torch.compile() support (Torch 2.x) — ~20-30 % faster inference after
    warm-up.
  • torch.backends.cudnn.benchmark = True for cuDNN kernel auto-tuning.
  • GPU warm-up call on init so the first real request isn't penalised.
"""

import re
import torch
import numpy as np
from scipy import signal as scipy_signal

# ── Optional fast resampler ───────────────────────────────────────────────────
try:
    import soxr
    _HAS_SOXR = True
except ImportError:
    _HAS_SOXR = False

from qwen_tts import Qwen3TTSModel

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID        = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEVICE          = "cuda:0"
DTYPE           = torch.float32
TARGET_SR       = 24_000   # sample rate sent to clients (change to 16000 if needed)

SPEAKERS = ["serena", "vivian", "ono_anna", "sohee", "aiden", "ryan"]

VOICE_PROFILES = {
    "serena":   {"speed": 0.97, "warmth": 0.25},
    "vivian":   {"speed": 0.96, "warmth": 0.25},
    "ono_anna": {"speed": 0.98, "warmth": 0.20},
    "sohee":    {"speed": 0.98, "warmth": 0.20},
    "aiden":    {"speed": 0.95, "warmth": 0.30},
    "ryan":     {"speed": 0.95, "warmth": 0.30},
}

# Emotion → speed/warmth overrides (used when Gateway sends metadata)
EMOTION_OVERRIDES = {
    "empathetic": {"speed": 0.88, "warmth": 0.45},
    "excited":    {"speed": 1.10, "warmth": 0.15},
    "calm":       {"speed": 0.90, "warmth": 0.35},
    "neutral":    {},
}

# Sentence-boundary pattern (used after the first chunk)
SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')

# Break-points for the aggressive first-chunk rule
_BREAK_RE = re.compile(r'([,\.!?:\n])')

# cuDNN auto-tuning — free ~5 % throughput on fixed-size inputs
torch.backends.cudnn.benchmark = True


# ── Audio helpers ─────────────────────────────────────────────────────────────
def _to_numpy(wavs) -> np.ndarray:
    if isinstance(wavs, torch.Tensor):
        return wavs.squeeze().cpu().float().numpy()
    if isinstance(wavs, (list, tuple)):
        first = wavs[0]
        if isinstance(first, torch.Tensor):
            return first.squeeze().cpu().float().numpy()
        return np.array(first, dtype=np.float32).squeeze()
    return np.array(wavs, dtype=np.float32).squeeze()


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample on the server; soxr is ~10× faster than scipy."""
    if src_sr == dst_sr:
        return audio
    if _HAS_SOXR:
        return soxr.resample(audio, src_sr, dst_sr, quality="HQ")
    # scipy fallback
    target_len = int(len(audio) * dst_sr / src_sr)
    return scipy_signal.resample(audio, target_len)


def _change_speed(audio: np.ndarray, speed: float, sr: int) -> np.ndarray:
    if abs(speed - 1.0) < 1e-3:
        return audio
    return scipy_signal.resample(audio, int(len(audio) / speed))


def _add_warmth(audio: np.ndarray, sr: int, warmth: float) -> np.ndarray:
    if warmth == 0:
        return audio
    b, a = scipy_signal.butter(2, 250 / (sr / 2), btype="low")
    warm = scipy_signal.filtfilt(b, a, audio)
    return np.clip(audio + warm * warmth, -1.0, 1.0)


def _cut_harshness(audio: np.ndarray, sr: int) -> np.ndarray:
    b, a = scipy_signal.butter(4, 7000 / (sr / 2), btype="low")
    return scipy_signal.filtfilt(b, a, audio)


def _normalize(audio: np.ndarray, peak: float = 0.88) -> np.ndarray:
    p = np.max(np.abs(audio))
    return audio / p * peak if p > 0 else audio


def _naturalize(audio: np.ndarray, sr: int, profile: dict) -> np.ndarray:
    audio = _change_speed(audio, profile["speed"], sr)
    audio = _cut_harshness(audio, sr)
    audio = _add_warmth(audio, sr, profile["warmth"])
    audio = _normalize(audio)
    return np.clip(audio, -1.0, 1.0)


def _to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] → int16 PCM bytes."""
    return (audio * 32767).astype(np.int16).tobytes()


def _build_profile(speaker: str, metadata: dict | None) -> dict:
    """
    Merge base voice profile with optional per-chunk metadata overrides.
    metadata example: {"emotion": "empathetic", "speed": 0.8}
    """
    profile = dict(VOICE_PROFILES.get(speaker, VOICE_PROFILES["aiden"]))

    if not metadata:
        return profile

    # Apply emotion preset first, then explicit speed/warmth keys
    emotion = metadata.get("emotion", "neutral")
    overrides = dict(EMOTION_OVERRIDES.get(emotion, {}))
    overrides.update({k: metadata[k] for k in ("speed", "warmth") if k in metadata})

    profile.update(overrides)
    return profile


# ── Engine ────────────────────────────────────────────────────────────────────
class TTSEngine:
    """
    Thread-safe TTS engine (one GPU model, many callers via run_in_executor).
    synthesize_chunk(text, speaker, language, metadata) → (pcm_bytes, sample_rate)
    """

    def __init__(self, compile_model: bool = False):
        print("[TTSEngine] Loading model …")
        self.model = Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            device_map=DEVICE,
            dtype=DTYPE,
            attn_implementation="eager",
        )
        if hasattr(self.model, "model") and hasattr(self.model.model, "generation_config"):
            self.model.model.generation_config.pad_token_id = 2150

        # Optional: torch.compile for ~20-30 % inference speedup (Torch 2.x only)
        # NOTE: first inference takes ~60 s to compile; subsequent calls are fast.
        if compile_model:
            try:
                self.model = torch.compile(self.model)
                print("[TTSEngine] torch.compile() applied.")
            except Exception as e:
                print(f"[TTSEngine] torch.compile() skipped: {e}")

        # Warm-up: run a silent inference so GPU caches are primed for the
        # first real request.
        print("[TTSEngine] Warming up GPU …")
        try:
            self.synthesize_chunk("Hello.", "aiden", "English")
            print("[TTSEngine] GPU warm-up complete.")
        except Exception as e:
            print(f"[TTSEngine] GPU warm-up failed (non-fatal): {e}")

        print("[TTSEngine] Ready.")

    def synthesize_chunk(
        self,
        text: str,
        speaker: str,
        language: str = "English",
        metadata: dict | None = None,
    ) -> tuple[bytes, int]:
        """
        Synthesize one sentence.

        Parameters
        ----------
        text     : sentence to synthesize
        speaker  : voice name (must be in SPEAKERS)
        language : language hint passed to model
        metadata : optional dict with keys "emotion", "speed", "warmth"

        Returns
        -------
        (pcm_bytes, sample_rate)  — pcm_bytes is int16 little-endian
        """
        text = text.strip()
        if not text:
            return b"", TARGET_SR

        profile = _build_profile(speaker, metadata)

        with torch.inference_mode():
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
            )

        audio = _to_numpy(wavs)
        audio = _naturalize(audio, sr, profile)

        # Resample to target SR on the server (GPU server >> JS client)
        audio = _resample(audio, sr, TARGET_SR)

        return _to_pcm16(audio), TARGET_SR


# ── Chunk splitter ────────────────────────────────────────────────────────────
class StreamingSentenceSplitter:
    """
    Buffers incoming LLM tokens and yields ready-to-synthesise segments.

    Strategy
    --------
    First chunk  — emit as soon as we have ≥ FIRST_CHUNK_CHARS characters
                   *and* hit any break-point (comma, period, colon …).
                   This gets audio to the user in ~150 ms instead of waiting
                   for a full sentence to complete.
    Subsequent   — revert to full sentence boundaries for natural prosody.
    """

    FIRST_CHUNK_CHARS = 25   # emit first audio after this many chars

    def __init__(self, min_chars: int = 20):
        self._buf           = ""
        self._min           = min_chars
        self._is_first      = True   # True until first chunk is emitted

    # ── public ────────────────────────────────────────────────────────────────

    def feed(self, token: str) -> list[str]:
        """Feed one token/chunk. Returns list of ready segments."""
        self._buf += token
        return self._flush()

    def finish(self) -> list[str]:
        """Signal end-of-stream. Returns any remaining buffered text."""
        tail = self._buf.strip()
        self._buf = ""
        self._is_first = True   # reset for potential session reuse
        return [tail] if tail else []

    # ── private ───────────────────────────────────────────────────────────────

    def _flush(self) -> list[str]:
        ready = []

        if self._is_first:
            ready = self._flush_first()
        
        # After first chunk (or if first chunk wasn't emitted yet), try normal splits
        if not self._is_first:
            ready += self._flush_sentences()

        return ready

    def _flush_first(self) -> list[str]:
        """
        Emit first segment as soon as buffer is long enough AND has a break-point.
        Falls back to splitting on whitespace if no punctuation found within
        FIRST_CHUNK_CHARS * 2 characters (avoids blocking on a very long word run).
        """
        if len(self._buf) < self.FIRST_CHUNK_CHARS:
            return []

        # Try split at first punctuation mark
        m = _BREAK_RE.search(self._buf)
        if m:
            split_at = m.end()
            chunk = self._buf[:split_at].strip()
            self._buf = self._buf[split_at:]
            self._is_first = False
            return [chunk] if chunk else []

        # No punctuation yet — fall back to word boundary after 2× threshold
        if len(self._buf) >= self.FIRST_CHUNK_CHARS * 2:
            last_space = self._buf.rfind(" ", 0, self.FIRST_CHUNK_CHARS * 2)
            if last_space > 0:
                chunk = self._buf[:last_space].strip()
                self._buf = self._buf[last_space:]
                self._is_first = False
                return [chunk] if chunk else []

        return []

    def _flush_sentences(self) -> list[str]:
        """Standard sentence-boundary splitting."""
        ready = []
        if len(self._buf) < self._min:
            return ready
        parts = SENTENCE_RE.split(self._buf)
        if len(parts) > 1:
            ready = [p.strip() for p in parts[:-1] if p.strip()]
            self._buf = parts[-1]
        return ready