"""
tts_engine.py — Core TTS engine.

v7 — LATENCY-OPTIMIZED + GPU-2 TARGET:
  • Always targets GPU_INDEX (default: 2) via CUDA_VISIBLE_DEVICES remapping.
  • torch.compile() enabled by default — 20–35 % steady-state speedup.
  • language is ALWAYS explicitly "English" — model is sensitive to this;
    a mismatch causes slow/degraded inference.
  • _MIN_NEW_TOKENS raised to 20 to avoid underrun artifacts.
  • Warmup uses _fast_path=True for intermediate phrases (only final is full).
  • Delta-synthesis path unchanged: full-context synthesis + client-side delta.

  StreamingSentenceSplitter retained for legacy LLM-streaming mode.
"""

import re
import gc
import math
import os
import torch
import numpy as np
from scipy import signal as scipy_signal
from huggingface_hub import snapshot_download

from qwen_tts import Qwen3TTSModel

# ── GPU config ────────────────────────────────────────────────────────────────
# Set CUDA_VISIBLE_DEVICES BEFORE any torch call so the physical GPU_INDEX
# is exposed as logical cuda:0 inside this process.
GPU_INDEX = 2
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(GPU_INDEX))

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_ID  = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEVICE    = "cuda:0"          # logical 0 = physical GPU_INDEX after remapping
DTYPE     = torch.float16
TARGET_SR = 24_000
LANGUAGE  = "English"         # ← single source-of-truth; never pass None / "en"

SPEAKERS = ["serena", "vivian", "ono_anna", "sohee", "aiden", "ryan"]

VOICE_PROFILES = {
    "serena":   {"speed": 0.97, "warmth": 0.25},
    "vivian":   {"speed": 0.96, "warmth": 0.25},
    "ono_anna": {"speed": 0.98, "warmth": 0.20},
    "sohee":    {"speed": 0.98, "warmth": 0.20},
    "aiden":    {"speed": 0.95, "warmth": 0.30},
    "ryan":     {"speed": 0.95, "warmth": 0.30},
}

EMOTION_OVERRIDES = {
    "empathetic": {"speed": 0.88, "warmth": 0.45},
    "excited":    {"speed": 1.10, "warmth": 0.15},
    "calm":       {"speed": 0.90, "warmth": 0.35},
    "neutral":    {},
}

# ── Splitter config (legacy / LLM-streaming mode only) ───────────────────────
_FIRST_CHUNK_CAP = 4    # fires after ~1 token — minimises TTFA
_HARD_CAP_CHARS  = 18   # shorter phrases = faster GPU round-trip
_WORD_CAP        = 2    # split every 2 words for rapid dispatch
_MIN_CHUNK_CHARS = 2    # allow single short words through

# ── Token budget ──────────────────────────────────────────────────────────────
_TOKENS_PER_CHAR = 2.5
_MAX_NEW_TOKENS  = 200
_MIN_NEW_TOKENS  = 10   # lowered from 20 — short phrases don't need 20 tokens

_BREAK_RE = re.compile(r'([,\.!?;:\n])')

torch.backends.cudnn.benchmark = True

# Pre-compute filter coefficients once at import time
_SR_FOR_COEFFS = TARGET_SR
_WARMTH_B, _WARMTH_A = scipy_signal.butter(2, 250  / (_SR_FOR_COEFFS / 2), btype="low")
_HARSH_B,  _HARSH_A  = scipy_signal.butter(4, 7000 / (_SR_FOR_COEFFS / 2), btype="low")


# ── HF model path resolver ────────────────────────────────────────────────────
def _resolve_model_path(model_id: str) -> str:
    print(f"[TTSEngine] Resolving model from HF Hub: {model_id}")
    local_path = snapshot_download(
        repo_id=model_id,
        repo_type="model",
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
    )
    print(f"[TTSEngine] Model cached at: {local_path}")
    return local_path


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
    if src_sr == dst_sr:
        return audio
    target_len = int(len(audio) * dst_sr / src_sr)
    return scipy_signal.resample(audio, target_len)


def _change_speed(audio: np.ndarray, speed: float, sr: int) -> np.ndarray:
    if abs(speed - 1.0) < 1e-3:
        return audio
    return scipy_signal.resample(audio, int(len(audio) / speed))


def _add_warmth_fast(audio: np.ndarray, warmth: float) -> np.ndarray:
    if warmth == 0:
        return audio
    warm = scipy_signal.lfilter(_WARMTH_B, _WARMTH_A, audio)
    return np.clip(audio + warm * warmth, -1.0, 1.0)


def _cut_harshness_fast(audio: np.ndarray) -> np.ndarray:
    return scipy_signal.lfilter(_HARSH_B, _HARSH_A, audio)


def _normalize(audio: np.ndarray, peak: float = 0.88) -> np.ndarray:
    p = np.max(np.abs(audio))
    return audio / p * peak if p > 0 else audio


def _trim_silence(
    audio: np.ndarray,
    threshold: float = 0.008,
    pad_ms: int = 8,
    sr: int = TARGET_SR,
    trim_head: bool = True,
    trim_tail: bool = True,
) -> np.ndarray:
    mask = np.abs(audio) > threshold
    if not np.any(mask):
        return audio
    pad   = int(pad_ms * sr / 1000)
    start = max(0, int(np.argmax(mask)) - pad) if trim_head else 0
    end   = (
        min(len(audio), len(audio) - int(np.argmax(mask[::-1])) + pad)
        if trim_tail else len(audio)
    )
    return audio[start:end]


_XFADE_SAMPLES = int(0.008 * TARGET_SR)


def _apply_tail_fade(audio: np.ndarray, n: int = _XFADE_SAMPLES) -> np.ndarray:
    if len(audio) <= n:
        return audio
    fade       = audio.copy()
    ramp       = np.linspace(1.0, 0.0, n, dtype=np.float32)
    fade[-n:] *= ramp
    return fade


def _naturalize(
    audio: np.ndarray,
    sr: int,
    profile: dict,
    trim_tail: bool = True,
) -> np.ndarray:
    audio = _change_speed(audio, profile["speed"], sr)
    audio = _cut_harshness_fast(audio)
    audio = _add_warmth_fast(audio, profile["warmth"])
    audio = _normalize(audio)
    audio = _trim_silence(audio, sr=sr, trim_head=True, trim_tail=trim_tail)
    if not trim_tail:
        audio = _apply_tail_fade(audio)
    return np.clip(audio, -1.0, 1.0)


def _to_pcm16(audio: np.ndarray) -> bytes:
    return (audio * 32767).astype(np.int16).tobytes()


def _fast_normalize_only(audio: np.ndarray) -> bytes:
    p = np.max(np.abs(audio))
    if p > 0:
        audio = audio * (0.88 / p)
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _build_profile(speaker: str, metadata: dict | None) -> dict:
    profile = dict(VOICE_PROFILES.get(speaker, VOICE_PROFILES["aiden"]))
    if not metadata:
        return profile
    emotion  = metadata.get("emotion", "neutral")
    override = dict(EMOTION_OVERRIDES.get(emotion, {}))
    override.update({k: metadata[k] for k in ("speed", "warmth") if k in metadata})
    profile.update(override)
    return profile


# ── Delta extractor ───────────────────────────────────────────────────────────
_DELTA_TOLERANCE_MS   = 40
_DELTA_TOLERANCE_SAMP = int(_DELTA_TOLERANCE_MS * TARGET_SR / 1000)
_MIN_DELTA_SAMP       = int(4 * TARGET_SR / 1000)


def extract_delta(
    prev_audio: np.ndarray | None,
    full_audio: np.ndarray,
    align: str = "xcorr",
) -> np.ndarray:
    """
    Return only the new audio portion compared to the previous synthesis.

    align="xcorr"  — cross-correlation alignment (handles slight timing drift)
    align="length" — simple length subtraction (faster, less robust)
    """
    if prev_audio is None or len(prev_audio) == 0:
        return full_audio

    expected = len(prev_audio)

    if align == "length" or expected >= len(full_audio):
        delta = full_audio[expected:]
        return delta if len(delta) >= _MIN_DELTA_SAMP else np.array([], dtype=np.float32)

    lo      = max(0, expected - _DELTA_TOLERANCE_SAMP)
    hi      = min(len(full_audio) - _MIN_DELTA_SAMP, expected + _DELTA_TOLERANCE_SAMP)
    ref_len = min(int(0.020 * TARGET_SR), len(prev_audio))
    ref     = prev_audio[-ref_len:].astype(np.float32)
    ref     = ref - ref.mean()
    ref_n   = np.linalg.norm(ref)

    if ref_n < 1e-6:
        delta = full_audio[expected:]
        return delta if len(delta) >= _MIN_DELTA_SAMP else np.array([], dtype=np.float32)

    best_pos, best_score = expected, -np.inf
    step = max(1, ref_len // 4)
    for pos in range(lo, min(hi, len(full_audio) - ref_len), step):
        cand  = full_audio[pos : pos + ref_len].astype(np.float32)
        cand  = cand - cand.mean()
        cn    = np.linalg.norm(cand)
        if cn < 1e-6:
            continue
        score = float(np.dot(ref, cand) / (ref_n * cn))
        if score > best_score:
            best_score = score
            best_pos   = pos + ref_len

    delta = full_audio[best_pos:]
    return delta if len(delta) >= _MIN_DELTA_SAMP else np.array([], dtype=np.float32)


# ── Engine ────────────────────────────────────────────────────────────────────
class TTSEngine:
    def __init__(self, compile_model: bool = True):
        """
        Load the TTS model onto DEVICE (physical GPU_INDEX via remapping).

        compile_model=True  — torch.compile() for 20–35 % faster steady-state
                              inference (first call is slower while tracing).
        """
        print(f"[TTSEngine] Loading model on physical GPU {GPU_INDEX} "
              f"(logical {DEVICE}) …")

        model_path = _resolve_model_path(MODEL_ID)

        self.model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=DEVICE,
            dtype=DTYPE,
            attn_implementation="sdpa",
        )

        # Pin all sub-modules to DEVICE and set eval mode
        _moved = 0
        for attr_name in dir(self.model):
            if attr_name.startswith("_"):
                continue
            try:
                attr = getattr(self.model, attr_name)
                if isinstance(attr, torch.nn.Module):
                    attr.to(DEVICE).eval()
                    _moved += 1
            except Exception:
                pass
        torch.cuda.synchronize()
        print(f"[TTSEngine] {_moved} sub-module(s) pinned to {DEVICE} → eval().")

        if hasattr(self.model, "model") and hasattr(
            self.model.model, "generation_config"
        ):
            self.model.model.generation_config.pad_token_id = 2150

        # ── torch.compile — enabled by default for latency ────────────────────
        if compile_model:
            try:
                self.model = torch.compile(self.model)
                print("[TTSEngine] ✅ torch.compile() applied — "
                      "first inference will be slow (tracing), then fast.")
            except Exception as e:
                print(f"[TTSEngine] ⚠️  torch.compile() skipped: {e}")

        self._cuda_stream = (
            torch.cuda.Stream(device=DEVICE) if torch.cuda.is_available() else None
        )

        # ── Warmup ────────────────────────────────────────────────────────────
        # Intermediate warmup phrases use _fast_path=True (skip heavy filters)
        # so the GPU heats up quickly without wasting time on post-processing.
        # Only the final phrase uses the full pipeline to prime all code paths.
        print("[TTSEngine] 🔥 Warming up GPU …")
        _warmup_phrases = [
            ("Hey!", True),
            ("Hey just wanted", True),
            ("Hey just wanted to check in and see", True),
            ("Hey just wanted to check in and see how you're doing today.", False),
        ]
        try:
            for phrase, fast in _warmup_phrases:
                raw, sr = self.synthesize_raw(phrase, "aiden")
                self.postprocess(
                    raw, sr, "aiden",
                    metadata=None,
                    _fast_path=fast,
                    is_last=not fast,
                )
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            print("[TTSEngine] ✅ GPU is HOT.")
        except Exception as e:
            print(f"[TTSEngine] ⚠️  Warm-up failed (non-fatal): {e}")

        print("[TTSEngine] Ready.")

    # ── Core synthesis ────────────────────────────────────────────────────────
    def synthesize_raw(
        self,
        text: str,
        speaker: str,
        language: str = LANGUAGE,   # ← always defaults to the constant "English"
    ) -> tuple[np.ndarray, int]:
        """
        Run the model and return raw float32 audio + sample-rate.
        The `language` parameter defaults to the module-level LANGUAGE constant
        so callers never need to remember to pass it.
        """
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32), TARGET_SR

        dyn_tokens = max(
            _MIN_NEW_TOKENS,
            min(int(math.ceil(len(text) * _TOKENS_PER_CHAR)), _MAX_NEW_TOKENS),
        )

        with torch.inference_mode():
            if self._cuda_stream is not None:
                with torch.cuda.stream(self._cuda_stream):
                    wavs, sr = self.model.generate_custom_voice(
                        text=text,
                        language=language,
                        speaker=speaker,
                        max_new_tokens=dyn_tokens,
                    )
                self._cuda_stream.synchronize()
            else:
                wavs, sr = self.model.generate_custom_voice(
                    text=text,
                    language=language,
                    speaker=speaker,
                    max_new_tokens=dyn_tokens,
                )

        return _to_numpy(wavs), sr

    # ── Post-processing ───────────────────────────────────────────────────────
    def postprocess(
        self,
        audio: np.ndarray,
        sr: int,
        speaker: str,
        metadata: dict | None = None,
        _fast_path: bool = False,
        is_last: bool = False,
    ) -> tuple[bytes, int]:
        if audio.size == 0:
            return b"", TARGET_SR

        trim_tail = is_last

        if _fast_path:
            audio = _trim_silence(audio, sr=sr, trim_head=True, trim_tail=trim_tail)
            if not trim_tail:
                audio = _apply_tail_fade(audio)
            pcm = _fast_normalize_only(audio)
            if sr != TARGET_SR:
                arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
                arr = _resample(arr, sr, TARGET_SR)
                pcm = _to_pcm16(arr)
            return pcm, TARGET_SR

        profile = _build_profile(speaker, metadata)
        audio   = _naturalize(audio, sr, profile, trim_tail=trim_tail)
        audio   = _resample(audio, sr, TARGET_SR)
        return _to_pcm16(audio), TARGET_SR

    def synthesize_chunk(
        self,
        text: str,
        speaker: str,
        language: str = LANGUAGE,
        metadata: dict | None = None,
    ) -> tuple[bytes, int]:
        audio, sr = self.synthesize_raw(text, speaker, language)
        return self.postprocess(audio, sr, speaker, metadata)


# ── Streaming splitter (legacy / LLM-streaming mode) ─────────────────────────
class StreamingSentenceSplitter:
    """
    Used only in LLM-streaming mode (main.py).
    Delta-synthesis clients bypass this entirely via the 'no_split' session flag.
    """

    def __init__(self) -> None:
        self._buf        = ""
        self._first_sent = False

    def _word_count(self, s: str) -> int:
        return len(s.split())

    def feed(self, token: str) -> list[str]:
        self._buf += token
        out: list[str] = []

        while True:
            m = _BREAK_RE.search(self._buf)
            if m:
                end       = m.end(1)
                chunk     = self._buf[:end].strip()
                self._buf = self._buf[end:].lstrip()
                if len(chunk) >= _MIN_CHUNK_CHARS:
                    out.append(chunk)
                    self._first_sent = True
                continue

            if not self._first_sent and len(self._buf) >= _FIRST_CHUNK_CAP:
                split = self._buf[:_FIRST_CHUNK_CAP].rfind(" ")
                if split > 2:
                    chunk     = self._buf[:split].strip()
                    self._buf = self._buf[split:].lstrip()
                    if len(chunk) >= _MIN_CHUNK_CHARS:
                        out.append(chunk)
                        self._first_sent = True
                    continue

            if self._word_count(self._buf) >= _WORD_CAP:
                words     = self._buf.split()
                chunk     = " ".join(words[:_WORD_CAP])
                self._buf = self._buf[len(chunk):].lstrip()
                if len(chunk) >= _MIN_CHUNK_CHARS:
                    out.append(chunk)
                    self._first_sent = True
                continue

            if len(self._buf) >= _HARD_CAP_CHARS:
                split = self._buf[:_HARD_CAP_CHARS].rfind(" ")
                if split > 2:
                    chunk     = self._buf[:split].strip()
                    self._buf = self._buf[split:].lstrip()
                else:
                    chunk     = self._buf[:_HARD_CAP_CHARS].strip()
                    self._buf = self._buf[_HARD_CAP_CHARS:]
                if len(chunk) >= _MIN_CHUNK_CHARS:
                    out.append(chunk)
                    self._first_sent = True
                continue

            break

        return out

    def finish(self) -> list[str]:
        tail      = self._buf.strip()
        self._buf = ""
        if tail:
            self._first_sent = True
        return [tail] if tail else []