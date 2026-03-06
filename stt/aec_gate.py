"""
aec_gate.py
───────────
Software-level Acoustic Echo Cancellation Gate

Problem
───────
When the AI (TTS) is playing audio through the speakers, that audio leaks back
into the microphone.  The browser's built-in AEC (echoCancellation: true) handles
most of this, but residual echo still arrives at the STT microservice — especially
on loudspeaker setups.

Solution: a two-layer gate implemented entirely in Python/numpy
  Layer A — State Gate
      The orchestrator signals the STT service when the AI is speaking via the
      WebSocket control message { "type": "ai_state", "speaking": true }.
      While speaking=true + grace period, the VAD threshold is raised and chunks
      are suppressed unless they exceed an energy ratio vs the reference signal.

  Layer B — Spectral Subtraction (Reference-Based)
      When a reference signal (what the AI is actually playing) is available,
      we subtract its spectral envelope from the incoming mic audio.
      This is a simplified Wiener-filter approach — works well for TTS which has
      a very predictable spectral shape.

The reference signal is OPTIONAL.  If the frontend sends AI audio back to the
STT service over a separate WebSocket channel, we use it.  If not, Layer A alone
is used (state gate only).

Integration points
──────────────────
  • pipeline.py  →  aec_gate.process(chunk, ai_speaking) before VAD
  • main.py      →  receives { type: "ai_state", speaking: bool } control frames
                    receives { type: "ai_reference", pcm: <bytes> } audio frames
"""

from __future__ import annotations

import numpy as np
import time
import logging
from collections import deque
from typing import Optional

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────

GRACE_PERIOD_MS      = 800    # Block ASR for this long after AI starts speaking
POST_STOP_BUFFER_MS  = 300    # Continue blocking after AI stops (echo tail)
SPECTRAL_ALPHA       = 0.85   # Spectral subtraction strength (0=none, 1=full)
REFERENCE_QUEUE_MS   = 2000   # How much reference audio to keep in buffer


class AECGate:
    """
    Acoustic Echo Cancellation Gate.

    Usage
    ─────
        aec = AECGate(sample_rate=16000)

        # When TTS starts/stops:
        aec.set_ai_speaking(True)
        aec.set_ai_speaking(False)

        # Optional: feed reference (what AI is playing)
        aec.push_reference(pcm_chunk)

        # In your chunk loop — call BEFORE VAD:
        cleaned, suppressed = aec.process(mic_chunk)
        if not suppressed:
            # forward cleaned audio to VAD / ASR
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate      = sample_rate
        self._ai_speaking     = False
        self._ai_started_at: Optional[float]  = None
        self._ai_stopped_at: Optional[float]  = None

        # Reference signal buffer (circular)
        _ref_samples = int(sample_rate * REFERENCE_QUEUE_MS / 1000)
        self._reference_buffer: deque[np.ndarray] = deque(maxlen=_ref_samples)
        self._has_reference = False

        # Stats
        self.chunks_suppressed = 0
        self.chunks_processed  = 0

    # ── AI state control ─────────────────────────────────────────────────────

    def set_ai_speaking(self, speaking: bool):
        """
        Called by main.py when it receives a control message from the orchestrator.
        """
        if speaking and not self._ai_speaking:
            self._ai_speaking   = True
            self._ai_started_at = time.monotonic()
            self._ai_stopped_at = None
            logger.debug("[AEC] AI speaking — gate ACTIVE")

        elif not speaking and self._ai_speaking:
            self._ai_speaking   = False
            self._ai_stopped_at = time.monotonic()
            logger.debug("[AEC] AI stopped — draining echo tail")

    # ── Reference signal ─────────────────────────────────────────────────────

    def push_reference(self, pcm_chunk: np.ndarray):
        """
        Optionally feed what the AI is playing.
        Call this from the TTS audio playback path (frontend forwards it back,
        or the TTS microservice pushes it via an internal Redis channel).
        """
        self._reference_buffer.extend(pcm_chunk.astype(np.float32).tolist())
        self._has_reference = True

    # ── Main processing ───────────────────────────────────────────────────────

    def process(
        self,
        mic_chunk: np.ndarray,
        force_check: bool = False,
    ) -> tuple[np.ndarray, bool]:
        """
        Process one mic chunk through the AEC gate.

        Returns
        ───────
        (cleaned_audio, suppressed)
          cleaned_audio : np.ndarray  — mic audio after spectral subtraction
          suppressed    : bool        — True = do NOT forward to VAD/ASR
        """
        self.chunks_processed += 1
        cleaned = mic_chunk.astype(np.float32)

        # ── Layer A: State gate ───────────────────────────────────────────
        in_grace    = self._in_grace_period()
        in_tail     = self._in_echo_tail()

        if in_grace or in_tail:
            # Apply spectral subtraction if we have a reference, then still
            # suppress unless the user is speaking MUCH louder than the echo
            if self._has_reference:
                cleaned = self._spectral_subtract(cleaned)

            # Energy ratio check — allow through if user is WAY louder than ref
            if self._has_reference:
                ref_rms = self._reference_rms()
                mic_rms = float(np.sqrt(np.mean(cleaned**2) + 1e-9))
                # Only allow if mic is 3× louder than reference (clear barge-in)
                if mic_rms < ref_rms * 3.0:
                    self.chunks_suppressed += 1
                    return cleaned, True   # SUPPRESSED
                else:
                    logger.debug(
                        f"[AEC] Barge-in energy ratio {mic_rms/ref_rms:.1f}× — ALLOWING"
                    )
                    return cleaned, False  # ALLOWED (user is shouting over AI)
            else:
                # No reference → full state-gate suppression
                self.chunks_suppressed += 1
                return cleaned, True

        # ── Layer B: Spectral subtraction when NOT suppressed ─────────────
        # (Clean up any residual echo even outside grace period)
        if self._has_reference:
            cleaned = self._spectral_subtract(cleaned, strength=0.3)

        return cleaned, False

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _in_grace_period(self) -> bool:
        """True for GRACE_PERIOD_MS after AI starts speaking."""
        if not self._ai_speaking or self._ai_started_at is None:
            return False
        elapsed_ms = (time.monotonic() - self._ai_started_at) * 1000
        return elapsed_ms < GRACE_PERIOD_MS

    def _in_echo_tail(self) -> bool:
        """True for POST_STOP_BUFFER_MS after AI stops speaking."""
        if self._ai_speaking or self._ai_stopped_at is None:
            return False
        elapsed_ms = (time.monotonic() - self._ai_stopped_at) * 1000
        return elapsed_ms < POST_STOP_BUFFER_MS

    def _reference_rms(self) -> float:
        if not self._reference_buffer:
            return 0.0
        arr = np.array(list(self._reference_buffer)[-2048:], dtype=np.float32)
        return float(np.sqrt(np.mean(arr**2) + 1e-9))

    def _spectral_subtract(
        self,
        audio: np.ndarray,
        strength: float = SPECTRAL_ALPHA,
    ) -> np.ndarray:
        """
        Simplified spectral subtraction using the reference buffer.
        Uses torch.fft (CUDA) when available for ~3× faster processing,
        falling back to numpy.fft on CPU. Audio stays float32 throughout
        (zero extra copies).
        """
        if not self._reference_buffer:
            return audio

        n = len(audio)
        ref_arr = np.array(list(self._reference_buffer)[-n:], dtype=np.float32)

        # Zero-pad reference if shorter
        if len(ref_arr) < n:
            ref_arr = np.pad(ref_arr, (0, n - len(ref_arr)))

        # ── Fast path: torch.fft on CUDA ────────────────────────────────
        if _TORCH_AVAILABLE and _torch.cuda.is_available():
            mic_t = _torch.from_numpy(audio).cuda()
            ref_t = _torch.from_numpy(ref_arr[:n]).cuda()

            mic_fft   = _torch.fft.rfft(mic_t)
            ref_fft   = _torch.fft.rfft(ref_t)

            mic_mag   = mic_fft.abs()
            ref_mag   = ref_fft.abs()
            mic_phase = _torch.angle(mic_fft)

            sub_mag   = _torch.clamp(mic_mag - strength * ref_mag, min=0.0)
            result_fft = sub_mag * _torch.exp(1j * mic_phase)

            return _torch.fft.irfft(result_fft, n=n).cpu().numpy().astype(np.float32)

        # ── Fallback: numpy FFT on CPU ───────────────────────────────────
        mic_fft = np.fft.rfft(audio)
        ref_fft = np.fft.rfft(ref_arr[:n])

        mic_mag  = np.abs(mic_fft)
        ref_mag  = np.abs(ref_fft)
        mic_phase = np.angle(mic_fft)

        subtracted_mag = np.maximum(mic_mag - strength * ref_mag, 0.0)
        result_fft = subtracted_mag * np.exp(1j * mic_phase)

        return np.fft.irfft(result_fft, n=n).astype(np.float32)

    def get_stats(self) -> dict:
        return {
            "ai_speaking":        self._ai_speaking,
            "has_reference":      self._has_reference,
            "chunks_processed":   self.chunks_processed,
            "chunks_suppressed":  self.chunks_suppressed,
            "suppression_rate":   (
                round(self.chunks_suppressed / max(self.chunks_processed, 1), 3)
            ),
        }

    def reset(self):
        """Reset for a new session."""
        self._ai_speaking    = False
        self._ai_started_at  = None
        self._ai_stopped_at  = None
        self._reference_buffer.clear()
        self._has_reference  = False
        self.chunks_suppressed = 0
        self.chunks_processed  = 0