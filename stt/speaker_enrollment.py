"""
speaker_enrollment.py  (v6 — First-Chunk Instant Enrollment)
─────────────────────────────────────────────────────────────
Design goal
───────────
Lock the speaker profile on the VERY FIRST voice chunk that passes the
VAD + RMS gate, even if it is a single short word.  ASR is unblocked
immediately on that same chunk — zero enrollment latency.

How it works
────────────
• No accumulation window.  Each incoming voice chunk is embedded directly
  using _extract_mfcc_full() with aggressive zero-padding so even a 20ms
  chunk produces a valid (39,) feature vector.

• On the first qualifying chunk (_locked = False, is_voice = True,
  rms > RMS_FLOOR) the embedding is computed and _lock_profile() is
  called immediately.  The chunk is also forwarded to ASR (send_to_asr=True).

• Subsequent chunks use the normal two-layer similarity check
  (anchor + adaptive) exactly as in v5.

• The adaptive profile continues to drift slowly so the system adjusts
  to the speaker's voice over the session.

Trade-off vs v5
───────────────
v5 collected 2–6 s to build a robust centroid across many phonemes.
v6 locks on one chunk — the anchor may represent only one phoneme, so
similarity_threshold and anchor_slack are loosened slightly:
  adaptive_threshold = 0.65  (was 0.75)
  anchor_slack       = 0.25  (was 0.20)  →  anchor = 0.40
This avoids false rejections when the first word (e.g. "yes") differs
spectrally from follow-up speech.

Feature vector layout  (shape 39)
──────────────────────────────────
  [  0:13 ]  mean MFCCs    (c0–c12)
  [ 13:26 ]  mean deltas
  [ 26:39 ]  mean delta²

Parameters you can tune in __init__
────────────────────────────────────
  similarity_threshold  float  0.65   Main gate (loosened for 1-chunk enroll)
  anchor_slack          float  0.25   anchor = threshold − slack  → 0.40
  adapt_rate            float  0.05   Slightly faster drift than v5
  adapt_min_chunks      int    3      Start adapting sooner
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

try:
    import python_speech_features as psf
    PSF_AVAILABLE = True
except ImportError:
    PSF_AVAILABLE = False
    logger.warning("python_speech_features not installed — using spectral fallback")


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration defaults
# ─────────────────────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.65   # Loosened: anchor covers only 1 chunk phoneme
ANCHOR_SLACK         = 0.25   # anchor threshold = 0.65 − 0.25 = 0.40
ADAPT_RATE           = 0.05   # Slightly faster drift to cover more phonemes early
ADAPT_MIN_CHUNKS     = 3      # Start adapting after only 3 accepted chunks
RMS_FLOOR            = 0.008  # Skip truly silent / noise-only chunks

N_MFCC               = 13
WINLEN               = 0.025
WINSTEP              = 0.010


# ─────────────────────────────────────────────────────────────────────────────
#  Feature extractors
# ─────────────────────────────────────────────────────────────────────────────

def _extract_mfcc_full(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Returns shape (39,) = mean(MFCC) + mean(delta) + mean(delta²).

    v6: Pads aggressively to at least 200ms so even a single 20ms streaming
    chunk produces a valid 39-dim feature vector.  The zero-padding only
    mildly pulls mean values toward zero — acceptable for instant enrollment.
    """
    # Pad to at least 200ms — guarantees enough frames for delta/delta2
    min_len = max(int(sample_rate * 0.20), int(sample_rate * WINLEN * 4))
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))

    mfcc = psf.mfcc(
        audio,
        samplerate   = sample_rate,
        winlen       = WINLEN,
        winstep      = WINSTEP,
        numcep       = N_MFCC,
        nfilt        = 26,
        nfft         = 512,
        preemph      = 0.97,
        ceplifter    = 22,
        appendEnergy = True,
    )
    # mfcc shape: (n_frames, 13)

    if mfcc.shape[0] < 3:
        # Too few frames for delta — return mean-only (13,)
        return mfcc.mean(axis=0).astype(np.float32)

    # Compute delta and delta-delta using simple finite differences
    delta   = _compute_delta(mfcc)     # (n_frames, 13)
    delta2  = _compute_delta(delta)    # (n_frames, 13)

    # Stack: mean across time → (39,)
    feat = np.concatenate([
        mfcc.mean(axis=0),
        delta.mean(axis=0),
        delta2.mean(axis=0),
    ]).astype(np.float32)

    return feat


def _compute_delta(features: np.ndarray, N: int = 2) -> np.ndarray:
    """
    Compute delta features using the standard regression formula.
    N = number of frames on each side.
    Edges are padded by replication.
    """
    n_frames, n_feats = features.shape
    padded = np.pad(features, ((N, N), (0, 0)), mode="edge")
    denom  = 2 * sum(i**2 for i in range(1, N + 1))
    delta  = np.zeros_like(features)
    for t in range(n_frames):
        for n in range(1, N + 1):
            delta[t] += n * (padded[t + N + n] - padded[t + N - n])
    return delta / denom


def _extract_spectral(audio: np.ndarray) -> np.ndarray:
    """
    Returns shape (8,) — richer spectral fallback when psf is not installed.
    Uses 6 log-spaced frequency bands + RMS + zero-crossing rate.
    """
    n     = len(audio)
    spec  = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(n)

    # 6 log-spaced band boundaries (0–8kHz at 16kHz SR)
    bands = np.logspace(np.log10(0.01), np.log10(0.5), 7)
    band_energies = []
    for i in range(len(bands) - 1):
        lo = int(bands[i]     * n)
        hi = int(bands[i + 1] * n)
        if hi > lo:
            band_energies.append(spec[lo:hi].mean())
        else:
            band_energies.append(0.0)

    rms  = float(np.sqrt(np.mean(audio ** 2) + 1e-9))
    zcr  = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)

    return np.array(band_energies + [rms, zcr], dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        # Shape mismatch can happen during transition from 13→39 feature size;
        # fall back to comparing the shared prefix only.
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 1.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _extract(audio: np.ndarray, sample_rate: int, use_mfcc: bool) -> np.ndarray:
    """Unified extractor dispatch."""
    if use_mfcc:
        return _extract_mfcc_full(audio, sample_rate)
    return _extract_spectral(audio)


# ─────────────────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────────────────

class SpeakerEnrollmentService:
    """
    Robust adaptive two-layer speaker enrollment (v5).

    Key improvements over v4
    ────────────────────────
    • 39-dim features (MFCC + delta + delta²) instead of 13-dim mean-only
    • Enrollment builds from per-chunk embeddings, not raw audio concat
    • Minimum enrollment duration raised to 2.0s
    • Quality gate: rejects enrollment chunks that are too short or too quiet
    • Thresholds recalibrated for 39-dim feature space
    • Slower adapt rate (3%) for more stable post-enrollment profiles
    """

    def __init__(
        self,
        sample_rate: int            = 16000,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        anchor_slack: float         = ANCHOR_SLACK,
        adapt_rate: float           = ADAPT_RATE,
        adapt_min_chunks: int       = ADAPT_MIN_CHUNKS,
    ):
        self.sample_rate          = sample_rate
        self.similarity_threshold = similarity_threshold
        self.anchor_threshold     = max(0.0, similarity_threshold - anchor_slack)
        self.adapt_rate           = adapt_rate
        self.adapt_min_chunks     = adapt_min_chunks

        # ── Internal state ─────────────────────────────────────────────────
        self._locked: bool                        = False
        self._use_mfcc: bool                      = PSF_AVAILABLE

        # Layer 1: immutable anchor (locked on first voice chunk)
        self._anchor_profile: Optional[np.ndarray]   = None
        # Layer 2: adaptive profile (starts = anchor, drifts with each accepted chunk)
        self._adaptive_profile: Optional[np.ndarray] = None

        # ── Stats ──────────────────────────────────────────────────────────
        self.enrolled_seconds: float  = 0.0
        self.last_similarity:  float  = 1.0
        self.last_anchor_sim:  float  = 1.0
        self.chunks_accepted:  int    = 0
        self.chunks_rejected:  int    = 0
        self._adapt_count:     int    = 0

        logger.info(
            f"[Enrollment] v6 initialized — INSTANT (first-chunk) mode  "
            f"adaptive_threshold={similarity_threshold:.2f}  "
            f"anchor_threshold={self.anchor_threshold:.2f}  "
            f"adapt_rate={adapt_rate}  "
            f"extractor={'mfcc+delta+delta2' if self._use_mfcc else 'spectral'}"
        )

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_enrolled(self) -> bool:
        return self._locked and self._anchor_profile is not None

    @property
    def enrollment_progress(self) -> float:
        # v6: either not enrolled (0.0) or enrolled (1.0) — no partial progress
        return 1.0 if self.is_enrolled else 0.0

    # ── Main entry point ──────────────────────────────────────────────────────

    def evaluate(self, audio_chunk: np.ndarray, is_voice: bool) -> "EnrollmentDecision":
        if len(audio_chunk) == 0:
            return EnrollmentDecision(send_to_asr=False, reason="empty_chunk")

        audio_chunk = audio_chunk.astype(np.float32)

        # ── Phase 1: ENROLLING — instant lock on first valid voice chunk ─────
        if not self.is_enrolled:
            if is_voice:
                rms = float(np.sqrt(np.mean(audio_chunk ** 2)))

                if rms >= RMS_FLOOR:
                    # Lock immediately on this single chunk.
                    # _extract_mfcc_full zero-pads to 200ms internally so even
                    # a 20ms chunk produces a valid 39-dim embedding.
                    embedding = _extract(audio_chunk, self.sample_rate, self._use_mfcc)
                    self._lock_profile_instant(embedding, len(audio_chunk))

                    logger.info(
                        f"[Enrollment] ⚡ INSTANT LOCK — first voice chunk  "
                        f"rms={rms:.4f}  samples={len(audio_chunk)}  "
                        f"feat_dim={embedding.shape[0]}"
                    )

                    # Forward this chunk to ASR immediately — no blocking.
                    return EnrollmentDecision(
                        send_to_asr = True,
                        reason      = "enrolled_instant",
                        similarity  = 1.0,
                        enrolled    = True,
                        progress    = 1.0,
                    )

            # Not yet a voice chunk (or rms too low) — waiting for first word
            return EnrollmentDecision(
                send_to_asr = False,
                reason      = "waiting_for_voice",
                similarity  = None,
                enrolled    = False,
                progress    = 0.0,
            )

        # ── Phase 2: ENROLLED — two-layer similarity check ────────────────────
        if not is_voice:
            return EnrollmentDecision(
                send_to_asr = False,
                reason      = "silence",
                enrolled    = True,
                progress    = 1.0,
            )

        embedding = _extract(audio_chunk, self.sample_rate, self._use_mfcc)

        # ── Layer 1: Anchor check (hard floor) ────────────────────────────────
        anchor_sim = _cosine_similarity(self._anchor_profile, embedding)
        self.last_anchor_sim = anchor_sim

        if anchor_sim < self.anchor_threshold:
            self.chunks_rejected += 1
            logger.debug(
                f"[Enrollment] REJECTED by ANCHOR "
                f"anchor_sim={anchor_sim:.3f} < {self.anchor_threshold:.3f}"
            )
            return EnrollmentDecision(
                send_to_asr = False,
                reason      = "speaker_mismatch",
                similarity  = anchor_sim,   # use anchor_sim so client never sees None
                enrolled    = True,
                progress    = 1.0,
            )

        # ── Layer 2: Adaptive check ───────────────────────────────────────────
        adaptive_sim = _cosine_similarity(self._adaptive_profile, embedding)
        self.last_similarity = adaptive_sim

        if adaptive_sim < self.similarity_threshold:
            self.chunks_rejected += 1
            logger.debug(
                f"[Enrollment] REJECTED by ADAPTIVE "
                f"adaptive_sim={adaptive_sim:.3f} < {self.similarity_threshold:.3f}  "
                f"anchor_sim={anchor_sim:.3f}"
            )
            return EnrollmentDecision(
                send_to_asr = False,
                reason      = "speaker_mismatch",
                similarity  = adaptive_sim,
                enrolled    = True,
                progress    = 1.0,
            )

        # ── ACCEPTED — blend into adaptive profile ────────────────────────────
        self.chunks_accepted += 1

        if self.chunks_accepted >= self.adapt_min_chunks:
            self._adapt_profile(embedding)

        logger.debug(
            f"[Enrollment] ACCEPTED  "
            f"adaptive={adaptive_sim:.3f}  anchor={anchor_sim:.3f}  "
            f"adapt_count={self._adapt_count}"
        )

        return EnrollmentDecision(
            send_to_asr = True,
            reason      = "speaker_match",
            similarity  = adaptive_sim,
            enrolled    = True,
            progress    = 1.0,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _lock_profile_instant(self, embedding: np.ndarray, n_samples: int):
        """
        v6: Lock profile immediately from a single embedding.
        The anchor is set to this normalised embedding.
        The adaptive profile starts as a copy and drifts over time.
        """
        if self._locked:
            return

        self._locked = True

        profile = embedding.astype(np.float32).copy()
        norm = np.linalg.norm(profile)
        if norm > 1e-9:
            profile /= norm

        self._anchor_profile   = profile.copy()
        self._adaptive_profile = profile.copy()
        self.enrolled_seconds  = n_samples / self.sample_rate

        logger.info(
            f"[Enrollment] ✅ Profile LOCKED (instant) — "
            f"{self.enrolled_seconds*1000:.0f}ms  "
            f"feat_dim={profile.shape[0]}  "
            f"extractor={'mfcc+delta+delta2' if self._use_mfcc else 'spectral'}  "
            f"anchor_threshold={self.anchor_threshold:.2f}  "
            f"adaptive_threshold={self.similarity_threshold:.2f}"
        )

    def _adapt_profile(self, embedding: np.ndarray):
        """
        Exponential moving average blend into the adaptive profile.
        The anchor is NEVER touched.
        """
        self._adaptive_profile = (
            (1.0 - self.adapt_rate) * self._adaptive_profile
            + self.adapt_rate * embedding
        ).astype(np.float32)

        norm = np.linalg.norm(self._adaptive_profile)
        if norm > 1e-9:
            self._adaptive_profile /= norm

        self._adapt_count += 1

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self):
        self._anchor_profile   = None
        self._adaptive_profile = None
        self._locked           = False
        self.enrolled_seconds  = 0.0
        self.last_similarity   = 1.0
        self.last_anchor_sim   = 1.0
        self.chunks_accepted   = 0
        self.chunks_rejected   = 0
        self._adapt_count      = 0
        logger.info("[Enrollment] Reset — ready for next first-chunk enrollment")

    def get_stats(self) -> dict:
        return {
            "enrolled":           self.is_enrolled,
            "enrolled_seconds":   round(self.enrolled_seconds, 2),
            "progress":           round(self.enrollment_progress, 2),
            "last_similarity":    round(self.last_similarity, 3),
            "last_anchor_sim":    round(self.last_anchor_sim, 3),
            "chunks_accepted":    self.chunks_accepted,
            "chunks_rejected":    self.chunks_rejected,
            "adapt_count":        self._adapt_count,
            "adaptive_threshold": self.similarity_threshold,
            "anchor_threshold":   round(self.anchor_threshold, 3),
            "feat_dim":           (
                39 if (self._use_mfcc and self._anchor_profile is not None
                       and len(self._anchor_profile) == 39)
                else 13 if self._use_mfcc else 8
            ),
            "extractor": "mfcc+delta+delta2" if self._use_mfcc else "spectral",
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Decision dataclass  (API unchanged — pipeline.py compatible)
# ─────────────────────────────────────────────────────────────────────────────

class EnrollmentDecision:
    __slots__ = ("send_to_asr", "reason", "similarity", "enrolled", "progress")

    def __init__(
        self,
        send_to_asr: bool,
        reason: str,
        similarity: Optional[float] = None,
        enrolled: bool  = False,
        progress: float = 0.0,
    ):
        self.send_to_asr = send_to_asr
        self.reason      = reason
        self.similarity  = similarity
        self.enrolled    = enrolled
        self.progress    = progress

    def to_dict(self) -> dict:
        return {
            "send_to_asr": self.send_to_asr,
            "reason":      self.reason,
            "similarity":  round(self.similarity, 3) if self.similarity is not None else None,
            "enrolled":    self.enrolled,
            "progress":    round(self.progress, 2),
        }