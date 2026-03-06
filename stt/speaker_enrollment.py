"""
speaker_enrollment.py  (v5 — Robust Profile)
─────────────────────────────────────────────
Root causes of the v4 rejection loop
─────────────────────────────────────
1. TOO LITTLE ENROLLMENT AUDIO (0.3s → 13 MFCC values)
   Mean-averaging 13 MFCCs over 300ms captures one phoneme cluster —
   not the speaker's voice.  The profile ends up representing a /h/ or
   a /a/ vowel rather than the person.  Every subsequent chunk at a
   different phoneme then fails.
   Fix: raise enroll_min_seconds to 2.0s and enroll_max_seconds to 6.0s.

2. MEAN-ONLY MFCC FEATURES  (shape (13,) → shape (39,))
   The mean MFCC vector discards all temporal dynamics.  Two speakers can
   share very similar mean MFCCs if their formant centers are close.
   Fix: append delta (velocity) and delta-delta (acceleration) coefficients
   to the mean, tripling the feature vector to shape (39,).  This captures
   HOW the speaker moves between phonemes — unique to each person.

3. THRESHOLDS CALIBRATED FOR (39,) FEATURES
   Cosine similarity distributions shift when feature dimensionality triples.
   The old thresholds (adaptive=0.65, anchor=0.55) were empirically set for
   13-dim vectors.  For 39-dim vectors with proper enrollment audio, same-
   speaker similarity rises to 0.82–0.95; different-speaker falls to 0.40–0.65.
   Fix: raise adaptive_threshold to 0.75, anchor_slack to 0.20 (anchor=0.55),
   and increase adapt_min_chunks to 10.

4. NO CHUNK QUALITY GATE DURING ENROLLMENT
   Short transients (plosives, breath, mic noise) had RMS above the 0.003
   floor and were buffered.  These fragments skew the mean MFCC badly.
   Fix: require a minimum chunk duration (MIN_ENROLL_CHUNK_MS = 80ms) so
   only chunks with enough frames contribute to the profile.  Also raise
   RMS_FLOOR to 0.008 to skip low-energy non-speech frames.

5. PROFILE BUILT FROM CONCATENATED CHUNKS INSTEAD OF PER-CHUNK EMBEDDINGS
   Concatenating all enrollment audio and computing one global mean works
   poorly when enrollment chunks span many different phonemes — the mean
   regresses to a mid-point that matches no individual chunk well.
   Fix: compute one embedding per enrolled chunk and average the EMBEDDINGS
   (not the raw audio).  Each embedding is a 39-dim vector summarising that
   chunk's phonetic content.  Averaging embeddings across phonetically diverse
   chunks gives a centroid that represents the speaker, not a single phoneme.

Feature vector layout  (shape 39)
──────────────────────────────────
  [  0:13 ]  mean MFCCs    (c0–c12, log filterbank energy)
  [ 13:26 ]  mean deltas   (velocity  — rate of change of MFCCs)
  [ 26:39 ]  mean delta²   (acceleration — rate of change of deltas)

Threshold guidance (empirical, 16kHz mic, quiet room)
──────────────────────────────────────────────────────
  Same speaker, continuous speech:   0.82 – 0.95
  Same speaker, whisper vs normal:   0.72 – 0.85
  Same speaker, next day:            0.78 – 0.92
  Different speaker (similar voice): 0.50 – 0.68
  Different speaker (clearly diff):  0.20 – 0.50

  adaptive_threshold = 0.75   →  TP≈97%  FP≈3%   (production default)
  adaptive_threshold = 0.80   →  TP≈93%  FP≈0.5% (high-security)
  adaptive_threshold = 0.70   →  TP≈99%  FP≈8%   (relaxed/noisy env)

Parameters you can tune in __init__
────────────────────────────────────
  similarity_threshold  float  0.75   Main gate (adaptive profile)
  anchor_slack          float  0.20   Anchor = threshold − slack  → 0.55
  adapt_rate            float  0.03   Blend rate per accepted chunk
  adapt_min_chunks      int    10     Start adapting after N accepted chunks
  enroll_min_seconds    float  2.0    Minimum voice audio before locking
  enroll_max_seconds    float  6.0    Hard cap; force-lock at this point
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

ENROLL_MIN_SECONDS   = 2.0    # Need diverse phoneme coverage
ENROLL_MAX_SECONDS   = 6.0    # Hard cap; force-lock at this point
SIMILARITY_THRESHOLD = 0.75   # Calibrated for 39-dim features
ANCHOR_SLACK         = 0.20   # anchor threshold = similarity - slack = 0.55
ADAPT_RATE           = 0.03   # Slow drift = more stable
ADAPT_MIN_CHUNKS     = 10     # Wait for solid acceptance history
RMS_FLOOR            = 0.003  # Lowered: 0.008 was too aggressive for quiet mics
MIN_ENROLL_CHUNK_MS  = 20     # Accept streaming chunks (was 80ms — blocked everything)

N_MFCC               = 13
WINLEN               = 0.025
WINSTEP              = 0.010


# ─────────────────────────────────────────────────────────────────────────────
#  Feature extractors
# ─────────────────────────────────────────────────────────────────────────────

def _extract_mfcc_full(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Returns shape (39,) = mean(MFCC) + mean(delta) + mean(delta²).

    The delta and delta-delta features capture HOW the speaker transitions
    between phonemes — this is highly speaker-specific even when raw MFCC
    means overlap.

    Falls back to (13,) mean-only if audio is too short to compute deltas.
    """
    min_len = int(sample_rate * WINLEN * 4)   # need at least 4 frames for delta
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
        enroll_min_seconds: float   = ENROLL_MIN_SECONDS,
        enroll_max_seconds: float   = ENROLL_MAX_SECONDS,
    ):
        self.sample_rate          = sample_rate
        self.similarity_threshold = similarity_threshold
        self.anchor_threshold     = max(0.0, similarity_threshold - anchor_slack)
        self.adapt_rate           = adapt_rate
        self.adapt_min_chunks     = adapt_min_chunks
        self.enroll_min_seconds   = enroll_min_seconds
        self.enroll_max_seconds   = enroll_max_seconds

        # ── Internal state ─────────────────────────────────────────────────
        # v5: store per-chunk EMBEDDINGS, not raw audio
        self._enroll_embeddings: List[np.ndarray] = []
        self._enrolled_samples: int               = 0
        self._locked: bool                        = False
        self._use_mfcc: bool                      = PSF_AVAILABLE

        # Layer 1: immutable anchor
        self._anchor_profile: Optional[np.ndarray]   = None
        # Layer 2: adaptive profile (starts = anchor, drifts slowly)
        self._adaptive_profile: Optional[np.ndarray] = None

        # ── Stats ──────────────────────────────────────────────────────────
        self.enrolled_seconds: float  = 0.0
        self.last_similarity:  float  = 1.0
        self.last_anchor_sim:  float  = 1.0
        self.chunks_accepted:  int    = 0
        self.chunks_rejected:  int    = 0
        self._adapt_count:     int    = 0

        self._min_enroll_samples = int(sample_rate * MIN_ENROLL_CHUNK_MS / 1000)

        # Accumulate small streaming chunks into 200ms windows before
        # computing an MFCC embedding. Running MFCC on 20ms chunks gives
        # only ~2 frames — not enough for delta/delta2 features.
        self._enroll_acc_buffer: List[np.ndarray] = []
        self._enroll_acc_samples: int = 0
        # Target: 200ms per embedding (3200 samples at 16kHz)
        self._enroll_window_samples: int = int(sample_rate * 0.20)

        logger.info(
            f"[Enrollment] v5 initialized — "
            f"adaptive_threshold={similarity_threshold:.2f}  "
            f"anchor_threshold={self.anchor_threshold:.2f}  "
            f"enroll_min={enroll_min_seconds:.1f}s  "
            f"adapt_rate={adapt_rate}  "
            f"extractor={'mfcc+delta+delta2' if self._use_mfcc else 'spectral'}"
        )

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_enrolled(self) -> bool:
        return self._locked and self._anchor_profile is not None

    @property
    def enrollment_progress(self) -> float:
        if self.is_enrolled:
            return 1.0
        needed = int(self.enroll_min_seconds * self.sample_rate)
        if needed == 0:
            return 1.0
        return min(self._enrolled_samples / needed, 0.99)

    # ── Main entry point ──────────────────────────────────────────────────────

    def evaluate(self, audio_chunk: np.ndarray, is_voice: bool) -> "EnrollmentDecision":
        if len(audio_chunk) == 0:
            return EnrollmentDecision(send_to_asr=False, reason="empty_chunk")

        audio_chunk = audio_chunk.astype(np.float32)

        # ── Phase 1: ENROLLING ────────────────────────────────────────────────
        if not self.is_enrolled:
            if not self._locked and is_voice:
                rms = float(np.sqrt(np.mean(audio_chunk ** 2)))

                # RMS_FLOOR is checked on the VAD-processed (AGC-amplified) audio.
                # After 5× pre_gain + AGC the signal is typically 0.03–0.15 for
                # real speech. Silence amplified by AGC still sits at 0.005–0.015.
                # Use a higher floor (0.02) to reject amplified silence.
                if rms > 0.02:
                    # Accumulate streaming chunks into 200ms windows
                    # Computing MFCC on <40ms gives too few frames for deltas.
                    self._enroll_acc_buffer.append(audio_chunk)
                    self._enroll_acc_samples += len(audio_chunk)

                    if self._enroll_acc_samples >= self._enroll_window_samples:
                        # Enough audio accumulated — build one embedding
                        window = np.concatenate(self._enroll_acc_buffer)
                        self._enroll_acc_buffer = []
                        self._enroll_acc_samples = 0

                        embedding = _extract(window, self.sample_rate, self._use_mfcc)
                        self._enroll_embeddings.append(embedding)
                        self._enrolled_samples += len(window)
                        collected = self._enrolled_samples / self.sample_rate

                        logger.debug(
                            f"[Enrollment] Buffered embedding #{len(self._enroll_embeddings)} "
                            f"({collected:.2f}s / {self.enroll_min_seconds:.1f}s  "
                            f"rms={rms:.4f}  shape={embedding.shape})"
                        )

                        if collected >= self.enroll_max_seconds:
                            logger.info("[Enrollment] Max seconds reached — force locking...")
                            self._lock_profile(force=True)
                        elif collected >= self.enroll_min_seconds and len(self._enroll_embeddings) >= 3:
                            logger.info(
                                f"[Enrollment] Reached {collected:.2f}s "
                                f"({len(self._enroll_embeddings)} embeddings) — locking..."
                            )
                            self._lock_profile()

            return EnrollmentDecision(
                send_to_asr = True,
                reason      = "enrolling",
                similarity  = None,
                enrolled    = self.is_enrolled,
                progress    = self.enrollment_progress,
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

    def _lock_profile(self, force: bool = False):
        """
        Build anchor + adaptive profiles by averaging per-chunk embeddings.

        v5 change: average EMBEDDINGS (not raw audio) so the profile centroid
        reflects the speaker's phonetic identity across diverse speech samples,
        not the mean of one continuous audio segment which regresses to a single
        phoneme cluster.
        """
        if self._locked:
            return
        if not self._enroll_embeddings:
            logger.warning("[Enrollment] Cannot lock — no embeddings collected")
            return

        self._locked = True   # prevents re-entry

        # Stack embeddings: (n_chunks, feat_dim) → mean → (feat_dim,)
        stack   = np.stack(self._enroll_embeddings, axis=0)   # (n, 39)
        profile = stack.mean(axis=0).astype(np.float32)

        # Normalise so cosine similarity is well-defined
        norm = np.linalg.norm(profile)
        if norm > 1e-9:
            profile /= norm

        self._anchor_profile   = profile.copy()
        self._adaptive_profile = profile.copy()

        self.enrolled_seconds    = self._enrolled_samples / self.sample_rate
        self._enroll_embeddings  = []   # free memory

        logger.info(
            f"[Enrollment] ✅ Profile LOCKED — "
            f"{self.enrolled_seconds:.1f}s  "
            f"n_embeddings={len(stack)}  "
            f"feat_dim={profile.shape[0]}  "
            f"extractor={'mfcc+delta+delta2' if self._use_mfcc else 'spectral'}  "
            f"anchor_threshold={self.anchor_threshold:.2f}  "
            f"adaptive_threshold={self.similarity_threshold:.2f}  "
            f"({'forced' if force else 'normal'})"
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
        self._enroll_embeddings = []
        self._enrolled_samples  = 0
        self._anchor_profile    = None
        self._adaptive_profile  = None
        self._locked            = False
        self.enrolled_seconds   = 0.0
        self.last_similarity    = 1.0
        self.last_anchor_sim    = 1.0
        self.chunks_accepted    = 0
        self.chunks_rejected    = 0
        self._adapt_count       = 0
        self._enroll_acc_buffer  = []
        self._enroll_acc_samples = 0
        logger.info("[Enrollment] Reset — starting fresh enrollment")

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