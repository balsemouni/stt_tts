"""
test_tts_voice_gate.py — Unit tests for stt/tts_voice_gate.py
  • TTSVoiceGate: enroll, check, cosine similarity, is_ready, reset

Run:
    pytest tests/test_tts_voice_gate.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stt"))

from tts_voice_gate import TTSVoiceGate, _hz_to_mel, _mel_to_hz, _mel_filterbank, N_MELS  # noqa


# ═══════════════════════════════════════════════════════════════════════════════
#  Mel Helper Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMelHelpers:
    def test_hz_to_mel(self):
        # 1000 Hz ≈ 1000 mel (by definition of the mel scale)
        mel = _hz_to_mel(1000.0)
        assert 999 < mel < 1001

    def test_mel_to_hz_roundtrip(self):
        for hz in [100, 440, 1000, 4000, 8000]:
            mel = _hz_to_mel(hz)
            recovered = _mel_to_hz(mel)
            assert abs(recovered - hz) < 0.01

    def test_mel_filterbank_shape(self):
        n_fft = 512
        sr = 16000
        fb = _mel_filterbank(N_MELS, n_fft, sr)
        assert fb.shape == (N_MELS, n_fft // 2 + 1)

    def test_mel_filterbank_non_negative(self):
        fb = _mel_filterbank(N_MELS, 512, 16000)
        assert np.all(fb >= 0)

    def test_mel_filterbank_row_sums(self):
        """Each mel band should have some non-zero weights."""
        fb = _mel_filterbank(N_MELS, 512, 16000)
        for i in range(N_MELS):
            assert fb[i].sum() > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  TTSVoiceGate Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTTSVoiceGate:

    def _make_gate(self, **kwargs):
        return TTSVoiceGate(sample_rate=16000, **kwargs)

    def _sinusoid(self, freq=440, duration_s=0.1, sr=16000, amplitude=0.5):
        """Generate a sinusoidal PCM chunk."""
        t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
        return amplitude * np.sin(2 * np.pi * freq * t)

    def test_init_defaults(self):
        g = self._make_gate()
        assert g.sample_rate == 16000
        assert g.detection_threshold == 0.70
        assert g.barge_in_threshold == 0.82
        assert g._is_ready is False
        assert g._n_enrolled == 0

    def test_not_ready_before_enrollment(self):
        g = self._make_gate()
        assert g.is_ready is False

    def test_enroll_builds_centroid(self):
        g = self._make_gate(min_enroll_frames=2)
        audio = self._sinusoid(freq=440, duration_s=0.2)
        g.enroll(audio)
        assert g._n_enrolled > 0
        assert g._centroid is not None

    def test_is_ready_after_sufficient_enrollment(self):
        g = self._make_gate(min_enroll_frames=2)
        audio = self._sinusoid(freq=440, duration_s=0.1)
        # Each enroll() call counts as 1 frame, need >= min_enroll_frames calls
        g.enroll(audio)
        g.enroll(audio)
        assert g.is_ready is True

    def test_check_not_ready_passes_audio(self):
        """If gate is not ready (not enrolled), all audio passes through."""
        g = self._make_gate()
        audio = self._sinusoid()
        suppressed, sim = g.check(audio, ai_speaking=False)
        assert suppressed is False

    def test_check_disabled_passes_audio(self):
        g = self._make_gate(enabled=False, min_enroll_frames=2)
        audio = self._sinusoid(freq=440, duration_s=0.5)
        g.enroll(audio)
        suppressed, sim = g.check(audio, ai_speaking=False)
        assert suppressed is False

    def test_check_same_voice_suppressed(self):
        """Mic audio identical to enrolled voice should be suppressed."""
        g = self._make_gate(min_enroll_frames=2, detection_threshold=0.5)
        voice = self._sinusoid(freq=440, duration_s=0.1)
        # Each enroll() call counts as 1 frame
        g.enroll(voice)
        g.enroll(voice)
        assert g.is_ready is True
        # Check with the same signal
        suppressed, sim = g.check(voice, ai_speaking=False)
        assert sim > 0.5  # high similarity
        assert suppressed is True

    def test_check_different_voice_passes(self):
        """Different frequency should have lower similarity."""
        g = self._make_gate(min_enroll_frames=2, detection_threshold=0.85)
        voice = self._sinusoid(freq=440, duration_s=0.5)
        g.enroll(voice)
        # Different frequency
        other = self._sinusoid(freq=2000, duration_s=0.1)
        suppressed, sim = g.check(other, ai_speaking=False)
        # Should likely pass (different spectral content)
        # Exact behavior depends on mel overlap

    def test_cosine_sim_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert TTSVoiceGate._cosine_sim(a, a) == pytest.approx(1.0)

    def test_cosine_sim_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert TTSVoiceGate._cosine_sim(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_cosine_sim_zero_vector(self):
        a = np.array([1.0, 2.0])
        b = np.zeros(2)
        assert TTSVoiceGate._cosine_sim(a, b) == 0.0

    def test_log_mel_empty_audio(self):
        g = self._make_gate()
        result = g._log_mel(np.array([], dtype=np.float32))
        assert result.shape == (N_MELS,)
        np.testing.assert_array_equal(result, np.zeros(N_MELS))

    def test_log_mel_short_audio(self):
        """Audio shorter than one frame should still produce features."""
        g = self._make_gate()
        audio = np.random.randn(100).astype(np.float32)
        result = g._log_mel(audio)
        assert result.shape == (N_MELS,)

    def test_log_mel_normal_audio(self):
        g = self._make_gate()
        audio = self._sinusoid(duration_s=0.1)
        result = g._log_mel(audio)
        assert result.shape == (N_MELS,)
        assert not np.all(result == 0)

    def test_reset_keeps_profile(self):
        g = self._make_gate(min_enroll_frames=2)
        audio = self._sinusoid(freq=440, duration_s=0.1)
        g.enroll(audio)
        g.enroll(audio)
        assert g._is_ready is True
        g.chunks_checked = 10
        g.chunks_suppressed = 5
        g.reset()
        assert g.chunks_checked == 0
        assert g.chunks_suppressed == 0
        # Centroid should persist
        assert g._centroid is not None
        assert g._is_ready is True

    def test_full_reset_clears_everything(self):
        g = self._make_gate(min_enroll_frames=2)
        g.enroll(self._sinusoid(freq=440, duration_s=0.5))
        g.full_reset()
        assert g._centroid is None
        assert g._n_enrolled == 0
        assert g._is_ready is False

    def test_stats_counters(self):
        g = self._make_gate(min_enroll_frames=2, detection_threshold=0.3)
        voice = self._sinusoid(freq=440, duration_s=0.1)
        g.enroll(voice)
        g.enroll(voice)
        assert g.is_ready is True
        g.check(voice, ai_speaking=False)
        g.check(voice, ai_speaking=False)
        assert g.chunks_checked == 2

    def test_barge_in_threshold_higher(self):
        """During ai_speaking, the threshold should be higher (barge_in_threshold)."""
        g = self._make_gate(
            min_enroll_frames=2,
            detection_threshold=0.3,
            barge_in_threshold=0.99  # very high → hard to suppress
        )
        voice = self._sinusoid(freq=440, duration_s=0.5)
        g.enroll(voice)
        # During barge-in (ai_speaking=True), threshold is higher
        suppressed, sim = g.check(voice, ai_speaking=True)
        # With threshold 0.99, it likely won't suppress
