"""
test_agc.py — Unit tests for stt/agc.py
  • SimpleAGC: process, gain adjustment, reset, edge cases

Run:
    pytest tests/test_agc.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stt"))

from agc import SimpleAGC  # noqa


class TestSimpleAGC:

    def test_init_defaults(self):
        agc = SimpleAGC()
        assert agc.target_rms == 0.02
        assert agc.max_gain == 10.0
        assert agc.current_gain == 1.0

    def test_init_custom(self):
        agc = SimpleAGC(target_rms=0.05, max_gain=5.0)
        assert agc.target_rms == 0.05
        assert agc.max_gain == 5.0

    def test_process_silence(self):
        """Very quiet audio should be returned unchanged (rms < 1e-5)."""
        agc = SimpleAGC()
        audio = np.zeros(320, dtype=np.float32)
        result = agc.process(audio)
        np.testing.assert_array_equal(result, audio)

    def test_process_empty_array(self):
        agc = SimpleAGC()
        audio = np.array([], dtype=np.float32)
        result = agc.process(audio)
        assert len(result) == 0

    def test_process_quiet_audio_amplifies(self):
        """Quiet audio should be amplified toward target RMS."""
        agc = SimpleAGC(target_rms=0.02, max_gain=10.0)
        # Very quiet sinusoid
        t = np.linspace(0, 0.02, 320, dtype=np.float32)
        audio = 0.001 * np.sin(2 * np.pi * 440 * t)
        result = agc.process(audio)
        # Result should be louder than input
        assert np.sqrt(np.mean(result**2)) > np.sqrt(np.mean(audio**2))

    def test_process_loud_audio_attenuates(self):
        """Loud audio should maintain or reduce gain."""
        agc = SimpleAGC(target_rms=0.02, max_gain=10.0)
        agc.current_gain = 5.0  # pretend we were amplifying
        t = np.linspace(0, 0.02, 320, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # very loud
        result = agc.process(audio)
        # Gain should decrease (release)
        assert agc.current_gain < 5.0

    def test_max_gain_respected(self):
        """Gain should never exceed max_gain."""
        agc = SimpleAGC(target_rms=0.5, max_gain=3.0)
        t = np.linspace(0, 0.02, 320, dtype=np.float32)
        audio = 0.001 * np.sin(2 * np.pi * 440 * t)
        # Process multiple times to converge
        for _ in range(100):
            agc.process(audio)
        assert agc.current_gain <= 3.0 + 0.01  # small tolerance for float

    def test_smooth_gain_transition(self):
        """Gain should change smoothly, not jump to target immediately."""
        agc = SimpleAGC()
        agc.current_gain = 1.0
        t = np.linspace(0, 0.02, 320, dtype=np.float32)
        audio = 0.001 * np.sin(2 * np.pi * 440 * t)
        agc.process(audio)
        # After one pass, gain should increase but not jump to max
        assert 1.0 < agc.current_gain < 10.0

    def test_reset(self):
        agc = SimpleAGC()
        agc.current_gain = 5.0
        agc.reset()
        assert agc.current_gain == 1.0

    def test_output_shape_matches_input(self):
        agc = SimpleAGC()
        audio = np.random.randn(512).astype(np.float32) * 0.01
        result = agc.process(audio)
        assert result.shape == audio.shape

    def test_output_dtype_float(self):
        agc = SimpleAGC()
        audio = np.random.randn(320).astype(np.float32) * 0.01
        result = agc.process(audio)
        assert result.dtype == np.float32 or result.dtype == np.float64
