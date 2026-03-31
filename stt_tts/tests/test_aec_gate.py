"""
test_aec_gate.py — Unit tests for stt/aec_gate.py
  • AECGate: set_ai_speaking, echo tail, process (suppress/pass), spectral subtraction, stats, reset

Run:
    pytest tests/test_aec_gate.py -v
"""

import sys
import os
import time
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stt"))

from aec_gate import AECGate, POST_STOP_BUFFER_MS  # noqa


class TestAECGate:

    def test_init_defaults(self):
        gate = AECGate()
        assert gate._ai_speaking is False
        assert gate.chunks_processed == 0
        assert gate.chunks_suppressed == 0

    def test_set_ai_speaking_true(self):
        gate = AECGate()
        gate.set_ai_speaking(True)
        assert gate._ai_speaking is True
        assert gate._ai_started_at is not None

    def test_set_ai_speaking_false(self):
        gate = AECGate()
        gate.set_ai_speaking(True)
        gate.set_ai_speaking(False)
        assert gate._ai_speaking is False
        assert gate._ai_stopped_at is not None

    def test_process_suppresses_during_ai_speaking(self):
        gate = AECGate()
        gate.set_ai_speaking(True)
        audio = np.random.randn(320).astype(np.float32)
        cleaned, suppressed = gate.process(audio)
        assert suppressed is True
        assert gate.chunks_suppressed == 1

    def test_process_suppresses_during_echo_tail(self):
        gate = AECGate()
        gate.set_ai_speaking(True)
        gate.set_ai_speaking(False)
        # Just stopped → in echo tail
        audio = np.random.randn(320).astype(np.float32)
        cleaned, suppressed = gate.process(audio)
        assert suppressed is True

    def test_process_passes_when_idle(self):
        gate = AECGate()
        audio = np.random.randn(320).astype(np.float32)
        cleaned, suppressed = gate.process(audio)
        assert suppressed is False
        assert gate.chunks_processed == 1

    def test_echo_tail_expires(self):
        gate = AECGate()
        gate.set_ai_speaking(True)
        gate.set_ai_speaking(False)
        # Manually set stopped time past the buffer
        gate._ai_stopped_at = time.monotonic() - (POST_STOP_BUFFER_MS / 1000 + 1)
        assert gate._in_echo_tail() is False
        audio = np.random.randn(320).astype(np.float32)
        _, suppressed = gate.process(audio)
        assert suppressed is False

    def test_in_echo_tail_true(self):
        gate = AECGate()
        gate.set_ai_speaking(True)
        gate.set_ai_speaking(False)
        assert gate._in_echo_tail() is True

    def test_in_echo_tail_false_when_never_stopped(self):
        gate = AECGate()
        assert gate._in_echo_tail() is False

    def test_push_reference(self):
        gate = AECGate()
        ref = np.random.randn(512).astype(np.float32)
        gate.push_reference(ref)
        assert gate._has_reference is True
        assert len(gate._reference_buffer) == 512

    def test_spectral_subtraction_with_reference(self):
        gate = AECGate()
        ref = np.random.randn(320).astype(np.float32) * 0.1
        gate.push_reference(ref)
        gate.set_ai_speaking(True)
        audio = np.random.randn(320).astype(np.float32)
        cleaned, suppressed = gate.process(audio)
        assert suppressed is True
        # Cleaned audio should differ from input (spectral subtraction applied)
        assert not np.array_equal(cleaned, audio)

    def test_get_stats(self):
        gate = AECGate()
        gate.set_ai_speaking(True)
        gate.process(np.zeros(320, dtype=np.float32))
        stats = gate.get_stats()
        assert stats["ai_speaking"] is True
        assert stats["chunks_processed"] == 1
        assert stats["chunks_suppressed"] == 1
        assert "suppression_rate" in stats
        assert stats["post_stop_buffer_ms"] == POST_STOP_BUFFER_MS

    def test_reset(self):
        gate = AECGate()
        gate.set_ai_speaking(True)
        gate.process(np.zeros(320, dtype=np.float32))
        gate.reset()
        assert gate._ai_speaking is False
        assert gate._ai_started_at is None
        assert gate._ai_stopped_at is None

    def test_process_increments_counter(self):
        gate = AECGate()
        for _ in range(5):
            gate.process(np.zeros(320, dtype=np.float32))
        assert gate.chunks_processed == 5

    def test_output_shape_matches_input(self):
        gate = AECGate()
        audio = np.random.randn(480).astype(np.float32)
        cleaned, _ = gate.process(audio)
        assert cleaned.shape == audio.shape

    def test_light_cleanup_outside_gate_with_reference(self):
        """Outside the gate window, with reference, light spectral subtraction applies."""
        gate = AECGate()
        ref = np.random.randn(320).astype(np.float32) * 0.5
        gate.push_reference(ref)
        # Not speaking, no tail → outside gate
        audio = np.random.randn(320).astype(np.float32)
        cleaned, suppressed = gate.process(audio)
        assert suppressed is False
        # Light cleanup should still modify audio slightly
        # (strength=0.25 spectral subtraction)
