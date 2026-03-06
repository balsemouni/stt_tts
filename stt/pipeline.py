"""
pipeline.py — STT Pipeline
───────────────────────────────────────────
Flow per chunk:
  1. AEC  — suppress echo while AI is playing audio
  2. VAD  — Silero voice probability + silence detection
  3. Enrollment — build speaker profile from first 0.5s of voice
  4. ASR  — Whisper runs on accumulated 600ms voice windows (not every tiny chunk)
  5. Flush — emit full segment when silence follows voice

Key rules:
  - Audio is buffered during voice activity until ASR_MIN_SAMPLES (600ms) is
    reached, then forwarded to Whisper.  This prevents Whisper receiving 20ms
    micro-chunks where it returns no words or unreliable timestamps.
  - On silence_event the buffer is flushed immediately before calling
    realtime_asr.flush() so no audio is lost.
  - ASR runs ONLY when is_voice=True. No speculative/noise transcription.
  - Flush fires ONLY on silence_event (first silent chunk after voice ends).
  - Enrollment runs in parallel: profile builds while ASR transcribes.
  - No hallucination filtering here — kept in main.py _filter_segment().
"""

from __future__ import annotations

import os, sys, time, numpy as np, logging
from collections import deque
from typing import List, Dict, Any, Optional

_HERE = os.path.dirname(__file__)
for _c in [_HERE, os.path.join(_HERE, ".."), os.path.join(_HERE, "agent")]:
    if _c not in sys.path:
        sys.path.insert(0, _c)

from vad import VoiceActivityDetector
from realtime_asr import RealTimeChunkASR
from aec_gate import AECGate
from speaker_enrollment import SpeakerEnrollmentService

logger = logging.getLogger(__name__)


class STTPipeline:

    def __init__(
        self,
        sample_rate: int            = 16000,
        device: str | None          = None,
        # VAD
        idle_threshold: float       = 0.15,
        barge_in_threshold: float   = 0.40,
        enable_noise_reduction: bool = False,
        vad_pre_gain: float         = 5.0,
        # ASR
        whisper_model_size: str     = "base.en",
        overlap_seconds: float      = 0.8,
        word_gap_ms: float          = 80.0,
        max_context_words: int      = 10,
        max_history_turns: int      = 3,
        # ASR chunk accumulation — buffer voice audio until this many ms
        # before calling Whisper.  Prevents micro-chunk starvation (was the
        # main reason words never appeared live).
        asr_min_buffer_ms: float    = 600.0,
        # AEC
        enable_aec: bool            = True,
        # Enrollment
        enable_enrollment: bool     = True,
        enroll_min_seconds: float   = 0.5,
        similarity_threshold: float = 0.72,
        # Compat params (ignored)
        ai_detector_model_path=None,
        ai_detection_threshold=0.7,
        enable_ai_filtering=False,
        barge_in_debounce_frames=2,
        barge_in_energy_ratio=2.5,
        barge_in_cooldown_ms=1500.0,
        speculative_vad_threshold=0.08,
    ):
        import torch
        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.sample_rate  = sample_rate
        self._ai_speaking = False
        self._last_is_voice = False
        self._words_this_utterance: List[str] = []

        # ── ASR chunk accumulator ─────────────────────────────────────────
        # Whisper needs at least ~600ms of audio to produce reliable word
        # timestamps.  Accumulate voice chunks here; drain when full or on
        # silence_event.
        self._asr_buffer: List[np.ndarray] = []
        self._asr_buffer_samples: int = 0
        self.ASR_MIN_SAMPLES: int = int(sample_rate * asr_min_buffer_ms / 1000.0)

        self.vad = VoiceActivityDetector(
            sample_rate            = sample_rate,
            device                 = _device,
            idle_threshold         = idle_threshold,
            barge_in_threshold     = barge_in_threshold,
            enable_noise_reduction = enable_noise_reduction,
            pre_gain               = vad_pre_gain,
        )

        self.realtime_asr = RealTimeChunkASR(
            model_size        = whisper_model_size,
            device            = _device,
            sample_rate       = sample_rate,
            overlap_seconds   = overlap_seconds,
            word_gap_ms       = word_gap_ms,
            max_context_words = max_context_words,
            max_history_turns = max_history_turns,
        )

        self.aec = AECGate(sample_rate=sample_rate) if enable_aec else None

        self.enrollment = (
            SpeakerEnrollmentService(
                sample_rate          = sample_rate,
                similarity_threshold = similarity_threshold,
                enroll_min_seconds   = enroll_min_seconds,
            )
            if enable_enrollment else None
        )

    # ── AI state ─────────────────────────────────────────────────────────────

    def notify_ai_speaking(self, speaking: bool):
        self._ai_speaking = speaking
        if self.aec:
            self.aec.set_ai_speaking(speaking)

    def push_ai_reference(self, pcm: np.ndarray):
        if self.aec:
            self.aec.push_reference(pcm)

    def add_assistant_turn(self, text: str):
        self.realtime_asr.add_assistant_turn(text)

    # ── Internal ASR runner ───────────────────────────────────────────────────

    def _run_asr_on_buffer(self, events: List[Dict[str, Any]]):
        """
        Concatenate the accumulated audio buffer, send to Whisper, emit
        word/partial events.  Clears the buffer after inference.
        """
        if not self._asr_buffer:
            return

        combined = np.concatenate(self._asr_buffer)
        self._asr_buffer = []
        self._asr_buffer_samples = 0

        result = self.realtime_asr.transcribe_chunk(combined)

        partial = result.get("partial", "").strip()
        if partial and hasattr(self.vad, "set_partial_text"):
            self.vad.set_partial_text(partial)

        for word in result.get("words", []):
            w = word if isinstance(word, str) else word.get("word", "")
            w = w.strip().strip(".,!?;:")
            if w and any(c.isalpha() for c in w):
                self._words_this_utterance.append(w)
                events.append({"type": "word", "word": w})

        if partial and any(c.isalpha() for c in partial):
            events.append({"type": "partial", "word": partial})

    # ── Main ─────────────────────────────────────────────────────────────────

    def process_chunk(self, audio_chunk: np.ndarray) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        # ── Step 1: AEC ───────────────────────────────────────────────────
        if self.aec:
            cleaned, suppressed = self.aec.process(audio_chunk)
            if suppressed:
                return [{"type": "aec", "suppressed": True}]
        else:
            cleaned = audio_chunk

        # ── Step 2: VAD ───────────────────────────────────────────────────
        audio, is_voice, prob, rms, silence_event = self.vad.process_chunk(
            cleaned, ai_is_speaking=self._ai_speaking
        )

        if is_voice != self._last_is_voice:
            events.append({"type": "vad", "prob": round(prob, 3),
                           "is_voice": is_voice, "rms": round(float(rms), 5)})
            self._last_is_voice = is_voice

        # ── Step 3: Enrollment ────────────────────────────────────────────
        block_asr = False
        if self.enrollment:
            decision = self.enrollment.evaluate(audio, is_voice=is_voice)
            events.append({"type": "enrollment", **decision.to_dict()})

            if decision.enrolled and not decision.send_to_asr:
                block_asr = True
                if silence_event:
                    self._words_this_utterance.clear()
                    self._asr_buffer = []
                    self._asr_buffer_samples = 0
                    events.append({"type": "segment_rejected",
                                   "reason": "speaker_mismatch",
                                   "similarity": decision.similarity})

        # ── Step 4: ASR — buffer voice chunks, run Whisper at 600ms ──────
        #
        # Problem this fixes: Whisper called on tiny 20ms chunks returns
        # 0 words or unreliable timestamps.  We accumulate voice audio
        # until ASR_MIN_SAMPLES (600ms default) then fire one inference
        # call — large enough for Whisper to produce stable word-level
        # timestamps and emit words live (not just at end-of-utterance).
        #
        # silence_event drains whatever is left in the buffer so the last
        # words before a pause are never lost.
        if is_voice and not block_asr:
            self._asr_buffer.append(audio)
            self._asr_buffer_samples += len(audio)

            # Fire ASR once buffer is large enough
            if self._asr_buffer_samples >= self.ASR_MIN_SAMPLES:
                self._run_asr_on_buffer(events)

        # ── Step 5: Silence flush ─────────────────────────────────────────
        # Fire exactly once: on the first silent chunk after voice ends.
        if silence_event and not block_asr:
            # Drain any remaining buffered voice audio before final flush
            if self._asr_buffer:
                self._run_asr_on_buffer(events)

            flush_result = self.realtime_asr.flush()

            for word in flush_result.get("words", []):
                w = word if isinstance(word, str) else word.get("word", "")
                w = w.strip().strip(".,!?;:")
                if w and any(c.isalpha() for c in w):
                    self._words_this_utterance.append(w)

            flush_text = flush_result.get("text", "").strip()
            if flush_text and not self._words_this_utterance:
                for w in flush_text.split():
                    w = w.strip().strip(".,!?;:")
                    if w and any(c.isalpha() for c in w):
                        self._words_this_utterance.append(w)

            if self._words_this_utterance:
                text = " ".join(self._words_this_utterance)
                events.append({"type": "segment", "text": text})
                logger.info(f"[pipeline] segment: {text!r}")

            self._words_this_utterance.clear()

        return events

    # ── REST path ─────────────────────────────────────────────────────────

    def transcribe_full(self, audio: np.ndarray) -> str:
        result  = self.realtime_asr.transcribe_chunk(audio)
        flushed = self.realtime_asr.flush()
        words   = result.get("words", []) + flushed.get("words", [])
        return " ".join(words).strip() or flushed.get("text", "")

    def flush(self) -> str:
        # Drain accumulator before final flush
        if self._asr_buffer:
            dummy_events: List[Dict[str, Any]] = []
            self._run_asr_on_buffer(dummy_events)
            for ev in dummy_events:
                if ev.get("type") == "word":
                    w = ev.get("word", "").strip()
                    if w:
                        self._words_this_utterance.append(w)

        text = self.realtime_asr.flush().get("text", "")
        self._words_this_utterance.clear()
        self._asr_buffer = []
        self._asr_buffer_samples = 0
        return text

    def reset(self):
        self.vad.reset()
        self.realtime_asr.reset()
        self._last_is_voice = False
        self._words_this_utterance.clear()
        self._ai_speaking = False
        # Reset accumulator
        self._asr_buffer = []
        self._asr_buffer_samples = 0
        if self.aec:
            self.aec.reset()
        if self.enrollment:
            self.enrollment.reset()

    def get_stats(self) -> dict:
        stats = {"vad": self.vad.get_state()}
        if self.aec:
            stats["aec"] = self.aec.get_stats()
        if self.enrollment:
            stats["enrollment"] = self.enrollment.get_stats()
        return stats