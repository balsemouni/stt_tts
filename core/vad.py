# agent/vad.py

import torch
import numpy as np

from agent.dsp.agc import SimpleAGC
from agent.dsp.deepfilter import DeepFilterNoiseReducer  
from agent.dsp.buffer import SpeechBuffer


class VoiceActivityDetector:
    """
    AGC → DeepFilterNet → Silero VAD → SpeechBuffer
    Simple barge-in: detect user voice even when AI is speaking
    """

    def __init__(
        self,
        sample_rate=16000,
        device="cpu",
        idle_threshold=0.40,
        barge_in_threshold=0.80,
        min_rms=0.012,
        silence_limit_ms=800,
        enable_noise_reduction=True,  # Option to disable for faster processing
    ):
        self.sample_rate = sample_rate
        self.device = device
        self.silence_limit_ms = silence_limit_ms
        self.enable_noise_reduction = enable_noise_reduction

        # --- Silero VAD ---
        print("Loading Silero VAD model...")
        self.vad_model, _ = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            force_reload=False
        )
        self.vad_model.to(self.device).eval()
        print("✅ VAD model loaded")

        # --- DSP chain ---
        self.agc = SimpleAGC()
        
        # DeepFilterNet noise reduction
        if self.enable_noise_reduction:
            self.denoiser = DeepFilterNoiseReducer(
                sample_rate=sample_rate,
                device=device
            )
        else:
            self.denoiser = None
            print("⚠️ Noise reduction disabled")
        
        self.buffer = SpeechBuffer(sample_rate)

        # --- Thresholds ---
        self.idle_threshold = idle_threshold
        self.barge_in_threshold = barge_in_threshold
        self.min_rms = min_rms

    @staticmethod
    def rms(audio: np.ndarray) -> float:
        """Calculate RMS (Root Mean Square) energy of audio"""
        return np.sqrt(np.mean(audio ** 2) + 1e-8)

    def process_chunk(self, audio_chunk: np.ndarray, ai_is_speaking=False):
        """
        Process audio chunk and detect voice activity.
        
        Args:
            audio_chunk: Input audio numpy array
            ai_is_speaking: Whether the AI is currently speaking (for barge-in)
            
        Returns:
            segment (np.ndarray | None): Complete speech segment when detected
            is_voice (bool): Whether voice is detected in this chunk
            prob (float): VAD probability
            rms_val (float): RMS volume level
        """

        # 1) AGC - Automatic Gain Control
        audio = self.agc.process(audio_chunk)

        # 2) Deep learning noise reduction (DeepFilterNet)
        if self.enable_noise_reduction and self.denoiser is not None:
            audio = self.denoiser.process(audio)

        # 3) Silero VAD (requires float32)
        tensor = torch.from_numpy(audio).to(
            device=self.device,
            dtype=torch.float32
        )

        with torch.no_grad():
            prob = self.vad_model(tensor, self.sample_rate).item()

        rms_val = self.rms(audio)

        # 4) Decision logic - use lower threshold when AI is speaking (barge-in)
        threshold = (
            self.barge_in_threshold
            if ai_is_speaking
            else self.idle_threshold
        )
        is_voice = prob > threshold and rms_val > self.min_rms

        # 5) Buffering (pre/post-roll handled internally)
        segment = self.buffer.push(audio, is_voice)

        return segment, is_voice, prob, rms_val
    
    def toggle_noise_reduction(self, enabled: bool):
        """Enable or disable noise reduction on the fly"""
        self.enable_noise_reduction = enabled
        print(f"🔧 Noise reduction {'enabled' if enabled else 'disabled'}")