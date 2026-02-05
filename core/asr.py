from faster_whisper import WhisperModel

class SpeechRecognizer:
    """Transcribes audio to text using Faster Whisper"""

    def __init__(self, model_size="base.en", device="cuda"):
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"Loading Whisper model on {device.upper()}...")

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

    def transcribe(self, audio_data):
        segments, _ = self.model.transcribe(audio_data, beam_size=1)
        return " ".join(s.text for s in segments).strip()