import os
import torch
import numpy as np
import asyncio
import threading

from core.memory import ConversationMemory
from core.microphone import MicrophoneHandler
from core.vad import VoiceActivityDetector
from core.asr import SpeechRecognizer
from core.llm import LLMHandler
from core.tts import TextToSpeech
from core.ui import UIHandler


class ProVoiceAgent:
    """
    Professional voice agent with barge-in support.
    User can interrupt AI speech at any time.
    """
    
    def __init__(self):
        self.API_KEY = os.getenv("API_KEY")

        self.SAMPLE_RATE = 16000
        self.CHUNK_SIZE = 512

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing on {self.device.upper()}")

        # Initialize components
        self.memory = ConversationMemory()
        self.microphone = MicrophoneHandler(self.SAMPLE_RATE, self.CHUNK_SIZE)
        self.vad = VoiceActivityDetector(
            sample_rate=self.SAMPLE_RATE,
            device=self.device,
        )
        self.asr = SpeechRecognizer("base.en", self.device)
        self.llm = LLMHandler(self.API_KEY)
        self.tts = TextToSpeech()
        self.ui = UIHandler()

        # State tracking
        self.speech_buffer = []
        self.user_is_talking = False
        self.processing_lock = asyncio.Lock()


    def vad_thread(self):
        """
        Main VAD loop running in separate thread.
        Detects user speech and triggers barge-in when needed.
        """
        silence_ms = 0
        recording = False

        while True:
            chunk = self.microphone.get_audio_chunk()
            if chunk is None:
                continue

            # Process audio chunk through VAD pipeline
            ai_speaking = self.tts.get_speaking_state()
            segment, is_voice, prob, rms = self.vad.process_chunk(
                chunk,
                ai_is_speaking=ai_speaking
            )

            if is_voice:
                # User started speaking
                silence_ms = 0
                
                if not recording:
                    # BARGE-IN: Stop AI if user is speaking
                    if ai_speaking:
                        self.tts.stop()
                        self.ui.print_interrupt()
                    
                    # Start new recording
                    recording = True
                    self.speech_buffer.clear()
                    self.user_is_talking = True

            elif recording:
                # Count silence duration
                silence_ms += (len(chunk) / self.SAMPLE_RATE) * 1000
                
                if silence_ms > self.vad.silence_limit_ms:
                    # End of speech detected
                    recording = False
                    self.user_is_talking = False
                    
                    # Process the recorded audio
                    asyncio.run_coroutine_threadsafe(
                        self.process_audio(),
                        self.loop
                    )

            # Add segment to buffer if available
            if segment is not None:
                self.speech_buffer.append(segment)

            # Update UI
            self.ui.draw_status(
                self.user_is_talking,
                ai_speaking,
                prob,
                rms,
            )


    async def process_audio(self):
        """
        Process recorded audio: ASR -> LLM -> TTS
        """
        # Use lock to prevent overlapping processing
        async with self.processing_lock:
            if not self.speech_buffer:
                return

            # Get audio data
            audio = np.concatenate(self.speech_buffer)
            self.speech_buffer.clear()

            # Transcribe speech to text
            text = self.asr.transcribe(audio)
            if not text:
                return

            self.ui.print_user(text)

            # Get LLM response
            msgs = self.memory.get_messages()
            msgs.append({"role": "user", "content": text})

            reply = await self.llm.get_response(msgs, self.loop)

            self.ui.print_ai(reply)
            self.memory.add_exchange(text, reply)

            # Speak response (can be interrupted by barge-in)
            await self.tts.speak(reply)


    async def run(self):
        """
        Main entry point - starts the voice agent
        """
        self.loop = asyncio.get_running_loop()

        # Start VAD thread
        threading.Thread(
            target=self.vad_thread,
            daemon=True
        ).start()

        # Start microphone
        with self.microphone.start():
            print("\n✅ Voice Agent Ready")
            print("💡 Barge-in enabled - you can interrupt the AI anytime!\n")
            
            while True:
                await asyncio.sleep(1)