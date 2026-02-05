import edge_tts
import pygame
import asyncio
import io
import threading

class TextToSpeech:
    """Handles text-to-speech with immediate interrupt capability"""

    def __init__(self, voice="en-US-AndrewNeural"):
        self.voice = voice
        pygame.mixer.init(frequency=24000, channels=1, buffer=512)
        self.interrupt_event = asyncio.Event()
        self.is_speaking = False
        self._lock = threading.Lock()

    async def speak(self, text):
        """
        Speak the given text. Can be interrupted immediately by stop().
        """
        if not text:
            return

        with self._lock:
            self.interrupt_event.clear()
            self.is_speaking = True

        try:
            # Generate audio using Edge TTS
            communicate = edge_tts.Communicate(text, self.voice)
            audio = b""

            # Stream audio chunks - check for interrupt during generation
            async for chunk in communicate.stream():
                if self.interrupt_event.is_set():
                    with self._lock:
                        self.is_speaking = False
                    return
                    
                if chunk["type"] == "audio":
                    audio += chunk["data"]

            # Check again before playback
            if self.interrupt_event.is_set():
                with self._lock:
                    self.is_speaking = False
                return

            # Play the generated audio
            pygame.mixer.music.load(io.BytesIO(audio))
            pygame.mixer.music.play()

            # Monitor playback and check for interrupts frequently
            while pygame.mixer.music.get_busy():
                if self.interrupt_event.is_set():
                    # IMMEDIATE STOP
                    pygame.mixer.music.stop()
                    try:
                        pygame.mixer.music.unload()
                    except:
                        pass
                    break
                await asyncio.sleep(0.02)  # Check every 20ms for fast response

        except Exception as e:
            print(f"\n❌ TTS error: {e}")
        finally:
            with self._lock:
                self.is_speaking = False

    def stop(self):
        """
        Immediately stop TTS - both generation and playback.
        This is called during barge-in to stop the bot's voice.
        """
        with self._lock:
            if self.is_speaking:
                # Stop audio playback IMMEDIATELY
                try:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                except:
                    pass
                
                # Signal to stop generation
                self.interrupt_event.set()
                self.is_speaking = False
    
    def get_speaking_state(self):
        """Thread-safe check if TTS is currently speaking"""
        with self._lock:
            return self.is_speaking