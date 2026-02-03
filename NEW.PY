import os
import torch
import numpy as np
import queue
import time
import asyncio
import threading
import sounddevice as sd
from faster_whisper import WhisperModel
import edge_tts
import pygame
import io
import sys
from collections import deque
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# --- 1. MEMORY MANAGEMENT ---
class ConversationMemory:
    def __init__(self, limit=8):
        self.history = deque(maxlen=limit)
        self.system_prompt = (
            "You are a helpful, witty AI voice assistant. "
            "Keep responses very short (1-2 sentences) and conversational."
        )

    def add_exchange(self, user, assistant):
        self.history.append({"role": "user", "content": user})
        self.history.append({"role": "assistant", "content": assistant})

    def get_messages(self):
        msgs = [{"role": "system", "content": self.system_prompt}]
        msgs.extend(list(self.history))
        return msgs

# --- 2. THE PRO VOICE AGENT ---
class ProVoiceAgent:
    def __init__(self):
        # Configuration - Replace with your key
        self.API_KEY = "" 
        self.SAMPLE_RATE = 16000
        self.CHUNK_SIZE = 512 
        
        # VAD & Sensitivity
        self.IDLE_VAD_THRESHOLD = 0.40     
        self.BARGE_IN_VAD_THRESHOLD = 0.80 # Strict when AI is talking
        self.MIN_VOLUME_RMS = 0.010        
        self.SILENCE_LIMIT_MS = 600        
        
        # Device Setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing AI on {self.device.upper()}...")
        
        # Load Models
        self.asr = WhisperModel("base.en", device=self.device, compute_type="float16" if self.device=="cuda" else "int8")
        self.vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        self.vad_model.to(self.device)
        self.llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", nvidia_api_key=self.API_KEY)
        self.memory = ConversationMemory()
        
        # Audio Player Init
        pygame.mixer.init(frequency=24000)
        
        # State Control
        self.audio_queue = queue.Queue()
        self.speech_buffer = []
        self.is_speaking = False      
        self.user_is_talking = False  
        self.interrupt_event = asyncio.Event() 
        self.process_audio_event = None 
        self.loop = None

    def mic_callback(self, indata, frames, time, status):
        """Continuous mic capture"""
        if status: print(status)
        self.audio_queue.put(indata.copy())

    def get_rms(self, audio_chunk):
        return np.sqrt(np.mean(audio_chunk**2))

    def vad_monitor_thread(self):
        """Background loop that monitors for voice and triggers barge-in"""
        silence_counter = 0
        is_recording = False

        while True:
            try:
                chunk = self.audio_queue.get(timeout=1).flatten()
            except queue.Empty:
                continue

            # Voice Probability
            tensor_seg = torch.from_numpy(chunk).to(self.device)
            with torch.no_grad():
                prob = self.vad_model(tensor_seg, self.SAMPLE_RATE).item()
            
            rms = self.get_rms(chunk)

            # Determine if we should listen
            # If AI is speaking, we need a higher probability to "Barge In"
            current_threshold = self.BARGE_IN_VAD_THRESHOLD if self.is_speaking else self.IDLE_VAD_THRESHOLD

            if prob > current_threshold and rms > self.MIN_VOLUME_RMS:
                silence_counter = 0
                
                if not is_recording:
                    # --- BARGE-IN TRIGGER ---
                    if self.is_speaking:
                        self.loop.call_soon_threadsafe(self.trigger_interrupt)
                    
                    is_recording = True
                    self.user_is_talking = True
                    self.speech_buffer = [] 

            elif is_recording:
                silence_counter += (len(chunk) / self.SAMPLE_RATE) * 1000
                if silence_counter > self.SILENCE_LIMIT_MS:
                    is_recording = False
                    self.user_is_talking = False
                    if self.loop:
                        self.loop.call_soon_threadsafe(self.process_audio_event.set)

            # Only record audio if it's the user speaking (not the AI's echo)
            if is_recording and not self.is_speaking:
                self.speech_buffer.append(chunk)

            self.draw_ui(prob, rms)

    def trigger_interrupt(self):
        """Immediately stops the AI voice and cleans up"""
        if self.is_speaking:
            pygame.mixer.music.stop() # Instant audio stop
            pygame.mixer.music.unload()
            self.interrupt_event.set() # Stop the TTS stream loop
            self.is_speaking = False
            self.speech_buffer = [] 
            sys.stdout.write("\n🛑 [INTERRUPT] User is speaking. AI stopped.\n")

    def draw_ui(self, prob, rms):
        status = "🎤 USER" if self.user_is_talking else "🤖 AI  " if self.is_speaking else "👂 LIST"
        bar = "█" * int(prob * 10) + "-" * (10 - int(prob * 10))
        sys.stdout.write(f"\r{status} | VAD: [{bar}] {prob:.2f} | Vol: {rms:.4f}  ")
        sys.stdout.flush()

    async def speak(self, text):
        """Edge-TTS to Pygame with interruption support"""
        if not text: return
        self.interrupt_event.clear()
        self.is_speaking = True
        
        try:
            communicate = edge_tts.Communicate(text, "en-US-AndrewNeural")
            audio_data = b""
            
            # Stream audio data
            async for chunk in communicate.stream():
                if self.interrupt_event.is_set(): 
                    return # Exit if user interrupted during generation
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            # Play audio
            f = io.BytesIO(audio_data)
            pygame.mixer.music.load(f)
            pygame.mixer.music.play()
            
            # Wait for playback to finish, but allow interruption
            while pygame.mixer.music.get_busy():
                if self.interrupt_event.is_set():
                    pygame.mixer.music.stop()
                    break
                await asyncio.sleep(0.05)
                
        except Exception as e:
            print(f"\nTTS Error: {e}")
        finally:
            self.is_speaking = False

    async def main_loop(self):
        self.loop = asyncio.get_running_loop()
        self.process_audio_event = asyncio.Event()
        
        # Start VAD background thread
        threading.Thread(target=self.vad_monitor_thread, daemon=True).start()
        
        # Open Microphone
        mic_stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE, 
            channels=1, 
            callback=self.mic_callback, 
            blocksize=self.CHUNK_SIZE, 
            dtype='float32'
        )
        
        with mic_stream:
            print(f"\n✅ AI Voice Assistant Online. Talk anytime!")
            while True:
                # 1. Wait for end of user speech
                await self.process_audio_event.wait()
                self.process_audio_event.clear()

                if not self.speech_buffer: continue

                # 2. Transcription
                audio_np = np.concatenate(self.speech_buffer)
                self.speech_buffer = [] 
                
                segments, _ = self.asr.transcribe(audio_np, beam_size=1)
                user_text = " ".join([s.text for s in segments]).strip()
                
                if len(user_text) < 2: continue 
                print(f"\n👤 User: {user_text}")

                # 3. LLM Logic
                try:
                    messages = self.memory.get_messages()
                    messages.append({"role": "user", "content": user_text})
                    
                    response = await self.loop.run_in_executor(
                        None, lambda: self.llm.invoke(messages)
                    )
                    ai_reply = response.content
                    
                    print(f"🤖 AI: {ai_reply}")
                    self.memory.add_exchange(user_text, ai_reply)

                    # 4. Speak with Barge-in capability
                    await self.speak(ai_reply)

                except Exception as e:
                    print(f"\n❌ Pipeline Error: {e}")

if __name__ == "__main__":
    agent = ProVoiceAgent()
    try:
        asyncio.run(agent.main_loop())
    except KeyboardInterrupt:
        print("\n⏹️ Session Closed.")
