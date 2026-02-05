import sounddevice as sd
#Audio data comes in fast. This queue acts as a buffer so you don't lose sound data if the rest of your code is busy for a millisecond.
import queue

class MicrophoneHandler:
    """Handles audio input from microphone"""

    def __init__(self, sample_rate=16000, chunk_size=512):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        #Creates the storage bucket for incoming sound.
        self.audio_queue = queue.Queue()
        self.stream = None

    #The audio hardware calls this automatically every time it has new sound data ready.
    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        #It takes the raw audio data and puts a copy into the queue.
        self.audio_queue.put(indata.copy())
    #sample_rate the number of times per second an analog sound wave is measured (sampled) to convert it into digital data,
    #Configures and opens the microphone stream.
    def start(self):
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            #Records in Mono (standard for voice AI)
            channels=1,
            callback=self.callback,
            blocksize=self.chunk_size,
            dtype="float32",
        )
        self.stream.start()
        return self.stream

    def get_audio_chunk(self, timeout=1):
        try:
            return self.audio_queue.get(timeout=timeout).flatten()
        except queue.Empty:
            return None

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()