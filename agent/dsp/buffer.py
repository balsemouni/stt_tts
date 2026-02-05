from collections import deque
import numpy as np

class SpeechBuffer:
    def __init__(self, sample_rate, pre_ms=200, post_ms=400):
        self.pre_frames = int(sample_rate * pre_ms / 1000)
        self.post_frames = int(sample_rate * post_ms / 1000)

        self.pre_buffer = deque(maxlen=self.pre_frames)
        self.post_counter = 0
        self.recording = False
        self.frames = []

    def push(self, frame, is_voice):
        self.pre_buffer.append(frame)

        if is_voice:
            if not self.recording:
                self.frames = list(self.pre_buffer)
                self.recording = True
            self.frames.append(frame)
            self.post_counter = 0

        elif self.recording:
            self.frames.append(frame)
            self.post_counter += len(frame)
            if self.post_counter > self.post_frames:
                self.recording = False
                return np.concatenate(self.frames)

        return None