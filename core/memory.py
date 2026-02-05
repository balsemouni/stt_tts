from collections import deque

class ConversationMemory:
    """Manages conversation history with a rolling window keep last 8 messages"""

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