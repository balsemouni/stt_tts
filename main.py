import asyncio
from agent.voice_agent import ProVoiceAgent

if __name__ == "__main__":
    agent = ProVoiceAgent()
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        print("\n⏹️ Session closed")