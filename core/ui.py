import sys

class UIHandler:
    """Handles console UI for the voice agent"""
    
    @staticmethod
    def draw_status(user_talking, ai_speaking, vad_prob, rms):
        """Draw real-time status bar"""
        if user_talking:
            status = "🎤 USER"
        elif ai_speaking:
            status = "🤖 AI"
        else:
            status = "👂 LISTENING"
            
        bar = "█" * int(vad_prob * 10) + "░" * (10 - int(vad_prob * 10))
        
        sys.stdout.write(
            f"\r{status} | VAD [{bar}] {vad_prob:.2f} | Vol {rms:.4f}  "
        )
        sys.stdout.flush()

    @staticmethod
    def print_user(text):
        """Print user message"""
        print(f"\n👤 User: {text}")

    @staticmethod
    def print_ai(text):
        """Print AI response"""
        print(f"🤖 AI: {text}")

    @staticmethod
    def print_interrupt():
        """Print interruption message"""
        print("\n⚡ [BARGE-IN] User interrupted AI")