# agent/dsp/deepfilter.py

import torch
import numpy as np

try:
    from df.enhance import init_df, enhance
    from df.io import resample
    DEEPFILTER_AVAILABLE = True
except ImportError:
    DEEPFILTER_AVAILABLE = False


class DeepFilterNoiseReducer:
    """Minimal DeepFilterNet noise reducer"""
    
    def __init__(self, sample_rate=16000, device="cpu"):
        self.input_sample_rate = sample_rate
        self.device = device
        self.model = None
        
        if not DEEPFILTER_AVAILABLE:
            print("⚠️ DeepFilterNet not available")
            return
        
        try:
            print("🧠 Loading DeepFilterNet...")
            self.model, self.df_state, _ = init_df(post_filter=True, log_level="ERROR")
            self.model = self.model.to(self.device).eval()
            self.df_sample_rate = self.df_state.sr()
            self.needs_resampling = (self.input_sample_rate != self.df_sample_rate)
            print(f"✅ DeepFilterNet loaded ({self.df_sample_rate}Hz)")
        except Exception as e:
            print(f"❌ DeepFilterNet failed: {e}")
            self.model = None
    
    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Apply noise reduction to audio chunk"""
        if self.model is None:
            return audio_chunk
        
        try:
            # Ensure audio is 1D
            if audio_chunk.ndim > 1:
                audio_chunk = audio_chunk.flatten()
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_chunk).float()
            
            # Resample to 48kHz if needed
            if self.needs_resampling:
                audio_tensor = resample(audio_tensor, self.input_sample_rate, self.df_sample_rate)
            
            audio_tensor = audio_tensor.to(self.device)
            
            # Process: add batch (1) and channel (1) dimensions
            with torch.no_grad():
                audio_input = audio_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
                enhanced = enhance(self.model, self.df_state, audio_input)
                enhanced = enhanced.squeeze(0).squeeze(0)  # Remove dimensions
            
            # Resample back if needed
            if self.needs_resampling:
                enhanced = resample(enhanced.cpu(), self.df_sample_rate, self.input_sample_rate)
                return enhanced.numpy()
            else:
                return enhanced.cpu().numpy()
                
        except Exception as e:
            # Silently return original audio on error
            return audio_chunk
    
    def __call__(self, audio_chunk: np.ndarray) -> np.ndarray:
        return self.process(audio_chunk)