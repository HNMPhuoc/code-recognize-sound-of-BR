import librosa
import numpy as np
from audio_classifier.config import Config

def load_audio(file_path: str):
    """Đọc file âm thanh và resample về 16kHz."""
    waveform, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE, mono=True)
    return waveform.astype(np.float32)
