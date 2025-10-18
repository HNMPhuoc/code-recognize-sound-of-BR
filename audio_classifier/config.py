import os

class Config:
    MODEL_URL = "https://tfhub.dev/google/yamnet/1"
    SAMPLE_RATE = 16000
    TEMP_DIR = os.path.join(os.getcwd(), "temp")
