import os
import uuid
from audio_classifier.config import Config

def save_temp_file(file):
    """Lưu file upload tạm thời vào thư mục temp."""
    os.makedirs(Config.TEMP_DIR, exist_ok=True)
    file_path = os.path.join(Config.TEMP_DIR, f"{uuid.uuid4()}.wav")
    with open(file_path, "wb") as f:
        f.write(file)
    return file_path
