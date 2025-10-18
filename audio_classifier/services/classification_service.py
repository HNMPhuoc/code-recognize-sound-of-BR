from audio_classifier.core.processing.preprocessor import load_audio
from audio_classifier.core.ml_models.predictor import YamnetPredictor
from audio_classifier.core.utils import save_temp_file

def classify_audio(file_bytes: bytes):
    """Pipeline: lưu file -> tiền xử lý -> dự đoán."""
    path = save_temp_file(file_bytes)
    waveform = load_audio(path)
    predictor = YamnetPredictor()
    result = predictor.predict(waveform)
    return result
