# audio_classifier/core/ml_models/predictor.py
import os
import math
from typing import Optional, List, Dict, Any

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa

from audio_classifier.config import Config

# Reduce TF logging noise for clarity
tf.get_logger().setLevel("ERROR")


def _clean_label(raw: str) -> str:
    """Lấy phần nhãn đọc được từ class_map (vd: '66,/t/dd00013,Children playing' -> 'Children playing')."""
    parts = raw.split(",")
    return parts[-1].strip() if parts else raw.strip()


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class YamnetPredictor:
    """
    Cải tiến predictor:
    - Chạy YAMNet (cần waveform 16k)
    - Trích feature (centroid, rolloff, zcr, mfcc)
    - Kết hợp YAMNet + heuristic để phân loại Bird / Mouse / Other
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("🔹 Loading YAMNet model from:", Config.MODEL_URL)
            cls._instance = super(YamnetPredictor, cls).__new__(cls)
            cls._instance.model = hub.load(Config.MODEL_URL)
            # load class map if possible
            try:
                class_map_path = cls._instance.model.class_map_path().numpy()
                with tf.io.gfile.GFile(class_map_path) as f:
                    raw_names = [ln.strip() for ln in f.readlines()]
                cls._instance.class_names = [_clean_label(x) for x in raw_names]
            except Exception:
                # fallback if not available
                cls._instance.class_names = []
        return cls._instance

    @staticmethod
    def _ensure_waveform(waveform: Optional[np.ndarray], file_path: Optional[str]) -> (np.ndarray, int): # type: ignore
        """Trả về waveform 1D và sampling rate (orig_sr). Nếu truyền file_path -> load bằng librosa."""
        if waveform is not None:
            # assume it's already 1D numpy
            return np.asarray(waveform, dtype=np.float32).flatten(), Config.SAMPLE_RATE
        if file_path is None:
            raise ValueError("Cần waveform hoặc file_path")
        # load with librosa at Config.SAMPLE_RATE (mặc định project dùng 16k; nhưng BirdCLEF notebook dùng 32k)
        # ta load ở native Config.SAMPLE_RATE để giữ tương thích
        wav, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE, mono=True)
        return wav.astype(np.float32).flatten(), sr

    @staticmethod
    def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return waveform
        return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)

    @staticmethod
    def _extract_features(waveform: np.ndarray, sr: int) -> Dict[str, float]:
        """Tính các đặc trưng cơ bản dùng cho heuristic."""
        # ensure float
        y = waveform.astype(float)
        # spectral centroid (Hz)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = float(np.mean(centroid)) if centroid.size else 0.0
        # spectral rolloff (Hz)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = float(np.mean(rolloff)) if rolloff.size else 0.0
        # zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = float(np.mean(zcr)) if zcr.size else 0.0
        # MFCC (1st coefficient mean)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc1 = float(np.mean(mfcc[0])) if mfcc.size else 0.0
        # bandwidth / spectral contrast could be added
        return {
            "centroid_mean_hz": centroid_mean,
            "rolloff_mean_hz": rolloff_mean,
            "zcr_mean": zcr_mean,
            "mfcc1_mean": mfcc1
        }

    def _is_yamnet_bird(self, yam_top_labels: List[str]) -> float:
        """Trả về confidence-like score (0..1) nếu YAMNet top labels liên quan đến chim."""
        bird_keywords = ["bird", "chirp", "chatter", "tweet", "avian", "sparrow", "rooster", "crow", "canary"]
        score = 0.0
        for i, lbl in enumerate(yam_top_labels):
            lbl_l = lbl.lower()
            for kw in bird_keywords:
                if kw in lbl_l:
                    score = max(score, 1.0 - 0.1 * i)  # top label ảnh hưởng mạnh hơn
        return score

    def _is_yamnet_mouse(self, yam_top_labels: List[str]) -> float:
        mouse_keywords = ["mouse", "squeak", "rodent"]
        score = 0.0
        for i, lbl in enumerate(yam_top_labels):
            lbl_l = lbl.lower()
            for kw in mouse_keywords:
                if kw in lbl_l:
                    score = max(score, 1.0 - 0.15 * i)
        return score

    def predict(
        self,
        waveform: Optional[np.ndarray] = None,
        file_path: Optional[str] = None,
        top_k: int = 3,
        smoothing_window: int = 3,
        return_features: bool = True
    ) -> Dict[str, Any]:
        """
        Args:
            waveform: np.ndarray (mono) sampled at Config.SAMPLE_RATE preferably
            file_path: nếu không có waveform -> load từ file
            top_k: số lượng YAMNet top labels trả về
            smoothing_window: hiện chưa dùng phức tạp; reserved
            return_features: có trả feature để debug hay không
        Returns:
            dict chứa top_results (YAMNet), features, final_category và category_scores
        """
        # 1) Load waveform (1D) ở Config.SAMPLE_RATE
        wav, orig_sr = self._ensure_waveform(waveform, file_path)

        # 2) Prepare waveform for YAMNet: YAMNet expects 16 kHz
        yamnet_sr = 16000
        yam_wave = self._resample(wav, orig_sr, yamnet_sr)

        # 3) Run YAMNet
        try:
            scores, embeddings, spectrogram = self.model(yam_wave)
            scores = scores.numpy()  # shape (frames, classes)
            # mean scores across frames
            mean_scores = np.mean(scores, axis=0)
        except Exception as ex:
            raise RuntimeError(f"YAMNet inference lỗi: {ex}")

        # top-k from YAMNet
        top_indices = np.argsort(mean_scores)[::-1][:top_k]
        top_results = []
        yam_top_labels = []
        yam_top_confidences = []
        for idx in top_indices:
            label = self.class_names[idx] if idx < len(self.class_names) else str(idx)
            conf = float(mean_scores[idx])
            yam_top_labels.append(label)
            yam_top_confidences.append(conf)
            top_results.append({"label": label, "confidence": conf})

        # 4) Extract audio features on original sampling rate (use orig_sr)
        features = self._extract_features(wav, sr=orig_sr)

        # 5) Heuristic scoring
        # YAMNet-based cues
        yam_bird_score = self._is_yamnet_bird(yam_top_labels) * max(yam_top_confidences) if yam_top_confidences else 0.0
        yam_mouse_score = self._is_yamnet_mouse(yam_top_labels) * max(yam_top_confidences) if yam_top_confidences else 0.0

        # feature-based cues (normalized via sigmoid)
        # centroid typical ranges: human speech ~1000-3000, bird often >2000 depending on recording
        centroid = features["centroid_mean_hz"]
        zcr = features["zcr_mean"]
        rolloff = features["rolloff_mean_hz"]

        # craft normalized feature scores (heuristic)
        bird_feature_score = _sigmoid((centroid - 2500.0) / 1500.0) * 0.9 + (_sigmoid((zcr - 0.04) * 50.0) * 0.1)
        mouse_feature_score = _sigmoid((centroid - 5000.0) / 1200.0) * 0.9 + (_sigmoid((zcr - 0.08) * 50.0) * 0.1)

        # combine YAMNet and feature signals
        combined_bird = 0.65 * yam_bird_score + 0.35 * bird_feature_score
        combined_mouse = 0.7 * yam_mouse_score + 0.3 * mouse_feature_score

        # also create "other" score as leftover (if none strong)
        combined_other = 1.0 - max(combined_bird, combined_mouse)

        # decide final category
        cat_scores = {"Bird": combined_bird, "Mouse": combined_mouse, "Other": combined_other}
        final_category = max(cat_scores.items(), key=lambda x: x[1])[0]

        # format outputs
        result = {
            "top_results": top_results,
            "yamnet_top_labels": yam_top_labels,
            "features": features if return_features else None,
            "category_scores": cat_scores,
            "final_category": final_category
        }
        return result

    def predict_file(self, file_path: str, **kwargs):
        return self.predict(file_path=file_path, **kwargs)
