import os
import math
from typing import Optional, List, Dict, Any

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa

from audio_classifier.config import Config


tf.get_logger().setLevel("ERROR")


def _clean_label(raw: str) -> str:
    parts = raw.split(",")
    return parts[-1].strip() if parts else raw.strip()


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class YamnetPredictor:
    """
    Cải tiến predictor:
    - Dùng YAMNet trích embedding (1024-D)
    - Nếu có ANN classifier (bird_mouse_classifier.h5) -> dùng để dự đoán Bird/Mouse/Other
    - Nếu không, fallback về heuristic hiện tại
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("🔹 Loading YAMNet model from:", Config.MODEL_URL)
            cls._instance = super(YamnetPredictor, cls).__new__(cls)
            cls._instance.model = hub.load(Config.MODEL_URL)

            # Load class map nếu có
            try:
                class_map_path = cls._instance.model.class_map_path().numpy()
                with tf.io.gfile.GFile(class_map_path) as f:
                    raw_names = [ln.strip() for ln in f.readlines()]
                cls._instance.class_names = [_clean_label(x) for x in raw_names]
            except Exception:
                cls._instance.class_names = []

            # Load ANN classifier nếu có
            ann_path = os.path.join(os.path.dirname(__file__), "../../bird_mouse_classifier.h5")
            ann_path = os.path.abspath(ann_path)
            if os.path.exists(ann_path):
                try:
                    cls._instance.ann_model = tf.keras.models.load_model(ann_path)
                    print(f"✅ Loaded ANN classifier: {ann_path}")
                except Exception as e:
                    print(f"⚠️ Không thể load ANN classifier: {e}")
                    cls._instance.ann_model = None
            else:
                print("ℹ️ Chưa có ANN classifier (bird_mouse_classifier.h5), dùng heuristic.")
                cls._instance.ann_model = None
        return cls._instance

    # =============== Helper functions ===============

    @staticmethod
    def _ensure_waveform(waveform: Optional[np.ndarray], file_path: Optional[str]) -> (np.ndarray, int):
        if waveform is not None:
            return np.asarray(waveform, dtype=np.float32).flatten(), Config.SAMPLE_RATE
        if file_path is None:
            raise ValueError("Cần waveform hoặc file_path")
        wav, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE, mono=True)
        return wav.astype(np.float32).flatten(), sr

    @staticmethod
    def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return waveform
        return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)

    @staticmethod
    def _extract_features(waveform: np.ndarray, sr: int) -> Dict[str, float]:
        y = waveform.astype(float)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return {
            "centroid_mean_hz": float(np.mean(centroid)),
            "rolloff_mean_hz": float(np.mean(rolloff)),
            "zcr_mean": float(np.mean(zcr)),
            "mfcc1_mean": float(np.mean(mfcc[0])),
        }

    # =============== Heuristic scoring ===============

    def _is_yamnet_bird(self, yam_top_labels: List[str]) -> float:
        bird_keywords = ["bird", "chirp", "chatter", "tweet", "avian", "sparrow", "rooster", "crow", "canary"]
        score = 0.0
        for i, lbl in enumerate(yam_top_labels):
            lbl_l = lbl.lower()
            for kw in bird_keywords:
                if kw in lbl_l:
                    score = max(score, 1.0 - 0.1 * i)
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

    # =============== Main Predict Function ===============

    def predict(self, waveform: Optional[np.ndarray] = None, file_path: Optional[str] = None, top_k: int = 3) -> Dict[str, Any]:
        wav, orig_sr = self._ensure_waveform(waveform, file_path)
        yam_wave = self._resample(wav, orig_sr, 16000)

        # Run YAMNet
        scores, embeddings, spectrogram = self.model(yam_wave)
        scores = scores.numpy()
        mean_scores = np.mean(scores, axis=0)
        embedding_mean = np.mean(embeddings.numpy(), axis=0)  # shape (1024,)

        top_indices = np.argsort(mean_scores)[::-1][:top_k]
        top_results, yam_top_labels, yam_top_confidences = [], [], []
        for idx in top_indices:
            label = self.class_names[idx] if idx < len(self.class_names) else str(idx)
            conf = float(mean_scores[idx])
            yam_top_labels.append(label)
            yam_top_confidences.append(conf)
            top_results.append({"label": label, "confidence": conf})

        features = self._extract_features(wav, sr=orig_sr)

        # ========== Nếu có ANN classifier ==========
        if self.ann_model is not None:
            # Chuẩn hóa embedding về 2D input
            emb_input = np.expand_dims(embedding_mean, axis=0)
            pred = self.ann_model.predict(emb_input, verbose=0)[0]
            labels = ["Bird", "Mouse", "Other"]
            result = {
                "top_results": top_results,
                "yamnet_top_labels": yam_top_labels,
                "category_scores": dict(zip(labels, map(float, pred))),
                "final_category": labels[int(np.argmax(pred))],
                "features": features,
                "used_model": "ANN+YAMNet"
            }
            return result

        # ========== Nếu KHÔNG có ANN (fallback heuristic) ==========
        yam_bird_score = self._is_yamnet_bird(yam_top_labels) * max(yam_top_confidences)
        yam_mouse_score = self._is_yamnet_mouse(yam_top_labels) * max(yam_top_confidences)

        centroid = features["centroid_mean_hz"]
        zcr = features["zcr_mean"]
        bird_feature_score = _sigmoid((centroid - 2500.0) / 1500.0) * 0.9 + (_sigmoid((zcr - 0.04) * 50.0) * 0.1)
        mouse_feature_score = _sigmoid((centroid - 5000.0) / 1200.0) * 0.9 + (_sigmoid((zcr - 0.08) * 50.0) * 0.1)

        combined_bird = 0.65 * yam_bird_score + 0.35 * bird_feature_score
        combined_mouse = 0.7 * yam_mouse_score + 0.3 * mouse_feature_score
        combined_other = 1.0 - max(combined_bird, combined_mouse)

        cat_scores = {"Bird": combined_bird, "Mouse": combined_mouse, "Other": combined_other}
        final_category = max(cat_scores.items(), key=lambda x: x[1])[0]

        result = {
            "top_results": top_results,
            "yamnet_top_labels": yam_top_labels,
            "features": features,
            "category_scores": cat_scores,
            "final_category": final_category,
            "used_model": "Heuristic+YAMNet"
        }
        return result

    def predict_file(self, file_path: str, **kwargs):
        return self.predict(file_path=file_path, **kwargs)
