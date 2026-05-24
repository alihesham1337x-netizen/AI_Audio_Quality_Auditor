import csv
import os
from typing import List, Dict

import numpy as np

try:
    import tensorflow as tf
    import tensorflow_hub as hub
except ImportError:
    tf = None
    hub = None

YAMNET_MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_LABELS_PATH = os.path.join(os.path.dirname(__file__), "yamnet_class_map.csv")
DEFAULT_LABELS = ["Speech", "Dog Barking", "Music", "Television", "Vehicle", "Construction", "Crowd", "Silence", "Other Noise"]

TARGET_CATEGORY_MAP = {
    "speech": "Speech",
    "child speech": "Speech",
    "conversation": "Speech",
    "narration": "Speech",
    "babbling": "Speech",
    "shout": "Speech",
    "yell": "Speech",
    "whispering": "Speech",
    "laughter": "Speech",
    "dog": "Dog Barking",
    "bark": "Dog Barking",
    "vehicle": "Vehicle",
    "car": "Vehicle",
    "truck": "Vehicle",
    "bus": "Vehicle",
    "motorcycle": "Vehicle",
    "engine": "Vehicle",
    "music": "Music",
    "television": "Television",
    "tv": "Television",
    "construction": "Construction",
    "jackhammer": "Construction",
    "drilling": "Construction",
    "crowd": "Crowd",
    "cheering": "Crowd",
    "screaming": "Crowd",
    "children shouting": "Crowd",
    "baby cry": "Child screaming",
    "crying": "Child screaming",
    "glass": "Other Noise",
    "keyboard": "Keyboard Smash",
    "typing": "Keyboard Smash",
    "door knock": "Other Noise",
    "doorbell": "Other Noise",
    "alarm": "Other Noise",
    "silence": "Silence",
}


def load_yamnet_labels(path: str) -> List[str]:
    labels = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(row["display_name"])
    except Exception:
        pass
    return labels


class EnvironmentalClassifier:
    def __init__(self):
        self.model = None
        self.labels = []
        self.label_map = TARGET_CATEGORY_MAP
        self._load_model()

    def _load_model(self):
        if tf is None or hub is None:
            return
        try:
            self.model = hub.load(YAMNET_MODEL_HANDLE)
            self.labels = load_yamnet_labels(YAMNET_LABELS_PATH)
        except Exception:
            self.model = None
            self.labels = []

    def classify_segment(self, audio: np.ndarray, sample_rate: int):
        if self.model is not None and self.labels:
            return self._yamnet_classify(audio, sample_rate)
        return self._heuristic_label(audio, sample_rate)

    def _yamnet_classify(self, audio: np.ndarray, sample_rate: int):
        waveform = audio.astype(np.float32)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1).astype(np.float32)

        try:
            scores, embeddings, spectrogram = self.model(waveform)
            if isinstance(scores, tf.Tensor):
                scores = scores.numpy()
            mean_scores = np.mean(scores, axis=0)
            top_indices = list(np.argsort(mean_scores)[::-1][:5])
            for index in top_indices:
                if index < len(self.labels):
                    label_name = self.labels[index].lower()
                    mapped = self._map_to_target_category(label_name)
                    if mapped is not None:
                        return mapped, float(mean_scores[index])
            best_index = top_indices[0]
            label_name = self.labels[best_index] if best_index < len(self.labels) else "Unknown"
            return self._map_to_target_category(label_name.lower()) or "Other Noise", float(mean_scores[best_index])
        except Exception:
            return self._heuristic_label(audio, sample_rate)

    def _map_to_target_category(self, label_name: str):
        for keyword, category in self.label_map.items():
            if keyword in label_name:
                return category
        return None

    def _heuristic_label(self, audio: np.ndarray, sample_rate: int):
        rms = np.sqrt(np.mean(np.square(audio)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))))
        spectrum = np.abs(np.fft.rfft(audio))
        centroid = float(np.sum(spectrum * np.arange(len(spectrum))) / (np.sum(spectrum) + 1e-9))
        if rms < 0.002:
            return "Silence", 0.9
        if zcr > 0.15 and rms > 0.02:
            return "Keyboard Smash", 0.75
        if centroid > 4000 and rms > 0.03:
            return "Crowd", 0.65
        if rms > 0.05:
            return "Other Noise", 0.55
        return "Speech", 0.35
