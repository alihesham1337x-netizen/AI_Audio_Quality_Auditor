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

TARGET_CATEGORY_MAP = {
    # ── Speech (never flagged) ──────────────────────────────────────────────
    "speech": "Speech",
    "child speech": "Speech",
    "conversation": "Speech",
    "narration": "Speech",
    "babbling": "Speech",
    "shout": "Speech",
    "yell": "Speech",
    "whispering": "Speech",
    "laughter": "Speech",
    # ── Dog / animal ────────────────────────────────────────────────────────
    "dog": "Dog Barking",
    "bark": "Dog Barking",
    "animal": "Dog Barking",
    "cat": "Dog Barking",
    # ── Background chatter / crowd ──────────────────────────────────────────
    "crowd": "Background Chatter",
    "cheering": "Background Chatter",
    "chatter": "Background Chatter",
    "hubbub": "Background Chatter",
    "children shouting": "Background Chatter",
    "screaming": "Background Chatter",
    "baby cry": "Background Chatter",
    "crying": "Background Chatter",
    # ── Microphone / audio artifacts ────────────────────────────────────────
    "static": "Mic Static",
    "noise": "Mic Static",
    "hiss": "Mic Static",
    "hum": "Mic Static",
    "buzz": "Mic Static",
    "echo": "Mic Static",
    "feedback": "Mic Static",
    "distortion": "Mic Static",
    "white noise": "Mic Static",
    "pink noise": "Mic Static",
    # ── Hits / bangs / impacts ──────────────────────────────────────────────
    "bang": "Impact Noise",
    "knock": "Impact Noise",
    "thud": "Impact Noise",
    "slam": "Impact Noise",
    "tap": "Impact Noise",
    "click": "Impact Noise",
    "clap": "Impact Noise",
    "door": "Impact Noise",
    "glass": "Impact Noise",
    "drum": "Impact Noise",
    "percussion": "Impact Noise",
    "gunshot": "Impact Noise",
    "explosion": "Impact Noise",
    # ── Fan / air / HVAC ────────────────────────────────────────────────────
    "wind": "Fan/Air Noise",
    "fan": "Fan/Air Noise",
    "air conditioning": "Fan/Air Noise",
    "hvac": "Fan/Air Noise",
    "ventilation": "Fan/Air Noise",
    "blowing": "Fan/Air Noise",
    "whoosh": "Fan/Air Noise",
    # ── Music / TV / media ──────────────────────────────────────────────────
    "music": "Music/TV",
    "television": "Music/TV",
    "tv": "Music/TV",
    "radio": "Music/TV",
    "song": "Music/TV",
    "singing": "Music/TV",
    # ── Vehicle / outdoor ───────────────────────────────────────────────────
    "vehicle": "Vehicle Noise",
    "car": "Vehicle Noise",
    "truck": "Vehicle Noise",
    "bus": "Vehicle Noise",
    "motorcycle": "Vehicle Noise",
    "engine": "Vehicle Noise",
    "traffic": "Vehicle Noise",
    "horn": "Vehicle Noise",
    # ── Construction / machinery ────────────────────────────────────────────
    "construction": "Construction",
    "jackhammer": "Construction",
    "drilling": "Construction",
    "saw": "Construction",
    "machinery": "Construction",
    # ── Keyboard / typing ───────────────────────────────────────────────────
    "keyboard": "Keyboard/Typing",
    "typing": "Keyboard/Typing",
    # ── Silence ─────────────────────────────────────────────────────────────
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
        """
        Heuristic classifier using acoustic features.
        Targets: Dog Barking, Background Chatter, Mic Static, Impact Noise,
                 Fan/Air Noise, Music/TV, Vehicle Noise, Construction, Keyboard/Typing.
        Avoids: normal speech (single speaker, tonal, mid-frequency).
        """
        rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))))

        # Spectral analysis
        n_fft = min(len(audio), 2048)
        spectrum = np.abs(np.fft.rfft(audio, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        total_power = np.sum(spectrum ** 2) + 1e-12

        # Frequency band energies
        sub_bass  = float(np.sum(spectrum[freqs < 150]  ** 2) / total_power)
        low_mid   = float(np.sum(spectrum[(freqs >= 150) & (freqs < 500)]  ** 2) / total_power)
        speech_b  = float(np.sum(spectrum[(freqs >= 500) & (freqs < 3000)] ** 2) / total_power)
        high_freq = float(np.sum(spectrum[freqs >= 3000] ** 2) / total_power)

        # Spectral centroid
        centroid = float(np.sum(spectrum * freqs) / (np.sum(spectrum) + 1e-9))

        # Spectral flatness (0=tonal/speech, 1=noise)
        log_mean   = float(np.mean(np.log(spectrum + 1e-12)))
        arith_mean = float(np.mean(spectrum) + 1e-12)
        flatness   = float(np.clip(np.exp(log_mean) / arith_mean, 0.0, 1.0))

        # Spectral bandwidth
        bandwidth = float(np.sqrt(
            np.sum(spectrum ** 2 * (freqs - centroid) ** 2) / total_power
        ))

        dur = len(audio) / sample_rate

        # ── Silence ──────────────────────────────────────────────────────────
        if rms < 0.003:
            return "Silence", 0.92

        # ── Speech guard — exit early if this looks like normal speech ────────
        # Normal speech: energy concentrated in 500–3000Hz (speech_b > 0.55),
        # low flatness (tonal), moderate ZCR, centroid 600–1800Hz.
        # If these all hold, return Speech immediately — don't try to classify
        # as noise. This prevents speech from matching noise rules below.
        is_speech_like = (
            speech_b > 0.55
            and flatness < 0.25
            and 500 < centroid < 2000
            and zcr < 0.18
        )
        if is_speech_like:
            return "Speech", max(0.35, min(0.60, 0.42 + rms * 1.5))

        # ── Impact Noise: hits, bangs, door slams ────────────────────────────
        # Sharp transient: strong sub-bass + low-mid, short duration,
        # low ZCR (not speech-like), low flatness (impulsive not broadband)
        if (rms > 0.02
                and sub_bass + low_mid > 0.45
                and zcr < 0.10
                and flatness < 0.12
                and dur < 0.5):
            return "Impact Noise", 0.78

        # ── Dog Barking ───────────────────────────────────────────────────────
        # Dogs bark in 300–2000Hz but with energy ABOVE the speech band too.
        # Key differentiators from speech:
        #   - Higher centroid than speech (barks are sharper/higher pitched)
        #   - Lower speech_b (energy not concentrated in 500–3kHz like speech)
        #   - Moderate flatness (not as tonal as speech)
        #   - Short bursts
        if (rms > 0.025
                and centroid > 1800          # higher than typical speech
                and speech_b < 0.55          # NOT speech-band dominant
                and 0.10 < flatness < 0.40
                and dur < 1.5):
            return "Dog Barking", 0.68

        # ── Fan / Air Noise ───────────────────────────────────────────────────
        # Continuous broadband hiss: high flatness, energy spread across bands,
        # low-to-moderate RMS, low ZCR
        if (flatness > 0.30
                and high_freq > 0.15
                and zcr < 0.18
                and rms < 0.06):
            return "Fan/Air Noise", 0.65

        # ── Mic Static / Echo ─────────────────────────────────────────────────
        # Very flat spectrum (white/pink noise), any RMS
        if flatness > 0.40 and rms > 0.008:
            return "Mic Static", 0.70

        # ── Background Chatter ────────────────────────────────────────────────
        # Multiple voices: wider bandwidth than single speech,
        # higher centroid, higher ZCR from overlapping voices
        if (rms > 0.04
                and bandwidth > 1400
                and centroid > 1800
                and zcr > 0.14):
            return "Background Chatter", 0.65

        # ── Keyboard / Typing ─────────────────────────────────────────────────
        if zcr > 0.28 and rms > 0.020:
            return "Keyboard/Typing", 0.70

        # ── Music / TV ────────────────────────────────────────────────────────
        if centroid > 2200 and bandwidth > 1600 and rms > 0.03:
            return "Music/TV", 0.60

        # ── Vehicle Noise ─────────────────────────────────────────────────────
        # Strong sub-bass + low-mid rumble, sustained, low centroid
        if (sub_bass + low_mid > 0.55
                and rms > 0.025
                and centroid < 1000
                and dur > 0.3):
            return "Vehicle Noise", 0.62

        # ── Construction ──────────────────────────────────────────────────────
        if rms > 0.07 and bandwidth > 1800:
            return "Construction", 0.60

        # ── Default: Speech ───────────────────────────────────────────────────
        return "Speech", max(0.30, min(0.55, 0.40 + rms * 1.5))
