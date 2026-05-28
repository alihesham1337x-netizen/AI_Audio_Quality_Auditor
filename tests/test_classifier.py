import numpy as np

from src.audio_auditor.classifier import EnvironmentalClassifier


def test_classifier_silence_heuristic():
    classifier = EnvironmentalClassifier()
    classifier.model = None
    classifier.labels = []

    label, confidence = classifier.classify_segment(np.zeros(1600, dtype=np.float32), 16000)
    assert label == "Silence"
    assert confidence == 0.9


def test_classifier_heuristic_returns_label_and_confidence():
    classifier = EnvironmentalClassifier()
    classifier.model = None
    classifier.labels = []
    audio = np.random.uniform(-0.05, 0.05, 1600).astype(np.float32)

    label, confidence = classifier.classify_segment(audio, 16000)
    assert isinstance(label, str)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0
