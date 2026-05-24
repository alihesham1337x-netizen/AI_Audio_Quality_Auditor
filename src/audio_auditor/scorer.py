from typing import Dict

LABEL_SEVERITY = {
    "Dog Barking": 6,
    "Keyboard Smash": 8,
    "Music": 7,
    "Television": 7,
    "Vehicle": 7,
    "Construction": 8,
    "Crowd": 6,
    "Silence": 1,
    "Other Noise": 5,
    "Speech": 1,
}


class SeverityScorer:
    def score_event(self, label: str, confidence: float, features: Dict[str, float]) -> float:
        base = LABEL_SEVERITY.get(label, 4)
        loudness = min(max(features.get("rms_mean", 0.0) * 100.0, 0.0), 10.0)
        duration = min(max(features.get("duration_s", 0.0), 0.0), 5.0)
        score = base * (0.5 + confidence * 0.5) + loudness * 0.4 + duration * 0.2
        return min(max(score, 1.0), 10.0)
