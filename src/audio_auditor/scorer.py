from typing import Dict

LABEL_SEVERITY = {
    "Dog Barking": 7,
    "Keyboard Smash": 8,
    "Music": 7,
    "Television": 6,
    "Vehicle": 6,
    "Construction": 8,
    "Crowd": 6,
    "Child screaming": 7,
    "Silence": 1,
    "Other Noise": 5,
    "Speech": 1,
}


class SeverityScorer:
    def score_event(self, label: str, confidence: float, features: Dict[str, float]) -> float:
        base = LABEL_SEVERITY.get(label, 4)
        # rms_mean after normalization is typically 0.01–0.15.
        # Scale to 0–6 range so loudness meaningfully contributes without
        # dominating the score for quiet events.
        loudness = min(max(features.get("rms_mean", 0.0) * 60.0, 0.0), 6.0)
        # Duration bonus capped at 4 s
        duration = min(max(features.get("duration_s", 0.0), 0.0), 4.0)
        score = base * (0.5 + confidence * 0.5) + loudness * 0.35 + duration * 0.15
        return min(max(score, 1.0), 10.0)
