from typing import Dict

# Severity base scores (1–10).
# Speech and Silence are never flagged — they get dropped before scoring.
# Background noise types are scored by how disruptive they are to call quality.
LABEL_SEVERITY = {
    # ── High severity — clearly audible, disruptive ──────────────────────────
    "Dog Barking":        8,   # sudden, loud, unmistakable
    "Impact Noise":       8,   # bangs, slams, hits — jarring
    "Construction":       8,   # loud machinery
    "Background Chatter": 7,   # other people talking in background
    "Keyboard/Typing":    7,   # loud typing audible to caller
    # ── Medium severity — noticeable but less disruptive ─────────────────────
    "Music/TV":           7,   # media playing in background
    "Vehicle Noise":      6,   # traffic, engine rumble
    "Mic Static":         6,   # echo, hiss, feedback — audio quality issue
    "Fan/Air Noise":      5,   # HVAC, fan — constant but low-level
    # ── Generic catch-all ────────────────────────────────────────────────────
    "Other Noise":        5,
    # ── Never flagged ────────────────────────────────────────────────────────
    "Speech":             1,
    "Silence":            1,
}


class SeverityScorer:
    def score_event(self, label: str, confidence: float, features: Dict[str, float]) -> float:
        base = LABEL_SEVERITY.get(label, 4)
        # Loudness: rms_mean 0.01–0.15 → scaled to 0–5
        loudness = min(max(features.get("rms_mean", 0.0) * 55.0, 0.0), 5.0)
        # Duration bonus capped at 4s — sustained noise is worse
        duration = min(max(features.get("duration_s", 0.0), 0.0), 4.0)
        score = base * (0.5 + confidence * 0.5) + loudness * 0.35 + duration * 0.15
        return min(max(score, 1.0), 10.0)
