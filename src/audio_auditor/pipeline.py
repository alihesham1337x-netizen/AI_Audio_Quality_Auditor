import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import numpy as np

from .audio_loader import safe_load_audio
from .vad import VoiceActivityDetector
from .detection import detect_transient_events, detect_sustained_noise_segments, filter_events_by_speech
from .features import extract_features
from .classifier import EnvironmentalClassifier
from .scorer import SeverityScorer
from .reporting import AuditReport


@dataclass
class AudioAuditResult:
    file_path: str
    overall_decision: str
    total_severity: float
    events: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class AudioAuditPipeline:
    def __init__(self):
        self.vad = VoiceActivityDetector()
        self.classifier = EnvironmentalClassifier()
        self.scorer = SeverityScorer()

    def audit_file(
        self,
        file_path: str,
        sustained_energy_threshold: float = 0.018,
        transient_z_threshold: float = 2.5,
        transient_flatness_min: float = 0.15,
    ) -> AudioAuditResult:
        waveform, sr = safe_load_audio(file_path)
        agent_audio = waveform[:, 0] if waveform.ndim == 2 else waveform

        speech_mask, non_speech_mask = self.vad.detect_speech(agent_audio, sr)

        raw_events = detect_transient_events(
            agent_audio,
            sr,
            z_threshold=transient_z_threshold,
            min_duration_s=0.06,
            speech_mask=speech_mask,
            flatness_min=transient_flatness_min,
        )
        raw_events.extend(
            detect_sustained_noise_segments(
                agent_audio,
                sr,
                non_speech_mask,
                energy_threshold=sustained_energy_threshold,
            )
        )

        filtered_events = filter_events_by_speech(raw_events, speech_mask, sr)

        events = []
        for event in filtered_events:
            start_sample = int(event["start_time"] * sr)
            end_sample = int(event["end_time"] * sr)
            segment_waveform = agent_audio[start_sample:end_sample]
            if len(segment_waveform) == 0:
                continue

            features = extract_features(segment_waveform, sr)
            label, confidence = self.classifier.classify_segment(segment_waveform, sr)

            speech_overlap = 0.0
            if end_sample > start_sample:
                speech_overlap = float(np.mean(speech_mask[start_sample:end_sample]))

            # Labels that are definitively NOT noise — drop immediately
            SPEECH_LABELS = {"Speech", "Silence"}
            # Labels that are definitively noise — keep immediately
            NOISE_LABELS = {
                "Dog Barking", "Background Chatter", "Mic Static",
                "Impact Noise", "Fan/Air Noise", "Music/TV",
                "Vehicle Noise", "Construction", "Keyboard/Typing",
                "Other Noise",
            }

            if label in NOISE_LABELS:
                # Classifier already identified it as a specific noise type — keep
                pass
            elif label in SPEECH_LABELS:
                is_noise_floor = event.get("detector") == "noise_floor"
                is_speech_spectral = event.get("detector") == "speech_spectral"
                is_transient = event.get("detector") not in ("noise_floor", "gap_energy", "speech_spectral")
                score = event.get("score", 0.0)
                strength = event.get("strength", 0.0)
                spike_ratio = event.get("spike_ratio", 999.0)

                if is_noise_floor:
                    label = "Other Noise"
                    confidence = max(confidence, 0.50)
                elif is_speech_spectral:
                    label = "Other Noise"
                    confidence = max(confidence, 0.48)
                elif label == "Speech":
                    # Classifier says Speech — trust it, never re-label
                    continue
                elif is_transient and score >= 2.0 and spike_ratio <= 3.5:
                    # Silence segment with real onset spike (not a speech onset)
                    label = "Impact Noise"
                    confidence = max(confidence, 0.52)
                elif features.get("rms_mean", 0.0) >= 0.010 and strength >= 0.08:
                    # Sustained Silence with meaningful energy
                    label = "Other Noise"
                    confidence = max(confidence, 0.52)
                else:
                    continue
            else:
                # Unknown label — treat as Other Noise
                label = "Other Noise"
                confidence = max(confidence, 0.45)

            # Final guard: drop weak events that are still majority speech.
            # High-confidence specific noise types bypass this guard.
            if (speech_overlap >= 0.75
                    and confidence < 0.45
                    and event.get("detector") not in ("noise_floor", "speech_spectral")
                    and event.get("score", 0.0) < 5.0
                    and label not in ("Dog Barking", "Impact Noise", "Background Chatter",
                                      "Mic Static", "Fan/Air Noise")):
                continue

            severity = self.scorer.score_event(label, confidence, features)
            events.append({
                "start": event["start_time"],
                "end": event["end_time"],
                "channel": "agent-left",
                "label": label,
                "confidence": confidence,
                "severity": severity,
                "strength": event.get("strength", 0.0),
                "transient_score": event.get("score", 0.0),
            })

        report = AuditReport(events=events)
        # Sort events by start time before reporting
        events = sorted(events, key=lambda e: e["start"])
        report = AuditReport(events=events)
        summary = report.summarize()

        return AudioAuditResult(
            file_path=file_path,
            overall_decision=summary["decision"],
            total_severity=summary["total_severity"],
            events=events,
            metadata={
                "duration_sec": len(waveform) / sr,
                "sample_rate": sr,
                "event_count": len(events),
            },
        )

    def audit_batch(self, file_paths: List[str]) -> List[AudioAuditResult]:
        results = []
        for path in file_paths:
            try:
                results.append(self.audit_file(path))
            except Exception as exc:
                results.append(AudioAuditResult(
                    file_path=path,
                    overall_decision="ERROR",
                    total_severity=0.0,
                    events=[],
                    metadata={"error": str(exc)},
                ))
        return results

    def export_json(self, results: List[AudioAuditResult], out_path: str) -> None:
        payload = [asdict(result) for result in results]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
