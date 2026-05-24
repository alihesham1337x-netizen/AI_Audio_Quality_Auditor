import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

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

    def audit_file(self, file_path: str, sustained_energy_threshold: float = 0.02) -> AudioAuditResult:
        waveform, sr = safe_load_audio(file_path)

        speech_mask, non_speech_mask = self.vad.detect_speech(waveform, sr)
        raw_events = detect_transient_events(waveform, sr)
        raw_events.extend(detect_sustained_noise_segments(waveform, sr, non_speech_mask, energy_threshold=sustained_energy_threshold))
        events = []

        filtered_events = filter_events_by_speech(raw_events, speech_mask, sr)
        for event in filtered_events:
            start_sample = int(event["start_time"] * sr)
            end_sample = int(event["end_time"] * sr)
            segment_waveform = waveform[start_sample:end_sample]
            features = extract_features(segment_waveform, sr)
            label, confidence = self.classifier.classify_segment(segment_waveform, sr)
            severity = self.scorer.score_event(label, confidence, features)
            events.append({
                "start": event["start_time"],
                "end": event["end_time"],
                "label": label,
                "confidence": confidence,
                "severity": severity,
                "strength": event.get("strength", 0.0),
                "transient_score": event.get("score", 0.0),
            })

        report = AuditReport(events=events)
        summary = report.summarize()

        return AudioAuditResult(
            file_path=file_path,
            overall_decision=summary["decision"],
            total_severity=summary["total_severity"],
            events=events,
            metadata={"duration_sec": len(waveform) / sr, "sample_rate": sr, "event_count": len(events)},
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
