import csv
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class AuditReport:
    events: List[Dict]

    def summarize(self) -> Dict:
        total_severity = sum(event.get("severity", 0.0) for event in self.events)
        violations = [event for event in self.events if event.get("severity", 0.0) >= 4.0]
        result = "PASS"
        if total_severity >= 25 or len(violations) >= 3:
            result = "FAIL"
        elif total_severity >= 8 or len(violations) >= 1:
            result = "WARNING"
        return {
            "total_events": len(self.events),
            "total_severity": total_severity,
            "violations": len(violations),
            "decision": result,
        }

    def export_json(self, output_path: str) -> None:
        import json

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"events": self.events, **self.summarize()}, f, indent=2)

    def export_csv(self, output_path: str) -> None:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["start", "end", "label", "confidence", "severity"])
            writer.writeheader()
            for event in self.events:
                writer.writerow({
                    "start": event.get("start"),
                    "end": event.get("end"),
                    "label": event.get("label"),
                    "confidence": event.get("confidence"),
                    "severity": event.get("severity"),
                })
