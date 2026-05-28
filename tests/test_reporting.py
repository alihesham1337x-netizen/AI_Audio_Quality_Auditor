from src.audio_auditor.reporting import AuditReport


def test_audit_report_pass():
    report = AuditReport(events=[])
    summary = report.summarize()

    assert summary["total_severity"] == 0
    assert summary["decision"] == "PASS"
    assert summary["total_events"] == 0


def test_audit_report_warning():
    report = AuditReport(events=[{"severity": 4.0}])
    summary = report.summarize()

    assert summary["decision"] == "WARNING"
    assert summary["violations"] == 1


def test_audit_report_fail():
    report = AuditReport(events=[{"severity": 10.0}, {"severity": 10.0}, {"severity": 10.0}])
    summary = report.summarize()

    assert summary["decision"] == "FAIL"
    assert summary["total_severity"] == 30.0
    assert summary["violations"] == 3
