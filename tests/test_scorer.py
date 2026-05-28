from src.audio_auditor.scorer import SeverityScorer


def test_severity_score_range():
    scorer = SeverityScorer()
    features = {"rms_mean": 0.03, "duration_s": 2.0}

    score = scorer.score_event("Construction", 0.8, features)
    assert 1.0 <= score <= 10.0


def test_severity_score_increases_with_confidence():
    scorer = SeverityScorer()
    features = {"rms_mean": 0.03, "duration_s": 2.0}

    low_score = scorer.score_event("Construction", 0.2, features)
    high_score = scorer.score_event("Construction", 0.9, features)

    assert high_score > low_score
