import numpy as np

from src.audio_auditor.detection import detect_transient_events, filter_events_by_speech


def test_detect_transient_events_empty():
    events = detect_transient_events(np.zeros(16000), 16000)
    assert isinstance(events, list)
    assert len(events) == 0


def test_filter_events_by_speech():
    audio_length = 16000
    speech_mask = np.zeros(audio_length, dtype=bool)
    events = [{"start_time": 0.0, "end_time": 0.2, "confidence": 0.8, "score": 5.0, "strength": 1.0}]
    filtered = filter_events_by_speech(events, speech_mask, 16000)
    assert len(filtered) == 1
    speech_mask[0:1600] = True
    filtered = filter_events_by_speech(events, speech_mask, 16000)
    assert len(filtered) == 0
