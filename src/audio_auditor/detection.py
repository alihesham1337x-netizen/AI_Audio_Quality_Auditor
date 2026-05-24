import numpy as np
import librosa
from scipy.signal import medfilt


def detect_transient_events(audio: np.ndarray, sample_rate: int, hop_ms: int = 25, z_threshold: float = 3.5, min_duration_s: float = 0.08):
    """Detect transient acoustic spikes using adaptive onset and z-score thresholds."""
    if len(audio) == 0:
        return []

    hop_length = max(1, int(sample_rate * hop_ms / 1000))
    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop_length, aggregate=np.mean)
    if len(onset_env) == 0:
        return []

    baseline = medfilt(onset_env, kernel_size=9)
    residual = onset_env - baseline
    std = np.std(residual) + 1e-8
    z_scores = residual / std

    candidate_frames = np.where(z_scores >= z_threshold)[0]
    events = []
    for frame_idx in candidate_frames:
        start_frame = max(0, frame_idx - 2)
        end_frame = min(len(onset_env) - 1, frame_idx + 2)
        start_time = start_frame * hop_ms / 1000.0
        end_time = (end_frame + 1) * hop_ms / 1000.0
        duration = end_time - start_time
        if duration < min_duration_s:
            end_time = start_time + min_duration_s
        confidence = float(min(1.0, z_scores[frame_idx] / (z_threshold + 2.0)))
        events.append({
            "start_time": start_time,
            "end_time": min(end_time, len(audio) / sample_rate),
            "strength": float(onset_env[frame_idx]),
            "confidence": confidence,
            "score": float(z_scores[frame_idx]),
        })

    return merge_nearby_events(events)


def merge_nearby_events(events, max_gap_s: float = 0.75):
    """Merge transient detections that are close in time into a single event."""
    if not events:
        return []

    events = sorted(events, key=lambda e: e["start_time"])
    merged = [events[0].copy()]
    for event in events[1:]:
        last = merged[-1]
        if event["start_time"] <= last["end_time"] + max_gap_s:
            last["end_time"] = max(last["end_time"], event["end_time"])
            last["confidence"] = max(last["confidence"], event["confidence"])
            last["score"] = max(last["score"], event["score"])
            last["strength"] = max(last["strength"], event["strength"])
        else:
            merged.append(event.copy())
    return merged


def detect_sustained_noise_segments(audio: np.ndarray, sample_rate: int, non_speech_mask: np.ndarray, frame_ms: int = 100, energy_threshold: float = 0.02, min_duration_s: float = 0.5):
    """Detect longer non-speech noise segments using frame-level RMS energy."""
    if len(audio) == 0 or len(non_speech_mask) != len(audio):
        return []

    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    events = []
    candidate_frames = []
    sample_indices = []

    for start in range(0, len(audio), frame_len):
        end = min(len(audio), start + frame_len)
        if not np.any(non_speech_mask[start:end]):
            candidate_frames.append(None)
            sample_indices.append((start, end))
            continue
        frame = audio[start:end]
        rms = float(np.sqrt(np.mean(frame ** 2) + 1e-12))
        candidate_frames.append(rms)
        sample_indices.append((start, end))

    energies = [r for r in candidate_frames if r is not None]
    if not energies:
        return []

    baseline = float(np.median(energies))
    threshold = max(energy_threshold, baseline * 2.0)

    active_start = None
    active_end = None
    active_strength = 0.0

    for rms, (start, end) in zip(candidate_frames, sample_indices):
        if rms is not None and rms >= threshold:
            if active_start is None:
                active_start = start
            active_end = end
            active_strength = max(active_strength, rms)
        else:
            if active_start is not None:
                duration = (active_end - active_start) / sample_rate
                if duration >= min_duration_s:
                    events.append({
                        "start_time": active_start / sample_rate,
                        "end_time": active_end / sample_rate,
                        "strength": active_strength,
                        "confidence": float(min(1.0, (active_strength - threshold) / (threshold + 1e-9))),
                        "score": float(active_strength / (threshold + 1e-9)),
                    })
                active_start = None
                active_end = None
                active_strength = 0.0

    if active_start is not None:
        duration = (active_end - active_start) / sample_rate
        if duration >= min_duration_s:
            events.append({
                "start_time": active_start / sample_rate,
                "end_time": active_end / sample_rate,
                "strength": active_strength,
                "confidence": float(min(1.0, (active_strength - threshold) / (threshold + 1e-9))),
                "score": float(active_strength / (threshold + 1e-9)),
            })

    return merge_nearby_events(events)


def filter_events_by_speech(events, speech_mask: np.ndarray, sample_rate: int, speech_ratio_threshold: float = 0.35):
    """Remove events that overlap too heavily with speech regions."""
    filtered = []
    for event in events:
        start_sample = int(event["start_time"] * sample_rate)
        end_sample = int(event["end_time"] * sample_rate)
        overlap = speech_mask[start_sample:end_sample]
        if len(overlap) == 0:
            filtered.append(event)
            continue
        speech_ratio = float(np.mean(overlap))
        if speech_ratio < speech_ratio_threshold:
            filtered.append(event)
    return filtered
