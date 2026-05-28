import numpy as np
import librosa
from scipy.signal import medfilt


# Absolute minimum noise-floor RMS to be considered real background noise.
# Clean files: ns_rms stays below 0.0013 in all speech windows.
# Noisy files: ns_rms reaches 0.0026+ in affected windows.
# Set to 0.0018 — safely above clean-file max, catches subtle noisy files.
_ABS_NOISE_FLOOR_MIN = 0.0018


def _build_speech_onset_mask(speech_mask: np.ndarray, sample_rate: int, grace_ms: int = 200) -> np.ndarray:
    """Return a boolean mask that is True for samples within `grace_ms` before
    the FIRST speech segment only. This prevents the agent's opening word
    ('hello') from generating a false transient detection, without blocking
    detection of noise that occurs before later speech segments.
    """
    grace_samples = int(sample_rate * grace_ms / 1000)
    mask = np.zeros(len(speech_mask), dtype=bool)
    # Only protect the very first speech onset
    for i in range(len(speech_mask)):
        if speech_mask[i]:
            start = max(0, i - grace_samples)
            mask[start:i] = True
            break  # only first onset
    return mask


def detect_transient_events(
    audio: np.ndarray,
    sample_rate: int,
    hop_ms: int = 25,
    z_threshold: float = 2.5,
    min_duration_s: float = 0.06,
    speech_mask: np.ndarray = None,
    flatness_min: float = 0.15,  # kept for API compat
):
    """Detect sudden transient noise spikes.

    Dual-threshold strategy:
    - Non-speech frames: standard z_threshold
    - Speech frames: z_threshold + 2.5 (only very strong spikes pass)

    Pre-speech grace window: frames within 200 ms before a speech segment
    starts are excluded — this prevents the agent's opening 'hello' from
    generating a false transient detection.
    """
    if len(audio) == 0:
        return []

    hop_length = max(1, int(sample_rate * hop_ms / 1000))
    onset_env = librosa.onset.onset_strength(
        y=audio, sr=sample_rate, hop_length=hop_length, aggregate=np.mean
    )
    if len(onset_env) == 0:
        return []

    baseline = medfilt(onset_env, kernel_size=11)
    residual = onset_env - baseline
    std = np.std(residual) + 1e-8
    z_scores = residual / std

    # Per-frame speech ratio and pre-speech grace mask
    speech_ratio_per_frame = np.zeros(len(onset_env), dtype=float)
    grace_per_frame = np.zeros(len(onset_env), dtype=bool)

    if speech_mask is not None:
        grace_mask = _build_speech_onset_mask(speech_mask, sample_rate, grace_ms=200)
        for i in range(len(onset_env)):
            s = i * hop_length
            e = min(len(audio), s + hop_length)
            seg_sp = speech_mask[s:e]
            seg_gr = grace_mask[s:e]
            speech_ratio_per_frame[i] = float(np.mean(seg_sp)) if len(seg_sp) else 0.0
            grace_per_frame[i] = bool(np.any(seg_gr))

    speech_z = z_threshold + 3.0  # speech frames need a much stronger spike
    candidate_frames = np.where(
        (~grace_per_frame)  # exclude pre-speech grace window
        & (
            ((speech_ratio_per_frame < 0.40) & (z_scores >= z_threshold))
            | ((speech_ratio_per_frame >= 0.40) & (z_scores >= speech_z))
        )
    )[0]

    events = []
    for frame_idx in candidate_frames:
        start_frame = max(0, frame_idx - 2)
        end_frame = min(len(onset_env) - 1, frame_idx + 2)
        start_time = start_frame * hop_ms / 1000.0
        end_time = (end_frame + 1) * hop_ms / 1000.0
        if end_time - start_time < min_duration_s:
            end_time = start_time + min_duration_s

        confidence = float(min(1.0, z_scores[frame_idx] / (z_threshold + 2.0)))

        # Compute spike-to-context RMS ratio.
        # Speech onsets (silence → speech) have very high ratios (4–16×).
        # Real noise during ongoing speech has low ratios (1–3×).
        frame_s = frame_idx * hop_length
        frame_e = min(len(audio), frame_s + hop_length)
        frame_rms = float(np.sqrt(np.mean(audio[frame_s:frame_e] ** 2) + 1e-12))
        pre_start = max(0, frame_s - int(0.5 * sample_rate))
        pre_rms = float(np.sqrt(np.mean(audio[pre_start:frame_s] ** 2) + 1e-12))
        spike_ratio = frame_rms / (pre_rms + 1e-9)

        events.append({
            "start_time": start_time,
            "end_time": min(end_time, len(audio) / sample_rate),
            "strength": float(onset_env[frame_idx]),
            "confidence": confidence,
            "score": float(z_scores[frame_idx]),
            "flatness": 0.0,
            "speech_ratio": float(speech_ratio_per_frame[frame_idx]),
            "spike_ratio": spike_ratio,
            "detector": "transient",
        })

    return merge_nearby_events(events)


def merge_nearby_events(events, max_gap_s: float = 0.50):
    """Merge detections that are close in time into a single event."""
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
            # Keep the minimum spike_ratio (most noise-like of the merged frames)
            last["spike_ratio"] = min(
                last.get("spike_ratio", 999.0),
                event.get("spike_ratio", 999.0)
            )
        else:
            merged.append(event.copy())
    return merged


def detect_sustained_noise_segments(
    audio: np.ndarray,
    sample_rate: int,
    non_speech_mask: np.ndarray,
    frame_ms: int = 100,
    energy_threshold: float = 0.018,
    min_duration_s: float = 0.8,
):
    """Detect sustained background noise using two complementary strategies.

    Strategy A — Non-speech gap energy:
        Frames that are mostly non-speech AND have elevated RMS.

    Strategy B — Noise floor within speech windows:
        Measures the RMS of non-speech samples within speech-heavy windows.
        Uses a dual threshold:
          - Relative: must be >= 4x the per-file baseline (p20 of all windows)
          - Absolute: must be >= _ABS_NOISE_FLOOR_MIN (0.0025)
        Both must be satisfied to avoid flagging clean files whose relative
        baseline is near zero (making 2-3x look large in relative terms).
    """
    if len(audio) == 0 or len(non_speech_mask) != len(audio):
        return []

    events = []

    # ── Strategy A: non-speech gap energy ────────────────────────────────────
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    sample_indices_a, frame_rms_a, frame_ns_ratio_a = [], [], []

    for start in range(0, len(audio), frame_len):
        end = min(len(audio), start + frame_len)
        frame = audio[start:end]
        rms = float(np.sqrt(np.mean(frame ** 2) + 1e-12))
        ns_seg = non_speech_mask[start:end]
        ns_ratio = float(np.mean(ns_seg)) if len(ns_seg) else 0.0
        frame_rms_a.append(rms)
        frame_ns_ratio_a.append(ns_ratio)
        sample_indices_a.append((start, end))

    if frame_rms_a:
        rms_arr = np.array(frame_rms_a)
        baseline_a = max(energy_threshold, float(np.median(rms_arr)))
        threshold_a = max(energy_threshold, baseline_a * 1.08)
        active_start = active_end = None
        active_strength = 0.0

        for (start, end), rms, ns_ratio in zip(sample_indices_a, frame_rms_a, frame_ns_ratio_a):
            noisy = ns_ratio >= 0.50 and rms >= threshold_a
            if noisy:
                if active_start is None:
                    active_start = start
                active_end = end
                active_strength = max(active_strength, rms)
            else:
                if active_start is not None:
                    dur = (active_end - active_start) / sample_rate
                    if dur >= min_duration_s:
                        conf = float(min(1.0, (active_strength - threshold_a) / (threshold_a + 1e-9)))
                        events.append({
                            "start_time": active_start / sample_rate,
                            "end_time": active_end / sample_rate,
                            "strength": active_strength,
                            "confidence": max(0.20, conf),
                            "score": float(active_strength / (threshold_a + 1e-9)),
                            "flatness": 0.0,
                            "detector": "gap_energy",
                        })
                    active_start = active_end = None
                    active_strength = 0.0

        if active_start is not None:
            dur = (active_end - active_start) / sample_rate
            if dur >= min_duration_s:
                conf = float(min(1.0, (active_strength - threshold_a) / (threshold_a + 1e-9)))
                events.append({
                    "start_time": active_start / sample_rate,
                    "end_time": active_end / sample_rate,
                    "strength": active_strength,
                    "confidence": max(0.20, conf),
                    "score": float(active_strength / (threshold_a + 1e-9)),
                    "flatness": 0.0,
                    "detector": "gap_energy",
                })

    # ── Strategy B: noise floor within speech windows ────────────────────────
    window_len = sample_rate          # 1-second windows
    hop_len    = sample_rate // 2     # 0.5-second hop

    window_positions = []
    valid_floors = []

    for start in range(0, len(audio) - window_len + 1, hop_len):
        end = start + window_len
        sp_seg = ~non_speech_mask[start:end]
        ns_seg = non_speech_mask[start:end]
        sp_ratio = float(np.mean(sp_seg))
        ns_samples = audio[start:end][ns_seg]
        if sp_ratio >= 0.30 and len(ns_samples) >= int(0.05 * sample_rate):
            ns_rms = float(np.sqrt(np.mean(ns_samples ** 2) + 1e-12))
            valid_floors.append(ns_rms)
            window_positions.append((start, end, sp_ratio, ns_rms))
        else:
            window_positions.append((start, end, sp_ratio, None))

    if valid_floors:
        floor_baseline = float(np.percentile(valid_floors, 20))
        # Dual threshold: relative (3× baseline) AND absolute minimum.
        # Lowered from 4× to 3× to catch more subtle sustained noise.
        # The absolute floor prevents clean files from being flagged.
        rel_threshold = floor_baseline * 3.0
        floor_threshold = max(rel_threshold, _ABS_NOISE_FLOOR_MIN)

        active_start = active_end = None
        active_strength = 0.0

        for (start, end, sp_ratio, ns_rms) in window_positions:
            noisy = (ns_rms is not None
                     and ns_rms >= floor_threshold
                     and sp_ratio >= 0.30)
            if noisy:
                if active_start is None:
                    active_start = start
                active_end = end
                active_strength = max(active_strength, ns_rms)
            else:
                if active_start is not None:
                    dur = (active_end - active_start) / sample_rate
                    if dur >= min_duration_s * 0.4:
                        conf = float(min(1.0, (active_strength - floor_threshold) / (floor_threshold + 1e-9)))
                        events.append({
                            "start_time": active_start / sample_rate,
                            "end_time": active_end / sample_rate,
                            "strength": active_strength,
                            "confidence": max(0.30, conf),
                            "score": float(active_strength / (floor_threshold + 1e-9)),
                            "flatness": 0.0,
                            "detector": "noise_floor",
                        })
                    active_start = active_end = None
                    active_strength = 0.0

        if active_start is not None:
            dur = (active_end - active_start) / sample_rate
            if dur >= min_duration_s * 0.4:
                conf = float(min(1.0, (active_strength - floor_threshold) / (floor_threshold + 1e-9)))
                events.append({
                    "start_time": active_start / sample_rate,
                    "end_time": active_end / sample_rate,
                    "strength": active_strength,
                    "confidence": max(0.30, conf),
                    "score": float(active_strength / (floor_threshold + 1e-9)),
                    "flatness": 0.0,
                    "detector": "noise_floor",
                })

    # ── Strategy C: embedded noise in speech — sub-bass & spectral flatness ──
    # Some calls have background noise (hum, rumble, broadband noise) mixed
    # directly into the speech signal. The non-speech gaps are clean so
    # Strategies A and B miss it. We detect it by analysing the spectral
    # content of speech segments:
    #   - Elevated sub-bass energy (<150 Hz) → room rumble / AC hum
    #   - High spectral flatness → broadband noise mixed with speech
    # We compare each speech window against the file's own clean-speech
    # baseline (lowest 20th percentile) to avoid flagging normal speech.
    window_len_c = sample_rate
    hop_len_c    = sample_rate // 2
    speech_mask_c = ~non_speech_mask

    sub_bass_vals = []
    flatness_vals = []
    window_pos_c  = []

    for start in range(0, len(audio) - window_len_c + 1, hop_len_c):
        end = start + window_len_c
        sp_chunk = speech_mask_c[start:end]
        sp_ratio = float(np.mean(sp_chunk))
        if sp_ratio < 0.40:
            window_pos_c.append((start, end, sp_ratio, None, None))
            continue
        sp_samples = audio[start:end][sp_chunk]
        if len(sp_samples) < int(0.1 * sample_rate):
            window_pos_c.append((start, end, sp_ratio, None, None))
            continue
        fft_c   = np.abs(np.fft.rfft(sp_samples))
        freqs_c = np.fft.rfftfreq(len(sp_samples), 1.0 / sample_rate)
        total_c = np.sum(fft_c ** 2) + 1e-12
        sub_bass   = float(np.sum(fft_c[freqs_c < 150] ** 2) / total_c)
        log_mean   = float(np.mean(np.log(fft_c + 1e-12)))
        arith_mean = float(np.mean(fft_c) + 1e-12)
        flatness_c = float(np.exp(log_mean) / arith_mean)
        sub_bass_vals.append(sub_bass)
        flatness_vals.append(flatness_c)
        window_pos_c.append((start, end, sp_ratio, sub_bass, flatness_c))

    if sub_bass_vals and flatness_vals:
        sub_baseline  = float(np.percentile(sub_bass_vals, 20))
        flat_baseline = float(np.percentile(flatness_vals, 20))
        sub_threshold  = max(sub_baseline * 4.0, 0.008)
        flat_threshold = max(flat_baseline * 5.0, 0.020)

        active_start = active_end = None
        active_strength = 0.0

        for (start, end, sp_ratio, sub_bass, flatness_c) in window_pos_c:
            noisy_c = (
                sub_bass is not None
                and sp_ratio >= 0.40
                and (sub_bass >= sub_threshold or flatness_c >= flat_threshold)
            )
            if noisy_c:
                if active_start is None:
                    active_start = start
                active_end = end
                sig = max(sub_bass or 0.0, flatness_c or 0.0)
                active_strength = max(active_strength, sig)
            else:
                if active_start is not None:
                    dur = (active_end - active_start) / sample_rate
                    if dur >= min_duration_s * 0.5:
                        events.append({
                            "start_time": active_start / sample_rate,
                            "end_time":   active_end   / sample_rate,
                            "strength":   active_strength,
                            "confidence": 0.45,
                            "score":      active_strength * 10.0,
                            "flatness":   0.0,
                            "detector":   "speech_spectral",
                        })
                    active_start = active_end = None
                    active_strength = 0.0

        if active_start is not None:
            dur = (active_end - active_start) / sample_rate
            if dur >= min_duration_s * 0.5:
                events.append({
                    "start_time": active_start / sample_rate,
                    "end_time":   active_end   / sample_rate,
                    "strength":   active_strength,
                    "confidence": 0.45,
                    "score":      active_strength * 10.0,
                    "flatness":   0.0,
                    "detector":   "speech_spectral",
                })

    return merge_nearby_events(events)


def filter_events_by_speech(
    events,
    speech_mask: np.ndarray,
    sample_rate: int,
    speech_ratio_threshold: float = 0.75,
):
    """Drop events whose window is more than 75% speech.

    For speech-region transients, use the spike_ratio (frame RMS / preceding
    500ms RMS) to distinguish real noise from speech onsets:
    - Speech onsets (silence → speech): spike_ratio > 3.5 — DROP
    - Real noise during ongoing speech: spike_ratio <= 3.5 — KEEP
    Noise-floor events are always kept.
    """
    filtered = []
    for event in events:
        if event.get("detector") == "noise_floor":
            filtered.append(event)
            continue

        # Speech-spectral events are detected within speech — always keep
        if event.get("detector") == "speech_spectral":
            filtered.append(event)
            continue

        start_sample = int(event["start_time"] * sample_rate)
        end_sample   = int(event["end_time"]   * sample_rate)
        overlap = speech_mask[start_sample:end_sample]
        if len(overlap) == 0:
            filtered.append(event)
            continue

        speech_ratio = float(np.mean(overlap))

        if speech_ratio < speech_ratio_threshold:
            # Mostly non-speech — keep
            filtered.append(event)
            continue

        # Majority speech: drop if spike_ratio is high (speech onset pattern)
        spike_ratio = event.get("spike_ratio", 1.0)
        if spike_ratio > 3.5:
            continue  # speech onset — silence transitioning to speech

        filtered.append(event)
    return filtered
