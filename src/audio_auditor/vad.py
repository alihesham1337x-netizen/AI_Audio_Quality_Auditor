import numpy as np

try:
    import webrtcvad
except ImportError:
    webrtcvad = None

# Support both the old 'silero' package and the new 'silero-vad' package
_silero_model = None
_silero_get_timestamps = None

try:
    import torch
    try:
        # New silero-vad package (pip install silero-vad)
        from silero_vad import load_silero_vad, get_speech_timestamps
        _silero_model = load_silero_vad()
        _silero_get_timestamps = get_speech_timestamps
    except ImportError:
        # Old silero package fallback
        from silero import vad as _old_silero
        _silero_get_timestamps = _old_silero.get_speech_timestamps
        _silero_model = True  # placeholder, old API loads model internally
except Exception:
    pass


def frame_generator(frame_duration_ms: int, audio: np.ndarray, sample_rate: int):
    """Yield audio frames of fixed duration from a waveform."""
    n = int(sample_rate * (frame_duration_ms / 1000.0))
    offset = 0
    while offset + n <= len(audio):
        yield audio[offset: offset + n]
        offset += n


def detect_speech_webrtc(audio: np.ndarray, sample_rate: int, frame_ms: int = 30):
    """Detect speech regions using WebRTC VAD."""
    if webrtcvad is None:
        raise RuntimeError("webrtcvad is not installed")

    vad = webrtcvad.Vad(2)
    mask = np.zeros(len(audio), dtype=bool)
    frame_len = int(sample_rate * frame_ms / 1000.0)

    for i, frame in enumerate(frame_generator(frame_ms, audio, sample_rate)):
        pcm = (frame * 32767).astype(np.int16).tobytes()
        if vad.is_speech(pcm, sample_rate):
            start = i * frame_len
            mask[start: start + frame_len] = True
    return mask


def detect_speech_energy(audio: np.ndarray, sample_rate: int, frame_ms: int = 30, threshold: float = 0.01):
    """Fallback energy-based speech detection."""
    frame_len = int(sample_rate * frame_ms / 1000.0)
    if frame_len <= 0:
        return np.zeros(len(audio), dtype=bool)

    mask = np.zeros(len(audio), dtype=bool)
    for i, frame in enumerate(frame_generator(frame_ms, audio, sample_rate)):
        rms = np.sqrt(np.mean(frame ** 2) + 1e-9)
        if rms >= threshold:
            start = i * frame_len
            mask[start: start + frame_len] = True
    return mask


def detect_speech_silero(audio: np.ndarray, sample_rate: int):
    """Detect speech regions using Silero VAD."""
    import torch

    if _silero_model is None or _silero_get_timestamps is None:
        raise RuntimeError("Silero VAD is not available")
    if sample_rate != 16000:
        raise ValueError("Silero VAD expects 16 kHz audio")

    wav = torch.from_numpy(audio).float()
    if wav.dim() == 1:
        pass  # already 1D
    else:
        wav = wav.squeeze()

    # New silero-vad API: pass model explicitly
    try:
        speech_timestamps = _silero_get_timestamps(wav, _silero_model, sampling_rate=sample_rate)
    except TypeError:
        # Old API fallback
        speech_timestamps = _silero_get_timestamps(wav, sampling_rate=sample_rate)

    mask = np.zeros(len(audio), dtype=bool)
    for segment in speech_timestamps:
        mask[segment["start"]: segment["end"]] = True
    return mask


class VoiceActivityDetector:
    def __init__(self, engine: str = "silero"):
        self.engine = engine

    def detect_speech(self, audio: np.ndarray, sample_rate: int):
        """Return speech and non-speech boolean masks."""
        speech_mask = None

        if self.engine == "silero":
            try:
                speech_mask = detect_speech_silero(audio, sample_rate)
            except Exception:
                if webrtcvad is not None:
                    try:
                        speech_mask = detect_speech_webrtc(audio, sample_rate)
                    except Exception:
                        pass
                if speech_mask is None:
                    speech_mask = detect_speech_energy(audio, sample_rate)

        elif self.engine == "webrtc":
            try:
                speech_mask = detect_speech_webrtc(audio, sample_rate)
            except Exception:
                try:
                    speech_mask = detect_speech_silero(audio, sample_rate)
                except Exception:
                    speech_mask = detect_speech_energy(audio, sample_rate)
        else:
            speech_mask = detect_speech_energy(audio, sample_rate)

        if speech_mask is None:
            speech_mask = detect_speech_energy(audio, sample_rate)

        non_speech_mask = ~speech_mask
        return speech_mask, non_speech_mask

    def extract_candidate_regions(self, audio: np.ndarray, sample_rate: int, non_speech_mask: np.ndarray, min_duration_s: float = 0.12):
        """Extract non-speech candidate segments for anomaly detection."""
        indices = np.where(non_speech_mask)[0]
        if len(indices) == 0:
            return []

        regions = []
        start = indices[0]
        prev = indices[0]
        for idx in indices[1:]:
            if idx != prev + 1:
                end = prev + 1
                duration = (end - start) / sample_rate
                if duration >= min_duration_s:
                    regions.append((start, end))
                start = idx
            prev = idx
        end = prev + 1
        if (end - start) / sample_rate >= min_duration_s:
            regions.append((start, end))

        return [
            {
                "waveform": audio[start:end],
                "start_time": start / sample_rate,
                "end_time": end / sample_rate,
            }
            for start, end in regions
        ]
