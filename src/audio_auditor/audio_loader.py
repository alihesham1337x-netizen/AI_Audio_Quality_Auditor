import os
from typing import Tuple, Optional

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16000


def load_audio(file_path: str, target_sr: int = TARGET_SR, mono: bool = False) -> Tuple[np.ndarray, int]:
    """Load an audio file, resample to target sample rate, and optionally convert to mono."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        audio, sr = sf.read(file_path, always_2d=False)
    except Exception:
        audio, sr = librosa.load(file_path, sr=None, mono=False)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        # Normalize stereo shape to (samples, channels) for both soundfile and librosa.
        if audio.shape[0] <= 2 and audio.shape[1] > 2:
            audio = audio.T
        if mono:
            audio = np.mean(audio, axis=1)

    if sr != target_sr:
        if audio.ndim == 1:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        else:
            audio = np.stack(
                [librosa.resample(audio[:, ch], orig_sr=sr, target_sr=target_sr) for ch in range(audio.shape[1])],
                axis=1,
            )
        sr = target_sr

    return audio.astype(np.float32), sr


def normalize_audio(audio: np.ndarray, ref_db: float = -30.0, eps: float = 1e-6) -> np.ndarray:
    """Normalize loudness to a reference dBFS level."""
    rms = np.sqrt(np.mean(np.square(audio)) + eps)
    scalar = 10 ** (ref_db / 20) / (rms + eps)
    return audio * scalar


def safe_load_audio(file_path: str, chunk_size_sec: float = 30.0) -> Tuple[np.ndarray, int]:
    """Load audio safely and return 16kHz waveform.

    Stereo audio is preserved when available so channel-specific analysis can
    focus on the left (agent) channel.
    """
    audio, sr = load_audio(file_path, mono=False)
    audio = normalize_audio(audio)
    return audio, sr
