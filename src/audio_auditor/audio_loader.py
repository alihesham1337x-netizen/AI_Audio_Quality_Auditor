import os
from typing import Tuple, Optional

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16000


def load_audio(file_path: str, target_sr: int = TARGET_SR, mono: bool = True) -> Tuple[np.ndarray, int]:
    """Load an audio file, resample to target sample rate, and convert to mono."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        audio, sr = sf.read(file_path, always_2d=False)
    except Exception:
        audio, sr = librosa.load(file_path, sr=None, mono=False)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio.astype(np.float32), sr


def normalize_audio(audio: np.ndarray, ref_db: float = -30.0, eps: float = 1e-6) -> np.ndarray:
    """Normalize loudness to a reference dBFS level."""
    rms = np.sqrt(np.mean(np.square(audio)) + eps)
    scalar = 10 ** (ref_db / 20) / (rms + eps)
    return audio * scalar


def safe_load_audio(file_path: str, chunk_size_sec: float = 30.0) -> Tuple[np.ndarray, int]:
    """Load audio safely and return mono 16kHz waveform.

    This implementation is currently full-file, but can be extended for streaming.
    """
    audio, sr = load_audio(file_path)
    audio = normalize_audio(audio)
    return audio, sr
