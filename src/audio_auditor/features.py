import numpy as np
import librosa


def extract_features(audio: np.ndarray, sample_rate: int) -> dict:
    """Extract a compact acoustic feature set for candidate event classification."""
    if len(audio) == 0:
        return {}

    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=512)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=1024, hop_length=512)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, roll_percent=0.85)[0]
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)[0]
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)

    return {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "centroid_mean": float(np.mean(centroid)),
        "bandwidth_mean": float(np.mean(bandwidth)),
        "rolloff_mean": float(np.mean(rolloff)),
        "contrast_mean": float(np.mean(contrast)),
        "mfcc_mean": [float(x) for x in np.mean(mfcc, axis=1)],
        "duration_s": float(len(audio) / sample_rate),
    }
