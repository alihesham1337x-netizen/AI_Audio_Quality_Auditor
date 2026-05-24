import numpy as np

try:
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
except ImportError:
    apply_model = None
    get_model = None


class SourceSeparator:
    def __init__(self):
        self.model = None
        if get_model is not None:
            try:
                self.model = get_model("demucs")
            except Exception:
                self.model = None

    def separate(self, audio: np.ndarray, sample_rate: int):
        """Return foreground and background stems."""
        if self.model is None or apply_model is None:
            return {
                "foreground": audio,
                "background": np.zeros_like(audio),
            }

        # Placeholder: Demucs expects a batch tensor, not raw numpy, in real usage.
        return {
            "foreground": audio,
            "background": np.zeros_like(audio),
        }
