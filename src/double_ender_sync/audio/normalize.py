import numpy as np


def convert_to_mono(samples: np.ndarray) -> np.ndarray:
    """Convert audio to mono for analysis. Keeps float32 dtype."""
    if samples.ndim == 1:
        return samples.astype(np.float32, copy=False)
    if samples.ndim != 2:
        raise ValueError(f"Expected 1D or 2D audio array, got shape={samples.shape!r}")
    return np.mean(samples, axis=1, dtype=np.float32)
