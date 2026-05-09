import numpy as np
from scipy.signal import resample_poly


def resample_to_sample_rate(samples: np.ndarray, src_sample_rate: int, dst_sample_rate: int) -> np.ndarray:
    """Resample audio with the project-wide band-limited converter."""
    if src_sample_rate <= 0 or dst_sample_rate <= 0:
        raise ValueError("Sample rates must be positive integers")
    if src_sample_rate == dst_sample_rate:
        return samples.astype(np.float32, copy=False)

    gcd = np.gcd(src_sample_rate, dst_sample_rate)
    up = dst_sample_rate // gcd
    down = src_sample_rate // gcd
    resampled = resample_poly(samples, up=up, down=down)
    return resampled.astype(np.float32, copy=False)


def resample_for_analysis(samples: np.ndarray, src_sample_rate: int, dst_sample_rate: int) -> np.ndarray:
    return resample_to_sample_rate(samples, src_sample_rate=src_sample_rate, dst_sample_rate=dst_sample_rate)
