import numpy as np
from scipy.signal import correlate


def extract_anchor_feature(samples: np.ndarray) -> np.ndarray:
    centered = samples - np.mean(samples)
    norm = float(np.linalg.norm(centered))
    if norm <= 1e-12:
        return centered.astype(np.float32)
    return (centered / norm).astype(np.float32)


def normalized_correlation_scores(search_signal: np.ndarray, feature: np.ndarray) -> np.ndarray:
    if feature.size == 0 or feature.size > search_signal.size:
        return np.array([], dtype=np.float32)

    search = np.asarray(search_signal, dtype=np.float32)
    feat = np.asarray(feature, dtype=np.float32)
    window_size = feat.size
    window_count = search.size - window_size + 1

    feat_centered = feat - feat.mean()
    feat_norm = float(np.linalg.norm(feat_centered))
    if feat_norm <= 1e-12:
        return np.full(window_count, -1.0, dtype=np.float32)

    # Numerator: cross-correlation with centered feature.
    numerator = correlate(search, feat_centered, mode="valid", method="fft")

    # Denominator: per-window norm of centered search windows via prefix sums.
    prefix = np.concatenate(([0.0], np.cumsum(search, dtype=np.float64)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(search * search, dtype=np.float64)))
    sum_w = prefix[window_size:] - prefix[:-window_size]
    sum_sq_w = prefix_sq[window_size:] - prefix_sq[:-window_size]
    centered_sq = sum_sq_w - (sum_w * sum_w / window_size)
    centered_sq = np.maximum(centered_sq, 0.0)
    centered_norm = np.sqrt(centered_sq)

    valid = centered_norm > 1e-12
    scores = np.full(window_count, -1.0, dtype=np.float32)
    if np.any(valid):
        scores[valid] = (numerator[valid] / (centered_norm[valid] * feat_norm)).astype(np.float32)
    return scores
