from dataclasses import dataclass
import numpy as np
from scipy.signal import correlate
import warnings


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


@dataclass(frozen=True)
class NccPeakDiagnostics:
    best_score: float
    best_lag_samples: int
    second_score: float | None
    second_lag_samples: int | None
    margin: float | None
    prominence: float
    width_samples: float
    plateau_size_samples: float


def ncc_peak_diagnostics(
    scores: np.ndarray,
    sample_rate: int,
    nms_exclusion_seconds: float = 0.05,
) -> NccPeakDiagnostics | None:
    if scores.size == 0:
        return None

    nms_exclusion_samples = max(1, int(round(nms_exclusion_seconds * sample_rate)))
    all_peaks, properties = _find_all_local_maxima(scores)

    endpoint_candidates: list[int] = []
    if 0 not in all_peaks:
        endpoint_candidates.append(0)
    if len(scores) - 1 not in all_peaks and len(scores) > 1:
        endpoint_candidates.append(len(scores) - 1)
    if endpoint_candidates:
        peak_list = list(all_peaks)
        peak_list.extend(endpoint_candidates)
        all_peaks = np.array(peak_list, dtype=np.intp)

    if len(all_peaks) == 0:
        peak_idx = int(np.argmax(scores))
        all_peaks = np.array([peak_idx])
        properties = {
            "prominences": np.array([0.0]),
            "widths": np.array([0.0]),
            "plateau_sizes": np.array([0.0]),
        }
    else:
        prop_len = len(all_peaks)
        prominences_arr = properties.get("prominences", np.zeros(prop_len - len(endpoint_candidates)))
        widths_arr = properties.get("widths", np.zeros(prop_len - len(endpoint_candidates)))
        plateau_arr = properties.get("plateau_sizes", np.zeros(prop_len - len(endpoint_candidates)))
        for ep_idx in endpoint_candidates:
            try:
                from scipy.signal import peak_prominences, peak_widths
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    ep_prom = float(peak_prominences(scores, np.array([ep_idx]))[0])
                    ep_width = float(peak_widths(scores, np.array([ep_idx]), rel_height=0.5)[0])
                    if ep_prom <= 0.0:
                        ep_prom = max(0.0, float(scores[ep_idx]) - float(np.min(scores)))
                        ep_width = 0.0
            except Exception:
                ep_prom = float(scores[ep_idx]) - float(np.min(scores))
                ep_width = 0.0
            prominences_arr = np.append(prominences_arr, ep_prom)
            widths_arr = np.append(widths_arr, ep_width)
            plateau_arr = np.append(plateau_arr, 0.0)
        properties = {
            "prominences": prominences_arr,
            "widths": widths_arr,
            "plateau_sizes": plateau_arr,
        }

    sorted_order = _argsort_peaks_descending(scores, all_peaks)
    sorted_peaks = all_peaks[sorted_order]

    best_idx = int(sorted_peaks[0])
    best_score = float(scores[best_idx])
    prominences = properties.get("prominences", np.zeros(len(all_peaks)))
    widths = properties.get("widths", np.zeros(len(all_peaks)))
    plateau_sizes = properties.get("plateau_sizes", np.zeros(len(all_peaks)))

    best_prominence = float(prominences[sorted_order[0]]) if sorted_order[0] < len(prominences) else 0.0
    best_width = float(widths[sorted_order[0]]) if sorted_order[0] < len(widths) else 0.0
    best_plateau = float(plateau_sizes[sorted_order[0]]) if sorted_order[0] < len(plateau_sizes) else 0.0

    second_score: float | None = None
    margin: float | None = None
    second_lag_samples: int | None = None

    for peak_idx in sorted_peaks[1:]:
        if abs(int(peak_idx) - best_idx) > nms_exclusion_samples:
            second_lag_samples = int(peak_idx)
            second_score = float(scores[peak_idx])
            margin = best_score - second_score
            break

    return NccPeakDiagnostics(
        best_score=best_score,
        best_lag_samples=best_idx,
        second_score=second_score,
        second_lag_samples=second_lag_samples,
        margin=margin,
        prominence=best_prominence,
        width_samples=best_width,
        plateau_size_samples=best_plateau,
    )


def _find_all_local_maxima(scores: np.ndarray) -> tuple[np.ndarray, dict]:
    try:
        from scipy.signal import find_peaks
    except ImportError:
        peak_idx = int(np.argmax(scores))
        return np.array([peak_idx]), {
            "prominences": np.array([0.0]),
            "widths": np.array([0.0]),
            "plateau_sizes": np.array([0.0]),
        }

    peaks, properties = find_peaks(
        np.asarray(scores, dtype=np.float64),
        prominence=0.0,
        width=0.0,
        plateau_size=0.0,
    )
    return peaks, properties


def _argsort_peaks_descending(scores: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    neg_scores = np.array([-float(scores[p]) for p in peaks], dtype=np.float64)
    return np.lexsort((peaks, neg_scores))


def gcc_phat_scores(
    search_signal: np.ndarray,
    feature: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    if feature.size == 0 or feature.size > search_signal.size:
        return np.array([], dtype=np.float32)

    search = np.asarray(search_signal, dtype=np.float64)
    feat = np.asarray(feature, dtype=np.float64)
    window_size = feat.size

    fft_size = search.size + feat.size - 1
    nfft = 1
    while nfft < fft_size:
        nfft <<= 1

    X = np.fft.rfft(search, n=nfft)
    Y = np.fft.rfft(feat, n=nfft)
    cross = X * np.conj(Y)
    denominator = np.maximum(np.abs(cross), epsilon)
    gcc = cross / denominator
    gcc_signal = np.fft.irfft(gcc, n=nfft)

    valid_count = search.size - window_size + 1
    result = gcc_signal[:valid_count].astype(np.float32)
    return result


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
