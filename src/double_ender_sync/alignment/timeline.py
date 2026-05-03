from dataclasses import dataclass
import logging

import numpy as np

from double_ender_sync.analysis.drift import DriftEstimate
from double_ender_sync.types import AudioTrack

LOGGER = logging.getLogger(__name__)


@dataclass
class GlobalAlignmentResult:
    output_samples: np.ndarray
    output_sample_rate: int
    output_duration_seconds: float
    stretch_ratio: float
    offset_seconds: float


def apply_global_time_correction(
    track: AudioTrack,
    master: AudioTrack,
    drift_estimate: DriftEstimate,
    stretch_method: str = "resample",
) -> GlobalAlignmentResult:
    """Apply phase 3 global correction and place the result on master timeline."""
    if drift_estimate.stretch_ratio <= 0:
        raise ValueError("stretch_ratio must be positive")

    if track.original_samples is None:
        raise ValueError("track.original_samples is required for global correction")

    mono_source = _to_mono(track.original_samples)
    stretched = _stretch_samples(mono_source, drift_estimate.stretch_ratio, stretch_method)

    output_length = int(round(master.duration_seconds * master.sample_rate))
    output = np.zeros(output_length, dtype=np.float32)

    start_index = int(round(drift_estimate.offset_seconds * master.sample_rate))
    src_start = 0
    if start_index < 0:
        src_start = -start_index
        start_index = 0

    available = min(stretched.shape[0] - src_start, output.shape[0] - start_index)
    if available > 0:
        output[start_index : start_index + available] = stretched[src_start : src_start + available]

    return GlobalAlignmentResult(
        output_samples=output,
        output_sample_rate=master.sample_rate,
        output_duration_seconds=master.duration_seconds,
        stretch_ratio=drift_estimate.stretch_ratio,
        offset_seconds=drift_estimate.offset_seconds,
    )


def _to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples.astype(np.float32, copy=False)
    return np.mean(samples, axis=1, dtype=np.float32)


def _stretch_samples(samples: np.ndarray, stretch_ratio: float, stretch_method: str) -> np.ndarray:
    if stretch_method == "resample":
        return _resample_by_ratio(samples, stretch_ratio)
    if stretch_method == "pitch_preserving":
        return _pitch_preserving_time_stretch(samples, stretch_ratio)
    raise ValueError("stretch_method must be one of: resample, pitch_preserving")


def _resample_by_ratio(samples: np.ndarray, stretch_ratio: float) -> np.ndarray:
    if samples.size == 0:
        return samples.astype(np.float32, copy=False)
    target_len = max(1, int(round(samples.shape[0] * stretch_ratio)))
    source_positions = np.linspace(0, samples.shape[0] - 1, num=samples.shape[0], dtype=np.float64)
    target_positions = np.linspace(0, samples.shape[0] - 1, num=target_len, dtype=np.float64)
    stretched = np.interp(target_positions, source_positions, samples.astype(np.float64, copy=False))
    return stretched.astype(np.float32, copy=False)


def _pitch_preserving_time_stretch(samples: np.ndarray, stretch_ratio: float) -> np.ndarray:
    if samples.size == 0:
        return samples.astype(np.float32, copy=False)

    import importlib

    try:
        librosa = importlib.import_module("librosa")
    except ModuleNotFoundError as exc:
        raise RuntimeError("pitch_preserving stretch requires librosa. Install with: pip install librosa") from exc
    rate = 1.0 / stretch_ratio
    stretched = librosa.effects.time_stretch(samples.astype(np.float32, copy=False), rate=rate)

    target_len = max(1, int(round(samples.shape[0] * stretch_ratio)))
    if stretched.shape[0] > target_len:
        stretched = stretched[:target_len]
    elif stretched.shape[0] < target_len:
        pad = np.zeros(target_len - stretched.shape[0], dtype=stretched.dtype)
        stretched = np.concatenate([stretched, pad])

    LOGGER.debug("pitch_preserving stretch applied rate=%.8f target_len=%d", rate, target_len)
    return stretched.astype(np.float32, copy=False)
