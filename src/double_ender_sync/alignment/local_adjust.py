from dataclasses import dataclass

import numpy as np

from double_ender_sync.analysis.vad import detect_speech_segments


@dataclass
class LocalAdjustmentEvent:
    split_time_seconds: float
    shift_seconds: float
    residual_ms: float
    confidence: float


@dataclass
class LocalAdjustmentResult:
    adjusted_samples: np.ndarray
    events: list[LocalAdjustmentEvent]
    warnings: list[str]


def apply_local_adjustment(
    globally_aligned_samples: np.ndarray,
    sample_rate: int,
    residual_events: list[dict],
    enabled: bool = False,
    residual_threshold_ms: float = 80.0,
    max_silence_search_seconds: float = 0.4,
    crossfade_ms: float = 12.0,
) -> LocalAdjustmentResult:
    """Apply optional phase-4 local timeline shifts at safe silence boundaries."""
    if not enabled:
        return LocalAdjustmentResult(
            adjusted_samples=globally_aligned_samples,
            events=[],
            warnings=["local_adjust disabled"],
        )

    adjusted = globally_aligned_samples.astype(np.float32, copy=True)
    events: list[LocalAdjustmentEvent] = []
    warnings: list[str] = []

    speech_segments = detect_speech_segments(adjusted, sample_rate=sample_rate)
    silence_regions = _compute_silence_regions(len(adjusted), sample_rate, speech_segments)

    skipped_rejected_count = 0
    for event in residual_events:
        if event.get("included_in_regression") is False:
            skipped_rejected_count += 1
            continue

        residual_ms = float(event.get("residual_ms", 0.0))
        if abs(residual_ms) < residual_threshold_ms:
            continue

        local_time = float(event.get("local_start", 0.0))
        split_time = _find_safe_split_time(local_time, silence_regions, max_silence_search_seconds)
        if split_time is None:
            warnings.append(
                f"no safe silence near {local_time:.3f}s for residual {residual_ms:.1f}ms; skipped"
            )
            continue

        shift_seconds = -residual_ms / 1000.0
        adjusted = _shift_from_time(adjusted, sample_rate, split_time, shift_seconds)
        _apply_crossfade(adjusted, sample_rate, split_time, crossfade_ms)
        confidence = max(0.0, 1.0 - min(1.0, abs(residual_ms) / 300.0))
        events.append(
            LocalAdjustmentEvent(
                split_time_seconds=split_time,
                shift_seconds=shift_seconds,
                residual_ms=residual_ms,
                confidence=confidence,
            )
        )

    if skipped_rejected_count:
        warnings.append(
            f"skipped {skipped_rejected_count} rejected drift anchor(s) during local adjustment"
        )

    if not events and not warnings:
        warnings.append("no residual events exceeded local adjustment threshold")

    return LocalAdjustmentResult(adjusted_samples=adjusted, events=events, warnings=warnings)


def _compute_silence_regions(
    total_samples: int,
    sample_rate: int,
    speech_segments: list,
) -> list[tuple[float, float]]:
    regions: list[tuple[float, float]] = []
    cursor = 0.0
    for segment in speech_segments:
        if segment.start > cursor:
            regions.append((cursor, segment.start))
        cursor = max(cursor, segment.end)
    total_seconds = total_samples / sample_rate
    if cursor < total_seconds:
        regions.append((cursor, total_seconds))
    return regions


def _find_safe_split_time(
    desired_time: float,
    silence_regions: list[tuple[float, float]],
    max_search_seconds: float,
) -> float | None:
    best_time = None
    best_distance = float("inf")
    for start, end in silence_regions:
        if end < desired_time - max_search_seconds or start > desired_time + max_search_seconds:
            continue
        candidate = min(max(desired_time, start), end)
        distance = abs(candidate - desired_time)
        if distance < best_distance:
            best_time = candidate
            best_distance = distance
    return best_time


def _shift_from_time(samples: np.ndarray, sample_rate: int, split_time: float, shift_seconds: float) -> np.ndarray:
    split_idx = int(round(split_time * sample_rate))
    shift_samples = int(round(shift_seconds * sample_rate))
    if shift_samples == 0 or split_idx >= len(samples):
        return samples

    out = np.copy(samples)
    tail = samples[split_idx:]

    if shift_samples > 0:
        insert = min(shift_samples, len(out) - split_idx)
        out[split_idx + insert :] = tail[: len(out) - (split_idx + insert)]
        out[split_idx : split_idx + insert] = 0.0
    else:
        pull = min(-shift_samples, len(tail))
        out[split_idx : split_idx + len(tail) - pull] = tail[pull:]
        out[len(out) - pull :] = 0.0
    return out


def _apply_crossfade(samples: np.ndarray, sample_rate: int, split_time: float, crossfade_ms: float) -> None:
    n = max(2, int(round((crossfade_ms / 1000.0) * sample_rate)))
    center = int(round(split_time * sample_rate))
    start = max(0, center - n // 2)
    end = min(len(samples), start + n)
    if end - start < 2:
        return
    window = np.hanning(end - start).astype(np.float32)
    baseline = samples[start:end].copy()
    samples[start:end] = baseline * window + baseline * (1.0 - window)
