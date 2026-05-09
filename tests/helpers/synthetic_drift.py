"""Deterministic synthetic drift fixtures for unit and calibration tests.

The helpers in this module intentionally generate only tiny in-memory arrays or
``AnchorMatch`` objects. They are safe for normal tests and avoid committing any
private podcast audio or large binary fixtures.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from double_ender_sync.analysis.drift import AnchorMatch


@dataclass(frozen=True)
class SyntheticAudioPair:
    """Small local/master pair with a known local-to-master time mapping."""

    local: np.ndarray
    master: np.ndarray
    sample_rate: int
    offset_seconds: float
    stretch_ratio: float


@dataclass(frozen=True)
class SyntheticAnchorSet:
    """Anchor fixtures plus the generating time mapping and duration."""

    matches: list[AnchorMatch]
    local_duration_seconds: float
    mapping: Callable[[float], float]


def anchor_match(
    local_start: float,
    master_start: float,
    *,
    duration_seconds: float = 1.0,
    confidence: float = 1.0,
    score: float = 0.9,
) -> AnchorMatch:
    """Create an ``AnchorMatch`` with consistent derived offset/end fields."""

    return AnchorMatch(
        local_start=float(local_start),
        local_end=float(local_start + duration_seconds),
        master_start=float(master_start),
        master_end=float(master_start + duration_seconds),
        offset_seconds=float(master_start - local_start),
        confidence=float(confidence),
        score=float(score),
    )


def matches_from_mapping(
    local_times: Iterable[float],
    mapping: Callable[[float], float],
    *,
    duration_seconds: float = 1.0,
    confidence: float = 1.0,
    score: float = 0.9,
) -> list[AnchorMatch]:
    """Build anchor matches by sampling a deterministic local-to-master mapping."""

    return [
        anchor_match(
            float(local_time),
            float(mapping(float(local_time))),
            duration_seconds=duration_seconds,
            confidence=confidence,
            score=score,
        )
        for local_time in local_times
    ]


def constant_drift_mapping(*, offset_seconds: float = 1.0, stretch_ratio: float = 1.0002) -> Callable[[float], float]:
    """Return a linear mapping with known offset and constant drift."""

    return lambda local_time: offset_seconds + (stretch_ratio * local_time)


def piecewise_drift_mapping(
    *,
    offset_seconds: float = 1.0,
    breakpoints: Sequence[float] = (300.0,),
    stretch_ratios: Sequence[float] = (1.00005, 1.00035),
) -> Callable[[float], float]:
    """Return a continuous piecewise-linear drift mapping."""

    if len(stretch_ratios) != len(breakpoints) + 1:
        raise ValueError("stretch_ratios must have one more item than breakpoints")
    sorted_breakpoints = tuple(float(point) for point in breakpoints)
    if sorted_breakpoints != tuple(sorted(sorted_breakpoints)):
        raise ValueError("breakpoints must be sorted in ascending order")

    def map_local_to_master(local_time: float) -> float:
        master_time = float(offset_seconds)
        segment_start = 0.0
        remaining_time = float(local_time)
        for breakpoint, stretch_ratio in zip(sorted_breakpoints, stretch_ratios):
            if remaining_time <= breakpoint:
                return master_time + (float(stretch_ratio) * (remaining_time - segment_start))
            master_time += float(stretch_ratio) * (breakpoint - segment_start)
            segment_start = breakpoint
        return master_time + (float(stretch_ratios[-1]) * (remaining_time - segment_start))

    return map_local_to_master


def smooth_spline_drift_mapping(
    *, offset_seconds: float = 1.0, duration_seconds: float = 600.0
) -> Callable[[float], float]:
    """Return a smooth, monotonic drift curve suitable for PCHIP/spline tests."""

    return lambda local_time: float(
        offset_seconds
        + local_time
        + (0.05 * np.sin((2.0 * np.pi * local_time) / duration_seconds))
    )


def _bounded_local_anchor_times(local_duration_seconds: float, interval_seconds: float) -> np.ndarray:
    """Return anchor start times that never exceed the advertised local duration."""

    if local_duration_seconds < 0.0:
        raise ValueError("local_duration_seconds must be non-negative")
    if interval_seconds <= 0.0:
        raise ValueError("interval_seconds must be greater than zero")

    # ``np.arange(stop + interval * 0.5, interval)`` rounds to the nearest grid
    # and can emit a final value beyond ``local_duration_seconds`` for off-grid
    # durations. Keep fixtures internally consistent by only returning times on
    # the interval grid that are inside the advertised local duration.
    count = int(np.floor(local_duration_seconds / interval_seconds)) + 1
    return np.arange(count, dtype=np.float64) * interval_seconds


def constant_drift_anchors(
    *,
    local_duration_seconds: float = 600.0,
    interval_seconds: float = 60.0,
    offset_seconds: float = 0.75,
    stretch_ratio: float = 1.0002,
) -> SyntheticAnchorSet:
    """Generate anchors for a known offset plus constant drift."""

    mapping = constant_drift_mapping(offset_seconds=offset_seconds, stretch_ratio=stretch_ratio)
    local_times = _bounded_local_anchor_times(local_duration_seconds, interval_seconds)
    return SyntheticAnchorSet(matches_from_mapping(local_times, mapping), local_duration_seconds, mapping)


def piecewise_drift_anchors(
    *,
    local_duration_seconds: float = 600.0,
    interval_seconds: float = 60.0,
    breakpoints: Sequence[float] = (300.0,),
    stretch_ratios: Sequence[float] = (1.00005, 1.00035),
) -> SyntheticAnchorSet:
    """Generate anchors for a continuous piecewise drift curve."""

    mapping = piecewise_drift_mapping(breakpoints=breakpoints, stretch_ratios=stretch_ratios)
    local_times = _bounded_local_anchor_times(local_duration_seconds, interval_seconds)
    return SyntheticAnchorSet(matches_from_mapping(local_times, mapping), local_duration_seconds, mapping)


def smooth_spline_drift_anchors(
    *,
    local_duration_seconds: float = 600.0,
    interval_seconds: float = 60.0,
) -> SyntheticAnchorSet:
    """Generate anchors for a smooth non-linear drift curve."""

    mapping = smooth_spline_drift_mapping(duration_seconds=local_duration_seconds)
    local_times = _bounded_local_anchor_times(local_duration_seconds, interval_seconds)
    return SyntheticAnchorSet(matches_from_mapping(local_times, mapping), local_duration_seconds, mapping)


def noisy_anchor_set(
    *,
    local_duration_seconds: float = 600.0,
    interval_seconds: float = 60.0,
    offset_seconds: float = 1.0,
    stretch_ratio: float = 1.0001,
    noise_ms: Sequence[float] = (
        0.0,
        12.0,
        -9.0,
        15.0,
        -11.0,
        8.0,
        -14.0,
        10.0,
        -7.0,
        9.0,
        -6.0,
    ),
) -> SyntheticAnchorSet:
    """Generate deterministic noisy observations around a linear ground truth."""

    mapping = constant_drift_mapping(offset_seconds=offset_seconds, stretch_ratio=stretch_ratio)
    local_times = _bounded_local_anchor_times(local_duration_seconds, interval_seconds)
    noise_values = list(noise_ms)
    if not noise_values:
        raise ValueError("noise_ms must contain at least one value")
    if len(noise_values) < len(local_times):
        repeats = int(np.ceil(len(local_times) / len(noise_values)))
        noise_values = (noise_values * repeats)[: len(local_times)]
    matches = [
        anchor_match(float(local_time), mapping(float(local_time)) + (float(noise_value) / 1000.0))
        for local_time, noise_value in zip(local_times, noise_values)
    ]
    return SyntheticAnchorSet(matches, local_duration_seconds, mapping)


def sparse_anchor_set(
    *,
    local_times: Sequence[float] = (0.0, 120.0, 240.0, 360.0, 480.0),
    local_duration_seconds: float = 600.0,
) -> SyntheticAnchorSet:
    """Generate too-few anchors for safe non-linear model selection."""

    mapping = piecewise_drift_mapping()
    return SyntheticAnchorSet(matches_from_mapping(local_times, mapping), local_duration_seconds, mapping)


def dropout_gap_anchor_set(*, local_duration_seconds: float = 300.0) -> SyntheticAnchorSet:
    """Generate anchors with a reconnect/dropout-like long gap between them."""

    mapping = constant_drift_mapping(offset_seconds=1.0, stretch_ratio=1.0)
    local_times = (0.0, 30.0, 60.0, 240.0)
    return SyntheticAnchorSet(matches_from_mapping(local_times, mapping), local_duration_seconds, mapping)


def offset_audio_pair(
    *,
    sample_rate: int = 8000,
    segment_duration: float = 0.8,
    silence_gap: float = 0.6,
    offset_seconds: float = 0.35,
    stretch_ratio: float = 1.0008,
    frequencies: Sequence[float] = (220.0, 330.0, 280.0, 410.0, 360.0, 260.0),
) -> SyntheticAudioPair:
    """Generate a tiny tone/silence local track rendered onto a master timeline."""

    segments: list[np.ndarray] = []
    for frequency in frequencies:
        t = np.arange(int(sample_rate * segment_duration)) / sample_rate
        tone = (0.22 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
        segments.append(tone)
        segments.append(np.zeros(int(sample_rate * silence_gap), dtype=np.float32))
    local = np.concatenate(segments)

    master_len = int(
        (stretch_ratio * (len(local) / sample_rate) + offset_seconds + 1.0) * sample_rate
    )
    master = np.zeros(master_len, dtype=np.float32)
    for index, sample in enumerate(local):
        target = int(round((stretch_ratio * (index / sample_rate) + offset_seconds) * sample_rate))
        if 0 <= target < master_len:
            master[target] += sample

    return SyntheticAudioPair(local, master, sample_rate, offset_seconds, stretch_ratio)
