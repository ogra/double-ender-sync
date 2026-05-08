import numpy as np
import pytest

from double_ender_sync.alignment.offset import estimate_initial_offset
from double_ender_sync.analysis.anchors import select_anchor_candidates
from double_ender_sync.analysis.drift import AnchorMatch, fit_linear_drift_model, match_anchors_for_drift
from double_ender_sync.analysis.vad import detect_speech_segments


def test_fit_linear_drift_model_estimates_stretch_ratio() -> None:
    sample_rate = 8000
    segment_duration = 0.8
    silence_gap = 0.6
    segments = []
    for f in [220, 330, 280, 410, 360, 260]:
        t = np.arange(int(sample_rate * segment_duration)) / sample_rate
        tone = (0.22 * np.sin(2 * np.pi * f * t)).astype(np.float32)
        segments.append(tone)
        segments.append(np.zeros(int(sample_rate * silence_gap), dtype=np.float32))
    local = np.concatenate(segments)

    true_stretch = 1.0008
    true_offset = 0.35
    master_len = int((true_stretch * (len(local) / sample_rate) + true_offset + 1.0) * sample_rate)
    master = np.zeros(master_len, dtype=np.float32)

    for i, sample in enumerate(local):
        target = int(round((true_stretch * (i / sample_rate) + true_offset) * sample_rate))
        if 0 <= target < master_len:
            master[target] += sample

    speech_segments = detect_speech_segments(local, sample_rate=sample_rate, frame_ms=40.0)
    anchors = select_anchor_candidates(local, sample_rate, speech_segments, min_anchor_duration=0.5, max_anchor_duration=0.9)
    initial = estimate_initial_offset(local, master, sample_rate, anchors)
    assert initial is not None

    matches = match_anchors_for_drift(local, master, sample_rate, anchors, initial.offset_seconds, search_radius_seconds=1.0)
    drift = fit_linear_drift_model(matches)

    assert drift is not None
    assert drift.anchor_count >= 2
    assert abs(drift.stretch_ratio - true_stretch) < 0.005
    assert abs(drift.offset_seconds - true_offset) < 0.1


def _match(local_start: float, master_start: float, confidence: float = 1.0) -> AnchorMatch:
    return AnchorMatch(
        local_start=local_start,
        local_end=local_start + 1.0,
        master_start=master_start,
        master_end=master_start + 1.0,
        offset_seconds=master_start - local_start,
        confidence=confidence,
        score=0.9,
    )


def test_fit_linear_drift_model_reports_very_weak_span_despite_high_anchor_count() -> None:
    matches = [_match(float(i * 10), float(i * 10 + 1.0)) for i in range(10)]

    drift = fit_linear_drift_model(matches, local_duration_seconds=600.0)

    assert drift is not None
    assert drift.anchor_count == 10
    assert drift.diagnostics is not None
    assert drift.diagnostics.local_span_seconds == 90.0
    assert drift.diagnostics.local_span_ratio == 0.15
    assert {warning.code for warning in drift.diagnostics.warnings} >= {"VERY_WEAK_DRIFT_ANCHOR_SPAN"}


def test_fit_linear_drift_model_rejects_outlier_deterministically_and_marks_matches() -> None:
    matches = [_match(local, local + 1.0) for local in [0.0, 100.0, 200.0, 400.0, 500.0]]
    outlier = _match(300.0, 321.0)
    matches.insert(3, outlier)

    drift = fit_linear_drift_model(matches, local_duration_seconds=600.0)

    assert drift is not None
    assert drift.anchor_count == 5
    assert drift.diagnostics is not None
    assert drift.diagnostics.outlier_count == 1
    assert outlier.included_in_regression is False
    assert outlier.rejected_reason == "residual_outlier"
    assert outlier.residual_ms is not None and outlier.residual_ms > 10000.0
    assert all(match.included_in_regression for match in matches if match is not outlier)
    assert {warning.code for warning in drift.diagnostics.warnings} >= {"DRIFT_OUTLIERS_REJECTED"}


def test_fit_linear_drift_model_full_timeline_residuals_stay_tight_with_many_anchors() -> None:
    true_stretch = 1.0002
    matches = [
        _match(float(local), true_stretch * float(local) + 0.75)
        for local in range(0, 601, 60)
    ]

    drift = fit_linear_drift_model(matches, local_duration_seconds=600.0)

    assert drift is not None
    assert drift.anchor_count == len(matches)
    assert drift.stretch_ratio == pytest.approx(np.float64(true_stretch).item())
    assert drift.residual_median_ms < 1e-6
    assert drift.residual_max_ms < 1e-6
    assert drift.diagnostics is not None
    assert drift.diagnostics.local_span_ratio == 1.0
    assert drift.diagnostics.warnings == []
