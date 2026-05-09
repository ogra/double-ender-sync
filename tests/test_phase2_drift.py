import json

import numpy as np
import pytest

from double_ender_sync.alignment.offset import estimate_initial_offset
from double_ender_sync.analysis.anchors import select_anchor_candidates
from double_ender_sync.analysis.drift import (
    AnchorMatch,
    fit_linear_drift_model,
    match_anchors_for_drift,
    select_drift_model,
)
from tests.helpers.synthetic_drift import (
    anchor_match as _match,
    constant_drift_anchors,
    dropout_gap_anchor_set,
    noisy_anchor_set,
    offset_audio_pair,
    piecewise_drift_anchors,
    piecewise_drift_mapping,
    smooth_spline_drift_anchors,
    smooth_spline_drift_mapping,
    sparse_anchor_set,
)
from double_ender_sync.analysis.vad import detect_speech_segments
from double_ender_sync.config import DriftModelConfig


def test_fit_linear_drift_model_estimates_stretch_ratio() -> None:
    audio_pair = offset_audio_pair()

    speech_segments = detect_speech_segments(
        audio_pair.local, sample_rate=audio_pair.sample_rate, frame_ms=40.0
    )
    anchors = select_anchor_candidates(
        audio_pair.local,
        audio_pair.sample_rate,
        speech_segments,
        min_anchor_duration=0.5,
        max_anchor_duration=0.9,
    )
    initial = estimate_initial_offset(
        audio_pair.local, audio_pair.master, audio_pair.sample_rate, anchors
    )
    assert initial is not None

    matches = match_anchors_for_drift(
        audio_pair.local,
        audio_pair.master,
        audio_pair.sample_rate,
        anchors,
        initial.offset_seconds,
        search_radius_seconds=1.0,
    )
    drift = fit_linear_drift_model(matches)

    assert drift is not None
    assert drift.anchor_count >= 2
    assert abs(drift.stretch_ratio - audio_pair.stretch_ratio) < 0.005
    assert abs(drift.offset_seconds - audio_pair.offset_seconds) < 0.1


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
    anchors = constant_drift_anchors()
    matches = anchors.matches

    drift = fit_linear_drift_model(matches, local_duration_seconds=anchors.local_duration_seconds)

    assert drift is not None
    assert drift.anchor_count == len(matches)
    assert drift.stretch_ratio == pytest.approx(np.float64(1.0002).item())
    assert drift.residual_median_ms < 1e-6
    assert drift.residual_max_ms < 1e-6
    assert drift.diagnostics is not None
    assert drift.diagnostics.local_span_ratio == 1.0
    assert drift.diagnostics.warnings == []


@pytest.mark.parametrize(
    "factory",
    [constant_drift_anchors, piecewise_drift_anchors, smooth_spline_drift_anchors, noisy_anchor_set],
)
def test_synthetic_anchor_generators_do_not_exceed_off_grid_local_duration(factory) -> None:
    anchors = factory(local_duration_seconds=650.0, interval_seconds=60.0)

    assert anchors.matches
    assert anchors.local_duration_seconds == 650.0
    assert max(match.local_start for match in anchors.matches) <= anchors.local_duration_seconds
    assert anchors.matches[-1].local_start == pytest.approx(600.0)


def test_linear_drift_mapping_matches_legacy_formula() -> None:
    drift = fit_linear_drift_model([
        _match(0.0, 0.75),
        _match(100.0, 100.77),
        _match(200.0, 200.79),
    ])

    assert drift is not None
    for local_time in [0.0, 12.5, 120.0, 240.0]:
        assert drift.map_local_to_master(local_time) == pytest.approx(
            drift.stretch_ratio * local_time + drift.offset_seconds
        )
    assert drift.local_rate_at(42.0) == pytest.approx(drift.stretch_ratio)


def test_linear_drift_residuals_and_report_metadata_are_explicit() -> None:
    drift = fit_linear_drift_model([
        _match(0.0, 1.0),
        _match(10.0, 11.01),
        _match(20.0, 21.02),
    ])

    assert drift is not None
    residuals = drift.residuals_ms([_match(5.0, drift.map_local_to_master(5.0) + 0.003)])
    assert residuals == pytest.approx([3.0])

    report = drift.to_report_dict()
    assert report["model_type"] == "linear"
    assert report["model_version"] == "1"
    assert report["model_parameters"] == {
        "offset_seconds": drift.offset_seconds,
        "stretch_ratio": drift.stretch_ratio,
    }
    assert report["offset_seconds"] == drift.offset_seconds
    assert report["stretch_ratio"] == drift.stretch_ratio
    assert report["local_rate_summary"] == {
        "min": drift.stretch_ratio,
        "max": drift.stretch_ratio,
        "mean": drift.stretch_ratio,
    }
    assert report["monotonicity_check"] == {"passed": True}


_piecewise_master_time = piecewise_drift_mapping()


def test_piecewise_linear_drift_recovers_single_breakpoint_and_reports_segments() -> None:
    from double_ender_sync.analysis.drift import PiecewiseLinearDrift, select_drift_model
    from double_ender_sync.config import DriftModelConfig

    matches = piecewise_drift_anchors().matches
    config = DriftModelConfig(
        drift_model="piecewise_linear",
        allow_nonlinear_drift=True,
        max_rate_change_ppm=500.0,
        min_residual_improvement_ms=1.0,
    )

    drift = select_drift_model(matches, config, local_duration_seconds=600.0)

    assert isinstance(drift, PiecewiseLinearDrift)
    assert len(drift.breakpoints) == 1
    assert drift.breakpoints[0] == pytest.approx(300.0, abs=35.0)
    assert drift.segments[0].stretch_ratio == pytest.approx(1.00005, abs=5e-5)
    assert drift.segments[1].stretch_ratio == pytest.approx(1.00035, abs=5e-5)
    assert drift.residual_median_ms < 1e-6

    report = drift.to_report_dict()
    assert report["model_type"] == "piecewise_linear"
    assert report["breakpoints"] == pytest.approx([drift.breakpoints[0]])
    assert len(report["segments"]) == 2
    assert report["model_parameters"]["linear_baseline"]["stretch_ratio"] == pytest.approx(drift.linear_baseline.stretch_ratio)


def test_piecewise_linear_sparse_anchors_falls_back_to_linear_with_reason() -> None:
    from double_ender_sync.analysis.drift import LinearDrift, select_drift_model
    from double_ender_sync.config import DriftModelConfig

    matches = sparse_anchor_set().matches
    config = DriftModelConfig(drift_model="auto", allow_nonlinear_drift=True)

    drift = select_drift_model(matches, config, local_duration_seconds=600.0)

    assert isinstance(drift, LinearDrift)
    assert drift.fallback_reason is not None
    assert "min_anchors_for_piecewise" in drift.fallback_reason


def test_noisy_anchor_set_rejects_empty_noise_sequence() -> None:
    with pytest.raises(ValueError, match="noise_ms must contain at least one value"):
        noisy_anchor_set(noise_ms=())


def test_auto_default_thresholds_do_not_overfit_noisy_linear_anchors() -> None:
    from double_ender_sync.analysis.drift import LinearDrift

    anchors = noisy_anchor_set()
    config = DriftModelConfig(drift_model="auto", allow_nonlinear_drift=True)

    drift = select_drift_model(
        anchors.matches, config, local_duration_seconds=anchors.local_duration_seconds
    )

    assert isinstance(drift, LinearDrift)
    assert drift.model_selection_policy == "nonlinear_experimental"
    assert drift.fallback_reason is not None
    assert "improvement" in drift.fallback_reason


def test_auto_default_thresholds_do_not_overfit_sparse_piecewise_anchors() -> None:
    from double_ender_sync.analysis.drift import LinearDrift

    anchors = sparse_anchor_set()
    config = DriftModelConfig(drift_model="auto", allow_nonlinear_drift=True)

    drift = select_drift_model(
        anchors.matches, config, local_duration_seconds=anchors.local_duration_seconds
    )

    assert isinstance(drift, LinearDrift)
    assert drift.model_selection_policy == "nonlinear_experimental"
    assert drift.fallback_reason is not None
    assert "min_anchors_for_piecewise" in drift.fallback_reason


def test_piecewise_linear_rejects_implausible_rate_change_with_useful_reason() -> None:
    from double_ender_sync.analysis.drift import LinearDrift, select_drift_model
    from double_ender_sync.config import DriftModelConfig

    def implausible(local_time: float) -> float:
        if local_time <= 300.0:
            return local_time
        return 300.0 + 1.01 * (local_time - 300.0)

    matches = [_match(float(local), implausible(float(local))) for local in range(0, 601, 60)]
    config = DriftModelConfig(
        drift_model="piecewise_linear",
        allow_nonlinear_drift=True,
        max_abs_rate_deviation_ppm=20_000.0,
        max_rate_change_ppm=500.0,
        min_residual_improvement_ms=1.0,
    )

    drift = select_drift_model(matches, config, local_duration_seconds=600.0)

    assert isinstance(drift, LinearDrift)
    assert drift.fallback_reason is not None
    assert "max_rate_change_ppm" in drift.fallback_reason


_two_breakpoint_master_time = piecewise_drift_mapping(
    breakpoints=(200.0, 400.0),
    stretch_ratios=(1.00005, 1.00035, 0.99995),
)


def test_piecewise_linear_honors_max_breakpoints_above_one() -> None:
    from double_ender_sync.analysis.drift import PiecewiseLinearDrift, select_drift_model
    from double_ender_sync.config import DriftModelConfig

    matches = [
        _match(float(local), _two_breakpoint_master_time(float(local)))
        for local in range(0, 601, 40)
    ]
    config = DriftModelConfig(
        drift_model="piecewise_linear",
        allow_nonlinear_drift=True,
        max_breakpoints=2,
        max_rate_change_ppm=500.0,
        min_residual_improvement_ms=1.0,
        min_relative_residual_improvement=0.1,
    )

    drift = select_drift_model(matches, config, local_duration_seconds=600.0)

    assert isinstance(drift, PiecewiseLinearDrift)
    assert len(drift.breakpoints) == 2
    assert drift.breakpoints[0] == pytest.approx(200.0, abs=40.0)
    assert drift.breakpoints[1] == pytest.approx(400.0, abs=70.0)
    assert len(drift.segments) == 3
    assert drift.segments[0].stretch_ratio == pytest.approx(1.00005, abs=1e-4)
    assert drift.segments[1].stretch_ratio == pytest.approx(1.00035, abs=1e-4)
    assert drift.segments[2].stretch_ratio == pytest.approx(0.99995, abs=1.5e-4)


def test_select_drift_model_reports_only_evaluated_candidates() -> None:
    from double_ender_sync.analysis.drift import LinearDrift, select_drift_model
    from double_ender_sync.config import DriftModelConfig

    matches = [_match(float(local), float(local) + 1.0) for local in range(0, 601, 60)]

    default_drift = select_drift_model(matches, DriftModelConfig(), local_duration_seconds=600.0)
    assert isinstance(default_drift, LinearDrift)
    assert default_drift.candidate_models == ("linear",)
    assert default_drift.model_selection_policy == "linear_default"

    linear_drift = select_drift_model(
        matches,
        DriftModelConfig(drift_model="linear", allow_nonlinear_drift=True),
        local_duration_seconds=600.0,
    )
    assert isinstance(linear_drift, LinearDrift)
    assert linear_drift.candidate_models == ("linear",)
    assert linear_drift.model_selection_policy == "linear_requested"

    skipped_piecewise = select_drift_model(
        matches,
        DriftModelConfig(allow_nonlinear_drift=True, max_breakpoints=0),
        local_duration_seconds=600.0,
    )
    assert isinstance(skipped_piecewise, LinearDrift)
    assert skipped_piecewise.candidate_models == ("linear", "spline")
    assert skipped_piecewise.model_selection_policy == "nonlinear_experimental"
    assert skipped_piecewise.selected_model_reason == "linear control model retained after non-linear evaluation"
    assert skipped_piecewise.fallback_reason is not None


def test_linear_drift_reports_unsupported_region_for_long_trusted_anchor_gap() -> None:
    anchors = dropout_gap_anchor_set()
    matches = anchors.matches
    config = DriftModelConfig(max_anchor_gap_seconds=90.0)

    drift = select_drift_model(matches, config=config, local_duration_seconds=300.0)

    assert drift is not None
    report = drift.to_report_dict()
    unsupported_regions = report["unsupported_regions"]
    assert unsupported_regions == [
        {
            "code": "ANCHOR_GAP_UNSUPPORTED_REGION",
            "reason": "anchor_gap_dropout_candidate",
            "local_start": 61.0,
            "local_end": 240.0,
            "master_start": 62.0,
            "master_end": 241.0,
            "local_gap_seconds": 179.0,
            "master_gap_seconds": 179.0,
            "threshold_seconds": 90.0,
        }
    ]
    assert {warning["code"] for warning in report["warnings"]} == {"ANCHOR_GAP_UNSUPPORTED_REGION"}
    assert drift.diagnostics is not None
    assert {warning.code for warning in drift.diagnostics.warnings} >= {"ANCHOR_GAP_UNSUPPORTED_REGION"}


def test_linear_drift_does_not_report_unsupported_region_for_normal_anchor_gaps() -> None:
    matches = [
        _match(0.0, 1.0),
        _match(30.0, 31.0),
        _match(60.0, 61.0),
        _match(90.0, 91.0),
    ]
    config = DriftModelConfig(max_anchor_gap_seconds=90.0)

    drift = select_drift_model(matches, config=config, local_duration_seconds=120.0)

    assert drift is not None
    report = drift.to_report_dict()
    assert report["unsupported_regions"] == []
    assert all(warning["code"] != "ANCHOR_GAP_UNSUPPORTED_REGION" for warning in report["warnings"])
    assert drift.diagnostics is not None
    assert all(warning.code != "ANCHOR_GAP_UNSUPPORTED_REGION" for warning in drift.diagnostics.warnings)



def test_anchor_gap_unsupported_region_uses_space_between_trusted_anchors() -> None:
    matches = [
        AnchorMatch(
            local_start=0.0,
            local_end=100.0,
            master_start=1.0,
            master_end=101.0,
            offset_seconds=1.0,
            confidence=1.0,
            score=0.9,
        ),
        AnchorMatch(
            local_start=105.0,
            local_end=106.0,
            master_start=106.0,
            master_end=107.0,
            offset_seconds=1.0,
            confidence=1.0,
            score=0.9,
        ),
        AnchorMatch(
            local_start=240.0,
            local_end=241.0,
            master_start=241.0,
            master_end=242.0,
            offset_seconds=1.0,
            confidence=1.0,
            score=0.9,
        ),
    ]
    config = DriftModelConfig(max_anchor_gap_seconds=90.0)

    drift = select_drift_model(matches, config=config, local_duration_seconds=300.0)

    assert drift is not None
    unsupported_regions = drift.to_report_dict()["unsupported_regions"]
    assert unsupported_regions == [
        {
            "code": "ANCHOR_GAP_UNSUPPORTED_REGION",
            "reason": "anchor_gap_dropout_candidate",
            "local_start": 106.0,
            "local_end": 240.0,
            "master_start": 107.0,
            "master_end": 241.0,
            "local_gap_seconds": 134.0,
            "master_gap_seconds": 134.0,
            "threshold_seconds": 90.0,
        }
    ]


def test_anchor_gap_warning_time_accepts_numpy_real_master_start() -> None:
    matches = [
        AnchorMatch(
            local_start=np.float32(0.0),
            local_end=np.float32(60.0),
            master_start=np.float32(1.0),
            master_end=np.float32(61.0),
            offset_seconds=np.float32(1.0),
            confidence=1.0,
            score=0.9,
        ),
        AnchorMatch(
            local_start=np.float32(240.0),
            local_end=np.float32(241.0),
            master_start=np.float32(241.0),
            master_end=np.float32(242.0),
            offset_seconds=np.float32(1.0),
            confidence=1.0,
            score=0.9,
        ),
    ]
    config = DriftModelConfig(max_anchor_gap_seconds=90.0)

    drift = select_drift_model(matches, config=config, local_duration_seconds=300.0)

    assert drift is not None
    assert drift.diagnostics is not None
    gap_warnings = [
        warning
        for warning in drift.diagnostics.warnings
        if warning.code == "ANCHOR_GAP_UNSUPPORTED_REGION"
    ]
    assert len(gap_warnings) == 1
    assert gap_warnings[0].time_seconds == pytest.approx(61.0)
    assert json.loads(json.dumps(drift.to_report_dict()["unsupported_regions"])) == [
        {
            "code": "ANCHOR_GAP_UNSUPPORTED_REGION",
            "reason": "anchor_gap_dropout_candidate",
            "local_start": 60.0,
            "local_end": 240.0,
            "master_start": 61.0,
            "master_end": 241.0,
            "local_gap_seconds": 180.0,
            "master_gap_seconds": 180.0,
            "threshold_seconds": 90.0,
        }
    ]


def test_piecewise_drift_preserves_linear_anchor_gap_diagnostics() -> None:
    from double_ender_sync.analysis.drift import PiecewiseLinearDrift

    local_times = [0.0, 60.0, 120.0, 300.0, 360.0, 420.0, 480.0, 540.0, 600.0]
    matches = [_match(local, _piecewise_master_time(local)) for local in local_times]
    config = DriftModelConfig(
        drift_model="piecewise_linear",
        allow_nonlinear_drift=True,
        max_anchor_gap_seconds=90.0,
        max_rate_change_ppm=500.0,
        min_residual_improvement_ms=1.0,
    )

    drift = select_drift_model(matches, config=config, local_duration_seconds=600.0)

    assert isinstance(drift, PiecewiseLinearDrift)
    report = drift.to_report_dict()
    assert report["unsupported_regions"] == [
        {
            "code": "ANCHOR_GAP_UNSUPPORTED_REGION",
            "reason": "anchor_gap_dropout_candidate",
            "local_start": 121.0,
            "local_end": 300.0,
            "master_start": _piecewise_master_time(120.0) + 1.0,
            "master_end": _piecewise_master_time(300.0),
            "local_gap_seconds": 179.0,
            "master_gap_seconds": pytest.approx(_piecewise_master_time(300.0) - (_piecewise_master_time(120.0) + 1.0)),
            "threshold_seconds": 90.0,
        }
    ]
    assert {warning["code"] for warning in report["warnings"]} >= {"ANCHOR_GAP_UNSUPPORTED_REGION"}

def test_drift_model_config_as_dict_reflects_resolved_selection_policy() -> None:
    default_config = DriftModelConfig().as_dict()
    assert default_config["selection_policy"] == "linear_default"
    assert "monotonicity_rate_epsilon" in default_config
    assert default_config["max_anchor_gap_seconds"] is None
    assert DriftModelConfig(max_anchor_gap_seconds=12.5).as_dict()["max_anchor_gap_seconds"] == 12.5
    assert "monotonicity_epsilon_seconds" not in default_config

    linear_requested = DriftModelConfig(drift_model="linear", allow_nonlinear_drift=True).as_dict()
    assert linear_requested["resolved_model"] == "linear"
    assert linear_requested["selection_policy"] == "linear_requested"

    auto_piecewise_enabled = DriftModelConfig(allow_nonlinear_drift=True).as_dict()
    assert auto_piecewise_enabled["resolved_model"] == "auto"
    assert auto_piecewise_enabled["selection_policy"] == "nonlinear_experimental"


def test_piecewise_breakpoint_search_keeps_current_best_score() -> None:
    from double_ender_sync.analysis.drift import _choose_piecewise_breakpoints, _fit_continuous_piecewise_model
    from double_ender_sync.config import DriftModelConfig

    local_times = [float(index * 10) for index in range(10)]
    master_times = [
        1.957,
        12.759,
        21.609,
        31.109,
        42.742,
        52.933,
        62.815,
        72.885,
        82.574,
        92.337,
    ]
    matches = [_match(local, master) for local, master in zip(local_times, master_times)]
    linear = fit_linear_drift_model(matches)
    assert linear is not None
    config = DriftModelConfig(
        allow_nonlinear_drift=True,
        max_breakpoints=2,
        min_anchors_for_piecewise=6,
        min_anchors_per_segment=2,
        min_residual_improvement_ms=0.0,
        min_relative_residual_improvement=0.0,
        max_abs_rate_deviation_ppm=1_000_000_000.0,
        max_rate_change_ppm=1_000_000_000.0,
    )

    breakpoints = _choose_piecewise_breakpoints(matches, linear, config)

    assert breakpoints == [65.0]
    selected_model = _fit_continuous_piecewise_model(matches, tuple(breakpoints), linear, None)
    worsening_model = _fit_continuous_piecewise_model(matches, (20.0, 65.0), linear, None)
    assert worsening_model.residual_median_ms > selected_model.residual_median_ms


def test_drift_model_config_skips_piecewise_anchor_validation_when_piecewise_disabled() -> None:
    from double_ender_sync.config import DriftModelConfig

    disabled_by_gate = DriftModelConfig(
        allow_nonlinear_drift=False,
        min_anchors_for_piecewise=1,
        min_anchors_per_segment=1,
    )
    assert disabled_by_gate.as_dict()["selection_policy"] == "linear_default"

    disabled_by_linear_request = DriftModelConfig(
        drift_model="linear",
        allow_nonlinear_drift=True,
        min_anchors_for_piecewise=1,
        min_anchors_per_segment=1,
    )
    assert disabled_by_linear_request.as_dict()["selection_policy"] == "linear_requested"

    disabled_by_zero_breakpoints = DriftModelConfig(
        allow_nonlinear_drift=True,
        max_breakpoints=0,
        min_anchors_for_piecewise=1,
        min_anchors_per_segment=1,
    )
    assert disabled_by_zero_breakpoints.as_dict()["max_breakpoints"] == 0

    with pytest.raises(ValueError, match="min_anchors_for_piecewise"):
        DriftModelConfig(
            allow_nonlinear_drift=True,
            max_breakpoints=1,
            min_anchors_for_piecewise=1,
            min_anchors_per_segment=1,
        )


def test_drift_model_config_skips_spline_validation_for_piecewise_only_policy() -> None:
    from double_ender_sync.config import DriftModelConfig

    config = DriftModelConfig(
        drift_model="piecewise_linear",
        allow_nonlinear_drift=True,
        min_anchors_for_spline=1,
        min_knot_spacing_seconds=-1.0,
        spline_validation_sample_count=1,
    )

    assert config.as_dict()["selection_policy"] == "piecewise_experimental"


@pytest.mark.parametrize(
    ("drift_model", "allow_nonlinear_drift"),
    [("linear", False), ("linear", True), ("auto", False), ("piecewise_linear", True)],
)
def test_drift_model_config_rejects_invalid_spline_knot_source_when_spline_is_not_evaluated(
    drift_model: str,
    allow_nonlinear_drift: bool,
) -> None:
    from double_ender_sync.config import DriftModelConfig

    with pytest.raises(ValueError, match="spline_knot_source"):
        DriftModelConfig(
            drift_model=drift_model,  # type: ignore[arg-type]
            allow_nonlinear_drift=allow_nonlinear_drift,
            spline_knot_source="bad",  # type: ignore[arg-type]
        )

def test_drift_model_config_rejects_spline_piecewise_boundaries_without_breakpoints() -> None:
    from double_ender_sync.config import DriftModelConfig

    with pytest.raises(ValueError, match="piecewise_boundaries.*max_breakpoints"):
        DriftModelConfig(
            drift_model="spline",
            allow_nonlinear_drift=True,
            spline_knot_source="piecewise_boundaries",
            max_breakpoints=0,
        )


def _smooth_spline_master_time(local_time: float, duration: float = 600.0) -> float:
    return smooth_spline_drift_mapping(duration_seconds=duration)(local_time)


def test_auto_nonlinear_selection_reports_auto_policy_for_accepted_piecewise() -> None:
    from double_ender_sync.analysis.drift import PiecewiseLinearDrift, select_drift_model
    from double_ender_sync.config import DriftModelConfig

    matches = [_match(float(local), _piecewise_master_time(float(local))) for local in range(0, 601, 60)]
    config = DriftModelConfig(
        drift_model="auto",
        allow_nonlinear_drift=True,
        max_rate_change_ppm=500.0,
        min_residual_improvement_ms=1.0,
    )

    drift = select_drift_model(matches, config, local_duration_seconds=600.0)

    assert isinstance(drift, PiecewiseLinearDrift)
    assert drift.model_selection_policy == "nonlinear_experimental"
    assert drift.to_report_dict()["model_selection_policy"] == "nonlinear_experimental"


def test_auto_nonlinear_selection_reports_auto_policy_for_accepted_spline() -> None:
    from double_ender_sync.analysis.drift import SplineDrift, select_drift_model
    from double_ender_sync.config import DriftModelConfig

    matches = smooth_spline_drift_anchors().matches
    config = DriftModelConfig(
        drift_model="auto",
        allow_nonlinear_drift=True,
        max_breakpoints=0,
        min_anchors_for_spline=5,
        min_knot_spacing_seconds=90.0,
        min_residual_improvement_ms=1.0,
        min_relative_residual_improvement=0.1,
        max_rate_change_ppm=1200.0,
    )

    drift = select_drift_model(matches, config, local_duration_seconds=600.0)

    assert isinstance(drift, SplineDrift)
    assert drift.model_selection_policy == "nonlinear_experimental"
    assert drift.to_report_dict()["model_selection_policy"] == "nonlinear_experimental"


def test_spline_drift_improves_smoothly_varying_drift_and_reports_knots() -> None:
    from double_ender_sync.analysis.drift import SplineDrift, select_drift_model
    from double_ender_sync.config import DriftModelConfig

    matches = smooth_spline_drift_anchors().matches
    config = DriftModelConfig(
        drift_model="spline",
        allow_nonlinear_drift=True,
        min_anchors_for_spline=5,
        min_knot_spacing_seconds=90.0,
        min_residual_improvement_ms=1.0,
        min_relative_residual_improvement=0.1,
        max_rate_change_ppm=1200.0,
    )

    drift = select_drift_model(matches, config, local_duration_seconds=600.0)

    assert isinstance(drift, SplineDrift)
    assert drift.interpolation_method == "pchip"
    assert drift.knot_source == "anchors"
    assert drift.residual_median_ms < 1e-6
    assert drift.linear_baseline.residual_median_ms > 10.0
    assert all(
        drift.map_local_to_master(left) < drift.map_local_to_master(right)
        for left, right in zip(np.linspace(0.0, 590.0, 20), np.linspace(10.0, 600.0, 20))
    )

    report = drift.to_report_dict()
    assert report["model_type"] == "spline"
    assert report["model_parameters"]["interpolation_method"] == "pchip"
    assert report["model_parameters"]["knot_source"] == "anchors"
    assert report["model_parameters"]["knot_decimation_applied"] is True
    assert len(report["knots"]) >= 5
    assert len(report["knot_residual_summaries"]) == len(report["knots"])
    assert {"anchor_count", "residual_median_ms", "residual_max_ms"}.issubset(
        report["knot_residual_summaries"][0]
    )
    assert report["knot_residual_summaries"] != report["knots"]
    assert report["monotonicity_check"]["passed"] is True
    assert report["local_rate_summary"]["min"] > 0.0


def test_spline_drift_reports_no_anchor_decimation_when_spacing_keeps_all_knots() -> None:
    from double_ender_sync.analysis.drift import SplineDrift, select_drift_model
    from double_ender_sync.config import DriftModelConfig

    matches = [
        _match(float(local), _smooth_spline_master_time(float(local), duration=300.0))
        for local in range(0, 301, 60)
    ]
    config = DriftModelConfig(
        drift_model="spline",
        allow_nonlinear_drift=True,
        min_anchors_for_spline=5,
        min_knot_spacing_seconds=0.0,
        min_residual_improvement_ms=0.0,
        min_relative_residual_improvement=0.0,
        max_abs_rate_deviation_ppm=10_000.0,
        max_rate_change_ppm=10_000.0,
    )

    drift = select_drift_model(matches, config, local_duration_seconds=300.0)

    assert isinstance(drift, SplineDrift)
    assert drift.to_report_dict()["model_parameters"]["knot_decimation_applied"] is False


def test_explicit_spline_can_prefit_piecewise_boundary_knots() -> None:
    from double_ender_sync.analysis.drift import SplineDrift, select_drift_model
    from double_ender_sync.config import DriftModelConfig

    matches = [_match(float(local), _piecewise_master_time(float(local))) for local in range(0, 601, 60)]
    config = DriftModelConfig(
        drift_model="spline",
        allow_nonlinear_drift=True,
        spline_knot_source="piecewise_boundaries",
        min_residual_improvement_ms=0.1,
        min_relative_residual_improvement=0.01,
        max_abs_rate_deviation_ppm=10_000.0,
        max_rate_change_ppm=10_000.0,
    )

    drift = select_drift_model(matches, config, local_duration_seconds=600.0)

    assert isinstance(drift, SplineDrift)
    assert drift.knot_source == "piecewise_boundaries"
    assert drift.candidate_models == ("linear", "piecewise_linear", "spline")
    assert drift.baseline_model_type == "linear"


def test_explicit_spline_piecewise_boundary_fallback_combines_prefit_and_spline_reasons() -> None:
    from double_ender_sync.analysis.drift import LinearDrift, select_drift_model
    from double_ender_sync.config import DriftModelConfig

    matches = [_match(float(local), _piecewise_master_time(float(local))) for local in range(0, 301, 60)]
    config = DriftModelConfig(
        drift_model="spline",
        allow_nonlinear_drift=True,
        spline_knot_source="piecewise_boundaries",
        min_anchors_for_piecewise=20,
        min_anchors_per_segment=3,
        min_anchors_for_spline=3,
    )

    drift = select_drift_model(matches, config, local_duration_seconds=300.0)

    assert isinstance(drift, LinearDrift)
    assert drift.fallback_reason is not None
    assert "min_anchors_for_piecewise" in drift.fallback_reason
    assert "no accepted piecewise model is available" in drift.fallback_reason


def test_spline_drift_reuses_supplied_pchip_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    import double_ender_sync.analysis.drift as drift_module
    from scipy.interpolate import PchipInterpolator

    from double_ender_sync.analysis.drift import LinearDrift

    knot_local_times = (0.0, 100.0, 200.0)
    knot_master_times = (1.0, 101.05, 201.2)
    interpolator = PchipInterpolator(
        np.array(knot_local_times, dtype=np.float64),
        np.array(knot_master_times, dtype=np.float64),
        extrapolate=True,
    )
    derivative = interpolator.derivative()

    def fail_if_reconstructed(*args: object, **kwargs: object) -> object:
        raise AssertionError("SplineDrift should reuse the supplied PCHIP interpolator")

    monkeypatch.setattr(drift_module, "PchipInterpolator", fail_if_reconstructed)

    drift = drift_module.SplineDrift(
        knot_local_times=knot_local_times,
        knot_master_times=knot_master_times,
        interpolation_method="pchip",
        knot_source="anchors",
        anchor_count=3,
        residual_median_ms=1.0,
        residual_max_ms=2.0,
        linear_baseline=LinearDrift(
            offset_seconds=1.0,
            stretch_ratio=1.0,
            anchor_count=3,
            residual_median_ms=10.0,
            residual_max_ms=20.0,
        ),
        baseline_model_type="linear",
        validation_sample_count=64,
        monotonicity_min_step_seconds=1.0,
        local_rate_min=1.0,
        local_rate_max=1.002,
        local_rate_mean=1.001,
        local_rate_change_max_ppm=10.0,
        pchip_interpolator=interpolator,
        pchip_derivative=derivative,
    )

    assert drift.map_local_to_master(50.0) == pytest.approx(float(interpolator(50.0)))
    assert drift.local_rate_at(50.0) == pytest.approx(float(derivative(50.0)))


def test_spline_drift_construction_value_error_returns_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    import double_ender_sync.analysis.drift as drift_module
    from double_ender_sync.analysis.drift import LinearDrift, fit_spline_drift_model
    from double_ender_sync.config import DriftModelConfig

    matches = [_match(float(local), float(local) + 1.0) for local in range(0, 301, 60)]
    for match in matches:
        match.included_in_regression = True
    linear = LinearDrift(
        offset_seconds=1.0,
        stretch_ratio=1.0,
        anchor_count=len(matches),
        residual_median_ms=100.0,
        residual_max_ms=100.0,
    )
    config = DriftModelConfig(
        drift_model="spline",
        allow_nonlinear_drift=True,
        min_anchors_for_spline=3,
        min_knot_spacing_seconds=0.0,
        min_residual_improvement_ms=0.0,
        min_relative_residual_improvement=0.0,
        max_abs_rate_deviation_ppm=1_000_000.0,
        max_rate_change_ppm=1_000_000.0,
    )

    def raise_value_error(*args: object, **kwargs: object) -> object:
        raise ValueError("SplineDrift master knots must be strictly increasing")

    monkeypatch.setattr(drift_module, "SplineDrift", raise_value_error)

    result = fit_spline_drift_model(matches, linear, config, local_duration_seconds=300.0)

    assert result.model is None
    assert result.fallback_reason is not None
    assert "spline candidate rejected: SplineDrift construction failed" in result.fallback_reason
    assert "master knots must be strictly increasing" in result.fallback_reason


def test_spline_drift_rejects_pathological_nonmonotonic_anchor_mapping() -> None:
    from double_ender_sync.analysis.drift import LinearDrift, fit_spline_drift_model
    from double_ender_sync.config import DriftModelConfig

    matches = [
        _match(0.0, 0.0),
        _match(10.0, 10.0),
        _match(20.0, 20.0),
        _match(30.0, 19.0),
        _match(40.0, 40.0),
        _match(50.0, 50.0),
    ]
    for match in matches:
        match.included_in_regression = True
    linear = LinearDrift(
        offset_seconds=0.0,
        stretch_ratio=1.0,
        anchor_count=len(matches),
        residual_median_ms=100.0,
        residual_max_ms=1000.0,
    )
    config = DriftModelConfig(
        drift_model="spline",
        allow_nonlinear_drift=True,
        min_anchors_for_spline=3,
        min_knot_spacing_seconds=0.0,
        min_residual_improvement_ms=0.0,
        min_relative_residual_improvement=0.0,
        max_abs_rate_deviation_ppm=1_000_000.0,
        max_rate_change_ppm=1_000_000.0,
    )

    result = fit_spline_drift_model(matches, linear, config, local_duration_seconds=50.0)

    assert result.model is None
    assert result.fallback_reason is not None
    assert "spline candidate rejected" in result.fallback_reason
    assert "monotonicity" in result.fallback_reason or "local-rate" in result.fallback_reason


def _smooth_kalman_master_time(local_time: float, duration: float = 600.0) -> float:
    phase = local_time / duration
    return 1.25 + local_time + 0.040 * np.sin(2.0 * np.pi * phase)


def test_kalman_drift_research_smoother_tracks_changing_offset_and_reports_uncertainty() -> None:
    from double_ender_sync.analysis.drift import KalmanDrift, select_drift_model
    from double_ender_sync.config import DriftModelConfig

    matches = [
        _match(float(local), _smooth_kalman_master_time(float(local)), confidence=0.95)
        for local in range(0, 601, 60)
    ]
    config = DriftModelConfig(
        drift_model="kalman",
        allow_nonlinear_drift=True,
        min_anchors_for_kalman=5,
        min_residual_improvement_ms=1.0,
        min_relative_residual_improvement=0.1,
        kalman_process_offset_noise_ms=20.0,
        kalman_process_rate_noise_ppm=500.0,
        kalman_observation_noise_ms=2.0,
        max_abs_rate_deviation_ppm=1000.0,
    )

    drift = select_drift_model(matches, config, local_duration_seconds=600.0)

    assert isinstance(drift, KalmanDrift)
    assert drift.model_selection_policy == "kalman_research_experimental"
    assert drift.residual_median_ms < drift.linear_baseline.residual_median_ms
    assert drift.map_local_to_master(150.0) == pytest.approx(_smooth_kalman_master_time(150.0), abs=0.010)
    assert all(
        drift.map_local_to_master(left) < drift.map_local_to_master(right)
        for left, right in zip(np.linspace(0.0, 590.0, 20), np.linspace(10.0, 600.0, 20))
    )

    report = drift.to_report_dict()
    assert report["model_type"] == "kalman"
    assert report["model_version"] == "research-1"
    assert report["model_parameters"]["state_definition"]["offset_seconds"] == "master_time - local_time"
    assert report["uncertainty_summary"]["median_one_sigma_ms"] >= 0.0
    assert report["covariance_summary"]["max_rate_std_ppm"] >= 0.0
    assert len(report["uncertainty_bands"]) == len(matches)
    assert report["uncertainty_bands"][0]["offset_lower_seconds"] <= report["uncertainty_bands"][0]["offset_seconds"]
    assert report["uncertainty_bands"][0]["offset_upper_seconds"] >= report["uncertainty_bands"][0]["offset_seconds"]
    assert len(report["anchor_residuals_ms"]) == len(matches)


def test_kalman_drift_preserves_linear_outlier_rejection_when_marking_residuals() -> None:
    from double_ender_sync.analysis.drift import KalmanDrift, select_drift_model
    from double_ender_sync.config import DriftModelConfig

    matches = [
        _match(float(local), _smooth_kalman_master_time(float(local)), confidence=0.95)
        for local in range(0, 601, 60)
    ]
    outlier = _match(300.0, 321.0, confidence=0.95)
    matches.insert(6, outlier)
    config = DriftModelConfig(
        drift_model="kalman",
        allow_nonlinear_drift=True,
        min_anchors_for_kalman=5,
        min_residual_improvement_ms=1.0,
        min_relative_residual_improvement=0.1,
        kalman_process_offset_noise_ms=20.0,
        kalman_process_rate_noise_ppm=500.0,
        kalman_observation_noise_ms=2.0,
        max_abs_rate_deviation_ppm=1000.0,
    )

    drift = select_drift_model(matches, config, local_duration_seconds=600.0)

    assert isinstance(drift, KalmanDrift)
    assert outlier.residual_ms is not None
    assert outlier.included_in_regression is False
    assert outlier.rejected_reason == "residual_outlier"
    assert all(match.included_in_regression for match in matches if match is not outlier)


def test_kalman_local_rate_matches_piecewise_mapping_derivative() -> None:
    from double_ender_sync.analysis.drift import KalmanDrift, select_drift_model
    from double_ender_sync.config import DriftModelConfig

    matches = [
        _match(float(local), _smooth_kalman_master_time(float(local)), confidence=0.95)
        for local in range(0, 601, 60)
    ]
    config = DriftModelConfig(
        drift_model="kalman",
        allow_nonlinear_drift=True,
        min_anchors_for_kalman=5,
        min_residual_improvement_ms=1.0,
        min_relative_residual_improvement=0.1,
        kalman_process_offset_noise_ms=20.0,
        kalman_process_rate_noise_ppm=500.0,
        kalman_observation_noise_ms=2.0,
        max_abs_rate_deviation_ppm=1000.0,
    )

    drift = select_drift_model(matches, config, local_duration_seconds=600.0)

    assert isinstance(drift, KalmanDrift)
    local_time = 150.0
    epsilon = 0.001
    finite_difference_rate = (
        drift.map_local_to_master(local_time + epsilon)
        - drift.map_local_to_master(local_time - epsilon)
    ) / (2.0 * epsilon)
    assert drift.local_rate_at(local_time) == pytest.approx(finite_difference_rate)
    report = drift.to_report_dict()
    reported_state = min(report["state_points"], key=lambda point: abs(point["local_time"] - local_time))
    assert reported_state["local_rate"] == pytest.approx(drift.local_rate_at(reported_state["local_time"]))
    assert "state_rate_deviation" in reported_state


def test_kalman_drift_rejects_adjacent_mapping_rate_changes_above_limit() -> None:
    from double_ender_sync.analysis.drift import LinearDrift, select_drift_model
    from double_ender_sync.config import DriftModelConfig

    matches = [
        _match(float(local), _smooth_kalman_master_time(float(local)), confidence=0.95)
        for local in range(0, 601, 60)
    ]
    config = DriftModelConfig(
        drift_model="kalman",
        allow_nonlinear_drift=True,
        min_anchors_for_kalman=5,
        min_residual_improvement_ms=1.0,
        min_relative_residual_improvement=0.1,
        kalman_process_offset_noise_ms=20.0,
        kalman_process_rate_noise_ppm=500.0,
        kalman_observation_noise_ms=2.0,
        max_abs_rate_deviation_ppm=1000.0,
        max_rate_change_ppm=50.0,
    )

    drift = select_drift_model(matches, config, local_duration_seconds=600.0)

    assert isinstance(drift, LinearDrift)
    assert drift.model_selection_policy == "kalman_research_experimental"
    assert drift.fallback_reason is not None
    assert "kalman candidate rejected: adjacent local-rate change" in drift.fallback_reason
    assert "max_rate_change_ppm=50" in drift.fallback_reason


def test_kalman_drift_numerical_failure_falls_back_with_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    from double_ender_sync.analysis import drift as drift_module
    from double_ender_sync.analysis.drift import LinearDrift, select_drift_model
    from double_ender_sync.config import DriftModelConfig

    matches = [
        _match(float(local), _smooth_kalman_master_time(float(local)), confidence=0.95)
        for local in range(0, 601, 60)
    ]
    config = DriftModelConfig(
        drift_model="kalman",
        allow_nonlinear_drift=True,
        min_anchors_for_kalman=5,
        min_residual_improvement_ms=1.0,
        min_relative_residual_improvement=0.1,
        kalman_process_offset_noise_ms=20.0,
        kalman_process_rate_noise_ppm=500.0,
        kalman_observation_noise_ms=2.0,
        max_abs_rate_deviation_ppm=1000.0,
    )

    def raise_linalg_error(*args: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        raise np.linalg.LinAlgError("synthetic ill-conditioned covariance")

    monkeypatch.setattr(drift_module, "_run_rts_backward_smoother", raise_linalg_error)

    drift = select_drift_model(matches, config, local_duration_seconds=600.0)

    assert isinstance(drift, LinearDrift)
    assert drift.model_selection_policy == "kalman_research_experimental"
    assert drift.fallback_reason is not None
    assert "kalman candidate rejected: numerical fitting failed" in drift.fallback_reason
    assert "synthetic ill-conditioned covariance" in drift.fallback_reason


def test_kalman_rts_smoother_rejects_singular_predicted_covariance() -> None:
    from double_ender_sync.analysis.drift import _run_rts_backward_smoother

    local_times = np.array([0.0, 1.0], dtype=np.float64)
    filtered_states = np.zeros((2, 2), dtype=np.float64)
    filtered_covariances = np.repeat(np.eye(2, dtype=np.float64)[None, :, :], repeats=2, axis=0)
    predicted_states = np.zeros((2, 2), dtype=np.float64)
    predicted_covariances = np.repeat(np.eye(2, dtype=np.float64)[None, :, :], repeats=2, axis=0)
    predicted_covariances[1] = 0.0

    with pytest.raises(ValueError, match="ill-conditioned predicted covariance"):
        _run_rts_backward_smoother(
            local_times=local_times,
            filtered_states=filtered_states,
            filtered_covariances=filtered_covariances,
            predicted_states=predicted_states,
            predicted_covariances=predicted_covariances,
        )


def test_kalman_drift_sparse_anchors_falls_back_with_reason() -> None:
    from double_ender_sync.analysis.drift import LinearDrift, select_drift_model
    from double_ender_sync.config import DriftModelConfig

    matches = [_match(float(local), float(local) + 1.0) for local in [0.0, 120.0, 240.0, 360.0]]
    config = DriftModelConfig(
        drift_model="kalman",
        allow_nonlinear_drift=True,
        min_anchors_for_kalman=5,
    )

    drift = select_drift_model(matches, config, local_duration_seconds=360.0)

    assert isinstance(drift, LinearDrift)
    assert drift.model_selection_policy == "kalman_research_experimental"
    assert drift.fallback_reason is not None
    assert "min_anchors_for_kalman" in drift.fallback_reason


def test_kalman_drift_is_not_attempted_by_auto_policy() -> None:
    from double_ender_sync.analysis.drift import select_drift_model
    from double_ender_sync.config import DriftModelConfig

    matches = [_match(float(local), _smooth_kalman_master_time(float(local))) for local in range(0, 601, 60)]
    config = DriftModelConfig(allow_nonlinear_drift=True, max_breakpoints=0)

    drift = select_drift_model(matches, config, local_duration_seconds=600.0)

    assert drift is not None
    assert "kalman" not in drift.candidate_models
