from pathlib import Path

import numpy as np
import pytest

from double_ender_sync.report.report import (
    build_alignment_diagnostics_report,
    write_sync_markers_csv,
    write_warnings_text,
)
from double_ender_sync.types import AudioTrack


def _track(name: str) -> AudioTrack:
    samples = np.zeros(16000, dtype=np.float32)
    return AudioTrack(
        path=Path(f"{name}.wav"),
        name=name,
        sample_rate=16000,
        duration_seconds=1.0,
        channels=1,
        original_samples=samples,
        analysis_samples=samples,
        analysis_sample_rate=16000,
    )


def test_alignment_diagnostics_report_contains_warnings_and_errors(tmp_path: Path) -> None:
    master = _track("master")
    speaker = _track("speaker-a")

    report = build_alignment_diagnostics_report(
        master=master,
        tracks=[speaker],
        analysis_sample_rate=16000,
        language="ja",
        verbose_report=True,
        track_details={
            "speaker-a": {
                "drift_estimate": {
                    "offset_seconds": 0.1,
                    "stretch_ratio": 1.0001,
                    "anchor_count": 4,
                    "residual_median_ms": 35.0,
                    "residual_max_ms": 110.0,
                },
                "drift_anchor_matches": [
                    {
                        "local_start": 0.2,
                        "local_end": 0.6,
                        "master_start": 0.3,
                        "master_end": 0.7,
                        "offset_seconds": 0.1,
                        "residual_ms": 50.0,
                        "confidence": 0.8,
                        "score": 0.7,
                    }
                ],
                "local_adjustment": {"warnings": ["no safe silence"]},
            }
        },
    )

    assert report["report_type"] == "alignment_diagnostics"
    assert report["schema_version"] == "1"
    assert "phase" not in report
    assert report["warnings"]
    assert report["analysis"]["language"] == "ja"
    assert "アンカー数" in report["warnings"][0]["message"]
    assert report["tracks"][0]["estimated_drift_at_end_ms"] > 0

    markers = write_sync_markers_csv(report, tmp_path)
    warnings = write_warnings_text(report, tmp_path)

    assert "speaker-a" in markers.read_text(encoding="utf-8")
    assert "LOCAL_ADJUST" in warnings.read_text(encoding="utf-8")


def test_alignment_diagnostics_report_promotes_anchor_coverage_diagnostics_to_warnings() -> None:
    master = _track("master")
    speaker = _track("speaker-a")

    report = build_alignment_diagnostics_report(
        master=master,
        tracks=[speaker],
        analysis_sample_rate=16000,
        language="en",
        track_details={
            "speaker-a": {
                "drift_estimate": {
                    "offset_seconds": 0.1,
                    "stretch_ratio": 1.0,
                    "anchor_count": 8,
                    "residual_median_ms": 5.0,
                    "residual_max_ms": 10.0,
                },
                "anchor_selection_diagnostics": {
                    "candidate_anchor_count": 8,
                    "selected_anchor_count": 4,
                    "target_anchor_count": 4,
                    "stratified_bin_count": 4,
                    "anchors_per_bin": 1,
                    "longest_unanchored_span_seconds": 180.0,
                    "sparse_bin_count": 3,
                    "bins": [],
                    "warnings": [
                        {
                            "code": "LONG_UNANCHORED_SPAN",
                            "message": "A long section of the local timeline has no selected drift anchors; inspect alignment manually.",
                            "time_seconds": 60.0,
                        }
                    ],
                },
            }
        },
    )

    assert report["tracks"][0]["anchor_selection_summary"]["sparse_bin_count"] == 3
    assert any(warning["code"] == "LONG_UNANCHORED_SPAN" for warning in report["warnings"])


def test_alignment_diagnostics_report_promotes_drift_fit_distribution_diagnostics_to_warnings() -> None:
    master = _track("master")
    speaker = _track("speaker-a")

    report = build_alignment_diagnostics_report(
        master=master,
        tracks=[speaker],
        analysis_sample_rate=16000,
        language="en",
        track_details={
            "speaker-a": {
                "drift_estimate": {
                    "offset_seconds": 0.1,
                    "stretch_ratio": 1.0,
                    "anchor_count": 10,
                    "residual_median_ms": 5.0,
                    "residual_max_ms": 10.0,
                },
                "drift_fit_diagnostics": {
                    "input_anchor_count": 10,
                    "matched_anchor_count": 10,
                    "fitted_anchor_count": 10,
                    "outlier_count": 0,
                    "local_span_seconds": 90.0,
                    "local_span_ratio": 0.15,
                    "warnings": [
                        {
                            "code": "WEAK_DRIFT_ANCHOR_SPAN",
                            "message": "Drift anchors have enough count but cover too little of the local timeline; inspect alignment manually.",
                            "time_seconds": 0.0,
                        }
                    ],
                },
            }
        },
    )

    assert report["tracks"][0]["drift_fit_summary"]["local_span_ratio"] == 0.15
    assert any(warning["code"] == "WEAK_DRIFT_ANCHOR_SPAN" for warning in report["warnings"])


def test_alignment_diagnostics_report_promotes_linear_drift_model_metadata() -> None:
    master = _track("master")
    speaker = _track("speaker-a")

    report = build_alignment_diagnostics_report(
        master=master,
        tracks=[speaker],
        analysis_sample_rate=16000,
        language="en",
        track_details={
            "speaker-a": {
                "drift_estimate": {
                    "model_type": "linear",
                    "model_version": "1",
                    "model_selection_policy": "linear_default",
                    "candidate_models": ["linear"],
                    "selected_model_reason": "linear is the default control model",
                    "fallback_reason": None,
                    "model_parameters": {"offset_seconds": 0.2, "stretch_ratio": 1.0002},
                    "offset_seconds": 0.2,
                    "stretch_ratio": 1.0002,
                    "anchor_count": 8,
                    "residual_median_ms": 4.0,
                    "residual_max_ms": 9.0,
                    "local_rate_summary": {"min": 1.0002, "max": 1.0002, "mean": 1.0002},
                    "monotonicity_check": {"passed": True},
                    "breakpoints": [],
                    "knots": [],
                    "unsupported_regions": [],
                }
            }
        },
    )

    track_report = report["tracks"][0]
    assert track_report["model_type"] == "linear"
    assert track_report["model_version"] == "1"
    assert track_report["model_parameters"] == {"offset_seconds": 0.2, "stretch_ratio": 1.0002}
    assert track_report["offset_seconds"] == 0.2
    assert track_report["stretch_ratio"] == 1.0002
    assert track_report["breakpoint_count"] == 0
    assert track_report["unsupported_region_summary"] == {"count": 0, "codes": []}
    assert "breakpoints" not in track_report
    assert "unsupported_regions" not in track_report


def test_alignment_diagnostics_report_promotes_model_metadata_without_linear_compatibility_keys() -> None:
    master = _track("master")
    speaker = _track("speaker-a")

    report = build_alignment_diagnostics_report(
        master=master,
        tracks=[speaker],
        analysis_sample_rate=16000,
        language="en",
        track_details={
            "speaker-a": {
                "drift_estimate": {
                    "model_type": "piecewise_linear",
                    "model_version": "1",
                    "model_selection_policy": "explicit",
                    "candidate_models": ["piecewise_linear"],
                    "selected_model_reason": "external model supplied diagnostics",
                    "model_parameters": {
                        "segments": [
                            {"local_start": 0.0, "local_end": 30.0, "local_rate": 1.0},
                            {"local_start": 30.0, "local_end": 60.0, "local_rate": 1.0004},
                        ]
                    },
                    "anchor_count": 9,
                    "residual_median_ms": 6.0,
                    "residual_max_ms": 18.0,
                    "local_rate_summary": {"min": 1.0, "max": 1.0004, "mean": 1.0002},
                    "monotonicity_check": {"passed": True},
                    "breakpoints": [30.0],
                    "unsupported_regions": [],
                }
            }
        },
    )

    track_report = report["tracks"][0]
    assert track_report["model_type"] == "piecewise_linear"
    assert track_report["model_parameters_summary"] == {"segments_count": 2}
    assert track_report["local_rate_summary"] == {"min": 1.0, "max": 1.0004, "mean": 1.0002}
    assert track_report["breakpoint_count"] == 1
    assert "model_parameters" not in track_report
    assert "breakpoints" not in track_report
    assert "stretch_ratio" not in track_report
    assert "estimated_drift_at_end_ms" not in track_report


def test_serialize_drift_estimate_uses_linear_model_report_shape() -> None:
    from double_ender_sync.analysis.drift import LinearDrift
    from double_ender_sync.report.report import serialize_drift_estimate

    serialized = serialize_drift_estimate(
        LinearDrift(
            offset_seconds=0.25,
            stretch_ratio=0.9999,
            anchor_count=9,
            residual_median_ms=6.0,
            residual_max_ms=12.0,
        )
    )

    assert serialized is not None
    assert serialized["model_type"] == "linear"
    assert serialized["model_parameters"] == {"offset_seconds": 0.25, "stretch_ratio": 0.9999}
    assert serialized["anchor_count"] == 9
    assert serialized["diagnostics"] is None


def test_piecewise_model_does_not_duplicate_drift_fit_warnings() -> None:
    from double_ender_sync.analysis.drift import (
        DriftFitDiagnostics,
        DriftFitWarning,
        LinearDrift,
        PiecewiseLinearDrift,
        PiecewiseLinearSegment,
    )
    from double_ender_sync.report.report import serialize_drift_estimate

    master = _track("master")
    speaker = _track("speaker-a")
    fit_warning = DriftFitWarning(
        code="WEAK_DRIFT_ANCHOR_SPAN",
        message="Drift anchors have enough count but cover too little of the local timeline; inspect alignment manually.",
        time_seconds=0.0,
    )
    diagnostics = DriftFitDiagnostics(
        input_anchor_count=8,
        matched_anchor_count=8,
        fitted_anchor_count=8,
        outlier_count=0,
        local_span_start_seconds=0.0,
        local_span_end_seconds=90.0,
        local_span_seconds=90.0,
        local_span_ratio=0.15,
        residual_rejection_threshold_ms=None,
        warnings=[fit_warning],
    )
    drift = PiecewiseLinearDrift(
        breakpoints=(45.0,),
        segments=(
            PiecewiseLinearSegment(
                local_start=0.0,
                local_end=45.0,
                master_start=0.1,
                master_end=45.1,
                stretch_ratio=1.0,
                offset_seconds=0.1,
                anchor_count=4,
                residual_median_ms=3.0,
                residual_max_ms=8.0,
            ),
            PiecewiseLinearSegment(
                local_start=45.0,
                local_end=90.0,
                master_start=45.1,
                master_end=90.118,
                stretch_ratio=1.0004,
                offset_seconds=0.082,
                anchor_count=4,
                residual_median_ms=4.0,
                residual_max_ms=9.0,
            ),
        ),
        anchor_count=8,
        residual_median_ms=4.0,
        residual_max_ms=9.0,
        linear_baseline=LinearDrift(
            offset_seconds=0.1,
            stretch_ratio=1.0002,
            anchor_count=8,
            residual_median_ms=12.0,
            residual_max_ms=30.0,
            diagnostics=diagnostics,
        ),
        diagnostics=diagnostics,
    )

    serialized = serialize_drift_estimate(drift)
    assert serialized is not None
    assert serialized["warnings"] == []
    assert serialized["diagnostics"]["warnings"] == [
        {
            "code": "WEAK_DRIFT_ANCHOR_SPAN",
            "message": fit_warning.message,
            "time_seconds": 0.0,
        }
    ]

    report = build_alignment_diagnostics_report(
        master=master,
        tracks=[speaker],
        analysis_sample_rate=16000,
        language="en",
        track_details={"speaker-a": {"drift_estimate": serialized}},
    )

    promoted_warning_codes = [warning["code"] for warning in report["tracks"][0]["warnings"]]
    assert promoted_warning_codes.count("WEAK_DRIFT_ANCHOR_SPAN") == 1


def test_alignment_diagnostics_report_verbose_mode_includes_configuration_snapshot() -> None:
    master = _track("master")
    speaker = _track("speaker-a")

    report = build_alignment_diagnostics_report(
        master=master,
        tracks=[speaker],
        analysis_sample_rate=16000,
        track_details={
            "speaker-a": {
                "drift_estimate": {
                    "offset_seconds": 0.0,
                    "stretch_ratio": 1.0,
                    "anchor_count": 8,
                    "residual_median_ms": 1.0,
                    "residual_max_ms": 2.0,
                }
            }
        },
        verbose_report=True,
        configuration_snapshot={"drift_model_selection": {"drift_model": "auto"}},
    )

    assert report["analysis"]["report_detail_level"] == "verbose"
    assert report["analysis"]["verbose_report"] is True
    assert report["analysis"]["configuration_snapshot"] == {"drift_model_selection": {"drift_model": "auto"}}


def test_alignment_diagnostics_report_default_mode_omits_configuration_snapshot() -> None:
    master = _track("master")
    speaker = _track("speaker-a")

    report = build_alignment_diagnostics_report(
        master=master,
        tracks=[speaker],
        analysis_sample_rate=16000,
        track_details={
            "speaker-a": {
                "drift_estimate": {
                    "offset_seconds": 0.0,
                    "stretch_ratio": 1.0,
                    "anchor_count": 8,
                    "residual_median_ms": 1.0,
                    "residual_max_ms": 2.0,
                }
            }
        },
        verbose_report=False,
        configuration_snapshot={"drift_model_selection": {"drift_model": "auto"}},
    )

    assert report["analysis"]["report_detail_level"] == "default"
    assert report["analysis"]["verbose_report"] is False
    assert "configuration_snapshot" not in report["analysis"]


def test_alignment_diagnostics_default_report_summarizes_verbose_only_detail_arrays() -> None:
    master = _track("master")
    speaker = _track("speaker-a")
    track_details = {
        "speaker-a": {
            "speech_segments": [{"start": 0.0, "end": 0.5, "confidence": 0.9}],
            "anchor_candidates": [{"local_start": 0.0, "local_end": 0.5, "confidence": 0.8}],
            "drift_anchor_matches": [
                {
                    "local_start": 0.0,
                    "local_end": 0.5,
                    "master_start": 0.1,
                    "master_end": 0.6,
                    "offset_seconds": 0.1,
                    "residual_ms": 3.0,
                    "confidence": 0.9,
                    "score": 0.8,
                    "included_in_regression": True,
                }
            ],
            "anchor_selection_diagnostics": {
                "candidate_anchor_count": 1,
                "selected_anchor_count": 1,
                "target_anchor_count": 4,
                "stratified_bin_count": 1,
                "longest_unanchored_span_seconds": 0.0,
                "sparse_bin_count": 0,
                "bins": [{"index": 0, "start_seconds": 0.0, "end_seconds": 1.0}],
                "warnings": [],
            },
            "drift_fit_diagnostics": {
                "input_anchor_count": 1,
                "matched_anchor_count": 1,
                "fitted_anchor_count": 1,
                "outlier_count": 0,
                "local_span_seconds": 0.5,
                "local_span_ratio": 0.5,
                "warnings": [],
            },
            "drift_estimate": {
                "model_type": "spline",
                "model_version": "1",
                "model_selection_policy": "explicit",
                "candidate_models": ["spline"],
                "selected_model_reason": "test diagnostic model",
                "model_parameters": {"knots": [{"local_time": 0.0}, {"local_time": 1.0}]},
                "anchor_count": 8,
                "residual_median_ms": 3.0,
                "residual_max_ms": 7.0,
                "local_rate_summary": {"min": 0.999, "max": 1.001, "mean": 1.0},
                "monotonicity_check": {"passed": True},
                "breakpoints": [0.5],
                "knots": [{"local_time": 0.0}, {"local_time": 1.0}],
                "state_points": [{"local_time": 0.0, "rate": 1.0}],
                "anchor_residuals": [{"local_time": 0.0, "residual_ms": 3.0}],
                "unsupported_regions": [{"start_seconds": 0.8, "end_seconds": 0.9, "code": "TEST_REGION"}],
                "diagnostics": {"debug": True},
            },
        }
    }

    report = build_alignment_diagnostics_report(
        master=master,
        tracks=[speaker],
        analysis_sample_rate=16000,
        language="en",
        track_details=track_details,
        verbose_report=False,
    )

    track_report = report["tracks"][0]
    assert track_report["speech_segment_summary"] == {"count": 1}
    assert track_report["anchor_candidate_summary"] == {"count": 1}
    assert track_report["drift_anchor_match_summary"]["count"] == 1
    assert track_report["anchor_selection_summary"]["sparse_bin_count"] == 0
    assert track_report["drift_fit_summary"]["local_span_ratio"] == 0.5
    assert track_report["model_parameters_summary"] == {"knots_count": 2}
    assert track_report["breakpoint_count"] == 1
    assert track_report["knot_count"] == 2
    assert track_report["state_point_count"] == 1
    assert track_report["anchor_residual_count"] == 1
    assert track_report["unsupported_region_summary"] == {"count": 1, "codes": ["TEST_REGION"]}
    for omitted_key in (
        "speech_segments",
        "anchor_candidates",
        "anchor_selection_diagnostics",
        "drift_anchor_matches",
        "drift_fit_diagnostics",
        "drift_estimate",
        "model_parameters",
        "breakpoints",
        "knots",
        "state_points",
        "anchor_residuals",
        "unsupported_regions",
        "diagnostics",
    ):
        assert omitted_key not in track_report


def test_alignment_diagnostics_default_markers_csv_uses_track_details_without_verbose_report(tmp_path: Path) -> None:
    master = _track("master")
    speaker = _track("speaker-a")
    track_details = {
        "speaker-a": {
            "drift_estimate": {
                "offset_seconds": 0.1,
                "stretch_ratio": 1.0,
                "anchor_count": 8,
                "residual_median_ms": 5.0,
                "residual_max_ms": 10.0,
            },
            "drift_anchor_matches": [
                {
                    "local_start": 0.2,
                    "local_end": 0.6,
                    "master_start": 0.3,
                    "master_end": 0.7,
                    "offset_seconds": 0.1,
                    "residual_ms": 5.0,
                    "confidence": 0.8,
                    "score": 0.7,
                }
            ],
        }
    }

    report = build_alignment_diagnostics_report(
        master=master,
        tracks=[speaker],
        analysis_sample_rate=16000,
        language="en",
        track_details=track_details,
        verbose_report=False,
    )

    assert "drift_anchor_matches" not in report["tracks"][0]

    markers = write_sync_markers_csv(report, tmp_path, track_details=track_details)

    lines = markers.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "speaker-a,0.2,0.6,0.3,0.7,0.1,5.0,0.8,0.7" in lines[1]


def test_alignment_diagnostics_default_markers_csv_requires_details_when_compact_report_has_matches(tmp_path: Path) -> None:
    master = _track("master")
    speaker = _track("speaker-a")
    report = build_alignment_diagnostics_report(
        master=master,
        tracks=[speaker],
        analysis_sample_rate=16000,
        language="en",
        track_details={
            "speaker-a": {
                "drift_estimate": {
                    "offset_seconds": 0.1,
                    "stretch_ratio": 1.0,
                    "anchor_count": 8,
                    "residual_median_ms": 5.0,
                    "residual_max_ms": 10.0,
                },
                "drift_anchor_matches": [{"local_start": 0.2, "residual_ms": 5.0}],
            }
        },
        verbose_report=False,
    )

    with pytest.raises(ValueError, match="Pass the original track_details"):
        write_sync_markers_csv(report, tmp_path)


def test_alignment_diagnostics_anchor_match_summary_uses_statistical_median() -> None:
    master = _track("master")
    speaker = _track("speaker-a")

    report = build_alignment_diagnostics_report(
        master=master,
        tracks=[speaker],
        analysis_sample_rate=16000,
        language="en",
        track_details={
            "speaker-a": {
                "drift_estimate": {
                    "offset_seconds": 0.0,
                    "stretch_ratio": 1.0,
                    "anchor_count": 8,
                    "residual_median_ms": 3.0,
                    "residual_max_ms": 4.0,
                },
                "drift_anchor_matches": [
                    {"residual_ms": 2.0, "included_in_regression": True},
                    {"residual_ms": 4.0, "included_in_regression": True},
                    {"residual_ms": 1000.0, "included_in_regression": False},
                ],
            }
        },
        verbose_report=False,
    )

    summary = report["tracks"][0]["drift_anchor_match_summary"]
    assert summary["included_count"] == 2
    assert summary["rejected_count"] == 1
    assert summary["residual_ms"] == {"min": 2.0, "max": 4.0, "median": 3.0}


def test_alignment_diagnostics_verbose_report_includes_detail_arrays_and_diagnostics() -> None:
    master = _track("master")
    speaker = _track("speaker-a")
    track_details = {
        "speaker-a": {
            "speech_segments": [{"start": 0.0, "end": 0.5, "confidence": 0.9}],
            "anchor_candidates": [{"local_start": 0.0, "local_end": 0.5, "confidence": 0.8}],
            "drift_anchor_matches": [{"local_start": 0.0, "residual_ms": 3.0}],
            "anchor_selection_diagnostics": {"bins": [{"index": 0}], "warnings": []},
            "drift_fit_diagnostics": {"local_span_ratio": 0.5, "warnings": []},
            "drift_estimate": {
                "model_type": "spline",
                "model_version": "1",
                "model_selection_policy": "explicit",
                "candidate_models": ["spline"],
                "selected_model_reason": "test diagnostic model",
                "model_parameters": {"knots": [{"local_time": 0.0}, {"local_time": 1.0}]},
                "anchor_count": 8,
                "residual_median_ms": 3.0,
                "residual_max_ms": 7.0,
                "breakpoints": [0.5],
                "knots": [{"local_time": 0.0}, {"local_time": 1.0}],
                "state_points": [{"local_time": 0.0, "rate": 1.0}],
                "anchor_residuals": [{"local_time": 0.0, "residual_ms": 3.0}],
                "unsupported_regions": [{"start_seconds": 0.8, "end_seconds": 0.9, "code": "TEST_REGION"}],
                "diagnostics": {"debug": True},
            },
        }
    }

    report = build_alignment_diagnostics_report(
        master=master,
        tracks=[speaker],
        analysis_sample_rate=16000,
        language="en",
        track_details=track_details,
        verbose_report=True,
    )

    track_report = report["tracks"][0]
    assert track_report["speech_segments"] == track_details["speaker-a"]["speech_segments"]
    assert track_report["anchor_candidates"] == track_details["speaker-a"]["anchor_candidates"]
    assert track_report["drift_anchor_matches"] == track_details["speaker-a"]["drift_anchor_matches"]
    assert track_report["anchor_selection_diagnostics"] == track_details["speaker-a"]["anchor_selection_diagnostics"]
    assert track_report["drift_fit_diagnostics"] == track_details["speaker-a"]["drift_fit_diagnostics"]
    assert track_report["drift_estimate"] == track_details["speaker-a"]["drift_estimate"]
    assert track_report["model_parameters"] == {"knots": [{"local_time": 0.0}, {"local_time": 1.0}]}
    assert track_report["breakpoints"] == [0.5]
    assert track_report["knots"] == [{"local_time": 0.0}, {"local_time": 1.0}]
    assert track_report["state_points"] == [{"local_time": 0.0, "rate": 1.0}]
    assert track_report["anchor_residuals"] == [{"local_time": 0.0, "residual_ms": 3.0}]
    assert track_report["unsupported_regions"] == [{"start_seconds": 0.8, "end_seconds": 0.9, "code": "TEST_REGION"}]
    assert track_report["diagnostics"] == {"debug": True}
