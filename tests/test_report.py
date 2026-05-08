from pathlib import Path

import numpy as np

from double_ender_sync.report.report import build_phase5_report, write_sync_markers_csv, write_warnings_text
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


def test_phase5_report_contains_warnings_and_errors(tmp_path: Path) -> None:
    master = _track("master")
    speaker = _track("speaker-a")

    report = build_phase5_report(
        master=master,
        tracks=[speaker],
        analysis_sample_rate=16000,
        language="ja",
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

    assert report["phase"] == "phase5_reporting"
    assert report["warnings"]
    assert report["analysis"]["language"] == "ja"
    assert "アンカー数" in report["warnings"][0]["message"]
    assert report["tracks"][0]["estimated_drift_at_end_ms"] > 0

    markers = write_sync_markers_csv(report, tmp_path)
    warnings = write_warnings_text(report, tmp_path)

    assert "speaker-a" in markers.read_text(encoding="utf-8")
    assert "LOCAL_ADJUST" in warnings.read_text(encoding="utf-8")


def test_phase5_report_promotes_anchor_coverage_diagnostics_to_warnings() -> None:
    master = _track("master")
    speaker = _track("speaker-a")

    report = build_phase5_report(
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

    assert report["tracks"][0]["anchor_selection_diagnostics"]["sparse_bin_count"] == 3
    assert any(warning["code"] == "LONG_UNANCHORED_SPAN" for warning in report["warnings"])


def test_phase5_report_promotes_drift_fit_distribution_diagnostics_to_warnings() -> None:
    master = _track("master")
    speaker = _track("speaker-a")

    report = build_phase5_report(
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

    assert report["tracks"][0]["drift_fit_diagnostics"]["local_span_ratio"] == 0.15
    assert any(warning["code"] == "WEAK_DRIFT_ANCHOR_SPAN" for warning in report["warnings"])
