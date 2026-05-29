import csv
import json
import statistics
from pathlib import Path

from double_ender_sync.analysis.anchors import AnchorCandidate, AnchorSelectionDiagnostics
from double_ender_sync.analysis.vad import SpeechSegment
from double_ender_sync.alignment.offset import OffsetEstimate
from double_ender_sync.analysis.drift import AnchorMatch, DriftModel, DriftFitDiagnostics
from double_ender_sync.i18n.catalog import TranslationCatalog
from double_ender_sync.i18n.resolver import resolve_language
from double_ender_sync.types import AudioTrack


def build_alignment_diagnostics_report(
    master: AudioTrack,
    tracks: list[AudioTrack],
    analysis_sample_rate: int,
    track_details: dict[str, dict],
    language: str | None = None,
    vad_metadata: dict | None = None,
    anchor_selection_metadata: dict | None = None,
    verbose_report: bool = False,
    configuration_snapshot: dict | None = None,
) -> dict:
    resolved_language = resolve_language(explicit_lang=language)
    catalog = TranslationCatalog(resolved_language)
    report_tracks: list[dict] = []
    global_warnings: list[dict] = []
    global_errors: list[dict] = []

    for track in tracks:
        detail = track_details.get(track.name, {})
        track_report = _build_track_alignment_report(track, detail, catalog, verbose_report=verbose_report)
        report_tracks.append(track_report)
        global_warnings.extend(track_report.get("warnings", []))
        global_errors.extend(track_report.get("errors", []))

    analysis = {
        "sample_rate": analysis_sample_rate,
        "channels": "mono",
        "dtype": "float32",
        "language": resolved_language,
        "vad": vad_metadata or {},
        "anchor_selection": anchor_selection_metadata or {},
        "report_detail_level": "verbose" if verbose_report else "default",
        "verbose_report": verbose_report,
    }
    if verbose_report:
        analysis["configuration_snapshot"] = configuration_snapshot or {}

    return {
        "report_type": "alignment_diagnostics",
        "schema_version": "1",
        "analysis": analysis,
        "master": _track_metadata(master),
        "tracks": report_tracks,
        "warnings": global_warnings,
        "errors": global_errors,
    }


# Backward-compatible API name retained for existing callers. New code should use
# build_alignment_diagnostics_report so user-facing report terminology is not
# tied to the internal MVP phase roadmap.
build_phase5_report = build_alignment_diagnostics_report


def write_sync_report(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "sync-report.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output_path


def write_sync_markers_csv(report: dict, out_dir: Path, track_details: dict[str, dict] | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "sync-markers.csv"
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["track", "local_start", "local_end", "master_start", "master_end", "offset_seconds", "residual_ms", "confidence", "score"],
        )
        writer.writeheader()
        for track in report.get("tracks", []):
            for marker in _sync_marker_matches(track, track_details):
                writer.writerow(
                    {
                        "track": track.get("name"),
                        "local_start": marker.get("local_start"),
                        "local_end": marker.get("local_end"),
                        "master_start": marker.get("master_start"),
                        "master_end": marker.get("master_end"),
                        "offset_seconds": marker.get("offset_seconds"),
                        "residual_ms": marker.get("residual_ms"),
                        "confidence": marker.get("confidence"),
                        "score": marker.get("score"),
                    }
                )
    return output_path


def _sync_marker_matches(track: dict, track_details: dict[str, dict] | None = None) -> list[dict]:
    """Return anchor matches for marker export without requiring verbose reports.

    Verbose reports carry ``drift_anchor_matches`` directly. Default reports keep
    that heavyweight array out of ``sync-report.json``, so CLI callers can pass
    the original per-track details to preserve editor-facing marker CSV rows.
    """

    markers = track.get("drift_anchor_matches")
    if isinstance(markers, list):
        return markers

    if track_details is None:
        summary = track.get("drift_anchor_match_summary")
        summary_count = summary.get("count", 0) if isinstance(summary, dict) else 0
        if summary_count:
            track_name = track.get("name", "unknown")
            raise ValueError(
                "Cannot write sync-markers.csv for track "
                f"{track_name!r}: drift_anchor_matches are omitted from the compact report. "
                "Pass the original track_details to write_sync_markers_csv or enable verbose reporting."
            )
        return []

    track_name = track.get("name")
    if not isinstance(track_name, str):
        return []

    detail = track_details.get(track_name, {})
    detail_markers = detail.get("drift_anchor_matches", [])
    return detail_markers if isinstance(detail_markers, list) else []


def write_warnings_text(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "warnings.txt"
    lines: list[str] = []
    for warning in report.get("warnings", []):
        timecode = warning.get("time_seconds")
        time_part = "n/a" if timecode is None else f"{timecode:.3f}s"
        lines.append(f"[{warning.get('severity', 'warning').upper()}] track={warning.get('speaker_track', 'unknown')} time={time_part} code={warning.get('code', 'n/a')} msg={warning.get('message', '')}")
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path


def _track_metadata(track: AudioTrack) -> dict:
    return {
        "path": str(track.path),
        "name": track.name,
        "sample_rate": track.sample_rate,
        "duration_seconds": round(track.duration_seconds, 6),
        "channels": track.channels,
        "analysis_sample_rate": track.analysis_sample_rate,
        "analysis_samples": int(track.analysis_samples.shape[0]),
    }


def _build_track_alignment_report(track: AudioTrack, detail: dict, catalog: TranslationCatalog, *, verbose_report: bool = False) -> dict:
    data = _track_metadata(track)
    _copy_default_track_detail_fields(data, detail)
    if verbose_report:
        _copy_verbose_track_detail_fields(data, detail)

    drift_estimate = detail.get("drift_estimate")
    warnings: list[dict] = []
    errors: list[dict] = []

    if drift_estimate is None:
        errors.append(_issue(track.name, "error", "ALIGNMENT_FAILED", catalog.t("warnings.alignment_failed")))
    else:
        _promote_drift_model_metadata(data, drift_estimate, verbose_report=verbose_report)

        offset_seconds = _compat_metric(drift_estimate, "offset_seconds")
        stretch_ratio = _compat_metric(drift_estimate, "stretch_ratio")
        anchor_count = _compat_metric(drift_estimate, "anchor_count")
        residual_median_ms = _compat_metric(drift_estimate, "residual_median_ms")
        residual_max_ms = _compat_metric(drift_estimate, "residual_max_ms")

        if offset_seconds is not None:
            data["offset_seconds"] = offset_seconds
        if stretch_ratio is not None:
            data["stretch_ratio"] = stretch_ratio
            data["estimated_drift_at_end_ms"] = ((stretch_ratio - 1.0) * track.duration_seconds) * 1000.0
        if anchor_count is not None:
            data["anchor_count"] = anchor_count
        if residual_median_ms is not None:
            data["residual_median_ms"] = residual_median_ms
        if residual_max_ms is not None:
            data["residual_max_ms"] = residual_max_ms

        if anchor_count is not None:
            if anchor_count < 3:
                errors.append(_issue(track.name, "error", "INSUFFICIENT_ANCHORS", catalog.t("warnings.insufficient_anchors")))
            elif anchor_count < 8:
                warnings.append(_issue(track.name, "warning", "LOW_ANCHOR_COUNT", catalog.t("warnings.low_anchor_count")))

        if residual_median_ms is not None and residual_max_ms is not None:
            if residual_median_ms >= 80 or residual_max_ms >= 250:
                errors.append(_issue(track.name, "error", "HIGH_RESIDUALS", catalog.t("warnings.high_residuals")))
            elif residual_median_ms >= 30 or residual_max_ms >= 100:
                warnings.append(_issue(track.name, "warning", "ELEVATED_RESIDUALS", catalog.t("warnings.elevated_residuals")))

        for model_warning in drift_estimate.get("warnings", []):
            warnings.append(
                _issue(
                    track.name,
                    "warning",
                    model_warning.get("code", "DRIFT_MODEL_WARNING"),
                    model_warning.get("message", catalog.t("warnings.drift_model_diagnostics_fallback")),
                    model_warning.get("time_seconds"),
                )
            )

    drift_fit_diagnostics = detail.get("drift_fit_diagnostics") or {}
    if not drift_fit_diagnostics and isinstance(drift_estimate, dict):
        drift_fit_diagnostics = drift_estimate.get("diagnostics") or {}
    for drift_warning in drift_fit_diagnostics.get("warnings", []):
        warnings.append(
            _issue(
                track.name,
                "warning",
                drift_warning.get("code", "DRIFT_FIT_DIAGNOSTIC"),
                drift_warning.get("message", catalog.t("warnings.drift_fit_diagnostics_fallback")),
                drift_warning.get("time_seconds"),
            )
        )

    local_adjustment = detail.get("local_adjustment") or {}
    anchor_selection_diagnostics = detail.get("anchor_selection_diagnostics") or {}
    for coverage_warning in anchor_selection_diagnostics.get("warnings", []):
        warnings.append(
            _issue(
                track.name,
                "warning",
                coverage_warning.get("code", "ANCHOR_COVERAGE"),
                coverage_warning.get("message", catalog.t("warnings.anchor_coverage_fallback")),
                coverage_warning.get("time_seconds"),
            )
        )

    for local_warning in local_adjustment.get("warnings", []):
        warnings.append(_issue(track.name, "warning", "LOCAL_ADJUST", local_warning))

    data["warnings"] = _deduplicate_issues(warnings)
    data["errors"] = _deduplicate_issues(errors)
    return data


_DEFAULT_TRACK_DETAIL_KEYS = ("initial_offset", "global_correction", "vad", "anchor_selection", "anchor_matching", "drift_model_selection")
_VERBOSE_TRACK_DETAIL_KEYS = (
    "speech_segments",
    "anchor_candidates",
    "anchor_selection_diagnostics",
    "drift_anchor_matches",
    "drift_fit_diagnostics",
    "drift_estimate",
    "local_adjustment",
)


def _copy_default_track_detail_fields(data: dict, detail: dict) -> None:
    """Copy only compact per-track fields into default reports."""

    for key in _DEFAULT_TRACK_DETAIL_KEYS:
        if key in detail:
            data[key] = detail[key]

    if "speech_segments" in detail:
        data["speech_segment_summary"] = _count_summary(detail["speech_segments"])
    if "anchor_candidates" in detail:
        data["anchor_candidate_summary"] = _count_summary(detail["anchor_candidates"])
    if "drift_anchor_matches" in detail:
        data["drift_anchor_match_summary"] = _anchor_match_summary(detail["drift_anchor_matches"])
    if "anchor_selection_diagnostics" in detail:
        data["anchor_selection_summary"] = _anchor_selection_summary(detail["anchor_selection_diagnostics"])
    if "drift_fit_diagnostics" in detail:
        data["drift_fit_summary"] = _drift_fit_summary(detail["drift_fit_diagnostics"])
    if "local_adjustment" in detail:
        data["local_adjustment_summary"] = _local_adjustment_summary(detail["local_adjustment"])


def _copy_verbose_track_detail_fields(data: dict, detail: dict) -> None:
    """Copy verbose/debug-only arrays and diagnostics into verbose reports."""

    for key in _VERBOSE_TRACK_DETAIL_KEYS:
        if key in detail:
            data[key] = detail[key]


def _count_summary(items: object) -> dict:
    return {"count": len(items) if isinstance(items, list) else 0}


def _anchor_match_summary(matches: object) -> dict:
    if not isinstance(matches, list):
        return {"count": 0, "included_count": 0, "rejected_count": 0}
    included_count = sum(1 for match in matches if not isinstance(match, dict) or match.get("included_in_regression", True))
    residuals = [
        float(match["residual_ms"])
        for match in matches
        if (
            isinstance(match, dict)
            and match.get("included_in_regression", True)
            and match.get("residual_ms") is not None
        )
    ]
    summary = {
        "count": len(matches),
        "included_count": included_count,
        "rejected_count": len(matches) - included_count,
    }
    if residuals:
        summary["residual_ms"] = {
            "min": min(residuals),
            "max": max(residuals),
            "median": statistics.median(residuals),
        }
    return summary


def _anchor_selection_summary(diagnostics: object) -> dict:
    if not isinstance(diagnostics, dict):
        return {}
    return {
        key: diagnostics[key]
        for key in (
            "candidate_anchor_count",
            "selected_anchor_count",
            "target_anchor_count",
            "stratified_bin_count",
            "longest_unanchored_span_seconds",
            "sparse_bin_count",
        )
        if key in diagnostics
    }


def _drift_fit_summary(diagnostics: object) -> dict:
    if not isinstance(diagnostics, dict):
        return {}
    return {
        key: diagnostics[key]
        for key in (
            "input_anchor_count",
            "matched_anchor_count",
            "fitted_anchor_count",
            "outlier_count",
            "local_span_seconds",
            "local_span_ratio",
            "residual_rejection_threshold_ms",
        )
        if key in diagnostics
    }


def _local_adjustment_summary(local_adjustment: object) -> dict:
    if not isinstance(local_adjustment, dict):
        return {"event_count": 0, "warning_count": 0}
    events = local_adjustment.get("events", [])
    warnings = local_adjustment.get("warnings", [])
    return {
        "event_count": len(events) if isinstance(events, list) else 0,
        "warning_count": len(warnings) if isinstance(warnings, list) else 0,
    }


def _model_parameter_summary(model_parameters: object) -> dict:
    if not isinstance(model_parameters, dict):
        return {}
    summary: dict = {}
    for key, value in model_parameters.items():
        if isinstance(value, list):
            summary[f"{key}_count"] = len(value)
        elif isinstance(value, tuple):
            summary[f"{key}_count"] = len(value)
        elif isinstance(value, dict):
            summary[f"{key}_keys"] = sorted(value)
        else:
            summary[key] = value
    return summary


def _promote_drift_model_metadata(data: dict, drift_estimate: dict, *, verbose_report: bool = False) -> None:
    """Expose drift-model metadata at the track level without forcing linear fields."""

    model_type = drift_estimate.get("model_type", "linear")
    stretch_ratio = _compat_metric(drift_estimate, "stretch_ratio")
    data["model_type"] = model_type
    data["model_version"] = drift_estimate.get("model_version", "1")
    data["model_selection_policy"] = drift_estimate.get("model_selection_policy", "linear_default")
    data["candidate_models"] = drift_estimate.get("candidate_models", [model_type])
    data["selected_model_reason"] = drift_estimate.get("selected_model_reason", "linear is the default control model")
    data["fallback_reason"] = drift_estimate.get("fallback_reason")

    model_parameters = drift_estimate.get("model_parameters")
    if model_parameters is None:
        model_parameters = {
            key: value
            for key in ("offset_seconds", "stretch_ratio")
            if (value := drift_estimate.get(key)) is not None
        }
    if verbose_report or model_type == "linear":
        data["model_parameters"] = model_parameters
    else:
        data["model_parameters_summary"] = _model_parameter_summary(model_parameters)

    if "local_rate_summary" in drift_estimate:
        data["local_rate_summary"] = drift_estimate["local_rate_summary"]
    elif stretch_ratio is not None:
        data["local_rate_summary"] = {"min": stretch_ratio, "max": stretch_ratio, "mean": stretch_ratio}

    if "monotonicity_check" in drift_estimate:
        data["monotonicity_check"] = drift_estimate["monotonicity_check"]
    elif stretch_ratio is not None:
        data["monotonicity_check"] = {"passed": stretch_ratio > 0.0}

    breakpoints = drift_estimate.get("breakpoints", [])
    knots = drift_estimate.get("knots", [])
    state_points = drift_estimate.get("state_points", [])
    anchor_residuals = drift_estimate.get("anchor_residuals", [])
    data["breakpoint_count"] = len(breakpoints) if isinstance(breakpoints, list) else 0
    data["knot_count"] = len(knots) if isinstance(knots, list) else 0
    data["state_point_count"] = len(state_points) if isinstance(state_points, list) else 0
    data["anchor_residual_count"] = len(anchor_residuals) if isinstance(anchor_residuals, list) else 0

    unsupported_regions = drift_estimate.get("unsupported_regions", [])
    data["unsupported_region_summary"] = {
        "count": len(unsupported_regions) if isinstance(unsupported_regions, list) else 0,
        "codes": sorted(
            {str(region.get("code")) for region in unsupported_regions if isinstance(region, dict) and region.get("code")}
        ) if isinstance(unsupported_regions, list) else [],
    }

    if verbose_report:
        data["breakpoints"] = breakpoints
        data["knots"] = knots
        data["unsupported_regions"] = unsupported_regions
        for key in (
            "segments",
            "segment_residual_summaries",
            "knot_residual_summaries",
            "state_points",
            "anchor_residuals",
            "diagnostics",
        ):
            if key in drift_estimate:
                data[key] = drift_estimate[key]


def _compat_metric(drift_estimate: dict, key: str) -> float | int | None:
    """Return a legacy metric from the top level or model parameters when available."""

    if key in drift_estimate:
        return drift_estimate[key]
    model_parameters = drift_estimate.get("model_parameters")
    if isinstance(model_parameters, dict):
        return model_parameters.get(key)
    return None


def _deduplicate_issues(issues: list[dict]) -> list[dict]:
    seen: set[tuple[object, object, object, object]] = set()
    deduplicated: list[dict] = []
    for issue in issues:
        key = (issue.get("severity"), issue.get("code"), issue.get("time_seconds"), issue.get("message"))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(issue)
    return deduplicated


def _issue(track_name: str, severity: str, code: str, message: str, time_seconds: float | None = None) -> dict:
    return {
        "speaker_track": track_name,
        "time_seconds": time_seconds,
        "severity": severity,
        "code": code,
        "message": message,
    }


def serialize_segments(segments: list[SpeechSegment]) -> list[dict]:
    return [{"start": s.start, "end": s.end, "confidence": s.confidence} for s in segments]


def serialize_anchors(anchors: list[AnchorCandidate]) -> list[dict]:
    return [
        {
            "local_start": a.local_start,
            "local_end": a.local_end,
            "confidence": a.confidence,
            "rms": a.rms,
            "bin_index": a.bin_index,
            "snr_db": a.snr_db,
            "spectral_flatness": a.spectral_flatness,
            "quality_multiplier": a.quality_multiplier,
            "duration_seconds": a.duration_seconds,
        }
        for a in anchors
    ]


def serialize_anchor_selection_diagnostics(diagnostics: AnchorSelectionDiagnostics) -> dict:
    return {
        "candidate_anchor_count": diagnostics.candidate_anchor_count,
        "selected_anchor_count": diagnostics.selected_anchor_count,
        "target_anchor_count": diagnostics.target_anchor_count,
        "stratified_bin_count": diagnostics.stratified_bin_count,
        "anchors_per_bin": diagnostics.anchors_per_bin,
        "longest_unanchored_span_seconds": diagnostics.longest_unanchored_span_seconds,
        "sparse_bin_count": diagnostics.sparse_bin_count,
        "adaptive_duration": {
            "min_seconds": diagnostics.adaptive_duration_min_seconds,
            "median_seconds": diagnostics.adaptive_duration_median_seconds,
            "max_seconds": diagnostics.adaptive_duration_max_seconds,
        },
        "rejected_candidate_counts": diagnostics.rejected_candidate_counts,
        "bins": [
            {
                "index": b.index,
                "start_seconds": b.start_seconds,
                "end_seconds": b.end_seconds,
                "candidate_count": b.candidate_count,
                "selected_count": b.selected_count,
            }
            for b in diagnostics.bins
        ],
        "warnings": [
            {
                "code": w.code,
                "message": w.message,
                "time_seconds": w.time_seconds,
            }
            for w in diagnostics.warnings
        ],
    }


def serialize_offset(offset: OffsetEstimate | None) -> dict | None:
    if offset is None:
        return None
    return {
        "offset_seconds": offset.offset_seconds,
        "confidence": offset.confidence,
        "local_anchor_start": offset.local_anchor_start,
        "master_anchor_start": offset.master_anchor_start,
        "score": offset.score,
    }


def serialize_anchor_matches(matches: list[AnchorMatch]) -> list[dict]:
    return [
        {
            "local_start": m.local_start,
            "local_end": m.local_end,
            "master_start": m.master_start,
            "master_end": m.master_end,
            "offset_seconds": m.offset_seconds,
            "confidence": m.confidence,
            "score": m.score,
            "residual_ms": m.residual_ms,
            "included_in_regression": m.included_in_regression,
            "rejected_reason": m.rejected_reason,
            "ncc_best_score": m.ncc_best_score,
            "ncc_second_score": m.ncc_second_score,
            "ncc_margin": m.ncc_margin,
            "ncc_prominence": m.ncc_prominence,
            "ncc_width_seconds": m.ncc_width_seconds,
            "ncc_plateau_size_seconds": m.ncc_plateau_size_seconds,
            "ncc_peak_lag_seconds": m.ncc_peak_lag_seconds,
            "gcc_phat_peak_lag_seconds": m.gcc_phat_peak_lag_seconds,
            "gcc_phat_agreement_seconds": m.gcc_phat_agreement_seconds,
            "match_quality": m.match_quality,
            "match_uniqueness": m.match_uniqueness,
            "match_sharpness": m.match_sharpness,
            "match_agreement": m.match_agreement,
        }
        for m in matches
    ]


def serialize_drift_fit_diagnostics(diagnostics: DriftFitDiagnostics | None) -> dict | None:
    if diagnostics is None:
        return None
    return {
        "input_anchor_count": diagnostics.input_anchor_count,
        "matched_anchor_count": diagnostics.matched_anchor_count,
        "fitted_anchor_count": diagnostics.fitted_anchor_count,
        "outlier_count": diagnostics.outlier_count,
        "local_span_start_seconds": diagnostics.local_span_start_seconds,
        "local_span_end_seconds": diagnostics.local_span_end_seconds,
        "local_span_seconds": diagnostics.local_span_seconds,
        "local_span_ratio": diagnostics.local_span_ratio,
        "residual_rejection_threshold_ms": diagnostics.residual_rejection_threshold_ms,
        "warnings": [
            {
                "code": warning.code,
                "message": warning.message,
                "time_seconds": warning.time_seconds,
            }
            for warning in diagnostics.warnings
        ],
    }


def serialize_drift_estimate(estimate: DriftModel | None) -> dict | None:
    if estimate is None:
        return None
    data = estimate.to_report_dict()
    diagnostics = getattr(estimate, "diagnostics", None)
    data["diagnostics"] = serialize_drift_fit_diagnostics(diagnostics)
    return data
