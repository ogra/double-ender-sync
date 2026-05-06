import csv
import json
from pathlib import Path

from double_ender_sync.analysis.anchors import AnchorCandidate
from double_ender_sync.analysis.vad import SpeechSegment
from double_ender_sync.alignment.offset import OffsetEstimate
from double_ender_sync.analysis.drift import AnchorMatch, DriftEstimate
from double_ender_sync.i18n.catalog import TranslationCatalog
from double_ender_sync.i18n.resolver import resolve_language
from double_ender_sync.types import AudioTrack


def build_phase5_report(
    master: AudioTrack,
    tracks: list[AudioTrack],
    analysis_sample_rate: int,
    track_details: dict[str, dict],
    language: str | None = None,
    vad_metadata: dict | None = None,
) -> dict:
    resolved_language = resolve_language(explicit_lang=language)
    catalog = TranslationCatalog(resolved_language)
    report_tracks: list[dict] = []
    global_warnings: list[dict] = []
    global_errors: list[dict] = []

    for track in tracks:
        detail = track_details.get(track.name, {})
        track_report = _build_track_phase5_report(track, detail, catalog)
        report_tracks.append(track_report)
        global_warnings.extend(track_report.get("warnings", []))
        global_errors.extend(track_report.get("errors", []))

    return {
        "phase": "phase5_reporting",
        "analysis": {"sample_rate": analysis_sample_rate, "channels": "mono", "dtype": "float32", "language": resolved_language, "vad": vad_metadata or {}},
        "master": _track_metadata(master),
        "tracks": report_tracks,
        "warnings": global_warnings,
        "errors": global_errors,
    }


def write_sync_report(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "sync-report.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output_path


def write_sync_markers_csv(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "sync-markers.csv"
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["track", "local_start", "local_end", "master_start", "master_end", "offset_seconds", "residual_ms", "confidence", "score"],
        )
        writer.writeheader()
        for track in report.get("tracks", []):
            for marker in track.get("drift_anchor_matches", []):
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


def _build_track_phase5_report(track: AudioTrack, detail: dict, catalog: TranslationCatalog) -> dict:
    data = _track_metadata(track)
    data.update(detail)

    drift_estimate = detail.get("drift_estimate")
    warnings: list[dict] = []
    errors: list[dict] = []

    if drift_estimate is None:
        errors.append(_issue(track.name, "error", "ALIGNMENT_FAILED", catalog.t("warnings.alignment_failed")))
    else:
        estimated_drift_at_end_ms = ((drift_estimate["stretch_ratio"] - 1.0) * track.duration_seconds) * 1000.0
        data["offset_seconds"] = drift_estimate["offset_seconds"]
        data["stretch_ratio"] = drift_estimate["stretch_ratio"]
        data["anchor_count"] = drift_estimate["anchor_count"]
        data["residual_median_ms"] = drift_estimate["residual_median_ms"]
        data["residual_max_ms"] = drift_estimate["residual_max_ms"]
        data["estimated_drift_at_end_ms"] = estimated_drift_at_end_ms

        if drift_estimate["anchor_count"] < 3:
            errors.append(_issue(track.name, "error", "INSUFFICIENT_ANCHORS", catalog.t("warnings.insufficient_anchors")))
        elif drift_estimate["anchor_count"] < 8:
            warnings.append(_issue(track.name, "warning", "LOW_ANCHOR_COUNT", catalog.t("warnings.low_anchor_count")))

        if drift_estimate["residual_median_ms"] >= 80 or drift_estimate["residual_max_ms"] >= 250:
            errors.append(_issue(track.name, "error", "HIGH_RESIDUALS", catalog.t("warnings.high_residuals")))
        elif drift_estimate["residual_median_ms"] >= 30 or drift_estimate["residual_max_ms"] >= 100:
            warnings.append(_issue(track.name, "warning", "ELEVATED_RESIDUALS", catalog.t("warnings.elevated_residuals")))

    local_adjustment = detail.get("local_adjustment") or {}
    for local_warning in local_adjustment.get("warnings", []):
        warnings.append(_issue(track.name, "warning", "LOCAL_ADJUST", local_warning))

    data["warnings"] = warnings
    data["errors"] = errors
    return data


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
    return [{"local_start": a.local_start, "local_end": a.local_end, "confidence": a.confidence, "rms": a.rms} for a in anchors]


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
    return [{"local_start": m.local_start, "local_end": m.local_end, "master_start": m.master_start, "master_end": m.master_end, "offset_seconds": m.offset_seconds, "confidence": m.confidence, "score": m.score, "residual_ms": m.residual_ms} for m in matches]


def serialize_drift_estimate(estimate: DriftEstimate | None) -> dict | None:
    if estimate is None:
        return None
    return {
        "offset_seconds": estimate.offset_seconds,
        "stretch_ratio": estimate.stretch_ratio,
        "anchor_count": estimate.anchor_count,
        "residual_median_ms": estimate.residual_median_ms,
        "residual_max_ms": estimate.residual_max_ms,
    }
