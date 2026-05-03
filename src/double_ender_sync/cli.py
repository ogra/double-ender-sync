import argparse
import logging
import sys
import time
from pathlib import Path

from double_ender_sync.alignment.offset import estimate_initial_offset
from double_ender_sync.analysis.anchors import select_anchor_candidates
from double_ender_sync.analysis.drift import fit_linear_drift_model, match_anchors_for_drift
from double_ender_sync.analysis.vad import detect_speech_segments
from double_ender_sync.audio.io import AudioLoadError, cleanup_temp_files, load_audio_track
from double_ender_sync.audio.render import write_synced_track
from double_ender_sync.alignment.timeline import apply_global_time_correction
from double_ender_sync.alignment.local_adjust import apply_local_adjustment
from double_ender_sync.i18n.resolver import resolve_language
from double_ender_sync.report.report import (
    build_phase5_report,
    serialize_anchors,
    serialize_anchor_matches,
    serialize_drift_estimate,
    serialize_offset,
    serialize_segments,
    write_sync_markers_csv,
    write_sync_report,
    write_warnings_text,
)


LOGGER = logging.getLogger("double_ender_sync")

EXIT_RUNTIME_ERROR = 1
EXIT_USAGE_ERROR = 2
EXIT_STRETCH_CONFIRMATION_REQUIRED = 3


class ProgressTracker:
    def __init__(self, total_units: int, progress_callback=None, event_callback=None) -> None:
        self.total_units = max(total_units, 1)
        self.completed_units = 0
        self.started_at = time.perf_counter()
        self.progress_callback = progress_callback
        self.event_callback = event_callback

    def update(self, message: str, step_units: int = 1) -> None:
        self.completed_units = min(self.total_units, self.completed_units + step_units)
        elapsed = time.perf_counter() - self.started_at
        ratio = self.completed_units / self.total_units
        percent = ratio * 100.0
        eta_seconds = 0.0
        if self.completed_units > 0:
            average_seconds = elapsed / self.completed_units
            eta_seconds = max((self.total_units - self.completed_units) * average_seconds, 0.0)

        progress_line = f"[progress] {percent:5.1f}% | ETA {eta_seconds:6.1f}s | {message}"
        print(progress_line)
        if self.event_callback is not None:
            self.event_callback(progress_line)
        if self.progress_callback is not None:
            self.progress_callback(percent, eta_seconds, message)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Double-ender sync tool")
    parser.add_argument("--master", required=True, type=Path, help="Master mixed reference WAV file")
    parser.add_argument("--track", action="append", required=True, type=Path, help="Speaker local track WAV file")
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    parser.add_argument("--analysis-sample-rate", type=int, default=16000, help="Internal mono analysis sample rate")
    parser.add_argument("--normalize-output", action="store_true", help="Normalize final synced output wav peak to 0 dBFS")
    parser.add_argument("--local-adjust-enabled", action="store_true", help="Enable optional phase-4 local adjustment")
    parser.add_argument("--local-adjust-threshold-ms", type=float, default=80.0, help="Residual threshold in ms for local adjustment")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging for pipeline and memory checkpoints")
    parser.add_argument("--stretch-ratio-warning-threshold", type=float, default=0.003, help="Warn when abs(stretch_ratio-1.0) exceeds this value")
    parser.add_argument("--stretch-ratio-auto-continue", action="store_true", help="Continue automatically when stretch ratio warning threshold is exceeded")
    parser.add_argument("--stretch-method", choices=["resample", "pitch_preserving"], default="resample", help="Global time correction method")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional log file path (default: <out>/double-ender-sync.log)")
    parser.add_argument("--lang", type=str, default=None, help="UI/report language code (e.g. en, ja). Regional codes like en-US are normalized to en.")
    return parser.parse_args(argv)


def _configure_logging(debug_enabled: bool, log_file: Path) -> None:
    log_level = logging.DEBUG if debug_enabled else logging.INFO
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def main(argv: list[str] | None = None, progress_callback=None, event_callback=None) -> int:
    args = parse_args(argv)
    log_file = args.log_file if args.log_file is not None else args.out / "double-ender-sync.log"
    _configure_logging(debug_enabled=args.debug, log_file=log_file)
    LOGGER.info("Starting alignment run")
    resolved_lang = resolve_language(explicit_lang=args.lang)
    LOGGER.debug("Resolved language=%s (explicit=%s)", resolved_lang, args.lang)
    LOGGER.debug("Parsed args: %s", vars(args))

    if args.analysis_sample_rate <= 0:
        print("error: --analysis-sample-rate must be positive", file=sys.stderr)
        return EXIT_USAGE_ERROR
    if args.stretch_ratio_warning_threshold < 0:
        print("error: --stretch-ratio-warning-threshold must be >= 0", file=sys.stderr)
        return EXIT_USAGE_ERROR

    loaded_tracks = []
    total_tracks = len(args.track)
    progress = ProgressTracker(
        total_units=(2 + (total_tracks * 8)),
        progress_callback=progress_callback,
        event_callback=event_callback,
    )

    try:
        master = load_audio_track(args.master, analysis_sample_rate=args.analysis_sample_rate, include_original_samples=False)
        loaded_tracks.append(master)
        progress.update("Master track loaded")
        LOGGER.debug(
            "Loaded master: name=%s duration=%.3fs sr=%d analysis_samples=%d",
            master.name,
            master.duration_seconds,
            master.sample_rate,
            len(master.analysis_samples),
        )
    except AudioLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    track_details: dict[str, dict] = {}
    processed_tracks = []
    try:
        progress.update(f"Starting processing for {total_tracks} track(s)")
        for track_path in args.track:
            LOGGER.info("Processing track: %s", track_path.name)
            try:
                track = load_audio_track(track_path, analysis_sample_rate=args.analysis_sample_rate, include_original_samples=False)
            except AudioLoadError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            loaded_tracks.append(track)
            progress.update(f"{track.name}: analysis audio loaded")

            speech_segments = detect_speech_segments(track.analysis_samples, sample_rate=args.analysis_sample_rate)
            LOGGER.debug("%s: speech_segments=%d", track.name, len(speech_segments))
            progress.update(f"{track.name}: speech detection completed")
            anchor_candidates = select_anchor_candidates(track.analysis_samples, args.analysis_sample_rate, speech_segments)
            LOGGER.debug("%s: anchor_candidates=%d", track.name, len(anchor_candidates))
            progress.update(f"{track.name}: anchor selection completed")
            LOGGER.debug("%s: starting initial offset estimation", track.name)
            offset_started = time.perf_counter()
            offset_estimate = estimate_initial_offset(
                local_samples=track.analysis_samples,
                master_samples=master.analysis_samples,
                sample_rate=args.analysis_sample_rate,
                anchors=anchor_candidates,
            )
            LOGGER.debug("%s: finished initial offset estimation elapsed=%.3fs", track.name, time.perf_counter() - offset_started)
            progress.update(f"{track.name}: initial offset estimation completed")

            drift_matches = []
            drift_estimate = None
            if offset_estimate is not None and anchor_candidates:
                LOGGER.debug("%s: initial_offset_seconds=%.6f", track.name, offset_estimate.offset_seconds)
                drift_matches = match_anchors_for_drift(
                    local_samples=track.analysis_samples,
                    master_samples=master.analysis_samples,
                    sample_rate=args.analysis_sample_rate,
                    anchors=anchor_candidates,
                    initial_offset_seconds=offset_estimate.offset_seconds,
                )
                LOGGER.debug("%s: completed drift anchor matching", track.name)
                drift_estimate = fit_linear_drift_model(drift_matches)
                LOGGER.debug("%s: drift_matches=%d drift_estimated=%s", track.name, len(drift_matches), drift_estimate is not None)
            progress.update(f"{track.name}: drift estimation completed")

            synced_output_path = None
            global_correction = None
            local_adjustment = None
            if drift_estimate is not None:
                stretch_delta = abs(drift_estimate.stretch_ratio - 1.0)
                if stretch_delta > args.stretch_ratio_warning_threshold:
                    warning_message = (
                        f"warning: {track.name}: stretch_ratio={drift_estimate.stretch_ratio:.6f} "
                        f"(delta={stretch_delta:.6f}) exceeds threshold={args.stretch_ratio_warning_threshold:.6f}"
                    )
                    LOGGER.warning(warning_message)
                    print(warning_message, file=sys.stderr)
                    if not args.stretch_ratio_auto_continue:
                        if not sys.stdin.isatty():
                            print(
                                "error: stretch ratio exceeded threshold and no TTY confirmation is available. "
                                'Use --stretch-ratio-auto-continue to proceed.',
                                file=sys.stderr,
                            )
                            return EXIT_STRETCH_CONFIRMATION_REQUIRED
                        response = input("Continue alignment for this track? [y/N]: ").strip().lower()
                        if response not in {"y", "yes"}:
                            print("error: aborted by user due to excessive stretch ratio", file=sys.stderr)
                            return EXIT_STRETCH_CONFIRMATION_REQUIRED
                LOGGER.debug("%s: loading original samples for final render", track.name)
                track_with_original = load_audio_track(track_path, analysis_sample_rate=args.analysis_sample_rate, include_original_samples=True)
                loaded_tracks.append(track_with_original)
                progress.update(f"{track.name}: original audio loaded")
                try:
                    global_correction = apply_global_time_correction(
                        track=track_with_original,
                        master=master,
                        drift_estimate=drift_estimate,
                        stretch_method=args.stretch_method,
                    )
                except RuntimeError as exc:
                    print(f'error: {exc}', file=sys.stderr)
                    LOGGER.exception("Global correction failed")
                    return EXIT_RUNTIME_ERROR
                LOGGER.debug(
                    "%s: global_correction output_sr=%d output_duration=%.3f",
                    track.name,
                    global_correction.output_sample_rate,
                    global_correction.output_duration_seconds,
                )
                local_adjustment = apply_local_adjustment(
                    globally_aligned_samples=global_correction.output_samples,
                    sample_rate=global_correction.output_sample_rate,
                    residual_events=serialize_anchor_matches(drift_matches),
                    enabled=args.local_adjust_enabled,
                    residual_threshold_ms=args.local_adjust_threshold_ms,
                )
                synced_output_path = args.out / f"{track.name}.synced.wav"
                write_synced_track(
                    synced_output_path,
                    local_adjustment.adjusted_samples,
                    global_correction.output_sample_rate,
                    normalize_output=args.normalize_output,
                )
                LOGGER.info("%s: wrote synced track: %s", track.name, synced_output_path)
                progress.update(f"{track.name}: synced track written")
            else:
                LOGGER.warning("%s: skipped global correction due to missing drift estimate", track.name)
                progress.update(f"{track.name}: rendering skipped (insufficient drift estimate)", step_units=2)

            track_details[track.name] = {
                "speech_segments": serialize_segments(speech_segments),
                "anchor_candidates": serialize_anchors(anchor_candidates),
                "initial_offset": serialize_offset(offset_estimate),
                "drift_anchor_matches": serialize_anchor_matches(drift_matches),
                "drift_estimate": serialize_drift_estimate(drift_estimate),
                "global_correction": None if global_correction is None else {
                    "local_adjust_enabled": args.local_adjust_enabled,
                    "output_sample_rate": global_correction.output_sample_rate,
                    "output_duration_seconds": global_correction.output_duration_seconds,
                    "stretch_ratio": global_correction.stretch_ratio,
                    "offset_seconds": global_correction.offset_seconds,
                    "output_path": str(synced_output_path) if synced_output_path else None,
                    "normalize_output": args.normalize_output,
                    "stretch_method": args.stretch_method,
                    "stretch_ratio_warning_threshold": args.stretch_ratio_warning_threshold,
                },
                "local_adjustment": None if local_adjustment is None else {
                    "events": [
                        {
                            "split_time_seconds": e.split_time_seconds,
                            "shift_seconds": e.shift_seconds,
                            "residual_ms": e.residual_ms,
                            "confidence": e.confidence,
                        }
                        for e in local_adjustment.events
                    ],
                    "warnings": local_adjustment.warnings,
                },
            }
            processed_tracks.append(track)
            progress.update(f"{track.name}: report details collected")

        report = build_phase5_report(
            master=master,
            tracks=processed_tracks,
            analysis_sample_rate=args.analysis_sample_rate,
            track_details=track_details,
            language=resolved_lang,
        )
        report_path = write_sync_report(report, args.out)
        markers_path = write_sync_markers_csv(report, args.out)
        warnings_path = write_warnings_text(report, args.out)
        progress.update("Reports generated")
        LOGGER.info("Completed alignment run")
        print(f"Wrote {report_path}")
        if event_callback is not None:
            event_callback(f"Wrote {report_path}")
        print(f"Wrote {markers_path}")
        if event_callback is not None:
            event_callback(f"Wrote {markers_path}")
        print(f"Wrote {warnings_path}")
        if event_callback is not None:
            event_callback(f"Wrote {warnings_path}")
        print(f"Wrote {log_file}")
        if event_callback is not None:
            event_callback(f"Wrote {log_file}")
        return 0
    finally:
        cleanup_temp_files(loaded_tracks)


if __name__ == "__main__":
    raise SystemExit(main())
