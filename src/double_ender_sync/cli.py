import argparse
import logging
import math
import sys
import time
from pathlib import Path

from double_ender_sync._version import get_cli_version_text
from double_ender_sync.alignment.offset import estimate_initial_offset
from double_ender_sync.analysis.anchors import select_anchor_candidates_with_diagnostics
from double_ender_sync.analysis.drift import match_anchors_for_drift, select_drift_model
from double_ender_sync.analysis.vad import DEFAULT_PYANNOTE_MODEL, build_vad_strategy, detect_speech_segments
from double_ender_sync.audio.io import AudioLoadError, cleanup_temp_files, load_audio_track
from double_ender_sync.audio.render import write_synced_track
from double_ender_sync.config import DEFAULT_ANCHOR_SELECTION_CONFIG, DEFAULT_DRIFT_MODEL_CONFIG, AnchorSelectionConfig, DriftModelConfig
from double_ender_sync.alignment.timeline import apply_global_time_correction
from double_ender_sync.alignment.local_adjust import apply_local_adjustment
from double_ender_sync.i18n.resolver import resolve_language
from double_ender_sync.report.report import (
    build_alignment_diagnostics_report,
    serialize_anchors,
    serialize_anchor_matches,
    serialize_anchor_selection_diagnostics,
    serialize_drift_estimate,
    serialize_drift_fit_diagnostics,
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


def _parse_optional_anchor_count(value: str) -> int | None:
    if value.lower() in {"none", "unbounded", "unlimited"}:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be an integer or one of: none/unbounded/unlimited"
        ) from exc


def _parse_optional_positive_float(value: str) -> float | None:
    if value.lower() in {"none", "off", "disabled"}:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive float or one of: none/off/disabled") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite value > 0 or one of: none/off/disabled")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Double-ender sync tool")
    parser.add_argument("--version", "-V", action="version", version=get_cli_version_text())
    parser.add_argument("--master", required=True, type=Path, help="Master mixed reference WAV file")
    parser.add_argument("--track", action="append", required=True, type=Path, help="Speaker local track WAV file")
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    parser.add_argument("--analysis-sample-rate", type=int, default=16000, help="Internal mono analysis sample rate")
    parser.add_argument("--normalize-output", action="store_true", help="Normalize final synced output wav peak to 0 dBFS")
    parser.add_argument("--local-adjust-enabled", action="store_true", help="Enable optional phase-4 local adjustment")
    parser.add_argument("--local-adjust-threshold-ms", type=float, default=80.0, help="Residual threshold in ms for local adjustment")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging for pipeline and memory checkpoints")
    parser.add_argument("--verbose-report", action="store_true", help="Mark sync-report.json as verbose/debug output and include the effective configuration snapshot")
    parser.add_argument("--vad-strategy", choices=["silero", "adaptive_rms", "rms", "webrtc", "pyannote"], default="adaptive_rms", help="VAD backend strategy")
    parser.add_argument("--pyannote-model", default=None, help=f"pyannote model/pipeline id when --vad-strategy pyannote is selected (default: {DEFAULT_PYANNOTE_MODEL})")
    parser.add_argument("--stretch-ratio-warning-threshold", type=float, default=0.003, help="Warn when abs(stretch_ratio-1.0) exceeds this value")
    parser.add_argument("--stretch-ratio-auto-continue", action="store_true", help="Continue automatically when stretch ratio warning threshold is exceeded")
    parser.add_argument("--stretch-method", choices=["resample", "pitch_preserving"], default="resample", help="Global time correction method")
    parser.add_argument("--drift-model", choices=["auto", "linear", "piecewise_linear", "spline", "kalman"], default=DEFAULT_DRIFT_MODEL_CONFIG.drift_model, help="Drift model policy; non-linear and research Kalman models require --allow-nonlinear-drift")
    parser.add_argument("--allow-nonlinear-drift", action="store_true", default=DEFAULT_DRIFT_MODEL_CONFIG.allow_nonlinear_drift, help="Enable experimental non-linear drift candidates and explicit research Kalman smoothing")
    parser.add_argument("--min-anchors-for-piecewise", type=int, default=DEFAULT_DRIFT_MODEL_CONFIG.min_anchors_for_piecewise, help="Minimum fitted anchors before piecewise breakpoint search")
    parser.add_argument("--min-anchors-per-segment", type=int, default=DEFAULT_DRIFT_MODEL_CONFIG.min_anchors_per_segment, help="Minimum fitted anchors required in each piecewise segment")
    parser.add_argument("--max-breakpoints", type=int, default=DEFAULT_DRIFT_MODEL_CONFIG.max_breakpoints, help="Maximum piecewise breakpoints to evaluate for non-linear drift candidates")
    parser.add_argument("--min-residual-improvement-ms", type=float, default=DEFAULT_DRIFT_MODEL_CONFIG.min_residual_improvement_ms, help="Required median residual improvement before selecting a richer drift model")
    parser.add_argument("--min-relative-residual-improvement", type=float, default=DEFAULT_DRIFT_MODEL_CONFIG.min_relative_residual_improvement, help="Required relative median residual improvement before selecting a richer drift model")
    parser.add_argument("--max-abs-rate-deviation-ppm", type=float, default=DEFAULT_DRIFT_MODEL_CONFIG.max_abs_rate_deviation_ppm, help="Reject piecewise fits whose absolute local-rate deviation exceeds this ppm")
    parser.add_argument("--max-rate-change-ppm", type=float, default=DEFAULT_DRIFT_MODEL_CONFIG.max_rate_change_ppm, help="Reject piecewise fits whose adjacent segment rate change exceeds this ppm")
    parser.add_argument("--warn-abs-rate-deviation-ppm", type=float, default=DEFAULT_DRIFT_MODEL_CONFIG.warn_abs_rate_deviation_ppm, help="Warn when accepted local-rate deviation exceeds this ppm")
    parser.add_argument("--max-anchor-gap-seconds", type=_parse_optional_positive_float, default=DEFAULT_DRIFT_MODEL_CONFIG.max_anchor_gap_seconds, help="Report an unsupported region when trusted drift anchors are farther apart than this many seconds; use none/off/disabled to disable")
    parser.add_argument("--min-anchors-for-spline", type=int, default=DEFAULT_DRIFT_MODEL_CONFIG.min_anchors_for_spline, help="Minimum fitted anchors before monotonic cubic spline fitting")
    parser.add_argument("--spline-knot-source", choices=["auto", "piecewise_boundaries", "anchors"], default=DEFAULT_DRIFT_MODEL_CONFIG.spline_knot_source, help="Knot source for spline drift candidates")
    parser.add_argument("--min-knot-spacing-seconds", type=float, default=DEFAULT_DRIFT_MODEL_CONFIG.min_knot_spacing_seconds, help="Minimum spacing for anchor-decimated spline knots")
    parser.add_argument("--spline-validation-sample-count", type=int, default=DEFAULT_DRIFT_MODEL_CONFIG.spline_validation_sample_count, help="Sample count for spline monotonicity and local-rate validation")
    parser.add_argument("--min-anchors-for-kalman", type=int, default=DEFAULT_DRIFT_MODEL_CONFIG.min_anchors_for_kalman, help="Minimum fitted anchors before explicit research Kalman smoothing")
    parser.add_argument("--kalman-process-offset-noise-ms", type=float, default=DEFAULT_DRIFT_MODEL_CONFIG.kalman_process_offset_noise_ms, help="Kalman process noise for offset wander, in ms per sqrt(second)")
    parser.add_argument("--kalman-process-rate-noise-ppm", type=float, default=DEFAULT_DRIFT_MODEL_CONFIG.kalman_process_rate_noise_ppm, help="Kalman process noise for drift-rate variability, in ppm per sqrt(second)")
    parser.add_argument("--kalman-observation-noise-ms", type=float, default=DEFAULT_DRIFT_MODEL_CONFIG.kalman_observation_noise_ms, help="Base Kalman anchor-observation noise in ms before confidence scaling")
    parser.add_argument("--kalman-initial-offset-uncertainty-ms", type=float, default=DEFAULT_DRIFT_MODEL_CONFIG.kalman_initial_offset_uncertainty_ms, help="Initial Kalman offset uncertainty in ms")
    parser.add_argument("--kalman-initial-rate-uncertainty-ppm", type=float, default=DEFAULT_DRIFT_MODEL_CONFIG.kalman_initial_rate_uncertainty_ppm, help="Initial Kalman rate uncertainty in ppm")
    parser.add_argument("--kalman-validation-sample-count", type=int, default=DEFAULT_DRIFT_MODEL_CONFIG.kalman_validation_sample_count, help="Sample count for Kalman monotonicity and local-rate validation")
    parser.add_argument("--min-anchor-duration", type=float, default=DEFAULT_ANCHOR_SELECTION_CONFIG.min_anchor_duration_seconds, help="Minimum speech-segment duration in seconds for anchor candidates")
    parser.add_argument("--base-anchor-duration", type=float, default=None, help=f"Starting anchor duration in seconds before adaptive quality scaling (default: clamp {DEFAULT_ANCHOR_SELECTION_CONFIG.base_anchor_duration_seconds} within min/max bounds)")
    parser.add_argument("--max-anchor-duration", type=float, default=DEFAULT_ANCHOR_SELECTION_CONFIG.max_anchor_duration_seconds, help="Maximum duration in seconds after adaptive anchor-duration scaling")
    parser.add_argument("--anchor-density-per-minute", type=float, default=DEFAULT_ANCHOR_SELECTION_CONFIG.anchor_density_per_minute, help="Target selected anchor density per minute before the safety cap is applied")
    parser.add_argument("--max-anchor-density-per-minute", type=float, default=DEFAULT_ANCHOR_SELECTION_CONFIG.max_anchor_density_per_minute, help="Validation ceiling for --anchor-density-per-minute")
    parser.add_argument("--min-anchor-count", type=int, default=DEFAULT_ANCHOR_SELECTION_CONFIG.min_anchor_count, help="Minimum target anchor budget for short recordings when enough valid candidates exist")
    parser.add_argument("--max-anchor-count", type=_parse_optional_anchor_count, default=DEFAULT_ANCHOR_SELECTION_CONFIG.max_anchor_count, help="Safety cap for selected anchor candidates; use 'none' to disable")
    parser.add_argument("--stratified-bin-count", type=int, default=DEFAULT_ANCHOR_SELECTION_CONFIG.stratified_bin_count, help="Timeline bin count for stratified anchor selection (default: auto)")
    parser.add_argument("--anchors-per-bin", type=int, default=DEFAULT_ANCHOR_SELECTION_CONFIG.anchors_per_bin, help="Per-bin anchor quota before global budget fill (default: auto)")
    parser.add_argument("--min-snr-db", type=float, default=DEFAULT_ANCHOR_SELECTION_CONFIG.min_snr_db, help="Reject anchor candidates below this local SNR in dB (default: disabled)")
    parser.add_argument("--spectral-flatness-threshold", type=float, default=DEFAULT_ANCHOR_SELECTION_CONFIG.spectral_flatness_threshold, help="Reject anchor candidates above this spectral flatness value from 0.0 to 1.0 (default: disabled)")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional log file path (default: <out>/double-ender-sync.log)")
    parser.add_argument("--lang", type=str, default=None, help="UI/report language code (e.g. en, ja). Regional codes like en-US are normalized to en.")
    return parser.parse_args(argv)


def _resolve_base_anchor_duration(args: argparse.Namespace) -> float:
    """Return an effective base duration, preserving max-only CLI tuning.

    When users do not pass ``--base-anchor-duration`` explicitly, keep the
    default base when possible but clamp it into the configured min/max duration
    bounds so workflows that only tune ``--min-anchor-duration`` or
    ``--max-anchor-duration`` remain valid. Explicit base values are validated by
    ``AnchorSelectionConfig`` without clamping so user mistakes stay visible.
    """

    if args.base_anchor_duration is not None:
        return args.base_anchor_duration
    default_base = DEFAULT_ANCHOR_SELECTION_CONFIG.base_anchor_duration_seconds
    return max(args.min_anchor_duration, min(default_base, args.max_anchor_duration))


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
    try:
        anchor_selection_config = AnchorSelectionConfig(
            anchor_density_per_minute=args.anchor_density_per_minute,
            max_anchor_density_per_minute=args.max_anchor_density_per_minute,
            min_anchor_count=args.min_anchor_count,
            min_anchor_duration_seconds=args.min_anchor_duration,
            base_anchor_duration_seconds=_resolve_base_anchor_duration(args),
            max_anchor_duration_seconds=args.max_anchor_duration,
            max_anchor_count=args.max_anchor_count,
            stratified_bin_count=args.stratified_bin_count,
            anchors_per_bin=args.anchors_per_bin,
            min_snr_db=args.min_snr_db,
            spectral_flatness_threshold=args.spectral_flatness_threshold,
        )
    except ValueError as exc:
        print(f"error: invalid anchor selection options: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    try:
        drift_model_config = DriftModelConfig(
            drift_model=args.drift_model,
            allow_nonlinear_drift=args.allow_nonlinear_drift,
            min_anchors_for_piecewise=args.min_anchors_for_piecewise,
            min_anchors_per_segment=args.min_anchors_per_segment,
            max_breakpoints=args.max_breakpoints,
            min_residual_improvement_ms=args.min_residual_improvement_ms,
            min_relative_residual_improvement=args.min_relative_residual_improvement,
            max_abs_rate_deviation_ppm=args.max_abs_rate_deviation_ppm,
            max_rate_change_ppm=args.max_rate_change_ppm,
            warn_abs_rate_deviation_ppm=args.warn_abs_rate_deviation_ppm,
            max_anchor_gap_seconds=args.max_anchor_gap_seconds,
            min_anchors_for_spline=args.min_anchors_for_spline,
            spline_knot_source=args.spline_knot_source,
            min_knot_spacing_seconds=args.min_knot_spacing_seconds,
            spline_validation_sample_count=args.spline_validation_sample_count,
            min_anchors_for_kalman=args.min_anchors_for_kalman,
            kalman_process_offset_noise_ms=args.kalman_process_offset_noise_ms,
            kalman_process_rate_noise_ppm=args.kalman_process_rate_noise_ppm,
            kalman_observation_noise_ms=args.kalman_observation_noise_ms,
            kalman_initial_offset_uncertainty_ms=args.kalman_initial_offset_uncertainty_ms,
            kalman_initial_rate_uncertainty_ppm=args.kalman_initial_rate_uncertainty_ppm,
            kalman_validation_sample_count=args.kalman_validation_sample_count,
        )
    except ValueError as exc:
        print(f"error: invalid drift model options: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    anchor_selection_metadata = anchor_selection_config.as_dict()
    drift_model_selection_metadata = drift_model_config.as_dict()
    configuration_snapshot = None
    if args.verbose_report:
        configuration_snapshot = {
            "drift_model_selection": drift_model_selection_metadata,
            "anchor_selection": anchor_selection_metadata,
        }

    selected_pyannote_model = args.pyannote_model or DEFAULT_PYANNOTE_MODEL
    if args.pyannote_model is not None and args.vad_strategy != "pyannote":
        print("error: --pyannote-model is only valid when --vad-strategy pyannote is selected", file=sys.stderr)
        return EXIT_USAGE_ERROR
    if args.vad_strategy == "pyannote":
        LOGGER.info("Selected pyannote VAD model: %s", selected_pyannote_model)

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
        vad_strategy = build_vad_strategy(args.vad_strategy, pyannote_model=selected_pyannote_model)
        for track_path in args.track:
            LOGGER.info("Processing track: %s", track_path.name)
            try:
                track = load_audio_track(track_path, analysis_sample_rate=args.analysis_sample_rate, include_original_samples=False)
            except AudioLoadError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            loaded_tracks.append(track)
            progress.update(f"{track.name}: analysis audio loaded")

            speech_segments = detect_speech_segments(track.analysis_samples, sample_rate=args.analysis_sample_rate, vad_strategy=vad_strategy)
            LOGGER.debug("%s: speech_segments=%d", track.name, len(speech_segments))
            progress.update(f"{track.name}: speech detection completed")
            anchor_selection_result = select_anchor_candidates_with_diagnostics(
                track.analysis_samples,
                args.analysis_sample_rate,
                speech_segments,
                config=anchor_selection_config,
            )
            anchor_candidates = anchor_selection_result.candidates
            LOGGER.debug(
                "%s: anchor_candidates=%d candidate_anchors=%d bins=%d longest_unanchored_span=%.3fs",
                track.name,
                len(anchor_candidates),
                anchor_selection_result.diagnostics.candidate_anchor_count,
                anchor_selection_result.diagnostics.stratified_bin_count,
                anchor_selection_result.diagnostics.longest_unanchored_span_seconds,
            )
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
                drift_estimate = select_drift_model(drift_matches, drift_model_config, local_duration_seconds=track.duration_seconds)
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
                except (RuntimeError, ValueError) as exc:
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
                "anchor_selection_diagnostics": serialize_anchor_selection_diagnostics(anchor_selection_result.diagnostics),
                "initial_offset": serialize_offset(offset_estimate),
                "drift_anchor_matches": serialize_anchor_matches(drift_matches),
                "drift_fit_diagnostics": serialize_drift_fit_diagnostics(None if drift_estimate is None else drift_estimate.diagnostics),
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
                    "render_method": global_correction.render_method,
                    "monotonicity_check": None if global_correction.monotonicity_check is None else {
                        "passed": global_correction.monotonicity_check.passed,
                        "sample_count": global_correction.monotonicity_check.sample_count,
                        "epsilon_seconds": global_correction.monotonicity_check.epsilon_seconds,
                        "min_step_seconds": global_correction.monotonicity_check.min_step_seconds,
                        "message": global_correction.monotonicity_check.message,
                    },
                    "unsupported_regions": [
                        {
                            "start_seconds": region.start_seconds,
                            "end_seconds": region.end_seconds,
                            "reason": region.reason,
                        }
                        for region in global_correction.unsupported_regions
                    ],
                },
                "vad": {
                    "strategy": args.vad_strategy,
                    "pyannote_model": selected_pyannote_model if args.vad_strategy == "pyannote" else None,
                },
                "anchor_selection": anchor_selection_metadata,
                "drift_model_selection": drift_model_selection_metadata,
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

        report = build_alignment_diagnostics_report(
            master=master,
            tracks=processed_tracks,
            analysis_sample_rate=args.analysis_sample_rate,
            track_details=track_details,
            language=resolved_lang,
            vad_metadata={
                "strategy": args.vad_strategy,
                "pyannote_model": selected_pyannote_model if args.vad_strategy == "pyannote" else None,
            },
            anchor_selection_metadata=anchor_selection_metadata,
            verbose_report=args.verbose_report,
            configuration_snapshot=configuration_snapshot,
        )
        report["analysis"]["drift_model_selection"] = drift_model_selection_metadata
        report_path = write_sync_report(report, args.out)
        markers_path = write_sync_markers_csv(report, args.out, track_details=track_details)
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
