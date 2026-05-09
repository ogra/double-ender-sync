from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

from double_ender_sync import cli
from double_ender_sync._version import get_version as _get_version
from double_ender_sync.analysis.vad import DEFAULT_PYANNOTE_MODEL
from double_ender_sync.config import DEFAULT_ANCHOR_SELECTION_CONFIG, DEFAULT_DRIFT_MODEL_CONFIG, AnchorSelectionConfig, DriftModelConfig, DriftModelName, SplineKnotSource


def get_version() -> str:
    """Return the package version exposed by the Python API surface."""

    return _get_version()


@dataclass(frozen=True)
class AlignmentOptions:
    """Options for running alignment from Python code."""

    master: Path
    tracks: Sequence[Path]
    out: Path
    analysis_sample_rate: int = 16000
    normalize_output: bool = False
    local_adjust_enabled: bool = False
    local_adjust_threshold_ms: float = 80.0
    debug: bool = False
    log_file: Path | None = None
    stretch_ratio_warning_threshold: float = 0.003
    stretch_ratio_auto_continue: bool = False
    stretch_method: Literal["resample", "pitch_preserving"] = "resample"
    drift_model: DriftModelName = DEFAULT_DRIFT_MODEL_CONFIG.drift_model
    allow_nonlinear_drift: bool = DEFAULT_DRIFT_MODEL_CONFIG.allow_nonlinear_drift
    min_anchors_for_piecewise: int = DEFAULT_DRIFT_MODEL_CONFIG.min_anchors_for_piecewise
    min_anchors_per_segment: int = DEFAULT_DRIFT_MODEL_CONFIG.min_anchors_per_segment
    max_breakpoints: int = DEFAULT_DRIFT_MODEL_CONFIG.max_breakpoints
    min_residual_improvement_ms: float = DEFAULT_DRIFT_MODEL_CONFIG.min_residual_improvement_ms
    min_relative_residual_improvement: float = DEFAULT_DRIFT_MODEL_CONFIG.min_relative_residual_improvement
    max_abs_rate_deviation_ppm: float = DEFAULT_DRIFT_MODEL_CONFIG.max_abs_rate_deviation_ppm
    max_rate_change_ppm: float = DEFAULT_DRIFT_MODEL_CONFIG.max_rate_change_ppm
    warn_abs_rate_deviation_ppm: float = DEFAULT_DRIFT_MODEL_CONFIG.warn_abs_rate_deviation_ppm
    max_anchor_gap_seconds: float | None = DEFAULT_DRIFT_MODEL_CONFIG.max_anchor_gap_seconds
    min_anchors_for_spline: int = DEFAULT_DRIFT_MODEL_CONFIG.min_anchors_for_spline
    spline_knot_source: SplineKnotSource = DEFAULT_DRIFT_MODEL_CONFIG.spline_knot_source
    min_knot_spacing_seconds: float = DEFAULT_DRIFT_MODEL_CONFIG.min_knot_spacing_seconds
    spline_validation_sample_count: int = DEFAULT_DRIFT_MODEL_CONFIG.spline_validation_sample_count
    min_anchors_for_kalman: int = DEFAULT_DRIFT_MODEL_CONFIG.min_anchors_for_kalman
    # Kalman process-noise fields use per-sqrt(second) units to match the
    # CLI/config contract: offset is ms/sqrt(s), rate is ppm/sqrt(s).
    # Observation and initial uncertainty fields are plain one-sigma ms/ppm.
    kalman_process_offset_noise_ms: float = DEFAULT_DRIFT_MODEL_CONFIG.kalman_process_offset_noise_ms
    kalman_process_rate_noise_ppm: float = DEFAULT_DRIFT_MODEL_CONFIG.kalman_process_rate_noise_ppm
    kalman_observation_noise_ms: float = DEFAULT_DRIFT_MODEL_CONFIG.kalman_observation_noise_ms
    kalman_initial_offset_uncertainty_ms: float = DEFAULT_DRIFT_MODEL_CONFIG.kalman_initial_offset_uncertainty_ms
    kalman_initial_rate_uncertainty_ppm: float = DEFAULT_DRIFT_MODEL_CONFIG.kalman_initial_rate_uncertainty_ppm
    kalman_validation_sample_count: int = DEFAULT_DRIFT_MODEL_CONFIG.kalman_validation_sample_count
    vad_strategy: Literal["adaptive_rms", "rms", "silero", "webrtc", "pyannote"] = "adaptive_rms"
    pyannote_model: str = DEFAULT_PYANNOTE_MODEL
    anchor_selection: AnchorSelectionConfig = field(default_factory=lambda: DEFAULT_ANCHOR_SELECTION_CONFIG)
    lang: str | None = None
    verbose_report: bool = False


def build_cli_argv(options: AlignmentOptions) -> list[str]:
    """Convert :class:`AlignmentOptions` to CLI-compatible argv."""

    allowed_methods = {"resample", "pitch_preserving"}
    allowed_vad_strategies = {"adaptive_rms", "rms", "silero", "webrtc", "pyannote"}
    allowed_drift_models = {"auto", "linear", "piecewise_linear", "spline", "kalman"}
    allowed_spline_knot_sources = {"auto", "piecewise_boundaries", "anchors"}
    if options.stretch_method not in allowed_methods:
        raise ValueError(f"stretch_method must be one of {sorted(allowed_methods)}")
    if options.drift_model not in allowed_drift_models:
        raise ValueError(f"drift_model must be one of {sorted(allowed_drift_models)}")
    if options.spline_knot_source not in allowed_spline_knot_sources:
        raise ValueError(f"spline_knot_source must be one of {sorted(allowed_spline_knot_sources)}")
    DriftModelConfig(
        drift_model=options.drift_model,
        allow_nonlinear_drift=options.allow_nonlinear_drift,
        min_anchors_for_piecewise=options.min_anchors_for_piecewise,
        min_anchors_per_segment=options.min_anchors_per_segment,
        max_breakpoints=options.max_breakpoints,
        min_residual_improvement_ms=options.min_residual_improvement_ms,
        min_relative_residual_improvement=options.min_relative_residual_improvement,
        max_abs_rate_deviation_ppm=options.max_abs_rate_deviation_ppm,
        max_rate_change_ppm=options.max_rate_change_ppm,
        warn_abs_rate_deviation_ppm=options.warn_abs_rate_deviation_ppm,
        max_anchor_gap_seconds=options.max_anchor_gap_seconds,
        min_anchors_for_spline=options.min_anchors_for_spline,
        spline_knot_source=options.spline_knot_source,
        min_knot_spacing_seconds=options.min_knot_spacing_seconds,
        spline_validation_sample_count=options.spline_validation_sample_count,
        min_anchors_for_kalman=options.min_anchors_for_kalman,
        kalman_process_offset_noise_ms=options.kalman_process_offset_noise_ms,
        kalman_process_rate_noise_ppm=options.kalman_process_rate_noise_ppm,
        kalman_observation_noise_ms=options.kalman_observation_noise_ms,
        kalman_initial_offset_uncertainty_ms=options.kalman_initial_offset_uncertainty_ms,
        kalman_initial_rate_uncertainty_ppm=options.kalman_initial_rate_uncertainty_ppm,
        kalman_validation_sample_count=options.kalman_validation_sample_count,
    )
    if options.vad_strategy not in allowed_vad_strategies:
        raise ValueError(f"vad_strategy must be one of {sorted(allowed_vad_strategies)}")
    if options.vad_strategy != "pyannote" and options.pyannote_model != DEFAULT_PYANNOTE_MODEL:
        raise ValueError("pyannote_model is only valid when vad_strategy='pyannote'")

    argv: list[str] = [
        "--master",
        str(options.master),
        "--out",
        str(options.out),
        "--analysis-sample-rate",
        str(options.analysis_sample_rate),
        "--local-adjust-threshold-ms",
        str(options.local_adjust_threshold_ms),
        "--stretch-ratio-warning-threshold",
        str(options.stretch_ratio_warning_threshold),
        "--stretch-method",
        options.stretch_method,
        "--drift-model",
        options.drift_model,
        "--min-anchors-for-piecewise",
        str(options.min_anchors_for_piecewise),
        "--min-anchors-per-segment",
        str(options.min_anchors_per_segment),
        "--max-breakpoints",
        str(options.max_breakpoints),
        "--min-residual-improvement-ms",
        str(options.min_residual_improvement_ms),
        "--min-relative-residual-improvement",
        str(options.min_relative_residual_improvement),
        "--max-abs-rate-deviation-ppm",
        str(options.max_abs_rate_deviation_ppm),
        "--max-rate-change-ppm",
        str(options.max_rate_change_ppm),
        "--warn-abs-rate-deviation-ppm",
        str(options.warn_abs_rate_deviation_ppm),
        "--max-anchor-gap-seconds",
        "none" if options.max_anchor_gap_seconds is None else str(options.max_anchor_gap_seconds),
        "--min-anchors-for-spline",
        str(options.min_anchors_for_spline),
        "--spline-knot-source",
        options.spline_knot_source,
        "--min-knot-spacing-seconds",
        str(options.min_knot_spacing_seconds),
        "--spline-validation-sample-count",
        str(options.spline_validation_sample_count),
        "--min-anchors-for-kalman",
        str(options.min_anchors_for_kalman),
        "--kalman-process-offset-noise-ms",
        str(options.kalman_process_offset_noise_ms),
        "--kalman-process-rate-noise-ppm",
        str(options.kalman_process_rate_noise_ppm),
        "--kalman-observation-noise-ms",
        str(options.kalman_observation_noise_ms),
        "--kalman-initial-offset-uncertainty-ms",
        str(options.kalman_initial_offset_uncertainty_ms),
        "--kalman-initial-rate-uncertainty-ppm",
        str(options.kalman_initial_rate_uncertainty_ppm),
        "--kalman-validation-sample-count",
        str(options.kalman_validation_sample_count),
        "--vad-strategy",
        options.vad_strategy,
        "--min-anchor-duration",
        str(options.anchor_selection.min_anchor_duration_seconds),
        "--base-anchor-duration",
        str(options.anchor_selection.base_anchor_duration_seconds),
        "--max-anchor-duration",
        str(options.anchor_selection.max_anchor_duration_seconds),
        "--anchor-density-per-minute",
        str(options.anchor_selection.anchor_density_per_minute),
        "--max-anchor-density-per-minute",
        str(options.anchor_selection.max_anchor_density_per_minute),
        "--min-anchor-count",
        str(options.anchor_selection.min_anchor_count),
        "--max-anchor-count",
        "none" if options.anchor_selection.max_anchor_count is None else str(options.anchor_selection.max_anchor_count),
    ]

    if options.anchor_selection.stratified_bin_count is not None:
        argv.extend(["--stratified-bin-count", str(options.anchor_selection.stratified_bin_count)])
    if options.anchor_selection.anchors_per_bin is not None:
        argv.extend(["--anchors-per-bin", str(options.anchor_selection.anchors_per_bin)])
    if options.anchor_selection.min_snr_db is not None:
        argv.extend(["--min-snr-db", str(options.anchor_selection.min_snr_db)])
    if options.anchor_selection.spectral_flatness_threshold is not None:
        argv.extend(["--spectral-flatness-threshold", str(options.anchor_selection.spectral_flatness_threshold)])

    if options.vad_strategy == "pyannote":
        argv.extend(["--pyannote-model", options.pyannote_model])

    for track_path in options.tracks:
        argv.extend(["--track", str(track_path)])

    if options.normalize_output:
        argv.append("--normalize-output")
    if options.local_adjust_enabled:
        argv.append("--local-adjust-enabled")
    if options.allow_nonlinear_drift:
        argv.append("--allow-nonlinear-drift")
    if options.stretch_ratio_auto_continue:
        argv.append("--stretch-ratio-auto-continue")
    if options.debug:
        argv.append("--debug")
    if options.verbose_report:
        argv.append("--verbose-report")
    if options.log_file is not None:
        argv.extend(["--log-file", str(options.log_file)])
    if options.lang:
        argv.extend(["--lang", options.lang])

    return argv


def run_alignment(options: AlignmentOptions, progress_callback=None, event_callback=None) -> int:
    """Run alignment pipeline and return CLI-compatible exit code."""

    if not options.tracks:
        raise ValueError("tracks must contain at least one path")

    if progress_callback is None and event_callback is None:
        return cli.main(build_cli_argv(options))
    return cli.main(build_cli_argv(options), progress_callback=progress_callback, event_callback=event_callback)
