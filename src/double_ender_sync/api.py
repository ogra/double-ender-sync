from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

from double_ender_sync import cli
from double_ender_sync.analysis.vad import DEFAULT_PYANNOTE_MODEL
from double_ender_sync.config import DEFAULT_ANCHOR_SELECTION_CONFIG, AnchorSelectionConfig


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
    vad_strategy: Literal["adaptive_rms", "rms", "silero", "webrtc", "pyannote"] = "adaptive_rms"
    pyannote_model: str = DEFAULT_PYANNOTE_MODEL
    anchor_selection: AnchorSelectionConfig = field(default_factory=lambda: DEFAULT_ANCHOR_SELECTION_CONFIG)
    lang: str | None = None


def build_cli_argv(options: AlignmentOptions) -> list[str]:
    """Convert :class:`AlignmentOptions` to CLI-compatible argv."""

    allowed_methods = {"resample", "pitch_preserving"}
    allowed_vad_strategies = {"adaptive_rms", "rms", "silero", "webrtc", "pyannote"}
    if options.stretch_method not in allowed_methods:
        raise ValueError(f"stretch_method must be one of {sorted(allowed_methods)}")
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
    if options.stretch_ratio_auto_continue:
        argv.append("--stretch-ratio-auto-continue")
    if options.debug:
        argv.append("--debug")
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
