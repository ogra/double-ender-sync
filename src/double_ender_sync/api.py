from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from double_ender_sync import cli
from double_ender_sync.analysis.vad import DEFAULT_PYANNOTE_MODEL


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
    ]

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
