from pathlib import Path

import pytest

import double_ender_sync
from double_ender_sync.api import AlignmentOptions, build_cli_argv, get_version, run_alignment
from double_ender_sync.analysis.vad import MODERN_PYANNOTE_SEGMENTATION_MODEL


def test_api_exposes_package_version() -> None:
    assert get_version() == "0.2.5"
    assert double_ender_sync.__version__ == "0.2.5"


def test_build_cli_argv_includes_required_fields() -> None:
    options = AlignmentOptions(
        master=Path("input/master.wav"),
        tracks=[Path("input/speaker-a.wav"), Path("input/speaker-b.wav")],
        out=Path("output"),
    )

    argv = build_cli_argv(options)

    assert "--master" in argv
    assert "input/master.wav" in argv
    assert argv.count("--track") == 2
    assert "--out" in argv
    assert "output" in argv


def test_run_alignment_calls_cli_main(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    def fake_main(argv: list[str]) -> int:
        called["argv"] = argv
        return 0

    monkeypatch.setattr("double_ender_sync.api.cli.main", fake_main)

    options = AlignmentOptions(
        master=Path("input/master.wav"),
        tracks=[Path("input/speaker-a.wav")],
        out=Path("output"),
        debug=True,
    )

    exit_code = run_alignment(options)

    assert exit_code == 0
    assert "--debug" in called["argv"]


def test_run_alignment_requires_at_least_one_track() -> None:
    options = AlignmentOptions(master=Path("input/master.wav"), tracks=[], out=Path("output"))

    with pytest.raises(ValueError, match="at least one"):
        run_alignment(options)


def test_build_cli_argv_includes_stretch_options() -> None:
    options = AlignmentOptions(
        master=Path("input/master.wav"),
        tracks=[Path("input/speaker-a.wav")],
        out=Path("output"),
        stretch_ratio_warning_threshold=0.004,
        stretch_ratio_auto_continue=True,
        stretch_method="pitch_preserving",
    )

    argv = build_cli_argv(options)

    assert "--stretch-ratio-warning-threshold" in argv
    assert "0.004" in argv
    assert "--stretch-ratio-auto-continue" in argv
    assert "--stretch-method" in argv
    assert "pitch_preserving" in argv


def test_build_cli_argv_rejects_invalid_stretch_method() -> None:
    options = AlignmentOptions(master=Path("input/master.wav"), tracks=[Path("input/speaker-a.wav")], out=Path("output"), stretch_method="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="stretch_method"):
        build_cli_argv(options)


def test_build_cli_argv_includes_lang_when_set() -> None:
    options = AlignmentOptions(
        master=Path("input/master.wav"),
        tracks=[Path("input/speaker-a.wav")],
        out=Path("output"),
        lang="ja",
    )

    argv = build_cli_argv(options)

    assert "--lang" in argv
    assert "ja" in argv


def test_build_cli_argv_includes_vad_strategy() -> None:
    options = AlignmentOptions(
        master=Path("input/master.wav"),
        tracks=[Path("input/speaker-a.wav")],
        out=Path("output"),
        vad_strategy="silero",
    )
    argv = build_cli_argv(options)
    assert "--vad-strategy" in argv
    assert "silero" in argv


def test_build_cli_argv_includes_pyannote_vad_strategy() -> None:
    options = AlignmentOptions(
        master=Path("input/master.wav"),
        tracks=[Path("input/speaker-a.wav")],
        out=Path("output"),
        vad_strategy="pyannote",
    )
    argv = build_cli_argv(options)
    assert "--vad-strategy" in argv
    assert "pyannote" in argv


def test_build_cli_argv_rejects_invalid_vad_strategy() -> None:
    options = AlignmentOptions(
        master=Path("input/master.wav"),
        tracks=[Path("input/speaker-a.wav")],
        out=Path("output"),
        vad_strategy="bad",  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="vad_strategy"):
        build_cli_argv(options)


def test_build_cli_argv_accepts_webrtc_vad_strategy() -> None:
    options = AlignmentOptions(
        master=Path("input/master.wav"),
        tracks=[Path("input/speaker-a.wav")],
        out=Path("output"),
        vad_strategy="webrtc",
    )
    argv = build_cli_argv(options)
    assert "--vad-strategy" in argv
    assert "webrtc" in argv


def test_build_cli_argv_includes_pyannote_model_for_pyannote_strategy() -> None:
    options = AlignmentOptions(
        master=Path("input/master.wav"),
        tracks=[Path("input/speaker-a.wav")],
        out=Path("output"),
        vad_strategy="pyannote",
        pyannote_model=MODERN_PYANNOTE_SEGMENTATION_MODEL,
    )

    argv = build_cli_argv(options)

    assert "--pyannote-model" in argv
    assert MODERN_PYANNOTE_SEGMENTATION_MODEL in argv


def test_build_cli_argv_rejects_pyannote_model_for_non_pyannote_strategy() -> None:
    options = AlignmentOptions(
        master=Path("input/master.wav"),
        tracks=[Path("input/speaker-a.wav")],
        out=Path("output"),
        vad_strategy="adaptive_rms",
        pyannote_model=MODERN_PYANNOTE_SEGMENTATION_MODEL,
    )

    with pytest.raises(ValueError, match="pyannote_model"):
        build_cli_argv(options)


def test_build_cli_argv_includes_linear_drift_model_policy() -> None:
    options = AlignmentOptions(
        master=Path("input/master.wav"),
        tracks=[Path("input/speaker-a.wav")],
        out=Path("output"),
        drift_model="linear",
    )

    argv = build_cli_argv(options)

    assert "--drift-model" in argv
    assert "linear" in argv
    assert "--allow-nonlinear-drift" not in argv


def test_build_cli_argv_rejects_piecewise_without_experimental_gate() -> None:
    options = AlignmentOptions(
        master=Path("input/master.wav"),
        tracks=[Path("input/speaker-a.wav")],
        out=Path("output"),
        drift_model="piecewise_linear",  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="requires allow_nonlinear_drift"):
        build_cli_argv(options)


def test_build_cli_argv_includes_piecewise_gate_and_thresholds() -> None:
    options = AlignmentOptions(
        master=Path("input/master.wav"),
        tracks=[Path("input/speaker-a.wav")],
        out=Path("output"),
        drift_model="piecewise_linear",
        allow_nonlinear_drift=True,
        max_breakpoints=1,
        min_residual_improvement_ms=2.5,
    )

    argv = build_cli_argv(options)

    assert "--allow-nonlinear-drift" in argv
    assert "--drift-model" in argv
    assert "piecewise_linear" in argv
    assert "--max-breakpoints" in argv
    assert "1" in argv
    assert "--min-residual-improvement-ms" in argv
    assert "2.5" in argv


def test_build_cli_argv_rejects_invalid_spline_knot_source_even_for_linear_policy() -> None:
    options = AlignmentOptions(
        master=Path("input/master.wav"),
        tracks=[Path("input/speaker-a.wav")],
        out=Path("output"),
        drift_model="linear",
        spline_knot_source="bad",  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="spline_knot_source"):
        build_cli_argv(options)


def test_build_cli_argv_includes_spline_drift_policy() -> None:
    options = AlignmentOptions(
        master=Path("input/master.wav"),
        tracks=[Path("input/speaker-a.wav")],
        out=Path("output"),
        drift_model="spline",
        allow_nonlinear_drift=True,
        min_anchors_for_spline=7,
        spline_knot_source="anchors",
        min_knot_spacing_seconds=45.0,
    )

    argv = build_cli_argv(options)

    assert "--allow-nonlinear-drift" in argv
    assert "--drift-model" in argv
    assert "spline" in argv
    assert "--min-anchors-for-spline" in argv
    assert "7" in argv
    assert "--spline-knot-source" in argv
    assert "anchors" in argv
    assert "--min-knot-spacing-seconds" in argv
    assert "45.0" in argv


def test_build_cli_argv_includes_max_anchor_gap_seconds() -> None:
    options = AlignmentOptions(
        master=Path("input/master.wav"),
        tracks=[Path("input/speaker-a.wav")],
        out=Path("output"),
        max_anchor_gap_seconds=123.5,
    )

    argv = build_cli_argv(options)

    assert "--max-anchor-gap-seconds" in argv
    assert "123.5" in argv


def test_build_cli_argv_includes_verbose_report_flag() -> None:
    options = AlignmentOptions(
        master=Path("input/master.wav"),
        tracks=[Path("input/speaker-a.wav")],
        out=Path("output"),
        verbose_report=True,
    )

    argv = build_cli_argv(options)

    assert "--verbose-report" in argv


def test_build_cli_argv_includes_anchor_matching_options() -> None:
    from double_ender_sync.config import AnchorMatchingConfig

    custom_matching = AnchorMatchingConfig(
        nms_exclusion_seconds=0.08,
        ncc_min_score=0.55,
        ncc_min_margin=0.12,
        ncc_min_prominence=0.08,
        ncc_good_width_seconds=0.003,
        ncc_bad_width_seconds=0.07,
        ncc_margin_low=0.04,
        ncc_margin_high=0.25,
        ncc_prominence_low=0.02,
        ncc_prominence_high=0.18,
        gcc_phat_enabled=True,
        gcc_phat_only_when_ambiguous=True,
        gcc_phat_agreement_tolerance_seconds=0.04,
        min_confidence_for_fit=0.06,
    )

    options = AlignmentOptions(
        master=Path("input/master.wav"),
        tracks=[Path("input/speaker-a.wav")],
        out=Path("output"),
        anchor_matching=custom_matching,
    )
    argv = build_cli_argv(options)

    assert "--nms-exclusion-seconds" in argv
    assert "0.08" in argv
    assert "--ncc-min-score" in argv
    assert "0.55" in argv
    assert "--ncc-min-margin" in argv
    assert "0.12" in argv
    assert "--ncc-min-prominence" in argv
    assert "0.08" in argv
    assert "--ncc-good-width-seconds" in argv
    assert "0.003" in argv
    assert "--ncc-bad-width-seconds" in argv
    assert "0.07" in argv
    assert "--ncc-margin-low" in argv
    assert "0.04" in argv
    assert "--ncc-margin-high" in argv
    assert "0.25" in argv
    assert "--ncc-prominence-low" in argv
    assert "0.02" in argv
    assert "--ncc-prominence-high" in argv
    assert "0.18" in argv
    assert "--gcc-phat-agreement-tolerance-seconds" in argv
    assert "0.04" in argv
    assert "--min-confidence-for-fit" in argv
    assert "0.06" in argv
    assert "--gcc-phat-enabled" in argv
    assert "--no-gcc-phat" not in argv
    assert "--gcc-phat-only-when-ambiguous" in argv
    assert "--no-gcc-phat-ambiguous-only" not in argv


def test_build_cli_argv_includes_anchor_matching_no_gcc_phat() -> None:
    from double_ender_sync.config import AnchorMatchingConfig

    custom_matching = AnchorMatchingConfig(
        gcc_phat_enabled=False,
        gcc_phat_only_when_ambiguous=False,
    )

    options = AlignmentOptions(
        master=Path("input/master.wav"),
        tracks=[Path("input/speaker-a.wav")],
        out=Path("output"),
        anchor_matching=custom_matching,
    )
    argv = build_cli_argv(options)

    assert "--no-gcc-phat" in argv
    assert "--gcc-phat-enabled" not in argv
    assert "--no-gcc-phat-ambiguous-only" in argv
    assert "--gcc-phat-only-when-ambiguous" not in argv
