from pathlib import Path

import pytest

from double_ender_sync.api import AlignmentOptions, build_cli_argv, run_alignment


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
