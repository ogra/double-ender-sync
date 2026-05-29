import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from double_ender_sync.analysis.vad import AdaptiveRmsVadStrategy, DEFAULT_PYANNOTE_MODEL, MODERN_PYANNOTE_SEGMENTATION_MODEL
from double_ender_sync.cli import main, parse_args


def _write_tone(path: Path, sample_rate: int, hz: float, duration_seconds: float) -> None:
    t = np.arange(int(sample_rate * duration_seconds)) / sample_rate
    samples = (0.2 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    sf.write(path, samples, sample_rate)


def test_cli_prints_version(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "version 0.2.5"


def test_cli_prints_version_with_short_flag(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["-V"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "version 0.2.5"


def test_cli_generates_sync_report(tmp_path: Path) -> None:
    master = tmp_path / "master.wav"
    track_a = tmp_path / "speaker-a.wav"
    out_dir = tmp_path / "output"

    _write_tone(master, 48000, 220.0, 0.5)
    _write_tone(track_a, 44100, 330.0, 0.5)

    exit_code = main([
        "--master",
        str(master),
        "--track",
        str(track_a),
        "--out",
        str(out_dir),
    ])

    assert exit_code == 0
    report = json.loads((out_dir / "sync-report.json").read_text(encoding="utf-8"))

    assert report["report_type"] == "alignment_diagnostics"
    assert report["schema_version"] == "1"
    assert "phase" not in report
    assert report["analysis"]["sample_rate"] == 16000
    assert report["master"]["sample_rate"] == 48000
    assert report["tracks"][0]["sample_rate"] == 44100
    assert (out_dir / "sync-markers.csv").exists()
    assert (out_dir / "warnings.txt").exists()



def test_cli_reports_phase2_fields(tmp_path: Path) -> None:
    master = tmp_path / "master.wav"
    track_a = tmp_path / "speaker-a.wav"
    out_dir = tmp_path / "output"

    _write_tone(master, 16000, 220.0, 1.0)
    _write_tone(track_a, 16000, 220.0, 1.0)

    exit_code = main(["--master", str(master), "--track", str(track_a), "--out", str(out_dir)])
    assert exit_code == 0

    report = json.loads((out_dir / "sync-report.json").read_text(encoding="utf-8"))
    track_report = report["tracks"][0]
    assert "speech_segment_summary" in track_report
    assert "anchor_candidate_summary" in track_report
    assert "initial_offset" in track_report
    assert "drift_anchor_match_summary" in track_report
    assert "speech_segments" not in track_report
    assert "anchor_candidates" not in track_report
    assert "drift_anchor_matches" not in track_report
    assert "drift_estimate" not in track_report
    assert "global_correction" in track_report


def test_cli_writes_debug_log_file(tmp_path: Path) -> None:
    master = tmp_path / "master.wav"
    track_a = tmp_path / "speaker-a.wav"
    out_dir = tmp_path / "output"

    _write_tone(master, 16000, 220.0, 1.0)
    _write_tone(track_a, 16000, 220.0, 1.0)

    exit_code = main([
        "--master", str(master),
        "--track", str(track_a),
        "--out", str(out_dir),
        "--debug",
    ])
    assert exit_code == 0

    log_file = out_dir / "double-ender-sync.log"
    assert log_file.exists()
    contents = log_file.read_text(encoding="utf-8")
    assert "Starting alignment run" in contents
    assert "anchor_candidates" in contents



def test_cli_prints_progress_and_eta(tmp_path: Path, capsys) -> None:
    master = tmp_path / "master.wav"
    track_a = tmp_path / "speaker-a.wav"
    out_dir = tmp_path / "output"

    _write_tone(master, 16000, 220.0, 1.0)
    _write_tone(track_a, 16000, 220.0, 1.0)

    exit_code = main(["--master", str(master), "--track", str(track_a), "--out", str(out_dir)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "[progress]" in captured.out
    assert "ETA" in captured.out


def test_cli_help_includes_normalize_output(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--normalize-output" in captured.out
    assert "--stretch-ratio-warning-threshold" in captured.out
    assert "--stretch-method" in captured.out
    assert "--stretch-ratio-auto-continue" in captured.out
    assert "--pyannote-model" in captured.out
    assert "--drift-model" in captured.out
    assert "--allow-nonlinear-drift" in captured.out
    assert "--verbose-report" in captured.out



@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_cli_rejects_non_finite_or_non_positive_max_anchor_gap_seconds(
    value: str,
    capsys,
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_args([
            "--master",
            "master.wav",
            "--track",
            "speaker.wav",
            "--out",
            "output",
            f"--max-anchor-gap-seconds={value}",
        ])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "must be a finite value > 0" in captured.err


def test_cli_accepts_disabled_max_anchor_gap_seconds() -> None:
    args = parse_args([
        "--master",
        "master.wav",
        "--track",
        "speaker.wav",
        "--out",
        "output",
        "--max-anchor-gap-seconds",
        "off",
    ])

    assert args.max_anchor_gap_seconds is None

def test_cli_rejects_negative_stretch_ratio_threshold(tmp_path: Path) -> None:
    master = tmp_path / "master.wav"
    track_a = tmp_path / "speaker-a.wav"
    out_dir = tmp_path / "output"
    _write_tone(master, 16000, 220.0, 0.5)
    _write_tone(track_a, 16000, 220.0, 0.5)

    exit_code = main(["--master", str(master), "--track", str(track_a), "--out", str(out_dir), "--stretch-ratio-warning-threshold", "-0.1"])
    assert exit_code == 2


def test_cli_rejects_pyannote_model_without_pyannote_strategy(tmp_path: Path) -> None:
    master = tmp_path / "master.wav"
    track_a = tmp_path / "speaker-a.wav"
    out_dir = tmp_path / "output"
    _write_tone(master, 16000, 220.0, 0.5)
    _write_tone(track_a, 16000, 220.0, 0.5)

    exit_code = main([
        "--master", str(master),
        "--track", str(track_a),
        "--out", str(out_dir),
        "--vad-strategy", "adaptive_rms",
        "--pyannote-model", MODERN_PYANNOTE_SEGMENTATION_MODEL,
    ])

    assert exit_code == 2


def test_cli_passes_modern_pyannote_model_to_vad_strategy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    master = tmp_path / "master.wav"
    track_a = tmp_path / "speaker-a.wav"
    out_dir = tmp_path / "output"
    _write_tone(master, 16000, 220.0, 1.0)
    _write_tone(track_a, 16000, 220.0, 1.0)
    captured: dict[str, object] = {}

    def fake_build_vad_strategy(name: str, pyannote_model: str | None = None) -> AdaptiveRmsVadStrategy:
        captured["name"] = name
        captured["pyannote_model"] = pyannote_model
        return AdaptiveRmsVadStrategy()

    monkeypatch.setattr("double_ender_sync.cli.build_vad_strategy", fake_build_vad_strategy)

    exit_code = main([
        "--master", str(master),
        "--track", str(track_a),
        "--out", str(out_dir),
        "--vad-strategy", "pyannote",
        "--pyannote-model", MODERN_PYANNOTE_SEGMENTATION_MODEL,
    ])

    assert exit_code == 0
    assert captured == {"name": "pyannote", "pyannote_model": MODERN_PYANNOTE_SEGMENTATION_MODEL}
    report = json.loads((out_dir / "sync-report.json").read_text(encoding="utf-8"))
    assert report["analysis"]["vad"] == {"strategy": "pyannote", "pyannote_model": MODERN_PYANNOTE_SEGMENTATION_MODEL}


def test_cli_uses_community_pyannote_model_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    master = tmp_path / "master.wav"
    track_a = tmp_path / "speaker-a.wav"
    out_dir = tmp_path / "output"
    _write_tone(master, 16000, 220.0, 1.0)
    _write_tone(track_a, 16000, 220.0, 1.0)
    captured: dict[str, object] = {}

    def fake_build_vad_strategy(name: str, pyannote_model: str | None = None) -> AdaptiveRmsVadStrategy:
        captured["name"] = name
        captured["pyannote_model"] = pyannote_model
        return AdaptiveRmsVadStrategy()

    monkeypatch.setattr("double_ender_sync.cli.build_vad_strategy", fake_build_vad_strategy)

    exit_code = main([
        "--master", str(master),
        "--track", str(track_a),
        "--out", str(out_dir),
        "--vad-strategy", "pyannote",
    ])

    assert exit_code == 0
    assert captured == {"name": "pyannote", "pyannote_model": DEFAULT_PYANNOTE_MODEL}
    report = json.loads((out_dir / "sync-report.json").read_text(encoding="utf-8"))
    assert report["analysis"]["vad"] == {"strategy": "pyannote", "pyannote_model": DEFAULT_PYANNOTE_MODEL}


def test_cli_report_includes_vad_metadata(tmp_path: Path) -> None:
    master = tmp_path / "master.wav"
    track_a = tmp_path / "speaker-a.wav"
    out_dir = tmp_path / "output"
    _write_tone(master, 16000, 220.0, 1.0)
    _write_tone(track_a, 16000, 220.0, 1.0)

    exit_code = main(["--master", str(master), "--track", str(track_a), "--out", str(out_dir)])

    assert exit_code == 0
    report = json.loads((out_dir / "sync-report.json").read_text(encoding="utf-8"))
    assert report["analysis"]["vad"] == {"strategy": "adaptive_rms", "pyannote_model": None}
    assert report["tracks"][0]["vad"] == {"strategy": "adaptive_rms", "pyannote_model": None}


def test_cli_accepts_allow_nonlinear_drift_for_piecewise_phase3(tmp_path: Path) -> None:
    master = tmp_path / "master.wav"
    track_a = tmp_path / "speaker-a.wav"
    out_dir = tmp_path / "output"
    _write_tone(master, 16000, 220.0, 0.5)
    _write_tone(track_a, 16000, 220.0, 0.5)

    exit_code = main([
        "--master", str(master),
        "--track", str(track_a),
        "--out", str(out_dir),
        "--allow-nonlinear-drift",
    ])

    assert exit_code == 0
    report = json.loads((out_dir / "sync-report.json").read_text(encoding="utf-8"))
    assert report["analysis"]["drift_model_selection"]["allow_nonlinear_drift"] is True
