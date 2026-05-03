import importlib
import numpy as np
from pathlib import Path

from double_ender_sync.alignment.timeline import apply_global_time_correction
from double_ender_sync.analysis.drift import DriftEstimate
from double_ender_sync.types import AudioTrack


def test_apply_global_time_correction_places_audio_on_master_timeline() -> None:
    sr = 8000
    local = np.ones(int(sr * 0.5), dtype=np.float32) * 0.25
    master = np.zeros(int(sr * 2.0), dtype=np.float32)

    track = AudioTrack(
        path=Path("speaker-a.wav"),
        name="speaker-a",
        sample_rate=sr,
        duration_seconds=0.5,
        channels=1,
        original_samples=local,
        analysis_samples=local,
        analysis_sample_rate=sr,
    )
    master_track = AudioTrack(
        path=Path("master.wav"),
        name="master",
        sample_rate=sr,
        duration_seconds=2.0,
        channels=1,
        original_samples=master,
        analysis_samples=master,
        analysis_sample_rate=sr,
    )

    drift = DriftEstimate(offset_seconds=0.75, stretch_ratio=1.0, anchor_count=4, residual_median_ms=10.0, residual_max_ms=20.0)
    result = apply_global_time_correction(track, master_track, drift)

    start = int(0.75 * sr)
    assert np.allclose(result.output_samples[start : start + len(local)], local)
    assert result.output_samples.shape[0] == master.shape[0]


def test_apply_global_time_correction_rejects_invalid_stretch_method() -> None:
    sr = 8000
    local = np.ones(int(sr * 0.25), dtype=np.float32)
    master = np.zeros(int(sr * 1.0), dtype=np.float32)
    track = AudioTrack(path=Path("speaker-a.wav"), name="speaker-a", sample_rate=sr, duration_seconds=0.25, channels=1, original_samples=local, analysis_samples=local, analysis_sample_rate=sr)
    master_track = AudioTrack(path=Path("master.wav"), name="master", sample_rate=sr, duration_seconds=1.0, channels=1, original_samples=master, analysis_samples=master, analysis_sample_rate=sr)
    drift = DriftEstimate(offset_seconds=0.0, stretch_ratio=1.0, anchor_count=4, residual_median_ms=1.0, residual_max_ms=2.0)

    import pytest
    with pytest.raises(ValueError, match="stretch_method"):
        apply_global_time_correction(track, master_track, drift, stretch_method="invalid")


def test_pitch_preserving_raises_when_librosa_missing(monkeypatch) -> None:
    sr = 8000
    local = np.ones(int(sr * 0.25), dtype=np.float32)
    master = np.zeros(int(sr * 1.0), dtype=np.float32)
    track = AudioTrack(path=Path("speaker-a.wav"), name="speaker-a", sample_rate=sr, duration_seconds=0.25, channels=1, original_samples=local, analysis_samples=local, analysis_sample_rate=sr)
    master_track = AudioTrack(path=Path("master.wav"), name="master", sample_rate=sr, duration_seconds=1.0, channels=1, original_samples=master, analysis_samples=master, analysis_sample_rate=sr)
    drift = DriftEstimate(offset_seconds=0.0, stretch_ratio=1.0, anchor_count=4, residual_median_ms=1.0, residual_max_ms=2.0)

    def fake_import_module(name: str):
        if name == "librosa":
            raise ModuleNotFoundError("no module named librosa")
        return importlib.import_module(name)

    monkeypatch.setattr("importlib.import_module", fake_import_module)

    import pytest
    with pytest.raises(RuntimeError, match="requires librosa"):
        apply_global_time_correction(track, master_track, drift, stretch_method="pitch_preserving")
