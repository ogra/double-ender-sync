import importlib
import numpy as np
from pathlib import Path

from double_ender_sync.alignment.timeline import apply_global_time_correction, build_inverse_time_map
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


class PiecewiseTestDrift:
    model_type = "piecewise_test"
    speaker_track = "speaker-a"

    def __init__(self, offset_seconds: float = 0.0) -> None:
        self.offset_seconds = offset_seconds

    def map_local_to_master(self, local_time_seconds: float) -> float:
        if local_time_seconds <= 0.5:
            return self.offset_seconds + local_time_seconds
        return self.offset_seconds + 0.5 + ((local_time_seconds - 0.5) * 1.5)

    def local_rate_at(self, local_time_seconds: float) -> float:
        return 1.0 if local_time_seconds <= 0.5 else 1.5

    def residuals_ms(self, anchors):
        return []

    def to_report_dict(self):
        return {"model_type": self.model_type}


class NonMonotonicTestDrift(PiecewiseTestDrift):
    model_type = "non_monotonic_test"

    def map_local_to_master(self, local_time_seconds: float) -> float:
        return -local_time_seconds


def test_inverse_time_map_round_trips_monotonic_model() -> None:
    drift = PiecewiseTestDrift(offset_seconds=0.25)

    inverse_map = build_inverse_time_map(drift, local_duration_seconds=1.0, sample_count=1001)

    master_times = np.array([0.25, 0.50, 0.75, 1.25], dtype=np.float64)
    expected_local = np.array([0.0, 0.25, 0.5, 5.0 / 6.0], dtype=np.float64)
    assert inverse_map.monotonicity_check.passed is True
    assert np.allclose(inverse_map.map_master_to_local(master_times), expected_local, atol=1e-3)


def test_inverse_time_map_rejects_non_monotonic_model() -> None:
    import pytest

    with pytest.raises(ValueError, match="strictly monotonic"):
        build_inverse_time_map(NonMonotonicTestDrift(), local_duration_seconds=1.0, sample_count=8)


def test_apply_global_time_correction_pads_outside_generic_mapping_support() -> None:
    sr = 100
    local = np.ones(sr, dtype=np.float32)
    master = np.zeros(int(sr * 2.0), dtype=np.float32)
    track = AudioTrack(path=Path("speaker-a.wav"), name="speaker-a", sample_rate=sr, duration_seconds=1.0, channels=1, original_samples=local, analysis_samples=local, analysis_sample_rate=sr)
    master_track = AudioTrack(path=Path("master.wav"), name="master", sample_rate=sr, duration_seconds=2.0, channels=1, original_samples=master, analysis_samples=master, analysis_sample_rate=sr)

    result = apply_global_time_correction(track, master_track, PiecewiseTestDrift(offset_seconds=0.5))

    assert result.render_method == "inverse_time_map"
    assert result.monotonicity_check is not None
    assert result.monotonicity_check.passed is True
    assert result.unsupported_regions is not None
    assert [(r.start_seconds, r.end_seconds, r.reason) for r in result.unsupported_regions] == [
        (0.0, 0.5, "before_drift_model_support"),
        (1.75, 2.0, "after_drift_model_support"),
    ]
    assert np.allclose(result.output_samples[:50], 0.0)
    assert np.max(result.output_samples[50:]) > 0.0


def test_apply_global_time_correction_rejects_pitch_preserving_for_generic_model() -> None:
    import pytest

    sr = 100
    local = np.ones(sr, dtype=np.float32)
    master = np.zeros(sr, dtype=np.float32)
    track = AudioTrack(path=Path("speaker-a.wav"), name="speaker-a", sample_rate=sr, duration_seconds=1.0, channels=1, original_samples=local, analysis_samples=local, analysis_sample_rate=sr)
    master_track = AudioTrack(path=Path("master.wav"), name="master", sample_rate=sr, duration_seconds=1.0, channels=1, original_samples=master, analysis_samples=master, analysis_sample_rate=sr)

    with pytest.raises(ValueError, match="LinearDrift only"):
        apply_global_time_correction(track, master_track, PiecewiseTestDrift(), stretch_method="pitch_preserving")


def test_linear_resample_uses_track_sample_rate_on_master_timeline() -> None:
    track_sr = 1000
    master_sr = 100
    local = np.ones(track_sr, dtype=np.float32)
    master = np.zeros(master_sr * 2, dtype=np.float32)
    track = AudioTrack(path=Path("speaker-a.wav"), name="speaker-a", sample_rate=track_sr, duration_seconds=1.0, channels=1, original_samples=local, analysis_samples=local, analysis_sample_rate=track_sr)
    master_track = AudioTrack(path=Path("master.wav"), name="master", sample_rate=master_sr, duration_seconds=2.0, channels=1, original_samples=master, analysis_samples=master, analysis_sample_rate=master_sr)
    drift = DriftEstimate(offset_seconds=0.5, stretch_ratio=1.0, anchor_count=4, residual_median_ms=1.0, residual_max_ms=2.0)

    result = apply_global_time_correction(track, master_track, drift)

    assert np.allclose(result.output_samples[:50], 0.0)
    assert np.allclose(result.output_samples[50:150], 1.0)
    assert np.allclose(result.output_samples[150:], 0.0)


class JumpGapTestDrift(PiecewiseTestDrift):
    model_type = "jump_gap_test"

    def map_local_to_master(self, local_time_seconds: float) -> float:
        if local_time_seconds <= 0.5:
            return local_time_seconds
        return local_time_seconds + 0.5


def test_inverse_time_map_marks_internal_master_gaps_unsupported() -> None:
    inverse_map = build_inverse_time_map(JumpGapTestDrift(), local_duration_seconds=1.0, sample_count=1001)

    assert [(round(r.start_seconds, 3), round(r.end_seconds, 3), r.reason) for r in inverse_map.unsupported_regions] == [
        (0.5, 1.0, "internal_drift_model_gap")
    ]
    assert inverse_map.monotonicity_check.message == "detected 1 internal unsupported master-time gap(s)"

    local_times = inverse_map.map_master_to_local(np.array([0.25, 0.75, 1.25], dtype=np.float64))
    assert np.isclose(local_times[0], 0.25)
    assert np.isnan(local_times[1])
    assert np.isclose(local_times[2], 0.75, atol=1e-3)


class RapidContinuousDrift(PiecewiseTestDrift):
    model_type = "rapid_continuous_test"

    def map_local_to_master(self, local_time_seconds: float) -> float:
        # Continuous and strictly monotonic, but with a narrow high-rate region
        # that can look like a gap in a coarse inverse-map table.
        return local_time_seconds + (0.5 / (1.0 + np.exp(-200.0 * (local_time_seconds - 0.5))))

    def local_rate_at(self, local_time_seconds: float) -> float:
        curve = 1.0 / (1.0 + np.exp(-200.0 * (local_time_seconds - 0.5)))
        return 1.0 + (100.0 * curve * (1.0 - curve))


def test_inverse_time_map_does_not_mark_refinable_rapid_rate_as_gap() -> None:
    inverse_map = build_inverse_time_map(RapidContinuousDrift(), local_duration_seconds=1.0, sample_count=64)

    assert inverse_map.unsupported_regions == []
    assert inverse_map.monotonicity_check.message is None


def test_build_inverse_time_map_rejects_zero_duration_with_clear_error() -> None:
    import pytest

    with pytest.raises(ValueError, match="local_duration_seconds must be positive"):
        build_inverse_time_map(PiecewiseTestDrift(), local_duration_seconds=0.0)


def test_apply_global_time_correction_pads_internal_master_gaps() -> None:
    sr = 100
    local = np.ones(sr, dtype=np.float32)
    master = np.zeros(sr * 2, dtype=np.float32)
    track = AudioTrack(path=Path("speaker-a.wav"), name="speaker-a", sample_rate=sr, duration_seconds=1.0, channels=1, original_samples=local, analysis_samples=local, analysis_sample_rate=sr)
    master_track = AudioTrack(path=Path("master.wav"), name="master", sample_rate=sr, duration_seconds=2.0, channels=1, original_samples=master, analysis_samples=master, analysis_sample_rate=sr)

    result = apply_global_time_correction(track, master_track, JumpGapTestDrift())

    assert any(region.reason == "internal_drift_model_gap" for region in result.unsupported_regions)
    assert np.max(result.output_samples[:50]) > 0.0
    assert np.allclose(result.output_samples[51:100], 0.0)
    assert np.max(result.output_samples[101:150]) > 0.0


def test_inverse_time_map_rendering_is_stable_across_chunks(monkeypatch) -> None:
    import double_ender_sync.alignment.timeline as timeline

    sr = 100
    local = np.linspace(0.0, 1.0, sr, dtype=np.float32)
    master = np.zeros(sr, dtype=np.float32)
    track = AudioTrack(path=Path("speaker-a.wav"), name="speaker-a", sample_rate=sr, duration_seconds=1.0, channels=1, original_samples=local, analysis_samples=local, analysis_sample_rate=sr)
    master_track = AudioTrack(path=Path("master.wav"), name="master", sample_rate=sr, duration_seconds=1.0, channels=1, original_samples=master, analysis_samples=master, analysis_sample_rate=sr)

    full_chunk = apply_global_time_correction(track, master_track, PiecewiseTestDrift()).output_samples
    monkeypatch.setattr(timeline, "_RENDER_CHUNK_SIZE_SAMPLES", 7)
    small_chunks = apply_global_time_correction(track, master_track, PiecewiseTestDrift()).output_samples

    assert np.allclose(small_chunks, full_chunk)


def test_apply_global_time_correction_renders_silence_for_zero_duration_generic_track() -> None:
    sr = 100
    local = np.array([], dtype=np.float32)
    master = np.zeros(sr, dtype=np.float32)
    track = AudioTrack(
        path=Path("speaker-a.wav"),
        name="speaker-a",
        sample_rate=sr,
        duration_seconds=0.0,
        channels=1,
        original_samples=local,
        analysis_samples=local,
        analysis_sample_rate=sr,
    )
    master_track = AudioTrack(
        path=Path("master.wav"),
        name="master",
        sample_rate=sr,
        duration_seconds=1.0,
        channels=1,
        original_samples=master,
        analysis_samples=master,
        analysis_sample_rate=sr,
    )

    result = apply_global_time_correction(track, master_track, PiecewiseTestDrift())

    assert result.render_method == "silence_empty_source"
    assert np.allclose(result.output_samples, 0.0)
    assert [(r.start_seconds, r.end_seconds, r.reason) for r in result.unsupported_regions] == [
        (0.0, 1.0, "empty_source_track")
    ]
    assert result.monotonicity_check is not None
    assert result.monotonicity_check.message == "source track is empty or has zero duration; rendered silence"


def test_pitch_preserving_uses_shared_sample_rate_resampler(monkeypatch) -> None:
    import double_ender_sync.alignment.timeline as timeline

    track_sr = 200
    master_sr = 100
    local = np.ones(track_sr, dtype=np.float32)
    master = np.zeros(master_sr, dtype=np.float32)
    track = AudioTrack(path=Path("speaker-a.wav"), name="speaker-a", sample_rate=track_sr, duration_seconds=1.0, channels=1, original_samples=local, analysis_samples=local, analysis_sample_rate=track_sr)
    master_track = AudioTrack(path=Path("master.wav"), name="master", sample_rate=master_sr, duration_seconds=1.0, channels=1, original_samples=master, analysis_samples=master, analysis_sample_rate=master_sr)
    drift = DriftEstimate(offset_seconds=0.0, stretch_ratio=1.0, anchor_count=4, residual_median_ms=1.0, residual_max_ms=2.0)
    calls = []

    def fake_time_stretch(samples: np.ndarray, stretch_ratio: float, stretch_method: str) -> np.ndarray:
        return samples

    def fake_resample(samples: np.ndarray, src_sample_rate: int, dst_sample_rate: int) -> np.ndarray:
        calls.append((src_sample_rate, dst_sample_rate))
        return np.full(dst_sample_rate, 0.5, dtype=np.float32)

    monkeypatch.setattr(timeline, "_stretch_samples", fake_time_stretch)
    monkeypatch.setattr(timeline, "resample_to_sample_rate", fake_resample)

    result = apply_global_time_correction(track, master_track, drift, stretch_method="pitch_preserving")

    assert calls == [(track_sr, master_sr)]
    assert np.allclose(result.output_samples, 0.5)
