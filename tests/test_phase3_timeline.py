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


def test_rubberband_raises_when_pyrubberband_missing(monkeypatch) -> None:
    import pytest
    sr = 8000
    local = np.ones(int(sr * 0.25), dtype=np.float32)
    master = np.zeros(int(sr * 1.0), dtype=np.float32)
    track = AudioTrack(path=Path("speaker-a.wav"), name="speaker-a", sample_rate=sr, duration_seconds=0.25, channels=1, original_samples=local, analysis_samples=local, analysis_sample_rate=sr)
    master_track = AudioTrack(path=Path("master.wav"), name="master", sample_rate=sr, duration_seconds=1.0, channels=1, original_samples=master, analysis_samples=master, analysis_sample_rate=sr)
    drift = DriftEstimate(offset_seconds=0.0, stretch_ratio=1.0, anchor_count=4, residual_median_ms=1.0, residual_max_ms=2.0)
    original_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "pyrubberband":
            raise ModuleNotFoundError("no module named pyrubberband")
        return original_import_module(name)

    monkeypatch.setattr("importlib.import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="requires pyrubberband"):
        apply_global_time_correction(track, master_track, drift, stretch_method="rubberband")


def test_rubberband_render_method_is_linear_rubberband() -> None:
    """rubberband stretch_method produces render_method='linear_rubberband' in result."""
    import shutil
    import pytest
    pytest.importorskip("pyrubberband")
    if shutil.which("rubberband") is None:
        pytest.skip("rubberband binary not found on PATH")

    sr = 8000
    local = np.ones(int(sr * 0.25), dtype=np.float32) * 0.1
    master = np.zeros(int(sr * 1.0), dtype=np.float32)
    track = AudioTrack(path=Path("speaker-a.wav"), name="speaker-a", sample_rate=sr, duration_seconds=0.25, channels=1, original_samples=local, analysis_samples=local, analysis_sample_rate=sr)
    master_track = AudioTrack(path=Path("master.wav"), name="master", sample_rate=sr, duration_seconds=1.0, channels=1, original_samples=master, analysis_samples=master, analysis_sample_rate=sr)
    drift = DriftEstimate(offset_seconds=0.0, stretch_ratio=1.0, anchor_count=4, residual_median_ms=1.0, residual_max_ms=2.0)

    result = apply_global_time_correction(track, master_track, drift, stretch_method="rubberband")
    assert result.render_method == "linear_rubberband"
    assert result.output_samples.dtype == np.float32
    assert result.output_samples.shape[0] == master.shape[0]


def test_rubberband_is_valid_stretch_method() -> None:
    from double_ender_sync.alignment.stretch import VALID_STRETCH_METHODS
    assert "rubberband" in VALID_STRETCH_METHODS


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
    import double_ender_sync.alignment.stretch as stretch
    import double_ender_sync.alignment.timeline as timeline

    track_sr = 200
    master_sr = 100
    local = np.ones(track_sr, dtype=np.float32)
    master = np.zeros(master_sr, dtype=np.float32)
    track = AudioTrack(path=Path("speaker-a.wav"), name="speaker-a", sample_rate=track_sr, duration_seconds=1.0, channels=1, original_samples=local, analysis_samples=local, analysis_sample_rate=track_sr)
    master_track = AudioTrack(path=Path("master.wav"), name="master", sample_rate=master_sr, duration_seconds=1.0, channels=1, original_samples=master, analysis_samples=master, analysis_sample_rate=master_sr)
    drift = DriftEstimate(offset_seconds=0.0, stretch_ratio=1.0, anchor_count=4, residual_median_ms=1.0, residual_max_ms=2.0)
    calls = []

    def fake_stretch_by_ratio(self, samples: np.ndarray, stretch_ratio: float, *, sample_rate: int | None = None) -> np.ndarray:
        return samples

    def fake_resample(samples: np.ndarray, src_sample_rate: int, dst_sample_rate: int) -> np.ndarray:
        calls.append((src_sample_rate, dst_sample_rate))
        return np.full(dst_sample_rate, 0.5, dtype=np.float32)

    monkeypatch.setattr(stretch.LibrosaStretcher, "stretch_by_ratio", fake_stretch_by_ratio)
    monkeypatch.setattr(timeline, "resample_to_sample_rate", fake_resample)

    result = apply_global_time_correction(track, master_track, drift, stretch_method="pitch_preserving")

    assert calls == [(track_sr, master_sr)]
    assert np.allclose(result.output_samples, 0.5)


def test_resample_path_does_not_invoke_numpy_interpolation_stretcher(monkeypatch) -> None:
    """The 'resample' path uses the chunked np.interp loop, not NumpyInterpolationStretcher."""
    import double_ender_sync.alignment.stretch as stretch

    sr = 8000
    local = np.ones(sr, dtype=np.float32) * 0.5
    master = np.zeros(sr, dtype=np.float32)
    track = AudioTrack(
        path=Path("speaker-a.wav"),
        name="speaker-a",
        sample_rate=sr,
        duration_seconds=1.0,
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
    drift = DriftEstimate(
        offset_seconds=0.0,
        stretch_ratio=1.0,
        anchor_count=4,
        residual_median_ms=1.0,
        residual_max_ms=2.0,
    )

    calls = []

    def spy_stretch_by_ratio(self, samples, stretch_ratio):
        calls.append(stretch_ratio)
        return samples

    monkeypatch.setattr(stretch.NumpyInterpolationStretcher, "stretch_by_ratio", spy_stretch_by_ratio)

    apply_global_time_correction(track, master_track, drift, stretch_method="resample")

    assert calls == [], "NumpyInterpolationStretcher.stretch_by_ratio must not be called in the resample path"


def test_soxr_raises_when_soxr_missing(monkeypatch) -> None:
    import pytest
    sr = 8000
    local = np.ones(int(sr * 0.25), dtype=np.float32)
    master = np.zeros(int(sr * 1.0), dtype=np.float32)
    track = AudioTrack(path=Path("speaker-a.wav"), name="speaker-a", sample_rate=sr, duration_seconds=0.25, channels=1, original_samples=local, analysis_samples=local, analysis_sample_rate=sr)
    master_track = AudioTrack(path=Path("master.wav"), name="master", sample_rate=sr, duration_seconds=1.0, channels=1, original_samples=master, analysis_samples=master, analysis_sample_rate=sr)
    drift = DriftEstimate(offset_seconds=0.0, stretch_ratio=1.0, anchor_count=4, residual_median_ms=1.0, residual_max_ms=2.0)
    original_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "soxr":
            raise ModuleNotFoundError("no module named soxr")
        return original_import_module(name)

    monkeypatch.setattr("importlib.import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="requires the soxr package"):
        apply_global_time_correction(track, master_track, drift, stretch_method="soxr")


def test_soxr_render_method_is_linear_soxr() -> None:
    """soxr stretch_method produces render_method='linear_soxr' in result."""
    import pytest
    pytest.importorskip("soxr")

    sr = 8000
    local = np.ones(int(sr * 0.25), dtype=np.float32) * 0.1
    master = np.zeros(int(sr * 1.0), dtype=np.float32)
    track = AudioTrack(path=Path("speaker-a.wav"), name="speaker-a", sample_rate=sr, duration_seconds=0.25, channels=1, original_samples=local, analysis_samples=local, analysis_sample_rate=sr)
    master_track = AudioTrack(path=Path("master.wav"), name="master", sample_rate=sr, duration_seconds=1.0, channels=1, original_samples=master, analysis_samples=master, analysis_sample_rate=sr)
    drift = DriftEstimate(offset_seconds=0.0, stretch_ratio=1.0, anchor_count=4, residual_median_ms=1.0, residual_max_ms=2.0)

    result = apply_global_time_correction(track, master_track, drift, stretch_method="soxr")
    assert result.render_method == "linear_soxr"
    assert result.output_samples.dtype == np.float32
    assert result.output_samples.shape[0] == master.shape[0]


def test_soxr_is_valid_stretch_method() -> None:
    from double_ender_sync.alignment.stretch import VALID_STRETCH_METHODS
    assert "soxr" in VALID_STRETCH_METHODS


def test_soxr_raises_for_non_linear_drift() -> None:
    """soxr rendering only supports LinearDrift; generic models raise ValueError."""
    import pytest
    pytest.importorskip("soxr")

    sr = 8000
    local = np.ones(int(sr * 0.25), dtype=np.float32)
    master = np.zeros(int(sr * 1.0), dtype=np.float32)
    track = AudioTrack(path=Path("speaker-a.wav"), name="speaker-a", sample_rate=sr, duration_seconds=0.25, channels=1, original_samples=local, analysis_samples=local, analysis_sample_rate=sr)
    master_track = AudioTrack(path=Path("master.wav"), name="master", sample_rate=sr, duration_seconds=1.0, channels=1, original_samples=master, analysis_samples=master, analysis_sample_rate=sr)

    class SimplePiecewiseDrift:
        model_type = "test_piecewise"
        speaker_track = "speaker-a"
        offset_seconds = 0.0
        stretch_ratio = 1.0

        def map_local_to_master(self, t: float) -> float:
            return t

        def local_rate_at(self, t: float) -> float:
            return 1.0

        def residuals_ms(self, anchors):
            return []

        @property
        def diagnostics(self):
            return None

    with pytest.raises(ValueError, match="'soxr'"):
        apply_global_time_correction(track, master_track, SimplePiecewiseDrift(), stretch_method="soxr")


def test_rubberband_does_not_raise_for_non_linear_drift(monkeypatch) -> None:
    """rubberband should route through timemap path for non-LinearDrift models."""
    import types

    fake_pyrubberband = types.ModuleType("pyrubberband")

    def fake_timemap_stretch(y, sr, time_map):
        n_dst = time_map[-1][1]
        return np.zeros(n_dst, dtype=np.float32)

    fake_pyrubberband.timemap_stretch = fake_timemap_stretch

    original_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "pyrubberband":
            return fake_pyrubberband
        return original_import_module(name)

    monkeypatch.setattr("importlib.import_module", fake_import_module)

    sr = 100
    local = np.ones(sr, dtype=np.float32)
    master = np.zeros(sr * 2, dtype=np.float32)
    track = AudioTrack(path=Path("speaker-a.wav"), name="speaker-a", sample_rate=sr, duration_seconds=1.0, channels=1, original_samples=local, analysis_samples=local, analysis_sample_rate=sr)
    master_track = AudioTrack(path=Path("master.wav"), name="master", sample_rate=sr, duration_seconds=2.0, channels=1, original_samples=master, analysis_samples=master, analysis_sample_rate=sr)

    result = apply_global_time_correction(track, master_track, PiecewiseTestDrift(offset_seconds=0.5), stretch_method="rubberband")
    assert result.render_method == "timemap_rubberband"
    assert result.output_samples.dtype == np.float32
    assert result.output_samples.shape[0] == master.shape[0]


def test_rubberband_timemap_raises_when_pyrubberband_missing_with_nonlinear_drift(monkeypatch) -> None:
    """rubberband + non-linear drift raises RuntimeError when pyrubberband is missing."""
    import pytest

    sr = 100
    local = np.ones(sr, dtype=np.float32)
    master = np.zeros(sr * 2, dtype=np.float32)
    track = AudioTrack(path=Path("speaker-a.wav"), name="speaker-a", sample_rate=sr, duration_seconds=1.0, channels=1, original_samples=local, analysis_samples=local, analysis_sample_rate=sr)
    master_track = AudioTrack(path=Path("master.wav"), name="master", sample_rate=sr, duration_seconds=2.0, channels=1, original_samples=master, analysis_samples=master, analysis_sample_rate=sr)
    original_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "pyrubberband":
            raise ModuleNotFoundError("no module named pyrubberband")
        return original_import_module(name)

    monkeypatch.setattr("importlib.import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="requires pyrubberband"):
        apply_global_time_correction(track, master_track, PiecewiseTestDrift(offset_seconds=0.5), stretch_method="rubberband")


def test_rubberband_timemap_render_method_is_timemap_rubberband() -> None:
    """rubberband + non-LinearDrift produces render_method='timemap_rubberband'."""
    import shutil
    import pytest
    pytest.importorskip("pyrubberband")
    if shutil.which("rubberband") is None:
        pytest.skip("rubberband binary not found on PATH")

    sr = 200
    local = np.ones(sr, dtype=np.float32) * 0.1
    master = np.zeros(int(sr * 2.5), dtype=np.float32)
    track = AudioTrack(path=Path("speaker-a.wav"), name="speaker-a", sample_rate=sr, duration_seconds=1.0, channels=1, original_samples=local, analysis_samples=local, analysis_sample_rate=sr)
    master_track = AudioTrack(path=Path("master.wav"), name="master", sample_rate=sr, duration_seconds=2.5, channels=1, original_samples=master, analysis_samples=master, analysis_sample_rate=sr)

    result = apply_global_time_correction(track, master_track, PiecewiseTestDrift(offset_seconds=0.5), stretch_method="rubberband")
    assert result.render_method == "timemap_rubberband"
    assert result.output_samples.dtype == np.float32
    assert result.output_samples.shape[0] == master.shape[0]
    assert result.unsupported_regions is not None


def test_rubberband_timemap_silences_internal_master_time_gap(monkeypatch) -> None:
    """render_with_rubberband_timemap must zero out unsupported_regions in the output.

    JumpGapTestDrift has an internal master-time gap from 0.5 s to 1.0 s.  The
    rubberband path would otherwise fill this interval with distorted audio
    produced by extreme stretching across the discontinuity.
    """
    import types

    fake_pyrubberband = types.ModuleType("pyrubberband")

    def fake_timemap_stretch(y, sr, time_map):
        # Return non-zero audio for the full dst length so any failure to
        # silence the gap is detectable.
        n_dst = time_map[-1][1]
        return np.ones(n_dst, dtype=np.float32) * 0.9

    fake_pyrubberband.timemap_stretch = fake_timemap_stretch

    original_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "pyrubberband":
            return fake_pyrubberband
        return original_import_module(name)

    monkeypatch.setattr("importlib.import_module", fake_import_module)

    sr = 100
    local = np.ones(sr, dtype=np.float32)
    master = np.zeros(sr * 2, dtype=np.float32)
    track = AudioTrack(path=Path("speaker-a.wav"), name="speaker-a", sample_rate=sr, duration_seconds=1.0, channels=1, original_samples=local, analysis_samples=local, analysis_sample_rate=sr)
    master_track = AudioTrack(path=Path("master.wav"), name="master", sample_rate=sr, duration_seconds=2.0, channels=1, original_samples=master, analysis_samples=master, analysis_sample_rate=sr)

    result = apply_global_time_correction(track, master_track, JumpGapTestDrift(), stretch_method="rubberband")

    assert result.render_method == "timemap_rubberband"
    # Gap from 0.5 s to 1.0 s must be silent (50 samples at sr=100)
    gap_start = int(0.5 * sr)
    gap_end = int(1.0 * sr)
    assert np.allclose(result.output_samples[gap_start:gap_end], 0.0), (
        "unsupported master-time gap must be silenced in timemap rubberband output"
    )
    # The gap must be reported as unsupported
    assert any(
        r.reason == "internal_drift_model_gap"
        for r in result.unsupported_regions
    )


def test_build_rubberband_time_map_identity() -> None:
    """_build_rubberband_time_map with uniform stretch produces correct endpoints."""
    from double_ender_sync.alignment.timeline import _build_rubberband_time_map
    import numpy as np

    sr = 1000
    n_src = sr
    local_times = np.linspace(0.0, 1.0, num=11)
    master_times = np.linspace(0.0, 1.0, num=11)

    time_map = _build_rubberband_time_map(n_src, sr, local_times, master_times)

    assert time_map[0][0] == 0
    assert time_map[-1][0] == n_src
    assert time_map[-1][1] == n_src
    # dst must be strictly monotonically increasing
    dst_values = [d for _, d in time_map]
    assert all(dst_values[i] < dst_values[i + 1] for i in range(len(dst_values) - 1))


def test_build_rubberband_time_map_deduplicates_src(monkeypatch) -> None:
    """_build_rubberband_time_map deduplicates consecutive equal src positions."""
    from double_ender_sync.alignment.timeline import _build_rubberband_time_map
    import numpy as np

    sr = 10
    n_src = 10
    # Very few samples so rounding creates many duplicates
    local_times = np.linspace(0.0, 1.0, num=100)
    master_times = np.linspace(0.0, 1.0, num=100)

    time_map = _build_rubberband_time_map(n_src, sr, local_times, master_times)

    src_values = [s for s, _ in time_map]
    assert len(src_values) == len(set(src_values)), "src values must be unique"
    assert time_map[-1][0] == n_src
