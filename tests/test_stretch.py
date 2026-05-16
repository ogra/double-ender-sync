"""Tests for the Stretcher protocol and its concrete implementations."""

import numpy as np
import pytest

from double_ender_sync.alignment.stretch import (
    AudiostretchyStretcher,
    LibrosaStretcher,
    NumpyInterpolationStretcher,
    RubberbandStretcher,
    SoxrStretcher,
    VALID_STRETCH_METHODS,
    make_stretcher,
)


# ---------------------------------------------------------------------------
# Protocol conformance — duck-typing checks
#
# Stretcher is not @runtime_checkable (Python 3.12+ raises TypeError for
# isinstance() against protocols with non-method members such as name: str),
# so conformance is verified by inspecting the required attributes directly.
# ---------------------------------------------------------------------------


def test_numpy_interpolation_stretcher_satisfies_protocol() -> None:
    s = NumpyInterpolationStretcher()
    assert isinstance(s.name, str)
    assert callable(s.stretch_by_ratio)


def test_librosa_stretcher_satisfies_protocol() -> None:
    s = LibrosaStretcher()
    assert isinstance(s.name, str)
    assert callable(s.stretch_by_ratio)


def test_rubberband_stretcher_satisfies_protocol() -> None:
    s = RubberbandStretcher()
    assert isinstance(s.name, str)
    assert callable(s.stretch_by_ratio)


def test_soxr_stretcher_satisfies_protocol() -> None:
    s = SoxrStretcher()
    assert isinstance(s.name, str)
    assert callable(s.stretch_by_ratio)


def test_numpy_stretcher_name() -> None:
    assert NumpyInterpolationStretcher().name == "resample"


def test_librosa_stretcher_name() -> None:
    assert LibrosaStretcher().name == "pitch_preserving"


def test_rubberband_stretcher_name() -> None:
    assert RubberbandStretcher().name == "rubberband"


def test_soxr_stretcher_name() -> None:
    assert SoxrStretcher().name == "soxr"


# ---------------------------------------------------------------------------
# make_stretcher factory
# ---------------------------------------------------------------------------


def test_make_stretcher_resample_returns_numpy_interpolation_stretcher() -> None:
    s = make_stretcher("resample")
    assert isinstance(s, NumpyInterpolationStretcher)


def test_make_stretcher_pitch_preserving_returns_librosa_stretcher() -> None:
    s = make_stretcher("pitch_preserving")
    assert isinstance(s, LibrosaStretcher)


def test_make_stretcher_rubberband_returns_rubberband_stretcher() -> None:
    s = make_stretcher("rubberband")
    assert isinstance(s, RubberbandStretcher)


def test_make_stretcher_soxr_returns_soxr_stretcher() -> None:
    s = make_stretcher("soxr")
    assert isinstance(s, SoxrStretcher)

def test_make_stretcher_audiostretchy_returns_audiostretchy_stretcher() -> None:
    s = make_stretcher("audiostretchy")
    assert isinstance(s, AudiostretchyStretcher)

def test_make_stretcher_invalid_raises_value_error_mentioning_stretch_method() -> None:
    with pytest.raises(ValueError, match="stretch_method"):
        make_stretcher("invalid_method")


def test_valid_stretch_methods_covers_known_methods() -> None:
    assert "resample" in VALID_STRETCH_METHODS
    assert "pitch_preserving" in VALID_STRETCH_METHODS
    assert "rubberband" in VALID_STRETCH_METHODS
    assert "soxr" in VALID_STRETCH_METHODS
    assert "audiostretchy" in VALID_STRETCH_METHODS

class TestAudiostretchyStretcher:
    def setup_method(self) -> None:
        self.stretcher = AudiostretchyStretcher()

    def test_requires_sample_rate(self) -> None:
        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(ValueError, match="sample_rate"):
            self.stretcher.stretch_by_ratio(samples, 1.0, sample_rate=None)

    def test_stretches_with_audio_stretch_class(self, monkeypatch) -> None:
        class FakeAudioStretch:
            def __init__(self):
                self.samples = None
                self.samplerate = 0
                self.num_channels = 0

            def stretch(self, ratio: float = 1.0):
                n = self.samples.shape[1]
                out_n = max(1, int(round(n * ratio)))
                self.samples = np.full((1, out_n), 0.25, dtype=np.float32)

        class FakeModule:
            AudioStretch = FakeAudioStretch

        monkeypatch.setattr(
            AudiostretchyStretcher,
            "_import_audiostretchy_f32",
            staticmethod(lambda: FakeModule),
        )

        samples = np.ones(200, dtype=np.float32)
        out = self.stretcher.stretch_by_ratio(samples, 1.5, sample_rate=48000)
        assert out.dtype == np.float32
        assert out.shape[0] == round(200 * 1.5)


# ---------------------------------------------------------------------------
# NumpyInterpolationStretcher
# ---------------------------------------------------------------------------


class TestNumpyInterpolationStretcher:
    def setup_method(self) -> None:
        self.stretcher = NumpyInterpolationStretcher()

    def test_empty_input_returns_empty_float32(self) -> None:
        out = self.stretcher.stretch_by_ratio(np.array([], dtype=np.float32), 1.0)
        assert out.dtype == np.float32
        assert out.size == 0

    def test_identity_ratio_preserves_length(self) -> None:
        samples = np.ones(1000, dtype=np.float32)
        out = self.stretcher.stretch_by_ratio(samples, 1.0)
        assert out.shape[0] == samples.shape[0]

    def test_identity_ratio_preserves_values(self) -> None:
        rng = np.random.default_rng(42)
        samples = rng.random(500).astype(np.float32)
        out = self.stretcher.stretch_by_ratio(samples, 1.0)
        assert np.allclose(out, samples, atol=1e-6)

    def test_output_length_expand(self) -> None:
        samples = np.ones(1000, dtype=np.float32)
        ratio = 1.5
        out = self.stretcher.stretch_by_ratio(samples, ratio)
        assert out.shape[0] == round(1000 * ratio)

    def test_output_length_compress(self) -> None:
        samples = np.ones(1000, dtype=np.float32)
        ratio = 0.75
        out = self.stretcher.stretch_by_ratio(samples, ratio)
        assert out.shape[0] == round(1000 * ratio)

    def test_output_is_float32(self) -> None:
        samples = np.ones(200, dtype=np.float32)
        out = self.stretcher.stretch_by_ratio(samples, 1.003)
        assert out.dtype == np.float32

    def test_zero_ratio_raises_value_error(self) -> None:
        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(ValueError, match="stretch_ratio"):
            self.stretcher.stretch_by_ratio(samples, 0.0)

    def test_negative_ratio_raises_value_error(self) -> None:
        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(ValueError, match="stretch_ratio"):
            self.stretcher.stretch_by_ratio(samples, -1.0)

    def test_small_positive_ratio_returns_at_least_one_sample(self) -> None:
        samples = np.ones(10, dtype=np.float32)
        out = self.stretcher.stretch_by_ratio(samples, 0.001)
        assert out.shape[0] >= 1

    def test_constant_signal_preserved_after_stretch(self) -> None:
        samples = np.full(500, 0.5, dtype=np.float32)
        out = self.stretcher.stretch_by_ratio(samples, 1.1)
        assert np.allclose(out, 0.5, atol=1e-6)

    def test_stretch_ratio_close_to_one_length_precision(self) -> None:
        samples = np.ones(48000, dtype=np.float32)
        ratio = 1.003
        out = self.stretcher.stretch_by_ratio(samples, ratio)
        expected_len = round(48000 * ratio)
        assert out.shape[0] == expected_len


# ---------------------------------------------------------------------------
# LibrosaStretcher — unit tests that do not require librosa to be installed
# ---------------------------------------------------------------------------


class TestLibrosaStretcher:
    def setup_method(self) -> None:
        self.stretcher = LibrosaStretcher()

    def test_empty_input_returns_empty_float32(self) -> None:
        out = self.stretcher.stretch_by_ratio(np.array([], dtype=np.float32), 1.0)
        assert out.dtype == np.float32
        assert out.size == 0

    def test_zero_ratio_raises_value_error(self) -> None:
        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(ValueError, match="stretch_ratio"):
            self.stretcher.stretch_by_ratio(samples, 0.0)

    def test_negative_ratio_raises_value_error(self) -> None:
        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(ValueError, match="stretch_ratio"):
            self.stretcher.stretch_by_ratio(samples, -0.5)

    def test_raises_runtime_error_when_librosa_missing(self, monkeypatch) -> None:
        def fake_import(name: str):
            raise ModuleNotFoundError("no module named librosa")

        monkeypatch.setattr("double_ender_sync.alignment.stretch.importlib.import_module", fake_import)

        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(RuntimeError, match="requires librosa"):
            self.stretcher.stretch_by_ratio(samples, 1.0)

    def test_error_message_mentions_install_command(self, monkeypatch) -> None:
        def fake_import(name: str):
            raise ModuleNotFoundError("no module named librosa")

        monkeypatch.setattr("double_ender_sync.alignment.stretch.importlib.import_module", fake_import)

        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(RuntimeError, match="double-ender-sync\\[stretch\\]"):
            self.stretcher.stretch_by_ratio(samples, 1.0)


# ---------------------------------------------------------------------------
# LibrosaStretcher — happy-path tests (skipped when librosa is not installed)
# ---------------------------------------------------------------------------


class TestLibrosaStretcherWithLibrosa:
    """Integration tests that require librosa to be installed.

    These are skipped automatically when the ``[stretch]`` optional extra is
    absent, so they never block CI environments that do not include librosa.
    """

    def setup_method(self) -> None:
        pytest.importorskip("librosa")
        self.stretcher = LibrosaStretcher()

    def test_output_is_float32(self) -> None:
        samples = np.ones(4000, dtype=np.float32)
        out = self.stretcher.stretch_by_ratio(samples, 1.0)
        assert out.dtype == np.float32

    def test_output_length_identity_ratio(self) -> None:
        n = 8000
        samples = np.ones(n, dtype=np.float32)
        out = self.stretcher.stretch_by_ratio(samples, 1.0)
        assert out.shape[0] == n

    def test_output_length_expand(self) -> None:
        n = 8000
        ratio = 1.1
        samples = np.ones(n, dtype=np.float32)
        out = self.stretcher.stretch_by_ratio(samples, ratio)
        assert out.shape[0] == round(n * ratio)

    def test_output_length_compress(self) -> None:
        n = 8000
        ratio = 0.9
        samples = np.ones(n, dtype=np.float32)
        out = self.stretcher.stretch_by_ratio(samples, ratio)
        assert out.shape[0] == round(n * ratio)

    def test_empty_input_returns_empty_without_calling_librosa(self) -> None:
        out = self.stretcher.stretch_by_ratio(np.array([], dtype=np.float32), 1.0)
        assert out.size == 0
        assert out.dtype == np.float32

    def test_does_not_raise_for_valid_audio(self) -> None:
        rng = np.random.default_rng(0)
        samples = rng.random(4000).astype(np.float32)
        out = self.stretcher.stretch_by_ratio(samples, 1.003)
        assert out.size > 0


# ---------------------------------------------------------------------------
# RubberbandStretcher — unit tests (no binary required for error-path tests)
# ---------------------------------------------------------------------------


class TestRubberbandStretcher:
    def setup_method(self) -> None:
        self.stretcher = RubberbandStretcher()

    def test_empty_input_returns_empty_float32(self) -> None:
        out = self.stretcher.stretch_by_ratio(np.array([], dtype=np.float32), 1.0)
        assert out.dtype == np.float32
        assert out.size == 0

    def test_zero_ratio_raises_value_error(self) -> None:
        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(ValueError, match="stretch_ratio"):
            self.stretcher.stretch_by_ratio(samples, 0.0)

    def test_negative_ratio_raises_value_error(self) -> None:
        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(ValueError, match="stretch_ratio"):
            self.stretcher.stretch_by_ratio(samples, -0.5)

    def test_none_sample_rate_raises_value_error(self) -> None:
        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(ValueError, match="sample_rate"):
            self.stretcher.stretch_by_ratio(samples, 1.0, sample_rate=None)

    def test_zero_sample_rate_raises_value_error(self) -> None:
        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(ValueError, match="sample_rate"):
            self.stretcher.stretch_by_ratio(samples, 1.0, sample_rate=0)

    def test_negative_sample_rate_raises_value_error(self) -> None:
        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(ValueError, match="sample_rate"):
            self.stretcher.stretch_by_ratio(samples, 1.0, sample_rate=-48000)

    def test_raises_runtime_error_when_pyrubberband_missing(self, monkeypatch) -> None:
        def fake_import(name: str):
            raise ModuleNotFoundError("no module named pyrubberband")

        monkeypatch.setattr("double_ender_sync.alignment.stretch.importlib.import_module", fake_import)

        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(RuntimeError, match="requires pyrubberband"):
            self.stretcher.stretch_by_ratio(samples, 1.0, sample_rate=48000)

    def test_error_message_mentions_install_command(self, monkeypatch) -> None:
        def fake_import(name: str):
            raise ModuleNotFoundError("no module named pyrubberband")

        monkeypatch.setattr("double_ender_sync.alignment.stretch.importlib.import_module", fake_import)

        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(RuntimeError, match="double-ender-sync\\[stretch\\]"):
            self.stretcher.stretch_by_ratio(samples, 1.0, sample_rate=48000)

    def test_raises_runtime_error_when_rubberband_binary_missing(self, monkeypatch) -> None:
        import types

        fake_pyrubberband = types.ModuleType("pyrubberband")

        def fake_time_stretch(y, sr, rate, rbargs=None):
            raise RuntimeError("Failed to execute rubberband. Please verify that rubberband-cli is installed.")

        fake_pyrubberband.time_stretch = fake_time_stretch

        def fake_import(name: str):
            if name == "pyrubberband":
                return fake_pyrubberband
            import importlib as _importlib
            return _importlib.import_module(name)

        monkeypatch.setattr("double_ender_sync.alignment.stretch.importlib.import_module", fake_import)

        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(RuntimeError, match="pyrubberband failed"):
            self.stretcher.stretch_by_ratio(samples, 1.0, sample_rate=48000)


# ---------------------------------------------------------------------------
# RubberbandStretcher — happy-path tests (skipped when binary is not available)
# ---------------------------------------------------------------------------


class TestRubberbandStretcherWithBinary:
    """Integration tests that require pyrubberband and the ``rubberband`` binary.

    The ``rubberband`` executable is provided by the ``rubberband-cli`` package
    on Debian/Ubuntu or by ``brew install rubberband`` on macOS.  Skipped
    automatically when either is absent so CI environments without the system
    binary are unaffected.
    """

    def setup_method(self) -> None:
        pytest.importorskip("pyrubberband")
        import shutil
        if shutil.which("rubberband") is None:
            pytest.skip("rubberband binary not found on PATH")
        self.stretcher = RubberbandStretcher()

    def test_output_is_float32(self) -> None:
        samples = np.ones(4000, dtype=np.float32)
        out = self.stretcher.stretch_by_ratio(samples, 1.0, sample_rate=16000)
        assert out.dtype == np.float32

    def test_output_length_identity_ratio(self) -> None:
        n = 8000
        samples = np.ones(n, dtype=np.float32)
        out = self.stretcher.stretch_by_ratio(samples, 1.0, sample_rate=16000)
        assert out.shape[0] == n

    def test_output_length_expand(self) -> None:
        n = 8000
        ratio = 1.1
        samples = np.ones(n, dtype=np.float32)
        out = self.stretcher.stretch_by_ratio(samples, ratio, sample_rate=16000)
        assert out.shape[0] == round(n * ratio)

    def test_output_length_compress(self) -> None:
        n = 8000
        ratio = 0.9
        samples = np.ones(n, dtype=np.float32)
        out = self.stretcher.stretch_by_ratio(samples, ratio, sample_rate=16000)
        assert out.shape[0] == round(n * ratio)

    def test_empty_input_returns_empty_without_calling_rubberband(self) -> None:
        out = self.stretcher.stretch_by_ratio(np.array([], dtype=np.float32), 1.0, sample_rate=16000)
        assert out.size == 0
        assert out.dtype == np.float32

    def test_does_not_raise_for_valid_audio(self) -> None:
        rng = np.random.default_rng(42)
        samples = rng.random(4000).astype(np.float32)
        out = self.stretcher.stretch_by_ratio(samples, 1.003, sample_rate=16000)
        assert out.size > 0

    def test_sample_rate_kwarg_is_accepted(self) -> None:
        samples = np.ones(4000, dtype=np.float32)
        out = self.stretcher.stretch_by_ratio(samples, 1.0, sample_rate=48000)
        assert out.dtype == np.float32
        assert out.size > 0


# ---------------------------------------------------------------------------
# SoxrStretcher — unit tests (no soxr required for error-path tests)
# ---------------------------------------------------------------------------


class TestSoxrStretcher:
    def setup_method(self) -> None:
        self.stretcher = SoxrStretcher()

    def test_empty_input_returns_empty_float32(self) -> None:
        out = self.stretcher.stretch_by_ratio(np.array([], dtype=np.float32), 1.0)
        assert out.dtype == np.float32
        assert out.size == 0

    def test_zero_ratio_raises_value_error(self) -> None:
        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(ValueError, match="stretch_ratio"):
            self.stretcher.stretch_by_ratio(samples, 0.0)

    def test_negative_ratio_raises_value_error(self) -> None:
        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(ValueError, match="stretch_ratio"):
            self.stretcher.stretch_by_ratio(samples, -0.5)

    def test_raises_runtime_error_when_soxr_missing(self, monkeypatch) -> None:
        def fake_import(name: str):
            raise ModuleNotFoundError("no module named soxr")

        monkeypatch.setattr("double_ender_sync.alignment.stretch.importlib.import_module", fake_import)

        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(RuntimeError, match="requires the soxr package"):
            self.stretcher.stretch_by_ratio(samples, 1.0)

    def test_error_message_mentions_install_command(self, monkeypatch) -> None:
        def fake_import(name: str):
            raise ModuleNotFoundError("no module named soxr")

        monkeypatch.setattr("double_ender_sync.alignment.stretch.importlib.import_module", fake_import)

        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(RuntimeError, match="double-ender-sync\\[hq-resample\\]"):
            self.stretcher.stretch_by_ratio(samples, 1.0)

    def test_sample_rate_kwarg_is_ignored_but_accepted(self) -> None:
        pytest.importorskip("soxr")
        samples = np.ones(1000, dtype=np.float32)
        out = self.stretcher.stretch_by_ratio(samples, 1.0, sample_rate=48000)
        assert out.dtype == np.float32
        assert out.size > 0


# ---------------------------------------------------------------------------
# SoxrStretcher — happy-path tests (skipped when soxr is not installed)
# ---------------------------------------------------------------------------


class TestSoxrStretcherWithSoxr:
    """Integration tests that require the soxr package.

    Skipped automatically when ``[hq-resample]`` optional extra is absent.
    """

    def setup_method(self) -> None:
        pytest.importorskip("soxr")
        self.stretcher = SoxrStretcher()

    def test_output_is_float32(self) -> None:
        samples = np.ones(4000, dtype=np.float32)
        out = self.stretcher.stretch_by_ratio(samples, 1.0)
        assert out.dtype == np.float32

    def test_output_length_identity_ratio(self) -> None:
        n = 8000
        samples = np.ones(n, dtype=np.float32)
        out = self.stretcher.stretch_by_ratio(samples, 1.0)
        assert out.shape[0] == n

    def test_output_length_expand(self) -> None:
        n = 8000
        ratio = 1.1
        samples = np.ones(n, dtype=np.float32)
        out = self.stretcher.stretch_by_ratio(samples, ratio)
        assert out.shape[0] == round(n * ratio)

    def test_output_length_compress(self) -> None:
        n = 8000
        ratio = 0.9
        samples = np.ones(n, dtype=np.float32)
        out = self.stretcher.stretch_by_ratio(samples, ratio)
        assert out.shape[0] == round(n * ratio)

    def test_output_length_small_ratio(self) -> None:
        n = 48000
        ratio = 1.003
        samples = np.ones(n, dtype=np.float32)
        out = self.stretcher.stretch_by_ratio(samples, ratio)
        assert out.shape[0] == round(n * ratio)

    def test_empty_input_returns_empty_without_calling_soxr(self) -> None:
        out = self.stretcher.stretch_by_ratio(np.array([], dtype=np.float32), 1.0)
        assert out.size == 0
        assert out.dtype == np.float32

    def test_does_not_raise_for_valid_audio(self) -> None:
        rng = np.random.default_rng(0)
        samples = rng.random(4000).astype(np.float32)
        out = self.stretcher.stretch_by_ratio(samples, 1.003)
        assert out.size > 0

    def test_alias_suppression_at_small_stretch_ratio(self) -> None:
        """Spectral energy above Nyquist/2 must be ≤ −60 dBFS after 0.3% stretch.

        Generates a 12 kHz sine at 48 kHz sample rate (well below Nyquist) and
        stretches by 1.003.  With linear np.interp, fold-back energy at
        high frequencies is audible; soxr VHQ should suppress it to ≤ −60 dBFS.
        """
        sample_rate = 48000
        duration_seconds = 1.0
        n = int(sample_rate * duration_seconds)
        t = np.arange(n, dtype=np.float64) / sample_rate
        freq_hz = 12000.0
        samples = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)

        ratio = 1.003
        out = self.stretcher.stretch_by_ratio(samples, ratio)

        spectrum = np.abs(np.fft.rfft(out.astype(np.float64)))
        freqs = np.fft.rfftfreq(len(out), d=1.0 / sample_rate)

        # Alias energy lives well above the signal frequency and below Nyquist.
        alias_mask = freqs > freq_hz * 1.5
        signal_mask = (freqs > freq_hz * 0.9) & (freqs < freq_hz * 1.1)

        signal_energy = float(np.max(spectrum[signal_mask])) if np.any(signal_mask) else 1.0
        alias_energy = float(np.max(spectrum[alias_mask])) if np.any(alias_mask) else 0.0

        alias_db = 20.0 * np.log10(max(alias_energy / signal_energy, 1e-12))
        assert alias_db <= -60.0, f"alias_db={alias_db:.1f} dBFS exceeds −60 dBFS threshold"

    def test_constant_signal_preserved_after_stretch(self) -> None:
        samples = np.full(4000, 0.5, dtype=np.float32)
        out = self.stretcher.stretch_by_ratio(samples, 1.1)
        interior = out[100:-100]
        assert np.allclose(interior, 0.5, atol=1e-4)


# ---------------------------------------------------------------------------
# RubberbandStretcher.stretch_by_timemap — unit tests (no binary required)
# ---------------------------------------------------------------------------


class TestRubberbandStretcherTimemap:
    def setup_method(self) -> None:
        self.stretcher = RubberbandStretcher()

    def test_empty_input_returns_empty_float32(self) -> None:
        out = self.stretcher.stretch_by_timemap(
            np.array([], dtype=np.float32),
            [(0, 0), (0, 0)],
            sample_rate=44100,
        )
        assert out.dtype == np.float32
        assert out.size == 0

    def test_raises_when_pyrubberband_missing(self, monkeypatch) -> None:
        def fake_import(name: str):
            raise ModuleNotFoundError("no module named pyrubberband")

        monkeypatch.setattr("double_ender_sync.alignment.stretch.importlib.import_module", fake_import)

        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(RuntimeError, match="requires pyrubberband"):
            self.stretcher.stretch_by_timemap(samples, [(0, 0), (100, 100)], sample_rate=44100)

    def test_error_message_mentions_install_command(self, monkeypatch) -> None:
        def fake_import(name: str):
            raise ModuleNotFoundError("no module named pyrubberband")

        monkeypatch.setattr("double_ender_sync.alignment.stretch.importlib.import_module", fake_import)

        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(RuntimeError, match="double-ender-sync\\[stretch\\]"):
            self.stretcher.stretch_by_timemap(samples, [(0, 0), (100, 100)], sample_rate=44100)

    def test_invalid_sample_rate_raises_value_error(self) -> None:
        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(ValueError, match="sample_rate"):
            self.stretcher.stretch_by_timemap(samples, [(0, 0), (100, 100)], sample_rate=-1)

    def test_zero_sample_rate_raises_value_error(self) -> None:
        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(ValueError, match="sample_rate"):
            self.stretcher.stretch_by_timemap(samples, [(0, 0), (100, 100)], sample_rate=0)

    def test_raises_when_rubberband_binary_missing(self, monkeypatch) -> None:
        import types

        fake_pyrubberband = types.ModuleType("pyrubberband")

        def fake_timemap_stretch(y, sr, time_map):
            raise RuntimeError("Failed to execute rubberband.")

        fake_pyrubberband.timemap_stretch = fake_timemap_stretch

        def fake_import(name: str):
            if name == "pyrubberband":
                return fake_pyrubberband
            import importlib as _importlib
            return _importlib.import_module(name)

        monkeypatch.setattr("double_ender_sync.alignment.stretch.importlib.import_module", fake_import)

        samples = np.ones(100, dtype=np.float32)
        with pytest.raises(RuntimeError, match="pyrubberband failed"):
            self.stretcher.stretch_by_timemap(samples, [(0, 0), (100, 100)], sample_rate=44100)


# ---------------------------------------------------------------------------
# RubberbandStretcher.stretch_by_timemap — happy-path tests
# ---------------------------------------------------------------------------


class TestRubberbandStretcherTimemapWithBinary:
    """Integration tests requiring pyrubberband and the rubberband binary."""

    def setup_method(self) -> None:
        pytest.importorskip("pyrubberband")
        import shutil
        if shutil.which("rubberband") is None:
            pytest.skip("rubberband binary not found on PATH")
        self.stretcher = RubberbandStretcher()
        self.sr = 8000

    def test_output_is_float32(self) -> None:
        n = self.sr
        samples = np.ones(n, dtype=np.float32)
        out = self.stretcher.stretch_by_timemap(
            samples, [(0, 0), (n, n)], sample_rate=self.sr
        )
        assert out.dtype == np.float32

    def test_identity_timemap_preserves_length(self) -> None:
        n = self.sr
        samples = np.ones(n, dtype=np.float32)
        out = self.stretcher.stretch_by_timemap(
            samples, [(0, 0), (n, n)], sample_rate=self.sr
        )
        assert out.shape[0] == n

    def test_expand_timemap_produces_longer_output(self) -> None:
        n = self.sr
        n_dst = int(n * 1.5)
        samples = np.ones(n, dtype=np.float32)
        out = self.stretcher.stretch_by_timemap(
            samples, [(0, 0), (n, n_dst)], sample_rate=self.sr
        )
        assert out.shape[0] == n_dst

    def test_piecewise_timemap_non_uniform_stretch(self) -> None:
        n = self.sr
        half = n // 2
        n_dst = half + int(half * 1.5)
        time_map = [(0, 0), (half, half), (n, n_dst)]
        samples = np.ones(n, dtype=np.float32)
        out = self.stretcher.stretch_by_timemap(samples, time_map, sample_rate=self.sr)
        assert out.shape[0] == n_dst
        assert out.dtype == np.float32
