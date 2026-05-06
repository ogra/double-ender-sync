import pickle
import re

import numpy as np
import pytest
import types
import sys


class _FakeTorchTensor:
    def __init__(self, arr):
        self.arr = arr

    def unsqueeze(self, dim: int):
        assert dim == 0
        return self


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch):
    fake_torch_module = types.ModuleType("torch")
    fake_torch_module.from_numpy = lambda arr: _FakeTorchTensor(arr)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch_module)
    return fake_torch_module


@pytest.fixture
def _fake_torch_for_pyannote_waveform_input(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch)


from double_ender_sync.analysis.vad import (
    AdaptiveRmsVadStrategy,
    DEFAULT_PYANNOTE_MODEL,
    LEGACY_PYANNOTE_VOICE_ACTIVITY_DETECTION_MODEL,
    MODERN_PYANNOTE_SEGMENTATION_MODEL,
    PyannoteRuntimeCompatibilityError,
    PyannoteVadStrategy,
    _canonicalize_pyannote_model_name,
    _ensure_pyannote_model_runtime_compatible,
    _is_pyannote_community_diarization_model,
    _is_pyannote_segmentation_model,
    _parse_version_prefix,
    RmsVadStrategy,
    WebRtcVadStrategy,
    _looks_like_hf_auth_error,
    _looks_like_torch_weights_only_error,
    _patch_pyannote_hf_download_compat,
    _resolve_huggingface_auth_token,
    build_vad_strategy,
    detect_speech_segments,
)


def test_build_vad_strategy_supports_known_names() -> None:
    assert isinstance(build_vad_strategy("rms"), RmsVadStrategy)
    assert isinstance(build_vad_strategy("adaptive_rms"), AdaptiveRmsVadStrategy)
    assert isinstance(build_vad_strategy("webrtc"), WebRtcVadStrategy)


def test_build_vad_strategy_passes_configured_pyannote_model() -> None:
    strategy = build_vad_strategy("pyannote", pyannote_model=MODERN_PYANNOTE_SEGMENTATION_MODEL)
    assert isinstance(strategy, PyannoteVadStrategy)
    assert strategy.model_name == MODERN_PYANNOTE_SEGMENTATION_MODEL


def test_build_vad_strategy_defaults_to_community_diarization_model() -> None:
    strategy = build_vad_strategy("pyannote")
    assert isinstance(strategy, PyannoteVadStrategy)
    assert strategy.model_name == DEFAULT_PYANNOTE_MODEL


def test_pyannote_segmentation_model_detection() -> None:
    assert _is_pyannote_segmentation_model(MODERN_PYANNOTE_SEGMENTATION_MODEL) is True
    assert _is_pyannote_segmentation_model(LEGACY_PYANNOTE_VOICE_ACTIVITY_DETECTION_MODEL) is False


def test_canonicalize_pyannote_model_name_strips_whitespace() -> None:
    assert _canonicalize_pyannote_model_name("  pyannote/segmentation-3.0  ") == MODERN_PYANNOTE_SEGMENTATION_MODEL
    assert _canonicalize_pyannote_model_name("\tpyannote/segmentation-3.0\n") == MODERN_PYANNOTE_SEGMENTATION_MODEL


def test_canonicalize_pyannote_model_name_normalizes_case() -> None:
    assert _canonicalize_pyannote_model_name("Pyannote/Segmentation-3.0") == MODERN_PYANNOTE_SEGMENTATION_MODEL
    assert _canonicalize_pyannote_model_name("PYANNOTE/SEGMENTATION-3.0") == MODERN_PYANNOTE_SEGMENTATION_MODEL


def test_canonicalize_pyannote_model_name_preserves_unrelated_names() -> None:
    assert _canonicalize_pyannote_model_name(LEGACY_PYANNOTE_VOICE_ACTIVITY_DETECTION_MODEL) == LEGACY_PYANNOTE_VOICE_ACTIVITY_DETECTION_MODEL
    assert _canonicalize_pyannote_model_name("  pyannote/voice-activity-detection  ") == LEGACY_PYANNOTE_VOICE_ACTIVITY_DETECTION_MODEL


def test_is_pyannote_segmentation_model_handles_whitespace_and_case() -> None:
    assert _is_pyannote_segmentation_model("  pyannote/segmentation-3.0  ") is True
    assert _is_pyannote_segmentation_model("Pyannote/Segmentation-3.0") is True
    assert _is_pyannote_segmentation_model("PYANNOTE/SEGMENTATION-3.0") is True


def test_is_pyannote_segmentation_model_other_segmentation_versions() -> None:
    # Other version numbers in the same segmentation family should also be detected.
    assert _is_pyannote_segmentation_model("pyannote/segmentation-3.1") is True
    assert _is_pyannote_segmentation_model("pyannote/segmentation-2.0") is True


def test_pyannote_community_diarization_model_detection() -> None:
    assert _is_pyannote_community_diarization_model(DEFAULT_PYANNOTE_MODEL) is True
    assert _is_pyannote_community_diarization_model("pyannote-community/speaker-diarization-community-1") is True
    assert _is_pyannote_community_diarization_model(MODERN_PYANNOTE_SEGMENTATION_MODEL) is False


def test_parse_version_prefix_handles_release_suffixes() -> None:
    assert _parse_version_prefix("4.0.1") == (4, 0, 1)
    assert _parse_version_prefix("3.3.2.post1") == (3, 3, 2)
    assert _parse_version_prefix("4.0rc1") == (4, 0)
    assert _parse_version_prefix("4") == (4, 0)
    assert _parse_version_prefix("4rc1") == (4, 0)


def test_pyannote_community_model_rejects_old_pyannote_audio_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_version(distribution_name: str) -> str:
        assert distribution_name == "pyannote.audio"
        return "3.3.2"

    monkeypatch.setattr("double_ender_sync.analysis.vad.importlib_metadata.version", _fake_version)

    with pytest.raises(PyannoteRuntimeCompatibilityError, match=r"pyannote\.audio >= 4\.0.*3\.3\.2"):
        _ensure_pyannote_model_runtime_compatible(DEFAULT_PYANNOTE_MODEL)


def test_pyannote_community_model_accepts_pyannote_audio_four(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_version(distribution_name: str) -> str:
        assert distribution_name == "pyannote.audio"
        return "4.0.0"

    monkeypatch.setattr("double_ender_sync.analysis.vad.importlib_metadata.version", _fake_version)

    _ensure_pyannote_model_runtime_compatible(DEFAULT_PYANNOTE_MODEL)


def test_pyannote_community_model_accepts_bare_pyannote_audio_four(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_version(distribution_name: str) -> str:
        assert distribution_name == "pyannote.audio"
        return "4rc1"

    monkeypatch.setattr("double_ender_sync.analysis.vad.importlib_metadata.version", _fake_version)

    _ensure_pyannote_model_runtime_compatible(DEFAULT_PYANNOTE_MODEL)


def test_pyannote_strategy_preserves_runtime_compatibility_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_version(distribution_name: str) -> str:
        assert distribution_name == "pyannote.audio"
        return "3.3.2"

    monkeypatch.setattr("double_ender_sync.analysis.vad.importlib_metadata.version", _fake_version)

    samples = np.zeros(16000, dtype=np.float32)
    with pytest.raises(PyannoteRuntimeCompatibilityError) as exc_info:
        detect_speech_segments(samples, sample_rate=16000, vad_strategy=build_vad_strategy("pyannote"))

    message = str(exc_info.value)
    assert "requires pyannote.audio >= 4.0" in message
    assert "3.3.2" in message
    assert "failed to initialize or run" not in message


def test_pyannote_runtime_compatibility_allows_non_community_models(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raising_version(distribution_name: str) -> str:
        raise AssertionError("version should not be checked for segmentation models")

    monkeypatch.setattr("double_ender_sync.analysis.vad.importlib_metadata.version", _raising_version)

    _ensure_pyannote_model_runtime_compatible(MODERN_PYANNOTE_SEGMENTATION_MODEL)


def test_adaptive_rms_detects_speech_above_noise_floor() -> None:
    sample_rate = 16000
    noise = np.full(int(sample_rate * 0.5), 0.01, dtype=np.float32)
    speech = np.full(int(sample_rate * 1.0), 0.08, dtype=np.float32)
    samples = np.concatenate([noise, speech, noise])

    segments = detect_speech_segments(samples, sample_rate=sample_rate, vad_strategy=AdaptiveRmsVadStrategy())

    assert segments
    assert segments[0].start <= 0.6
    assert segments[0].end >= 1.4
    assert segments[0].confidence > 0.2


def test_pyannote_strategy_requires_runtime_or_raises_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = np.zeros(16000, dtype=np.float32)

    import builtins

    real_import = builtins.__import__

    def _raising_import(name, *args, **kwargs):
        if name == "pyannote.audio":
            raise ModuleNotFoundError("pyannote.audio")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raising_import)

    with pytest.raises(RuntimeError, match="requires optional extras"):
        detect_speech_segments(samples, sample_rate=16000, vad_strategy=build_vad_strategy("pyannote"))


def test_webrtc_strategy_requires_runtime_or_raises_missing_extra() -> None:
    samples = np.zeros(16000, dtype=np.float32)
    strategy = build_vad_strategy("webrtc")
    try:
        segments = detect_speech_segments(samples, sample_rate=16000, vad_strategy=strategy)
    except RuntimeError as exc:
        assert "requires optional extras" in str(exc)
    else:
        assert segments == []


def test_webrtc_strategy_honors_hop_ms(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("webrtcvad")
    call_count = {"count": 0}

    class _FakeVad:
        def __init__(self, mode: int):
            assert mode == 2

        def is_speech(self, frame_bytes: bytes, sample_rate: int) -> bool:
            assert sample_rate == 16000
            assert len(frame_bytes) == int(16000 * 0.03) * 2
            call_count["count"] += 1
            return True

    fake_module.Vad = _FakeVad  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "webrtcvad", fake_module)

    samples = np.ones(int(16000 * 0.1), dtype=np.float32)
    strategy = build_vad_strategy("webrtc")
    segments = detect_speech_segments(samples, sample_rate=16000, frame_ms=30.0, hop_ms=10.0, min_speech_ms=0.0, vad_strategy=strategy)

    assert call_count["count"] == 8
    assert len(segments) == 1
    assert segments[0].start == 0.0
    assert segments[0].end == pytest.approx(0.08)


def test_silero_strategy_requires_runtime_or_raises_missing_extra() -> None:
    samples = np.zeros(16000, dtype=np.float32)
    strategy = build_vad_strategy("silero")
    try:
        segments = detect_speech_segments(samples, sample_rate=16000, vad_strategy=strategy)
    except RuntimeError as exc:
        assert "requires optional extras" in str(exc)
    else:
        assert segments == []


def test_silero_strategy_converts_timestamps_to_speech_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("silero_vad")

    def _load_silero_vad(*, onnx: bool):
        assert onnx is True
        return object()

    def _get_speech_timestamps(samples, model, sampling_rate: int, threshold: float, min_speech_duration_ms: int):
        assert sampling_rate == 16000
        assert threshold == 0.5
        assert min_speech_duration_ms == 300
        return [{"start": 1600, "end": 4800}, {"start": 5000, "end": 5000}]

    fake_module.load_silero_vad = _load_silero_vad  # type: ignore[attr-defined]
    fake_module.get_speech_timestamps = _get_speech_timestamps  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "silero_vad", fake_module)

    samples = np.zeros(16000, dtype=np.float32)
    segments = detect_speech_segments(samples, sample_rate=16000, vad_strategy=build_vad_strategy("silero"))

    assert len(segments) == 1
    assert segments[0].start == 0.1
    assert segments[0].end == 0.3
    assert segments[0].confidence == 0.5


def test_rms_strategy_keeps_configured_threshold() -> None:
    sample_rate = 16000
    quiet = np.full(sample_rate, 0.015, dtype=np.float32)
    strategy = RmsVadStrategy(energy_threshold=0.01)

    segments = detect_speech_segments(quiet, sample_rate=sample_rate, vad_strategy=strategy)

    assert segments
    assert strategy.energy_threshold == 0.01


def test_default_detect_speech_segments_honors_energy_threshold() -> None:
    sample_rate = 16000
    quiet = np.full(sample_rate, 0.015, dtype=np.float32)

    segments_low = detect_speech_segments(quiet, sample_rate=sample_rate, energy_threshold=0.01)
    segments_high = detect_speech_segments(quiet, sample_rate=sample_rate, energy_threshold=0.02)

    assert segments_low
    assert not segments_high


def test_pyannote_strategy_converts_timeline_to_segments(monkeypatch: pytest.MonkeyPatch, _fake_torch_for_pyannote_waveform_input: None) -> None:
    fake_module = types.ModuleType("pyannote.audio")

    class _Segment:
        def __init__(self, start: float, end: float):
            self.start = start
            self.end = end

    class _Timeline:
        def support(self):
            return [_Segment(0.2, 0.8), _Segment(0.9, 1.0)]

    class _Result:
        def get_timeline(self):
            return _Timeline()

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            assert model_name == DEFAULT_PYANNOTE_MODEL
            return cls()

        def __call__(self, file_or_waveform):
            assert isinstance(file_or_waveform, dict)
            assert "waveform" in file_or_waveform
            assert file_or_waveform["sample_rate"] == 16000
            return _Result()

    fake_module.Pipeline = _Pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_module)

    samples = np.zeros(16000, dtype=np.float32)
    segments = detect_speech_segments(
        samples,
        sample_rate=16000,
        min_speech_ms=200.0,
        vad_strategy=PyannoteVadStrategy(),
    )

    assert len(segments) == 1
    assert segments[0].start == pytest.approx(0.2)
    assert segments[0].end == pytest.approx(0.8)


def test_pyannote_strategy_converts_community_diarization_output_to_speech_segments(monkeypatch: pytest.MonkeyPatch, _fake_torch_for_pyannote_waveform_input: None) -> None:
    fake_module = types.ModuleType("pyannote.audio")

    class _Segment:
        def __init__(self, start: float, end: float):
            self.start = start
            self.end = end

    class _Timeline:
        def support(self):
            return [_Segment(0.1, 0.4), _Segment(0.5, 0.9)]

    class _SpeakerDiarization:
        def get_timeline(self):
            return _Timeline()

    class _Result:
        speaker_diarization = _SpeakerDiarization()

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            assert model_name == DEFAULT_PYANNOTE_MODEL
            return cls()

        def __call__(self, file_or_waveform):
            return _Result()

    fake_module.Pipeline = _Pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_module)

    samples = np.zeros(16000, dtype=np.float32)
    segments = detect_speech_segments(
        samples,
        sample_rate=16000,
        min_speech_ms=200.0,
        vad_strategy=PyannoteVadStrategy(),
    )

    assert [(segment.start, segment.end) for segment in segments] == [(0.1, 0.4), (0.5, 0.9)]


def test_pyannote_strategy_forwards_auth_token(monkeypatch: pytest.MonkeyPatch, _fake_torch_for_pyannote_waveform_input: None) -> None:
    fake_module = types.ModuleType("pyannote.audio")

    class _Segment:
        def __init__(self, start: float, end: float):
            self.start = start
            self.end = end

    class _Timeline:
        def support(self):
            return [_Segment(0.2, 0.8)]

    class _Result:
        def get_timeline(self):
            return _Timeline()

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            assert model_name == DEFAULT_PYANNOTE_MODEL
            assert kwargs == {"token": "hf_test_token"}
            return cls()

        def __call__(self, file_or_waveform):
            return _Result()

    fake_module.Pipeline = _Pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_module)

    samples = np.zeros(16000, dtype=np.float32)
    segments = detect_speech_segments(
        samples,
        sample_rate=16000,
        min_speech_ms=200.0,
        vad_strategy=PyannoteVadStrategy(auth_token="hf_test_token"),
    )

    assert len(segments) == 1
    assert segments[0].start == pytest.approx(0.2)
    assert segments[0].end == pytest.approx(0.8)


def test_pyannote_strategy_retries_legacy_auth_kwarg(monkeypatch: pytest.MonkeyPatch, _fake_torch_for_pyannote_waveform_input: None) -> None:
    fake_module = types.ModuleType("pyannote.audio")
    calls = {"count": 0}

    class _Segment:
        def __init__(self, start: float, end: float):
            self.start = start
            self.end = end

    class _Timeline:
        def support(self):
            return [_Segment(0.2, 0.8)]

    class _Result:
        def get_timeline(self):
            return _Timeline()

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                assert kwargs == {"token": "hf_test_token"}
                raise TypeError("from_pretrained() got an unexpected keyword argument 'token'")
            assert kwargs == {"use_auth_token": "hf_test_token"}
            return cls()

        def __call__(self, file_or_waveform):
            return _Result()

    fake_module.Pipeline = _Pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_module)

    samples = np.zeros(16000, dtype=np.float32)
    segments = detect_speech_segments(
        samples,
        sample_rate=16000,
        min_speech_ms=200.0,
        vad_strategy=PyannoteVadStrategy(auth_token="hf_test_token"),
    )

    assert calls["count"] == 2
    assert len(segments) == 1

def test_pyannote_strategy_patches_hf_download_after_legacy_kwarg_path(monkeypatch: pytest.MonkeyPatch, _fake_torch_for_pyannote_waveform_input: None) -> None:
    fake_module = types.ModuleType("pyannote.audio")
    fake_core_module = types.ModuleType("pyannote.audio.core")
    fake_pipeline_module = types.ModuleType("pyannote.audio.core.pipeline")
    fake_model_module = types.ModuleType("pyannote.audio.core.model")
    calls = {"count": 0, "pipeline_download_kwargs": None, "model_download_kwargs": None}

    def _pipeline_hf_hub_download(*args, **kwargs):
        calls["pipeline_download_kwargs"] = kwargs
        return "ok"

    def _model_hf_hub_download(*args, **kwargs):
        calls["model_download_kwargs"] = kwargs
        return "ok"

    fake_pipeline_module.hf_hub_download = _pipeline_hf_hub_download  # type: ignore[attr-defined]
    fake_model_module.hf_hub_download = _model_hf_hub_download  # type: ignore[attr-defined]

    class _Segment:
        def __init__(self, start: float, end: float):
            self.start = start
            self.end = end

    class _Timeline:
        def support(self):
            return [_Segment(0.2, 0.8)]

    class _Result:
        def get_timeline(self):
            return _Timeline()

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                assert kwargs == {"token": "hf_test_token"}
                raise TypeError("from_pretrained() got an unexpected keyword argument 'token'")
            if calls["count"] == 2:
                assert kwargs == {"use_auth_token": "hf_test_token"}
                raise TypeError("hf_hub_download() got an unexpected keyword argument 'use_auth_token'")
            fake_pipeline_module.hf_hub_download(repo_id=model_name, use_auth_token=kwargs.get("use_auth_token"))
            fake_model_module.hf_hub_download(repo_id="segmentation", use_auth_token=kwargs.get("use_auth_token"))
            return cls()

        def __call__(self, file_or_waveform):
            return _Result()

    fake_module.Pipeline = _Pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core", fake_core_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core.pipeline", fake_pipeline_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core.model", fake_model_module)

    samples = np.zeros(16000, dtype=np.float32)
    segments = detect_speech_segments(
        samples,
        sample_rate=16000,
        min_speech_ms=200.0,
        vad_strategy=PyannoteVadStrategy(auth_token="hf_test_token"),
    )

    assert calls["count"] == 3
    assert calls["pipeline_download_kwargs"] == {"repo_id": DEFAULT_PYANNOTE_MODEL, "token": "hf_test_token"}
    assert calls["model_download_kwargs"] == {"repo_id": "segmentation", "token": "hf_test_token"}
    assert len(segments) == 1


def test_patch_pyannote_hf_download_compat_aliases_use_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pyannote_module = types.ModuleType("pyannote.audio")
    fake_core_module = types.ModuleType("pyannote.audio.core")
    fake_pipeline_module = types.ModuleType("pyannote.audio.core.pipeline")
    fake_model_module = types.ModuleType("pyannote.audio.core.model")
    calls: dict[str, object] = {"kwargs": None}

    def _hf_hub_download(*args, **kwargs):
        calls["kwargs"] = kwargs
        return "ok"

    fake_pipeline_module.hf_hub_download = _hf_hub_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core", fake_core_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core.pipeline", fake_pipeline_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core.model", fake_model_module)

    _patch_pyannote_hf_download_compat()
    result = fake_pipeline_module.hf_hub_download(repo_id="model", use_auth_token="abc")

    assert result == "ok"
    assert calls["kwargs"] == {"repo_id": "model", "token": "abc"}


def test_patch_pyannote_hf_download_compat_aliases_model_module_use_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pyannote_module = types.ModuleType("pyannote.audio")
    fake_core_module = types.ModuleType("pyannote.audio.core")
    fake_pipeline_module = types.ModuleType("pyannote.audio.core.pipeline")
    fake_model_module = types.ModuleType("pyannote.audio.core.model")
    calls: dict[str, object] = {"kwargs": None}

    def _hf_hub_download(*args, **kwargs):
        calls["kwargs"] = kwargs
        return "model-ok"

    fake_model_module.hf_hub_download = _hf_hub_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core", fake_core_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core.pipeline", fake_pipeline_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core.model", fake_model_module)

    _patch_pyannote_hf_download_compat()
    result = fake_model_module.hf_hub_download(repo_id="segmentation", use_auth_token="abc")

    assert result == "model-ok"
    assert calls["kwargs"] == {"repo_id": "segmentation", "token": "abc"}


def test_pyannote_strategy_uses_waveform_input_without_file_decode(monkeypatch: pytest.MonkeyPatch, _fake_torch_for_pyannote_waveform_input: None) -> None:
    fake_pyannote_module = types.ModuleType("pyannote.audio")
    calls = {"count": 0}

    class _Segment:
        def __init__(self, start: float, end: float):
            self.start = start
            self.end = end

    class _Timeline:
        def support(self):
            return [_Segment(0.1, 0.6)]

    class _Result:
        def get_timeline(self):
            return _Timeline()

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            return cls()

        def __call__(self, file_or_waveform):
            calls["count"] += 1
            assert isinstance(file_or_waveform, dict)
            assert "waveform" in file_or_waveform
            assert file_or_waveform["sample_rate"] == 16000
            return _Result()

    fake_pyannote_module.Pipeline = _Pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote_module)

    samples = np.zeros(16000, dtype=np.float32)
    segments = detect_speech_segments(
        samples,
        sample_rate=16000,
        min_speech_ms=200.0,
        vad_strategy=PyannoteVadStrategy(),
    )

    assert len(segments) == 1
    assert segments[0].start == pytest.approx(0.1)
    assert segments[0].end == pytest.approx(0.6)


def test_pyannote_strategy_surfaces_optional_extra_hint_when_torch_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pyannote_module = types.ModuleType("pyannote.audio")

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            return cls()

    fake_pyannote_module.Pipeline = _Pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote_module)
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    import builtins

    real_import = builtins.__import__

    def _raising_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raising_import)

    samples = np.zeros(16000, dtype=np.float32)
    with pytest.raises(RuntimeError, match="vad-pyannote"):
        detect_speech_segments(samples, sample_rate=16000, vad_strategy=PyannoteVadStrategy())


def test_pyannote_strategy_retries_torch_weights_only_checkpoint_load(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pyannote_module = types.ModuleType("pyannote.audio")
    fake_torch_module = types.ModuleType("torch")
    calls = {"torch_load_kwargs": []}

    class _Segment:
        def __init__(self, start: float, end: float):
            self.start = start
            self.end = end

    class _Timeline:
        def support(self):
            return [_Segment(0.2, 0.8)]

    class _Result:
        def get_timeline(self):
            return _Timeline()

    def _fake_torch_load(*args, **kwargs):
        calls["torch_load_kwargs"].append(kwargs.copy())
        if kwargs.get("weights_only") is False:
            return {"checkpoint": "ok"}
        raise pickle.UnpicklingError(
            "Weights only load failed. WeightsUnpickler error: Unsupported global: "
            "GLOBAL pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint"
        )

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            fake_torch_module.load("checkpoint.ckpt")
            return cls()

        def __call__(self, file_or_waveform):
            return _Result()

    fake_torch_module.load = _fake_torch_load  # type: ignore[attr-defined]
    fake_torch_module.from_numpy = lambda arr: _FakeTorchTensor(arr)  # type: ignore[attr-defined]
    fake_pyannote_module.Pipeline = _Pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote_module)

    samples = np.zeros(16000, dtype=np.float32)
    segments = detect_speech_segments(
        samples,
        sample_rate=16000,
        min_speech_ms=200.0,
        vad_strategy=PyannoteVadStrategy(),
    )

    assert calls["torch_load_kwargs"] == [{}, {"weights_only": False}]
    assert len(segments) == 1
    assert fake_torch_module.load is _fake_torch_load


def test_torch_load_weights_only_false_context_overrides_explicit_true(monkeypatch: pytest.MonkeyPatch) -> None:
    from double_ender_sync.analysis.vad import _torch_load_weights_only_false_context

    fake_torch_module = types.ModuleType("torch")
    calls: list[dict] = []

    def _fake_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return kwargs.copy()

    fake_torch_module.load = _fake_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch_module)

    with _torch_load_weights_only_false_context():
        result = fake_torch_module.load("checkpoint.ckpt", weights_only=True)

    assert result == {"weights_only": False}
    assert calls == [{"weights_only": False}]


def test_torch_load_weights_only_false_context_does_not_affect_other_threads() -> None:
    """Verify the thread-local flag scopes weights_only=False to the calling thread only."""
    import threading
    from double_ender_sync.analysis.vad import _torch_load_weights_only_false_context

    fake_torch_module = types.ModuleType("torch")

    def _fake_load(*args, **kwargs):
        return kwargs.copy()

    fake_torch_module.load = _fake_load  # type: ignore[attr-defined]

    barrier = threading.Barrier(2)
    result_holder: dict = {}

    def other_thread_fn():
        barrier.wait()  # wait until the context is active in the main thread
        result_holder["kwargs"] = fake_torch_module.load("ckpt")
        barrier.wait()  # signal that the call is done

    t = threading.Thread(target=other_thread_fn)
    t.start()

    original_modules_torch = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch_module
    try:
        with _torch_load_weights_only_false_context():
            barrier.wait()  # let the other thread call torch.load
            barrier.wait()  # wait for the other thread to finish
    finally:
        if original_modules_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original_modules_torch

    t.join()

    # The other thread must NOT have received weights_only=False
    assert result_holder["kwargs"] == {}


def test_looks_like_torch_weights_only_error_detects_checkpoint_error() -> None:
    exc = pickle.UnpicklingError("Weights only load failed. Unsupported global: ModelCheckpoint")
    assert _looks_like_torch_weights_only_error(exc) is True
    assert _looks_like_torch_weights_only_error(RuntimeError("some unrelated runtime error")) is False
    # A non-UnpicklingError with the same message must NOT trigger the unsafe retry.
    assert _looks_like_torch_weights_only_error(RuntimeError("Weights only load failed. WeightsUnpickler error")) is False
    # A wrapped UnpicklingError (chained via __cause__) must still be detected.
    inner = pickle.UnpicklingError("Weights only load failed. WeightsUnpickler error")
    outer = RuntimeError("checkpoint load error")
    outer.__cause__ = inner
    assert _looks_like_torch_weights_only_error(outer) is True


def test_resolve_huggingface_auth_token_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    assert _resolve_huggingface_auth_token(None) is None

    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hf_env_token")
    assert _resolve_huggingface_auth_token(None) == "hf_env_token"
    assert _resolve_huggingface_auth_token("hf_arg_token") == "hf_arg_token"
    # An empty string is an explicitly-provided value; it must NOT fall back to env vars.
    assert _resolve_huggingface_auth_token("") == ""


def test_looks_like_hf_auth_error_detects_403_message() -> None:
    exc = RuntimeError("HTTP 403 Forbidden: Could not download model from huggingface")
    assert _looks_like_hf_auth_error(exc) is True
    assert _looks_like_hf_auth_error(RuntimeError("some unrelated runtime error")) is False
    # Overly-broad markers that were removed must not trigger auth-error detection.
    assert _looks_like_hf_auth_error(RuntimeError("could not download model")) is False
    assert _looks_like_hf_auth_error(RuntimeError("use_auth_token is deprecated")) is False
    assert _looks_like_hf_auth_error(RuntimeError("huggingface network error")) is False
    # Filesystem/cache "permission denied" must not be misclassified as an HF auth error.
    assert _looks_like_hf_auth_error(RuntimeError("permission denied: /tmp/cache/model.bin")) is False


def test_pyannote_strategy_detect_forwards_env_token(monkeypatch: pytest.MonkeyPatch, _fake_torch_for_pyannote_waveform_input: None) -> None:
    """detect() should pass HF_TOKEN env var to Pipeline.from_pretrained when no explicit token is set."""
    fake_module = types.ModuleType("pyannote.audio")
    token_captured: dict = {}

    class _Segment:
        def __init__(self, start: float, end: float) -> None:
            self.start = start
            self.end = end

    class _Timeline:
        def support(self):
            return [_Segment(0.1, 0.5)]

    class _Result:
        def get_timeline(self):
            return _Timeline()

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            token_captured["token"] = kwargs.get("token")
            return cls()

        def __call__(self, file_or_waveform):
            return _Result()

    fake_module.Pipeline = _Pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_module)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_env_test_token")

    samples = np.zeros(16000, dtype=np.float32)
    detect_speech_segments(samples, sample_rate=16000, min_speech_ms=0.0, vad_strategy=PyannoteVadStrategy())

    assert token_captured["token"] == "hf_env_test_token"


def test_pyannote_strategy_remaps_hf_auth_error_to_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 401/403/gated failure from Pipeline.from_pretrained should produce a clear user-facing RuntimeError."""
    fake_module = types.ModuleType("pyannote.audio")

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            raise RuntimeError("403 Forbidden: Access to gated repo pyannote/speaker-diarization-community-1")

    fake_module.Pipeline = _Pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_module)

    samples = np.zeros(16000, dtype=np.float32)
    with pytest.raises(RuntimeError, match=r"HF_TOKEN.*HUGGINGFACE_HUB_TOKEN|pyannote/speaker-diarization-community-1"):
        detect_speech_segments(samples, sample_rate=16000, vad_strategy=PyannoteVadStrategy())


def test_pyannote_segmentation_model_path_builds_voice_activity_detection(monkeypatch: pytest.MonkeyPatch, _fake_torch_for_pyannote_waveform_input: None) -> None:
    fake_pyannote_root = types.ModuleType("pyannote")
    fake_audio_module = types.ModuleType("pyannote.audio")
    fake_pipelines_module = types.ModuleType("pyannote.audio.pipelines")
    calls: dict[str, object] = {}

    class _Segment:
        def __init__(self, start: float, end: float) -> None:
            self.start = start
            self.end = end

    class _Timeline:
        def support(self):
            return [_Segment(0.2, 0.8)]

    class _Result:
        def get_timeline(self):
            return _Timeline()

    class _Model:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            calls["model_name"] = model_name
            calls["model_kwargs"] = kwargs
            return "segmentation-model"

    class _VoiceActivityDetection:
        def __init__(self, segmentation):
            calls["segmentation"] = segmentation

        def instantiate(self, hyperparameters):
            calls["hyperparameters"] = hyperparameters
            return self

        def __call__(self, file_or_waveform):
            assert isinstance(file_or_waveform, dict)
            assert "waveform" in file_or_waveform
            assert file_or_waveform["sample_rate"] == 16000
            return _Result()

    fake_audio_module.Model = _Model  # type: ignore[attr-defined]
    fake_pipelines_module.VoiceActivityDetection = _VoiceActivityDetection  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote", fake_pyannote_root)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio.pipelines", fake_pipelines_module)
    monkeypatch.setenv("HF_TOKEN", "hf_token")

    samples = np.zeros(16000, dtype=np.float32)
    segments = detect_speech_segments(
        samples,
        sample_rate=16000,
        min_speech_ms=200.0,
        vad_strategy=build_vad_strategy("pyannote", pyannote_model=MODERN_PYANNOTE_SEGMENTATION_MODEL),
    )

    assert calls["model_name"] == MODERN_PYANNOTE_SEGMENTATION_MODEL
    assert calls["model_kwargs"] == {"token": "hf_token"}
    assert calls["segmentation"] == "segmentation-model"
    assert calls["hyperparameters"] == {"min_duration_on": 0.1, "min_duration_off": 0.1}
    assert len(segments) == 1


def test_pyannote_segmentation_model_error_names_selected_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pyannote_root = types.ModuleType("pyannote")
    fake_audio_module = types.ModuleType("pyannote.audio")
    fake_pipelines_module = types.ModuleType("pyannote.audio.pipelines")

    class _Model:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            raise RuntimeError("boom")

    class _VoiceActivityDetection:
        pass

    fake_audio_module.Model = _Model  # type: ignore[attr-defined]
    fake_pipelines_module.VoiceActivityDetection = _VoiceActivityDetection  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote", fake_pyannote_root)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio.pipelines", fake_pipelines_module)

    samples = np.zeros(16000, dtype=np.float32)
    with pytest.raises(RuntimeError, match=re.escape(MODERN_PYANNOTE_SEGMENTATION_MODEL)):
        detect_speech_segments(
            samples,
            sample_rate=16000,
            vad_strategy=PyannoteVadStrategy(model_name=MODERN_PYANNOTE_SEGMENTATION_MODEL),
        )
