from __future__ import annotations

import contextlib
import logging
import os
import pickle
from importlib import metadata as importlib_metadata
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from packaging.version import InvalidVersion as _InvalidVersion
from packaging.version import Version as _PackagingVersion

LOGGER = logging.getLogger(__name__)
DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
LEGACY_PYANNOTE_VOICE_ACTIVITY_DETECTION_MODEL = "pyannote/voice-activity-detection"
MODERN_PYANNOTE_SEGMENTATION_MODEL = "pyannote/segmentation-3.0"
MODERN_PYANNOTE_SEGMENTATION_MODEL_PREFIX = MODERN_PYANNOTE_SEGMENTATION_MODEL.rsplit("-", 1)[0]
PYANNOTE_SEGMENTATION_MODEL_PREFIXES = (MODERN_PYANNOTE_SEGMENTATION_MODEL_PREFIX,)
PYANNOTE_VAD_MIN_DURATION_ON_SECONDS = 0.10
PYANNOTE_VAD_MIN_DURATION_OFF_SECONDS = 0.10
PYANNOTE_AUDIO_COMMUNITY_DIARIZATION_MIN_VERSION = (4, 0)
PYANNOTE_AUDIO_COMMUNITY_DIARIZATION_MIN_MAJOR = 4


class PyannoteRuntimeCompatibilityError(RuntimeError):
    """Raised when the installed pyannote.audio is incompatible with the selected pipeline."""


@dataclass
class SpeechSegment:
    start: float
    end: float
    confidence: float


class VadStrategy(Protocol):
    """Interface for pluggable VAD implementations."""

    def detect(
        self,
        samples: np.ndarray,
        sample_rate: int,
        frame_ms: float = 30.0,
        hop_ms: float = 10.0,
        min_speech_ms: float = 300.0,
    ) -> list[SpeechSegment]:
        ...


@dataclass
class RmsVadStrategy:
    energy_threshold: float = 0.02

    def detect(
        self,
        samples: np.ndarray,
        sample_rate: int,
        frame_ms: float = 30.0,
        hop_ms: float = 10.0,
        min_speech_ms: float = 300.0,
    ) -> list[SpeechSegment]:
        return _detect_speech_segments_from_energy(
            samples=samples,
            sample_rate=sample_rate,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            min_speech_ms=min_speech_ms,
            energy_threshold=self.energy_threshold,
        )


@dataclass
class AdaptiveRmsVadStrategy:
    noise_floor_percentile: float = 20.0
    mad_scale: float = 1.0
    min_energy_floor: float = 1e-4

    def detect(
        self,
        samples: np.ndarray,
        sample_rate: int,
        frame_ms: float = 30.0,
        hop_ms: float = 10.0,
        min_speech_ms: float = 300.0,
    ) -> list[SpeechSegment]:
        frame_size = max(1, int(sample_rate * frame_ms / 1000.0))
        hop_size = max(1, int(sample_rate * hop_ms / 1000.0))
        energies, starts = _compute_frame_energies(samples, frame_size, hop_size)
        if not energies:
            return []

        energy_array = np.asarray(energies, dtype=np.float32)
        noise_floor = float(np.percentile(energy_array, self.noise_floor_percentile))
        noise_reference = energy_array[energy_array <= noise_floor]
        if noise_reference.size == 0:
            noise_reference = energy_array
        reference_median = float(np.median(noise_reference))
        deviations = np.abs(noise_reference - reference_median)
        mad = float(np.median(deviations))
        if mad < 1e-3:
            adaptive_threshold = max(self.min_energy_floor, noise_floor)
        else:
            adaptive_threshold = max(self.min_energy_floor, noise_floor + (self.mad_scale * mad))

        return _segments_from_flags(
            speech_flags=(energy >= adaptive_threshold for energy in energies),
            starts=starts,
            hop_size=hop_size,
            sample_rate=sample_rate,
            energies=energies,
            min_speech_ms=min_speech_ms,
        )


class SileroVadStrategy:
    """Silero VAD strategy backed by the optional ``silero-vad`` package."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def detect(
        self,
        samples: np.ndarray,
        sample_rate: int,
        frame_ms: float = 30.0,
        hop_ms: float = 10.0,
        min_speech_ms: float = 300.0,
    ) -> list[SpeechSegment]:
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad
        except Exception as exc:  # pragma: no cover - dependent on optional install
            raise RuntimeError(
                "Silero VAD requires optional extras: pip install double-ender-sync[vad-ml]"
            ) from exc

        if sample_rate <= 0:
            return []
        mono = np.asarray(samples, dtype=np.float32)
        if mono.size == 0:
            return []

        model = load_silero_vad(onnx=True)
        timestamps = get_speech_timestamps(
            mono,
            model,
            sampling_rate=sample_rate,
            threshold=float(self.threshold),
            min_speech_duration_ms=int(min_speech_ms),
        )
        segments: list[SpeechSegment] = []
        for ts in timestamps:
            start_sample = int(ts.get("start", 0))
            end_sample = int(ts.get("end", 0))
            if end_sample <= start_sample:
                continue
            start = start_sample / sample_rate
            end = end_sample / sample_rate
            segments.append(SpeechSegment(start=start, end=end, confidence=float(self.threshold)))
        return segments


class WebRtcVadStrategy:
    def __init__(self, mode: int = 2) -> None:
        self.mode = max(0, min(3, int(mode)))

    def detect(
        self,
        samples: np.ndarray,
        sample_rate: int,
        frame_ms: float = 30.0,
        hop_ms: float = 10.0,
        min_speech_ms: float = 300.0,
    ) -> list[SpeechSegment]:
        try:
            import webrtcvad
        except Exception as exc:  # pragma: no cover - dependent on optional install
            raise RuntimeError(
                "WebRTC VAD requires optional extras: pip install double-ender-sync[vad-ml]"
            ) from exc

        if sample_rate <= 0:
            return []
        mono = np.asarray(samples, dtype=np.float32)
        if mono.size == 0:
            return []

        # WebRTC VAD accepts only specific sample rates / frame sizes.
        target_sample_rate = sample_rate if sample_rate in {8000, 16000, 32000, 48000} else 16000
        if target_sample_rate != sample_rate:
            from scipy.signal import resample_poly

            mono = resample_poly(mono, target_sample_rate, sample_rate).astype(np.float32, copy=False)

        frame_ms_int = int(frame_ms)
        if frame_ms_int not in {10, 20, 30}:
            frame_ms_int = 30
        frame_size = int(target_sample_rate * frame_ms_int / 1000)
        if frame_size <= 0:
            return []
        hop_size = max(1, int(target_sample_rate * hop_ms / 1000.0))

        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        normalized = mono / max(peak, 1.0)
        pcm16 = np.clip(normalized * 32767.0, -32768, 32767).astype(np.int16)
        vad = webrtcvad.Vad(self.mode)

        flags: list[bool] = []
        starts: list[int] = []
        energies: list[float] = []
        for start in range(0, len(pcm16) - frame_size + 1, hop_size):
            frame = pcm16[start : start + frame_size]
            starts.append(start)
            energies.append(float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2) + 1e-12)))
            flags.append(bool(vad.is_speech(frame.tobytes(), target_sample_rate)))

        if not flags:
            return []
        return _segments_from_flags(
            speech_flags=flags,
            starts=starts,
            hop_size=hop_size,
            sample_rate=target_sample_rate,
            energies=energies,
            min_speech_ms=min_speech_ms,
        )


class PyannoteVadStrategy:
    """pyannote-based VAD strategy backed by optional ``pyannote.audio``."""

    def __init__(self, model_name: str = DEFAULT_PYANNOTE_MODEL, auth_token: str | None = None) -> None:
        self.model_name = _canonicalize_pyannote_model_name(model_name)
        self.auth_token = auth_token

    def detect(
        self,
        samples: np.ndarray,
        sample_rate: int,
        frame_ms: float = 30.0,
        hop_ms: float = 10.0,
        min_speech_ms: float = 300.0,
    ) -> list[SpeechSegment]:
        if sample_rate <= 0:
            return []
        mono = np.asarray(samples, dtype=np.float32)
        if mono.size == 0:
            return []

        try:
            resolved_auth_token = _resolve_huggingface_auth_token(self.auth_token)
            LOGGER.info("Loading pyannote VAD model: %s", self.model_name)
            pipeline = _load_selected_pyannote_vad_pipeline(
                model_name=self.model_name,
                auth_token=resolved_auth_token,
            )
            # Pass waveform input directly instead of asking pyannote to decode a
            # temporary file path. This keeps the project off pyannote's legacy
            # torchaudio-backed file I/O path and aligns the runtime with the
            # TorchCodec transition in newer PyTorch audio stacks.
            waveform_input = _build_pyannote_waveform_input(mono=mono, sample_rate=sample_rate)
            diarization = pipeline(waveform_input)
        except Exception as exc:
            if isinstance(exc, PyannoteRuntimeCompatibilityError):
                raise
            if isinstance(exc, RuntimeError) and (
                "Install runtime dependencies" in str(exc)
                or "requires optional extras" in str(exc)
            ):
                raise
            if "FFmpeg extension is not available" in str(exc):
                raise RuntimeError(
                    "pyannote VAD failed because FFmpeg extension is unavailable. "
                    f"{_ffmpeg_extension_install_hint()}"
                ) from exc
            if _looks_like_hf_auth_error(exc):
                raise RuntimeError(
                    f"pyannote VAD could not access model '{self.model_name}' on Hugging Face "
                    "(HTTP 401/403 or gated/private model). "
                    "If you passed an auth_token directly, verify it is valid and not expired. "
                    f"Otherwise set HF_TOKEN or HUGGINGFACE_HUB_TOKEN, then accept model terms at "
                    f"https://hf.co/{self.model_name}."
                ) from exc
            raise RuntimeError(
                f"pyannote VAD failed to initialize or run with selected model '{self.model_name}'. "
                "Verify model access/token and local runtime dependencies."
            ) from exc

        return _pyannote_result_to_speech_segments(diarization, min_speech_ms=min_speech_ms)


def _pyannote_result_to_speech_segments(diarization: Any, min_speech_ms: float) -> list[SpeechSegment]:
    """Convert pyannote VAD/diarization outputs to speech-only timeline segments."""
    min_duration = min_speech_ms / 1000.0
    segments: list[SpeechSegment] = []
    for segment in _iter_pyannote_timeline_segments(diarization):
        start = float(segment.start)
        end = float(segment.end)
        if end - start < min_duration:
            continue
        segments.append(SpeechSegment(start=start, end=end, confidence=0.5))
    return segments


def _iter_pyannote_timeline_segments(diarization: Any):
    """Yield timeline segments from pyannote pipeline outputs across versions.

    Legacy VAD pipelines commonly return an Annotation-like object with
    ``get_timeline().support()``.  Newer diarization pipelines such as
    ``pyannote/speaker-diarization-community-1`` may wrap the annotation under
    ``speaker_diarization``.  The alignment engine only needs the union of
    speech activity, so diarization speaker labels are intentionally ignored.
    """
    get_timeline = getattr(diarization, "get_timeline", None)
    if callable(get_timeline):
        yield from get_timeline().support()
        return

    speaker_diarization = getattr(diarization, "speaker_diarization", None)
    if speaker_diarization is None:
        return

    get_timeline = getattr(speaker_diarization, "get_timeline", None)
    if callable(get_timeline):
        yield from get_timeline().support()
        return

    for item in speaker_diarization:
        if isinstance(item, tuple):
            yield item[0]
        else:
            yield item


def _build_pyannote_waveform_input(mono: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Build pyannote waveform input without invoking pyannote/torchaudio file decoding."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional runtime dependency details
        raise RuntimeError(
            "pyannote VAD requires optional extras for direct waveform input: "
            "pip install double-ender-sync[vad-pyannote]"
        ) from exc
    clipped = np.clip(np.asarray(mono, dtype=np.float32), -1.0, 1.0)
    waveform = torch.from_numpy(clipped).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def _canonicalize_pyannote_model_name(model_name: str) -> str:
    stripped = model_name.strip()
    if stripped.lower() == MODERN_PYANNOTE_SEGMENTATION_MODEL.lower():
        return MODERN_PYANNOTE_SEGMENTATION_MODEL
    return stripped


def _is_pyannote_segmentation_model(model_name: str) -> bool:
    normalized = _canonicalize_pyannote_model_name(model_name).lower()
    return any(normalized.startswith(prefix) for prefix in PYANNOTE_SEGMENTATION_MODEL_PREFIXES)


def _is_pyannote_community_diarization_model(model_name: str) -> bool:
    normalized = _canonicalize_pyannote_model_name(model_name).lower()
    return normalized in {
        "pyannote/speaker-diarization-community-1",
        "pyannote-community/speaker-diarization-community-1",
    }


def _parse_version_prefix(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for raw_part in version.split("."):
        digits = ""
        for char in raw_part:
            if not char.isdigit():
                break
            digits += char
        if digits == "":
            break
        parts.append(int(digits))
    if parts:
        # Normalize bare major versions such as ``4`` or ``4rc1`` so they
        # compare correctly against two-component minimum versions like (4, 0).
        while len(parts) < 2:
            parts.append(0)
    return tuple(parts)


def _ensure_pyannote_model_runtime_compatible(model_name: str) -> None:
    """Fail early for pyannote pipelines whose configs require newer pyannote.audio.

    ``pyannote/speaker-diarization-community-1`` is a pyannote.audio 4.x
    pipeline. Older 3.x runtimes can download its config but then fail with
    implementation-detail errors such as ``SpeakerDiarization.__init__() got
    an unexpected keyword argument 'plda'``. Detect that mismatch before
    loading so users get an actionable dependency message.
    """
    if not _is_pyannote_community_diarization_model(model_name):
        return

    try:
        version_str = importlib_metadata.version("pyannote.audio")
    except importlib_metadata.PackageNotFoundError:
        return

    try:
        parsed_version = _PackagingVersion(version_str)
        compatible = parsed_version.major >= PYANNOTE_AUDIO_COMMUNITY_DIARIZATION_MIN_MAJOR
    except _InvalidVersion:
        # Fall back to tuple-based comparison when packaging fails to parse
        # an unusual version string.
        compatible = _parse_version_prefix(version_str) >= PYANNOTE_AUDIO_COMMUNITY_DIARIZATION_MIN_VERSION

    if compatible:
        return

    raise PyannoteRuntimeCompatibilityError(
        f"pyannote model '{model_name}' requires pyannote.audio >= 4.0, "
        f"but the installed pyannote.audio version is {version_str}. "
        "Upgrade the pyannote extra with `pip install -U 'double-ender-sync[vad-pyannote]'` "
        "or select `--pyannote-model pyannote/segmentation-3.0` for the legacy-compatible segmentation VAD path."
    )


def _load_selected_pyannote_vad_pipeline(model_name: str, auth_token: str | None):
    """Load either the legacy pyannote VAD pipeline or a modern segmentation-model VAD path."""
    if _is_pyannote_segmentation_model(model_name):
        try:
            from pyannote.audio import Model
            from pyannote.audio.pipelines import VoiceActivityDetection
        except Exception as exc:  # pragma: no cover - dependent on optional install
            raise RuntimeError(
                "pyannote VAD requires optional extras: pip install double-ender-sync[vad-pyannote]"
            ) from exc
        return _load_pyannote_segmentation_vad_pipeline(
            Model=Model,
            VoiceActivityDetection=VoiceActivityDetection,
            model_name=model_name,
            auth_token=auth_token,
        )

    _ensure_pyannote_model_runtime_compatible(model_name)

    try:
        from pyannote.audio import Pipeline
    except Exception as exc:  # pragma: no cover - dependent on optional install
        raise RuntimeError(
            "pyannote VAD requires optional extras: pip install double-ender-sync[vad-pyannote]"
        ) from exc

    return _load_pyannote_pipeline(Pipeline=Pipeline, model_name=model_name, auth_token=auth_token)


def _load_pyannote_segmentation_vad_pipeline(Model, VoiceActivityDetection, model_name: str, auth_token: str | None):
    """Build pyannote.audio 3.x VAD from a segmentation model.

    The short on/off durations preserve sub-second speech islands useful for
    anchor selection while still smoothing tiny boundary flickers. These values
    are intentionally not final-editing decisions; downstream anchor selection
    and reporting remain responsible for confidence and uncertainty.
    """
    segmentation_model = _load_pyannote_model(Model=Model, model_name=model_name, auth_token=auth_token)
    pipeline = VoiceActivityDetection(segmentation=segmentation_model)
    hyperparameters = {
        "min_duration_on": PYANNOTE_VAD_MIN_DURATION_ON_SECONDS,
        "min_duration_off": PYANNOTE_VAD_MIN_DURATION_OFF_SECONDS,
    }
    instantiate = getattr(pipeline, "instantiate", None)
    if callable(instantiate):
        pipeline = instantiate(hyperparameters)
    else:
        for key, value in hyperparameters.items():
            setattr(pipeline, key, value)
    return pipeline


def _load_pyannote_model(Model, model_name: str, auth_token: str | None):
    """Load a pyannote model while preserving legacy/current auth compatibility."""
    if auth_token is None:
        try:
            return _call_pyannote_from_pretrained_with_torch_compat(lambda: Model.from_pretrained(model_name))
        except TypeError as exc:
            if "use_auth_token" not in str(exc):
                raise
            _patch_pyannote_hf_download_compat()
            return _call_pyannote_from_pretrained_with_torch_compat(lambda: Model.from_pretrained(model_name))

    try:
        return _call_pyannote_from_pretrained_with_torch_compat(lambda: Model.from_pretrained(model_name, token=auth_token))
    except TypeError as exc:
        if "token" not in str(exc):
            raise
        try:
            return _call_pyannote_from_pretrained_with_torch_compat(lambda: Model.from_pretrained(model_name, use_auth_token=auth_token))
        except TypeError as legacy_exc:
            if "use_auth_token" not in str(legacy_exc):
                raise
            _patch_pyannote_hf_download_compat()
            return _call_pyannote_from_pretrained_with_torch_compat(lambda: Model.from_pretrained(model_name, use_auth_token=auth_token))


def _resolve_huggingface_auth_token(auth_token: str | None) -> str | None:
    if auth_token is not None:
        return auth_token
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def _looks_like_hf_auth_error(exc: Exception) -> bool:
    message = str(exc).lower()
    explicit_markers = (
        "403",
        "forbidden",
        "authentication",
        "authorization",
        "invalid token",
        "access token",
        "gated repo",
        "gated model",
        "accept the conditions",
    )
    if any(marker in message for marker in explicit_markers):
        return True
    has_auth_status = "401" in message or "403" in message
    has_auth_context = any(
        marker in message
        for marker in ("token", "auth", "unauthorized", "forbidden", "gated", "permission")
    )
    return has_auth_status and has_auth_context


def _load_pyannote_pipeline(Pipeline, model_name: str, auth_token: str | None):
    """Load pyannote pipeline while tolerating runtime compatibility differences."""
    if auth_token is None:
        try:
            return _call_pyannote_from_pretrained_with_torch_compat(
                lambda: Pipeline.from_pretrained(model_name)
            )
        except TypeError as exc:
            if "use_auth_token" not in str(exc):
                raise
            _patch_pyannote_hf_download_compat()
            return _call_pyannote_from_pretrained_with_torch_compat(
                lambda: Pipeline.from_pretrained(model_name)
            )

    try:
        return _call_pyannote_from_pretrained_with_torch_compat(
            lambda: Pipeline.from_pretrained(model_name, token=auth_token)
        )
    except TypeError as exc:
        # Older pyannote releases still use the legacy keyword.
        if "token" not in str(exc):
            raise
        try:
            return _call_pyannote_from_pretrained_with_torch_compat(
                lambda: Pipeline.from_pretrained(model_name, use_auth_token=auth_token)
            )
        except TypeError as legacy_exc:
            if "use_auth_token" not in str(legacy_exc):
                raise
            _patch_pyannote_hf_download_compat()
            return _call_pyannote_from_pretrained_with_torch_compat(
                lambda: Pipeline.from_pretrained(model_name, use_auth_token=auth_token)
            )


def _call_pyannote_from_pretrained_with_torch_compat(load_pipeline: Callable[[], Any]) -> Any:
    """Retry pyannote checkpoint loading with legacy torch.load semantics when needed.

    PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
    pyannote pipeline checkpoints can still include trusted PyTorch Lightning
    callback objects, so older pyannote checkpoints may raise a
    ``Weights only load failed`` unpickling error while loading the pipeline.
    Retry only that specific failure with ``weights_only=False`` scoped to this
    pyannote model-load call.
    """
    try:
        return load_pipeline()
    except Exception as exc:
        if not _looks_like_torch_weights_only_error(exc):
            raise
        with _torch_load_weights_only_false_context():
            return load_pipeline()


_torch_load_compat_tls = threading.local()

_TORCH_WEIGHTS_ONLY_MARKERS = ("Weights only load failed", "WeightsUnpickler")


def _looks_like_torch_weights_only_error(exc: Exception) -> bool:
    """Return True only for the specific PyTorch 2.6+ weights-only checkpoint failure.

    Requires both a known unpickling exception type (``pickle.UnpicklingError``
    or a torch-native pickle error bubbled through the exception chain) **and**
    the exact PyTorch ``Weights only load failed`` / ``WeightsUnpickler``
    message markers.  This prevents unrelated failures from accidentally
    triggering an ``weights_only=False`` retry.
    """
    candidate: BaseException | None = exc
    while candidate is not None:
        if isinstance(candidate, pickle.UnpicklingError):
            message = str(candidate)
            if any(m in message for m in _TORCH_WEIGHTS_ONLY_MARKERS):
                return True
        candidate = candidate.__cause__ or (
            candidate.__context__ if not candidate.__suppress_context__ else None
        )
    return False


@contextlib.contextmanager
def _torch_load_weights_only_false_context() -> Iterator[None]:
    """Temporarily inject ``weights_only=False`` into ``torch.load`` for the current thread only.

    The monkeypatch is applied globally so that calls originating from inside
    pyannote's ``Pipeline.from_pretrained`` (which we cannot modify) are
    intercepted.  However, only the calling thread has the thread-local
    ``_torch_load_compat_tls.active`` flag set, so concurrent ``torch.load``
    calls on other threads are **not** affected by the override.
    """
    torch_module = sys.modules.get("torch")
    original_load = getattr(torch_module, "load", None) if torch_module is not None else None
    if original_load is None:
        yield
        return

    def _torch_load_with_legacy_checkpoint_default(*args, **kwargs):
        if getattr(_torch_load_compat_tls, "active", False):
            # Some loaders call ``torch.load`` without ``weights_only`` and rely on
            # PyTorch 2.6+'s default, while newer Lightning/Fabric shims can pass
            # ``weights_only=True`` explicitly.  This context is only entered after
            # pyannote has already failed with the specific safe-unpickler checkpoint
            # error, so force the legacy value for this scoped retry.
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch_module.load = _torch_load_with_legacy_checkpoint_default
    _torch_load_compat_tls.active = True
    try:
        yield
    finally:
        _torch_load_compat_tls.active = False
        torch_module.load = original_load


def _patch_pyannote_hf_download_compat() -> None:
    """Patch pyannote's cached hf_hub_download imports for hub auth kwarg compatibility.

    Some older pyannote.audio releases still call ``hf_hub_download`` with the
    removed ``use_auth_token`` keyword. Newer huggingface_hub versions expect
    ``token`` instead, and pyannote keeps module-local references to
    ``hf_hub_download`` in more than one place while loading a pipeline and its
    underlying model. Patch every known local reference so the legacy pyannote
    path keeps working with current huggingface_hub.
    """
    for module_name in (
        "pyannote.audio.core.pipeline",
        "pyannote.audio.core.model",
    ):
        _patch_module_hf_download_compat(module_name)


def _patch_module_hf_download_compat(module_name: str) -> None:
    module = sys.modules.get(module_name)
    if module is None:
        try:
            module = __import__(module_name, fromlist=["hf_hub_download"])
        except Exception:
            return

    original = getattr(module, "hf_hub_download", None)
    if original is None or getattr(original, "_double_ender_sync_compat", False):
        return

    def _hf_hub_download_compat(*args, **kwargs):
        if "use_auth_token" in kwargs and "token" not in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        else:
            kwargs.pop("use_auth_token", None)
        return original(*args, **kwargs)

    setattr(_hf_hub_download_compat, "_double_ender_sync_compat", True)
    module.hf_hub_download = _hf_hub_download_compat



def build_vad_strategy(name: str, pyannote_model: str | None = None) -> VadStrategy:
    normalized = name.strip().lower()
    if normalized == "rms":
        return RmsVadStrategy()
    if normalized == "adaptive_rms":
        return AdaptiveRmsVadStrategy()
    if normalized == "silero":
        return SileroVadStrategy()
    if normalized == "webrtc":
        return WebRtcVadStrategy()
    if normalized == "pyannote":
        return PyannoteVadStrategy(model_name=pyannote_model or DEFAULT_PYANNOTE_MODEL)
    raise ValueError(f"Unsupported VAD strategy: {name}")


def _ffmpeg_extension_install_hint() -> str:
    return (
        "Install runtime dependencies, then retry: "
        "(1) install FFmpeg (macOS: `brew install ffmpeg@6`, Ubuntu/Debian: `sudo apt-get install -y ffmpeg`, "
        "Windows: `winget install Gyan.FFmpeg`), "
        "(2) install Python packages: `pip install -U torch torchcodec pyannote.audio`."
    )


def detect_speech_segments(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    energy_threshold: float = 0.02,
    min_speech_ms: float = 300.0,
    vad_strategy: VadStrategy | None = None,
) -> list[SpeechSegment]:
    strategy = vad_strategy if vad_strategy is not None else RmsVadStrategy(energy_threshold=energy_threshold)
    return strategy.detect(
        samples=samples,
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        min_speech_ms=min_speech_ms,
    )


def _compute_frame_energies(samples: np.ndarray, frame_size: int, hop_size: int) -> tuple[list[float], list[int]]:
    if len(samples) < frame_size:
        return [], []
    energies: list[float] = []
    starts: list[int] = []
    for start in range(0, len(samples) - frame_size + 1, hop_size):
        frame = samples[start : start + frame_size]
        energies.append(float(np.sqrt(np.mean(frame * frame) + 1e-12)))
        starts.append(start)
    return energies, starts


def _detect_speech_segments_from_energy(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: float,
    hop_ms: float,
    min_speech_ms: float,
    energy_threshold: float,
) -> list[SpeechSegment]:
    frame_size = max(1, int(sample_rate * frame_ms / 1000.0))
    hop_size = max(1, int(sample_rate * hop_ms / 1000.0))
    energies, starts = _compute_frame_energies(samples, frame_size, hop_size)
    if not energies:
        return []

    return _segments_from_flags(
        speech_flags=(energy >= energy_threshold for energy in energies),
        starts=starts,
        hop_size=hop_size,
        sample_rate=sample_rate,
        energies=energies,
        min_speech_ms=min_speech_ms,
    )


def _segments_from_flags(
    speech_flags,
    starts: list[int],
    hop_size: int,
    sample_rate: int,
    energies: list[float],
    min_speech_ms: float,
) -> list[SpeechSegment]:
    flags = list(speech_flags)
    segments: list[SpeechSegment] = []
    run_start = None
    for idx, is_speech in enumerate(flags):
        if is_speech and run_start is None:
            run_start = idx
        if not is_speech and run_start is not None:
            segments.append(_build_segment(run_start, idx - 1, starts, hop_size, sample_rate, energies))
            run_start = None

    if run_start is not None:
        segments.append(_build_segment(run_start, len(flags) - 1, starts, hop_size, sample_rate, energies))

    min_duration = min_speech_ms / 1000.0
    return [segment for segment in segments if (segment.end - segment.start) >= min_duration]


def _build_segment(run_start: int, run_end: int, starts: list[int], hop_size: int, sample_rate: int, energies: list[float]) -> SpeechSegment:
    start_sample = starts[run_start]
    end_sample = starts[run_end] + hop_size
    confidence = max(0.0, min(1.0, float(np.mean(energies[run_start : run_end + 1]) / 0.1)))
    return SpeechSegment(start=start_sample / sample_rate, end=end_sample / sample_rate, confidence=confidence)
