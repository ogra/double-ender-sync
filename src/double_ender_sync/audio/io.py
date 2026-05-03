from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

from double_ender_sync.audio.normalize import convert_to_mono
from double_ender_sync.audio.resample import resample_for_analysis
from double_ender_sync.types import AudioTrack


class AudioLoadError(Exception):
    """Raised when an audio file cannot be loaded."""


def load_audio_track(path: Path, analysis_sample_rate: int, include_original_samples: bool = True) -> AudioTrack:
    if not path.exists():
        raise AudioLoadError(f"Audio file does not exist: {path}")
    if not path.is_file():
        raise AudioLoadError(f"Audio path is not a file: {path}")

    try:
        info = sf.info(path)
        sample_rate = info.samplerate
        channels = info.channels
        duration_seconds = info.frames / sample_rate
    except Exception as exc:  # noqa: BLE001
        raise AudioLoadError(f"Failed to inspect audio file: {path}: {exc}") from exc

    temp_files: list[Path] = []
    original_samples: np.ndarray | None = None
    if include_original_samples:
        try:
            original_samples, read_sample_rate = sf.read(path, dtype="float32", always_2d=False)
        except Exception as exc:  # noqa: BLE001
            raise AudioLoadError(f"Failed to decode audio file: {path}: {exc}") from exc
        if read_sample_rate != sample_rate:
            raise AudioLoadError(f"Inconsistent sample rate read for {path}")
        mono = convert_to_mono(original_samples)
    else:
        mono, mono_path = _read_mono_streaming(path, channels=channels, frame_count=info.frames)
        temp_files.append(mono_path)

    analysis = resample_for_analysis(mono, src_sample_rate=sample_rate, dst_sample_rate=analysis_sample_rate)
    analysis_mmap, analysis_path = _persist_analysis_samples(analysis)
    temp_files.append(analysis_path)

    return AudioTrack(
        path=path,
        name=path.stem,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        channels=channels,
        original_samples=original_samples,
        analysis_samples=analysis_mmap,
        analysis_sample_rate=analysis_sample_rate,
        temp_files=temp_files,
    )


def _read_mono_streaming(path: Path, channels: int, frame_count: int, block_size: int = 262144) -> tuple[np.ndarray, Path]:
    """Read mono analysis input via sf.blocks to reduce peak RAM on long recordings."""
    with tempfile.NamedTemporaryFile(prefix="double-ender-mono-", suffix=".bin", delete=False) as handle:
        mono_path = Path(handle.name)

    mono = np.memmap(mono_path, dtype=np.float32, mode="w+", shape=(frame_count,))
    cursor = 0
    for block in sf.blocks(path, blocksize=block_size, dtype="float32", always_2d=(channels > 1)):
        mono_block = convert_to_mono(block)
        end = cursor + len(mono_block)
        mono[cursor:end] = mono_block
        cursor = end

    if cursor != frame_count:
        raise AudioLoadError(f"Failed to stream all frames for {path}: expected {frame_count}, got {cursor}")

    return mono, mono_path


def _persist_analysis_samples(analysis_samples: np.ndarray) -> tuple[np.ndarray, Path]:
    """Persist analysis samples to a temporary mmap-backed file to reduce RAM pressure."""
    with tempfile.NamedTemporaryFile(prefix="double-ender-analysis-", suffix=".npy", delete=False) as handle:
        tmp_path = Path(handle.name)
    np.save(tmp_path, analysis_samples.astype(np.float32, copy=False))
    return np.load(tmp_path, mmap_mode="r"), tmp_path


def cleanup_temp_files(tracks: list[AudioTrack]) -> None:
    for track in tracks:
        for temp_path in track.temp_files:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                continue
