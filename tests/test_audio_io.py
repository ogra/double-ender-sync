from pathlib import Path

import numpy as np
import soundfile as sf

from double_ender_sync.audio.io import AudioLoadError, load_audio_track


def test_load_audio_track_extracts_metadata_and_analysis_audio(tmp_path: Path) -> None:
    sample_rate = 48000
    duration_seconds = 1.0
    frame_count = int(sample_rate * duration_seconds)

    t = np.arange(frame_count) / sample_rate
    left = 0.2 * np.sin(2 * np.pi * 440 * t)
    right = 0.2 * np.sin(2 * np.pi * 880 * t)
    stereo = np.stack([left, right], axis=1).astype(np.float32)

    wav_path = tmp_path / "speaker-a.wav"
    sf.write(wav_path, stereo, sample_rate)

    track = load_audio_track(wav_path, analysis_sample_rate=16000)

    assert track.sample_rate == sample_rate
    assert track.channels == 2
    assert abs(track.duration_seconds - duration_seconds) < 1e-6
    assert track.original_samples.dtype == np.float32
    assert track.analysis_samples.dtype == np.float32
    assert track.analysis_sample_rate == 16000
    assert abs(len(track.analysis_samples) - 16000) <= 1


def test_load_audio_track_raises_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.wav"
    try:
        load_audio_track(missing, analysis_sample_rate=16000)
    except AudioLoadError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("Expected AudioLoadError")


def test_load_audio_track_streaming_mode_skips_original_samples(tmp_path: Path) -> None:
    sample_rate = 48000
    frame_count = sample_rate * 2
    mono = (0.2 * np.sin(2 * np.pi * 220 * (np.arange(frame_count) / sample_rate))).astype(np.float32)
    wav_path = tmp_path / "speaker-stream.wav"
    sf.write(wav_path, mono, sample_rate)

    track = load_audio_track(wav_path, analysis_sample_rate=16000, include_original_samples=False)

    assert track.original_samples is None
    assert track.channels == 1
    assert abs(track.duration_seconds - 2.0) < 1e-6
    assert abs(len(track.analysis_samples) - 32000) <= 1
