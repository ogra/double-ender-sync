from pathlib import Path

import numpy as np
import soundfile as sf

from double_ender_sync.audio.render import write_synced_track


def test_write_synced_track_normalize_disabled_by_default(tmp_path: Path) -> None:
    output = tmp_path / "out.wav"
    samples = np.array([0.25, -0.5, 0.75], dtype=np.float32)

    write_synced_track(output, samples, sample_rate=48000)

    written, _ = sf.read(output, dtype="float32")
    assert float(np.max(np.abs(written))) == 0.75


def test_write_synced_track_normalize_enabled(tmp_path: Path) -> None:
    output = tmp_path / "out-normalized.wav"
    samples = np.array([0.25, -0.5, 0.75], dtype=np.float32)

    write_synced_track(output, samples, sample_rate=48000, normalize_output=True)

    written, _ = sf.read(output, dtype="float32")
    assert np.isclose(float(np.max(np.abs(written))), 1.0, atol=1e-4)
