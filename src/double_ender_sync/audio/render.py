from pathlib import Path

import numpy as np
import soundfile as sf


def write_synced_track(
    output_path: Path,
    samples: np.ndarray,
    sample_rate: int,
    normalize_output: bool = False,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = samples
    if normalize_output:
        peak = float(np.max(np.abs(rendered))) if rendered.size else 0.0
        if peak > 0.0:
            rendered = rendered / peak
    clipped = np.clip(rendered, -1.0, 1.0)
    sf.write(output_path, clipped.astype(np.float32, copy=False), sample_rate)
    return output_path
