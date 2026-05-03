from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class AudioTrack:
    path: Path
    name: str
    sample_rate: int
    duration_seconds: float
    channels: int
    original_samples: np.ndarray | None
    analysis_samples: np.ndarray
    analysis_sample_rate: int
    temp_files: list[Path] = field(default_factory=list)
