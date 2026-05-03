from dataclasses import dataclass

import numpy as np


@dataclass
class SpeechSegment:
    start: float
    end: float
    confidence: float


def detect_speech_segments(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    energy_threshold: float = 0.02,
    min_speech_ms: float = 300.0,
) -> list[SpeechSegment]:
    frame_size = max(1, int(sample_rate * frame_ms / 1000.0))
    hop_size = max(1, int(sample_rate * hop_ms / 1000.0))
    if len(samples) < frame_size:
        return []

    energies: list[float] = []
    starts: list[int] = []
    for start in range(0, len(samples) - frame_size + 1, hop_size):
        frame = samples[start : start + frame_size]
        energies.append(float(np.sqrt(np.mean(frame * frame) + 1e-12)))
        starts.append(start)

    speech_flags = [energy >= energy_threshold for energy in energies]

    segments: list[SpeechSegment] = []
    run_start = None
    for idx, is_speech in enumerate(speech_flags):
        if is_speech and run_start is None:
            run_start = idx
        if not is_speech and run_start is not None:
            segments.append(_build_segment(run_start, idx - 1, starts, hop_size, sample_rate, energies))
            run_start = None

    if run_start is not None:
        segments.append(_build_segment(run_start, len(speech_flags) - 1, starts, hop_size, sample_rate, energies))

    min_duration = min_speech_ms / 1000.0
    return [segment for segment in segments if (segment.end - segment.start) >= min_duration]


def _build_segment(run_start: int, run_end: int, starts: list[int], hop_size: int, sample_rate: int, energies: list[float]) -> SpeechSegment:
    start_sample = starts[run_start]
    end_sample = starts[run_end] + hop_size
    confidence = max(0.0, min(1.0, float(np.mean(energies[run_start : run_end + 1]) / 0.1)))
    return SpeechSegment(start=start_sample / sample_rate, end=end_sample / sample_rate, confidence=confidence)
