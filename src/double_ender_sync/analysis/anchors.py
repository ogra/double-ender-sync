from dataclasses import dataclass

import numpy as np

from double_ender_sync.analysis.vad import SpeechSegment


@dataclass
class AnchorCandidate:
    local_start: float
    local_end: float
    confidence: float
    rms: float


def select_anchor_candidates(
    samples: np.ndarray,
    sample_rate: int,
    speech_segments: list[SpeechSegment],
    min_anchor_duration: float = 1.0,
    max_anchor_duration: float = 4.0,
) -> list[AnchorCandidate]:
    anchors: list[AnchorCandidate] = []
    for segment in speech_segments:
        duration = segment.end - segment.start
        if duration < min_anchor_duration:
            continue

        anchor_start = segment.start
        anchor_end = min(segment.end, segment.start + max_anchor_duration)
        start_index = int(anchor_start * sample_rate)
        end_index = int(anchor_end * sample_rate)
        clip = samples[start_index:end_index]
        if clip.size == 0:
            continue

        rms = float(np.sqrt(np.mean(clip * clip) + 1e-12))
        anchors.append(
            AnchorCandidate(
                local_start=anchor_start,
                local_end=anchor_end,
                confidence=segment.confidence,
                rms=rms,
            )
        )

    anchors.sort(key=lambda anchor: (anchor.confidence, anchor.rms), reverse=True)
    return anchors[:5]
