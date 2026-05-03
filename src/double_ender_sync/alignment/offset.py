from dataclasses import dataclass
import logging

import numpy as np

from double_ender_sync.analysis.anchors import AnchorCandidate
from double_ender_sync.analysis.features import extract_anchor_feature, normalized_correlation_scores


LOGGER = logging.getLogger("double_ender_sync")


@dataclass
class OffsetEstimate:
    offset_seconds: float
    confidence: float
    local_anchor_start: float
    master_anchor_start: float
    score: float


def estimate_initial_offset(
    local_samples: np.ndarray,
    master_samples: np.ndarray,
    sample_rate: int,
    anchors: list[AnchorCandidate],
) -> OffsetEstimate | None:
    best: OffsetEstimate | None = None

    for anchor in anchors:
        local_start = int(anchor.local_start * sample_rate)
        local_end = int(anchor.local_end * sample_rate)
        local_clip = local_samples[local_start:local_end]
        if local_clip.size < int(0.5 * sample_rate):
            continue

        feature = extract_anchor_feature(local_clip)
        if len(feature) == 0 or len(feature) > master_samples.size:
            continue

        row_count = master_samples.size - len(feature) + 1
        LOGGER.debug(
            "offset-anchor local=[%.3f, %.3f]s feature_samples=%d search_rows=%d method=fft-ncc",
            anchor.local_start,
            anchor.local_end,
            len(feature),
            row_count,
        )

        scores = normalized_correlation_scores(master_samples, feature)

        best_score = float(np.max(scores))
        # Periodic signals can produce many nearly-identical maxima.
        # Prefer the earliest plausible peak to reduce forward drift bias.
        peak_tolerance = 1e-4
        near_peaks = np.flatnonzero(scores >= (best_score - peak_tolerance))
        best_index = int(near_peaks[0]) if near_peaks.size > 0 else int(np.argmax(scores))
        master_anchor_start = best_index / sample_rate
        offset_seconds = master_anchor_start - anchor.local_start
        confidence = max(0.0, min(1.0, (best_score + 1.0) / 2.0 * anchor.confidence))

        candidate = OffsetEstimate(
            offset_seconds=offset_seconds,
            confidence=confidence,
            local_anchor_start=anchor.local_start,
            master_anchor_start=master_anchor_start,
            score=best_score,
        )
        if best is None or candidate.score > best.score:
            best = candidate

        LOGGER.debug(
            "offset-anchor-result local_start=%.3f master_start=%.3f score=%.6f confidence=%.3f",
            anchor.local_start,
            master_anchor_start,
            best_score,
            confidence,
        )

    return best
