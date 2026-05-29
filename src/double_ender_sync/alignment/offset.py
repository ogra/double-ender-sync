from dataclasses import dataclass
import logging

import numpy as np

from double_ender_sync.analysis.anchors import AnchorCandidate
from double_ender_sync.analysis.features import (
    clamp01,
    extract_anchor_feature,
    ncc_peak_diagnostics,
    normalized_correlation_scores,
)


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

        diagnostics = ncc_peak_diagnostics(
            scores,
            sample_rate=sample_rate,
            nms_exclusion_seconds=0.05,
        )

        if diagnostics is None:
            continue

        best_score = diagnostics.best_score

        if best_score < 0.10:
            continue

        best_index = diagnostics.best_lag_samples
        earliest_idx = _earliest_near_tie(scores, best_score, best_index)
        best_index = earliest_idx

        master_anchor_start = best_index / sample_rate
        offset_seconds = master_anchor_start - anchor.local_start
        confidence = max(0.0, min(1.0, clamp01((best_score + 1.0) / 2.0) * anchor.confidence))

        candidate = OffsetEstimate(
            offset_seconds=offset_seconds,
            confidence=confidence,
            local_anchor_start=anchor.local_start,
            master_anchor_start=master_anchor_start,
            score=best_score,
        )
        if best is None or candidate.score > best.score + 1e-4:
            best = candidate

        LOGGER.debug(
            "offset-anchor-result local_start=%.3f master_start=%.3f score=%.6f confidence=%.3f",
            anchor.local_start,
            master_anchor_start,
            best_score,
            confidence,
        )

    return best


def _earliest_near_tie(scores: np.ndarray, best_score: float, best_index: int, tolerance: float = 1e-4) -> int:
    """Return the earliest lag whose score is within *tolerance* of *best_score*."""
    near_tie_mask = scores >= (best_score - tolerance)
    near_tie_indices = np.where(near_tie_mask)[0]
    if near_tie_indices.size == 0:
        return best_index
    return int(near_tie_indices[0])
