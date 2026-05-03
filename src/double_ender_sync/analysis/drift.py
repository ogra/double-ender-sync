from dataclasses import dataclass
import logging

import numpy as np

from double_ender_sync.analysis.anchors import AnchorCandidate
from double_ender_sync.analysis.features import extract_anchor_feature, normalized_correlation_scores


LOGGER = logging.getLogger("double_ender_sync")


@dataclass
class AnchorMatch:
    local_start: float
    local_end: float
    master_start: float
    master_end: float
    offset_seconds: float
    confidence: float
    score: float
    residual_ms: float | None = None


@dataclass
class DriftEstimate:
    offset_seconds: float
    stretch_ratio: float
    anchor_count: int
    residual_median_ms: float
    residual_max_ms: float


def match_anchors_for_drift(
    local_samples: np.ndarray,
    master_samples: np.ndarray,
    sample_rate: int,
    anchors: list[AnchorCandidate],
    initial_offset_seconds: float,
    search_radius_seconds: float = 6.0,
) -> list[AnchorMatch]:
    matches: list[AnchorMatch] = []
    master_duration = master_samples.shape[0] / sample_rate

    for anchor in anchors:
        local_start_idx = int(anchor.local_start * sample_rate)
        local_end_idx = int(anchor.local_end * sample_rate)
        local_clip = local_samples[local_start_idx:local_end_idx]
        if local_clip.size < int(0.5 * sample_rate):
            continue

        feature = extract_anchor_feature(local_clip)
        expected_master_start = anchor.local_start + initial_offset_seconds
        search_start = max(0.0, expected_master_start - search_radius_seconds)
        search_end = min(master_duration, expected_master_start + search_radius_seconds + (local_clip.size / sample_rate))

        search_start_idx = int(search_start * sample_rate)
        search_end_idx = int(search_end * sample_rate)
        search_region = master_samples[search_start_idx:search_end_idx]
        if search_region.size < len(feature):
            continue

        row_count = search_region.size - len(feature) + 1
        LOGGER.debug(
            "drift-anchor local=[%.3f, %.3f]s expected_master=%.3fs search=[%.3f, %.3f]s rows=%d method=fft-ncc",
            anchor.local_start,
            anchor.local_end,
            expected_master_start,
            search_start,
            search_end,
            row_count,
        )

        scores = normalized_correlation_scores(search_region, feature)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        master_start = (search_start_idx + best_idx) / sample_rate
        master_end = master_start + (local_clip.size / sample_rate)
        offset_seconds = master_start - anchor.local_start
        confidence = max(0.0, min(1.0, ((best_score + 1.0) / 2.0) * anchor.confidence))

        matches.append(
            AnchorMatch(
                local_start=anchor.local_start,
                local_end=anchor.local_end,
                master_start=master_start,
                master_end=master_end,
                offset_seconds=offset_seconds,
                confidence=confidence,
                score=best_score,
            )
        )

    return matches


def fit_linear_drift_model(anchor_matches: list[AnchorMatch]) -> DriftEstimate | None:
    if len(anchor_matches) < 2:
        return None

    local = np.array([m.local_start for m in anchor_matches], dtype=np.float64)
    master = np.array([m.master_start for m in anchor_matches], dtype=np.float64)
    weights = np.array([max(m.confidence, 1e-3) for m in anchor_matches], dtype=np.float64)

    kept_indices = np.arange(len(anchor_matches))
    for _ in range(2):
        x = local[kept_indices]
        y = master[kept_indices]
        w = weights[kept_indices]
        stretch, offset = _weighted_linear_fit(x, y, w)
        residuals_ms = (y - (stretch * x + offset)) * 1000.0
        median = float(np.median(np.abs(residuals_ms)))
        mad = float(np.median(np.abs(residuals_ms - np.median(residuals_ms))))
        threshold = max(40.0, 3.5 * mad, 2.5 * median)
        keep_mask = np.abs(residuals_ms) <= threshold
        if keep_mask.all() or keep_mask.sum() < 2:
            break
        kept_indices = kept_indices[keep_mask]

    x = local[kept_indices]
    y = master[kept_indices]
    w = weights[kept_indices]
    stretch, offset = _weighted_linear_fit(x, y, w)
    residuals_ms = (y - (stretch * x + offset)) * 1000.0

    for idx in kept_indices:
        r = (anchor_matches[idx].master_start - (stretch * anchor_matches[idx].local_start + offset)) * 1000.0
        anchor_matches[idx].residual_ms = float(r)

    return DriftEstimate(
        offset_seconds=float(offset),
        stretch_ratio=float(stretch),
        anchor_count=int(len(kept_indices)),
        residual_median_ms=float(np.median(np.abs(residuals_ms))),
        residual_max_ms=float(np.max(np.abs(residuals_ms))),
    )


def _weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([x, np.ones_like(x)])
    W = np.diag(w)
    beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)
    return float(beta[0]), float(beta[1])

