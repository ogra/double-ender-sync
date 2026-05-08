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
    included_in_regression: bool = False
    rejected_reason: str | None = None


@dataclass
class DriftFitWarning:
    code: str
    message: str
    time_seconds: float | None = None


@dataclass
class DriftFitDiagnostics:
    input_anchor_count: int
    matched_anchor_count: int
    fitted_anchor_count: int
    outlier_count: int
    local_span_start_seconds: float | None
    local_span_end_seconds: float | None
    local_span_seconds: float
    local_span_ratio: float | None
    residual_rejection_threshold_ms: float | None
    warnings: list[DriftFitWarning]


@dataclass
class DriftEstimate:
    offset_seconds: float
    stretch_ratio: float
    anchor_count: int
    residual_median_ms: float
    residual_max_ms: float
    diagnostics: DriftFitDiagnostics | None = None


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


def fit_linear_drift_model(
    anchor_matches: list[AnchorMatch],
    local_duration_seconds: float | None = None,
) -> DriftEstimate | None:
    if len(anchor_matches) < 2:
        return None

    for match in anchor_matches:
        match.included_in_regression = False
        match.rejected_reason = None
        match.residual_ms = None

    local = np.array([m.local_start for m in anchor_matches], dtype=np.float64)
    master = np.array([m.master_start for m in anchor_matches], dtype=np.float64)
    weights = np.array([max(m.confidence, 1e-3) for m in anchor_matches], dtype=np.float64)

    kept_indices = np.arange(len(anchor_matches))
    residual_rejection_threshold_ms: float | None = None
    for _ in range(2):
        x = local[kept_indices]
        y = master[kept_indices]
        w = weights[kept_indices]
        stretch, offset = _weighted_linear_fit(x, y, w)
        residuals_ms = (y - (stretch * x + offset)) * 1000.0
        median = float(np.median(np.abs(residuals_ms)))
        mad = float(np.median(np.abs(residuals_ms - np.median(residuals_ms))))
        threshold = max(40.0, 3.5 * mad, 2.5 * median)
        residual_rejection_threshold_ms = float(threshold)
        keep_mask = np.abs(residuals_ms) <= threshold
        if keep_mask.all() or keep_mask.sum() < 2:
            break
        kept_indices = kept_indices[keep_mask]

    x = local[kept_indices]
    y = master[kept_indices]
    w = weights[kept_indices]
    stretch, offset = _weighted_linear_fit(x, y, w)
    residuals_ms = (y - (stretch * x + offset)) * 1000.0

    kept_index_set = set(int(idx) for idx in kept_indices)
    for idx, match in enumerate(anchor_matches):
        residual_ms = (match.master_start - (stretch * match.local_start + offset)) * 1000.0
        match.residual_ms = float(residual_ms)
        match.included_in_regression = idx in kept_index_set
        if not match.included_in_regression:
            match.rejected_reason = "residual_outlier"

    diagnostics = _build_drift_fit_diagnostics(
        anchor_matches=anchor_matches,
        kept_indices=kept_indices,
        local_duration_seconds=local_duration_seconds,
        residual_rejection_threshold_ms=residual_rejection_threshold_ms,
    )

    return DriftEstimate(
        offset_seconds=float(offset),
        stretch_ratio=float(stretch),
        anchor_count=int(len(kept_indices)),
        residual_median_ms=float(np.median(np.abs(residuals_ms))),
        residual_max_ms=float(np.max(np.abs(residuals_ms))),
        diagnostics=diagnostics,
    )


def _weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([x, np.ones_like(x)])
    W = np.diag(w)
    beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)
    return float(beta[0]), float(beta[1])


def _build_drift_fit_diagnostics(
    anchor_matches: list[AnchorMatch],
    kept_indices: np.ndarray,
    local_duration_seconds: float | None,
    residual_rejection_threshold_ms: float | None,
) -> DriftFitDiagnostics:
    kept_matches = [anchor_matches[int(idx)] for idx in kept_indices]
    span_start: float | None = None
    span_end: float | None = None
    span_seconds = 0.0
    span_ratio: float | None = None
    warnings: list[DriftFitWarning] = []

    if kept_matches:
        span_start = min(match.local_start for match in kept_matches)
        span_end = max(match.local_start for match in kept_matches)
        span_seconds = max(0.0, span_end - span_start)
        if local_duration_seconds is not None and local_duration_seconds > 0:
            span_ratio = span_seconds / local_duration_seconds

    outlier_count = len(anchor_matches) - len(kept_matches)
    if outlier_count > 0:
        warnings.append(
            DriftFitWarning(
                code="DRIFT_OUTLIERS_REJECTED",
                message="Some drift anchor matches were excluded from the linear regression as residual outliers.",
            )
        )

    if span_ratio is not None and local_duration_seconds is not None:
        if local_duration_seconds >= 120.0 and span_ratio < 0.25:
            warnings.append(
                DriftFitWarning(
                    code="VERY_WEAK_DRIFT_ANCHOR_SPAN",
                    message="Drift anchors cover only a small part of the local timeline; the fitted stretch ratio may be unreliable.",
                    time_seconds=span_start,
                )
            )
        elif local_duration_seconds >= 120.0 and span_ratio < 0.50 and len(kept_matches) >= 3:
            warnings.append(
                DriftFitWarning(
                    code="WEAK_DRIFT_ANCHOR_SPAN",
                    message="Drift anchors have enough count but cover too little of the local timeline; inspect alignment manually.",
                    time_seconds=span_start,
                )
            )

    return DriftFitDiagnostics(
        input_anchor_count=len(anchor_matches),
        matched_anchor_count=len(anchor_matches),
        fitted_anchor_count=len(kept_matches),
        outlier_count=outlier_count,
        local_span_start_seconds=span_start,
        local_span_end_seconds=span_end,
        local_span_seconds=float(span_seconds),
        local_span_ratio=None if span_ratio is None else float(span_ratio),
        residual_rejection_threshold_ms=residual_rejection_threshold_ms,
        warnings=warnings,
    )
