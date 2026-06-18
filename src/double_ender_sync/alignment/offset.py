from dataclasses import dataclass, field
import logging

import numpy as np

from double_ender_sync.analysis.anchors import AnchorCandidate
from double_ender_sync.analysis.features import (
    clamp01,
    extract_anchor_feature,
    ncc_peak_diagnostics,
    normalized_correlation_scores,
)
from double_ender_sync.audio.resample import resample_to_sample_rate
from double_ender_sync.config import (
    DEFAULT_INITIAL_OFFSET_SAFETY_CONFIG,
    InitialOffsetSafetyConfig,
)


LOGGER = logging.getLogger("double_ender_sync")


@dataclass
class OffsetEstimate:
    offset_seconds: float
    confidence: float
    local_anchor_start: float
    master_anchor_start: float
    score: float
    estimation_method: str = "anchor_ncc"
    confidence_band: str | None = None
    fallback_attempted: bool = False
    fallback_selected: bool = False
    fallback_reason: str | None = None
    initial_offset_confidence_threshold: float | None = None
    selected_drift_search_radius_seconds: float | None = None
    max_drift_search_radius_seconds: float | None = None
    radius_reason: str | None = None
    anchor_ncc: dict | None = None
    coarse_fft_fallback: dict | None = None
    master_vad_rejection_count: int | None = None
    warnings: list[str] = field(default_factory=list)


def estimate_initial_offset(
    local_samples: np.ndarray,
    master_samples: np.ndarray,
    sample_rate: int,
    anchors: list[AnchorCandidate],
) -> OffsetEstimate | None:
    """Anchor-based initial offset estimator.

    This is the primary initial-offset path. It searches each anchor feature
    against the full master recording and returns the best-scoring match. The
    returned estimate carries safety-net diagnostics but does not run the
    coarse FFT fallback itself.
    """

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
            estimation_method="anchor_ncc",
            anchor_ncc={
                "offset_seconds": offset_seconds,
                "confidence": confidence,
                "local_anchor_start": anchor.local_start,
                "master_anchor_start": master_anchor_start,
                "score": best_score,
                "peak_margin": diagnostics.margin,
                "peak_prominence": diagnostics.prominence,
                "peak_width_seconds": diagnostics.width_samples / max(1.0, float(sample_rate)),
                "second_score": diagnostics.second_score,
                "second_lag_seconds": (
                    None
                    if diagnostics.second_lag_samples is None
                    else diagnostics.second_lag_samples / sample_rate
                ),
            },
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


def estimate_initial_offset_with_safety_net(
    local_samples: np.ndarray,
    master_samples: np.ndarray,
    sample_rate: int,
    anchors: list[AnchorCandidate],
    safety_config: InitialOffsetSafetyConfig | None = None,
) -> OffsetEstimate:
    """Initial offset estimator with coarse FFT fallback and confidence policy.

    Runs the anchor-based estimator first. If the anchor estimate is below the
    configured confidence threshold and the coarse fallback is enabled, it also
    computes a whole-recording downsampled FFT cross-correlation. The selected
    offset is the more reliable of the two, with explicit diagnostics when the
    result remains uncertain.
    """

    config = safety_config or DEFAULT_INITIAL_OFFSET_SAFETY_CONFIG
    anchor_estimate = estimate_initial_offset(local_samples, master_samples, sample_rate, anchors)

    anchor_confidence = anchor_estimate.confidence if anchor_estimate is not None else 0.0
    anchor_band = _select_confidence_band(anchor_confidence, config)
    fallback_estimate: OffsetEstimate | None = None
    fallback_attempted = False
    fallback_selected = False
    fallback_reason: str | None = None
    warnings: list[str] = []

    should_attempt_fallback = (
        config.coarse_fallback_enabled
        and anchor_confidence < config.initial_offset_min_confidence
    )

    if should_attempt_fallback:
        fallback_attempted = True
        fallback_estimate = _coarse_fft_fallback_offset(
            local_samples=local_samples,
            master_samples=master_samples,
            sample_rate=sample_rate,
            config=config,
        )
        if fallback_estimate is None:
            fallback_reason = "coarse_fallback_failed_or_ambiguous"
            warnings.append("coarse_offset_fallback_failed_or_ambiguous")
        else:
            fallback_confidence = fallback_estimate.confidence
            fallback_margin = fallback_estimate.coarse_fft_fallback.get("peak_margin") if fallback_estimate.coarse_fft_fallback else None
            margin_passes = (
                fallback_margin is None
                or fallback_margin >= config.coarse_fallback_min_peak_margin
            )
            confidence_advantage = fallback_confidence - anchor_confidence
            anchor_is_failed = anchor_band == "failed"

            if (
                fallback_confidence >= config.coarse_fallback_min_confidence
                and margin_passes
                and (confidence_advantage >= config.coarse_fallback_confidence_margin or anchor_is_failed)
            ):
                fallback_selected = True
                fallback_reason = "coarse_fallback_more_reliable_than_anchor_ncc"
            else:
                fallback_reason = _explain_fallback_rejection(
                    fallback_confidence=fallback_confidence,
                    anchor_confidence=anchor_confidence,
                    margin=fallback_margin,
                    min_confidence=config.coarse_fallback_min_confidence,
                    min_margin=config.coarse_fallback_min_peak_margin,
                    min_advantage=config.coarse_fallback_confidence_margin,
                )
                if anchor_estimate is not None and anchor_band != "failed":
                    warnings.append("coarse_offset_fallback_attempted_but_rejected")

        if fallback_estimate is not None and fallback_estimate.coarse_fft_fallback is not None:
            fallback_diagnostics = fallback_estimate.coarse_fft_fallback
            if fallback_diagnostics.get("memory_limited"):
                warnings.append("coarse_fallback_memory_limited")
            explicit_cap = fallback_diagnostics.get("explicit_cap_seconds")
            if explicit_cap is not None:
                original_local_duration = local_samples.size / sample_rate
                original_master_duration = master_samples.size / sample_rate
                if (
                    original_local_duration > explicit_cap
                    or original_master_duration > explicit_cap
                ):
                    warnings.append("coarse_fallback_duration_capped")

    selected = fallback_estimate if fallback_selected else anchor_estimate

    # Preserve any available diagnostic estimate so failure reasons are reported
    # even when the result is too uncertain for drift matching.
    diagnostic_estimate = selected
    if diagnostic_estimate is None:
        diagnostic_estimate = fallback_estimate if fallback_estimate is not None else anchor_estimate

    # The selected confidence/band must reflect the estimate that was actually
    # chosen for drift matching. Rejected fallback diagnostics are still kept
    # in coarse_fft_fallback/anchor_ncc, but they must not drive the band.
    if selected is None:
        selected_confidence = 0.0
        selected_band = "failed"
    else:
        selected_confidence = selected.confidence
        selected_band = _select_confidence_band(selected_confidence, config)
    has_usable_estimate = selected is not None and selected_band != "failed"

    if fallback_selected and anchor_estimate is not None:
        offset_disagreement = abs(selected.offset_seconds - anchor_estimate.offset_seconds)
        if offset_disagreement > 1.0 and selected.confidence < config.high_confidence_threshold:
            warnings.append("initial_offset_estimates_disagree")

    if selected_band in {"low", "failed"}:
        warnings.append("initial_offset_low_confidence")
    if fallback_selected:
        warnings.append("coarse_offset_fallback_used")

    radius, radius_reason = _select_drift_search_radius(
        confidence_band=selected_band,
        fallback_selected=fallback_selected,
        has_usable_estimate=has_usable_estimate,
        config=config,
    )

    if radius > config.high_confidence_search_radius_seconds:
        warnings.append("drift_search_radius_widened")

    result = OffsetEstimate(
        offset_seconds=diagnostic_estimate.offset_seconds if diagnostic_estimate is not None else 0.0,
        confidence=selected_confidence,
        local_anchor_start=diagnostic_estimate.local_anchor_start if diagnostic_estimate is not None else 0.0,
        master_anchor_start=diagnostic_estimate.master_anchor_start if diagnostic_estimate is not None else 0.0,
        score=diagnostic_estimate.score if diagnostic_estimate is not None else 0.0,
        estimation_method=(selected.estimation_method if selected is not None else "none"),
        confidence_band=selected_band,
        fallback_attempted=fallback_attempted,
        fallback_selected=fallback_selected,
        fallback_reason=fallback_reason,
        initial_offset_confidence_threshold=config.initial_offset_min_confidence,
        selected_drift_search_radius_seconds=radius,
        max_drift_search_radius_seconds=config.max_drift_search_radius_seconds,
        radius_reason=radius_reason,
        anchor_ncc=anchor_estimate.anchor_ncc if anchor_estimate is not None else None,
        coarse_fft_fallback=fallback_estimate.coarse_fft_fallback if fallback_estimate is not None else None,
        warnings=warnings,
    )
    return result


def _select_confidence_band(confidence: float, config: InitialOffsetSafetyConfig) -> str:
    if confidence >= config.high_confidence_threshold:
        return "high"
    if confidence >= config.medium_confidence_threshold:
        return "medium"
    if confidence >= config.low_confidence_threshold:
        return "low"
    return "failed"


def _select_drift_search_radius(
    confidence_band: str,
    fallback_selected: bool,
    has_usable_estimate: bool,
    config: InitialOffsetSafetyConfig,
) -> tuple[float, str]:
    if not has_usable_estimate:
        return 0.0, "no_usable_initial_estimate"
    if fallback_selected:
        if confidence_band == "high":
            radius = config.high_confidence_search_radius_seconds
            reason = "high_confidence_fallback_selected"
        elif confidence_band == "medium":
            radius = config.medium_confidence_search_radius_seconds
            reason = "medium_confidence_fallback_selected"
        else:
            radius = config.low_confidence_search_radius_seconds
            reason = "low_confidence_fallback_selected"
    else:
        if confidence_band == "high":
            radius = config.high_confidence_search_radius_seconds
            reason = "high_confidence_anchor_estimate"
        elif confidence_band == "medium":
            radius = config.medium_confidence_search_radius_seconds
            reason = "medium_confidence_anchor_estimate"
        else:
            radius = config.low_confidence_search_radius_seconds
            reason = "low_confidence_no_fallback_selected"
    radius = min(radius, config.max_drift_search_radius_seconds)
    return radius, reason


def _effective_fallback_duration_seconds(
    local_duration_seconds: float,
    master_duration_seconds: float,
    sample_rate: int,
    target_rate: int,
    config: InitialOffsetSafetyConfig,
) -> tuple[float | None, float | None, bool]:
    """Return the effective duration cap for the coarse fallback.

    Returns a tuple of (effective_cap_seconds, memory_cap_seconds, memory_limited).
    The effective cap is the minimum of the explicit user cap and the memory
    guard cap. If neither limits the input, returns (None, None, False).
    """

    explicit_cap = config.coarse_fallback_max_duration_seconds
    original_max_duration = max(local_duration_seconds, master_duration_seconds)

    # Memory guard: FFT correlation can allocate multiple complex64 work buffers.
    # With the current padding strategy, the FFT length can be on the order of
    # ~8 * duration * target_rate in the worst case (padding + power-of-two rounding).
    bytes_per_sample = 8  # complex64
    memory_headroom = 3
    max_fft_samples = int(
        config.coarse_fallback_max_memory_mb * 1024 * 1024 / bytes_per_sample / memory_headroom
    )
    memory_cap = max_fft_samples / (8.0 * target_rate)
    memory_guard_limited = memory_cap < original_max_duration

    if explicit_cap is None:
        effective_cap = memory_cap if memory_guard_limited else None
    else:
        effective_cap = min(explicit_cap, memory_cap)
        if effective_cap >= original_max_duration:
            effective_cap = None

    memory_limited = memory_guard_limited and (
        explicit_cap is None or memory_cap <= explicit_cap
    )
    memory_cap_to_report = memory_cap if memory_limited else None
    return effective_cap, memory_cap_to_report, memory_limited


def _coarse_fft_fallback_offset(
    local_samples: np.ndarray,
    master_samples: np.ndarray,
    sample_rate: int,
    config: InitialOffsetSafetyConfig,
) -> OffsetEstimate | None:
    """Whole-recording coarse offset fallback using downsampled FFT correlation."""

    if local_samples.size == 0 or master_samples.size == 0:
        return None

    original_local_duration = local_samples.size / sample_rate
    original_master_duration = master_samples.size / sample_rate
    target_rate = config.coarse_fallback_sample_rate

    effective_cap, memory_cap, memory_limited = _effective_fallback_duration_seconds(
        local_duration_seconds=original_local_duration,
        master_duration_seconds=original_master_duration,
        sample_rate=sample_rate,
        target_rate=target_rate,
        config=config,
    )
    if effective_cap is not None:
        max_samples = int(effective_cap * sample_rate)
        local_samples = local_samples[:max_samples]
        master_samples = master_samples[:max_samples]

    local_down = resample_to_sample_rate(local_samples, src_sample_rate=sample_rate, dst_sample_rate=target_rate)
    master_down = resample_to_sample_rate(master_samples, src_sample_rate=sample_rate, dst_sample_rate=target_rate)

    local_down = _robust_normalize(local_down)
    master_down = _robust_normalize(master_down)

    # Allow negative lags by padding the master at the front with silence equal
    # to the local length. A peak at lag L in the padded search corresponds to
    # original lag L - (len(local) - 1).
    # Also pad the tail so that positive lags where the local extends past the
    # master end are searchable as long as the overlap meets the minimum
    # threshold, e.g. when the local has post-roll or is longer than the
    # remaining master after the true offset.
    pad_length = local_down.size - 1
    min_overlap_samples = max(1, int(round(0.25 * local_down.size)))
    tail_pad = local_down.size - min_overlap_samples
    padded_master = np.concatenate([
        np.zeros(pad_length, dtype=np.float32),
        master_down,
        np.zeros(tail_pad, dtype=np.float32),
    ])

    if local_down.size > padded_master.size:
        return None

    scores = normalized_correlation_scores(padded_master, local_down)
    if scores.size == 0:
        return None

    # Ignore lags where the local window would overlap the master by less than
    # a quarter of its length. The full cross-correlation covers original lags
    # from -(local_length-1) to (master_length-1). A lag k<0 overlaps
    # min(|k|+1, master_length) samples; a lag k>=0 overlaps min(local_length,
    # master_length-k) samples. Converted to padded-lag space this becomes the
    # contiguous range below.
    first_valid = max(0, min_overlap_samples - 1)
    last_valid = min(len(scores) - 1, master_down.size + pad_length - min_overlap_samples)
    if first_valid > last_valid:
        return None

    valid_mask = np.zeros_like(scores, dtype=bool)
    valid_mask[first_valid : last_valid + 1] = True
    valid_scores = scores.copy()
    valid_scores[~valid_mask] = -1.0

    diagnostics = ncc_peak_diagnostics(
        valid_scores,
        sample_rate=target_rate,
        nms_exclusion_seconds=0.5,
    )
    if diagnostics is None:
        return None

    best_score = diagnostics.best_score
    if best_score < 0.10:
        return None

    padded_lag = diagnostics.best_lag_samples
    original_lag_samples = padded_lag - pad_length
    offset_seconds = original_lag_samples / target_rate
    if offset_seconds < 0.0:
        local_anchor_start = -offset_seconds
        master_anchor_start = 0.0
    else:
        local_anchor_start = 0.0
        master_anchor_start = offset_seconds
    confidence = clamp01((best_score + 1.0) / 2.0)
    margin = diagnostics.margin
    if margin is None and diagnostics.prominence is not None:
        margin = diagnostics.prominence

    LOGGER.debug(
        "coarse-fallback-offset offset=%.3fs score=%.4f margin=%s confidence=%.3f sample_rate=%d",
        offset_seconds,
        best_score,
        "None" if margin is None else f"{margin:.4f}",
        confidence,
        target_rate,
    )

    return OffsetEstimate(
        offset_seconds=offset_seconds,
        confidence=confidence,
        local_anchor_start=local_anchor_start,
        master_anchor_start=master_anchor_start,
        score=best_score,
        estimation_method="coarse_fft_fallback",
        coarse_fft_fallback={
            "offset_seconds": offset_seconds,
            "confidence": confidence,
            "peak_score": best_score,
            "peak_margin": margin,
            "peak_prominence": diagnostics.prominence,
            "second_score": diagnostics.second_score,
            "second_lag_seconds": (
                None
                if diagnostics.second_lag_samples is None
                else (diagnostics.second_lag_samples - pad_length) / target_rate
            ),
            "effective_sample_rate": target_rate,
            "local_duration_seconds": local_down.size / target_rate,
            "master_duration_seconds": master_down.size / target_rate,
            "original_local_duration_seconds": original_local_duration,
            "original_master_duration_seconds": original_master_duration,
            "searched_lag_min_seconds": (first_valid - pad_length) / target_rate,
            "searched_lag_max_seconds": (last_valid - pad_length) / target_rate,
            "memory_limited": memory_limited,
            "effective_max_duration_seconds": effective_cap,
            "memory_cap_seconds": memory_cap,
            "explicit_cap_seconds": config.coarse_fallback_max_duration_seconds,
        },
    )


def _robust_normalize(samples: np.ndarray) -> np.ndarray:
    """Zero-mean and scale to unit RMS."""
    samples_f32 = np.asarray(samples, dtype=np.float32)
    mean = float(np.mean(samples_f32, dtype=np.float64))
    centered = samples_f32 - mean
    rms = float(np.sqrt(np.mean(centered * centered, dtype=np.float64)))
    if rms <= 1e-12:
        return np.zeros_like(samples_f32, dtype=np.float32)
    return (centered / rms).astype(np.float32)


def _explain_fallback_rejection(
    fallback_confidence: float,
    anchor_confidence: float,
    margin: float | None,
    min_confidence: float,
    min_margin: float,
    min_advantage: float,
) -> str:
    reasons: list[str] = []
    if fallback_confidence < min_confidence:
        reasons.append(f"fallback_confidence {fallback_confidence:.3f} < {min_confidence}")
    if margin is not None and margin < min_margin:
        reasons.append(f"fallback_peak_margin {margin:.3f} < {min_margin}")
    if fallback_confidence - anchor_confidence < min_advantage:
        reasons.append(
            f"fallback_confidence_advantage {fallback_confidence - anchor_confidence:.3f} < {min_advantage}"
        )
    if not reasons:
        reasons.append("fallback did not clearly outperform anchor estimate")
    return "coarse_fallback_rejected: " + "; ".join(reasons)


def _earliest_near_tie(scores: np.ndarray, best_score: float, best_index: int, tolerance: float = 1e-4) -> int:
    """Return the earliest lag whose score is within *tolerance* of *best_score*."""
    near_tie_mask = scores >= (best_score - tolerance)
    near_tie_indices = np.where(near_tie_mask)[0]
    if near_tie_indices.size == 0:
        return best_index
    return int(near_tie_indices[0])
