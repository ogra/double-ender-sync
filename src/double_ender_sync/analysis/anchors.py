from dataclasses import dataclass
from math import ceil

import numpy as np

from double_ender_sync.analysis.vad import SpeechSegment
from double_ender_sync.config import DEFAULT_ANCHOR_SELECTION_CONFIG, AnchorSelectionConfig


@dataclass
class AnchorCandidate:
    local_start: float
    local_end: float
    confidence: float
    rms: float
    bin_index: int | None = None
    snr_db: float | None = None
    spectral_flatness: float | None = None
    quality_multiplier: float = 1.0
    duration_seconds: float | None = None


@dataclass
class AnchorSelectionBinSummary:
    index: int
    start_seconds: float
    end_seconds: float
    candidate_count: int
    selected_count: int


@dataclass
class AnchorSelectionCoverageWarning:
    code: str
    message: str
    time_seconds: float | None = None


@dataclass
class AnchorSelectionDiagnostics:
    candidate_anchor_count: int
    selected_anchor_count: int
    target_anchor_count: int | None
    stratified_bin_count: int
    anchors_per_bin: int | None
    longest_unanchored_span_seconds: float
    sparse_bin_count: int
    adaptive_duration_min_seconds: float | None
    adaptive_duration_median_seconds: float | None
    adaptive_duration_max_seconds: float | None
    rejected_candidate_counts: dict[str, int]
    bins: list[AnchorSelectionBinSummary]
    warnings: list[AnchorSelectionCoverageWarning]


@dataclass
class AnchorSelectionResult:
    candidates: list[AnchorCandidate]
    diagnostics: AnchorSelectionDiagnostics


def compute_target_anchor_budget(duration_seconds: float, config: AnchorSelectionConfig) -> int | None:
    """Compute the duration-aware target count for selected anchor candidates.

    The budget is a target cap for already-valid candidates, not a minimum that
    forces weak regions into the selection. ``None`` means no explicit cap.
    """

    if config.max_anchor_count == 0:
        return 0
    track_minutes = max(duration_seconds, 0.0) / 60.0
    density_target = int(ceil(track_minutes * config.anchor_density_per_minute))
    target_anchor_count = max(config.min_anchor_count, density_target)
    if config.max_anchor_count is None:
        return None
    return min(target_anchor_count, config.max_anchor_count)


def compute_stratified_bin_count(duration_seconds: float, target_anchor_count: int | None, config: AnchorSelectionConfig) -> int:
    """Compute timeline-bin count for coverage-aware anchor selection."""

    if config.stratified_bin_count is not None:
        return config.stratified_bin_count
    if duration_seconds <= 0:
        return 1
    timeline_bins = max(1, int(ceil(duration_seconds / 60.0)))
    if target_anchor_count is None:
        return timeline_bins
    return max(1, min(target_anchor_count, timeline_bins))


def select_anchor_candidates(
    samples: np.ndarray,
    sample_rate: int,
    speech_segments: list[SpeechSegment],
    min_anchor_duration: float | None = None,
    max_anchor_duration: float | None = None,
    config: AnchorSelectionConfig | None = None,
) -> list[AnchorCandidate]:
    return select_anchor_candidates_with_diagnostics(
        samples=samples,
        sample_rate=sample_rate,
        speech_segments=speech_segments,
        min_anchor_duration=min_anchor_duration,
        max_anchor_duration=max_anchor_duration,
        config=config,
    ).candidates


def select_anchor_candidates_with_diagnostics(
    samples: np.ndarray,
    sample_rate: int,
    speech_segments: list[SpeechSegment],
    min_anchor_duration: float | None = None,
    max_anchor_duration: float | None = None,
    config: AnchorSelectionConfig | None = None,
) -> AnchorSelectionResult:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive for anchor selection")

    anchor_config = config or DEFAULT_ANCHOR_SELECTION_CONFIG
    min_duration = anchor_config.min_anchor_duration_seconds if min_anchor_duration is None else min_anchor_duration
    max_duration = anchor_config.max_anchor_duration_seconds if max_anchor_duration is None else max_anchor_duration
    duration_seconds = samples.shape[0] / sample_rate
    budget = compute_target_anchor_budget(duration_seconds, anchor_config)
    bin_count = compute_stratified_bin_count(duration_seconds, budget, anchor_config)

    valid_anchors: list[AnchorCandidate] = []
    rejected_counts: dict[str, int] = {}
    for segment in speech_segments:
        duration = segment.end - segment.start
        if duration < min_duration:
            continue

        anchor_start = segment.start

        quality = choose_adaptive_anchor_duration(
            samples=samples,
            sample_rate=sample_rate,
            segment_start=segment.start,
            segment_end=segment.end,
            min_duration=min_duration,
            base_duration=anchor_config.base_anchor_duration_seconds,
            max_duration=max_duration,
            config=anchor_config,
        )
        if quality.rejection_reason is not None:
            rejected_counts[quality.rejection_reason] = rejected_counts.get(quality.rejection_reason, 0) + 1
            continue

        anchor_end = min(segment.end, segment.start + quality.duration_seconds)
        clip = _slice_seconds(samples, sample_rate, anchor_start, anchor_end)
        if clip.size == 0:
            continue

        rms = _rms(clip)
        midpoint = (anchor_start + anchor_end) / 2.0
        valid_anchors.append(
            AnchorCandidate(
                local_start=anchor_start,
                local_end=anchor_end,
                confidence=max(0.0, min(1.0, segment.confidence * quality.confidence_scale)),
                rms=rms,
                bin_index=_assign_bin_index(midpoint, duration_seconds, bin_count),
                snr_db=quality.snr_db,
                spectral_flatness=quality.spectral_flatness,
                quality_multiplier=quality.quality_multiplier,
                duration_seconds=anchor_end - anchor_start,
            )
        )

    selected = _select_stratified_anchors(valid_anchors, budget, bin_count, anchor_config)
    diagnostics = _build_selection_diagnostics(
        valid_anchors=valid_anchors,
        selected_anchors=selected,
        duration_seconds=duration_seconds,
        target_anchor_count=budget,
        bin_count=bin_count,
        config=anchor_config,
        rejected_counts=rejected_counts,
    )
    return AnchorSelectionResult(candidates=selected, diagnostics=diagnostics)


def _select_stratified_anchors(
    anchors: list[AnchorCandidate],
    budget: int | None,
    bin_count: int,
    config: AnchorSelectionConfig,
) -> list[AnchorCandidate]:
    sorted_anchors = sorted(anchors, key=_anchor_quality_key, reverse=True)
    if budget is None:
        return sorted_anchors
    if budget <= 0 or not sorted_anchors:
        return []

    anchors_per_bin = config.anchors_per_bin or int(ceil(budget / max(bin_count, 1)))
    by_bin: list[list[AnchorCandidate]] = [[] for _ in range(bin_count)]
    for anchor in sorted_anchors:
        by_bin[anchor.bin_index or 0].append(anchor)

    per_bin_selected: list[AnchorCandidate] = []
    selected_ids: set[int] = set()

    for bin_anchors in by_bin:
        for anchor in bin_anchors[:anchors_per_bin]:
            per_bin_selected.append(anchor)
            selected_ids.add(id(anchor))

    if len(per_bin_selected) >= budget:
        return sorted(per_bin_selected, key=_anchor_quality_key, reverse=True)[:budget]

    selected = list(per_bin_selected)
    for anchor in sorted_anchors:
        if id(anchor) in selected_ids:
            continue
        selected.append(anchor)
        selected_ids.add(id(anchor))
        if len(selected) >= budget:
            break

    return selected


def _build_selection_diagnostics(
    valid_anchors: list[AnchorCandidate],
    selected_anchors: list[AnchorCandidate],
    duration_seconds: float,
    target_anchor_count: int | None,
    bin_count: int,
    config: AnchorSelectionConfig,
    rejected_counts: dict[str, int] | None = None,
) -> AnchorSelectionDiagnostics:
    bin_width = duration_seconds / bin_count if duration_seconds > 0 else 0.0
    bin_summaries: list[AnchorSelectionBinSummary] = []
    selected_ids = {id(anchor) for anchor in selected_anchors}
    selected_counts = [0 for _ in range(bin_count)]
    candidate_counts = [0 for _ in range(bin_count)]
    for anchor in valid_anchors:
        candidate_counts[anchor.bin_index or 0] += 1
        if id(anchor) in selected_ids:
            selected_counts[anchor.bin_index or 0] += 1

    for index in range(bin_count):
        start = index * bin_width
        end = duration_seconds if index == bin_count - 1 else (index + 1) * bin_width
        bin_summaries.append(
            AnchorSelectionBinSummary(
                index=index,
                start_seconds=start,
                end_seconds=end,
                candidate_count=candidate_counts[index],
                selected_count=selected_counts[index],
            )
        )

    sparse_bin_count = sum(1 for count in selected_counts if count == 0)
    longest_unanchored_span = _compute_longest_unanchored_span_seconds(selected_anchors, duration_seconds)
    anchors_per_bin = config.anchors_per_bin
    if anchors_per_bin is None and target_anchor_count is not None:
        anchors_per_bin = int(ceil(target_anchor_count / max(bin_count, 1)))

    warnings = _build_coverage_warnings(
        selected_anchors=selected_anchors,
        duration_seconds=duration_seconds,
        bin_count=bin_count,
        sparse_bin_count=sparse_bin_count,
        longest_unanchored_span_seconds=longest_unanchored_span,
    )

    selected_durations = [anchor.duration_seconds for anchor in selected_anchors if anchor.duration_seconds is not None]

    return AnchorSelectionDiagnostics(
        candidate_anchor_count=len(valid_anchors),
        selected_anchor_count=len(selected_anchors),
        target_anchor_count=target_anchor_count,
        stratified_bin_count=bin_count,
        anchors_per_bin=anchors_per_bin,
        longest_unanchored_span_seconds=longest_unanchored_span,
        sparse_bin_count=sparse_bin_count,
        adaptive_duration_min_seconds=None if not selected_durations else float(np.min(selected_durations)),
        adaptive_duration_median_seconds=None if not selected_durations else float(np.median(selected_durations)),
        adaptive_duration_max_seconds=None if not selected_durations else float(np.max(selected_durations)),
        rejected_candidate_counts={} if rejected_counts is None else dict(sorted(rejected_counts.items())),
        bins=bin_summaries,
        warnings=warnings,
    )


def _build_coverage_warnings(
    selected_anchors: list[AnchorCandidate],
    duration_seconds: float,
    bin_count: int,
    sparse_bin_count: int,
    longest_unanchored_span_seconds: float,
) -> list[AnchorSelectionCoverageWarning]:
    if not selected_anchors:
        return []

    warnings: list[AnchorSelectionCoverageWarning] = []
    selected_bin_count = len({anchor.bin_index for anchor in selected_anchors})
    if bin_count >= 3 and selected_bin_count < 3:
        warnings.append(
            AnchorSelectionCoverageWarning(
                code="SPARSE_ANCHOR_COVERAGE",
                message="Selected drift anchors are concentrated in too few timeline regions; inspect alignment manually.",
            )
        )

    poor_span_threshold = max(120.0, duration_seconds * 0.5)
    if duration_seconds >= 120.0 and longest_unanchored_span_seconds > poor_span_threshold:
        warnings.append(
            AnchorSelectionCoverageWarning(
                code="LONG_UNANCHORED_SPAN",
                message="A long section of the local timeline has no selected drift anchors; inspect alignment manually.",
                time_seconds=_first_large_gap_start(selected_anchors, duration_seconds, poor_span_threshold),
            )
        )

    if bin_count >= 3 and sparse_bin_count == bin_count - selected_bin_count:
        # Kept for transparent diagnostics; the warning above carries the report-facing issue.
        pass

    return warnings


def _compute_longest_unanchored_span_seconds(anchors: list[AnchorCandidate], duration_seconds: float) -> float:
    if duration_seconds <= 0:
        return 0.0
    midpoints = sorted((anchor.local_start + anchor.local_end) / 2.0 for anchor in anchors)
    if not midpoints:
        return duration_seconds
    gaps = [max(0.0, midpoints[0])]
    gaps.extend(max(0.0, right - left) for left, right in zip(midpoints, midpoints[1:]))
    gaps.append(max(0.0, duration_seconds - midpoints[-1]))
    return float(max(gaps))


def _first_large_gap_start(anchors: list[AnchorCandidate], duration_seconds: float, threshold_seconds: float) -> float | None:
    midpoints = sorted((anchor.local_start + anchor.local_end) / 2.0 for anchor in anchors)
    previous = 0.0
    for midpoint in midpoints:
        if midpoint - previous > threshold_seconds:
            return previous
        previous = midpoint
    if duration_seconds - previous > threshold_seconds:
        return previous
    return None


@dataclass(frozen=True)
class AnchorQualityMetrics:
    duration_seconds: float
    snr_db: float | None
    spectral_flatness: float
    quality_multiplier: float
    confidence_scale: float
    rejection_reason: str | None = None


def choose_adaptive_anchor_duration(
    samples: np.ndarray,
    sample_rate: int,
    segment_start: float,
    segment_end: float,
    min_duration: float,
    base_duration: float,
    max_duration: float,
    config: AnchorSelectionConfig,
) -> AnchorQualityMetrics:
    """Choose a bounded anchor duration from local SNR and spectral flatness.

    Adaptation affects matching evidence only: weak or flat material asks for a
    longer bounded clip and may be downgraded or rejected, while regression
    weights continue to come from match confidence rather than duration alone.
    """

    available_duration = max(0.0, segment_end - segment_start)
    if available_duration < min_duration:
        return AnchorQualityMetrics(
            duration_seconds=available_duration,
            snr_db=None,
            spectral_flatness=1.0,
            quality_multiplier=1.0,
            confidence_scale=0.0,
            rejection_reason="too_short",
        )

    analysis_end = min(segment_end, segment_start + min(max_duration, available_duration))
    quality_clip = _slice_seconds(samples, sample_rate, segment_start, analysis_end)
    if quality_clip.size == 0:
        return AnchorQualityMetrics(
            duration_seconds=available_duration,
            snr_db=None,
            spectral_flatness=1.0,
            quality_multiplier=1.0,
            confidence_scale=0.0,
            rejection_reason="empty_analysis_window",
        )

    snr_db = compute_local_snr_db(
        samples,
        sample_rate,
        segment_start,
        analysis_end,
        noise_context_start=segment_start,
        noise_context_end=segment_end,
    )
    spectral_flatness = compute_spectral_flatness(quality_clip)

    if config.min_snr_db is not None and snr_db is not None and snr_db < config.min_snr_db:
        return AnchorQualityMetrics(
            duration_seconds=available_duration,
            snr_db=snr_db,
            spectral_flatness=spectral_flatness,
            quality_multiplier=1.0,
            confidence_scale=0.0,
            rejection_reason="low_snr",
        )
    if config.spectral_flatness_threshold is not None and spectral_flatness > config.spectral_flatness_threshold:
        return AnchorQualityMetrics(
            duration_seconds=available_duration,
            snr_db=snr_db,
            spectral_flatness=spectral_flatness,
            quality_multiplier=1.0,
            confidence_scale=0.0,
            rejection_reason="spectrally_flat",
        )

    snr_multiplier = _snr_duration_multiplier(snr_db)
    flatness_multiplier = _flatness_duration_multiplier(spectral_flatness)
    quality_multiplier = max(snr_multiplier, flatness_multiplier)
    requested_duration = max(min_duration, min(max_duration, base_duration * quality_multiplier))
    duration = min(available_duration, requested_duration)

    confidence_scale = 1.0
    if snr_db is not None:
        if snr_db < 6.0:
            confidence_scale *= 0.65
        elif snr_db < 12.0:
            confidence_scale *= 0.85
    if spectral_flatness > 0.75:
        confidence_scale *= 0.65
    elif spectral_flatness > 0.45:
        confidence_scale *= 0.85
    if duration < requested_duration:
        confidence_scale *= max(0.5, duration / requested_duration)

    return AnchorQualityMetrics(
        duration_seconds=duration,
        snr_db=snr_db,
        spectral_flatness=spectral_flatness,
        quality_multiplier=quality_multiplier,
        confidence_scale=confidence_scale,
    )


def compute_local_snr_db(
    samples: np.ndarray,
    sample_rate: int,
    segment_start: float,
    segment_end: float,
    noise_context_start: float | None = None,
    noise_context_end: float | None = None,
) -> float | None:
    """Estimate signal-to-nearby-noise ratio in dB for a speech segment.

    ``segment_start`` and ``segment_end`` bound the signal window. Optional
    noise-context bounds keep surrounding speech from being reused as the local
    noise estimate for a shorter anchor clipped from a longer VAD segment.
    Returns ``None`` when no surrounding context exists so clean full-file or
    continuous-speech segments are not treated as their own noise floor.
    """

    signal = _slice_seconds(samples, sample_rate, segment_start, segment_end)
    if signal.size == 0:
        return None
    signal_rms = _rms(signal)
    context_seconds = min(max(segment_end - segment_start, 0.25), 1.0)
    context_start = segment_start if noise_context_start is None else noise_context_start
    context_end = segment_end if noise_context_end is None else noise_context_end
    before = _slice_seconds(samples, sample_rate, max(0.0, context_start - context_seconds), context_start)
    after = _slice_seconds(samples, sample_rate, context_end, min(samples.shape[0] / sample_rate, context_end + context_seconds))
    if before.size and after.size:
        noise = np.concatenate([before, after])
    elif before.size:
        noise = before
    elif after.size:
        noise = after
    else:
        return None
    noise_rms = max(_rms(noise), 1e-9)
    return float(20.0 * np.log10(max(signal_rms, 1e-9) / noise_rms))


def compute_spectral_flatness(samples: np.ndarray) -> float:
    """Compute spectral flatness where 0 is tonal and 1 is noise-like."""

    if samples.size == 0:
        return 1.0
    centered = samples.astype(np.float64) - float(np.mean(samples))
    if not np.any(centered):
        return 1.0
    power = np.abs(np.fft.rfft(centered)) ** 2
    if power.size <= 1:
        return 1.0
    power = power[1:] + 1e-18
    geometric_mean = float(np.exp(np.mean(np.log(power))))
    arithmetic_mean = float(np.mean(power))
    if arithmetic_mean <= 0.0:
        return 1.0
    return float(max(0.0, min(1.0, geometric_mean / arithmetic_mean)))


def _snr_duration_multiplier(snr_db: float | None) -> float:
    if snr_db is None:
        return 1.0
    if snr_db >= 24.0:
        return 1.0
    if snr_db >= 12.0:
        return 1.0 + ((24.0 - snr_db) / 12.0) * 0.5
    if snr_db >= 6.0:
        return 1.5 + ((12.0 - snr_db) / 6.0) * 0.5
    return 2.0


def _flatness_duration_multiplier(spectral_flatness: float) -> float:
    if spectral_flatness <= 0.25:
        return 1.0
    if spectral_flatness <= 0.65:
        return 1.0 + ((spectral_flatness - 0.25) / 0.40) * 0.5
    return 2.0


def _slice_seconds(samples: np.ndarray, sample_rate: int, start_seconds: float, end_seconds: float) -> np.ndarray:
    start_index = max(0, int(start_seconds * sample_rate))
    end_index = max(start_index, min(samples.shape[0], int(end_seconds * sample_rate)))
    return samples[start_index:end_index]


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) * samples.astype(np.float64)) + 1e-18))


def _assign_bin_index(midpoint_seconds: float, duration_seconds: float, bin_count: int) -> int:
    if bin_count <= 1 or duration_seconds <= 0:
        return 0
    raw_index = int((midpoint_seconds / duration_seconds) * bin_count)
    return max(0, min(bin_count - 1, raw_index))


def _anchor_quality_key(anchor: AnchorCandidate) -> tuple[float, float, float, float, float]:
    snr_component = 0.0 if anchor.snr_db is None else anchor.snr_db
    flatness_component = 1.0 if anchor.spectral_flatness is None else -anchor.spectral_flatness
    return (anchor.confidence, snr_component, flatness_component, anchor.rms, -(anchor.local_start))
