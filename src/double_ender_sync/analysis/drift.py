from dataclasses import InitVar, dataclass, field
import logging
from numbers import Real
from typing import Any, Protocol, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator

from double_ender_sync.analysis.anchors import AnchorCandidate
from double_ender_sync.analysis.features import (
    NccPeakDiagnostics,
    clamp01,
    extract_anchor_feature,
    gcc_phat_scores,
    ncc_peak_diagnostics,
    normalized_correlation_scores,
)
from double_ender_sync.config import (
    AnchorMatchingConfig,
    DEFAULT_INITIAL_OFFSET_SAFETY_CONFIG,
    DriftModelConfig,
    InitialOffsetSafetyConfig,
)
from double_ender_sync.analysis.vad import SpeechSegment


LOGGER = logging.getLogger("double_ender_sync")
_MAX_KALMAN_COVARIANCE_CONDITION = 1e12


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

    ncc_best_score: float | None = None
    ncc_second_score: float | None = None
    ncc_margin: float | None = None
    ncc_prominence: float | None = None
    ncc_width_seconds: float | None = None
    ncc_plateau_size_seconds: float | None = None
    ncc_peak_lag_seconds: float | None = None
    gcc_phat_peak_lag_seconds: float | None = None
    gcc_phat_agreement_seconds: float | None = None
    match_quality: float | None = None
    match_uniqueness: float | None = None
    match_sharpness: float | None = None
    match_agreement: float | None = None


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


class DriftModel(Protocol):
    """Protocol for drift mappings from local time to master time.

    Phase 1 introduces the interface while keeping the linear implementation as
    the only active/default model. Richer models should implement the same
    observable mapping and report hooks without being forced into linear-only
    parameters.
    """

    model_type: str
    speaker_track: str

    def map_local_to_master(self, local_time_seconds: float) -> float:
        """Map a local-track time in seconds onto the master timeline."""
        ...

    def local_rate_at(self, local_time_seconds: float) -> float:
        """Return the local-to-master rate around a local-track time."""
        ...

    def residuals_ms(self, anchors: Sequence[AnchorMatch]) -> Sequence[float]:
        """Return signed residuals as observed master time minus predicted master time.

        Positive values mean an anchor was observed later on the master
        timeline than this model predicts. This matches ``AnchorMatch.residual_ms``
        from linear drift fitting.
        """
        ...

    def to_report_dict(self) -> dict[str, object]:
        """Return a report-friendly model description."""
        ...


@dataclass
class LinearDrift:
    """Linear drift model preserving the original stretch/offset behavior."""

    offset_seconds: float
    stretch_ratio: float
    anchor_count: int
    residual_median_ms: float
    residual_max_ms: float
    diagnostics: DriftFitDiagnostics | None = None
    speaker_track: str = ""
    model_type: str = "linear"
    model_version: str = "1"
    model_selection_policy: str = "linear_default"
    candidate_models: tuple[str, ...] = ("linear",)
    selected_model_reason: str = "linear is the default control model"
    fallback_reason: str | None = None
    unsupported_regions: tuple[dict[str, object], ...] = ()
    warnings: tuple[DriftFitWarning, ...] = ()

    def map_local_to_master(self, local_time_seconds: float) -> float:
        return (self.stretch_ratio * local_time_seconds) + self.offset_seconds

    def local_rate_at(self, local_time_seconds: float) -> float:
        return self.stretch_ratio

    def residuals_ms(self, anchors: Sequence[AnchorMatch]) -> Sequence[float]:
        return [
            1000.0 * (anchor.master_start - self.map_local_to_master(anchor.local_start))
            for anchor in anchors
        ]

    def to_report_dict(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "model_version": self.model_version,
            "model_selection_policy": self.model_selection_policy,
            "candidate_models": list(self.candidate_models),
            "selected_model_reason": self.selected_model_reason,
            "fallback_reason": self.fallback_reason,
            "model_parameters": {
                "offset_seconds": self.offset_seconds,
                "stretch_ratio": self.stretch_ratio,
            },
            "offset_seconds": self.offset_seconds,
            "stretch_ratio": self.stretch_ratio,
            "anchor_count": self.anchor_count,
            "residual_median_ms": self.residual_median_ms,
            "residual_max_ms": self.residual_max_ms,
            "local_rate_summary": {
                "min": self.stretch_ratio,
                "max": self.stretch_ratio,
                "mean": self.stretch_ratio,
            },
            "monotonicity_check": {
                "passed": self.stretch_ratio > 0.0,
            },
            "breakpoints": [],
            "knots": [],
            "unsupported_regions": list(self.unsupported_regions),
            "warnings": [
                {"code": warning.code, "message": warning.message, "time_seconds": warning.time_seconds}
                for warning in self.warnings
            ],
        }

@dataclass
class PiecewiseLinearSegment:
    local_start: float
    local_end: float
    master_start: float
    master_end: float
    stretch_ratio: float
    offset_seconds: float
    anchor_count: int
    residual_median_ms: float
    residual_max_ms: float

    def contains(self, local_time_seconds: float) -> bool:
        return self.local_start <= local_time_seconds <= self.local_end


@dataclass
class PiecewiseLinearDrift:
    """Continuous piecewise-linear local-to-master drift model."""

    breakpoints: tuple[float, ...]
    segments: tuple[PiecewiseLinearSegment, ...]
    anchor_count: int
    residual_median_ms: float
    residual_max_ms: float
    linear_baseline: LinearDrift
    diagnostics: DriftFitDiagnostics | None = None
    speaker_track: str = ""
    model_type: str = "piecewise_linear"
    model_version: str = "1"
    model_selection_policy: str = "piecewise_experimental"
    candidate_models: tuple[str, ...] = ("linear", "piecewise_linear")
    selected_model_reason: str = "piecewise residuals materially improve over the linear control model"
    fallback_reason: str | None = None
    warnings: tuple[DriftFitWarning, ...] = ()
    unsupported_regions: tuple[dict[str, object], ...] = ()

    @property
    def offset_seconds(self) -> float:
        return self.linear_baseline.offset_seconds

    @property
    def stretch_ratio(self) -> float:
        return self.linear_baseline.stretch_ratio

    def map_local_to_master(self, local_time_seconds: float) -> float:
        segment = self._segment_for_time(local_time_seconds)
        return (segment.stretch_ratio * local_time_seconds) + segment.offset_seconds

    def local_rate_at(self, local_time_seconds: float) -> float:
        return self._segment_for_time(local_time_seconds).stretch_ratio

    def residuals_ms(self, anchors: Sequence[AnchorMatch]) -> Sequence[float]:
        return [
            1000.0 * (anchor.master_start - self.map_local_to_master(anchor.local_start))
            for anchor in anchors
        ]

    def to_report_dict(self) -> dict[str, object]:
        rates = [segment.stretch_ratio for segment in self.segments]
        segment_dicts = [
            {
                "local_start": segment.local_start,
                "local_end": segment.local_end,
                "master_start": segment.master_start,
                "master_end": segment.master_end,
                "stretch_ratio": segment.stretch_ratio,
                "offset_seconds": segment.offset_seconds,
                "anchor_count": segment.anchor_count,
                "residual_median_ms": segment.residual_median_ms,
                "residual_max_ms": segment.residual_max_ms,
            }
            for segment in self.segments
        ]
        return {
            "model_type": self.model_type,
            "model_version": self.model_version,
            "model_selection_policy": self.model_selection_policy,
            "candidate_models": list(self.candidate_models),
            "selected_model_reason": self.selected_model_reason,
            "fallback_reason": self.fallback_reason,
            "model_parameters": {
                "breakpoints": list(self.breakpoints),
                "segments": segment_dicts,
                "linear_baseline": {
                    "offset_seconds": self.linear_baseline.offset_seconds,
                    "stretch_ratio": self.linear_baseline.stretch_ratio,
                    "residual_median_ms": self.linear_baseline.residual_median_ms,
                    "residual_max_ms": self.linear_baseline.residual_max_ms,
                },
            },
            "offset_seconds": self.linear_baseline.offset_seconds,
            "stretch_ratio": self.linear_baseline.stretch_ratio,
            "anchor_count": self.anchor_count,
            "residual_median_ms": self.residual_median_ms,
            "residual_max_ms": self.residual_max_ms,
            "breakpoints": list(self.breakpoints),
            "knots": [],
            "segments": segment_dicts,
            "segment_residual_summaries": segment_dicts,
            "local_rate_summary": {
                "min": min(rates),
                "max": max(rates),
                "mean": float(np.mean(rates)),
            },
            "monotonicity_check": {
                "passed": all(rate > 0.0 for rate in rates),
            },
            "unsupported_regions": list(self.unsupported_regions),
            "warnings": [
                {"code": warning.code, "message": warning.message, "time_seconds": warning.time_seconds}
                for warning in self.warnings
            ],
        }

    def _segment_for_time(self, local_time_seconds: float) -> PiecewiseLinearSegment:
        if local_time_seconds <= self.segments[0].local_start:
            return self.segments[0]
        for segment in self.segments:
            if segment.contains(local_time_seconds):
                return segment
        return self.segments[-1]


@dataclass
class SplineDrift:
    """Monotonic cubic PCHIP local-to-master drift model."""

    knot_local_times: tuple[float, ...]
    knot_master_times: tuple[float, ...]
    interpolation_method: str
    knot_source: str
    anchor_count: int
    residual_median_ms: float
    residual_max_ms: float
    linear_baseline: LinearDrift
    baseline_model_type: str
    validation_sample_count: int
    monotonicity_min_step_seconds: float
    local_rate_min: float
    local_rate_max: float
    local_rate_mean: float
    local_rate_change_max_ppm: float
    diagnostics: DriftFitDiagnostics | None = None
    speaker_track: str = ""
    model_type: str = "spline"
    model_version: str = "1"
    model_selection_policy: str = "spline_experimental"
    candidate_models: tuple[str, ...] = ("linear", "spline")
    selected_model_reason: str = "spline residuals materially improve over the simpler control model"
    fallback_reason: str | None = None
    warnings: tuple[DriftFitWarning, ...] = ()
    unsupported_regions: tuple[dict[str, object], ...] = ()
    knot_residual_summaries: tuple[dict[str, object], ...] = ()
    knot_decimation_applied: bool = False
    pchip_interpolator: InitVar[PchipInterpolator | None] = None
    pchip_derivative: InitVar[Any | None] = None

    def __post_init__(
        self,
        pchip_interpolator: PchipInterpolator | None,
        pchip_derivative: Any | None,
    ) -> None:
        if len(self.knot_local_times) != len(self.knot_master_times):
            raise ValueError(
                "SplineDrift requires equal knot counts "
                f"(local={len(self.knot_local_times)}, master={len(self.knot_master_times)})"
            )
        if len(self.knot_local_times) < 2:
            raise ValueError(f"SplineDrift requires at least 2 knots, got {len(self.knot_local_times)}")
        local = np.array(self.knot_local_times, dtype=np.float64)
        master = np.array(self.knot_master_times, dtype=np.float64)
        if not np.all(np.isfinite(local)) or not np.all(np.isfinite(master)):
            raise ValueError("SplineDrift knots must be finite local and master times")
        if np.any(np.diff(local) <= 0.0):
            raise ValueError(f"SplineDrift local knots must be strictly increasing: {self.knot_local_times}")
        if np.any(np.diff(master) <= 0.0):
            raise ValueError(f"SplineDrift master knots must be strictly increasing: {self.knot_master_times}")
        if pchip_interpolator is None:
            self._interpolator = PchipInterpolator(local, master, extrapolate=True)
            self._derivative = self._interpolator.derivative()
        else:
            self._interpolator = pchip_interpolator
            self._derivative = (
                pchip_derivative if pchip_derivative is not None else pchip_interpolator.derivative()
            )

    @property
    def offset_seconds(self) -> float:
        return self.linear_baseline.offset_seconds

    @property
    def stretch_ratio(self) -> float:
        return self.linear_baseline.stretch_ratio

    def map_local_to_master(self, local_time_seconds: float) -> float:
        return float(self._interpolator(float(local_time_seconds)))

    def local_rate_at(self, local_time_seconds: float) -> float:
        return float(self._derivative(float(local_time_seconds)))

    def residuals_ms(self, anchors: Sequence[AnchorMatch]) -> Sequence[float]:
        return [
            1000.0 * (anchor.master_start - self.map_local_to_master(anchor.local_start))
            for anchor in anchors
        ]

    def to_report_dict(self) -> dict[str, object]:
        knots = [
            {"local_time": local, "master_time": master}
            for local, master in zip(self.knot_local_times, self.knot_master_times)
        ]
        warning_dicts = [
            {"code": warning.code, "message": warning.message, "time_seconds": warning.time_seconds}
            for warning in self.warnings
        ]
        return {
            "model_type": self.model_type,
            "model_version": self.model_version,
            "model_selection_policy": self.model_selection_policy,
            "candidate_models": list(self.candidate_models),
            "selected_model_reason": self.selected_model_reason,
            "fallback_reason": self.fallback_reason,
            "model_parameters": {
                "interpolation_method": self.interpolation_method,
                "knot_source": self.knot_source,
                "knot_decimation_applied": self.knot_decimation_applied,
                "knot_count": len(self.knot_local_times),
                "knots": knots,
                "linear_baseline": {
                    "offset_seconds": self.linear_baseline.offset_seconds,
                    "stretch_ratio": self.linear_baseline.stretch_ratio,
                    "residual_median_ms": self.linear_baseline.residual_median_ms,
                    "residual_max_ms": self.linear_baseline.residual_max_ms,
                },
                "baseline_model_type": self.baseline_model_type,
            },
            "offset_seconds": self.linear_baseline.offset_seconds,
            "stretch_ratio": self.linear_baseline.stretch_ratio,
            "anchor_count": self.anchor_count,
            "residual_median_ms": self.residual_median_ms,
            "residual_max_ms": self.residual_max_ms,
            "breakpoints": [],
            "knots": knots,
            "knot_residual_summaries": list(self.knot_residual_summaries),
            "local_rate_summary": {
                "min": self.local_rate_min,
                "max": self.local_rate_max,
                "mean": self.local_rate_mean,
                "max_change_ppm": self.local_rate_change_max_ppm,
            },
            "monotonicity_check": {
                "passed": True,
                "sample_count": self.validation_sample_count,
                "min_step_seconds": self.monotonicity_min_step_seconds,
            },
            "unsupported_regions": list(self.unsupported_regions),
            "warnings": warning_dicts,
        }


@dataclass
class KalmanStatePoint:
    """Smoothed offset/rate state at one anchor time for research diagnostics."""

    local_time: float
    offset_seconds: float
    rate_deviation: float
    offset_std_ms: float
    rate_std_ppm: float


@dataclass
class KalmanDrift:
    """Research/experimental state-space drift model using RTS-smoothed anchor states.

    State vector units are explicit and reportable:
    ``offset_seconds`` maps to ``master_time - local_time``. The
    ``rate_deviation`` state is retained as Kalman diagnostic output, while the
    renderable mapping rate is derived from the same piecewise-linear offset
    interpolant used by ``map_local_to_master()``. Anchor observations are
    noisy offset measurements derived from ``anchor.master_start -
    anchor.local_start``.
    """

    state_points: tuple[KalmanStatePoint, ...]
    anchor_count: int
    residual_median_ms: float
    residual_max_ms: float
    linear_baseline: LinearDrift
    validation_sample_count: int
    monotonicity_min_step_seconds: float
    local_rate_min: float
    local_rate_max: float
    local_rate_mean: float
    covariance_summary: dict[str, float]
    uncertainty_summary: dict[str, float]
    anchor_residuals_ms: tuple[float, ...]
    diagnostics: DriftFitDiagnostics | None = None
    speaker_track: str = ""
    model_type: str = "kalman"
    model_version: str = "research-1"
    model_selection_policy: str = "kalman_research_experimental"
    candidate_models: tuple[str, ...] = ("linear", "kalman")
    selected_model_reason: str = "Kalman smoother residuals materially improve over the linear control model"
    fallback_reason: str | None = None
    warnings: tuple[DriftFitWarning, ...] = ()
    unsupported_regions: tuple[dict[str, object], ...] = ()
    _state_times: np.ndarray = field(init=False, repr=False)
    _state_offsets: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._state_times = np.array([point.local_time for point in self.state_points], dtype=np.float64)
        self._state_offsets = np.array([point.offset_seconds for point in self.state_points], dtype=np.float64)
        if len(self._state_times) < 2:
            raise ValueError("KalmanDrift requires at least two state points for interpolation")
        if np.any(np.diff(self._state_times) <= 0.0):
            raise ValueError("KalmanDrift state point local times must be strictly increasing")

    @property
    def offset_seconds(self) -> float:
        return self.linear_baseline.offset_seconds

    @property
    def stretch_ratio(self) -> float:
        return self.linear_baseline.stretch_ratio

    def map_local_to_master(self, local_time_seconds: float) -> float:
        return float(local_time_seconds) + self._interpolate_offset(float(local_time_seconds))

    def local_rate_at(self, local_time_seconds: float) -> float:
        return self._interpolate_mapping_rate(float(local_time_seconds))

    def residuals_ms(self, anchors: Sequence[AnchorMatch]) -> Sequence[float]:
        return [
            1000.0 * (anchor.master_start - self.map_local_to_master(anchor.local_start))
            for anchor in anchors
        ]

    def to_report_dict(self) -> dict[str, object]:
        states = [
            {
                "local_time": point.local_time,
                "offset_seconds": point.offset_seconds,
                "local_rate": self.local_rate_at(point.local_time),
                "state_rate_deviation": point.rate_deviation,
                "offset_std_ms": point.offset_std_ms,
                "rate_std_ppm": point.rate_std_ppm,
            }
            for point in self.state_points
        ]
        warning_dicts = [
            {"code": warning.code, "message": warning.message, "time_seconds": warning.time_seconds}
            for warning in self.warnings
        ]
        return {
            "model_type": self.model_type,
            "model_version": self.model_version,
            "model_selection_policy": self.model_selection_policy,
            "candidate_models": list(self.candidate_models),
            "selected_model_reason": self.selected_model_reason,
            "fallback_reason": self.fallback_reason,
            "model_parameters": {
                "state_definition": {
                    "offset_seconds": "master_time - local_time",
                    "rate_deviation": "latent Kalman rate state; mapping local_rate is derived from offset interpolation",
                    "observation": "anchor.master_start - anchor.local_start",
                },
                "smoother": "linear_gaussian_forward_filter_rts_backward_smoother",
                "state_count": len(self.state_points),
                "linear_baseline": {
                    "offset_seconds": self.linear_baseline.offset_seconds,
                    "stretch_ratio": self.linear_baseline.stretch_ratio,
                    "residual_median_ms": self.linear_baseline.residual_median_ms,
                    "residual_max_ms": self.linear_baseline.residual_max_ms,
                },
            },
            "offset_seconds": self.linear_baseline.offset_seconds,
            "stretch_ratio": self.linear_baseline.stretch_ratio,
            "anchor_count": self.anchor_count,
            "residual_median_ms": self.residual_median_ms,
            "residual_max_ms": self.residual_max_ms,
            "breakpoints": [],
            "knots": [],
            "state_points": states,
            "local_rate_summary": {
                "min": self.local_rate_min,
                "max": self.local_rate_max,
                "mean": self.local_rate_mean,
            },
            "monotonicity_check": {
                "passed": True,
                "sample_count": self.validation_sample_count,
                "min_step_seconds": self.monotonicity_min_step_seconds,
            },
            "uncertainty_summary": self.uncertainty_summary,
            "covariance_summary": self.covariance_summary,
            "uncertainty_bands": self._uncertainty_bands(),
            "anchor_residuals_ms": list(self.anchor_residuals_ms),
            "unsupported_regions": list(self.unsupported_regions),
            "warnings": warning_dicts,
        }

    def _uncertainty_bands(self) -> list[dict[str, float]]:
        return [
            {
                "local_time": point.local_time,
                "predicted_master_time": point.local_time + point.offset_seconds,
                "offset_seconds": point.offset_seconds,
                "offset_lower_seconds": point.offset_seconds - point.offset_std_ms / 1000.0,
                "offset_upper_seconds": point.offset_seconds + point.offset_std_ms / 1000.0,
                "master_time_lower": point.local_time + point.offset_seconds - point.offset_std_ms / 1000.0,
                "master_time_upper": point.local_time + point.offset_seconds + point.offset_std_ms / 1000.0,
                "sigma": 1.0,
            }
            for point in self.state_points
        ]

    def _interpolate_offset(self, local_time_seconds: float) -> float:
        return _interpolate_piecewise_linear_offset(local_time_seconds, self._state_times, self._state_offsets)

    def _interpolate_mapping_rate(self, local_time_seconds: float) -> float:
        return _interpolate_piecewise_linear_mapping_rate(local_time_seconds, self._state_times, self._state_offsets)


def _interpolate_piecewise_linear_offset(
    local_time_seconds: float,
    local_times: np.ndarray,
    offsets: np.ndarray,
) -> float:
    if len(local_times) < 2:
        return float("nan")
    local_time = float(local_time_seconds)
    if local_time <= float(local_times[0]):
        slope = _piecewise_offset_slope(0, local_times, offsets)
        return float(offsets[0] + slope * (local_time - float(local_times[0])))
    if local_time >= float(local_times[-1]):
        slope = _piecewise_offset_slope(len(local_times) - 2, local_times, offsets)
        return float(offsets[-1] + slope * (local_time - float(local_times[-1])))
    return float(np.interp(local_time, local_times, offsets))


def _interpolate_piecewise_linear_mapping_rate(
    local_time_seconds: float,
    local_times: np.ndarray,
    offsets: np.ndarray,
) -> float:
    if len(local_times) < 2:
        return float("nan")
    local_time = float(local_time_seconds)
    if local_time <= float(local_times[0]):
        offset_slope = _piecewise_offset_slope(0, local_times, offsets)
    elif local_time >= float(local_times[-1]):
        offset_slope = _piecewise_offset_slope(len(local_times) - 2, local_times, offsets)
    else:
        right_index = int(np.searchsorted(local_times, local_time, side="right"))
        offset_slope = _piecewise_offset_slope(right_index - 1, local_times, offsets)
    return float(1.0 + offset_slope)


def _piecewise_offset_slope(left_index: int, local_times: np.ndarray, offsets: np.ndarray) -> float:
    right_index = left_index + 1
    dt = float(local_times[right_index] - local_times[left_index])
    if dt <= 0.0:
        return float("nan")
    return float((offsets[right_index] - offsets[left_index]) / dt)


@dataclass(frozen=True)
class KalmanFitResult:
    model: KalmanDrift | None
    fallback_reason: str | None


@dataclass(frozen=True)
class SplineFitResult:
    model: SplineDrift | None
    fallback_reason: str | None


@dataclass(frozen=True)
class SplineKnotSelectionResult:
    knot_local_times: list[float] | None
    knot_master_times: list[float] | None
    knot_source: str | None
    fallback_reason: str | None = None
    decimation_applied: bool = False


@dataclass(frozen=True)
class PiecewiseLinearFitResult:
    model: PiecewiseLinearDrift | None
    fallback_reason: str | None



# Backward-compatible public name for callers/tests that still import the old
# linear-only estimate type. New code should prefer LinearDrift or DriftModel.
DriftEstimate = LinearDrift


def _compute_ncc_continuous_confidence(
    diagnostics: object,
    config: AnchorMatchingConfig,
    sample_rate: int,
) -> tuple[float, float, float]:
    ncc_score_denom = 1.0 - config.ncc_min_score
    if ncc_score_denom <= 1e-15:
        quality = 1.0 if diagnostics.best_score >= config.ncc_min_score else 0.0
    else:
        quality = clamp01(
            (diagnostics.best_score - config.ncc_min_score) / ncc_score_denom
        )

    margin = diagnostics.margin
    prominence = diagnostics.prominence

    prom_range = config.ncc_prominence_high - config.ncc_prominence_low

    def _safe_prominence_uniqueness(prom: float) -> float:
        if prom_range <= 1e-15:
            return 1.0 if prom >= config.ncc_prominence_high else 0.0
        return clamp01((prom - config.ncc_prominence_low) / prom_range)

    def _safe_margin_uniqueness(marg: float) -> float:
        margin_low_shifted = config.ncc_margin_low * 0.4
        margin_range = config.ncc_margin_high - margin_low_shifted
        if margin_range <= 1e-15:
            return 1.0 if marg >= margin_low_shifted else 0.0
        return clamp01((marg - margin_low_shifted) / margin_range)

    def _safe_margin_uniqueness_full(marg: float) -> float:
        margin_range = config.ncc_margin_high - config.ncc_margin_low
        if margin_range <= 1e-15:
            return 1.0 if marg >= config.ncc_margin_low else 0.0
        return clamp01((marg - config.ncc_margin_low) / margin_range)

    if margin is None:
        uniqueness = _safe_prominence_uniqueness(prominence)
    elif prominence >= config.ncc_prominence_high:
        uniqueness = min(
            _safe_margin_uniqueness(margin),
            _safe_prominence_uniqueness(prominence),
        )
    else:
        uniqueness = min(
            _safe_margin_uniqueness_full(margin),
            _safe_prominence_uniqueness(prominence),
        )

    width_seconds = diagnostics.width_samples / max(1.0, float(sample_rate))

    width_range = config.ncc_bad_width_seconds - config.ncc_good_width_seconds
    if width_range <= 1e-15:
        sharpness = 1.0 if width_seconds <= config.ncc_good_width_seconds else 0.0
    else:
        sharpness = 1.0 - clamp01(
            (width_seconds - config.ncc_good_width_seconds) / width_range
        )

    return quality, uniqueness, sharpness


def _match_diagnostics_passes_hard_gate(
    diagnostics: object,
    config: AnchorMatchingConfig,
) -> tuple[bool, str | None]:
    if diagnostics.best_score < config.ncc_min_score:
        return False, "ncc_best_score_below_min"

    margin = diagnostics.margin
    if margin is not None and margin < config.ncc_min_margin:
        return False, "ncc_margin_below_min"

    if diagnostics.prominence < config.ncc_min_prominence:
        return False, "ncc_prominence_below_min"

    return True, None


def match_anchors_for_drift(
    local_samples: np.ndarray,
    master_samples: np.ndarray,
    sample_rate: int,
    anchors: list[AnchorCandidate],
    initial_offset_seconds: float,
    search_radius_seconds: float = 6.0,
    matching_config: AnchorMatchingConfig | None = None,
    master_speech_segments: list[SpeechSegment] | None = None,
    safety_config: InitialOffsetSafetyConfig | None = None,
) -> list[AnchorMatch]:
    if matching_config is None:
        from double_ender_sync.config import DEFAULT_ANCHOR_MATCHING_CONFIG
        matching_config = DEFAULT_ANCHOR_MATCHING_CONFIG
    if safety_config is None:
        safety_config = DEFAULT_INITIAL_OFFSET_SAFETY_CONFIG

    matches: list[AnchorMatch] = []
    master_duration = master_samples.shape[0] / sample_rate

    vad_filter_enabled = safety_config.master_vad_filter_enabled
    # Treat both an empty segment list and None as uncertain: VAD may have
    # missed attenuated speech on a quiet master, so we rely on the configured
    # uncertain policy rather than confidently rejecting every match.
    vad_available = (
        master_speech_segments is not None and len(master_speech_segments) > 0
    )

    if vad_filter_enabled and not vad_available and safety_config.master_vad_uncertain_policy == "warn":
        LOGGER.warning(
            "master_vad_unavailable: drift-anchor matching continues without master VAD filtering"
        )

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

        if vad_filter_enabled and not vad_available and safety_config.master_vad_uncertain_policy == "reject":
            # VAD data is unavailable and policy is to reject. Clamp the
            # expected span to the master timeline so diagnostics never carry
            # negative or out-of-range timestamps.
            clamped_ms = max(0.0, min(master_duration, expected_master_start))
            clamped_me = max(0.0, min(master_duration, expected_master_start + (local_clip.size / sample_rate)))
            matches.append(
                AnchorMatch(
                    local_start=anchor.local_start,
                    local_end=anchor.local_end,
                    master_start=clamped_ms,
                    master_end=clamped_me,
                    offset_seconds=clamped_ms - anchor.local_start,
                    confidence=0.0,
                    score=0.0,
                    rejected_reason="master_vad_unavailable",
                )
            )
            continue
        # "warn" and "skip" policies, and the vad_available path, fall through
        # to NCC matching. When VAD is available the overlap check runs
        # post-NCC on the actual matched span (see below).

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

        peak_diagnostics = ncc_peak_diagnostics(
            scores,
            sample_rate=sample_rate,
            nms_exclusion_seconds=matching_config.nms_exclusion_seconds,
        )

        if peak_diagnostics is None:
            continue

        # Post-NCC master VAD check: measure speech overlap at the actual
        # matched span. If the best peak lands outside speech, try lower
        # peaks before giving up on the anchor.
        vad_rejected_reason: str | None = None
        if vad_filter_enabled and vad_available:
            peak_diagnostics, vad_rejected_reason = _select_vad_valid_peak(
                scores=scores,
                initial_diagnostics=peak_diagnostics,
                sample_rate=sample_rate,
                search_start_idx=search_start_idx,
                local_clip_size=local_clip.size,
                matching_config=matching_config,
                safety_config=safety_config,
                master_speech_segments=master_speech_segments,
            )

        best_score = peak_diagnostics.best_score
        best_lag_samples = peak_diagnostics.best_lag_samples
        second_score = peak_diagnostics.second_score
        margin = peak_diagnostics.margin
        prominence = peak_diagnostics.prominence
        width_samples = peak_diagnostics.width_samples
        plateau_size_samples = peak_diagnostics.plateau_size_samples

        ncc_width_seconds = width_samples / max(1.0, float(sample_rate))
        ncc_plateau_size_seconds = plateau_size_samples / max(1.0, float(sample_rate))
        ncc_peak_lag_seconds = best_lag_samples / sample_rate

        master_start = (search_start_idx + best_lag_samples) / sample_rate
        master_end = master_start + (local_clip.size / sample_rate)
        offset_seconds = master_start - anchor.local_start

        if vad_rejected_reason is not None:
            # Preserve NCC diagnostics so the report shows the actual match
            # strength even though master VAD rejected it.
            vad_quality, vad_uniqueness, vad_sharpness = _compute_ncc_continuous_confidence(
                peak_diagnostics, matching_config, sample_rate
            )
            matches.append(
                AnchorMatch(
                    local_start=anchor.local_start,
                    local_end=anchor.local_end,
                    master_start=master_start,
                    master_end=master_end,
                    offset_seconds=offset_seconds,
                    confidence=0.0,
                    score=best_score,
                    rejected_reason=vad_rejected_reason,
                    ncc_best_score=best_score,
                    ncc_second_score=second_score,
                    ncc_margin=margin,
                    ncc_prominence=prominence,
                    ncc_width_seconds=ncc_width_seconds,
                    ncc_plateau_size_seconds=ncc_plateau_size_seconds,
                    ncc_peak_lag_seconds=ncc_peak_lag_seconds,
                    match_quality=vad_quality,
                    match_uniqueness=vad_uniqueness,
                    match_sharpness=vad_sharpness,
                    match_agreement=1.0,
                )
            )
            LOGGER.debug(
                "drift-anchor-rejected local_start=%.3f matched_master=[%.3f, %.3f]s "
                "ncc_best=%.4f ncc_margin=%s reason=%s",
                anchor.local_start,
                master_start,
                master_end,
                best_score,
                f"{margin:.4f}" if margin is not None else "None",
                vad_rejected_reason,
            )
            continue

        passes_gate, rejection_reason = _match_diagnostics_passes_hard_gate(
            peak_diagnostics, matching_config
        )

        legacy_score = best_score
        quality, uniqueness, sharpness = _compute_ncc_continuous_confidence(
            peak_diagnostics, matching_config, sample_rate
        )
        agreement = 1.0
        gcc_lag_seconds: float | None = None
        gcc_agreement_seconds: float | None = None
        ncc_is_ambiguous = not passes_gate or (
            margin is not None
            and margin < matching_config.ncc_min_margin * 2.0
            and prominence < matching_config.ncc_min_prominence * 5.0
        ) or prominence < matching_config.ncc_min_prominence * 2.0

        if matching_config.gcc_phat_enabled and (
            not matching_config.gcc_phat_only_when_ambiguous or ncc_is_ambiguous
        ):
            try:
                gcc_scores = gcc_phat_scores(search_region, feature)
                if gcc_scores.size > 0:
                    gcc_best_lag = int(np.argmax(gcc_scores))
                    gcc_best_value = float(gcc_scores[gcc_best_lag])
                    gcc_lag_seconds = gcc_best_lag / sample_rate
                    gcc_agreement_seconds = abs(ncc_peak_lag_seconds - gcc_lag_seconds)
                    if gcc_best_value >= 0.05 and gcc_agreement_seconds <= matching_config.gcc_phat_agreement_tolerance_seconds * 3.0:
                        agreement = clamp01(
                            1.0
                            - gcc_agreement_seconds
                            / matching_config.gcc_phat_agreement_tolerance_seconds
                        )
                    else:
                        gcc_lag_seconds = None
                        gcc_agreement_seconds = None
                        agreement = 0.0
            except Exception:
                LOGGER.debug(
                    "drift-anchor gcc-phat failed for local_start=%.3f; using agreement=1.0",
                    anchor.local_start,
                )

        anchor_confidence = getattr(anchor, "confidence", 1.0)
        confidence_value = (
            anchor_confidence * quality * uniqueness * sharpness * agreement
        )
        confidence_value = max(0.0, min(1.0, confidence_value))

        if not passes_gate or confidence_value < matching_config.min_confidence_for_fit:
            rejection_reason = rejection_reason or "confidence_below_min_for_fit"

        match = AnchorMatch(
            local_start=anchor.local_start,
            local_end=anchor.local_end,
            master_start=master_start,
            master_end=master_end,
            offset_seconds=offset_seconds,
            confidence=confidence_value,
            score=legacy_score,
            rejected_reason=rejection_reason,
            ncc_best_score=best_score,
            ncc_second_score=second_score,
            ncc_margin=margin,
            ncc_prominence=prominence,
            ncc_width_seconds=ncc_width_seconds,
            ncc_plateau_size_seconds=ncc_plateau_size_seconds,
            ncc_peak_lag_seconds=ncc_peak_lag_seconds,
            gcc_phat_peak_lag_seconds=gcc_lag_seconds,
            gcc_phat_agreement_seconds=gcc_agreement_seconds,
            match_quality=quality,
            match_uniqueness=uniqueness,
            match_sharpness=sharpness,
            match_agreement=agreement,
        )
        matches.append(match)

        margin_str = f"{margin:.4f}" if margin is not None else "None"
        LOGGER.debug(
            "drift-anchor-result local_start=%.3f master_start=%.3f "
            "ncc_best=%.4f ncc_margin=%s ncc_prom=%.4f "
            "quality=%.4f uniqueness=%.4f sharpness=%.4f agreement=%.4f "
            "confidence=%.4f rejected=%s",
            anchor.local_start,
            master_start,
            best_score,
            margin_str,
            prominence,
            quality,
            uniqueness,
            sharpness,
            agreement,
            confidence_value,
            str(rejection_reason) if rejection_reason else "None",
        )

    return matches


def _select_vad_valid_peak(
    scores: np.ndarray,
    initial_diagnostics: NccPeakDiagnostics,
    sample_rate: int,
    search_start_idx: int,
    local_clip_size: int,
    matching_config: AnchorMatchingConfig,
    safety_config: InitialOffsetSafetyConfig,
    master_speech_segments: list[SpeechSegment],
    max_candidates: int = 5,
) -> tuple[NccPeakDiagnostics, str | None]:
    """Return the best NCC peak whose matched span overlaps master speech.

    If the highest-scoring peak falls outside the master VAD speech intervals,
    lower-scoring peaks are tried in order until one overlaps speech or the
    candidate budget is exhausted. The second return value is a VAD rejection
    reason when no candidate is valid.
    """

    nms_samples = max(1, int(round(matching_config.nms_exclusion_seconds * sample_rate)))
    suppressed_scores = scores.copy()
    diagnostics = initial_diagnostics

    for _ in range(max_candidates):
        best_lag = diagnostics.best_lag_samples
        master_start = (search_start_idx + best_lag) / sample_rate
        master_end = master_start + (local_clip_size / sample_rate)

        overlap_ratio = _master_vad_overlap_ratio(
            master_start,
            master_end,
            master_speech_segments,
            padding_seconds=safety_config.master_vad_padding_seconds,
        )
        if overlap_ratio >= safety_config.master_vad_min_overlap_ratio:
            return diagnostics, None

        # Suppress this peak region and recompute diagnostics for the next best.
        start = max(0, best_lag - nms_samples)
        end = min(len(suppressed_scores), best_lag + nms_samples + 1)
        suppressed_scores[start:end] = -1.0

        next_diagnostics = ncc_peak_diagnostics(
            suppressed_scores,
            sample_rate=sample_rate,
            nms_exclusion_seconds=matching_config.nms_exclusion_seconds,
        )
        if next_diagnostics is None or next_diagnostics.best_score < matching_config.ncc_min_score:
            break
        diagnostics = next_diagnostics

    # No VAD-valid candidate found; report the original best peak as rejected.
    best_lag = initial_diagnostics.best_lag_samples
    master_start = (search_start_idx + best_lag) / sample_rate
    master_end = master_start + (local_clip_size / sample_rate)
    overlap_ratio = _master_vad_overlap_ratio(
        master_start,
        master_end,
        master_speech_segments,
        padding_seconds=safety_config.master_vad_padding_seconds,
    )
    rejected_reason = (
        "master_vad_low_overlap_ratio"
        if overlap_ratio > 0.0
        else "master_vad_no_overlap"
    )
    return initial_diagnostics, rejected_reason


def _master_vad_overlap_ratio(
    search_start_seconds: float,
    search_end_seconds: float,
    speech_segments: list[SpeechSegment],
    padding_seconds: float = 0.25,
) -> float:
    """Return the fraction of the search interval that overlaps master speech."""

    search_start = min(search_start_seconds, search_end_seconds)
    search_end = max(search_start_seconds, search_end_seconds)
    search_duration = max(0.0, search_end - search_start)
    if search_duration <= 0.0:
        return 0.0

    # Build padded speech intervals and merge overlaps so that overlapping
    # segments (or segments that overlap after padding) do not inflate the
    # overlap ratio through double-counting.
    intervals = [
        (segment.start - padding_seconds, segment.end + padding_seconds)
        for segment in speech_segments
        if segment.end + padding_seconds > segment.start - padding_seconds
    ]
    if not intervals:
        return 0.0

    intervals.sort()
    merged: list[tuple[float, float]] = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    overlap_seconds = 0.0
    for start, end in merged:
        clip_start = max(search_start, start)
        clip_end = min(search_end, end)
        if clip_end > clip_start:
            overlap_seconds += clip_end - clip_start

    return min(1.0, overlap_seconds / search_duration)


def fit_linear_drift_model(
    anchor_matches: list[AnchorMatch],
    local_duration_seconds: float | None = None,
    config: DriftModelConfig | None = None,
    matching_config: AnchorMatchingConfig | None = None,
) -> DriftEstimate | None:
    if matching_config is None:
        from double_ender_sync.config import DEFAULT_ANCHOR_MATCHING_CONFIG
        matching_config = DEFAULT_ANCHOR_MATCHING_CONFIG

    _FIT_PRODUCED_REASONS = frozenset({
        "excluded_by_min_confidence",
        "confidence_below_min_for_fit",
        "residual_outlier",
        "outside_spline_support",
        "outside_piecewise_support",
    })
    for match in anchor_matches:
        if match.rejected_reason in _FIT_PRODUCED_REASONS:
            match.rejected_reason = None

    eligible_matches = [
        match for match in anchor_matches
        if match.rejected_reason is None
    ]
    eligible_matches = [
        match for match in eligible_matches
        if match.confidence >= matching_config.min_confidence_for_fit
    ]

    for match in anchor_matches:
        match.included_in_regression = False
        if match not in eligible_matches and match.rejected_reason is None:
            match.rejected_reason = "excluded_by_min_confidence"
        match.residual_ms = None

    if len(eligible_matches) < 2:
        return None

    local = np.array([m.local_start for m in eligible_matches], dtype=np.float64)
    master = np.array([m.master_start for m in eligible_matches], dtype=np.float64)
    weights = np.array([max(m.confidence, 1e-3) for m in eligible_matches], dtype=np.float64)

    kept_indices = np.arange(len(eligible_matches))
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

    eligible_idx_by_id = {id(match): idx for idx, match in enumerate(eligible_matches)}

    for full_idx, match in enumerate(anchor_matches):
        residual_ms = (match.master_start - (stretch * match.local_start + offset)) * 1000.0
        match.residual_ms = float(residual_ms)
        eligible_idx = eligible_idx_by_id.get(id(match))
        if eligible_idx is None:
            continue
        match.included_in_regression = eligible_idx in kept_index_set
        if not match.included_in_regression and match.rejected_reason is None:
            match.rejected_reason = "residual_outlier"

    anchor_idx_by_id = {id(match): idx for idx, match in enumerate(anchor_matches)}
    kept_full_indices = np.array([
        anchor_idx_by_id[id(eligible_matches[int(idx)])]
        for idx in kept_indices
    ], dtype=np.intp)

    unsupported_regions = _detect_anchor_gap_unsupported_regions(
        anchor_matches=anchor_matches,
        kept_indices=kept_full_indices,
        max_anchor_gap_seconds=None if config is None else config.max_anchor_gap_seconds,
    )
    diagnostics = _build_drift_fit_diagnostics(
        anchor_matches=anchor_matches,
        kept_indices=kept_full_indices,
        local_duration_seconds=local_duration_seconds,
        residual_rejection_threshold_ms=residual_rejection_threshold_ms,
        unsupported_regions=unsupported_regions,
        eligible_match_count=len(eligible_matches),
    )

    return LinearDrift(
        offset_seconds=float(offset),
        stretch_ratio=float(stretch),
        anchor_count=int(len(kept_indices)),
        residual_median_ms=float(np.median(np.abs(residuals_ms))),
        residual_max_ms=float(np.max(np.abs(residuals_ms))),
        diagnostics=diagnostics,
        unsupported_regions=tuple(unsupported_regions),
        warnings=tuple(
            warning
            for warning in diagnostics.warnings
            if warning.code == "ANCHOR_GAP_UNSUPPORTED_REGION"
        ),
    )


def _weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([x, np.ones_like(x)])
    W = np.diag(w)
    beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)
    return float(beta[0]), float(beta[1])


def _detect_anchor_gap_unsupported_regions(
    anchor_matches: list[AnchorMatch],
    kept_indices: np.ndarray,
    max_anchor_gap_seconds: float | None,
) -> list[dict[str, object]]:
    if max_anchor_gap_seconds is None:
        return []

    kept_matches = sorted((anchor_matches[int(idx)] for idx in kept_indices), key=lambda match: match.local_start)
    regions: list[dict[str, object]] = []
    for previous, current in zip(kept_matches, kept_matches[1:]):
        local_start = float(previous.local_end)
        local_end = float(current.local_start)
        master_start = float(previous.master_end)
        master_end = float(current.master_start)
        local_gap_seconds = local_end - local_start
        master_gap_seconds = master_end - master_start
        if local_gap_seconds <= 0.0 or master_gap_seconds <= 0.0:
            continue
        if local_gap_seconds <= max_anchor_gap_seconds and master_gap_seconds <= max_anchor_gap_seconds:
            continue
        regions.append(
            {
                "code": "ANCHOR_GAP_UNSUPPORTED_REGION",
                "reason": "anchor_gap_dropout_candidate",
                "local_start": local_start,
                "local_end": local_end,
                "master_start": master_start,
                "master_end": master_end,
                "local_gap_seconds": float(local_gap_seconds),
                "master_gap_seconds": float(master_gap_seconds),
                "threshold_seconds": float(max_anchor_gap_seconds),
            }
        )
    return regions


def _deduplicate_drift_warnings(warnings: Sequence[DriftFitWarning]) -> tuple[DriftFitWarning, ...]:
    deduplicated: list[DriftFitWarning] = []
    seen: set[tuple[str, str, float | None]] = set()
    for warning in warnings:
        key = (warning.code, warning.message, warning.time_seconds)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(warning)
    return tuple(deduplicated)


def _deduplicate_unsupported_regions(regions: Sequence[dict[str, object]]) -> tuple[dict[str, object], ...]:
    deduplicated: list[dict[str, object]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for region in regions:
        key = tuple(sorted((str(field), repr(value)) for field, value in region.items()))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(region)
    return tuple(deduplicated)


def _propagate_linear_gap_diagnostics(model: DriftModel, linear_baseline: LinearDrift) -> None:
    gap_regions = tuple(linear_baseline.unsupported_regions)
    gap_warnings = tuple(
        warning
        for warning in linear_baseline.warnings
        if warning.code == "ANCHOR_GAP_UNSUPPORTED_REGION"
    )
    if not gap_regions and not gap_warnings:
        return

    existing_regions = tuple(getattr(model, "unsupported_regions", ()))
    existing_warnings = tuple(getattr(model, "warnings", ()))
    setattr(model, "unsupported_regions", _deduplicate_unsupported_regions((*existing_regions, *gap_regions)))
    setattr(model, "warnings", _deduplicate_drift_warnings((*existing_warnings, *gap_warnings)))


def _region_time_seconds(value: object) -> float | None:
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)
    return None


def _build_drift_fit_diagnostics(
    anchor_matches: list[AnchorMatch],
    kept_indices: np.ndarray,
    local_duration_seconds: float | None,
    residual_rejection_threshold_ms: float | None,
    unsupported_regions: Sequence[dict[str, object]] = (),
    eligible_match_count: int | None = None,
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

    eligible_count = eligible_match_count if eligible_match_count is not None else len(anchor_matches)
    outlier_count = eligible_count - len(kept_matches)
    if outlier_count > 0:
        warnings.append(
            DriftFitWarning(
                code="DRIFT_OUTLIERS_REJECTED",
                message="Some drift anchor matches were excluded from the linear regression as residual outliers.",
            )
        )

    for region in unsupported_regions:
        warnings.append(
            DriftFitWarning(
                code="ANCHOR_GAP_UNSUPPORTED_REGION",
                message=(
                    "Trusted drift anchors contain a long local/master timeline gap; "
                    "treat this as a dropout or reconnect-like candidate and inspect manually."
                ),
                time_seconds=_region_time_seconds(region.get("master_start")),
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


def select_drift_model(
    anchor_matches: list[AnchorMatch],
    config: DriftModelConfig,
    local_duration_seconds: float | None = None,
    matching_config: AnchorMatchingConfig | None = None,
) -> DriftModel | None:
    """Fit the configured drift model with conservative non-linear fallback.

    LinearDrift remains the control fit for every run. PiecewiseLinearDrift and
    SplineDrift can be attempted by the gated ``auto`` policy or explicit
    requests. The Kalman research model is evaluated only for explicit
    ``drift_model="kalman"`` requests with the same non-linear safety gate, so
    experimental behavior cannot run silently and ``auto`` never attempts Kalman.
    """
    linear = fit_linear_drift_model(anchor_matches, local_duration_seconds=local_duration_seconds, config=config, matching_config=matching_config)
    if linear is None:
        return None

    kalman_enabled = config.drift_model == "kalman" and config.allow_nonlinear_drift
    piecewise_enabled = (
        config.drift_model in {"auto", "piecewise_linear"}
        and config.allow_nonlinear_drift
        and config.max_breakpoints > 0
    )
    spline_enabled = config.drift_model in {"auto", "spline"} and config.allow_nonlinear_drift
    piecewise_prefit_for_spline = (
        config.drift_model == "spline"
        and spline_enabled
        and config.spline_knot_source in {"auto", "piecewise_boundaries"}
        and config.max_breakpoints > 0
    )
    candidate_models = ["linear"]
    if piecewise_enabled or piecewise_prefit_for_spline:
        candidate_models.append("piecewise_linear")
    if spline_enabled:
        candidate_models.append("spline")
    if kalman_enabled:
        candidate_models.append("kalman")
    linear.candidate_models = tuple(candidate_models)

    if config.drift_model == "linear":
        linear.model_selection_policy = "linear_requested"
        linear.selected_model_reason = "linear drift model was explicitly requested"
        linear.candidate_models = ("linear",)
        return linear

    if not config.allow_nonlinear_drift:
        linear.model_selection_policy = "linear_default"
        linear.selected_model_reason = "linear is the default control model"
        linear.fallback_reason = "non-linear drift gate is disabled; non-linear candidates were not attempted"
        return linear

    if kalman_enabled:
        kalman_result = fit_kalman_drift_model(
            anchor_matches=anchor_matches,
            linear_baseline=linear,
            config=config,
            local_duration_seconds=local_duration_seconds,
        )
        if kalman_result.model is not None:
            kalman_result.model.candidate_models = tuple(candidate_models)
            _propagate_linear_gap_diagnostics(kalman_result.model, linear)
            return kalman_result.model
        linear.model_selection_policy = "kalman_research_experimental"
        linear.candidate_models = tuple(candidate_models)
        linear.fallback_reason = kalman_result.fallback_reason
        linear.selected_model_reason = "linear control model retained after Kalman research evaluation"
        return linear

    selected: DriftModel = linear
    piecewise_model: PiecewiseLinearDrift | None = None
    piecewise_fallback_reason: str | None = None

    if piecewise_enabled or piecewise_prefit_for_spline:
        piecewise_result = fit_piecewise_linear_drift_model(
            anchor_matches=anchor_matches,
            linear_baseline=linear,
            config=config,
            local_duration_seconds=local_duration_seconds,
        )
        piecewise_model = piecewise_result.model
        piecewise_fallback_reason = piecewise_result.fallback_reason
        if piecewise_model is not None:
            piecewise_model.candidate_models = tuple(candidate_models)
            _propagate_linear_gap_diagnostics(piecewise_model, linear)
            if config.drift_model == "auto":
                piecewise_model.model_selection_policy = "nonlinear_experimental"
            if piecewise_enabled:
                selected = piecewise_model
        elif config.drift_model == "piecewise_linear":
            linear.model_selection_policy = "piecewise_experimental"
            linear.fallback_reason = piecewise_fallback_reason
            linear.selected_model_reason = "linear control model retained after piecewise evaluation"
            return linear
    elif config.drift_model == "piecewise_linear":
        linear.model_selection_policy = "piecewise_experimental"
        linear.selected_model_reason = "linear control model retained because max_breakpoints is 0"
        linear.fallback_reason = "piecewise candidate skipped because max_breakpoints is 0"
        return linear

    if spline_enabled:
        baseline_for_spline = selected
        spline_result = fit_spline_drift_model(
            anchor_matches=anchor_matches,
            linear_baseline=linear,
            config=config,
            local_duration_seconds=local_duration_seconds,
            piecewise_model=piecewise_model,
            comparison_baseline=baseline_for_spline,
        )
        if spline_result.model is not None:
            spline_result.model.candidate_models = tuple(candidate_models)
            _propagate_linear_gap_diagnostics(spline_result.model, linear)
            if config.drift_model == "auto":
                spline_result.model.model_selection_policy = "nonlinear_experimental"
            return spline_result.model
        if config.drift_model == "spline":
            linear.model_selection_policy = "spline_experimental"
            reasons = [reason for reason in (piecewise_fallback_reason, spline_result.fallback_reason) if reason]
            linear.fallback_reason = "; ".join(reasons) if reasons else None
            linear.selected_model_reason = "linear control model retained after spline evaluation"
            return linear
        if isinstance(selected, LinearDrift):
            selected.model_selection_policy = "nonlinear_experimental"
            reasons = [reason for reason in (piecewise_fallback_reason, spline_result.fallback_reason) if reason]
            selected.fallback_reason = "; ".join(reasons) if reasons else None
            selected.selected_model_reason = "linear control model retained after non-linear evaluation"
        else:
            selected.model_selection_policy = "nonlinear_experimental"
            if spline_result.fallback_reason is not None:
                selected.fallback_reason = spline_result.fallback_reason
        return selected

    if isinstance(selected, PiecewiseLinearDrift):
        if config.drift_model == "auto":
            selected.model_selection_policy = "nonlinear_experimental"
        return selected

    selected.model_selection_policy = "nonlinear_experimental"
    selected.selected_model_reason = "linear control model retained; no richer model applied"
    return selected


def fit_kalman_drift_model(
    anchor_matches: list[AnchorMatch],
    linear_baseline: LinearDrift,
    config: DriftModelConfig,
    local_duration_seconds: float | None = None,
) -> KalmanFitResult:
    """Fit a research/experimental Kalman/RTS smoother drift candidate.

    The latent state is ``[offset_seconds, rate_deviation]`` where
    ``offset_seconds = master_time - local_time`` and ``rate_deviation =
    local_rate - 1.0``. Observations are anchor offsets. Process noise models
    slow offset wander and drift-rate variability between anchor local times.
    """
    reliable = sorted(
        [match for match in anchor_matches if match.included_in_regression],
        key=lambda match: match.local_start,
    )
    if len(reliable) < int(config.min_anchors_for_kalman):
        return KalmanFitResult(
            None,
            (
                "kalman candidate skipped: "
                f"{len(reliable)} fitted anchors available, but min_anchors_for_kalman="
                f"{config.min_anchors_for_kalman}"
            ),
        )

    local_times = np.array([match.local_start for match in reliable], dtype=np.float64)
    observed_offsets = np.array([match.master_start - match.local_start for match in reliable], dtype=np.float64)
    confidences = np.array([max(match.confidence, 1e-3) for match in reliable], dtype=np.float64)
    if np.any(np.diff(local_times) <= 0.0):
        return KalmanFitResult(None, "kalman candidate rejected: fitted anchor local times must be strictly increasing")
    if not np.all(np.isfinite(observed_offsets)):
        return KalmanFitResult(None, "kalman candidate rejected: observed anchor offsets contain non-finite values")

    try:
        filtered_states, filtered_covariances, predicted_states, predicted_covariances = _run_kalman_forward_filter(
            local_times=local_times,
            observed_offsets=observed_offsets,
            confidences=confidences,
            linear_baseline=linear_baseline,
            config=config,
        )
        smoothed_states, smoothed_covariances = _run_rts_backward_smoother(
            local_times=local_times,
            filtered_states=filtered_states,
            filtered_covariances=filtered_covariances,
            predicted_states=predicted_states,
            predicted_covariances=predicted_covariances,
        )
    except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
        return KalmanFitResult(None, f"kalman candidate rejected: numerical fitting failed ({exc})")

    if not np.all(np.isfinite(smoothed_states)) or not np.all(np.isfinite(smoothed_covariances)):
        return KalmanFitResult(None, "kalman candidate rejected: numerical fitting produced non-finite states")

    predicted_master = local_times + smoothed_states[:, 0]
    residuals_ms = (np.array([match.master_start for match in reliable], dtype=np.float64) - predicted_master) * 1000.0
    residual_median_ms = float(np.median(np.abs(residuals_ms)))
    residual_max_ms = float(np.max(np.abs(residuals_ms)))
    validation = _validate_kalman_candidate(
        local_times=local_times,
        smoothed_states=smoothed_states,
        residual_median_ms=residual_median_ms,
        residual_max_ms=residual_max_ms,
        linear_baseline=linear_baseline,
        config=config,
        local_duration_seconds=local_duration_seconds,
    )
    if validation[0] is not None:
        return KalmanFitResult(None, validation[0])
    metrics = validation[1]
    assert metrics is not None

    offset_std_ms = np.sqrt(np.maximum(smoothed_covariances[:, 0, 0], 0.0)) * 1000.0
    rate_std_ppm = np.sqrt(np.maximum(smoothed_covariances[:, 1, 1], 0.0)) * 1_000_000.0
    state_points = tuple(
        KalmanStatePoint(
            local_time=float(local_time),
            offset_seconds=float(state[0]),
            rate_deviation=float(state[1]),
            offset_std_ms=float(offset_std),
            rate_std_ppm=float(rate_std),
        )
        for local_time, state, offset_std, rate_std in zip(
            local_times,
            smoothed_states,
            offset_std_ms,
            rate_std_ppm,
        )
    )

    warnings: list[DriftFitWarning] = []
    max_abs_rate_deviation_ppm = max(abs(metrics["rate_min"] - 1.0), abs(metrics["rate_max"] - 1.0)) * 1_000_000.0
    if max_abs_rate_deviation_ppm > float(config.warn_abs_rate_deviation_ppm):
        warnings.append(
            DriftFitWarning(
                code="KALMAN_RATE_DEVIATION_WARNING",
                message=(
                    "Kalman research model accepted, but local-rate deviation "
                    f"{max_abs_rate_deviation_ppm:.3f} ppm exceeds "
                    f"warn_abs_rate_deviation_ppm={config.warn_abs_rate_deviation_ppm}. "
                    "Inspect alignment manually."
                ),
            )
        )

    covariance_summary = {
        "median_offset_std_ms": float(np.median(offset_std_ms)),
        "max_offset_std_ms": float(np.max(offset_std_ms)),
        "median_rate_std_ppm": float(np.median(rate_std_ppm)),
        "max_rate_std_ppm": float(np.max(rate_std_ppm)),
    }
    uncertainty_summary = {
        "median_one_sigma_ms": covariance_summary["median_offset_std_ms"],
        "max_one_sigma_ms": covariance_summary["max_offset_std_ms"],
    }

    model = KalmanDrift(
        state_points=state_points,
        anchor_count=len(reliable),
        residual_median_ms=residual_median_ms,
        residual_max_ms=residual_max_ms,
        linear_baseline=linear_baseline,
        validation_sample_count=int(config.kalman_validation_sample_count),
        monotonicity_min_step_seconds=metrics["min_step_seconds"],
        local_rate_min=metrics["rate_min"],
        local_rate_max=metrics["rate_max"],
        local_rate_mean=metrics["rate_mean"],
        covariance_summary=covariance_summary,
        uncertainty_summary=uncertainty_summary,
        anchor_residuals_ms=tuple(float(value) for value in residuals_ms),
        diagnostics=linear_baseline.diagnostics,
        warnings=tuple(warnings),
    )
    _mark_model_residuals(anchor_matches, model)
    return KalmanFitResult(model, None)


def _run_kalman_forward_filter(
    local_times: np.ndarray,
    observed_offsets: np.ndarray,
    confidences: np.ndarray,
    linear_baseline: LinearDrift,
    config: DriftModelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(local_times)
    filtered_states = np.zeros((count, 2), dtype=np.float64)
    filtered_covariances = np.zeros((count, 2, 2), dtype=np.float64)
    predicted_states = np.zeros((count, 2), dtype=np.float64)
    predicted_covariances = np.zeros((count, 2, 2), dtype=np.float64)

    state = np.array([
        linear_baseline.map_local_to_master(float(local_times[0])) - float(local_times[0]),
        linear_baseline.stretch_ratio - 1.0,
    ], dtype=np.float64)
    covariance = np.diag([
        (float(config.kalman_initial_offset_uncertainty_ms) / 1000.0) ** 2,
        (float(config.kalman_initial_rate_uncertainty_ppm) / 1_000_000.0) ** 2,
    ])
    observation_matrix = np.array([[1.0, 0.0]], dtype=np.float64)

    for index, (local_time, observation, confidence) in enumerate(zip(local_times, observed_offsets, confidences)):
        if index > 0:
            dt = float(local_time - local_times[index - 1])
            transition = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
            process_noise = _kalman_process_noise(dt, config)
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process_noise
        predicted_states[index] = state
        predicted_covariances[index] = covariance

        observation_sigma_seconds = (float(config.kalman_observation_noise_ms) / 1000.0) / max(float(confidence), 1e-3)
        observation_covariance = np.array([[observation_sigma_seconds**2]], dtype=np.float64)
        innovation = np.array([float(observation)], dtype=np.float64) - (observation_matrix @ state)
        innovation_covariance = observation_matrix @ covariance @ observation_matrix.T + observation_covariance
        innovation_variance = float(innovation_covariance[0, 0])
        if innovation_variance <= 0.0 or not np.isfinite(innovation_variance):
            raise ValueError(
                "invalid Kalman innovation variance "
                f"at anchor index {index}: {innovation_variance!r}"
            )
        kalman_gain = (covariance @ observation_matrix.T) / innovation_variance
        state = state + (kalman_gain @ innovation)
        identity = np.eye(2)
        innovation_projection = identity - kalman_gain @ observation_matrix
        covariance = innovation_projection @ covariance @ innovation_projection.T + kalman_gain @ observation_covariance @ kalman_gain.T
        covariance = (covariance + covariance.T) / 2.0
        filtered_states[index] = state
        filtered_covariances[index] = covariance

    return filtered_states, filtered_covariances, predicted_states, predicted_covariances


def _run_rts_backward_smoother(
    local_times: np.ndarray,
    filtered_states: np.ndarray,
    filtered_covariances: np.ndarray,
    predicted_states: np.ndarray,
    predicted_covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    smoothed_states = filtered_states.copy()
    smoothed_covariances = filtered_covariances.copy()
    for index in range(len(local_times) - 2, -1, -1):
        dt = float(local_times[index + 1] - local_times[index])
        transition = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        predicted_covariance = predicted_covariances[index + 1]
        if not np.all(np.isfinite(predicted_covariance)):
            raise ValueError(f"non-finite predicted covariance at smoother index {index + 1}")
        covariance_condition = float(np.linalg.cond(predicted_covariance))
        if not np.isfinite(covariance_condition) or covariance_condition > _MAX_KALMAN_COVARIANCE_CONDITION:
            raise ValueError(
                "ill-conditioned predicted covariance "
                f"at smoother index {index + 1}: condition={covariance_condition:.6g}"
            )
        smoother_left = filtered_covariances[index] @ transition.T
        smoother_gain = np.linalg.solve(predicted_covariance.T, smoother_left.T).T
        smoothed_states[index] = filtered_states[index] + smoother_gain @ (
            smoothed_states[index + 1] - predicted_states[index + 1]
        )
        smoothed_covariances[index] = filtered_covariances[index] + smoother_gain @ (
            smoothed_covariances[index + 1] - predicted_covariances[index + 1]
        ) @ smoother_gain.T
        smoothed_covariances[index] = (smoothed_covariances[index] + smoothed_covariances[index].T) / 2.0
    return smoothed_states, smoothed_covariances


def _kalman_process_noise(dt: float, config: DriftModelConfig) -> np.ndarray:
    dt = max(float(dt), 0.0)
    offset_sigma = float(config.kalman_process_offset_noise_ms) / 1000.0
    rate_sigma = float(config.kalman_process_rate_noise_ppm) / 1_000_000.0
    return np.array(
        [
            [offset_sigma**2 * dt + (rate_sigma**2 * dt**3 / 3.0), rate_sigma**2 * dt**2 / 2.0],
            [rate_sigma**2 * dt**2 / 2.0, rate_sigma**2 * dt],
        ],
        dtype=np.float64,
    )


def _validate_kalman_candidate(
    local_times: np.ndarray,
    smoothed_states: np.ndarray,
    residual_median_ms: float,
    residual_max_ms: float,
    linear_baseline: LinearDrift,
    config: DriftModelConfig,
    local_duration_seconds: float | None = None,
) -> tuple[str | None, dict[str, float] | None]:
    validation_count = int(config.kalman_validation_sample_count)
    support_start = 0.0 if local_duration_seconds is not None and local_duration_seconds > 0.0 else float(local_times[0])
    support_end = (
        float(local_duration_seconds)
        if local_duration_seconds is not None and local_duration_seconds > 0.0
        else float(local_times[-1])
    )
    if support_end <= support_start:
        return "kalman candidate rejected: validation support must have positive duration", None
    sample_times = np.linspace(support_start, support_end, num=validation_count, dtype=np.float64)
    sample_offsets = np.array(
        [_interpolate_piecewise_linear_offset(float(sample_time), local_times, smoothed_states[:, 0]) for sample_time in sample_times],
        dtype=np.float64,
    )
    sample_master_times = sample_times + sample_offsets
    if not np.all(np.isfinite(sample_master_times)):
        return "kalman candidate rejected: monotonicity check produced non-finite master times", None
    steps = np.diff(sample_master_times)
    min_step = float(np.min(steps)) if steps.size else 0.0
    if steps.size and np.any(steps <= float(config.monotonicity_rate_epsilon)):
        return (
            "kalman candidate rejected: monotonicity check failed "
            f"(sample_count={validation_count}, min_step_seconds={min_step:.12g}, "
            f"monotonicity_rate_epsilon={config.monotonicity_rate_epsilon})"
        ), None

    rates = np.array(
        [
            _interpolate_piecewise_linear_mapping_rate(float(sample_time), local_times, smoothed_states[:, 0])
            for sample_time in sample_times
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(rates)):
        return "kalman candidate rejected: local-rate validation produced non-finite values", None
    rate_min = float(np.min(rates))
    rate_max = float(np.max(rates))
    rate_mean = float(np.mean(rates))
    max_abs_rate_deviation_ppm = max(abs(rate_min - 1.0), abs(rate_max - 1.0)) * 1_000_000.0
    if max_abs_rate_deviation_ppm > float(config.max_abs_rate_deviation_ppm):
        return (
            "kalman candidate rejected: local-rate deviation "
            f"{max_abs_rate_deviation_ppm:.3f} ppm exceeds "
            f"max_abs_rate_deviation_ppm={config.max_abs_rate_deviation_ppm}"
        ), None

    segment_rates = np.array(
        [
            _interpolate_piecewise_linear_mapping_rate(float(left), local_times, smoothed_states[:, 0])
            for left in local_times[:-1]
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(segment_rates)):
        return "kalman candidate rejected: segment local-rate validation produced non-finite values", None
    rate_changes = np.abs(np.diff(segment_rates)) * 1_000_000.0
    max_rate_change = float(np.max(rate_changes)) if rate_changes.size else 0.0
    if max_rate_change > float(config.max_rate_change_ppm):
        return (
            "kalman candidate rejected: adjacent local-rate change "
            f"{max_rate_change:.3f} ppm exceeds max_rate_change_ppm={config.max_rate_change_ppm}"
        ), None

    if linear_baseline.residual_median_ms - residual_median_ms < float(config.min_residual_improvement_ms):
        return (
            "kalman candidate rejected: median residual improvement "
            f"{linear_baseline.residual_median_ms - residual_median_ms:.3f} ms is below "
            f"min_residual_improvement_ms={config.min_residual_improvement_ms}"
        ), None
    if linear_baseline.residual_median_ms > 0.0:
        relative_improvement = (linear_baseline.residual_median_ms - residual_median_ms) / linear_baseline.residual_median_ms
        if relative_improvement < float(config.min_relative_residual_improvement):
            return (
                "kalman candidate rejected: relative residual improvement "
                f"{relative_improvement:.6f} is below "
                f"min_relative_residual_improvement={config.min_relative_residual_improvement}"
            ), None
    if residual_max_ms > linear_baseline.residual_max_ms + 1e-9:
        return (
            "kalman candidate rejected: max residual worsened compared with linear control "
            f"(kalman={residual_max_ms:.3f} ms, linear={linear_baseline.residual_max_ms:.3f} ms)"
        ), None
    return None, {
        "min_step_seconds": min_step,
        "rate_min": rate_min,
        "rate_max": rate_max,
        "rate_mean": rate_mean,
    }


def _mark_model_residuals(anchor_matches: list[AnchorMatch], model: DriftModel) -> None:
    for match in anchor_matches:
        match.residual_ms = float(1000.0 * (match.master_start - model.map_local_to_master(match.local_start)))

def fit_spline_drift_model(
    anchor_matches: list[AnchorMatch],
    linear_baseline: LinearDrift,
    config: DriftModelConfig,
    local_duration_seconds: float | None = None,
    piecewise_model: PiecewiseLinearDrift | None = None,
    comparison_baseline: DriftModel | None = None,
) -> SplineFitResult:
    """Fit a monotonic cubic PCHIP drift candidate with conservative guards."""
    reliable = sorted(
        [match for match in anchor_matches if match.included_in_regression],
        key=lambda match: match.local_start,
    )
    if len(reliable) < int(config.min_anchors_for_spline):
        return SplineFitResult(
            None,
            (
                "spline candidate skipped: "
                f"{len(reliable)} fitted anchors available, but min_anchors_for_spline="
                f"{config.min_anchors_for_spline}"
            ),
        )

    knot_result = _choose_spline_knots(
        reliable,
        linear_baseline,
        config,
        local_duration_seconds,
        piecewise_model,
    )
    if knot_result.knot_local_times is None or knot_result.knot_master_times is None:
        return SplineFitResult(None, knot_result.fallback_reason)
    knot_local_times = knot_result.knot_local_times
    knot_master_times = knot_result.knot_master_times
    knot_source = knot_result.knot_source
    assert knot_source is not None

    try:
        interpolator = PchipInterpolator(
            np.array(knot_local_times, dtype=np.float64),
            np.array(knot_master_times, dtype=np.float64),
            extrapolate=True,
        )
        derivative = interpolator.derivative()
    except ValueError as exc:
        return SplineFitResult(
            None,
            f"spline candidate rejected: PCHIP construction failed for knot_source={knot_source}: {exc}",
        )

    residuals_ms = np.array(
        [1000.0 * (match.master_start - float(interpolator(match.local_start))) for match in reliable],
        dtype=np.float64,
    )
    validation = _validate_spline_candidate(
        interpolator=interpolator,
        derivative=derivative,
        knot_local_times=knot_local_times,
        linear_baseline=linear_baseline,
        comparison_baseline=comparison_baseline or linear_baseline,
        residual_median_ms=float(np.median(np.abs(residuals_ms))),
        residual_max_ms=float(np.max(np.abs(residuals_ms))),
        config=config,
    )
    if validation[0] is not None:
        return SplineFitResult(None, validation[0])
    metrics = validation[1]
    assert metrics is not None

    warnings: tuple[DriftFitWarning, ...] = ()
    if metrics["max_abs_rate_deviation_ppm"] > float(config.warn_abs_rate_deviation_ppm):
        warnings = (
            DriftFitWarning(
                code="SPLINE_RATE_DEVIATION_WARNING",
                message=(
                    "Spline model accepted, but local-rate deviation "
                    f"{metrics['max_abs_rate_deviation_ppm']:.3f} ppm exceeds "
                    f"warn_abs_rate_deviation_ppm={config.warn_abs_rate_deviation_ppm}. "
                    "Inspect alignment manually."
                ),
            ),
        )

    try:
        model = SplineDrift(
            knot_local_times=tuple(float(value) for value in knot_local_times),
            knot_master_times=tuple(float(value) for value in knot_master_times),
            interpolation_method="pchip",
            knot_source=knot_source,
            anchor_count=len(reliable),
            residual_median_ms=float(np.median(np.abs(residuals_ms))),
            residual_max_ms=float(np.max(np.abs(residuals_ms))),
            linear_baseline=linear_baseline,
            baseline_model_type=(comparison_baseline or linear_baseline).model_type,
            validation_sample_count=int(config.spline_validation_sample_count),
            monotonicity_min_step_seconds=metrics["min_step_seconds"],
            local_rate_min=metrics["rate_min"],
            local_rate_max=metrics["rate_max"],
            local_rate_mean=metrics["rate_mean"],
            local_rate_change_max_ppm=metrics["max_rate_change_ppm"],
            diagnostics=linear_baseline.diagnostics,
            warnings=warnings,
            knot_residual_summaries=tuple(
                _summarize_spline_knot_residuals(reliable, knot_local_times, knot_master_times, interpolator)
            ),
            knot_decimation_applied=knot_result.decimation_applied,
            pchip_interpolator=interpolator,
            pchip_derivative=derivative,
        )
    except ValueError as exc:
        return SplineFitResult(
            None,
            f"spline candidate rejected: SplineDrift construction failed for knot_source={knot_source}: {exc}",
        )
    _mark_spline_residuals(anchor_matches, model)
    return SplineFitResult(model, None)


def _summarize_spline_knot_residuals(
    reliable: list[AnchorMatch],
    knot_local_times: list[float],
    knot_master_times: list[float],
    interpolator: PchipInterpolator,
) -> list[dict[str, object]]:
    """Summarize fitted-anchor residuals nearest to each spline support knot."""
    summaries: list[dict[str, object]] = []
    if not knot_local_times:
        return summaries

    knot_array = np.array(knot_local_times, dtype=np.float64)
    for index, (local_time, master_time) in enumerate(zip(knot_local_times, knot_master_times)):
        if len(knot_array) == 1:
            left_bound = float("-inf")
            right_bound = float("inf")
        elif index == 0:
            left_bound = float("-inf")
            right_bound = float((knot_array[index] + knot_array[index + 1]) / 2.0)
        elif index == len(knot_array) - 1:
            left_bound = float((knot_array[index - 1] + knot_array[index]) / 2.0)
            right_bound = float("inf")
        else:
            left_bound = float((knot_array[index - 1] + knot_array[index]) / 2.0)
            right_bound = float((knot_array[index] + knot_array[index + 1]) / 2.0)

        residuals = [
            1000.0 * (match.master_start - float(interpolator(match.local_start)))
            for match in reliable
            if left_bound <= match.local_start < right_bound
        ]

        abs_residuals = np.abs(np.array(residuals, dtype=np.float64)) if residuals else np.array([], dtype=np.float64)
        summaries.append(
            {
                "local_time": float(local_time),
                "master_time": float(master_time),
                "anchor_count": len(residuals),
                "residual_median_ms": None if abs_residuals.size == 0 else float(np.median(abs_residuals)),
                "residual_max_ms": None if abs_residuals.size == 0 else float(np.max(abs_residuals)),
            }
        )
    return summaries


def _choose_spline_knots(
    reliable: list[AnchorMatch],
    linear_baseline: LinearDrift,
    config: DriftModelConfig,
    local_duration_seconds: float | None,
    piecewise_model: PiecewiseLinearDrift | None,
) -> SplineKnotSelectionResult:
    requested_source = config.spline_knot_source
    if requested_source in {"auto", "piecewise_boundaries"} and piecewise_model is not None:
        local_times = [piecewise_model.segments[0].local_start]
        local_times.extend(piecewise_model.breakpoints)
        local_times.append(piecewise_model.segments[-1].local_end)
        master_times = [piecewise_model.map_local_to_master(local_time) for local_time in local_times]
        if len(local_times) >= 3:
            return SplineKnotSelectionResult(local_times, master_times, "piecewise_boundaries")
        if requested_source == "piecewise_boundaries":
            return SplineKnotSelectionResult(
                None,
                None,
                None,
                "spline candidate skipped: piecewise boundary knot source produced fewer than 3 knots",
            )

    if requested_source == "piecewise_boundaries":
        return SplineKnotSelectionResult(
            None,
            None,
            None,
            "spline candidate skipped: spline_knot_source='piecewise_boundaries' but no accepted piecewise model is available",
        )

    local_times, master_times, decimation_applied = _decimate_anchor_knots(
        reliable,
        linear_baseline,
        local_duration_seconds,
        float(config.min_knot_spacing_seconds),
    )
    if len(local_times) < 3:
        return SplineKnotSelectionResult(
            None,
            None,
            None,
            (
                "spline candidate skipped: anchor-decimated knot source produced "
                f"{len(local_times)} knots; at least 3 are required"
            ),
        )
    return SplineKnotSelectionResult(local_times, master_times, "anchors", decimation_applied=decimation_applied)


def _decimate_anchor_knots(
    reliable: list[AnchorMatch],
    linear_baseline: LinearDrift,
    local_duration_seconds: float | None,
    min_knot_spacing_seconds: float,
) -> tuple[list[float], list[float], bool]:
    knot_pairs: list[tuple[float, float]] = []
    decimation_applied = False
    if local_duration_seconds is not None and local_duration_seconds > 0.0:
        knot_pairs.append((0.0, linear_baseline.map_local_to_master(0.0)))

    last_anchor_time = knot_pairs[-1][0] if knot_pairs else None
    for match in reliable:
        if last_anchor_time is not None and abs(match.local_start - last_anchor_time) <= 1e-9:
            knot_pairs[-1] = (float(match.local_start), float(match.master_start))
            last_anchor_time = float(match.local_start)
            continue
        if last_anchor_time is not None and match.local_start - last_anchor_time < min_knot_spacing_seconds:
            decimation_applied = True
            continue
        knot_pairs.append((float(match.local_start), float(match.master_start)))
        last_anchor_time = float(match.local_start)

    if reliable:
        final_anchor = (float(reliable[-1].local_start), float(reliable[-1].master_start))
        if not knot_pairs or abs(knot_pairs[-1][0] - final_anchor[0]) > 1e-9:
            if knot_pairs and final_anchor[0] - knot_pairs[-1][0] < min_knot_spacing_seconds and len(knot_pairs) > 1:
                decimation_applied = True
                knot_pairs[-1] = final_anchor
            else:
                knot_pairs.append(final_anchor)

    if local_duration_seconds is not None and local_duration_seconds > 0.0:
        final_pair = (float(local_duration_seconds), linear_baseline.map_local_to_master(float(local_duration_seconds)))
        if not knot_pairs or final_pair[0] - knot_pairs[-1][0] > 1e-9:
            knot_pairs.append(final_pair)

    deduped: list[tuple[float, float]] = []
    for local_time, master_time in sorted(knot_pairs):
        if deduped and local_time <= deduped[-1][0] + 1e-9:
            continue
        deduped.append((local_time, master_time))

    local_times = [pair[0] for pair in deduped]
    master_times = [pair[1] for pair in deduped]
    return local_times, master_times, decimation_applied


def _validate_spline_candidate(
    interpolator: PchipInterpolator,
    derivative: Any,
    knot_local_times: list[float],
    linear_baseline: LinearDrift,
    comparison_baseline: DriftModel,
    residual_median_ms: float,
    residual_max_ms: float,
    config: DriftModelConfig,
) -> tuple[str | None, dict[str, float] | None]:
    validation_count = int(config.spline_validation_sample_count)
    sample_times = np.linspace(knot_local_times[0], knot_local_times[-1], num=validation_count, dtype=np.float64)
    sample_master_times = np.asarray(interpolator(sample_times), dtype=np.float64)
    if not np.all(np.isfinite(sample_master_times)):
        return "spline candidate rejected: monotonicity check produced non-finite master times", None

    steps = np.diff(sample_master_times)
    min_step = float(np.min(steps)) if steps.size else 0.0
    if steps.size and np.any(steps <= float(config.monotonicity_rate_epsilon)):
        return (
            "spline candidate rejected: monotonicity check failed "
            f"(sample_count={validation_count}, min_step_seconds={min_step:.12g}, "
            f"monotonicity_rate_epsilon={config.monotonicity_rate_epsilon})"
        ), None

    rates = np.asarray(derivative(sample_times), dtype=np.float64)
    if not np.all(np.isfinite(rates)):
        return "spline candidate rejected: local-rate check produced non-finite derivative values", None
    rate_min = float(np.min(rates))
    rate_max = float(np.max(rates))
    rate_mean = float(np.mean(rates))
    if rate_min <= float(config.monotonicity_rate_epsilon):
        return (
            "spline candidate rejected: local-rate derivative is not strictly positive "
            f"(min_rate={rate_min:.12g}, monotonicity_rate_epsilon={config.monotonicity_rate_epsilon})"
        ), None

    max_abs_deviation = float(np.max(np.abs(rates - 1.0)) * 1_000_000.0)
    if max_abs_deviation > float(config.max_abs_rate_deviation_ppm):
        return (
            "spline candidate rejected: absolute local-rate deviation "
            f"{max_abs_deviation:.3f} ppm exceeds max_abs_rate_deviation_ppm={config.max_abs_rate_deviation_ppm}"
        ), None

    knot_rates = np.asarray(derivative(np.array(knot_local_times, dtype=np.float64)), dtype=np.float64)
    max_rate_change = float(np.max(np.abs(np.diff(knot_rates))) * 1_000_000.0) if knot_rates.size > 1 else 0.0
    if max_rate_change > float(config.max_rate_change_ppm):
        return (
            "spline candidate rejected: adjacent knot local-rate change "
            f"{max_rate_change:.3f} ppm exceeds max_rate_change_ppm={config.max_rate_change_ppm}"
        ), None

    baseline_median = getattr(comparison_baseline, "residual_median_ms", linear_baseline.residual_median_ms)
    median_improvement = float(baseline_median) - residual_median_ms
    if median_improvement < float(config.min_residual_improvement_ms):
        return (
            "spline residual improvement below min_residual_improvement_ms "
            f"({median_improvement:.3f} ms < {config.min_residual_improvement_ms} ms)"
        ), None

    if float(baseline_median) > 0.0:
        relative_improvement = median_improvement / float(baseline_median)
        if relative_improvement < float(config.min_relative_residual_improvement):
            return (
                "spline relative residual improvement below threshold "
                f"({relative_improvement:.3f} < {config.min_relative_residual_improvement})"
            ), None

    baseline_max = getattr(comparison_baseline, "residual_max_ms", linear_baseline.residual_max_ms)
    required_max_improvement = max(float(config.min_residual_improvement_ms), float(baseline_max) * 0.10)
    max_improvement = float(baseline_max) - residual_max_ms
    if max_improvement < required_max_improvement:
        return (
            "spline worst-case residual improvement below threshold "
            f"({max_improvement:.3f} ms < {required_max_improvement:.3f} ms)"
        ), None

    return None, {
        "min_step_seconds": min_step,
        "rate_min": rate_min,
        "rate_max": rate_max,
        "rate_mean": rate_mean,
        "max_abs_rate_deviation_ppm": max_abs_deviation,
        "max_rate_change_ppm": max_rate_change,
    }


def _mark_spline_residuals(anchor_matches: list[AnchorMatch], model: SplineDrift) -> None:
    residuals = model.residuals_ms(anchor_matches)
    support_start = model.knot_local_times[0]
    support_end = model.knot_local_times[-1]
    for match, residual in zip(anchor_matches, residuals):
        was_linear_inlier = match.included_in_regression and match.rejected_reason is None
        inside_support = support_start <= match.local_start <= support_end
        match.residual_ms = float(residual)
        match.included_in_regression = was_linear_inlier and inside_support
        if was_linear_inlier and not inside_support:
            match.rejected_reason = "outside_spline_support"


def fit_piecewise_linear_drift_model(
    anchor_matches: list[AnchorMatch],
    linear_baseline: LinearDrift,
    config: DriftModelConfig,
    local_duration_seconds: float | None = None,
) -> PiecewiseLinearFitResult:
    reliable = [match for match in anchor_matches if match.included_in_regression]
    if len(reliable) < int(config.min_anchors_for_piecewise):
        return PiecewiseLinearFitResult(
            None,
            (
                "piecewise candidate skipped: "
                f"{len(reliable)} fitted anchors available, but min_anchors_for_piecewise="
                f"{config.min_anchors_for_piecewise}"
            ),
        )

    reliable.sort(key=lambda match: match.local_start)
    breakpoints = _choose_piecewise_breakpoints(reliable, linear_baseline, config)
    if not breakpoints:
        return PiecewiseLinearFitResult(None, "piecewise candidate skipped: no breakpoint satisfied segment anchor coverage")

    candidate = _fit_continuous_piecewise_model(
        reliable,
        tuple(breakpoints),
        linear_baseline,
        local_duration_seconds,
    )
    rejection_reason = _validate_piecewise_candidate(candidate, linear_baseline, config)
    if rejection_reason is not None:
        return PiecewiseLinearFitResult(None, rejection_reason)

    _mark_piecewise_residuals(anchor_matches, candidate)
    return PiecewiseLinearFitResult(candidate, None)


def _choose_piecewise_breakpoints(
    matches: list[AnchorMatch],
    linear_baseline: LinearDrift,
    config: DriftModelConfig,
) -> list[float]:
    selected: list[float] = []
    current_score = linear_baseline.residual_median_ms
    max_breakpoints = int(config.max_breakpoints)
    for _ in range(max_breakpoints):
        best_breakpoint: float | None = None
        best_score = current_score
        candidates = _candidate_breakpoints(matches, selected, int(config.min_anchors_per_segment))
        for breakpoint in candidates:
            trial_breakpoints = sorted([*selected, breakpoint])
            trial = _fit_continuous_piecewise_model(matches, tuple(trial_breakpoints), linear_baseline, None)
            if _segment_anchor_counts_ok(trial, int(config.min_anchors_per_segment)):
                score = trial.residual_median_ms
                if score < best_score:
                    best_score = score
                    best_breakpoint = breakpoint
        if best_breakpoint is None:
            break
        selected.append(best_breakpoint)
        selected.sort()
        current_score = best_score
    return selected


def _candidate_breakpoints(
    matches: list[AnchorMatch],
    existing: list[float],
    min_anchors_per_segment: int,
) -> list[float]:
    local_times = [match.local_start for match in matches]
    candidates: list[float] = []
    for index in range(min_anchors_per_segment, len(local_times) - min_anchors_per_segment + 1):
        for breakpoint in (local_times[index - 1], (local_times[index - 1] + local_times[index]) / 2.0):
            if any(abs(breakpoint - current) < 1e-9 for current in existing):
                continue
            if any(abs(breakpoint - current) < 1e-9 for current in candidates):
                continue
            candidates.append(float(breakpoint))
    return candidates


def _fit_continuous_piecewise_model(
    matches: list[AnchorMatch],
    breakpoints: tuple[float, ...],
    linear_baseline: LinearDrift,
    local_duration_seconds: float | None,
) -> PiecewiseLinearDrift:
    x = np.array([match.local_start for match in matches], dtype=np.float64)
    y = np.array([match.master_start for match in matches], dtype=np.float64)
    weights = np.array([max(match.confidence, 1e-3) for match in matches], dtype=np.float64)

    columns = [np.ones_like(x), x]
    for breakpoint in breakpoints:
        columns.append(np.maximum(0.0, x - breakpoint))
    design = np.column_stack(columns)
    sqrt_w = np.sqrt(weights)
    beta = np.linalg.pinv(design * sqrt_w[:, None]) @ (y * sqrt_w)

    intercept = float(beta[0])
    base_slope = float(beta[1])
    slope_changes = [float(value) for value in beta[2:]]
    segment_rates = _piecewise_segment_rates(base_slope, slope_changes)

    local_start = 0.0 if local_duration_seconds is not None else float(np.min(x))
    local_end = float(local_duration_seconds) if local_duration_seconds is not None else float(np.max(x))
    boundaries = [local_start, *breakpoints, local_end]
    segments: list[PiecewiseLinearSegment] = []
    predictions = _predict_piecewise(x, intercept, base_slope, breakpoints, slope_changes)
    residuals_ms = (y - predictions) * 1000.0

    for index, rate in enumerate(segment_rates):
        start = float(boundaries[index])
        end = float(boundaries[index + 1])
        offset = _piecewise_value(start, intercept, base_slope, breakpoints, slope_changes) - (rate * start)
        in_segment = (x >= start) & (x <= end if index == len(segment_rates) - 1 else x < end)
        segment_residuals = np.abs(residuals_ms[in_segment])
        if segment_residuals.size == 0:
            residual_median = 0.0
            residual_max = 0.0
        else:
            residual_median = float(np.median(segment_residuals))
            residual_max = float(np.max(segment_residuals))
        segments.append(
            PiecewiseLinearSegment(
                local_start=start,
                local_end=end,
                master_start=float(_piecewise_value(start, intercept, base_slope, breakpoints, slope_changes)),
                master_end=float(_piecewise_value(end, intercept, base_slope, breakpoints, slope_changes)),
                stretch_ratio=float(rate),
                offset_seconds=float(offset),
                anchor_count=int(np.sum(in_segment)),
                residual_median_ms=residual_median,
                residual_max_ms=residual_max,
            )
        )

    diagnostics = _build_piecewise_diagnostics(linear_baseline.diagnostics, segments)
    return PiecewiseLinearDrift(
        breakpoints=breakpoints,
        segments=tuple(segments),
        anchor_count=len(matches),
        residual_median_ms=float(np.median(np.abs(residuals_ms))),
        residual_max_ms=float(np.max(np.abs(residuals_ms))),
        linear_baseline=linear_baseline,
        diagnostics=diagnostics,
    )


def _predict_piecewise(
    x: np.ndarray,
    intercept: float,
    base_slope: float,
    breakpoints: tuple[float, ...],
    slope_changes: list[float],
) -> np.ndarray:
    y = intercept + (base_slope * x)
    for breakpoint, change in zip(breakpoints, slope_changes):
        y = y + (change * np.maximum(0.0, x - breakpoint))
    return y


def _piecewise_value(
    local_time: float,
    intercept: float,
    base_slope: float,
    breakpoints: tuple[float, ...],
    slope_changes: list[float],
) -> float:
    value = intercept + (base_slope * local_time)
    for breakpoint, change in zip(breakpoints, slope_changes):
        value += change * max(0.0, local_time - breakpoint)
    return float(value)


def _piecewise_segment_rates(base_slope: float, slope_changes: list[float]) -> list[float]:
    rates = [base_slope]
    current = base_slope
    for change in slope_changes:
        current += change
        rates.append(current)
    return [float(rate) for rate in rates]


def _segment_anchor_counts_ok(model: PiecewiseLinearDrift, min_anchors_per_segment: int) -> bool:
    return all(segment.anchor_count >= min_anchors_per_segment for segment in model.segments)


def _validate_piecewise_candidate(candidate: PiecewiseLinearDrift, linear_baseline: LinearDrift, config: DriftModelConfig) -> str | None:
    if not _segment_anchor_counts_ok(candidate, int(config.min_anchors_per_segment)):
        counts = [segment.anchor_count for segment in candidate.segments]
        return (
            "piecewise candidate rejected: each segment requires at least "
            f"{config.min_anchors_per_segment} anchors, got segment counts {counts}"
        )

    rates = [segment.stretch_ratio for segment in candidate.segments]
    if any(rate <= float(config.monotonicity_rate_epsilon) for rate in rates):
        return f"piecewise candidate rejected: non-monotonic segment rates {rates}"

    abs_rate_deviations = [abs(rate - 1.0) * 1_000_000.0 for rate in rates]
    max_abs_deviation = max(abs_rate_deviations)
    if max_abs_deviation > float(config.max_abs_rate_deviation_ppm):
        return (
            "piecewise candidate rejected: absolute local-rate deviation "
            f"{max_abs_deviation:.3f} ppm exceeds max_abs_rate_deviation_ppm={config.max_abs_rate_deviation_ppm}"
        )

    rate_changes = [abs(right - left) * 1_000_000.0 for left, right in zip(rates, rates[1:])]
    max_rate_change = max(rate_changes, default=0.0)
    if max_rate_change > float(config.max_rate_change_ppm):
        return (
            "piecewise candidate rejected: adjacent local-rate change "
            f"{max_rate_change:.3f} ppm exceeds max_rate_change_ppm={config.max_rate_change_ppm}"
        )

    median_improvement = linear_baseline.residual_median_ms - candidate.residual_median_ms
    if median_improvement < float(config.min_residual_improvement_ms):
        return (
            "piecewise residual improvement below min_residual_improvement_ms "
            f"({median_improvement:.3f} ms < {config.min_residual_improvement_ms} ms)"
        )

    if linear_baseline.residual_median_ms > 0.0:
        relative_improvement = median_improvement / linear_baseline.residual_median_ms
        if relative_improvement < float(config.min_relative_residual_improvement):
            return (
                "piecewise relative residual improvement below threshold "
                f"({relative_improvement:.3f} < {config.min_relative_residual_improvement})"
            )

    required_max_improvement = max(float(config.min_residual_improvement_ms), linear_baseline.residual_max_ms * 0.10)
    max_improvement = linear_baseline.residual_max_ms - candidate.residual_max_ms
    if max_improvement < required_max_improvement:
        return (
            "piecewise worst-case residual improvement below threshold "
            f"({max_improvement:.3f} ms < {required_max_improvement:.3f} ms)"
        )

    if max_abs_deviation > float(config.warn_abs_rate_deviation_ppm):
        candidate.warnings = (
            *candidate.warnings,
            DriftFitWarning(
                code="PIECEWISE_RATE_DEVIATION_WARNING",
                message=(
                    "Piecewise model accepted, but local-rate deviation "
                    f"{max_abs_deviation:.3f} ppm exceeds warn_abs_rate_deviation_ppm="
                    f"{config.warn_abs_rate_deviation_ppm}. Inspect alignment manually."
                ),
            ),
        )

    return None


def _mark_piecewise_residuals(anchor_matches: list[AnchorMatch], model: PiecewiseLinearDrift) -> None:
    residuals = model.residuals_ms(anchor_matches)
    for match, residual in zip(anchor_matches, residuals):
        was_linear_inlier = match.included_in_regression and match.rejected_reason is None
        inside_support = any(segment.contains(match.local_start) for segment in model.segments)
        match.residual_ms = float(residual)
        match.included_in_regression = was_linear_inlier and inside_support
        if was_linear_inlier and not inside_support:
            match.rejected_reason = "outside_piecewise_support"


def _build_piecewise_diagnostics(
    baseline_diagnostics: DriftFitDiagnostics | None,
    segments: list[PiecewiseLinearSegment],
) -> DriftFitDiagnostics | None:
    if baseline_diagnostics is None:
        return None
    warnings = [*baseline_diagnostics.warnings]
    return DriftFitDiagnostics(
        input_anchor_count=baseline_diagnostics.input_anchor_count,
        matched_anchor_count=baseline_diagnostics.matched_anchor_count,
        fitted_anchor_count=sum(segment.anchor_count for segment in segments),
        outlier_count=baseline_diagnostics.outlier_count,
        local_span_start_seconds=baseline_diagnostics.local_span_start_seconds,
        local_span_end_seconds=baseline_diagnostics.local_span_end_seconds,
        local_span_seconds=baseline_diagnostics.local_span_seconds,
        local_span_ratio=baseline_diagnostics.local_span_ratio,
        residual_rejection_threshold_ms=baseline_diagnostics.residual_rejection_threshold_ms,
        warnings=warnings,
    )
