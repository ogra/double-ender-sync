from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal


SplineKnotSource = Literal["auto", "piecewise_boundaries", "anchors"]


DriftModelName = Literal["auto", "linear", "piecewise_linear", "spline", "kalman"]


MasterVadUncertainPolicy = Literal["warn", "skip", "reject"]


@dataclass(frozen=True)
class InitialOffsetSafetyConfig:
    """Configuration for the initial-offset safety net.

    The safety net makes initial offset estimation more resilient by adding a
    coarse whole-recording FFT fallback for low-confidence anchor estimates,
    widening drift-anchor searches when confidence is medium/low, and using
    master-side speech evidence to avoid matching local speech into master
    silence.
    """

    initial_offset_min_confidence: float = 0.50
    high_confidence_threshold: float = 0.75
    medium_confidence_threshold: float = 0.50
    low_confidence_threshold: float = 0.25

    coarse_fallback_enabled: bool = True
    coarse_fallback_sample_rate: int = 8000
    coarse_fallback_min_peak_margin: float = 0.10
    coarse_fallback_max_duration_seconds: float | None = None
    coarse_fallback_max_memory_mb: float = 1024.0
    coarse_fallback_min_confidence: float = 0.50
    coarse_fallback_confidence_margin: float = 0.15

    max_drift_search_radius_seconds: float = 30.0
    high_confidence_search_radius_seconds: float = 6.0
    medium_confidence_search_radius_seconds: float = 12.0
    low_confidence_search_radius_seconds: float = 20.0

    master_vad_filter_enabled: bool = True
    master_vad_min_overlap_ratio: float = 0.25
    master_vad_padding_seconds: float = 0.25
    master_vad_uncertain_policy: MasterVadUncertainPolicy = "warn"

    def __post_init__(self) -> None:
        if not (0.0 < self.low_confidence_threshold < self.medium_confidence_threshold < self.high_confidence_threshold < 1.0):
            raise ValueError(
                "confidence thresholds must satisfy 0 < low < medium < high < 1"
            )
        if not (0.0 <= self.initial_offset_min_confidence <= 1.0):
            raise ValueError("initial_offset_min_confidence must be in [0.0, 1.0]")
        if self.initial_offset_min_confidence < self.low_confidence_threshold:
            raise ValueError(
                "initial_offset_min_confidence must be >= low_confidence_threshold"
            )
        if not (0.0 <= self.coarse_fallback_min_peak_margin <= 1.0):
            raise ValueError("coarse_fallback_min_peak_margin must be in [0.0, 1.0]")
        if self.coarse_fallback_sample_rate <= 0:
            raise ValueError("coarse_fallback_sample_rate must be positive")
        if self.coarse_fallback_max_duration_seconds is not None and (
            not isfinite(self.coarse_fallback_max_duration_seconds)
            or self.coarse_fallback_max_duration_seconds <= 0.0
        ):
            raise ValueError(
                "coarse_fallback_max_duration_seconds must be None or a positive finite value"
            )
        if not isfinite(self.coarse_fallback_max_memory_mb) or self.coarse_fallback_max_memory_mb <= 0.0:
            raise ValueError("coarse_fallback_max_memory_mb must be finite and positive")
        if not (0.0 <= self.coarse_fallback_min_confidence <= 1.0):
            raise ValueError("coarse_fallback_min_confidence must be in [0.0, 1.0]")
        if not (0.0 <= self.coarse_fallback_confidence_margin <= 1.0):
            raise ValueError("coarse_fallback_confidence_margin must be in [0.0, 1.0]")
        if not isfinite(self.max_drift_search_radius_seconds) or self.max_drift_search_radius_seconds <= 0.0:
            raise ValueError("max_drift_search_radius_seconds must be finite and positive")
        for name, value in (
            ("high_confidence_search_radius_seconds", self.high_confidence_search_radius_seconds),
            ("medium_confidence_search_radius_seconds", self.medium_confidence_search_radius_seconds),
            ("low_confidence_search_radius_seconds", self.low_confidence_search_radius_seconds),
        ):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        # The configured radii are policy defaults; the effective radius is
        # always capped at max_drift_search_radius_seconds, so the individual
        # values do not need to be ordered relative to each other.
        if not (0.0 <= self.master_vad_min_overlap_ratio <= 1.0):
            raise ValueError("master_vad_min_overlap_ratio must be in [0.0, 1.0]")
        if not isfinite(self.master_vad_padding_seconds) or self.master_vad_padding_seconds < 0.0:
            raise ValueError("master_vad_padding_seconds must be finite and non-negative")
        if self.master_vad_uncertain_policy not in {"warn", "skip", "reject"}:
            raise ValueError("master_vad_uncertain_policy must be one of: warn, skip, reject")

    def as_dict(self) -> dict[str, float | bool | int | str | None]:
        return {
            "initial_offset_min_confidence": self.initial_offset_min_confidence,
            "high_confidence_threshold": self.high_confidence_threshold,
            "medium_confidence_threshold": self.medium_confidence_threshold,
            "low_confidence_threshold": self.low_confidence_threshold,
            "coarse_fallback_enabled": self.coarse_fallback_enabled,
            "coarse_fallback_sample_rate": self.coarse_fallback_sample_rate,
            "coarse_fallback_min_peak_margin": self.coarse_fallback_min_peak_margin,
            "coarse_fallback_max_duration_seconds": self.coarse_fallback_max_duration_seconds,
            "coarse_fallback_max_memory_mb": self.coarse_fallback_max_memory_mb,
            "coarse_fallback_min_confidence": self.coarse_fallback_min_confidence,
            "coarse_fallback_confidence_margin": self.coarse_fallback_confidence_margin,
            "max_drift_search_radius_seconds": self.max_drift_search_radius_seconds,
            "high_confidence_search_radius_seconds": self.high_confidence_search_radius_seconds,
            "medium_confidence_search_radius_seconds": self.medium_confidence_search_radius_seconds,
            "low_confidence_search_radius_seconds": self.low_confidence_search_radius_seconds,
            "master_vad_filter_enabled": self.master_vad_filter_enabled,
            "master_vad_min_overlap_ratio": self.master_vad_min_overlap_ratio,
            "master_vad_padding_seconds": self.master_vad_padding_seconds,
            "master_vad_uncertain_policy": self.master_vad_uncertain_policy,
        }


@dataclass(frozen=True)
class DriftModelConfig:
    """Shared drift-model selection policy for CLI, API, and GUI runs.

    LinearDrift remains the default/control model. PiecewiseLinearDrift is
    selectable only when ``allow_nonlinear_drift`` is true and the policy is
    ``auto`` or ``piecewise_linear``. Explicit spline runs may prefit a
    piecewise model only to derive requested boundary knots. Kalman smoothing
    is research/experimental and is evaluated only when explicitly requested.
    """

    drift_model: DriftModelName = "auto"
    allow_nonlinear_drift: bool = False
    min_anchors_for_piecewise: int = 6
    min_anchors_per_segment: int = 3
    max_breakpoints: int = 1
    min_residual_improvement_ms: float = 5.0
    min_relative_residual_improvement: float = 0.2
    max_abs_rate_deviation_ppm: float = 1000.0
    max_rate_change_ppm: float = 500.0
    warn_abs_rate_deviation_ppm: float = 200.0
    max_anchor_gap_seconds: float | None = None
    monotonicity_rate_epsilon: float = 1e-9
    min_anchors_for_spline: int = 6
    spline_knot_source: SplineKnotSource = "auto"
    min_knot_spacing_seconds: float = 30.0
    spline_validation_sample_count: int = 1024
    min_anchors_for_kalman: int = 5
    # Kalman process-noise standard deviations are scaled per sqrt(second)
    # in the constant-rate process model: offset in ms/sqrt(s), rate in
    # ppm/sqrt(s). Observation and initial uncertainty fields below are plain
    # one-sigma ms/ppm values.
    kalman_process_offset_noise_ms: float = 5.0
    kalman_process_rate_noise_ppm: float = 50.0
    kalman_observation_noise_ms: float = 30.0
    kalman_initial_offset_uncertainty_ms: float = 250.0
    kalman_initial_rate_uncertainty_ppm: float = 500.0
    kalman_validation_sample_count: int = 1024

    def __post_init__(self) -> None:
        allowed = {"auto", "linear", "piecewise_linear", "spline", "kalman"}
        if self.drift_model not in allowed:
            raise ValueError(f"drift_model must be one of {sorted(allowed)}")
        if self.drift_model in {"piecewise_linear", "spline", "kalman"} and not self.allow_nonlinear_drift:
            raise ValueError(
                f"drift_model={self.drift_model!r} requires allow_nonlinear_drift=True "
                "because non-linear and Kalman drift models remain experimental"
            )
        if self.max_breakpoints < 0:
            raise ValueError("max_breakpoints must be >= 0")
        if self.spline_knot_source not in {"auto", "piecewise_boundaries", "anchors"}:
            raise ValueError("spline_knot_source must be one of: auto, piecewise_boundaries, anchors")
        spline_can_be_evaluated = self.drift_model in {"auto", "spline"} and self.allow_nonlinear_drift
        piecewise_can_be_evaluated = (
            (
                self.drift_model in {"auto", "piecewise_linear"}
                or (
                    self.drift_model == "spline"
                    and self.spline_knot_source in {"auto", "piecewise_boundaries"}
                )
            )
            and self.allow_nonlinear_drift
            and self.max_breakpoints > 0
        )
        if spline_can_be_evaluated:
            if self.spline_knot_source == "piecewise_boundaries" and self.max_breakpoints <= 0:
                raise ValueError(
                    "spline_knot_source='piecewise_boundaries' requires max_breakpoints > 0 "
                    "so a piecewise boundary prefit can be evaluated"
                )
            if self.min_anchors_for_spline < 3:
                raise ValueError("min_anchors_for_spline must be >= 3")
            if not isfinite(self.min_knot_spacing_seconds) or self.min_knot_spacing_seconds < 0:
                raise ValueError("min_knot_spacing_seconds must be finite and >= 0")
            if self.spline_validation_sample_count < 8:
                raise ValueError("spline_validation_sample_count must be >= 8")
        if piecewise_can_be_evaluated:
            if self.min_anchors_for_piecewise < 2:
                raise ValueError("min_anchors_for_piecewise must be >= 2")
            if self.min_anchors_per_segment < 2:
                raise ValueError("min_anchors_per_segment must be >= 2")
            if self.min_anchors_for_piecewise < self.min_anchors_per_segment * 2:
                raise ValueError("min_anchors_for_piecewise must be at least twice min_anchors_per_segment")
        if not isfinite(self.min_residual_improvement_ms) or self.min_residual_improvement_ms < 0:
            raise ValueError("min_residual_improvement_ms must be finite and >= 0")
        if not isfinite(self.min_relative_residual_improvement) or self.min_relative_residual_improvement < 0:
            raise ValueError("min_relative_residual_improvement must be finite and >= 0")
        if not isfinite(self.max_abs_rate_deviation_ppm) or self.max_abs_rate_deviation_ppm <= 0:
            raise ValueError("max_abs_rate_deviation_ppm must be finite and positive")
        if not isfinite(self.max_rate_change_ppm) or self.max_rate_change_ppm < 0:
            raise ValueError("max_rate_change_ppm must be finite and >= 0")
        if not isfinite(self.warn_abs_rate_deviation_ppm) or self.warn_abs_rate_deviation_ppm < 0:
            raise ValueError("warn_abs_rate_deviation_ppm must be finite and >= 0")
        if self.max_anchor_gap_seconds is not None and (
            not isfinite(self.max_anchor_gap_seconds) or self.max_anchor_gap_seconds <= 0
        ):
            raise ValueError("max_anchor_gap_seconds must be None or finite and positive")
        if self.warn_abs_rate_deviation_ppm > self.max_abs_rate_deviation_ppm:
            raise ValueError("warn_abs_rate_deviation_ppm must be <= max_abs_rate_deviation_ppm")
        if not isfinite(self.monotonicity_rate_epsilon) or self.monotonicity_rate_epsilon < 0:
            raise ValueError("monotonicity_rate_epsilon must be finite and >= 0")
        kalman_can_be_evaluated = self.drift_model == "kalman" and self.allow_nonlinear_drift
        if kalman_can_be_evaluated:
            if self.min_anchors_for_kalman < 3:
                raise ValueError("min_anchors_for_kalman must be >= 3")
            if not isfinite(self.kalman_process_offset_noise_ms) or self.kalman_process_offset_noise_ms < 0:
                raise ValueError("kalman_process_offset_noise_ms must be finite and >= 0")
            if not isfinite(self.kalman_process_rate_noise_ppm) or self.kalman_process_rate_noise_ppm < 0:
                raise ValueError("kalman_process_rate_noise_ppm must be finite and >= 0")
            if not isfinite(self.kalman_observation_noise_ms) or self.kalman_observation_noise_ms <= 0:
                raise ValueError("kalman_observation_noise_ms must be finite and positive")
            if not isfinite(self.kalman_initial_offset_uncertainty_ms) or self.kalman_initial_offset_uncertainty_ms <= 0:
                raise ValueError("kalman_initial_offset_uncertainty_ms must be finite and positive")
            if not isfinite(self.kalman_initial_rate_uncertainty_ppm) or self.kalman_initial_rate_uncertainty_ppm <= 0:
                raise ValueError("kalman_initial_rate_uncertainty_ppm must be finite and positive")
            if self.kalman_validation_sample_count < 8:
                raise ValueError("kalman_validation_sample_count must be >= 8")

    def as_dict(self) -> dict[str, str | bool | int | float | None]:
        resolved_model = "linear"
        selection_policy = "linear_default"
        if self.drift_model == "linear":
            selection_policy = "linear_requested"
        elif self.allow_nonlinear_drift and self.drift_model == "piecewise_linear":
            resolved_model = "piecewise_linear"
            selection_policy = "piecewise_experimental"
        elif self.allow_nonlinear_drift and self.drift_model == "spline":
            resolved_model = "spline"
            selection_policy = "spline_experimental"
        elif self.allow_nonlinear_drift and self.drift_model == "kalman":
            resolved_model = "kalman"
            selection_policy = "kalman_research_experimental"
        elif self.allow_nonlinear_drift and self.drift_model == "auto":
            resolved_model = "auto"
            selection_policy = "nonlinear_experimental"
        return {
            "drift_model": self.drift_model,
            "allow_nonlinear_drift": self.allow_nonlinear_drift,
            "resolved_model": resolved_model,
            "selection_policy": selection_policy,
            "min_anchors_for_piecewise": self.min_anchors_for_piecewise,
            "min_anchors_per_segment": self.min_anchors_per_segment,
            "max_breakpoints": self.max_breakpoints,
            "min_residual_improvement_ms": self.min_residual_improvement_ms,
            "min_relative_residual_improvement": self.min_relative_residual_improvement,
            "max_abs_rate_deviation_ppm": self.max_abs_rate_deviation_ppm,
            "max_rate_change_ppm": self.max_rate_change_ppm,
            "warn_abs_rate_deviation_ppm": self.warn_abs_rate_deviation_ppm,
            "max_anchor_gap_seconds": self.max_anchor_gap_seconds,
            "monotonicity_rate_epsilon": self.monotonicity_rate_epsilon,
            "min_anchors_for_spline": self.min_anchors_for_spline,
            "spline_knot_source": self.spline_knot_source,
            "min_knot_spacing_seconds": self.min_knot_spacing_seconds,
            "spline_validation_sample_count": self.spline_validation_sample_count,
            "min_anchors_for_kalman": self.min_anchors_for_kalman,
            "kalman_process_offset_noise_ms": self.kalman_process_offset_noise_ms,
            "kalman_process_rate_noise_ppm": self.kalman_process_rate_noise_ppm,
            "kalman_observation_noise_ms": self.kalman_observation_noise_ms,
            "kalman_initial_offset_uncertainty_ms": self.kalman_initial_offset_uncertainty_ms,
            "kalman_initial_rate_uncertainty_ppm": self.kalman_initial_rate_uncertainty_ppm,
            "kalman_validation_sample_count": self.kalman_validation_sample_count,
        }


@dataclass(frozen=True)
class AnchorSelectionConfig:
    """Shared anchor-selection options for CLI, API, and GUI runs.

    Anchor selection uses a duration-aware target budget so long recordings can
    contribute more drift evidence than short clips while still respecting an
    explicit safety cap.
    """

    anchor_density_per_minute: float = 1.0
    max_anchor_density_per_minute: float = 2.0
    min_anchor_count: int = 5
    max_anchor_count: int | None = 120
    stratified_bin_count: int | None = None
    anchors_per_bin: int | None = None
    min_anchor_duration_seconds: float = 1.0
    base_anchor_duration_seconds: float = 4.0
    max_anchor_duration_seconds: float = 8.0
    min_snr_db: float | None = None
    spectral_flatness_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.anchor_density_per_minute <= 0:
            raise ValueError("anchor_density_per_minute must be positive")
        if self.max_anchor_density_per_minute <= 0:
            raise ValueError("max_anchor_density_per_minute must be positive")
        if self.anchor_density_per_minute > self.max_anchor_density_per_minute:
            raise ValueError("anchor_density_per_minute must be <= max_anchor_density_per_minute")
        if self.min_anchor_count < 0:
            raise ValueError("min_anchor_count must be >= 0")
        if self.max_anchor_count is not None and self.max_anchor_count < 0:
            raise ValueError("max_anchor_count must be >= 0 when set")
        if self.max_anchor_count is not None and self.max_anchor_count < self.min_anchor_count:
            raise ValueError("max_anchor_count must be >= min_anchor_count when set")
        if self.stratified_bin_count is not None and self.stratified_bin_count <= 0:
            raise ValueError("stratified_bin_count must be positive when set")
        if self.anchors_per_bin is not None and self.anchors_per_bin <= 0:
            raise ValueError("anchors_per_bin must be positive when set")
        if not isfinite(self.min_anchor_duration_seconds):
            raise ValueError("min_anchor_duration_seconds must be finite")
        if self.min_anchor_duration_seconds <= 0:
            raise ValueError("min_anchor_duration_seconds must be positive")
        if not isfinite(self.base_anchor_duration_seconds):
            raise ValueError("base_anchor_duration_seconds must be finite")
        if self.base_anchor_duration_seconds <= 0:
            raise ValueError("base_anchor_duration_seconds must be positive")
        if not isfinite(self.max_anchor_duration_seconds):
            raise ValueError("max_anchor_duration_seconds must be finite")
        if self.max_anchor_duration_seconds <= 0:
            raise ValueError("max_anchor_duration_seconds must be positive")
        if self.max_anchor_duration_seconds < self.min_anchor_duration_seconds:
            raise ValueError("max_anchor_duration_seconds must be >= min_anchor_duration_seconds")
        if self.base_anchor_duration_seconds < self.min_anchor_duration_seconds:
            raise ValueError("base_anchor_duration_seconds must be >= min_anchor_duration_seconds")
        if self.base_anchor_duration_seconds > self.max_anchor_duration_seconds:
            raise ValueError("base_anchor_duration_seconds must be <= max_anchor_duration_seconds")
        if self.min_snr_db is not None and not isfinite(self.min_snr_db):
            raise ValueError("min_snr_db must be finite when set")
        if self.spectral_flatness_threshold is not None and not (0.0 <= self.spectral_flatness_threshold <= 1.0):
            raise ValueError("spectral_flatness_threshold must be between 0.0 and 1.0 when set")

    def as_dict(self) -> dict[str, float | int | None]:
        """Return JSON/report-friendly configuration values."""

        return {
            "anchor_density_per_minute": self.anchor_density_per_minute,
            "max_anchor_density_per_minute": self.max_anchor_density_per_minute,
            "min_anchor_count": self.min_anchor_count,
            "max_anchor_count": self.max_anchor_count,
            "stratified_bin_count": self.stratified_bin_count,
            "anchors_per_bin": self.anchors_per_bin,
            "min_anchor_duration_seconds": self.min_anchor_duration_seconds,
            "base_anchor_duration_seconds": self.base_anchor_duration_seconds,
            "max_anchor_duration_seconds": self.max_anchor_duration_seconds,
            "min_snr_db": self.min_snr_db,
            "spectral_flatness_threshold": self.spectral_flatness_threshold,
        }


@dataclass(frozen=True)
class AnchorMatchingConfig:
    """Shared anchor-matching scoring options for CLI, API, and GUI runs.

    Controls NCC peak diagnostics, hard-gate rejection, continuous
    confidence computation, and optional GCC-PHAT agreement signals.
    """

    nms_exclusion_seconds: float = 0.05
    ncc_min_score: float = 0.45
    ncc_min_margin: float = 0.10
    ncc_min_prominence: float = 0.06
    ncc_good_width_seconds: float = 0.005
    ncc_bad_width_seconds: float = 0.050
    ncc_margin_low: float = 0.05
    ncc_margin_high: float = 0.20
    ncc_prominence_low: float = 0.03
    ncc_prominence_high: float = 0.15
    gcc_phat_enabled: bool = True
    gcc_phat_only_when_ambiguous: bool = True
    gcc_phat_agreement_tolerance_seconds: float = 0.030
    min_confidence_for_fit: float = 0.05

    def __post_init__(self) -> None:
        if not (0.0 < self.nms_exclusion_seconds <= 0.5):
            raise ValueError("nms_exclusion_seconds must be in (0.0, 0.5]")
        if not (-1.0 <= self.ncc_min_score < 1.0):
            raise ValueError("ncc_min_score must be in [-1.0, 1.0)")
        if not (0.0 <= self.ncc_min_margin <= 1.0):
            raise ValueError("ncc_min_margin must be in [0.0, 1.0]")
        if not (0.0 <= self.ncc_min_prominence <= 1.0):
            raise ValueError("ncc_min_prominence must be in [0.0, 1.0]")
        if not isfinite(self.ncc_bad_width_seconds):
            raise ValueError("ncc_bad_width_seconds must be finite")
        if not (0.0 < self.ncc_good_width_seconds < self.ncc_bad_width_seconds):
            raise ValueError("ncc_good_width_seconds must be in (0.0, ncc_bad_width_seconds)")
        if not isfinite(self.ncc_margin_high):
            raise ValueError("ncc_margin_high must be finite")
        if not (0.0 <= self.ncc_margin_low < self.ncc_margin_high):
            raise ValueError("ncc_margin_low must be in [0.0, ncc_margin_high)")
        if not isfinite(self.ncc_prominence_high):
            raise ValueError("ncc_prominence_high must be finite")
        if not (0.0 <= self.ncc_prominence_low < self.ncc_prominence_high):
            raise ValueError("ncc_prominence_low must be in [0.0, ncc_prominence_high)")
        if not (0.0 < self.gcc_phat_agreement_tolerance_seconds <= 1.0):
            raise ValueError("gcc_phat_agreement_tolerance_seconds must be in (0.0, 1.0]")
        if not (0.0 <= self.min_confidence_for_fit <= 1.0):
            raise ValueError("min_confidence_for_fit must be in [0.0, 1.0]")


    def as_dict(self) -> dict[str, float | bool]:
        return {
            "nms_exclusion_seconds": self.nms_exclusion_seconds,
            "ncc_min_score": self.ncc_min_score,
            "ncc_min_margin": self.ncc_min_margin,
            "ncc_min_prominence": self.ncc_min_prominence,
            "ncc_good_width_seconds": self.ncc_good_width_seconds,
            "ncc_bad_width_seconds": self.ncc_bad_width_seconds,
            "ncc_margin_low": self.ncc_margin_low,
            "ncc_margin_high": self.ncc_margin_high,
            "ncc_prominence_low": self.ncc_prominence_low,
            "ncc_prominence_high": self.ncc_prominence_high,
            "gcc_phat_enabled": self.gcc_phat_enabled,
            "gcc_phat_only_when_ambiguous": self.gcc_phat_only_when_ambiguous,
            "gcc_phat_agreement_tolerance_seconds": self.gcc_phat_agreement_tolerance_seconds,
            "min_confidence_for_fit": self.min_confidence_for_fit,
        }


DEFAULT_ANCHOR_SELECTION_CONFIG = AnchorSelectionConfig()
DEFAULT_DRIFT_MODEL_CONFIG = DriftModelConfig()
DEFAULT_ANCHOR_MATCHING_CONFIG = AnchorMatchingConfig()
DEFAULT_INITIAL_OFFSET_SAFETY_CONFIG = InitialOffsetSafetyConfig()
