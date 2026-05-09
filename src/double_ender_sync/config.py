from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal


SplineKnotSource = Literal["auto", "piecewise_boundaries", "anchors"]


DriftModelName = Literal["auto", "linear", "piecewise_linear", "spline", "kalman"]


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


DEFAULT_ANCHOR_SELECTION_CONFIG = AnchorSelectionConfig()
DEFAULT_DRIFT_MODEL_CONFIG = DriftModelConfig()
