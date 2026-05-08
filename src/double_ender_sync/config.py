from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


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
