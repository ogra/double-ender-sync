from dataclasses import dataclass, field
import logging

import numpy as np

from double_ender_sync.analysis.drift import DriftModel, LinearDrift
from double_ender_sync.audio.resample import resample_to_sample_rate
from double_ender_sync.types import AudioTrack

LOGGER = logging.getLogger(__name__)


@dataclass
class UnsupportedRegion:
    start_seconds: float
    end_seconds: float
    reason: str


@dataclass
class MonotonicityCheck:
    passed: bool
    sample_count: int
    epsilon_seconds: float
    min_step_seconds: float | None
    message: str | None = None


@dataclass
class InverseTimeMap:
    local_times_seconds: np.ndarray
    master_times_seconds: np.ndarray
    supported_master_start_seconds: float
    supported_master_end_seconds: float
    monotonicity_check: MonotonicityCheck
    unsupported_regions: list[UnsupportedRegion] = field(default_factory=list)

    def map_master_to_local(self, master_times_seconds: np.ndarray) -> np.ndarray:
        local_times = np.interp(
            master_times_seconds,
            self.master_times_seconds,
            self.local_times_seconds,
            left=np.nan,
            right=np.nan,
        )
        for region in self.unsupported_regions:
            in_gap = (master_times_seconds > region.start_seconds) & (master_times_seconds < region.end_seconds)
            local_times[in_gap] = np.nan
        return local_times


@dataclass
class GlobalAlignmentResult:
    output_samples: np.ndarray
    output_sample_rate: int
    output_duration_seconds: float
    stretch_ratio: float | None
    offset_seconds: float | None
    render_method: str = "linear_resample"
    unsupported_regions: list[UnsupportedRegion] = field(default_factory=list)
    monotonicity_check: MonotonicityCheck | None = None


def apply_global_time_correction(
    track: AudioTrack,
    master: AudioTrack,
    drift_estimate: DriftModel,
    stretch_method: str = "resample",
) -> GlobalAlignmentResult:
    """Apply global correction by rendering against a local-to-master time map.

    The public rendering contract is ``t_master = f(t_local)``.  Linear drift
    with ``stretch_method="resample"`` keeps the original constant-ratio
    implementation for compatibility, while the generic path below samples the
    source audio via an inverse time map so future monotonic drift models can be
    rendered without assuming one constant stretch ratio.
    """
    if track.original_samples is None:
        raise ValueError("track.original_samples is required for global correction")

    if stretch_method not in {"resample", "pitch_preserving"}:
        raise ValueError("stretch_method must be one of: resample, pitch_preserving")

    mono_source = _to_mono(track.original_samples)

    if mono_source.size == 0 or track.duration_seconds <= 0:
        return _render_empty_source_on_master_timeline(
            master=master,
            drift_estimate=drift_estimate,
        )

    if isinstance(drift_estimate, LinearDrift):
        if drift_estimate.stretch_ratio <= 0:
            raise ValueError("stretch_ratio must be positive")
        output = _render_linear_stretch_on_master_timeline(
            source_samples=mono_source,
            source_sample_rate=track.sample_rate,
            master_duration_seconds=master.duration_seconds,
            master_sample_rate=master.sample_rate,
            offset_seconds=drift_estimate.offset_seconds,
            stretch_ratio=drift_estimate.stretch_ratio,
            stretch_method=stretch_method,
        )
        supported_end_seconds = drift_estimate.map_local_to_master(track.duration_seconds)
        return GlobalAlignmentResult(
            output_samples=output,
            output_sample_rate=master.sample_rate,
            output_duration_seconds=master.duration_seconds,
            stretch_ratio=drift_estimate.stretch_ratio,
            offset_seconds=drift_estimate.offset_seconds,
            render_method="linear_pitch_preserving" if stretch_method == "pitch_preserving" else "linear_resample",
            unsupported_regions=_unsupported_regions_for_supported_interval(
                0.0,
                master.duration_seconds,
                drift_estimate.offset_seconds,
                supported_end_seconds,
            ),
            monotonicity_check=MonotonicityCheck(
                passed=True,
                sample_count=2,
                epsilon_seconds=0.0,
                min_step_seconds=supported_end_seconds - drift_estimate.offset_seconds,
            ),
        )

    if stretch_method == "pitch_preserving":
        raise ValueError("pitch_preserving rendering currently supports LinearDrift only")

    inverse_map = build_inverse_time_map(
        drift_model=drift_estimate,
        local_duration_seconds=track.duration_seconds,
    )
    output = render_with_inverse_time_map(
        source_samples=mono_source,
        source_sample_rate=track.sample_rate,
        output_duration_seconds=master.duration_seconds,
        output_sample_rate=master.sample_rate,
        inverse_time_map=inverse_map,
    )

    return GlobalAlignmentResult(
        output_samples=output,
        output_sample_rate=master.sample_rate,
        output_duration_seconds=master.duration_seconds,
        stretch_ratio=None,  # non-LinearDrift models do not expose stretch_ratio/offset_seconds
        offset_seconds=None,
        render_method="inverse_time_map",
        unsupported_regions=_unsupported_regions_for_inverse_map(
            0.0,
            master.duration_seconds,
            inverse_map,
        ),
        monotonicity_check=inverse_map.monotonicity_check,
    )


def _render_empty_source_on_master_timeline(
    master: AudioTrack,
    drift_estimate: DriftModel,
) -> GlobalAlignmentResult:
    output_length = int(round(master.duration_seconds * master.sample_rate))
    unsupported_regions = []
    if master.duration_seconds > 0:
        unsupported_regions.append(UnsupportedRegion(0.0, master.duration_seconds, "empty_source_track"))
    return GlobalAlignmentResult(
        output_samples=np.zeros(output_length, dtype=np.float32),
        output_sample_rate=master.sample_rate,
        output_duration_seconds=master.duration_seconds,
        stretch_ratio=drift_estimate.stretch_ratio if isinstance(drift_estimate, LinearDrift) else None,
        offset_seconds=drift_estimate.offset_seconds if isinstance(drift_estimate, LinearDrift) else None,
        render_method="silence_empty_source",
        unsupported_regions=unsupported_regions,
        monotonicity_check=MonotonicityCheck(
            passed=True,
            sample_count=0,
            epsilon_seconds=0.0,
            min_step_seconds=None,
            message="source track is empty or has zero duration; rendered silence",
        ),
    )


def _render_linear_stretch_on_master_timeline(
    source_samples: np.ndarray,
    source_sample_rate: int,
    master_duration_seconds: float,
    master_sample_rate: int,
    offset_seconds: float,
    stretch_ratio: float,
    stretch_method: str,
) -> np.ndarray:
    output_length = int(round(master_duration_seconds * master_sample_rate))
    output = np.zeros(output_length, dtype=np.float32)
    if output_length == 0 or source_samples.size == 0:
        return output

    if stretch_method == "pitch_preserving":
        stretched = _stretch_samples(source_samples, stretch_ratio, stretch_method)
        return _place_rendered_samples_on_master_timeline(
            rendered_samples=stretched,
            rendered_sample_rate=source_sample_rate,
            master_sample_rate=master_sample_rate,
            offset_seconds=offset_seconds,
            output=output,
        )

    sample_positions = np.arange(source_samples.shape[0], dtype=np.float64)
    source_values = source_samples.astype(np.float64, copy=False)
    for start_index in range(0, output_length, _RENDER_CHUNK_SIZE_SAMPLES):
        end_index = min(start_index + _RENDER_CHUNK_SIZE_SAMPLES, output_length)
        master_times = np.arange(start_index, end_index, dtype=np.float64) / float(master_sample_rate)
        local_times = (master_times - offset_seconds) / stretch_ratio
        source_positions = local_times * float(source_sample_rate)
        valid = (source_positions >= 0.0) & (source_positions <= float(source_samples.shape[0] - 1) + 1e-9)
        if np.any(valid):
            clipped_positions = np.clip(source_positions[valid], 0.0, float(source_samples.shape[0] - 1))
            chunk = output[start_index:end_index]
            chunk[valid] = np.interp(
                clipped_positions,
                sample_positions,
                source_values,
            ).astype(np.float32, copy=False)
    return output


def _place_rendered_samples_on_master_timeline(
    rendered_samples: np.ndarray,
    rendered_sample_rate: int,
    master_sample_rate: int,
    offset_seconds: float,
    output: np.ndarray,
) -> np.ndarray:
    if rendered_sample_rate != master_sample_rate:
        rendered_samples = resample_to_sample_rate(rendered_samples, rendered_sample_rate, master_sample_rate)

    start_index = int(round(offset_seconds * master_sample_rate))
    src_start = 0
    if start_index < 0:
        src_start = -start_index
        start_index = 0

    available = min(rendered_samples.shape[0] - src_start, output.shape[0] - start_index)
    if available > 0:
        output[start_index : start_index + available] = rendered_samples[src_start : src_start + available]
    return output


_RENDER_CHUNK_SIZE_SAMPLES = 262_144
_INTERNAL_GAP_STEP_MULTIPLIER = 8.0


def build_inverse_time_map(
    drift_model: DriftModel,
    local_duration_seconds: float,
    sample_count: int = 4096,
    monotonicity_epsilon_seconds: float = 1e-9,
    internal_gap_step_multiplier: float = _INTERNAL_GAP_STEP_MULTIPLIER,
) -> InverseTimeMap:
    """Build a table-based inverse for a monotonic local-to-master drift model."""
    if local_duration_seconds <= 0:
        raise ValueError("local_duration_seconds must be positive when building an inverse time map")
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")

    local_times = np.linspace(0.0, local_duration_seconds, num=sample_count, dtype=np.float64)
    master_times = np.array([drift_model.map_local_to_master(float(t)) for t in local_times], dtype=np.float64)

    if not np.all(np.isfinite(master_times)):
        check = MonotonicityCheck(
            passed=False,
            sample_count=sample_count,
            epsilon_seconds=monotonicity_epsilon_seconds,
            min_step_seconds=None,
            message="drift model produced non-finite master times",
        )
        raise ValueError(check.message)

    steps = np.diff(master_times)
    min_step = float(np.min(steps)) if steps.size else None
    if steps.size and np.any(steps <= monotonicity_epsilon_seconds):
        check = MonotonicityCheck(
            passed=False,
            sample_count=sample_count,
            epsilon_seconds=monotonicity_epsilon_seconds,
            min_step_seconds=min_step,
            message=(
                f"drift model must be strictly monotonic and invertible over the render domain"
                f" (sample_count={sample_count}, epsilon_seconds={monotonicity_epsilon_seconds!r},"
                f" min_step_seconds={min_step!r})"
            ),
        )
        raise ValueError(check.message)

    internal_regions = _detect_internal_unsupported_regions(
        drift_model,
        local_times,
        master_times,
        steps,
        internal_gap_step_multiplier,
        monotonicity_epsilon_seconds,
    )
    check = MonotonicityCheck(
        passed=True,
        sample_count=sample_count,
        epsilon_seconds=monotonicity_epsilon_seconds,
        min_step_seconds=min_step,
        message=None if not internal_regions else f"detected {len(internal_regions)} internal unsupported master-time gap(s)",
    )
    return InverseTimeMap(
        local_times_seconds=local_times,
        master_times_seconds=master_times,
        supported_master_start_seconds=float(master_times[0]),
        supported_master_end_seconds=float(master_times[-1]),
        monotonicity_check=check,
        unsupported_regions=internal_regions,
    )


def _detect_internal_unsupported_regions(
    drift_model: DriftModel,
    local_times: np.ndarray,
    master_times: np.ndarray,
    steps: np.ndarray,
    internal_gap_step_multiplier: float,
    epsilon_seconds: float,
) -> list[UnsupportedRegion]:
    if steps.size == 0 or internal_gap_step_multiplier <= 1.0:
        return []

    median_step = float(np.median(steps))
    gap_threshold = max(median_step * internal_gap_step_multiplier, epsilon_seconds)
    regions: list[UnsupportedRegion] = []
    for index, step in enumerate(steps):
        if float(step) <= gap_threshold:
            continue

        refined_region = _refine_internal_gap_candidate(
            drift_model=drift_model,
            local_start_seconds=float(local_times[index]),
            local_end_seconds=float(local_times[index + 1]),
            master_start_seconds=float(master_times[index]),
            master_end_seconds=float(master_times[index + 1]),
            gap_threshold_seconds=gap_threshold,
            epsilon_seconds=epsilon_seconds,
        )
        if refined_region is not None:
            regions.append(refined_region)
    return regions


def _refine_internal_gap_candidate(
    drift_model: DriftModel,
    local_start_seconds: float,
    local_end_seconds: float,
    master_start_seconds: float,
    master_end_seconds: float,
    gap_threshold_seconds: float,
    epsilon_seconds: float,
    max_refinements: int = 32,
) -> UnsupportedRegion | None:
    """Confirm a coarse large step is a discontinuity before silencing it.

    Continuous high-rate sections can produce large coarse table steps that shrink
    as the local interval is subdivided. A real unsupported master-time gap keeps
    a large master-time jump concentrated in an ever-smaller local interval, so
    only candidates that survive adaptive refinement are reported as unsupported.
    """
    local_start = local_start_seconds
    local_end = local_end_seconds
    master_start = master_start_seconds
    master_end = master_end_seconds

    for _ in range(max_refinements):
        total_step = master_end - master_start
        if total_step <= gap_threshold_seconds:
            return None

        local_mid = (local_start + local_end) / 2.0
        if local_mid <= local_start or local_mid >= local_end:
            break

        master_mid = float(drift_model.map_local_to_master(local_mid))
        if not np.isfinite(master_mid):
            return UnsupportedRegion(master_start, master_end, "internal_drift_model_gap")

        left_step = master_mid - master_start
        right_step = master_end - master_mid
        if left_step <= epsilon_seconds or right_step <= epsilon_seconds:
            return UnsupportedRegion(master_start, master_end, "internal_drift_model_gap")

        if left_step >= right_step:
            local_end = local_mid
            master_end = master_mid
        else:
            local_start = local_mid
            master_start = master_mid

    if master_end - master_start > gap_threshold_seconds:
        return UnsupportedRegion(master_start, master_end, "internal_drift_model_gap")
    return None


def render_with_inverse_time_map(
    source_samples: np.ndarray,
    source_sample_rate: int,
    output_duration_seconds: float,
    output_sample_rate: int,
    inverse_time_map: InverseTimeMap,
) -> np.ndarray:
    """Render source audio onto the master timeline using inverse lookup.

    Output samples whose master times fall outside the model support or source
    sample domain remain silent. Linear interpolation is used explicitly for the
    current MVP renderer; higher-quality interpolation can be added behind this
    extension point later.
    """
    output_length = int(round(output_duration_seconds * output_sample_rate))
    output = np.zeros(output_length, dtype=np.float32)
    if output_length == 0 or source_samples.size == 0:
        return output

    sample_positions = np.arange(source_samples.shape[0], dtype=np.float64)
    source_values = source_samples.astype(np.float64, copy=False)
    for start_index in range(0, output_length, _RENDER_CHUNK_SIZE_SAMPLES):
        end_index = min(start_index + _RENDER_CHUNK_SIZE_SAMPLES, output_length)
        master_times = np.arange(start_index, end_index, dtype=np.float64) / float(output_sample_rate)
        local_times = inverse_time_map.map_master_to_local(master_times)
        valid = np.isfinite(local_times)
        source_positions = local_times * float(source_sample_rate)
        valid &= source_positions >= 0.0
        valid &= source_positions <= float(source_samples.shape[0] - 1) + 1e-9

        if np.any(valid):
            clipped_positions = np.clip(source_positions[valid], 0.0, float(source_samples.shape[0] - 1))
            chunk = output[start_index:end_index]
            chunk[valid] = np.interp(
                clipped_positions,
                sample_positions,
                source_values,
            ).astype(np.float32, copy=False)
    return output


def _unsupported_regions_for_inverse_map(
    master_start_seconds: float,
    master_end_seconds: float,
    inverse_map: InverseTimeMap,
) -> list[UnsupportedRegion]:
    regions = _unsupported_regions_for_supported_interval(
        master_start_seconds,
        master_end_seconds,
        inverse_map.supported_master_start_seconds,
        inverse_map.supported_master_end_seconds,
    )
    for region in inverse_map.unsupported_regions:
        clipped_start = max(master_start_seconds, region.start_seconds)
        clipped_end = min(master_end_seconds, region.end_seconds)
        if clipped_start < clipped_end:
            regions.append(UnsupportedRegion(clipped_start, clipped_end, region.reason))
    regions.sort(key=lambda region: region.start_seconds)
    return regions


def _unsupported_regions_for_supported_interval(
    master_start_seconds: float,
    master_end_seconds: float,
    supported_start_seconds: float,
    supported_end_seconds: float,
) -> list[UnsupportedRegion]:
    regions: list[UnsupportedRegion] = []
    clipped_supported_start = max(master_start_seconds, supported_start_seconds)
    clipped_supported_end = min(master_end_seconds, supported_end_seconds)
    if clipped_supported_start > master_start_seconds:
        regions.append(UnsupportedRegion(master_start_seconds, clipped_supported_start, "before_drift_model_support"))
    if clipped_supported_end < master_end_seconds:
        regions.append(UnsupportedRegion(clipped_supported_end, master_end_seconds, "after_drift_model_support"))
    if clipped_supported_end < clipped_supported_start:
        return [UnsupportedRegion(master_start_seconds, master_end_seconds, "outside_drift_model_support")]
    return regions


def _to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples.astype(np.float32, copy=False)
    return np.mean(samples, axis=1, dtype=np.float32)


def _stretch_samples(samples: np.ndarray, stretch_ratio: float, stretch_method: str) -> np.ndarray:
    if stretch_method == "resample":
        return _resample_by_ratio(samples, stretch_ratio)
    if stretch_method == "pitch_preserving":
        return _pitch_preserving_time_stretch(samples, stretch_ratio)
    raise ValueError("stretch_method must be one of: resample, pitch_preserving")



def _resample_by_ratio(samples: np.ndarray, stretch_ratio: float) -> np.ndarray:
    if samples.size == 0:
        return samples.astype(np.float32, copy=False)
    target_len = max(1, int(round(samples.shape[0] * stretch_ratio)))
    source_positions = np.linspace(0, samples.shape[0] - 1, num=samples.shape[0], dtype=np.float64)
    target_positions = np.linspace(0, samples.shape[0] - 1, num=target_len, dtype=np.float64)
    stretched = np.interp(target_positions, source_positions, samples.astype(np.float64, copy=False))
    return stretched.astype(np.float32, copy=False)


def _pitch_preserving_time_stretch(samples: np.ndarray, stretch_ratio: float) -> np.ndarray:
    if samples.size == 0:
        return samples.astype(np.float32, copy=False)

    import importlib

    try:
        librosa = importlib.import_module("librosa")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pitch_preserving stretch requires librosa. Install with: pip install \"double-ender-sync[stretch]\""
        ) from exc
    rate = 1.0 / stretch_ratio
    stretched = librosa.effects.time_stretch(samples.astype(np.float32, copy=False), rate=rate)

    target_len = max(1, int(round(samples.shape[0] * stretch_ratio)))
    if stretched.shape[0] > target_len:
        stretched = stretched[:target_len]
    elif stretched.shape[0] < target_len:
        pad = np.zeros(target_len - stretched.shape[0], dtype=stretched.dtype)
        stretched = np.concatenate([stretched, pad])

    LOGGER.debug("pitch_preserving stretch applied rate=%.8f target_len=%d", rate, target_len)
    return stretched.astype(np.float32, copy=False)
