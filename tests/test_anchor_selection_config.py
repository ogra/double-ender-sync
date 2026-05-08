from pathlib import Path

import numpy as np
import pytest

from double_ender_sync.api import AlignmentOptions, build_cli_argv
from double_ender_sync.analysis.anchors import compute_target_anchor_budget, select_anchor_candidates, select_anchor_candidates_with_diagnostics
from double_ender_sync.analysis.vad import SpeechSegment
from double_ender_sync.cli import _resolve_base_anchor_duration, parse_args
from double_ender_sync.config import DEFAULT_ANCHOR_SELECTION_CONFIG, AnchorSelectionConfig

def test_anchor_selection_defaults_are_shared_by_cli_and_api() -> None:
    cli_args = parse_args([
        "--master",
        "master.wav",
        "--track",
        "speaker.wav",
        "--out",
        "output",
    ])
    api_options = AlignmentOptions(master=Path("master.wav"), tracks=[Path("speaker.wav")], out=Path("output"))

    assert cli_args.min_anchor_duration == DEFAULT_ANCHOR_SELECTION_CONFIG.min_anchor_duration_seconds
    assert cli_args.max_anchor_duration == DEFAULT_ANCHOR_SELECTION_CONFIG.max_anchor_duration_seconds
    assert cli_args.anchor_density_per_minute == DEFAULT_ANCHOR_SELECTION_CONFIG.anchor_density_per_minute
    assert cli_args.max_anchor_density_per_minute == DEFAULT_ANCHOR_SELECTION_CONFIG.max_anchor_density_per_minute
    assert cli_args.min_anchor_count == DEFAULT_ANCHOR_SELECTION_CONFIG.min_anchor_count
    assert cli_args.max_anchor_count == DEFAULT_ANCHOR_SELECTION_CONFIG.max_anchor_count
    assert cli_args.stratified_bin_count == DEFAULT_ANCHOR_SELECTION_CONFIG.stratified_bin_count
    assert cli_args.anchors_per_bin == DEFAULT_ANCHOR_SELECTION_CONFIG.anchors_per_bin
    assert api_options.anchor_selection == DEFAULT_ANCHOR_SELECTION_CONFIG

def test_api_threads_anchor_selection_options_to_cli_argv() -> None:
    options = AlignmentOptions(
        master=Path("master.wav"),
        tracks=[Path("speaker.wav")],
        out=Path("output"),
        anchor_selection=AnchorSelectionConfig(
            min_anchor_duration_seconds=0.75,
            base_anchor_duration_seconds=3.0,
            max_anchor_duration_seconds=3.5,
            anchor_density_per_minute=1.5,
            max_anchor_density_per_minute=2.5,
            min_anchor_count=2,
            max_anchor_count=4,
        ),
    )

    argv = build_cli_argv(options)

    assert argv[argv.index("--min-anchor-duration") + 1] == "0.75"
    assert argv[argv.index("--base-anchor-duration") + 1] == "3.0"
    assert argv[argv.index("--max-anchor-duration") + 1] == "3.5"
    assert argv[argv.index("--anchor-density-per-minute") + 1] == "1.5"
    assert argv[argv.index("--max-anchor-density-per-minute") + 1] == "2.5"
    assert argv[argv.index("--min-anchor-count") + 1] == "2"
    assert argv[argv.index("--max-anchor-count") + 1] == "4"

def test_api_preserves_custom_max_anchor_density_for_cli_validation() -> None:
    options = AlignmentOptions(
        master=Path("master.wav"),
        tracks=[Path("speaker.wav")],
        out=Path("output"),
        anchor_selection=AnchorSelectionConfig(
            anchor_density_per_minute=3.0,
            max_anchor_density_per_minute=4.0,
        ),
    )

    cli_args = parse_args(build_cli_argv(options))
    rebuilt_config = AnchorSelectionConfig(
        anchor_density_per_minute=cli_args.anchor_density_per_minute,
        max_anchor_density_per_minute=cli_args.max_anchor_density_per_minute,
        min_anchor_count=cli_args.min_anchor_count,
        max_anchor_count=cli_args.max_anchor_count,
        min_anchor_duration_seconds=cli_args.min_anchor_duration,
        max_anchor_duration_seconds=cli_args.max_anchor_duration,
    )

    assert rebuilt_config.anchor_density_per_minute == 3.0
    assert rebuilt_config.max_anchor_density_per_minute == 4.0

def test_anchor_selection_rejects_nonpositive_sample_rate_with_clear_error() -> None:
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        select_anchor_candidates(
            np.ones(10, dtype=np.float32),
            0,
            [SpeechSegment(start=0.0, end=1.0, confidence=1.0)],
        )

def test_anchor_selection_config_rejects_max_count_below_min_count() -> None:
    with pytest.raises(ValueError, match="max_anchor_count must be >= min_anchor_count"):
        AnchorSelectionConfig(min_anchor_count=10, max_anchor_count=5)

def test_anchor_selection_config_rejects_base_duration_outside_min_max_bounds() -> None:
    with pytest.raises(ValueError, match="base_anchor_duration_seconds must be >= min_anchor_duration_seconds"):
        AnchorSelectionConfig(
            min_anchor_duration_seconds=3.0,
            base_anchor_duration_seconds=2.0,
            max_anchor_duration_seconds=8.0,
        )

    with pytest.raises(ValueError, match="base_anchor_duration_seconds must be <= max_anchor_duration_seconds"):
        AnchorSelectionConfig(
            min_anchor_duration_seconds=1.0,
            base_anchor_duration_seconds=4.0,
            max_anchor_duration_seconds=3.5,
        )


def test_anchor_selection_config_rejects_non_finite_durations() -> None:
    with pytest.raises(ValueError, match="min_anchor_duration_seconds must be finite"):
        AnchorSelectionConfig(min_anchor_duration_seconds=float("nan"))
    with pytest.raises(ValueError, match="base_anchor_duration_seconds must be finite"):
        AnchorSelectionConfig(base_anchor_duration_seconds=float("inf"))
    with pytest.raises(ValueError, match="max_anchor_duration_seconds must be finite"):
        AnchorSelectionConfig(max_anchor_duration_seconds=float("-inf"))

def test_anchor_selection_default_uses_duration_aware_budget_not_fixed_top_five() -> None:
    sample_rate = 100
    samples = np.ones(sample_rate * 20, dtype=np.float32)
    segments = [SpeechSegment(start=float(i * 2), end=float(i * 2 + 1), confidence=1.0 - (i * 0.01)) for i in range(8)]

    anchors = select_anchor_candidates(samples, sample_rate, segments)

    assert DEFAULT_ANCHOR_SELECTION_CONFIG.max_anchor_count == 120
    assert len(anchors) == 5

def test_anchor_selection_config_can_override_current_cap() -> None:
    sample_rate = 100
    samples = np.ones(sample_rate * 20, dtype=np.float32)
    segments = [SpeechSegment(start=float(i * 2), end=float(i * 2 + 1), confidence=1.0 - (i * 0.01)) for i in range(8)]

    anchors = select_anchor_candidates(
        samples,
        sample_rate,
        segments,
        config=AnchorSelectionConfig(min_anchor_count=3, max_anchor_count=3),
    )

    assert len(anchors) == 3

def test_anchor_budget_uses_minimum_for_short_recordings() -> None:
    config = AnchorSelectionConfig(anchor_density_per_minute=1.0, min_anchor_count=5, max_anchor_count=120)

    assert compute_target_anchor_budget(45.0, config) == 5

def test_anchor_budget_scales_with_medium_recordings() -> None:
    config = AnchorSelectionConfig(anchor_density_per_minute=1.0, min_anchor_count=5, max_anchor_count=120)

    assert compute_target_anchor_budget(10 * 60.0, config) == 10

def test_anchor_budget_respects_maximum_safety_cap_for_long_recordings() -> None:
    config = AnchorSelectionConfig(anchor_density_per_minute=2.0, min_anchor_count=5, max_anchor_count=12)

    assert compute_target_anchor_budget(120 * 60.0, config) == 12

def test_long_recording_can_select_more_than_five_valid_anchors() -> None:
    sample_rate = 10
    duration_seconds = 10 * 60
    samples = np.ones(sample_rate * duration_seconds, dtype=np.float32)
    segments = [
        SpeechSegment(start=float(i * 30), end=float(i * 30 + 2), confidence=1.0 - (i * 0.001))
        for i in range(20)
    ]

    anchors = select_anchor_candidates(
        samples,
        sample_rate,
        segments,
        config=AnchorSelectionConfig(anchor_density_per_minute=1.0, min_anchor_count=5, max_anchor_count=120),
    )

    assert len(anchors) == 10

def test_short_recording_does_not_force_more_anchors_than_valid_candidates() -> None:
    sample_rate = 10
    samples = np.ones(sample_rate * 45, dtype=np.float32)
    segments = [SpeechSegment(start=float(i * 10), end=float(i * 10 + 2), confidence=1.0) for i in range(3)]

    anchors = select_anchor_candidates(
        samples,
        sample_rate,
        segments,
        config=AnchorSelectionConfig(anchor_density_per_minute=1.0, min_anchor_count=5, max_anchor_count=120),
    )

    assert len(anchors) == 3

def test_stratified_selection_retains_late_anchor_when_top_scores_cluster_early() -> None:
    sample_rate = 10
    duration_seconds = 3 * 60
    samples = np.ones(sample_rate * duration_seconds, dtype=np.float32)
    early_segments = [
        SpeechSegment(start=float(2 + i * 4), end=float(4 + i * 4), confidence=1.0 - (i * 0.01))
        for i in range(8)
    ]
    distributed_segments = [
        SpeechSegment(start=70.0, end=72.0, confidence=0.40),
        SpeechSegment(start=140.0, end=142.0, confidence=0.35),
    ]

    anchors = select_anchor_candidates(
        samples,
        sample_rate,
        early_segments + distributed_segments,
        config=AnchorSelectionConfig(anchor_density_per_minute=1.0, min_anchor_count=3, max_anchor_count=3),
    )

    selected_bins = {anchor.bin_index for anchor in anchors}
    assert selected_bins == {0, 1, 2}
    assert any(anchor.local_start >= 140.0 for anchor in anchors)

def test_stratified_selection_uses_quality_when_bin_override_exceeds_budget() -> None:
    sample_rate = 10
    duration_seconds = 10 * 60
    samples = np.ones(sample_rate * duration_seconds, dtype=np.float32)
    weak_early_segments = [
        SpeechSegment(start=float(5 + i * 60), end=float(7 + i * 60), confidence=0.10 + (i * 0.01))
        for i in range(5)
    ]
    strong_late_segments = [
        SpeechSegment(start=float(305 + i * 60), end=float(307 + i * 60), confidence=0.90 + (i * 0.01))
        for i in range(5)
    ]

    anchors = select_anchor_candidates(
        samples,
        sample_rate,
        weak_early_segments + strong_late_segments,
        config=AnchorSelectionConfig(
            anchor_density_per_minute=1.0,
            min_anchor_count=5,
            max_anchor_count=5,
            stratified_bin_count=10,
        ),
    )

    assert len(anchors) == 5
    assert {anchor.bin_index for anchor in anchors} == {5, 6, 7, 8, 9}
    assert all(anchor.local_start >= 300.0 for anchor in anchors)

def test_stratified_selection_reports_sparse_coverage_without_failing() -> None:
    from double_ender_sync.analysis.anchors import select_anchor_candidates_with_diagnostics

    sample_rate = 10
    duration_seconds = 4 * 60
    samples = np.ones(sample_rate * duration_seconds, dtype=np.float32)
    segments = [
        SpeechSegment(start=float(2 + i * 4), end=float(4 + i * 4), confidence=1.0 - (i * 0.01))
        for i in range(5)
    ]

    result = select_anchor_candidates_with_diagnostics(
        samples,
        sample_rate,
        segments,
        config=AnchorSelectionConfig(anchor_density_per_minute=1.0, min_anchor_count=4, max_anchor_count=4),
    )

    assert len(result.candidates) == 4
    assert result.diagnostics.stratified_bin_count == 4
    assert result.diagnostics.sparse_bin_count == 3
    assert result.diagnostics.longest_unanchored_span_seconds > 120.0
    assert {warning.code for warning in result.diagnostics.warnings} >= {"SPARSE_ANCHOR_COVERAGE", "LONG_UNANCHORED_SPAN"}

def test_api_threads_stratified_anchor_options_to_cli_argv() -> None:
    options = AlignmentOptions(
        master=Path("master.wav"),
        tracks=[Path("speaker.wav")],
        out=Path("output"),
        anchor_selection=AnchorSelectionConfig(stratified_bin_count=6, anchors_per_bin=2),
    )

    argv = build_cli_argv(options)

    assert argv[argv.index("--stratified-bin-count") + 1] == "6"
    assert argv[argv.index("--anchors-per-bin") + 1] == "2"

def test_anchor_selection_defaults_include_adaptive_duration_options() -> None:
    cli_args = parse_args([
        "--master",
        "master.wav",
        "--track",
        "speaker.wav",
        "--out",
        "output",
    ])

    assert cli_args.base_anchor_duration is None
    assert _resolve_base_anchor_duration(cli_args) == DEFAULT_ANCHOR_SELECTION_CONFIG.base_anchor_duration_seconds
    assert cli_args.min_snr_db == DEFAULT_ANCHOR_SELECTION_CONFIG.min_snr_db
    assert cli_args.spectral_flatness_threshold == DEFAULT_ANCHOR_SELECTION_CONFIG.spectral_flatness_threshold


def test_cli_derives_default_base_anchor_duration_within_custom_bounds() -> None:
    max_only_args = parse_args([
        "--master",
        "master.wav",
        "--track",
        "speaker.wav",
        "--out",
        "output",
        "--max-anchor-duration",
        "3.5",
    ])
    min_only_args = parse_args([
        "--master",
        "master.wav",
        "--track",
        "speaker.wav",
        "--out",
        "output",
        "--min-anchor-duration",
        "5.0",
    ])
    explicit_base_args = parse_args([
        "--master",
        "master.wav",
        "--track",
        "speaker.wav",
        "--out",
        "output",
        "--base-anchor-duration",
        "6.0",
        "--max-anchor-duration",
        "3.5",
    ])

    assert _resolve_base_anchor_duration(max_only_args) == 3.5
    assert _resolve_base_anchor_duration(min_only_args) == 5.0
    assert _resolve_base_anchor_duration(explicit_base_args) == 6.0

def test_api_threads_adaptive_anchor_options_to_cli_argv() -> None:
    options = AlignmentOptions(
        master=Path("master.wav"),
        tracks=[Path("speaker.wav")],
        out=Path("output"),
        anchor_selection=AnchorSelectionConfig(
            base_anchor_duration_seconds=3.0,
            max_anchor_duration_seconds=7.0,
            min_snr_db=5.0,
            spectral_flatness_threshold=0.7,
        ),
    )

    argv = build_cli_argv(options)

    assert argv[argv.index("--base-anchor-duration") + 1] == "3.0"
    assert argv[argv.index("--max-anchor-duration") + 1] == "7.0"
    assert argv[argv.index("--min-snr-db") + 1] == "5.0"
    assert argv[argv.index("--spectral-flatness-threshold") + 1] == "0.7"

def test_adaptive_anchor_duration_keeps_high_snr_tonal_anchor_at_base_duration() -> None:
    sample_rate = 1000
    samples = np.zeros(sample_rate * 20, dtype=np.float32)
    t = np.arange(sample_rate * 6) / sample_rate
    samples[5 * sample_rate : 11 * sample_rate] = (0.8 * np.sin(2 * np.pi * 10 * t)).astype(np.float32)

    result = select_anchor_candidates(
        samples,
        sample_rate,
        [SpeechSegment(start=5.0, end=11.0, confidence=1.0)],
        config=AnchorSelectionConfig(
            min_anchor_count=1,
            max_anchor_count=1,
            base_anchor_duration_seconds=4.0,
            max_anchor_duration_seconds=8.0,
        ),
    )

    assert len(result) == 1
    assert result[0].duration_seconds == pytest.approx(4.0)
    assert result[0].snr_db is not None and result[0].snr_db > 24.0
    assert result[0].spectral_flatness is not None and result[0].spectral_flatness < 0.25

def test_adaptive_anchor_duration_extends_and_downgrades_low_snr_anchor() -> None:
    sample_rate = 1000
    rng = np.random.default_rng(0)
    samples = (0.2 * rng.standard_normal(sample_rate * 20)).astype(np.float32)
    t = np.arange(sample_rate * 6) / sample_rate
    samples[5 * sample_rate : 11 * sample_rate] += (0.05 * np.sin(2 * np.pi * 10 * t)).astype(np.float32)

    result = select_anchor_candidates_with_diagnostics(
        samples,
        sample_rate,
        [SpeechSegment(start=5.0, end=11.0, confidence=1.0)],
        config=AnchorSelectionConfig(
            min_anchor_count=1,
            max_anchor_count=1,
            base_anchor_duration_seconds=4.0,
            max_anchor_duration_seconds=8.0,
        ),
    )

    anchor = result.candidates[0]
    assert anchor.duration_seconds == pytest.approx(6.0)
    assert anchor.quality_multiplier == pytest.approx(2.0)
    assert anchor.confidence < 1.0
    assert result.diagnostics.adaptive_duration_max_seconds == pytest.approx(6.0)

def test_adaptive_anchor_duration_does_not_penalize_full_file_speech_without_noise_context() -> None:
    sample_rate = 1000
    t = np.arange(sample_rate * 6) / sample_rate
    samples = (0.8 * np.sin(2 * np.pi * 10 * t)).astype(np.float32)

    result = select_anchor_candidates_with_diagnostics(
        samples,
        sample_rate,
        [SpeechSegment(start=0.0, end=6.0, confidence=1.0)],
        config=AnchorSelectionConfig(
            min_anchor_count=1,
            max_anchor_count=1,
            base_anchor_duration_seconds=4.0,
            max_anchor_duration_seconds=6.0,
            min_snr_db=12.0,
        ),
    )

    assert len(result.candidates) == 1
    anchor = result.candidates[0]
    assert anchor.snr_db is None
    assert anchor.duration_seconds == pytest.approx(4.0)
    assert anchor.confidence == pytest.approx(1.0)
    assert result.diagnostics.rejected_candidate_counts == {}

def test_adaptive_anchor_snr_uses_bounded_analysis_window() -> None:
    sample_rate = 1000
    samples = np.full(sample_rate * 12, 0.1, dtype=np.float32)
    t = np.arange(sample_rate * 4) / sample_rate
    samples[1 * sample_rate : 5 * sample_rate] = (0.14 * np.sin(2 * np.pi * 10 * t)).astype(np.float32)
    samples[5 * sample_rate : 9 * sample_rate] = 1.0

    result = select_anchor_candidates_with_diagnostics(
        samples,
        sample_rate,
        [SpeechSegment(start=1.0, end=9.0, confidence=1.0)],
        config=AnchorSelectionConfig(
            min_anchor_count=1,
            max_anchor_count=1,
            base_anchor_duration_seconds=2.0,
            max_anchor_duration_seconds=4.0,
            min_snr_db=12.0,
        ),
    )

    assert result.candidates == []
    assert result.diagnostics.rejected_candidate_counts == {"low_snr": 1}


def test_adaptive_anchor_duration_rejects_configured_low_quality_candidates() -> None:
    sample_rate = 1000
    rng = np.random.default_rng(1)
    samples = (0.2 * rng.standard_normal(sample_rate * 20)).astype(np.float32)
    segment = [SpeechSegment(start=5.0, end=11.0, confidence=1.0)]

    result = select_anchor_candidates_with_diagnostics(
        samples,
        sample_rate,
        segment,
        config=AnchorSelectionConfig(
            min_anchor_count=1,
            max_anchor_count=1,
            base_anchor_duration_seconds=4.0,
            max_anchor_duration_seconds=8.0,
            min_snr_db=6.0,
            spectral_flatness_threshold=0.5,
        ),
    )

    assert result.candidates == []
    assert result.diagnostics.rejected_candidate_counts in ({"low_snr": 1}, {"spectrally_flat": 1})
