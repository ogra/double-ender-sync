import numpy as np
import pytest

from double_ender_sync.alignment.offset import estimate_initial_offset_with_safety_net
from double_ender_sync.analysis.anchors import AnchorCandidate, select_anchor_candidates
from double_ender_sync.analysis.drift import match_anchors_for_drift
from double_ender_sync.analysis.vad import SpeechSegment, detect_speech_segments
from double_ender_sync.config import InitialOffsetSafetyConfig
from double_ender_sync.report.report import serialize_offset


def _make_tone(sample_rate: int, duration: float, frequency: float, amplitude: float = 0.25) -> np.ndarray:
    t = np.arange(int(sample_rate * duration)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def test_high_confidence_anchor_estimate_keeps_offset_and_method() -> None:
    sample_rate = 16000
    speech_duration = 1.2
    silence_lead = 0.4

    speech = _make_tone(sample_rate, speech_duration, 310)
    local = speech
    master = np.concatenate([np.zeros(int(sample_rate * silence_lead), dtype=np.float32), speech])

    segments = detect_speech_segments(local, sample_rate=sample_rate)
    anchors = select_anchor_candidates(local, sample_rate, segments, min_anchor_duration=0.5, max_anchor_duration=1.0)
    estimate = estimate_initial_offset_with_safety_net(local, master, sample_rate, anchors)

    assert estimate is not None
    assert abs(estimate.offset_seconds - silence_lead) < 0.03
    assert estimate.confidence > 0.4
    assert estimate.estimation_method == "anchor_ncc"
    assert estimate.confidence_band in {"high", "medium"}
    assert estimate.fallback_attempted is False
    assert estimate.fallback_selected is False
    if estimate.confidence_band == "high":
        assert estimate.selected_drift_search_radius_seconds == pytest.approx(6.0)
    else:
        assert estimate.selected_drift_search_radius_seconds == pytest.approx(12.0)


def test_low_confidence_estimate_triggers_fallback_and_recovers() -> None:
    """Anchor NCC sees a repeated early phrase and produces a wrong low-confidence offset.

    The coarse whole-recording fallback should recover the true offset.
    """
    sample_rate = 8000
    true_offset = 2.5
    tone_freq = 440

    # Build a local track with two identical tone bursts.
    burst = _make_tone(sample_rate, 0.8, tone_freq)
    silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    local = np.concatenate([burst, silence, burst, silence, _make_tone(sample_rate, 0.8, 660)])

    # Build a master that contains the same content starting at true_offset, plus
    # a decoy burst at 0.0 to confuse the first anchor.
    master = np.zeros(int(sample_rate * (local.size / sample_rate + true_offset + 1.0)), dtype=np.float32)
    decoy = burst[: min(len(burst), int(sample_rate * 0.8))]
    master[: len(decoy)] = decoy
    start = int(true_offset * sample_rate)
    master[start : start + len(local)] = local

    # Force a single short anchor at the first burst so the anchor NCC path
    # returns the decoy offset with low confidence.
    anchors = [AnchorCandidate(local_start=0.0, local_end=0.8, confidence=0.3, rms=0.1)]

    safety = InitialOffsetSafetyConfig(
        initial_offset_min_confidence=0.50,
        coarse_fallback_min_peak_margin=0.05,
        coarse_fallback_min_confidence=0.50,
    )
    estimate = estimate_initial_offset_with_safety_net(local, master, sample_rate, anchors, safety_config=safety)

    assert estimate is not None
    assert estimate.fallback_attempted is True
    assert estimate.fallback_selected is True
    assert estimate.estimation_method == "coarse_fft_fallback"
    assert abs(estimate.offset_seconds - true_offset) < 0.3


def test_diagnostics_serialize_without_removing_existing_fields() -> None:
    sample_rate = 16000
    speech = _make_tone(sample_rate, 1.2, 310)
    local = speech
    master = np.concatenate([np.zeros(int(sample_rate * 0.4), dtype=np.float32), speech])

    segments = detect_speech_segments(local, sample_rate=sample_rate)
    anchors = select_anchor_candidates(local, sample_rate, segments, min_anchor_duration=0.5, max_anchor_duration=1.0)
    estimate = estimate_initial_offset_with_safety_net(local, master, sample_rate, anchors)

    serialized = serialize_offset(estimate)
    assert serialized is not None
    assert "offset_seconds" in serialized
    assert "confidence" in serialized
    assert "local_anchor_start" in serialized
    assert "master_anchor_start" in serialized
    assert "score" in serialized
    assert "estimation_method" in serialized
    assert "confidence_band" in serialized


def test_ambiguous_both_estimates_remain_low_confidence() -> None:
    sample_rate = 8000
    rng = np.random.default_rng(0)
    tone = _make_tone(sample_rate, 1.0, 440, amplitude=0.25)
    # Add enough noise that the normalized correlation peak is below the
    # fallback confidence gate, while still above the hard 0.10 score floor.
    # The anchor confidence is calibrated to fall in the "low" band so the
    # fallback is attempted, but the fallback peak is too weak to be selected.
    noise_std = 0.35
    local = tone + rng.normal(0.0, noise_std, size=tone.shape).astype(np.float32)
    master = tone + rng.normal(0.0, noise_std, size=tone.shape).astype(np.float32)

    anchors = [AnchorCandidate(local_start=0.0, local_end=1.0, confidence=0.6, rms=0.1)]
    safety = InitialOffsetSafetyConfig(
        initial_offset_min_confidence=0.50,
        coarse_fallback_min_peak_margin=0.05,
        coarse_fallback_min_confidence=0.65,
    )
    estimate = estimate_initial_offset_with_safety_net(local, master, sample_rate, anchors, safety_config=safety)

    # The anchor should be below the fallback threshold and the fallback should
    # not be clearly better, so the result stays uncertain.
    assert estimate is not None
    assert estimate.confidence_band in {"low", "medium"}
    assert estimate.fallback_attempted is True
    assert estimate.fallback_selected is False
    assert "coarse_offset_fallback_attempted_but_rejected" in estimate.warnings
    assert estimate.confidence < safety.high_confidence_threshold


def test_coordinate_system_positive_and_negative_offsets() -> None:
    sample_rate = 8000
    tone = _make_tone(sample_rate, 1.0, 330)
    silence = np.zeros(int(sample_rate * 1.0), dtype=np.float32)

    # Positive offset: master has leading silence.
    local_pos = tone
    master_pos = np.concatenate([silence, tone])
    estimate_pos = estimate_initial_offset_with_safety_net(
        local_pos, master_pos, sample_rate, []
    )
    assert estimate_pos is not None
    assert abs(estimate_pos.offset_seconds - 1.0) < 0.15

    # Negative offset: local has leading silence, master starts at local's tone.
    local_neg = np.concatenate([silence, tone])
    master_neg = tone
    estimate_neg = estimate_initial_offset_with_safety_net(
        local_neg, master_neg, sample_rate, []
    )
    assert estimate_neg is not None
    assert abs(estimate_neg.offset_seconds - (-1.0)) < 0.15
    assert estimate_neg.master_anchor_start == pytest.approx(0.0)
    assert estimate_neg.local_anchor_start == pytest.approx(1.0)
    assert estimate_neg.offset_seconds == pytest.approx(
        estimate_neg.master_anchor_start - estimate_neg.local_anchor_start
    )


def test_fallback_not_called_when_anchor_high_confidence() -> None:
    sample_rate = 16000
    speech = _make_tone(sample_rate, 1.2, 310)
    local = speech
    master = np.concatenate([np.zeros(int(sample_rate * 0.4), dtype=np.float32), speech])

    segments = detect_speech_segments(local, sample_rate=sample_rate)
    anchors = select_anchor_candidates(local, sample_rate, segments, min_anchor_duration=0.5, max_anchor_duration=1.0)
    safety = InitialOffsetSafetyConfig(coarse_fallback_enabled=True)
    estimate = estimate_initial_offset_with_safety_net(local, master, sample_rate, anchors, safety_config=safety)

    assert estimate is not None
    assert estimate.fallback_attempted is False
    assert estimate.fallback_selected is False
    assert estimate.estimation_method == "anchor_ncc"


def test_initial_offset_min_confidence_gates_fallback_attempt() -> None:
    """Fallback attempt is gated by initial_offset_min_confidence, not the band thresholds.

    A medium-band anchor estimate should trigger the fallback when the user raises
    --initial-offset-min-confidence above the medium threshold.
    """
    sample_rate = 8000
    tone = _make_tone(sample_rate, 1.0, 440)
    silence = np.zeros(int(sample_rate * 1.5), dtype=np.float32)
    master = np.concatenate([silence, tone])
    local = tone

    anchors = [AnchorCandidate(local_start=0.0, local_end=1.0, confidence=0.6, rms=0.1)]

    low_threshold = InitialOffsetSafetyConfig(
        initial_offset_min_confidence=0.50,
        coarse_fallback_enabled=True,
    )
    estimate_low = estimate_initial_offset_with_safety_net(
        local, master, sample_rate, anchors, safety_config=low_threshold
    )

    high_threshold = InitialOffsetSafetyConfig(
        initial_offset_min_confidence=0.80,
        coarse_fallback_enabled=True,
    )
    estimate_high = estimate_initial_offset_with_safety_net(
        local, master, sample_rate, anchors, safety_config=high_threshold
    )

    assert estimate_low is not None
    assert estimate_high is not None
    assert estimate_low.confidence_band == "medium"
    assert estimate_low.fallback_attempted is False
    assert estimate_high.fallback_attempted is True


def test_high_confidence_preserves_six_second_radius() -> None:
    safety = InitialOffsetSafetyConfig()
    from double_ender_sync.alignment.offset import _select_drift_search_radius

    radius, reason = _select_drift_search_radius("high", False, True, safety)
    assert radius == pytest.approx(6.0)
    assert "high_confidence" in reason


def test_lower_confidence_increases_radius_deterministically() -> None:
    safety = InitialOffsetSafetyConfig()
    from double_ender_sync.alignment.offset import _select_drift_search_radius

    medium_radius, _ = _select_drift_search_radius("medium", False, True, safety)
    low_radius, _ = _select_drift_search_radius("low", False, True, safety)
    low_fallback_radius, _ = _select_drift_search_radius("low", True, True, safety)

    assert medium_radius == pytest.approx(12.0)
    assert low_radius == pytest.approx(20.0)
    assert low_fallback_radius == pytest.approx(20.0)


def test_radius_capped_for_very_low_confidence() -> None:
    safety = InitialOffsetSafetyConfig(
        high_confidence_search_radius_seconds=2.0,
        medium_confidence_search_radius_seconds=5.0,
        low_confidence_search_radius_seconds=20.0,
        max_drift_search_radius_seconds=10.0,
    )
    from double_ender_sync.alignment.offset import _select_drift_search_radius

    radius, _ = _select_drift_search_radius("low", True, True, safety)
    assert radius == pytest.approx(10.0)


def test_no_usable_initial_estimate_returns_zero_radius() -> None:
    safety = InitialOffsetSafetyConfig()
    from double_ender_sync.alignment.offset import _select_drift_search_radius

    radius, reason = _select_drift_search_radius("failed", False, False, safety)
    assert radius == pytest.approx(0.0)
    assert "no_usable" in reason


def test_wider_searches_reject_low_quality_matches() -> None:
    """A wider search radius should not admit matches that fail the hard gate."""
    sample_rate = 8000
    tone = _make_tone(sample_rate, 0.8, 220)
    local = tone
    master = np.concatenate([np.zeros(int(sample_rate * 0.5), dtype=np.float32), tone])

    segments = detect_speech_segments(local, sample_rate=sample_rate)
    anchors = select_anchor_candidates(local, sample_rate, segments, min_anchor_duration=0.3, max_anchor_duration=0.8)
    estimate = estimate_initial_offset_with_safety_net(local, master, sample_rate, anchors)
    assert estimate is not None

    # Use a permissive matching config so the true match is accepted, while any
    # low-quality alternative still fails the hard gate.
    from double_ender_sync.config import AnchorMatchingConfig
    permissive_matching = AnchorMatchingConfig(
        ncc_min_score=0.10, ncc_min_margin=0.0, ncc_min_prominence=0.0, min_confidence_for_fit=0.05
    )
    matches = match_anchors_for_drift(
        local, master, sample_rate, anchors,
        initial_offset_seconds=estimate.offset_seconds,
        search_radius_seconds=20.0,
        matching_config=permissive_matching,
    )
    assert any(match.rejected_reason is None for match in matches)


def test_vad_rejected_match_preserves_ncc_diagnostics() -> None:
    """VAD-rejected anchors must keep their NCC scores/diagnostics."""
    sample_rate = 8000
    tone = _make_tone(sample_rate, 0.8, 330, amplitude=0.25)
    silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)

    local = tone
    master = np.concatenate([silence, tone])

    segments = detect_speech_segments(local, sample_rate=sample_rate)
    anchors = select_anchor_candidates(local, sample_rate, segments, min_anchor_duration=0.3, max_anchor_duration=0.8)

    # Speech segment placed away from the true match so VAD rejects it.
    master_speech_segments = [SpeechSegment(start=2.0, end=2.1, confidence=1.0)]
    safety = InitialOffsetSafetyConfig(master_vad_filter_enabled=True)
    from double_ender_sync.config import AnchorMatchingConfig
    permissive_matching = AnchorMatchingConfig(
        ncc_min_score=0.0, ncc_min_margin=0.0, ncc_min_prominence=0.0, min_confidence_for_fit=0.0
    )
    matches = match_anchors_for_drift(
        local, master, sample_rate, anchors,
        initial_offset_seconds=0.5,
        search_radius_seconds=6.0,
        matching_config=permissive_matching,
        master_speech_segments=master_speech_segments,
        safety_config=safety,
    )
    rejected = [
        match for match in matches
        if match.rejected_reason in {"master_vad_no_overlap", "master_vad_low_overlap_ratio"}
    ]
    assert len(rejected) >= 1
    match = rejected[0]
    assert match.score > 0.0
    assert match.ncc_best_score is not None and match.ncc_best_score > 0.0
    assert match.ncc_margin is not None
    assert match.ncc_prominence is not None and match.ncc_prominence > 0.0
    assert match.confidence == 0.0


def test_vad_filter_falls_back_to_lower_ncc_peak_inside_speech() -> None:
    """When the highest NCC peak is outside master VAD, lower peaks are tried."""
    sample_rate = 8000
    real_tone = _make_tone(sample_rate, 0.8, 330, amplitude=0.25)
    decoy_tone = _make_tone(sample_rate, 0.8, 330, amplitude=0.50)
    silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)

    local = real_tone
    # Decoy is louder and therefore the best NCC peak, but it sits outside the
    # master VAD speech interval. The quieter real match is inside the interval.
    master = np.concatenate([silence, real_tone, silence, decoy_tone])

    segments = detect_speech_segments(local, sample_rate=sample_rate)
    anchors = select_anchor_candidates(local, sample_rate, segments, min_anchor_duration=0.3, max_anchor_duration=0.8)

    # VAD only marks the real match region as speech.
    master_speech_segments = [SpeechSegment(start=0.4, end=1.5, confidence=1.0)]
    safety = InitialOffsetSafetyConfig(master_vad_filter_enabled=True)
    from double_ender_sync.config import AnchorMatchingConfig
    permissive_matching = AnchorMatchingConfig(
        ncc_min_score=0.0, ncc_min_margin=0.0, ncc_min_prominence=0.0, min_confidence_for_fit=0.0
    )
    matches = match_anchors_for_drift(
        local, master, sample_rate, anchors,
        initial_offset_seconds=0.5,
        search_radius_seconds=6.0,
        matching_config=permissive_matching,
        master_speech_segments=master_speech_segments,
        safety_config=safety,
    )
    accepted = [match for match in matches if match.rejected_reason is None]
    assert len(accepted) >= 1
    # The accepted match should land on the real match, not the decoy.
    assert any(abs(match.master_start - 0.5) < 0.2 for match in accepted)


def test_local_speech_anchor_in_master_silence_rejected() -> None:
    sample_rate = 8000
    tone = _make_tone(sample_rate, 0.8, 330)
    local = tone
    master = np.concatenate([np.zeros(int(sample_rate * 0.5), dtype=np.float32), tone])

    segments = detect_speech_segments(local, sample_rate=sample_rate)
    anchors = select_anchor_candidates(local, sample_rate, segments, min_anchor_duration=0.3, max_anchor_duration=0.8)

    # Provide a non-empty speech segment list that does not overlap the actual
    # matched master span, so the post-NCC VAD check rejects with no overlap.
    master_speech_segments = [SpeechSegment(start=2.0, end=2.1, confidence=1.0)]
    safety = InitialOffsetSafetyConfig(master_vad_filter_enabled=True)
    from double_ender_sync.config import AnchorMatchingConfig
    permissive_matching = AnchorMatchingConfig(
        ncc_min_score=0.0, ncc_min_margin=0.0, ncc_min_prominence=0.0, min_confidence_for_fit=0.0
    )
    matches = match_anchors_for_drift(
        local, master, sample_rate, anchors,
        initial_offset_seconds=0.5,
        search_radius_seconds=6.0,
        matching_config=permissive_matching,
        master_speech_segments=master_speech_segments,
        safety_config=safety,
    )
    rejected = [match for match in matches if match.rejected_reason == "master_vad_no_overlap"]
    assert len(rejected) >= 1


def test_anchor_overlapping_master_speech_preserved() -> None:
    sample_rate = 8000
    tone = _make_tone(sample_rate, 0.8, 330)
    local = tone
    master = np.concatenate([np.zeros(int(sample_rate * 0.5), dtype=np.float32), tone])

    segments = detect_speech_segments(local, sample_rate=sample_rate)
    anchors = select_anchor_candidates(local, sample_rate, segments, min_anchor_duration=0.3, max_anchor_duration=0.8)

    master_speech_segments = [SpeechSegment(start=0.4, end=1.5, confidence=1.0)]
    safety = InitialOffsetSafetyConfig(
        master_vad_filter_enabled=True,
        master_vad_min_overlap_ratio=0.25,
    )
    from double_ender_sync.config import AnchorMatchingConfig
    permissive_matching = AnchorMatchingConfig(
        ncc_min_score=0.0, ncc_min_margin=0.0, ncc_min_prominence=0.0, min_confidence_for_fit=0.0
    )
    matches = match_anchors_for_drift(
        local, master, sample_rate, anchors,
        initial_offset_seconds=0.5,
        search_radius_seconds=6.0,
        matching_config=permissive_matching,
        master_speech_segments=master_speech_segments,
        safety_config=safety,
    )
    accepted = [match for match in matches if match.rejected_reason is None]
    assert len(accepted) >= 1


def test_vad_boundary_padding_prevents_false_rejection() -> None:
    sample_rate = 8000
    tone = _make_tone(sample_rate, 0.8, 330)
    local = tone
    # Master speech starts slightly after the expected master position.
    master = np.concatenate([np.zeros(int(sample_rate * 0.55), dtype=np.float32), tone])

    segments = detect_speech_segments(local, sample_rate=sample_rate)
    anchors = select_anchor_candidates(local, sample_rate, segments, min_anchor_duration=0.3, max_anchor_duration=0.8)

    # Without padding the search interval would barely miss the speech.
    master_speech_segments = [SpeechSegment(start=0.6, end=1.5, confidence=1.0)]
    safety = InitialOffsetSafetyConfig(
        master_vad_filter_enabled=True,
        master_vad_min_overlap_ratio=0.25,
        master_vad_padding_seconds=0.25,
    )
    from double_ender_sync.config import AnchorMatchingConfig
    permissive_matching = AnchorMatchingConfig(
        ncc_min_score=0.0, ncc_min_margin=0.0, ncc_min_prominence=0.0, min_confidence_for_fit=0.0
    )
    matches = match_anchors_for_drift(
        local, master, sample_rate, anchors,
        initial_offset_seconds=0.55,
        search_radius_seconds=6.0,
        matching_config=permissive_matching,
        master_speech_segments=master_speech_segments,
        safety_config=safety,
    )
    accepted = [match for match in matches if match.rejected_reason is None]
    assert len(accepted) >= 1


def test_master_vad_overlap_ratio_computation() -> None:
    from double_ender_sync.analysis.drift import _master_vad_overlap_ratio

    speech_segments = [SpeechSegment(start=1.0, end=2.0, confidence=1.0)]
    padding = 0.25
    # Search fully contains the padded speech segment.
    assert _master_vad_overlap_ratio(0.0, 3.0, speech_segments, padding_seconds=padding) == pytest.approx(1.5 / 3.0)
    # Partial overlap with the padded segment.
    assert _master_vad_overlap_ratio(1.5, 2.5, speech_segments, padding_seconds=padding) == pytest.approx(0.75 / 1.0)
    # No overlap even with padding.
    assert _master_vad_overlap_ratio(2.5, 3.5, speech_segments, padding_seconds=padding) == pytest.approx(0.0)


def test_master_vad_overlap_ratio_avoids_double_counting_overlapping_segments() -> None:
    from double_ender_sync.analysis.drift import _master_vad_overlap_ratio

    # Two speech segments that overlap after padding. The union of padded
    # intervals is [0.75, 2.25], length 1.5, not 2.0.
    speech_segments = [
        SpeechSegment(start=1.0, end=1.6, confidence=1.0),
        SpeechSegment(start=1.4, end=2.0, confidence=1.0),
    ]
    padding = 0.25
    assert _master_vad_overlap_ratio(0.0, 3.0, speech_segments, padding_seconds=padding) == pytest.approx(1.5 / 3.0)


def test_empty_master_vad_treated_as_uncertain() -> None:
    sample_rate = 8000
    tone = _make_tone(sample_rate, 0.8, 330)
    local = tone
    master = np.concatenate([np.zeros(int(sample_rate * 0.5), dtype=np.float32), tone])

    segments = detect_speech_segments(local, sample_rate=sample_rate)
    anchors = select_anchor_candidates(local, sample_rate, segments, min_anchor_duration=0.3, max_anchor_duration=0.8)
    estimate = estimate_initial_offset_with_safety_net(local, master, sample_rate, anchors)
    assert estimate is not None

    # Default uncertain policy is "warn": matching continues without VAD filtering.
    from double_ender_sync.config import AnchorMatchingConfig
    permissive_matching = AnchorMatchingConfig(
        ncc_min_score=0.0, ncc_min_margin=0.0, ncc_min_prominence=0.0, min_confidence_for_fit=0.0
    )
    warn_safety = InitialOffsetSafetyConfig(master_vad_filter_enabled=True)
    warn_matches = match_anchors_for_drift(
        local, master, sample_rate, anchors,
        initial_offset_seconds=estimate.offset_seconds,
        search_radius_seconds=6.0,
        matching_config=permissive_matching,
        master_speech_segments=[],
        safety_config=warn_safety,
    )
    assert any(match.rejected_reason is None for match in warn_matches)
    assert not any(
        match.rejected_reason in {"master_vad_no_overlap", "master_vad_low_overlap_ratio"}
        for match in warn_matches
    )

    # Reject policy still rejects every anchor as unavailable.
    reject_safety = InitialOffsetSafetyConfig(
        master_vad_filter_enabled=True,
        master_vad_uncertain_policy="reject",
    )
    reject_matches = match_anchors_for_drift(
        local, master, sample_rate, anchors,
        initial_offset_seconds=estimate.offset_seconds,
        search_radius_seconds=6.0,
        master_speech_segments=[],
        safety_config=reject_safety,
    )
    assert all(match.rejected_reason == "master_vad_unavailable" for match in reject_matches)


def test_none_master_vad_applies_uncertain_policy() -> None:
    sample_rate = 8000
    tone = _make_tone(sample_rate, 0.8, 330)
    local = tone
    master = np.concatenate([np.zeros(int(sample_rate * 0.5), dtype=np.float32), tone])

    segments = detect_speech_segments(local, sample_rate=sample_rate)
    anchors = select_anchor_candidates(local, sample_rate, segments, min_anchor_duration=0.3, max_anchor_duration=0.8)
    estimate = estimate_initial_offset_with_safety_net(local, master, sample_rate, anchors)
    assert estimate is not None

    # reject policy should mark every anchor as master_vad_unavailable.
    reject_safety = InitialOffsetSafetyConfig(
        master_vad_filter_enabled=True,
        master_vad_uncertain_policy="reject",
    )
    reject_matches = match_anchors_for_drift(
        local, master, sample_rate, anchors,
        initial_offset_seconds=estimate.offset_seconds,
        search_radius_seconds=6.0,
        master_speech_segments=None,
        safety_config=reject_safety,
    )
    assert all(match.rejected_reason == "master_vad_unavailable" for match in reject_matches)

    # warn/skip policies should fall through to normal matching.
    from double_ender_sync.config import AnchorMatchingConfig
    permissive_matching = AnchorMatchingConfig(
        ncc_min_score=0.0, ncc_min_margin=0.0, ncc_min_prominence=0.0, min_confidence_for_fit=0.0
    )
    for policy in ("warn", "skip"):
        permissive_safety = InitialOffsetSafetyConfig(
            master_vad_filter_enabled=True,
            master_vad_uncertain_policy=policy,  # type: ignore[arg-type]
        )
        permissive_matches = match_anchors_for_drift(
            local, master, sample_rate, anchors,
            initial_offset_seconds=estimate.offset_seconds,
            search_radius_seconds=6.0,
            matching_config=permissive_matching,
            master_speech_segments=None,
            safety_config=permissive_safety,
        )
        assert any(match.rejected_reason is None for match in permissive_matches)


def test_initial_offset_safety_config_validation() -> None:
    with pytest.raises(ValueError, match="confidence thresholds"):
        InitialOffsetSafetyConfig(low_confidence_threshold=0.5, medium_confidence_threshold=0.3)

    with pytest.raises(ValueError, match="initial_offset_min_confidence"):
        InitialOffsetSafetyConfig(initial_offset_min_confidence=0.1)

    with pytest.raises(ValueError, match="master_vad_uncertain_policy"):
        InitialOffsetSafetyConfig(master_vad_uncertain_policy="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_drift_search_radius_seconds"):
        InitialOffsetSafetyConfig(max_drift_search_radius_seconds=-5.0)


def test_report_compatibility_existing_linear_fields() -> None:
    from double_ender_sync.report.report import build_alignment_diagnostics_report
    from double_ender_sync.types import AudioTrack

    track = AudioTrack(
        path=__file__,
        name="test",
        sample_rate=16000,
        duration_seconds=2.0,
        channels=1,
        original_samples=None,
        analysis_samples=np.zeros(32000, dtype=np.float32),
        analysis_sample_rate=16000,
    )
    master = AudioTrack(
        path=__file__,
        name="master",
        sample_rate=16000,
        duration_seconds=2.0,
        channels=1,
        original_samples=None,
        analysis_samples=np.zeros(32000, dtype=np.float32),
        analysis_sample_rate=16000,
    )
    detail = {
        "drift_estimate": {
            "model_type": "linear",
            "offset_seconds": 1.0,
            "stretch_ratio": 1.0,
            "anchor_count": 5,
            "residual_median_ms": 10.0,
            "residual_max_ms": 20.0,
            "warnings": [],
        }
    }
    report = build_alignment_diagnostics_report(
        master=master,
        tracks=[track],
        analysis_sample_rate=16000,
        track_details={"test": detail},
    )
    track_report = report["tracks"][0]
    assert track_report["offset_seconds"] == 1.0
    assert track_report["stretch_ratio"] == 1.0
    assert track_report["anchor_count"] == 5
    assert track_report["residual_median_ms"] == 10.0
    assert track_report["residual_max_ms"] == 20.0


def test_report_serialization_includes_safety_fields() -> None:
    sample_rate = 16000
    speech = _make_tone(sample_rate, 1.2, 310)
    local = speech
    master = np.concatenate([np.zeros(int(sample_rate * 0.4), dtype=np.float32), speech])

    segments = detect_speech_segments(local, sample_rate=sample_rate)
    anchors = select_anchor_candidates(local, sample_rate, segments, min_anchor_duration=0.5, max_anchor_duration=1.0)
    estimate = estimate_initial_offset_with_safety_net(local, master, sample_rate, anchors)
    serialized = serialize_offset(estimate)

    assert serialized is not None
    assert serialized["estimation_method"] == "anchor_ncc"
    assert serialized["confidence_band"] in {"high", "medium"}
    if serialized["confidence_band"] == "high":
        assert serialized["selected_drift_search_radius_seconds"] == pytest.approx(6.0)
    else:
        assert serialized["selected_drift_search_radius_seconds"] == pytest.approx(12.0)
    assert "anchor_ncc" in serialized


def test_failed_initial_offset_preserves_diagnostics() -> None:
    sample_rate = 8000
    tone = _make_tone(sample_rate, 1.0, 440)
    local = tone
    silence = np.zeros(int(sample_rate * 1.5), dtype=np.float32)
    master = np.concatenate([silence, tone])

    safety = InitialOffsetSafetyConfig(coarse_fallback_enabled=False)
    estimate = estimate_initial_offset_with_safety_net(
        local, master, sample_rate, [], safety_config=safety
    )

    assert estimate is not None
    assert estimate.confidence_band == "failed"
    assert estimate.radius_reason == "no_usable_initial_estimate"
    assert estimate.selected_drift_search_radius_seconds == pytest.approx(0.0)
    assert "initial_offset_low_confidence" in estimate.warnings


def test_rejected_fallback_does_not_drive_high_confidence_band() -> None:
    """A rejected ambiguous fallback must not report a high-confidence band.

    When the anchor path finds nothing and the coarse fallback sees two equally
    good matches, the fallback estimate is rejected for low margin. The reported
    confidence band must be failed, even though the raw fallback diagnostic
    carries a near-perfect peak score.
    """
    sample_rate = 8000
    tone = _make_tone(sample_rate, 1.0, 440)
    silence = np.zeros(int(sample_rate * 1.0), dtype=np.float32)

    local = tone
    # Master contains two identical copies of the tone, so the fallback sees
    # two equal peaks and a near-zero margin.
    master = np.concatenate([silence, tone, silence, tone])

    safety = InitialOffsetSafetyConfig(
        coarse_fallback_enabled=True,
        coarse_fallback_min_peak_margin=0.5,  # force rejection
        coarse_fallback_min_confidence=0.0,
    )
    estimate = estimate_initial_offset_with_safety_net(
        local, master, sample_rate, [], safety_config=safety
    )

    assert estimate is not None
    assert estimate.fallback_attempted is True
    assert estimate.fallback_selected is False
    assert estimate.confidence_band == "failed"
    assert estimate.confidence == pytest.approx(0.0)
    assert estimate.radius_reason == "no_usable_initial_estimate"
    assert "initial_offset_low_confidence" in estimate.warnings
    assert estimate.coarse_fft_fallback is not None
    assert estimate.coarse_fft_fallback["peak_margin"] < safety.coarse_fallback_min_peak_margin


def test_default_fallback_is_not_duration_capped() -> None:
    sample_rate = 8000
    tone = _make_tone(sample_rate, 1.0, 440)
    local = tone
    silence = np.zeros(int(sample_rate * 1.5), dtype=np.float32)
    master = np.concatenate([silence, tone])

    estimate = estimate_initial_offset_with_safety_net(local, master, sample_rate, [])
    assert estimate is not None
    assert estimate.coarse_fft_fallback is not None
    assert estimate.coarse_fft_fallback["local_duration_seconds"] == pytest.approx(1.0)
    assert estimate.coarse_fft_fallback["master_duration_seconds"] == pytest.approx(2.5)
    assert "coarse_fallback_duration_capped" not in estimate.warnings
    assert "coarse_fallback_memory_limited" not in estimate.warnings


def test_explicit_fallback_duration_cap_emits_warning() -> None:
    sample_rate = 8000
    tone = _make_tone(sample_rate, 1.0, 440)
    local = tone
    silence = np.zeros(int(sample_rate * 1.5), dtype=np.float32)
    master = np.concatenate([silence, tone])

    # Cap below the full master duration (2.5s) but large enough to include
    # part of the tone so the fallback can still produce a match.
    safety = InitialOffsetSafetyConfig(coarse_fallback_max_duration_seconds=2.0)
    estimate = estimate_initial_offset_with_safety_net(local, master, sample_rate, [], safety_config=safety)
    assert estimate is not None
    assert estimate.coarse_fft_fallback is not None
    assert estimate.coarse_fft_fallback["master_duration_seconds"] == pytest.approx(2.0)
    assert "coarse_fallback_duration_capped" in estimate.warnings


def test_fallback_memory_guard_caps_long_recordings() -> None:
    """A tiny memory budget caps even a short recording."""
    from double_ender_sync.alignment.offset import _coarse_fft_fallback_offset

    sample_rate = 8000
    duration = 10.0
    tone = _make_tone(sample_rate, duration, 440)

    safety = InitialOffsetSafetyConfig(
        coarse_fallback_sample_rate=8000,
        coarse_fallback_max_memory_mb=1.0,  # very small budget to force a cap
    )
    estimate = _coarse_fft_fallback_offset(tone, tone, sample_rate, safety)
    assert estimate is not None
    fallback = estimate.coarse_fft_fallback
    assert fallback is not None
    assert fallback["memory_limited"] is True
    assert fallback["effective_max_duration_seconds"] is not None
    assert fallback["effective_max_duration_seconds"] < duration
    assert fallback["local_duration_seconds"] == pytest.approx(
        fallback["effective_max_duration_seconds"], rel=1e-2
    )
    assert fallback["master_duration_seconds"] == pytest.approx(
        fallback["effective_max_duration_seconds"], rel=1e-2
    )


def test_short_fallback_not_memory_limited() -> None:
    """Short recordings fit comfortably within the default memory budget."""
    from double_ender_sync.alignment.offset import _coarse_fft_fallback_offset

    sample_rate = 8000
    tone = _make_tone(sample_rate, 2.0, 440)

    safety = InitialOffsetSafetyConfig(
        coarse_fallback_sample_rate=8000,
        coarse_fallback_max_memory_mb=1024.0,
    )
    estimate = _coarse_fft_fallback_offset(tone, tone, sample_rate, safety)
    assert estimate is not None
    fallback = estimate.coarse_fft_fallback
    assert fallback is not None
    assert fallback["memory_limited"] is False
    assert fallback["effective_max_duration_seconds"] is None
    assert fallback["local_duration_seconds"] == pytest.approx(2.0)


def test_coarse_fallback_diagnostics_contain_required_fields() -> None:
    sample_rate = 8000
    tone = _make_tone(sample_rate, 1.0, 440)
    local = tone
    silence = np.zeros(int(sample_rate * 1.5), dtype=np.float32)
    master = np.concatenate([silence, tone])

    safety = InitialOffsetSafetyConfig(
        initial_offset_min_confidence=0.50,
        coarse_fallback_min_peak_margin=0.05,
        coarse_fallback_min_confidence=0.50,
    )
    estimate = estimate_initial_offset_with_safety_net(local, master, sample_rate, [], safety_config=safety)
    assert estimate is not None
    assert estimate.fallback_attempted is True
    assert estimate.fallback_selected is True
    assert estimate.coarse_fft_fallback is not None
    fallback = estimate.coarse_fft_fallback
    assert "offset_seconds" in fallback
    assert "peak_score" in fallback
    assert "peak_margin" in fallback
    assert "second_score" in fallback
    assert "effective_sample_rate" in fallback
    assert "searched_lag_min_seconds" in fallback
    assert "searched_lag_max_seconds" in fallback
