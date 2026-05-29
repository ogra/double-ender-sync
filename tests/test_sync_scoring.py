import numpy as np
import pytest

from double_ender_sync.analysis.features import (
    NccPeakDiagnostics,
    clamp01,
    extract_anchor_feature,
    gcc_phat_scores,
    ncc_peak_diagnostics,
    normalized_correlation_scores,
)
from double_ender_sync.config import AnchorMatchingConfig
from double_ender_sync.analysis.drift import (
    AnchorMatch,
    match_anchors_for_drift,
    fit_linear_drift_model,
)
from double_ender_sync.analysis.anchors import AnchorCandidate


class TestNccPeakDiagnostics:
    def test_single_clear_peak_high_confidence(self) -> None:
        scores = np.array([0.1, 0.2, 0.15, 0.9, 0.16, 0.12, 0.11], dtype=np.float32)
        result = ncc_peak_diagnostics(scores, sample_rate=16000, nms_exclusion_seconds=0.001)
        assert result is not None
        assert result.best_score == pytest.approx(0.9)
        assert result.best_lag_samples == 3
        assert result.prominence > 0.5

    def test_no_peaks_falls_back_to_argmax(self) -> None:
        scores = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = ncc_peak_diagnostics(scores, sample_rate=16000)
        assert result is not None
        assert result.best_score == pytest.approx(0.0)
        assert result.prominence == 0.0

    def test_multiple_equal_peaks_selects_earliest(self) -> None:
        scores = np.array([0.1, 0.2, 0.15, 1.0, 0.16, 0.12, 1.0, 0.11], dtype=np.float32)
        result = ncc_peak_diagnostics(scores, sample_rate=16000, nms_exclusion_seconds=0.001)
        assert result is not None
        assert result.best_score == pytest.approx(1.0)
        assert result.best_lag_samples == 3

    def test_empty_scores_returns_none(self) -> None:
        result = ncc_peak_diagnostics(np.array([], dtype=np.float32), sample_rate=16000)
        assert result is None

    def test_second_peak_outside_exclusion(self) -> None:
        scores = np.zeros(200, dtype=np.float32)
        scores[50] = 1.0
        scores[100] = 0.8
        result = ncc_peak_diagnostics(scores, sample_rate=16000, nms_exclusion_seconds=0.001)
        assert result is not None
        assert result.second_score is not None
        assert result.second_score == pytest.approx(0.8)
        assert result.margin == pytest.approx(0.2)

    def test_no_independent_second_peak(self) -> None:
        scores = np.zeros(1000, dtype=np.float32)
        scores[500] = 1.0
        result = ncc_peak_diagnostics(scores, sample_rate=16000, nms_exclusion_seconds=0.05)
        assert result is not None
        assert result.second_score is None
        assert result.margin is None


class TestGccPhat:
    def test_gcc_phat_lag_matches_ncc_for_broadband_signal(self) -> None:
        rng = np.random.RandomState(42)
        search = rng.randn(8000).astype(np.float32) * 0.1
        feature = rng.randn(400).astype(np.float32)
        insert_position = 3000
        search[insert_position : insert_position + len(feature)] = feature

        ncc = normalized_correlation_scores(search, feature)
        ncc_best = int(np.argmax(ncc))

        gcc = gcc_phat_scores(search, feature)
        gcc_best = int(np.argmax(gcc))

        gcc_peak_value = float(gcc[gcc_best])
        assert gcc_peak_value > 0.05
        assert abs(ncc_best - gcc_best) <= 3

    def test_gcc_phat_empty_feature(self) -> None:
        search = np.ones(100, dtype=np.float32)
        feature = np.zeros(0, dtype=np.float32)
        result = gcc_phat_scores(search, feature)
        assert result.size == 0

    def test_gcc_phat_feature_larger_than_search(self) -> None:
        search = np.ones(100, dtype=np.float32)
        feature = np.ones(200, dtype=np.float32)
        result = gcc_phat_scores(search, feature)
        assert result.size == 0


class TestAnchorMatchingConfig:
    def test_default_config_is_valid(self) -> None:
        config = AnchorMatchingConfig()
        assert config.ncc_min_score == 0.45
        assert config.ncc_min_margin == 0.10
        assert config.gcc_phat_enabled is True

    def test_invalid_nms_exclusion_raises(self) -> None:
        with pytest.raises(ValueError):
            AnchorMatchingConfig(nms_exclusion_seconds=0.0)
        with pytest.raises(ValueError):
            AnchorMatchingConfig(nms_exclusion_seconds=0.6)

    def test_invalid_ncc_min_score_raises(self) -> None:
        with pytest.raises(ValueError):
            AnchorMatchingConfig(ncc_min_score=-1.5)
        with pytest.raises(ValueError):
            AnchorMatchingConfig(ncc_min_score=1.5)

    def test_invalid_gcc_phat_tolerance_raises(self) -> None:
        with pytest.raises(ValueError):
            AnchorMatchingConfig(gcc_phat_agreement_tolerance_seconds=0.0)
        with pytest.raises(ValueError):
            AnchorMatchingConfig(gcc_phat_agreement_tolerance_seconds=1.5)

    def test_invalid_width_ordering_raises(self) -> None:
        with pytest.raises(ValueError):
            AnchorMatchingConfig(ncc_good_width_seconds=0.1, ncc_bad_width_seconds=0.05)

    def test_as_dict_returns_all_fields(self) -> None:
        config = AnchorMatchingConfig()
        d = config.as_dict()
        assert "nms_exclusion_seconds" in d
        assert "ncc_min_score" in d
        assert "gcc_phat_enabled" in d
        assert "min_confidence_for_fit" in d


class TestMatchAnchorsForDriftImproved:
    def test_rejects_low_score_with_gate(self) -> None:
        rng = np.random.RandomState(42)
        local = rng.randn(16000).astype(np.float32) * 0.1
        master = rng.randn(32000).astype(np.float32) * 0.1
        sr = 16000

        anchor = AnchorCandidate(local_start=0.0, local_end=1.0, confidence=1.0, rms=0.1)
        config = AnchorMatchingConfig(ncc_min_score=0.8)

        matches = match_anchors_for_drift(
            local, master, sr, [anchor], 0.0, matching_config=config
        )

        assert len(matches) == 1
        assert matches[0].rejected_reason == "ncc_best_score_below_min"

    def test_clear_single_match_high_confidence(self) -> None:
        rng = np.random.RandomState(42)
        sr = 16000
        local = rng.randn(int(sr * 3)).astype(np.float32) * 0.01
        master = rng.randn(int(sr * 8)).astype(np.float32) * 0.01
        feature = rng.randn(int(sr * 1.0)).astype(np.float32)
        local[int(sr * 1.0):int(sr * 2.0)] = feature
        master[int(sr * 3.0):int(sr * 4.0)] = feature

        anchor = AnchorCandidate(local_start=1.0, local_end=2.0, confidence=1.0, rms=0.1)
        config = AnchorMatchingConfig(
            ncc_min_score=0.3,
            ncc_min_margin=0.02,
            ncc_min_prominence=0.02,
            ncc_margin_low=0.01,
            gcc_phat_enabled=False,
        )

        matches = match_anchors_for_drift(
            local, master, sr, [anchor], 1.0,
            search_radius_seconds=3.0, matching_config=config,
        )

        assert len(matches) == 1
        assert matches[0].rejected_reason is None
        assert matches[0].confidence > 0.3
        assert matches[0].match_quality is not None
        assert matches[0].match_uniqueness is not None
        assert matches[0].match_sharpness is not None

    def test_repeated_phrase_low_uniqueness(self) -> None:
        rng = np.random.RandomState(42)
        sr = 16000
        feature = rng.randn(int(sr * 0.5)).astype(np.float32)
        master = rng.randn(int(sr * 10)).astype(np.float32) * 0.01
        master[int(sr * 2.0):int(sr * 2.5)] = feature + rng.randn(int(sr * 0.5)).astype(np.float32) * 0.001
        master[int(sr * 3.0):int(sr * 3.5)] = feature + rng.randn(int(sr * 0.5)).astype(np.float32) * 0.001

        local = rng.randn(int(sr * 2)).astype(np.float32) * 0.01
        local[int(sr * 1.0):int(sr * 1.5)] = feature

        anchor = AnchorCandidate(local_start=1.0, local_end=1.5, confidence=1.0, rms=0.1)
        config = AnchorMatchingConfig(
            ncc_min_score=0.3,
            ncc_min_margin=0.15,
            ncc_min_prominence=0.03,
            ncc_margin_low=0.05,
            ncc_margin_high=0.30,
            gcc_phat_enabled=False,
        )

        matches = match_anchors_for_drift(
            local, master, sr, [anchor], 1.0,
            search_radius_seconds=3.0, matching_config=config,
        )

        assert len(matches) == 1
        assert matches[0].ncc_margin is not None
        assert matches[0].ncc_margin < 0.20, f"Expected margin < 0.20, got {matches[0].ncc_margin}"
        assert matches[0].match_uniqueness is not None
        assert matches[0].match_uniqueness < 0.5

    def test_wide_peak_low_sharpness(self) -> None:
        sr = 16000
        feature_len = int(sr * 0.6)
        half = feature_len // 2
        feature = np.ones(feature_len, dtype=np.float32)
        feature[half:] = -1.0

        master = np.zeros(int(sr * 5), dtype=np.float32)
        insert_pos = int(sr * 2.0)
        master[insert_pos : insert_pos + len(feature)] = feature

        local = np.zeros(int(sr * 1.5), dtype=np.float32)
        local_insert = int(sr * 0.3)
        local[local_insert : local_insert + len(feature)] = feature

        anchor = AnchorCandidate(
            local_start=float(local_insert / sr),
            local_end=float((local_insert + len(feature)) / sr),
            confidence=1.0, rms=0.1,
        )
        config = AnchorMatchingConfig(
            ncc_min_score=0.0,
            ncc_min_margin=0.0,
            ncc_min_prominence=0.0,
            ncc_good_width_seconds=0.001,
            ncc_bad_width_seconds=0.2,
            gcc_phat_enabled=False,
        )

        matches = match_anchors_for_drift(
            local, master, sr, [anchor], 0.0,
            search_radius_seconds=3.0, matching_config=config,
        )

        assert len(matches) == 1
        assert matches[0].ncc_width_seconds is not None
        assert matches[0].match_sharpness is not None
        assert matches[0].match_sharpness < 0.3

    def test_hard_gate_rejected_matches_excluded_from_regression(self) -> None:
        rng = np.random.RandomState(42)
        sr = 16000
        local = rng.randn(int(sr * 5)).astype(np.float32) * 0.01
        master = rng.randn(int(sr * 8)).astype(np.float32) * 0.01
        feature = rng.randn(int(sr * 0.5)).astype(np.float32)
        local[int(sr * 1.0):int(sr * 1.5)] = feature
        master[int(sr * 3.0):int(sr * 3.5)] = feature

        anchor = AnchorCandidate(local_start=1.0, local_end=1.5, confidence=1.0, rms=0.1)
        config = AnchorMatchingConfig(
            ncc_min_score=0.999,
            ncc_min_margin=0.99,
            ncc_min_prominence=0.99,
            gcc_phat_enabled=False,
        )
        matches = match_anchors_for_drift(
            local, master, sr, [anchor], 1.0,
            search_radius_seconds=3.0, matching_config=config,
        )
        assert len(matches) == 1
        assert matches[0].rejected_reason is not None

        drift = fit_linear_drift_model(matches, matching_config=config)
        assert drift is None

    def test_min_confidence_for_fit_excludes_low_confidence(self) -> None:
        rng = np.random.RandomState(42)
        sr = 16000
        feature0 = rng.randn(int(sr * 0.8)).astype(np.float32)
        feature1 = rng.randn(int(sr * 0.8)).astype(np.float32)
        feature2 = rng.randn(int(sr * 0.8)).astype(np.float32)
        local = np.zeros(int(sr * 4), dtype=np.float32) + 0.01
        master = np.zeros(int(sr * 12), dtype=np.float32) + 0.01
        local[int(sr * 1.0):int(sr * 1.8)] = feature0
        master[int(sr * 3.0):int(sr * 3.8)] = feature0
        local[int(sr * 2.0):int(sr * 2.8)] = feature1
        master[int(sr * 5.0):int(sr * 5.8)] = feature1
        feat_wrong = rng.randn(int(sr * 0.8)).astype(np.float32)
        local[int(sr * 3.0):int(sr * 3.8)] = feature2
        master[int(sr * 7.0):int(sr * 7.8)] = feat_wrong  # mismatch to produce a low-confidence match

        anchors = [
            AnchorCandidate(local_start=1.0, local_end=1.8, confidence=1.0, rms=0.1),
            AnchorCandidate(local_start=2.0, local_end=2.8, confidence=1.0, rms=0.1),
            AnchorCandidate(local_start=3.0, local_end=3.8, confidence=1.0, rms=0.1),
        ]
        config = AnchorMatchingConfig(
            ncc_min_score=0.2,
            ncc_min_margin=0.005,
            ncc_min_prominence=0.005,
            ncc_margin_low=0.001,
            ncc_prominence_low=0.001,
            ncc_prominence_high=0.5,
            min_confidence_for_fit=0.30,
            gcc_phat_enabled=False,
        )

        matches = match_anchors_for_drift(
            local, master, sr, anchors, 1.0,
            search_radius_seconds=3.0, matching_config=config,
        )
        assert len(matches) == 3
        drift = fit_linear_drift_model(matches, matching_config=config)
        assert drift is not None
        assert drift.anchor_count >= 2
        low_conf_matches = [m for m in matches if m.confidence < 0.30]
        high_conf_matches = [m for m in matches if m.confidence >= 0.30]
        assert len(low_conf_matches) >= 1, "Expected at least one match below min_confidence_for_fit"
        assert len(high_conf_matches) >= 2
        for m in low_conf_matches:
            assert not m.included_in_regression
            assert m.rejected_reason is not None


class TestClamp01:
    def test_mid_range(self) -> None:
        assert clamp01(0.5) == 0.5

    def test_below_zero(self) -> None:
        assert clamp01(-0.2) == 0.0

    def test_above_one(self) -> None:
        assert clamp01(1.5) == 1.0

    def test_zero_and_one(self) -> None:
        assert clamp01(0.0) == 0.0
        assert clamp01(1.0) == 1.0


class TestExtractAnchorFeature:
    def test_normalization(self) -> None:
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = extract_anchor_feature(samples)
        assert abs(float(np.linalg.norm(result)) - 1.0) < 1e-6
        assert abs(float(np.mean(result))) < 1e-6

    def test_near_silence_returns_centered(self) -> None:
        samples = np.zeros(10, dtype=np.float32)
        result = extract_anchor_feature(samples)
        assert result.shape == samples.shape


class TestNormalizedCorrelationScores:
    def test_perfect_match_gives_one(self) -> None:
        rng = np.random.RandomState(42)
        feature = rng.randn(50).astype(np.float32)
        signal = rng.randn(1000).astype(np.float32)
        feature_centered = feature - np.mean(feature)
        feature_norm = np.linalg.norm(feature_centered)
        feature_unit = (feature_centered / feature_norm).astype(np.float32)
        signal[200:250] = feature

        scores = normalized_correlation_scores(signal, feature_unit)
        max_score = float(np.max(scores))
        assert max_score == pytest.approx(1.0, abs=0.01)
        best_idx = int(np.argmax(scores))
        assert best_idx == 200

    def test_no_match_gives_low_scores(self) -> None:
        rng = np.random.RandomState(42)
        signal = rng.randn(10000).astype(np.float32)
        feature = rng.randn(400).astype(np.float32)
        scores = normalized_correlation_scores(signal, feature)
        assert float(np.max(scores)) < 0.25

    def test_empty_feature_returns_empty(self) -> None:
        signal = np.ones(100, dtype=np.float32)
        feature = np.zeros(0, dtype=np.float32)
        result = normalized_correlation_scores(signal, feature)
        assert result.size == 0

    def test_feature_larger_than_signal_returns_empty(self) -> None:
        signal = np.ones(10, dtype=np.float32)
        feature = np.ones(20, dtype=np.float32)
        result = normalized_correlation_scores(signal, feature)
        assert result.size == 0
