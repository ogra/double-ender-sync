import numpy as np

from double_ender_sync.alignment.local_adjust import apply_local_adjustment


def test_local_adjustment_skips_when_disabled() -> None:
    samples = np.zeros(8000, dtype=np.float32)
    result = apply_local_adjustment(samples, 8000, residual_events=[], enabled=False)
    assert result.events == []
    assert "disabled" in result.warnings[0]


def test_local_adjustment_applies_shift_on_safe_silence() -> None:
    sr = 1000
    samples = np.zeros(4000, dtype=np.float32)
    samples[1000:1500] = 0.7
    samples[2500:3000] = 0.7

    residual_events = [{"local_start": 2.0, "residual_ms": 120.0}]
    result = apply_local_adjustment(
        globally_aligned_samples=samples,
        sample_rate=sr,
        residual_events=residual_events,
        enabled=True,
        residual_threshold_ms=80.0,
        max_silence_search_seconds=0.6,
    )

    assert len(result.events) == 1
    assert result.events[0].split_time_seconds == 2.0
    assert result.events[0].shift_seconds == -0.12
    assert np.allclose(result.adjusted_samples[0:1000], samples[0:1000])


def test_local_adjustment_ignores_rejected_drift_anchor() -> None:
    sr = 1000
    samples = np.zeros(4000, dtype=np.float32)
    samples[1000:1500] = 0.7
    samples[2500:3000] = 0.7

    residual_events = [
        {
            "local_start": 2.0,
            "residual_ms": 120.0,
            "included_in_regression": False,
            "rejected_reason": "residual_outlier",
        }
    ]
    result = apply_local_adjustment(
        globally_aligned_samples=samples,
        sample_rate=sr,
        residual_events=residual_events,
        enabled=True,
        residual_threshold_ms=80.0,
        max_silence_search_seconds=0.6,
    )

    assert result.events == []
    assert np.array_equal(result.adjusted_samples, samples)
    assert "skipped 1 rejected drift anchor" in result.warnings[0]
