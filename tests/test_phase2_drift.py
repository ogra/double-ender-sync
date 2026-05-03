import numpy as np

from double_ender_sync.alignment.offset import estimate_initial_offset
from double_ender_sync.analysis.anchors import select_anchor_candidates
from double_ender_sync.analysis.drift import fit_linear_drift_model, match_anchors_for_drift
from double_ender_sync.analysis.vad import detect_speech_segments


def test_fit_linear_drift_model_estimates_stretch_ratio() -> None:
    sample_rate = 8000
    segment_duration = 0.8
    silence_gap = 0.6
    segments = []
    for f in [220, 330, 280, 410, 360, 260]:
        t = np.arange(int(sample_rate * segment_duration)) / sample_rate
        tone = (0.22 * np.sin(2 * np.pi * f * t)).astype(np.float32)
        segments.append(tone)
        segments.append(np.zeros(int(sample_rate * silence_gap), dtype=np.float32))
    local = np.concatenate(segments)

    true_stretch = 1.0008
    true_offset = 0.35
    master_len = int((true_stretch * (len(local) / sample_rate) + true_offset + 1.0) * sample_rate)
    master = np.zeros(master_len, dtype=np.float32)

    for i, sample in enumerate(local):
        target = int(round((true_stretch * (i / sample_rate) + true_offset) * sample_rate))
        if 0 <= target < master_len:
            master[target] += sample

    speech_segments = detect_speech_segments(local, sample_rate=sample_rate, frame_ms=40.0)
    anchors = select_anchor_candidates(local, sample_rate, speech_segments, min_anchor_duration=0.5, max_anchor_duration=0.9)
    initial = estimate_initial_offset(local, master, sample_rate, anchors)
    assert initial is not None

    matches = match_anchors_for_drift(local, master, sample_rate, anchors, initial.offset_seconds, search_radius_seconds=1.0)
    drift = fit_linear_drift_model(matches)

    assert drift is not None
    assert drift.anchor_count >= 2
    assert abs(drift.stretch_ratio - true_stretch) < 0.005
    assert abs(drift.offset_seconds - true_offset) < 0.1
