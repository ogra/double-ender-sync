import numpy as np

from double_ender_sync.alignment.offset import estimate_initial_offset
from double_ender_sync.analysis.anchors import select_anchor_candidates
from double_ender_sync.analysis.vad import detect_speech_segments


def test_estimate_initial_offset_with_leading_silence() -> None:
    sample_rate = 16000
    speech_duration = 1.2
    silence_lead = 0.4

    t = np.arange(int(sample_rate * speech_duration)) / sample_rate
    speech = (0.25 * np.sin(2 * np.pi * 310 * t)).astype(np.float32)

    local = speech
    master = np.concatenate([np.zeros(int(sample_rate * silence_lead), dtype=np.float32), speech])

    segments = detect_speech_segments(local, sample_rate=sample_rate)
    anchors = select_anchor_candidates(local, sample_rate, segments, min_anchor_duration=0.5, max_anchor_duration=1.0)
    estimate = estimate_initial_offset(local, master, sample_rate=sample_rate, anchors=anchors)

    assert estimate is not None
    assert abs(estimate.offset_seconds - silence_lead) < 0.03
    assert estimate.confidence > 0.4
