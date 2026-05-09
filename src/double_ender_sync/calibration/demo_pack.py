#!/usr/bin/env python3
"""Generate deterministic synthetic calibration WAVs outside the repository.

This script creates small public-domain-style voiced/silence WAVs with known
local-to-master mappings. Generated WAVs and reports are calibration artifacts;
do not commit them to the repository.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf


def _constant(offset_seconds: float, stretch_ratio: float) -> Callable[[float], float]:
    return lambda local_time: offset_seconds + (stretch_ratio * local_time)


def _piecewise(offset_seconds: float = 1.0) -> Callable[[float], float]:
    def mapping(local_time: float) -> float:
        if local_time <= 12.0:
            return offset_seconds + (1.0000 * local_time)
        boundary_master = offset_seconds + (1.0000 * 12.0)
        return boundary_master + (1.0030 * (local_time - 12.0))

    return mapping


def _smooth(
    offset_seconds: float = 1.0, duration_seconds: float = 24.0
) -> Callable[[float], float]:
    return lambda local_time: float(
        offset_seconds
        + local_time
        + (0.035 * np.sin((2.0 * np.pi * local_time) / duration_seconds))
    )


def _local_program(sample_rate: int, duration_seconds: float) -> np.ndarray:
    total_samples = int(round(sample_rate * duration_seconds))
    samples = np.zeros(total_samples, dtype=np.float32)
    # Keep each voiced region above the CLI default --min-anchor-duration
    # (1.0s) so generated packs exercise anchor selection without extra flags.
    tone_duration = 1.20
    gap_duration = 0.40
    frequencies = (220.0, 330.0, 247.0, 392.0, 294.0, 440.0)
    cursor = 0.0
    tone_index = 0
    while cursor + tone_duration < duration_seconds:
        start = int(round(cursor * sample_rate))
        end = min(total_samples, start + int(round(tone_duration * sample_rate)))
        t = np.arange(end - start, dtype=np.float32) / sample_rate
        envelope = np.sin(np.linspace(0.0, np.pi, end - start, dtype=np.float32))
        frequency = frequencies[tone_index % len(frequencies)]
        chirp = np.sin(2.0 * np.pi * (frequency * t + 35.0 * t * t))
        rng = np.random.default_rng(10_000 + tone_index)
        noise = rng.standard_normal(end - start).astype(np.float32)
        segment = (0.35 * chirp.astype(np.float32)) + (0.65 * noise)
        samples[start:end] += (0.18 * envelope * segment).astype(np.float32)
        cursor += tone_duration + gap_duration
        tone_index += 1
    return samples


def _render_to_master(
    local: np.ndarray,
    mapping: Callable[[float], float],
    sample_rate: int,
    master_duration_seconds: float,
    *,
    dropout: tuple[float, float] | None = None,
) -> np.ndarray:
    master = np.zeros(int(round(master_duration_seconds * sample_rate)), dtype=np.float32)
    for local_index, sample in enumerate(local):
        local_time = local_index / sample_rate
        if dropout is not None and dropout[0] <= local_time <= dropout[1]:
            continue
        master_index = int(round(mapping(local_time) * sample_rate))
        if 0 <= master_index < len(master):
            master[master_index] += sample
    return np.clip(master, -0.95, 0.95)


def _write_case(
    output_dir: Path,
    name: str,
    mapping: Callable[[float], float],
    *,
    sample_rate: int,
    duration_seconds: float,
    dropout: tuple[float, float] | None = None,
) -> dict[str, object]:
    case_dir = output_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    local = _local_program(sample_rate, duration_seconds)
    master_duration = mapping(duration_seconds) + 1.0
    master = _render_to_master(local, mapping, sample_rate, master_duration, dropout=dropout)

    local_path = case_dir / "speaker-a.wav"
    master_path = case_dir / "master.wav"
    sf.write(local_path, local, sample_rate)
    sf.write(master_path, master, sample_rate)
    return {
        "name": name,
        "master": master_path.relative_to(output_dir).as_posix(),
        "track": local_path.relative_to(output_dir).as_posix(),
        "duration_seconds": duration_seconds,
        "sample_rate": sample_rate,
        "dropout_local_seconds": dropout,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, required=True, help="Output directory for generated WAV demo pack."
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=8000,
        help="Small deterministic sample rate for demo WAVs.",
    )
    parser.add_argument(
        "--duration", type=float, default=24.0, help="Local-track duration in seconds per case."
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cases = [
        _write_case(
            args.out,
            "known-offset",
            _constant(2.0, 1.0),
            sample_rate=args.sample_rate,
            duration_seconds=args.duration,
        ),
        _write_case(
            args.out,
            "constant-drift",
            _constant(1.0, 1.0020),
            sample_rate=args.sample_rate,
            duration_seconds=args.duration,
        ),
        _write_case(
            args.out,
            "piecewise-drift",
            _piecewise(),
            sample_rate=args.sample_rate,
            duration_seconds=args.duration,
        ),
        _write_case(
            args.out,
            "smooth-spline-drift",
            _smooth(duration_seconds=args.duration),
            sample_rate=args.sample_rate,
            duration_seconds=args.duration,
        ),
        _write_case(
            args.out,
            "dropout-like-gap",
            _constant(1.0, 1.0),
            sample_rate=args.sample_rate,
            duration_seconds=args.duration,
            dropout=(9.0, 14.0),
        ),
    ]
    manifest = {
        "note": "Generated calibration artifacts only. Do not commit WAVs or private audio.",
        "cases": cases,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote synthetic calibration demo pack to {args.out}")


if __name__ == "__main__":
    main()
