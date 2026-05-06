# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-05-06

### Changed

- Changed the default voice activity detection (VAD) behavior from a fixed RMS threshold to adaptive thresholding via the `adaptive_rms` strategy.

### Added

- Added selectable VAD strategies through the VAD strategy option:
  - `adaptive_rms` (default)
  - `rms` (legacy fixed-threshold behavior)
  - `silero`
  - `webrtc`
  - `pyannote`
- Added support for choosing the pyannote model or pipeline with `--pyannote-model <id>` when using the `pyannote` VAD strategy.
