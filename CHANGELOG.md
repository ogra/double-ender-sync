# Changelog

All notable changes to this project will be documented in this file.

## [0.2.2] - 2026-05-09

### Added

- Added consistent version display across CLI (`--version` / `-V`), Python API, and GUI footer.
- Added selectable non-linear drift models across CLI, API, and GUI.
- Included a calibration example script in source distribution so release artifacts contain the example tooling.

### Changed

- Updated `sync-report.json` data structure.

## [0.2.1] - 2026-05-08

### Added

- Added richer drift anchor selection controls, including dynamic anchor budgeting, stratified anchor selection, and adaptive anchor durations.
- Added user-facing runtime tuning documentation for alignment options and debug diagnostics.
- Added Japanese locale translations for user-facing messages.

### Changed

- Improved drift fitting diagnostics so reports expose clearer residual and anchor-selection details for manual review.
- Improved validation for anchor duration configuration values, including rejecting non-finite values with clearer errors.
- Clarified CLI help and documentation for debug verbosity, option key names, and the `pitch_preserving` stretch extra dependency.

### Fixed

- Fixed stratified anchor selection budget ordering.
- Fixed noise-likeness scoring for silent or constant centered signals by treating them as the worst noise-like case.

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
