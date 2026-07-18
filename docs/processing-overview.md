# Processing and Algorithm Overview

This document explains how `double-ender-sync` processes audio and which algorithms are used at each stage.
It is intended for contributors who are reading the codebase for the first time.

## 1. What the pipeline does

At a high level, the CLI aligns each local speaker recording to a master reference recording:

1. Load master and local tracks
2. Convert analysis audio to mono at a fixed analysis sample rate
3. Detect speech regions in each local track
4. Select anchor candidates from detected speech
5. Estimate initial offset against the master
6. Refine alignment with multi-anchor drift estimation
7. Apply global time correction using a linear model
8. Optionally apply local residual adjustment
9. Export synced WAV and diagnostic reports

The orchestration for these steps is in `src/double_ender_sync/cli.py`.

---

## 2. Data flow and module responsibilities

### CLI orchestration (`cli.py`)

`main()` coordinates the full run per track:

- Audio loading
- VAD/speech segmentation
- Anchor selection
- Initial offset estimation
- Drift matching + weighted linear fit
- Global correction + optional local adjustment
- Writing synced audio and report artifacts

It also handles argument validation, progress updates, logging, and top-level error handling.

### Audio I/O and rendering (`audio/`)

- `audio/io.py`: reads WAV data, metadata, and analysis buffers.
- `audio/normalize.py`: normalization helpers used when requested.
- `audio/resample.py`: sample-rate conversion used by analysis/rendering paths.
- `audio/render.py`: writes final synced WAV files.
- `alignment/timeline.py`: renders source samples onto the master timeline with bounded-memory chunks, source/master sample-rate-aware lookup, and explicit silence padding for unsupported time-map regions.

### Analysis (`analysis/`)

- `vad.py`: strategy-based speech segment detection (`silero` / `adaptive_rms` / `rms` / `webrtc`, including optional `pyannote` backend).
- `anchors.py`: converts speech segments into bounded anchor candidates.
- `features.py`: normalized feature extraction and fast normalized cross-correlation scoring.
- `drift.py`: anchor matching around expected positions, outlier rejection, weighted linear drift fit.

### Alignment (`alignment/`)

- `offset.py`: initial alignment estimate from selected anchors.
- `timeline.py`: applies global linear mapping from local timeline to master timeline.
- `local_adjust.py`: optional local split/shift corrections around large residuals.

### Reporting (`report/`)

- `report.py`: serializes stage outputs, writes `sync-report.json`, `sync-markers.csv`, and `warnings.txt`.

---

## 3. Core algorithms

## 3.1 Speech detection (pluggable VAD strategies)

`detect_speech_segments()` in `analysis/vad.py` dispatches through a `VadStrategy`:

- `adaptive_rms` (current robust baseline):
  - Frame mono signal (default 30 ms frame / 10 ms hop)
  - Compute frame RMS energy
  - Estimate noise floor from low percentile
  - Compute adaptive threshold via `noise_floor + k * MAD`
  - Merge contiguous speech frames and filter short segments (default minimum 300 ms)
- `rms`:
  - Fixed RMS threshold path (default threshold `0.02`)
- `silero`:
  - Verifies optional `onnxruntime` installation (`vad-ml` extra)
  - If dependency/runtime requirements are not met, raises a runtime error (no silent fallback)
- `webrtc`:
  - Verifies optional `webrtcvad-wheels` installation (`vad-ml` extra)
  - Resamples to WebRTC-supported rates as needed and applies frame-based speech decisions
- `pyannote`:
  - Verifies optional `pyannote.audio` installation and model/runtime accessibility; the `vad-pyannote` extra targets the current Python 3.14-compatible stack (`pyannote.audio>=4.0.4,<5`, `torch==2.13.0`, `torchaudio==2.11.0`, `torchcodec>=0.11.1,<0.12`).
  - Accepts a configurable model/pipeline id via `--pyannote-model` / `AlignmentOptions.pyannote_model`; the CLI rejects this option unless `--vad-strategy pyannote` is selected
  - Default is `pyannote/speaker-diarization-community-1`; diarization speaker labels are ignored and the union of speech regions becomes the VAD timeline. The legacy `pyannote/voice-activity-detection` pipeline remains available through `--pyannote-model pyannote/voice-activity-detection`.
  - Model identifiers beginning with `pyannote/segmentation` use the compatibility path originally added for pyannote.audio 3.x and still covered under the 4.x stack: `Model.from_pretrained(...)` plus `pyannote.audio.pipelines.VoiceActivityDetection`
  - The modern segmentation VAD path instantiates explicit 100 ms `min_duration_on` / `min_duration_off` smoothing. These are safe anchor-selection defaults, not final editing cuts.
  - Runs VAD from an in-memory Torch waveform dictionary so first-party pyannote calls do not need to decode temporary audio files through torchaudio-backed file I/O
  - Converts timeline speech regions into `SpeechSegment` outputs and records selected VAD metadata in reports
  - See `scripts/compare-vad-strategies.py` for a repeatable private-audio comparison procedure covering adaptive RMS, Silero, default community pyannote, legacy pyannote, and `pyannote/segmentation-3.0`

This design keeps call sites stable while allowing ML-backed VAD integration in later phases.

### Practical strategy trial order

For user-facing operation, the safest trial order with the current implementation is:

1. `adaptive_rms` (default): zero extra runtime requirements and generally robust baseline.
2. `rms`: simplest baseline for quick threshold-behavior comparison.
3. `webrtc`: optional `vad-ml` dependency; useful when energy-based methods mis-detect noisy regions.
4. `silero`: optional `vad-ml` dependency; often stronger speech/non-speech separation than plain RMS paths.
5. `pyannote`: optional heavy backend requiring additional runtime/model setup; reserve for difficult material. The pyannote default is `pyannote/speaker-diarization-community-1`; use explicit `--pyannote-model pyannote/segmentation-3.0` or `--pyannote-model pyannote/voice-activity-detection` when you need to reproduce those verified/legacy paths.

Recommended operator loop:

- Start from default (`adaptive_rms`).
- If anchor confidence or residual metrics are poor, move to the next strategy.
- Compare `anchor_count`, `residual_median_ms`, `residual_max_ms`, and warning volume across runs before selecting output for manual editing. The helper `python scripts/compare-vad-strategies.py --master input/master.wav --track input/speaker-a.wav --out output/vad-comparison` creates per-strategy reports and a compact JSON summary without requiring private audio in the repository.

## 3.2 Anchor candidate selection

`select_anchor_candidates()` in `analysis/anchors.py` transforms speech segments into anchors:

- Read shared defaults from `AnchorSelectionConfig` so CLI, API, and GUI use one source of truth
- Keep segments above minimum duration (default 1.0 s)
- Estimate local SNR from nearby non-segment context and spectral flatness from the candidate audio
- Choose a bounded adaptive duration from the configured minimum, base (default 4.0 s), and maximum (default 8.0 s) duration values
- Keep high-SNR, low-flatness anchors near the base duration, extend weaker/noisier anchors toward the maximum, and explicitly reject candidates when optional SNR/flatness gates are configured
- Compute anchor RMS, SNR, spectral flatness, quality multiplier, and duration as transparent quality signals
- Assign each candidate to a stratified timeline bin by midpoint
- Select top candidates per bin first so early, middle, and late regions can contribute when viable candidates exist
- Fill any remaining duration-aware budget by global quality ranking that includes confidence, SNR, flatness, and RMS
- Record diagnostics such as candidate count, selected count, per-bin counts, sparse bins, longest unanchored span, adaptive duration min/median/max, and rejected candidate counts

This prioritizes clearer, higher-energy, higher-confidence, and more distinctive regions while also preserving timeline coverage for drift estimation. Empty bins do not fail selection by themselves, but sparse overall coverage and low-quality rejections are reported so editors can inspect low-evidence regions.

## 3.3 Feature normalization and matching

`analysis/features.py` performs normalized matching:

- `extract_anchor_feature()` mean-centers anchor audio and L2-normalizes it.
- `normalized_correlation_scores()` computes normalized cross-correlation over a search region.

Implementation details:

- Numerator via FFT-based correlation (`scipy.signal.correlate(..., method="fft")`)
- Denominator via prefix sums to get per-window centered norms efficiently
- Output score range is approximately `[-1, 1]` where higher is better

This provides scale-insensitive similarity scoring and keeps search practical for long tracks.

## 3.4 Drift matching around expected positions

`match_anchors_for_drift()` in `analysis/drift.py`:

- For each local anchor, estimate expected master position from initial offset
- Search only within a local time window (default ±6 s, plus anchor length)
- Compute NCC scores and take the best-scoring match
- Produce `AnchorMatch` with:
  - local/master times
  - implied offset
  - score
  - confidence

Restricting search around expected positions reduces false matches and runtime.

## 3.5 Weighted linear drift model with outlier rejection

`fit_linear_drift_model()` fits:

$$
\text{master\_time} = \text{stretch\_ratio} \cdot \text{local\_time}+ \text{offset\_seconds}
$$

Algorithm:

1. Build arrays of local/master anchor starts
2. Use confidence-derived weights
3. Solve weighted least squares (`_weighted_linear_fit`)
4. Compute residuals in ms
5. Run iterative outlier rejection (up to 2 rounds) using robust threshold from median/MAD
6. Refit on inliers and emit summary metrics:
   - `offset_seconds`
   - `stretch_ratio`
   - `anchor_count`
   - `residual_median_ms`
   - `residual_max_ms`

If fewer than 2 matches exist, drift estimation returns `None`.

## 3.6 Global correction and optional local adjustment

- `alignment/timeline.py` applies the global linear model to produce master-aligned samples.
- `alignment/local_adjust.py` can optionally apply local split-and-shift corrections near large residual events.

Local adjustment is intentionally optional and should be used with report review.

---

## 4. Reporting and observability

The tool writes:

- `sync-report.json`: structured alignment diagnostics for master + per-track stages.
  Top-level metadata identifies the report as `report_type: "alignment_diagnostics"`
  with `schema_version: "1"`.
- `sync-markers.csv`: marker-style timeline information for manual review
- `warnings.txt`: human-readable warnings and confidence issues

The code is designed to expose uncertainty (insufficient anchors, large residuals, skipped rendering) rather than hiding it.

---

## 5. Practical reading order for new contributors

If you are new to this codebase, read in this order:

1. `src/double_ender_sync/cli.py` (pipeline control flow)
2. `src/double_ender_sync/analysis/vad.py` (speech segmentation)
3. `src/double_ender_sync/analysis/anchors.py` (anchor generation)
4. `src/double_ender_sync/alignment/offset.py` (initial offset)
5. `src/double_ender_sync/analysis/drift.py` + `features.py` (multi-anchor drift model)
6. `src/double_ender_sync/alignment/timeline.py` (global correction; renders through a monotonic `t_master = f(t_local)` mapping, pads outside-support and internal-gap regions with silence, and processes long outputs in chunks)
7. `src/double_ender_sync/report/report.py` (final output schema)

This sequence matches the runtime pipeline and makes it easier to map implementation details to output artifacts.
