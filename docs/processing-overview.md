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

### Analysis (`analysis/`)

- `vad.py`: RMS energy based speech segment detection.
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

## 3.1 Speech detection (RMS VAD)

`detect_speech_segments()` in `analysis/vad.py` uses short-time RMS energy:

- Frame the mono signal (default 30 ms frame / 10 ms hop)
- Compute RMS per frame
- Mark frames as speech if RMS >= threshold (default `0.02`)
- Merge contiguous speech frames into segments
- Filter out short segments (default minimum 300 ms)

This is intentionally simple and deterministic, designed for robust anchor generation rather than linguistic analysis.

## 3.2 Anchor candidate selection

`select_anchor_candidates()` in `analysis/anchors.py` transforms speech segments into anchors:

- Keep segments above minimum duration (default 1.0 s)
- Truncate each anchor to max duration (default 4.0 s)
- Compute anchor RMS as a quality signal
- Rank by `(confidence, rms)` descending
- Keep top N anchors (currently top 5)

This prioritizes clearer, higher-energy and higher-confidence regions while bounding compute cost.

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

\[
\text{master\_time} = \text{stretch\_ratio} \cdot \text{local\_time} + \text{offset\_seconds}
\]

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

- `sync-report.json`: structured diagnostics for master + per-track stages
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
6. `src/double_ender_sync/alignment/timeline.py` (global correction)
7. `src/double_ender_sync/report/report.py` (final output schema)

This sequence matches the runtime pipeline and makes it easier to map implementation details to output artifacts.
