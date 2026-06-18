# Initial Offset Safety Net

This document explains the initial-offset safety net added to the alignment pipeline.

The goal is conservative: if the first offset estimate is weak or ambiguous, the pipeline should widen its search carefully, expose that uncertainty in the report, and only switch to a coarser fallback when that fallback is materially more reliable.

## Why this exists

Initial offset estimation drives the downstream drift-anchor search window. If the first offset is wrong, drift fitting can start from the wrong region of the master and either fail noisily or converge on the wrong anchors.

The safety net reduces that risk by combining three controls:

1. A confidence gate for the primary anchor-based NCC estimate.
2. An optional coarse whole-recording FFT fallback for low-confidence cases.
3. A master-side speech-overlap filter so drift anchors are less likely to match local speech into master silence.

## Selection flow

1. The pipeline computes the primary initial offset from anchor NCC.
2. The result is assigned a confidence score and then mapped into a confidence band: `high`, `medium`, `low`, or `failed`.
3. If the selected anchor estimate is below `--initial-offset-min-confidence` and coarse fallback is enabled, the pipeline also runs the downsampled whole-recording FFT fallback.
4. The fallback is selected only if it passes its own confidence and peak-distinctness gates and is sufficiently better than the anchor estimate, unless the anchor estimate already failed.
5. The selected confidence band then chooses the drift-anchor search radius, capped by `--max-drift-search-radius-seconds`.
6. During drift-anchor matching, master-side VAD overlap checks can reject matches that land mostly in master silence.

## CLI option reference

### Initial offset confidence gate

- `--initial-offset-min-confidence`
  Minimum confidence required before the pipeline skips the coarse fallback attempt.

- `--high-confidence-threshold`
  Lower bound of the `high` confidence band.

- `--medium-confidence-threshold`
  Lower bound of the `medium` confidence band.

- `--low-confidence-threshold`
  Lower bound of the `low` confidence band.

Constraints:

- The thresholds must satisfy `0 < low < medium < high < 1`.
- `--initial-offset-min-confidence` must be greater than or equal to the low threshold.

### Coarse fallback controls

- `--coarse-fallback-enabled` / `--no-coarse-fallback`
  Enable or disable the coarse whole-recording fallback.

- `--coarse-fallback-sample-rate`
  Downsample rate used by the coarse fallback correlation.

- `--coarse-fallback-max-duration-seconds`
  Optional duration cap for the fallback analysis. Use `none`, `off`, or `disabled` to remove the cap.

- `--coarse-fallback-max-memory-mb`
  Soft FFT working-memory budget in MiB.

- `--coarse-fallback-min-peak-margin`
  Minimum distinctness required from the fallback peak.

- `--coarse-fallback-min-confidence`
  Minimum confidence required before the fallback estimate can be selected.

- `--coarse-fallback-confidence-margin`
  Minimum confidence advantage required over the anchor estimate before the fallback replaces it, unless the anchor estimate is already in the `failed` band.

Operational notes:

- Very long inputs may be truncated for the fallback when `--coarse-fallback-max-duration-seconds` is set.
- Memory pressure can reduce the working span or other internal sizing decisions. The report surfaces this rather than hiding it.

### Drift search radius policy

- `--max-drift-search-radius-seconds`
  Global cap for the selected drift-anchor search radius.

- `--high-confidence-search-radius-seconds`
  Radius used for `high` confidence initial offsets.

- `--medium-confidence-search-radius-seconds`
  Radius used for `medium` confidence initial offsets.

- `--low-confidence-search-radius-seconds`
  Radius used for `low` confidence initial offsets.

Behavior notes:

- `failed` initial offsets produce no usable drift search radius.
- The configured per-band radii do not need to be ordered because the effective value is capped independently by `--max-drift-search-radius-seconds`.

### Master VAD filter controls

- `--master-vad-filter-enabled` / `--no-master-vad-filter`
  Enable or disable master-side speech-overlap rejection during drift-anchor matching.

- `--master-vad-min-overlap-ratio`
  Minimum required ratio of master speech overlap for a matched span.

- `--master-vad-padding-seconds`
  Padding applied around master speech segments before overlap checks.

- `--master-vad-uncertain-policy {warn,skip,reject}`
  Decide what to do when master-side speech evidence is unavailable or inconclusive.

Policy meanings:

- `warn`: keep matching, but surface a warning.
- `skip`: bypass the VAD gate for the uncertain condition.
- `reject`: reject those matches conservatively.

## Report fields

Each track in `sync-report.json` now includes compact initial-offset safety data even in the default non-verbose report.

### `initial_offset`

Common fields:

- `offset_seconds`
- `confidence`
- `local_anchor_start`
- `master_anchor_start`
- `score`
- `estimation_method`
- `confidence_band`
- `fallback_attempted`
- `fallback_selected`
- `fallback_reason`
- `initial_offset_confidence_threshold`
- `selected_drift_search_radius_seconds`
- `max_drift_search_radius_seconds`
- `radius_reason`
- `warnings`

Additional diagnostic fields when available:

- `anchor_ncc`
  Carries the anchor-based estimate plus NCC peak diagnostics such as margin, prominence, peak width, and the second-best peak.

- `coarse_fft_fallback`
  Carries the fallback estimate plus its own diagnostics, including whether the fallback was memory-limited or duration-capped.

- `master_vad_rejection_count`
  Number of drift-anchor matches rejected by the master-VAD gate.

### `initial_offset_safety_diagnostics`

This compact section carries:

- `config`
  Effective initial-offset safety configuration used for the track.

- `warnings`
  Machine-readable warning codes such as:
  - `initial_offset_low_confidence`
  - `coarse_offset_fallback_used`
  - `coarse_offset_fallback_attempted_but_rejected`
  - `coarse_offset_fallback_failed_or_ambiguous`
  - `coarse_fallback_memory_limited`
  - `coarse_fallback_duration_capped`
  - `master_vad_unavailable`

- `master_vad_rejection_count`
  Count repeated here so compact reports can surface master-VAD impact without expanding the verbose match list.

### Promoted warnings

The report builder also promotes safety-net conditions into track/global warning entries, including:

- `INITIAL_OFFSET_LOW_CONFIDENCE`
- `COARSE_OFFSET_FALLBACK_USED`
- `MASTER_VAD_ANCHOR_REJECTED`
- `MASTER_VAD_NO_SPEECH_SEGMENTS`

## Python API

The Python API exposes the same behavior through `AlignmentOptions.initial_offset_safety`.

```python
from pathlib import Path

from double_ender_sync import AlignmentOptions, InitialOffsetSafetyConfig, run_alignment

options = AlignmentOptions(
    master=Path("input/master.wav"),
    tracks=[Path("input/speaker-a.wav")],
    out=Path("output"),
    initial_offset_safety=InitialOffsetSafetyConfig(
        initial_offset_min_confidence=0.50,
        coarse_fallback_enabled=True,
        coarse_fallback_sample_rate=8000,
        coarse_fallback_max_memory_mb=1024.0,
        high_confidence_search_radius_seconds=6.0,
        medium_confidence_search_radius_seconds=12.0,
        low_confidence_search_radius_seconds=20.0,
        master_vad_filter_enabled=True,
        master_vad_uncertain_policy="warn",
    ),
)

exit_code = run_alignment(options)
```

Validation is shared with the CLI/config layer. Invalid threshold ordering or unsupported policy combinations fail early instead of silently falling back to defaults.

## Practical tuning guidance

- If anchor matches are usually correct but some files mis-start, raise `--initial-offset-min-confidence` slightly before widening all drift settings.
- If fallback is too expensive on long files, lower `--coarse-fallback-sample-rate`, set `--coarse-fallback-max-duration-seconds`, or reduce `--coarse-fallback-max-memory-mb`.
- If the fallback is too eager, raise `--coarse-fallback-min-confidence` or `--coarse-fallback-confidence-margin`.
- If repeated speech still causes drift misalignment, keep the master VAD filter enabled and inspect `master_vad_rejection_count` together with per-track warnings.
- If many trustworthy files are being rejected by master VAD, try increasing `--master-vad-padding-seconds` first before weakening the overlap ratio.
