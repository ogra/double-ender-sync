# Calibration sources and synthetic drift fixtures

Calibration work is useful only when the source material and thresholds are documented clearly. The alignment engine must not hide uncertainty, and calibration data must not put private podcast recordings at risk.

## Allowed calibration sources

Use calibration material in this order of preference:

1. **Deterministic synthetic fixtures** in `tests/helpers/synthetic_drift.py` for unit tests and model-selection regressions.
   - Available scenarios include known offset audio, constant drift anchors, piecewise drift anchors, smooth spline-like drift anchors, noisy anchors, sparse anchors, and dropout-like anchor gaps.
   - These helpers are small, reproducible, and suitable for committed tests.
2. **Generated synthetic demo packs** from `double-ender-sync-generate-demo-pack` for manual CLI/API calibration runs.
   - The script writes WAVs and a manifest into a caller-provided output directory.
   - Generated WAVs are artifacts and must not be committed.
3. **Publicly redistributable audio** only when its license allows repository storage and the file is small enough for normal tests.
   - Prefer source-only generation recipes over binary fixtures.
   - Record the license and source URL in documentation if a public binary fixture is ever added.
4. **Private production audio** only for local/manual calibration outside the repository.
   - Keep reports, notes, and threshold summaries anonymized before sharing.
   - Never upload or commit private recordings.

## Prohibited material and practices

Do not commit any of the following:

- Private podcast recordings, call recordings, stems, exports, or excerpts.
- Generated calibration WAVs from `double-ender-sync-generate-demo-pack`.
- Large binary fixtures that make normal unit tests slow or hard to clone.
- Reports that expose private filesystem paths, guest names, episode titles, or transcript-like content.
- Threshold changes justified only by an undocumented private recording.

Do not add network calls or external audio uploads to calibration workflows. Calibration should be reproducible locally and offline unless an explicit future task says otherwise.

## Synthetic helper usage

Use the test helper module when adding regression coverage for drift behavior:

```python
from tests.helpers.synthetic_drift import noisy_anchor_set, sparse_anchor_set

anchors = noisy_anchor_set()
drift = select_drift_model(anchors.matches, config, anchors.local_duration_seconds)
```

The helpers return deterministic `AnchorMatch` lists and local durations so tests can assert model-selection behavior without duplicating anchor construction logic.

## Generating a demo pack

Run the generator into a temporary or ignored directory outside committed fixtures:

```bash
double-ender-sync-generate-demo-pack --out /tmp/double-ender-calibration
```

Then run the CLI against an individual case, for example:

```bash
double-ender-sync \
  --master /tmp/double-ender-calibration/constant-drift/master.wav \
  --track /tmp/double-ender-calibration/constant-drift/speaker-a.wav \
  --out /tmp/double-ender-calibration/constant-drift-output \
  --allow-nonlinear-drift \
  --verbose-report
```

The output directory should be treated as a local experiment artifact. Do not commit the generated WAVs, manifests, or reports unless a future task explicitly asks for a sanitized text summary.

## Recording threshold-tuning decisions

When changing drift or anchor thresholds, include a short calibration note in the PR description or a dedicated docs update. Record:

- The calibration source category: synthetic helper, generated demo pack, public audio, or private local-only audio.
- The exact command or test used.
- The threshold values before and after the change.
- The observed effect on `anchor_count`, `residual_median_ms`, `residual_max_ms`, `fallback_reason`, warning codes, and selected `model_type`.
- Whether non-linear drift was gated by `allow_nonlinear_drift`.
- Any low-confidence regions, unsupported regions, or dropout-like gaps that still require manual inspection.

For private audio, write only anonymized aggregate metrics and do not include filenames, participant names, paths, transcripts, or raw snippets.
