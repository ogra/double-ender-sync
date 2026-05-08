# Runtime tuning guide for long recordings

This guide describes a practical, report-driven workflow for reducing
`double-ender-sync` processing time on long podcast recordings while keeping
alignment decisions reviewable.

The examples assume a long session with one master file and two local speaker
files, such as a recording around 1 hour 45 minutes long. Adjust paths and output
directory names for your project.

## Tuning principles

Optimize for the fastest setting that still produces trustworthy diagnostics.
Do not choose a configuration only because it is faster.

For each trial run, compare both runtime and alignment quality:

- Total wall-clock runtime.
- `sync-report.json`.
- `warnings.txt`.
- `double-ender-sync.log` (`--debug` increases verbosity).
- Per-track `anchor_count`.
- Per-track `residual_median_ms` and `residual_max_ms`.
- Per-track `stretch_ratio`.
- Anchor-selection diagnostics, especially `selected_anchor_count` and
  `longest_unanchored_span_seconds`.
- Manual spot checks near the beginning, middle, and end of the synced audio.

A faster run is only a good candidate when residuals, warnings, anchor coverage,
and listening checks remain acceptable.

## Why anchor count matters

For long recordings, anchor matching is often a major part of total runtime. With
the default anchor density of `1.0` anchor per minute and the default safety cap
of `120` anchors, a 105-minute recording may select roughly 105 anchors per local
speaker track when enough valid speech regions exist.

Reducing selected anchors is usually the first and safest tuning lever because it
cuts repeated matching work while preserving the same analysis model and VAD
strategy.

## Step 0: create a baseline

Start with a default run and keep its outputs as the comparison point.

```bash
double-ender-sync \
  --master input/master.wav \
  --track input/speaker-1.wav \
  --track input/speaker-2.wav \
  --out output/baseline \
  --debug
```

Record the total runtime and inspect the generated reports before changing any
options.

## Step 1: cap selected anchors

Try reducing `--max-anchor-count` first. For long podcast episodes, start with 60
anchors per track, then try 40 or 30 only if quality remains stable.

```bash
double-ender-sync \
  --master input/master.wav \
  --track input/speaker-1.wav \
  --track input/speaker-2.wav \
  --out output/tune-anchor-60 \
  --max-anchor-count 60 \
  --debug
```

```bash
double-ender-sync \
  --master input/master.wav \
  --track input/speaker-1.wav \
  --track input/speaker-2.wav \
  --out output/tune-anchor-40 \
  --max-anchor-count 40 \
  --debug
```

```bash
double-ender-sync \
  --master input/master.wav \
  --track input/speaker-1.wav \
  --track input/speaker-2.wav \
  --out output/tune-anchor-30 \
  --max-anchor-count 30 \
  --debug
```

Prefer the lowest cap that does not introduce significant new warnings, sparse
coverage, or audible timing problems. If 30 anchors increases residuals or leaves
large timeline regions without anchors, move back to 40 or 60.

## Step 2: tune anchor density instead of an absolute cap

`--max-anchor-count` sets an absolute ceiling. `--anchor-density-per-minute`
scales the target anchor budget with recording length.

For a 105-minute recording, `0.5` anchors per minute targets roughly 53 anchors
per local track before caps and candidate availability are considered:

```bash
double-ender-sync \
  --master input/master.wav \
  --track input/speaker-1.wav \
  --track input/speaker-2.wav \
  --out output/tune-density-05 \
  --anchor-density-per-minute 0.5 \
  --debug
```

A more aggressive trial is `0.3` anchors per minute:

```bash
double-ender-sync \
  --master input/master.wav \
  --track input/speaker-1.wav \
  --track input/speaker-2.wav \
  --out output/tune-density-03 \
  --anchor-density-per-minute 0.3 \
  --debug
```

Use density when you want one setting that scales across episode lengths. Use a
maximum count when you mainly want to stop very long recordings from selecting too
many anchors.

## Step 3: lower the analysis sample rate

The default analysis sample rate is `16000`. Lower rates reduce feature and
matching cost, but can also reduce matching precision on difficult material.

After finding a reasonable anchor cap, try `12000` first:

```bash
double-ender-sync \
  --master input/master.wav \
  --track input/speaker-1.wav \
  --track input/speaker-2.wav \
  --out output/tune-sr-12000-anchor-60 \
  --analysis-sample-rate 12000 \
  --max-anchor-count 60 \
  --debug
```

Then try `8000` only if you need more speed:

```bash
double-ender-sync \
  --master input/master.wav \
  --track input/speaker-1.wav \
  --track input/speaker-2.wav \
  --out output/tune-sr-8000-anchor-60 \
  --analysis-sample-rate 8000 \
  --max-anchor-count 60 \
  --debug
```

If `8000` changes offsets, stretch ratios, residuals, or warnings noticeably,
return to `12000` or the default `16000`.

## Step 4: shorten anchors only after count and sample-rate tuning

Shorter anchors reduce per-anchor matching cost, but they can make ambiguous or
repeated speech easier to mismatch. Treat this as a secondary tuning lever.

```bash
double-ender-sync \
  --master input/master.wav \
  --track input/speaker-1.wav \
  --track input/speaker-2.wav \
  --out output/tune-short-anchors \
  --analysis-sample-rate 12000 \
  --max-anchor-count 60 \
  --min-anchor-duration 0.75 \
  --base-anchor-duration 2.0 \
  --max-anchor-duration 4.0 \
  --debug
```

If residuals worsen or the selected anchors look less distinctive, restore the
default anchor duration policy and keep the speed gains from anchor count or
analysis sample rate instead.

## Step 5: keep VAD lightweight for speed-focused tuning

The default `adaptive_rms` strategy is a good first choice for runtime tuning
because it is lightweight and has no extra ML runtime dependency.

You can try `rms` as a quick comparison:

```bash
double-ender-sync \
  --master input/master.wav \
  --track input/speaker-1.wav \
  --track input/speaker-2.wav \
  --out output/tune-vad-rms \
  --vad-strategy rms \
  --analysis-sample-rate 12000 \
  --max-anchor-count 60 \
  --debug
```

Avoid switching to heavier VAD backends such as `pyannote` or `silero` solely for
runtime reduction. They can be useful for difficult material, but they are not the
first options to try when the goal is a shorter run.

## Step 6: keep global correction simple while tuning

For speed-focused experiments, keep the default global correction method:

```bash
--stretch-method resample
```

Use `pitch_preserving` only when you specifically need pitch-preserving output
and have accepted the extra runtime cost.

> **Note:** `pitch_preserving` requires `librosa`, which is not installed by
> default. Install it with the `stretch` extra before using this method,
> otherwise the tool will raise a runtime error that may look like an alignment
> failure:
>
> ```bash
> pip install "double-ender-sync[stretch]"
> ```

Also keep local adjustment disabled during runtime comparisons unless you are
specifically validating local adjustment behavior:

```text
Do not add --local-adjust-enabled during basic runtime tuning.
```

## Recommended trial order

Run these configurations in order and stop when you find an acceptable speed and
quality balance.

1. Baseline defaults with `--debug`.
2. `--max-anchor-count 60`.
3. `--analysis-sample-rate 12000 --max-anchor-count 60`.
4. `--analysis-sample-rate 8000 --max-anchor-count 60`.
5. `--analysis-sample-rate 12000 --max-anchor-count 40`.
6. `--analysis-sample-rate 12000 --max-anchor-count 40 --min-anchor-duration 0.75 --base-anchor-duration 2.0 --max-anchor-duration 4.0`.

For many long recordings, a good first production candidate is:

```bash
--analysis-sample-rate 12000 --max-anchor-count 60
```

If that remains reliable, try:

```bash
--analysis-sample-rate 12000 --max-anchor-count 40
```

or:

```bash
--analysis-sample-rate 8000 --max-anchor-count 60
```

## Acceptance checklist

Before adopting a faster setting, confirm the following:

| Check | Suggested target |
| --- | --- |
| Runtime | Clearly faster than the baseline |
| Fitted anchors | At least 8, preferably 20 or more for long recordings |
| Median residual | Preferably below 30 ms |
| Maximum residual | Preferably below 100 ms |
| Warnings | No new severe alignment or coverage warnings |
| Anchor coverage | Anchors are not concentrated in only one part of the timeline |
| Listening review | Beginning, middle, and end remain aligned |

These values are practical starting points, not guarantees. When the report shows
uncertainty, trust the warnings and inspect the audio manually.
