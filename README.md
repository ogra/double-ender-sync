# double-ender-sync

`double-ender-sync` is a CLI tool that aligns each speaker's local recording to a mixed reference recording ("master") for podcast post-production.

It focuses on **time alignment and diagnostics**, not final audio mixing.

## Project status

This project is currently **experimental (alpha)**.

It can produce useful alignment results for some double-ender podcast recordings, but it is not yet a fully validated production-grade editor. Always review generated reports, markers, warnings, and synced audio manually before using outputs in final production.

## What this tool does

- Detects initial timing offset between each local track and the master.
- Estimates long-duration clock drift from multiple anchor points.
- Applies global time correction and exports synced WAV files.
- Produces alignment diagnostics (`sync-report.json`, markers, warnings) so editors can review confidence and problem areas.

Offset definition:

```text
offset_seconds = master_time - local_time
```

## Install

### Requirements

- Python 3.11+
- WAV input files for master and local tracks

### Recommended install flow (important)

Start with the **core CLI only** and add extras only when needed:

```bash
pip install double-ender-sync
```

- Core install includes CLI alignment pipeline (default `resample` stretch) and excludes GUI / pitch-preserving dependencies.
- Add GUI only when you need desktop operation:

```bash
pip install "double-ender-sync[gui]"
```

- Add pitch-preserving stretch support only when needed:

```bash
pip install "double-ender-sync[stretch]"
```

- Install everything (GUI + stretch + dev-oriented extras) only if you explicitly want full feature/development setup:

```bash
pip install "double-ender-sync[all]"
```

### From source

```bash
pip install .
```

### Development install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Run tests:

```bash
pip install -e ".[dev]"
pytest
```

If you need pitch-preserving stretch during development, install it explicitly:

```bash
pip install -e ".[stretch]"
```

After installation, the command is available as:

```bash
double-ender-sync --help
```

## Quick start

Input example:

```text
input/
  master.wav
  speaker-a.wav
  speaker-b.wav
```

Run:

```bash
double-ender-sync \
  --master input/master.wav \
  --track input/speaker-a.wav \
  --track input/speaker-b.wav \
  --out output/
```

## Output files

Typical output:

```text
output/
  speaker-a.synced.wav
  speaker-b.synced.wav
  sync-report.json
  sync-markers.csv
  warnings.txt
```

## Useful options

- `--analysis-sample-rate 16000`  
  Set analysis sample rate used for feature extraction/matching.
- `--local-adjust-enabled`  
  Enable **experimental** optional local adjustment around large residual errors. This is disabled by default and should only be used after manual report/audio review.
- `--local-adjust-threshold-ms 80`  
  Threshold for triggering local adjustment diagnostics/correction.
- `--normalize-output`  
  Normalize final synced WAV peak level before writing. Disabled by default.
- `--stretch-ratio-warning-threshold 0.003`  
  Warn when `abs(stretch_ratio - 1.0)` exceeds threshold (default `0.003` = 0.3%).
- `--stretch-ratio-auto-continue`  
  Skip interactive confirmation and continue even when stretch ratio warning threshold is exceeded.
- `--stretch-method {resample,pitch_preserving}`  
  Global correction method. `resample` is default. `pitch_preserving` uses librosa and prioritizes pitch stability for larger drift corrections.
- `--debug`  
  Enable debug logging to identify which stage is running when resource usage spikes.
- `--log-file output/debug.log`  
  Write logs to a specific file path (default: `output/double-ender-sync.log`).

Use `double-ender-sync --help` for the full option list.

### GUI (PySide6, drag & drop)

This project also provides an optional desktop GUI built with PySide6.

Install with GUI dependency (required for `double-ender-sync-gui`):

```bash
pip install -e ".[gui]"
```

Launch GUI:

```bash
double-ender-sync-gui
```

### Language option (`--lang`) common specification

Project-wide behavior for language resolution is fixed as follows:

- `--lang <code>` is accepted (for example: `en`, `ja`).
- If `--lang` is omitted, system locale is used (`LC_ALL` then `LANG`).
- If the normalized language is unsupported, fallback is `en`.
- Regional codes are normalized to their language part before support checks (for example: `en-US` -> `en`, `ja_JP.UTF-8` -> `ja`).
- GUI applies this resolver first, and the same resolver is reusable from CLI/API so each entry point does not need separate language detection logic.

Examples:

```bash
double-ender-sync-gui --lang en
double-ender-sync-gui --lang ja
double-ender-sync-gui
```

GUI features (current):

- Select `master.wav`
- Drag and drop multiple speaker `.wav` tracks
- Choose output directory
- Run the same alignment pipeline as CLI



## Python API (import from another project)

In addition to CLI usage, you can run the same pipeline from Python.

```python
from pathlib import Path

from double_ender_sync import AlignmentOptions, run_alignment

options = AlignmentOptions(
    master=Path("input/master.wav"),
    tracks=[Path("input/speaker-a.wav"), Path("input/speaker-b.wav")],
    out=Path("output"),
    analysis_sample_rate=16000,
    local_adjust_enabled=False,
    normalize_output=False,
)

exit_code = run_alignment(options)
if exit_code != 0:
    raise RuntimeError(f"alignment failed with exit code {exit_code}")
```

`run_alignment(...)` returns the same exit code semantics as the CLI `main(...)`.


## Translation operations rules

- Translation keys are domain-prefixed and stable (`gui.*`, `cli.*`, `api.*`, `errors.*`, `warnings.*`).
- Never use display text itself as a key.
- Missing key behavior is unified:
  - If the target locale does not have the key, fallback to `en`.
  - If `en` also does not have the key, show the key string and emit a warning log.
- Placeholder formatting is unified (for example: `"File not found: {path}"`).
  - Placeholder names must match exactly across all languages for the same key.

### Adding a new language

1. Add a locale file: `src/double_ender_sync/i18n/locales/<lang>.json`.
2. Add `<lang>` to `SUPPORTED_LANGUAGES` in `src/double_ender_sync/i18n/resolver.py`.
3. Run required key validation: `double-ender-sync-validate-locales` (or `python -m double_ender_sync.i18n.validate`).
4. Verify UI rendering manually:
   - launch `double-ender-sync-gui` with your locale selected,
   - confirm labels/dialog/errors render correctly,
   - run one alignment and check runtime messages/logs.

## Intended use case

This tool is intended for podcast double-ender workflows where:

- each participant records a local WAV file,
- a mixed call recording is available as timing reference,
- local recordings contain enough speech anchors across the session,
- final output is reviewed and edited by a human in a DAW.

It may perform poorly when:

- the master recording is heavily compressed/noisy or missing large sections,
- a local track contains very little speech,
- local and master recordings contain different edits,
- long dropouts or repeated phrases confuse anchor matching,
- timing changes are non-linear and not well approximated by a simple drift model.

## Reviewing the result

After running the tool, inspect:

- `warnings.txt` for low-confidence regions and skipped adjustments,
- `sync-markers.csv` for anchor/residual positions,
- `sync-report.json` for per-track offset/stretch/residual diagnostics,
- exported `.synced.wav` files by listening in your DAW.

Do not treat generated synced files as final mastered audio.

## Temporary files

This tool creates temporary memory-mapped files during analysis to reduce peak RAM usage for long recordings. These temporary files are cleaned up at the end of a normal CLI run.

## Current implementation status

Implemented pipeline includes:

1. audio loading and normalization for analysis,
2. speech-region detection (RMS-based),
3. anchor selection and matching against master,
4. initial offset estimation,
5. multi-anchor linear drift estimation,
6. global correction and synced WAV export,
7. detailed reporting with warnings/errors.

## Scope and non-goals

This project does **not** do final podcast mastering tasks such as:

- noise reduction,
- EQ/compression/loudness normalization,
- transcript-based editing,
- final mixdown/publishing.

The expected workflow is:

```text
raw recordings -> double-ender-sync -> synced WAV + report -> human DAW edit
```

## Licensing and distribution policy

Project code is MIT licensed.

Current policy is **source-only distribution** from this repository. No official prebuilt binaries are published.

Before publishing any binary builds in the future, review third-party obligations (especially LGPL-related components) and update distribution/legal documentation accordingly.

See:

- `THIRD_PARTY_NOTICES.md`
- `docs/licensing-source-only.md`
