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

- Add ML-VAD runtime dependencies for lightweight ML backends (`silero` / `webrtc`):

```bash
pip install "double-ender-sync[vad-ml]"
```

- Add the dedicated pyannote stack only when you need `--vad-strategy pyannote`:

```bash
pip install "double-ender-sync[vad-pyannote]"
```

> Note: as of 2026-05-06, the pyannote extra targets the current Python 3.14-compatible stack verified by `python -m pip index versions`: `pyannote.audio>=4.0.4,<5`, `torch==2.11.0`, `torchaudio==2.11.0`, and `torchcodec>=0.11.1,<0.12`. `pyannote.audio` pulls in `pyannote-core>=6.0.1` and therefore requires `numpy>=2.0`; the core package allows NumPy 1.26 or newer, so installing the pyannote or all extras may upgrade an existing environment to NumPy 2.x. The project does not call `torchaudio.load` directly; for `--vad-strategy pyannote`, it passes an in-memory Torch waveform into pyannote so pyannote does not decode temporary audio files through its legacy torchaudio-backed file I/O path. By default, the pyannote backend loads `pyannote/speaker-diarization-community-1`, which is a gated Hugging Face pipeline requiring accepted terms and a token. You can override the model/pipeline with `--pyannote-model`; `pyannote/segmentation-3.0` keeps using the verified segmentation-model VAD loader, and `pyannote/voice-activity-detection` remains available as an explicit legacy pipeline. PyTorch 2.6+ defaults checkpoint loads to `weights_only=True`; the pyannote backend retries only that known checkpoint-compatibility failure with `weights_only=False`, so use `pyannote` only with model checkpoints you trust. With pyannote.audio 4.0.4 and the default community pipeline, this retry path is not expected during normal loading; it is retained for explicitly selected legacy checkpoints. See `docs/pyannote-vad-modernization-plan.md` for the executable modernization subtasks.
>
> **macOS (Homebrew) runtime caveat for pyannote/torio**: some environments only succeed with FFmpeg 6 libraries (not the latest FFmpeg 8.x keg). If you see `libtorio_ffmpeg*.so` load failures or `Library not loaded: @rpath/libavutil.*.dylib`, install `ffmpeg@6` and point dynamic loading to it before running the CLI:
>
> ```bash
> brew install ffmpeg@6
> export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@6)/lib:${DYLD_LIBRARY_PATH:-}"
> double-ender-sync ... --vad-strategy pyannote
> ```

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

Synthetic drift fixtures for tests live in `tests/helpers/synthetic_drift.py`; use them instead of duplicating anchor-generation logic when adding offset, constant-drift, piecewise-drift, spline, noisy-anchor, sparse-anchor, or dropout-gap regressions. For manual calibration, see [Calibration sources and synthetic drift fixtures](docs/calibration.md) and generate temporary demo packs with `double-ender-sync-generate-demo-pack`. Generated WAVs, private podcast recordings, and private calibration reports must not be committed to this repository.

If you need pitch-preserving stretch during development, install it explicitly:

```bash
pip install -e ".[stretch]"
```

After installation, the command is available as:

```bash
double-ender-sync --help
```

Print the installed version from the CLI with either long or short version flags:

```bash
double-ender-sync --version
double-ender-sync -V
# version 0.2.2
```

The same version is exposed to Python callers through the package/API (`double_ender_sync.__version__` and `double_ender_sync.api.get_version()` both return `0.2.2`) and is shown in the GUI footer as `v0.2.2`.

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

`sync-report.json` is an alignment diagnostics report. Its top-level metadata
uses `report_type: "alignment_diagnostics"` and `schema_version: "1"`, followed
by `analysis`, `master`, `tracks`, `warnings`, and `errors` sections.

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
  Global correction method. `resample` is default and now renders through the general monotonic time-mapping interface for future drift models while preserving the linear default/control behavior, including tracks whose original sample rate differs from the master sample rate. `pitch_preserving` uses librosa, prioritizes pitch stability for larger drift corrections, and currently supports only `LinearDrift`; unsupported renderer/model combinations fail clearly rather than silently producing audio.
- `--min-anchor-duration 1.0` / `--base-anchor-duration 4.0` / `--max-anchor-duration 8.0`
  Configure the adaptive speech-derived anchor duration policy shared by CLI/API/GUI runs. High-SNR, distinctive material stays near the base duration; noisier or spectrally flatter material can extend toward the maximum instead of using a globally fixed clip length. If `--base-anchor-duration` is omitted, the CLI derives an effective default by clamping 4.0 seconds into the configured min/max bounds, so max-only or min-only tuning remains valid. Explicit base values are validated against the min/max bounds and fail clearly when inconsistent.
- `--min-snr-db <db>` / `--spectral-flatness-threshold <0.0-1.0>`
  Optional quality gates for rejecting low-SNR or noise-like anchor candidates. Defaults leave these hard rejections disabled while still recording SNR/flatness diagnostics and confidence downgrades.
- `--anchor-density-per-minute 1.0` / `--max-anchor-density-per-minute 2.0`
  Configure the duration-aware anchor budget and the validation ceiling for custom density values. API callers that raise `anchor_density_per_minute` should raise `max_anchor_density_per_minute` with it.
- `--min-anchor-count 5` / `--max-anchor-count 120`
  Configure the minimum target budget for short recordings and the safety cap for selected anchor candidates. Use `none` for `--max-anchor-count` only for debugging/unbounded experiments.
- `--stratified-bin-count <count>` / `--anchors-per-bin <count>`
  Override the automatic timeline stratification used for drift-anchor selection. By default, the selector derives roughly one-minute bins bounded by the target anchor budget, picks top candidates per bin first, then fills the remaining budget globally; sparse coverage and long unanchored spans are reported as warnings rather than hidden.
- `--drift-model {auto,linear,piecewise_linear,spline,kalman}`
  Select the drift model policy. The default `auto` keeps `LinearDrift` as the compatibility/control model while `--allow-nonlinear-drift` is disabled. `linear` always requests the linear control model. `piecewise_linear` and `spline` are experimental and require `--allow-nonlinear-drift`; `kalman` is research/experimental and is evaluated only when explicitly requested with `--allow-nonlinear-drift`. Piecewise fitting uses continuous segments, spline fitting uses monotonic cubic PCHIP interpolation through validated support knots, and Kalman fitting models offset/rate as noisy latent states smoothed across anchors. These richer models are accepted only when anchor coverage, residual improvement, monotonicity, and local-rate plausibility checks pass. Reports preserve legacy `offset_seconds` / `stretch_ratio` fields and add `model_type`, `model_version`, `model_selection_policy`, `model_parameters`, `breakpoints`, `segments`, `segment_residual_summaries`, `knots`, `knot_residual_summaries` with per-knot anchor counts and residual statistics, `fallback_reason`, `local_rate_summary`, and Kalman-specific `uncertainty_summary`, `covariance_summary`, `uncertainty_bands`, `state_points`, and `anchor_residuals_ms` diagnostics. The renderer API supports strictly monotonic, invertible `DriftModel` mappings; regions outside model support and detected internal master-time gaps are padded with silence and exposed in `global_correction.unsupported_regions`.
- `--allow-nonlinear-drift`
  Enable experimental non-linear drift candidate evaluation and explicit research Kalman smoothing. Disabled by default; when disabled, `auto` does not attempt piecewise, spline, or Kalman fitting and reports that the safety gate retained the linear control model. Even when enabled, `auto` does not select the Kalman smoother; choose `--drift-model kalman` explicitly for research experiments.
- `--min-anchors-for-piecewise 6` / `--min-anchors-per-segment 3` / `--max-breakpoints 1`
  Configure conservative piecewise-linear breakpoint search. Sparse-anchor recordings fall back to `LinearDrift` with an explicit `fallback_reason`.
- `--min-anchors-for-spline 6` / `--spline-knot-source {auto,piecewise_boundaries,anchors}` / `--min-knot-spacing-seconds 30.0` / `--spline-validation-sample-count 1024`
  Configure monotonic cubic spline fitting. `auto` prefers accepted piecewise boundaries when available and otherwise uses confidence-filtered, spacing-decimated anchors. Reports keep the configured knot-source vocabulary (`anchors`, `piecewise_boundaries`, or `auto` resolution) and expose `model_parameters.knot_decimation_applied` when anchor spacing decimation was used. Explicit `spline` runs with `piecewise_boundaries` prefit a piecewise model for knot support without selecting it as the final model. Spline candidates are rejected with an explicit `fallback_reason` if PCHIP construction, monotonicity, derivative/rate bounds, or residual-improvement checks fail.
- `--min-anchors-for-kalman 5` / `--kalman-process-offset-noise-ms 5.0` / `--kalman-process-rate-noise-ppm 50.0` / `--kalman-observation-noise-ms 30.0`
  Configure the Phase 5 state-space / Kalman RTS smoother. This feature is **research/experimental**, not a normal-user default: it models `offset_seconds = master_time - local_time` and `rate_deviation = local_rate - 1.0`, scales observation noise from anchor confidence, emits uncertainty/covariance diagnostics, and falls back to `LinearDrift` with a `fallback_reason` when anchor coverage, monotonicity, rate plausibility, or residual-improvement checks fail.
- `--kalman-initial-offset-uncertainty-ms 250.0` / `--kalman-initial-rate-uncertainty-ppm 500.0` / `--kalman-validation-sample-count 1024`
  Configure initial uncertainty and validation density for explicit Kalman research runs. These values are reportable so calibration experiments can compare noise assumptions without hiding uncertainty.
- `--min-residual-improvement-ms 5.0` / `--min-relative-residual-improvement 0.2`
  Require both absolute and relative median residual improvement before selecting a richer model; worst-case residuals must also improve.
- `--max-abs-rate-deviation-ppm 1000` / `--max-rate-change-ppm 500` / `--warn-abs-rate-deviation-ppm 200`
  Bound accepted local rates plus adjacent segment, knot, and Kalman mapping-rate changes for piecewise, spline, and Kalman models. Fits outside hard bounds fall back to a simpler model; accepted fits above the warning bound emit report warnings for manual inspection.
- `--verbose-report`
  Write the full debug form of `sync-report.json`. Default reports keep compact, editor-facing fields: track metadata, model metadata, linear compatibility fields (`offset_seconds`, `stretch_ratio`, anchor/residual metrics), warning/error lists, unsupported-region summaries, breakpoint/knot/state-point/anchor-residual counts, and compact summaries for speech segments, anchor candidates, drift-anchor matches, anchor-selection diagnostics, drift-fit diagnostics, and local adjustment. Default reports omit the heavyweight detail arrays and diagnostic payloads: `speech_segments`, `anchor_candidates`, full `drift_anchor_matches`, full `anchor_selection_diagnostics`, full `drift_fit_diagnostics`, nested `drift_estimate`, detailed model diagnostics, full `breakpoints`, `knots`, `state_points`, `anchor_residuals`, and full `unsupported_regions`. Verbose reports add those detailed fields plus an effective configuration snapshot under `analysis.configuration_snapshot`. Use this for model-selection debugging and calibration comparisons; the same option is available through `AlignmentOptions.verbose_report` and the GUI advanced settings so CLI/API/GUI runs share the same report semantics.
- `--debug`  
  Enable debug logging to identify which stage is running when resource usage spikes.
- `--vad-strategy {silero,adaptive_rms,rms,webrtc,pyannote}`  
  Select VAD backend strategy. Default is `adaptive_rms`. Current implementation behavior:
  - `silero`: requires `vad-ml` extra; if missing, the command fails with an explicit error.
  - `adaptive_rms`: adaptive thresholding (`noise_floor + k*MAD`) for low-cost robustness.
  - `rms`: fixed-threshold baseline.
  - `webrtc`: requires `vad-ml` extra; if missing, the command fails with an explicit error.
  - `pyannote`: requires `vad-pyannote` extra (plus pyannote runtime/model access); if unavailable, the command fails with an explicit error.
    - Default pyannote model/pipeline is `pyannote/speaker-diarization-community-1`. The alignment engine ignores diarization speaker labels and uses the union of detected speech regions as VAD segments for anchor selection. This model is gated on Hugging Face and its pipeline config requires pyannote.audio 4.x, so accept the model terms, provide `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`, and upgrade the pyannote extra if an older 3.x environment was already installed.
    - Override with `--pyannote-model <id>` only when `--vad-strategy pyannote` is selected. Passing `--pyannote-model` with a non-pyannote strategy is a usage error, not a silent no-op.
    - `--pyannote-model pyannote/segmentation-3.0` uses the existing segmentation-model loader (`Model.from_pretrained` + `VoiceActivityDetection`) with conservative 100 ms `min_duration_on` / `min_duration_off` smoothing for anchor selection. This verified path remains available for explicit selection.
    - For gated/private models, set `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) and accept model terms for the selected model at `https://hf.co/<model-id>`.
- `--pyannote-model <model-id>`
  Select the pyannote model/pipeline id. Only valid with `--vad-strategy pyannote`. The default is `pyannote/speaker-diarization-community-1`; pass `--pyannote-model pyannote/segmentation-3.0` to use the verified segmentation-model VAD path, or `--pyannote-model pyannote/voice-activity-detection` to reproduce the legacy pipeline behavior.
- `--log-file output/debug.log`  
  Write logs to a specific file path (default: `output/double-ender-sync.log`).

Use `double-ender-sync --help` for the full option list.

For long recordings, see [Runtime tuning guide for long recordings](docs/runtime-tuning.md) for a report-driven process to reduce processing time while preserving alignment confidence.

### VAD strategy selection guide (recommended trial order)

If you are unsure which `--vad-strategy` to use, try them in this order:

1. `adaptive_rms` (default)
   - Best first try for most environments: no extra ML runtime, low setup cost, and robust enough for many podcast recordings.
2. `rms`
   - Simple fixed-threshold baseline. Useful as a quick comparison when `adaptive_rms` seems too strict/too loose for your material.
3. `webrtc`
   - Lightweight ML-style VAD option after installing ML extras (`pip install "double-ender-sync[vad-ml]"` for PyPI installs, or `pip install -e ".[vad-ml]"` for source/editable installs). Good next step when RMS-based detection struggles with noise/silence boundaries.
4. `silero`
   - Typically stronger speech/non-speech discrimination than simple energy thresholds, but requires the same optional ML extras install shown above.
5. `pyannote`
   - Most heavyweight option (dependency/runtime/model requirements are larger). Install `vad-pyannote` for this backend; try this last when other strategies still produce low-confidence anchors. The pyannote default is now `pyannote/speaker-diarization-community-1` to benefit from newer diarization improvements; keep comparing reports and boundary spot-checks, and use `--pyannote-model pyannote/segmentation-3.0` or `--pyannote-model pyannote/voice-activity-detection` when you need those explicit paths.

Suggested workflow:

- First run with defaults (`adaptive_rms`) and inspect `sync-report.json` / `warnings.txt`.
- If warnings indicate low anchor confidence or poor residuals, retry with the next strategy in the list.
- Keep the report from each run and compare `anchor_count`, `residual_median_ms`, and `residual_max_ms` to choose the safest result.
- To compare the same private audio across the built-in trial set without committing audio, run `python scripts/compare-vad-strategies.py --master input/master.wav --track input/speaker-a.wav --out output/vad-comparison`. The summary file compares anchor counts, residuals, warning counts, and warning severities for `adaptive_rms`, `silero`, default community pyannote, legacy pyannote, and `pyannote/segmentation-3.0`; manually spot-check speech boundaries around selected anchors before changing recommendations.

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




## Runtime troubleshooting (pyannote + FFmpeg)

When `--vad-strategy pyannote` is enabled, runtime loading depends on Torch waveform support and PyTorch checkpoint compatibility. The normal VAD path passes audio as an in-memory waveform, avoiding pyannote's file-decoding path that can emit torchaudio deprecation warnings during the PyTorch TorchCodec transition.

- TorchCodec/FFmpeg native bindings may still be needed by pyannote itself or by user-selected pyannote pipelines that perform their own file decoding.
- Keep `torch`, `torchaudio`, and `torchcodec` on a compatible set. The pyannote extra pins `torch==2.11.0` and `torchaudio==2.11.0`, so it also constrains TorchCodec to `>=0.11.1,<0.12`. If `pip` previously installed an incompatible TorchCodec, reinstall the extra or run `pip install --force-reinstall "torchcodec>=0.11.1,<0.12"`. A `Symbol not found` error from `libtorchcodec_core*.dylib` that references `torch/lib/libc10.dylib` usually points to this Torch/TorchCodec ABI mismatch rather than to the selected FFmpeg keg.
- On **macOS**, Torch/Torio may try FFmpeg major versions in descending order and can fail on incompatible majors and succeed on 6.
- Homebrew latest FFmpeg is currently 8.x, but that does not guarantee ABI compatibility with prebuilt Python extensions in your environment.
- In that case, install `ffmpeg@6` in parallel and expose its library directory via `DYLD_LIBRARY_PATH`. If the error persists after FFmpeg 6 loads successfully, check the TorchCodec version compatibility before changing FFmpeg paths again.
- If logs mention `models--pyannote--segmentation`, that can be an internal segmentation dependency of the selected pyannote pipeline rather than an explicit CLI model switch. The selected pyannote model is logged and written under `analysis.vad.pyannote_model` in `sync-report.json`.
- If the default `pyannote/speaker-diarization-community-1` model fails with an error such as `SpeakerDiarization.__init__() got an unexpected keyword argument 'plda'`, the installed pyannote.audio runtime is too old for the community-1 pipeline config. Upgrade with `pip install -U "double-ender-sync[vad-pyannote]"`, or explicitly choose `--pyannote-model pyannote/segmentation-3.0` while you keep the older runtime.
- If PyTorch reports `Weights only load failed` while loading a pyannote checkpoint, the CLI now retries that specific pyannote pipeline load with `weights_only=False`; this mirrors pre-PyTorch-2.6 behavior and should only be used with trusted pyannote/Hugging Face checkpoints.
- If the runtime warns with text like `Model was trained with pyannote.audio 0.x` or `Model was trained with torch 1.x`, treat that as a signal to use the default `pyannote/speaker-diarization-community-1` pipeline or compare the explicit `pyannote/segmentation-3.0` loader; do not immediately downgrade the project-wide `torch` or `torchaudio` pins.

For automation, you can set env vars **temporarily from Python** before launching the CLI subprocess (cross-platform pattern):

```python
import os
import subprocess
import sys

env = os.environ.copy()

if sys.platform == "darwin":
    brew = subprocess.run(["brew", "--prefix", "ffmpeg@6"], capture_output=True, text=True)
    ffmpeg6_lib = brew.stdout.strip() + "/lib" if brew.returncode == 0 else "/opt/homebrew/opt/ffmpeg@6/lib"
    env["DYLD_LIBRARY_PATH"] = f"{ffmpeg6_lib}:{env.get('DYLD_LIBRARY_PATH', '')}".rstrip(":")
elif sys.platform.startswith("linux"):
    # Set only if you manage a custom FFmpeg location.
    custom_lib = "/usr/local/lib"
    env["LD_LIBRARY_PATH"] = f"{custom_lib}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")

subprocess.run(
    [
        "double-ender-sync",
        "--master",
        "input/master.wav",
        "--track",
        "input/speaker-a.wav",
        "--out",
        "output",
        "--vad-strategy",
        "pyannote",
    ],
    check=True,
    env=env,
)
```

This keeps the override local to the launched process and avoids permanently mutating user shell configuration.

## Python API (import from another project)

In addition to CLI usage, you can run the same pipeline from Python.

```python
from pathlib import Path

from double_ender_sync import AlignmentOptions, AnchorSelectionConfig, run_alignment

options = AlignmentOptions(
    master=Path("input/master.wav"),
    tracks=[Path("input/speaker-a.wav"), Path("input/speaker-b.wav")],
    out=Path("output"),
    analysis_sample_rate=16000,
    local_adjust_enabled=False,
    normalize_output=False,
    anchor_selection=AnchorSelectionConfig(anchor_density_per_minute=1.0, max_anchor_count=120),
    drift_model="auto",
    allow_nonlinear_drift=False,
    max_breakpoints=1,
    min_anchors_for_piecewise=6,
    min_anchors_per_segment=3,
    verbose_report=False,
)

exit_code = run_alignment(options)
if exit_code != 0:
    raise RuntimeError(f"alignment failed with exit code {exit_code}")
```

`run_alignment(...)` returns the same exit code semantics as the CLI `main(...)`. Anchor-selection and drift-model options use the same shared configuration defaults as CLI and GUI runs; `drift_model="auto"` still resolves to the linear control path unless `allow_nonlinear_drift=True` is set explicitly.


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
2. speech-region detection (strategy-based: adaptive RMS by default fallback, with ML-ready hooks),
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
