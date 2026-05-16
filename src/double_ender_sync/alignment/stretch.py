from __future__ import annotations

import importlib
import logging
from typing import Protocol

import numpy as np

LOGGER = logging.getLogger(__name__)


class Stretcher(Protocol):
    """Protocol for time-stretching audio samples by a given ratio.

    Implementations must be stateless with respect to audio data — each call to
    ``stretch_by_ratio`` should be independent.  The ``name`` attribute
    identifies the method for reporting and CLI selection.
    """

    name: str

    def stretch_by_ratio(
        self,
        samples: np.ndarray,
        stretch_ratio: float,
        *,
        sample_rate: int | None = None,
    ) -> np.ndarray:
        """Return ``samples`` stretched by ``stretch_ratio``.

        Args:
            samples: float32 mono audio array.
            stretch_ratio: > 1.0 expands the timeline (slower), < 1.0 compresses
                it (faster).  Must be positive.
            sample_rate: Source audio sample rate in Hz.  Required by backends
                that operate in the sample domain (e.g. ``pyrubberband``);
                ignored by frequency-domain backends.  Pass ``None`` (the
                default) only when the backend does not need it — callers
                that do not know which backend is in use should always supply
                the true sample rate.

        Returns:
            Float32 array of length ``max(1, round(len(samples) * stretch_ratio))``
            unless ``samples`` is empty, in which case an empty float32 array is
            returned.
        """
        ...


class NumpyInterpolationStretcher:
    """Stretcher using linear ``np.interp`` interpolation.

    This is the default/fallback implementation and requires no optional
    dependencies. It resamples by mapping evenly spaced target positions onto
    the original sample range and applying linear interpolation via
    ``np.interp``. Anti-aliasing is *not* applied, so audible aliasing is
    possible at stretch ratios that deviate by more than ~0.3% from 1.0.
    """

    name: str = "resample"

    def stretch_by_ratio(
        self,
        samples: np.ndarray,
        stretch_ratio: float,
        *,
        sample_rate: int | None = None,
    ) -> np.ndarray:
        if stretch_ratio <= 0:
            raise ValueError(f"stretch_ratio must be positive; got {stretch_ratio!r}")
        if samples.size == 0:
            return samples.astype(np.float32, copy=False)
        target_len = max(1, int(round(samples.shape[0] * stretch_ratio)))
        source_positions = np.linspace(0, samples.shape[0] - 1, num=samples.shape[0], dtype=np.float64)
        target_positions = np.linspace(0, samples.shape[0] - 1, num=target_len, dtype=np.float64)
        stretched = np.interp(target_positions, source_positions, samples.astype(np.float64, copy=False))
        return stretched.astype(np.float32, copy=False)


class LibrosaStretcher:
    """Pitch-preserving time-stretch backed by ``librosa.effects.time_stretch``.

    Requires the ``[stretch]`` optional extra (``pip install
    double-ender-sync[stretch]``).  A ``RuntimeError`` is raised at call time
    when ``librosa`` is not available so that the error surface remains
    consistent with the existing behaviour.
    """

    name: str = "pitch_preserving"

    def stretch_by_ratio(
        self,
        samples: np.ndarray,
        stretch_ratio: float,
        *,
        sample_rate: int | None = None,
    ) -> np.ndarray:
        if stretch_ratio <= 0:
            raise ValueError(f"stretch_ratio must be positive; got {stretch_ratio!r}")
        if samples.size == 0:
            return samples.astype(np.float32, copy=False)

        try:
            librosa = importlib.import_module("librosa")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                'pitch_preserving stretch requires librosa. '
                'Install with: pip install "double-ender-sync[stretch]"'
            ) from exc

        rate = 1.0 / stretch_ratio
        stretched = librosa.effects.time_stretch(samples.astype(np.float32, copy=False), rate=rate)

        target_len = max(1, int(round(samples.shape[0] * stretch_ratio)))
        if stretched.shape[0] > target_len:
            stretched = stretched[:target_len]
        elif stretched.shape[0] < target_len:
            pad = np.zeros(target_len - stretched.shape[0], dtype=stretched.dtype)
            stretched = np.concatenate([stretched, pad])

        LOGGER.debug("pitch_preserving stretch applied rate=%.8f target_len=%d", rate, target_len)
        return stretched.astype(np.float32, copy=False)


class RubberbandStretcher:
    """Pitch-preserving time-stretch backed by ``pyrubberband``.

    Uses the Rubber Band Library via the ``pyrubberband`` Python wrapper and the
    Rubber Band command-line executable, typically available on ``PATH`` as
    ``rubberband``.  Some distributions provide this via a package named
    ``rubberband-cli``.  Provides superior transient preservation and formant
    handling compared to phase-vocoder approaches.

    Requires:
    - ``pyrubberband>=0.3`` Python package (``pip install "double-ender-sync[stretch]"``)
    - Rubber Band executable on ``PATH`` (typically ``rubberband``; e.g.
      ``apt install rubberband-cli`` on some distributions)
    """

    name: str = "rubberband"

    def stretch_by_ratio(
        self,
        samples: np.ndarray,
        stretch_ratio: float,
        *,
        sample_rate: int | None = None,
    ) -> np.ndarray:
        if stretch_ratio <= 0:
            raise ValueError(f"stretch_ratio must be positive; got {stretch_ratio!r}")
        if samples.size == 0:
            return samples.astype(np.float32, copy=False)
        if sample_rate is None:
            raise ValueError(
                "RubberbandStretcher requires sample_rate to be specified explicitly. "
                "Pass the source audio sample rate (e.g. sample_rate=48000)."
            )
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(f"sample_rate must be a positive integer; got {sample_rate!r}")

        try:
            pyrubberband = importlib.import_module("pyrubberband")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                'rubberband stretch requires pyrubberband. '
                'Install with: pip install "double-ender-sync[stretch]"'
            ) from exc

        rate = 1.0 / stretch_ratio
        try:
            stretched = pyrubberband.time_stretch(
                samples.astype(np.float32, copy=False),
                sample_rate,
                rate,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"pyrubberband failed: {exc}. "
                "Ensure that the rubberband executable is on PATH "
                "(package: 'apt install rubberband-cli' on Debian/Ubuntu, "
                "'brew install rubberband' on macOS)."
            ) from exc

        target_len = max(1, int(round(samples.shape[0] * stretch_ratio)))
        if stretched.shape[0] > target_len:
            stretched = stretched[:target_len]
        elif stretched.shape[0] < target_len:
            pad = np.zeros(target_len - stretched.shape[0], dtype=stretched.dtype)
            stretched = np.concatenate([stretched, pad])

        LOGGER.debug("rubberband stretch applied rate=%.8f target_len=%d sr=%d", rate, target_len, sample_rate)
        return stretched.astype(np.float32, copy=False)

    def stretch_by_timemap(
        self,
        samples: np.ndarray,
        time_map: list[tuple[int, int]],
        *,
        sample_rate: int,
    ) -> np.ndarray:
        """Return ``samples`` stretched according to a non-uniform time map.

        ``time_map`` is a list of ``(src_sample, dst_sample)`` pairs in
        ascending order; the last entry's ``src_sample`` must equal
        ``len(samples)``.  This calls ``pyrubberband.timemap_stretch``
        which requires the Rubber Band executable on ``PATH``.
        """
        if samples.size == 0:
            return samples.astype(np.float32, copy=False)
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(f"sample_rate must be a positive integer; got {sample_rate!r}")
        if not time_map:
            raise ValueError("time_map must not be empty")
        if time_map[-1][0] != samples.shape[0]:
            raise ValueError(
                f"time_map last src_sample ({time_map[-1][0]}) must equal len(samples) ({samples.shape[0]})"
            )
        for i in range(1, len(time_map)):
            prev_src, prev_dst = time_map[i - 1]
            cur_src, cur_dst = time_map[i]
            if cur_src <= prev_src:
                raise ValueError(
                    f"time_map src values must be strictly increasing; "
                    f"got time_map[{i - 1}]={time_map[i - 1]!r}, time_map[{i}]={time_map[i]!r}"
                )
            if cur_dst < prev_dst:
                raise ValueError(
                    f"time_map dst values must be non-decreasing; "
                    f"got time_map[{i - 1}]={time_map[i - 1]!r}, time_map[{i}]={time_map[i]!r}"
                )

        try:
            pyrubberband = importlib.import_module("pyrubberband")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                'rubberband timemap stretch requires pyrubberband. '
                'Install with: pip install "double-ender-sync[stretch]"'
            ) from exc

        try:
            stretched = pyrubberband.timemap_stretch(
                samples.astype(np.float32, copy=False),
                sample_rate,
                time_map,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"pyrubberband failed: {exc}. "
                "Ensure that the rubberband executable is on PATH "
                "(package: 'apt install rubberband-cli' on Debian/Ubuntu, "
                "'brew install rubberband' on macOS)."
            ) from exc

        LOGGER.debug(
            "rubberband timemap_stretch applied time_map_len=%d sr=%d",
            len(time_map),
            sample_rate,
        )
        return stretched.astype(np.float32, copy=False)


class SoxrStretcher:
    """High-quality resampler backed by ``soxr`` (libsoxr) at VHQ preset.

    Uses ``soxr.resample`` with the VHQ (Very High Quality, 64-tap sinc)
    quality level to eliminate aliasing artifacts that are audible at even
    small stretch ratios (~0.3%).  Operates on the whole input array and
    does not require ``sample_rate`` (though the kwarg is accepted for API
    compatibility).

    Requires the ``[hq-resample]`` optional extra (``pip install
    double-ender-sync[hq-resample]``).  A ``RuntimeError`` is raised at call
    time when ``soxr`` is not available.
    """

    name: str = "soxr"

    def stretch_by_ratio(
        self,
        samples: np.ndarray,
        stretch_ratio: float,
        *,
        sample_rate: int | None = None,
    ) -> np.ndarray:
        if stretch_ratio <= 0:
            raise ValueError(f"stretch_ratio must be positive; got {stretch_ratio!r}")
        if samples.size == 0:
            return samples.astype(np.float32, copy=False)

        try:
            soxr = importlib.import_module("soxr")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                'soxr stretch requires the soxr package. '
                'Install with: pip install "double-ender-sync[hq-resample]"'
            ) from exc

        resampled = soxr.resample(
            samples.astype(np.float32, copy=False),
            1.0,
            stretch_ratio,
            quality="VHQ",
        )

        target_len = max(1, int(round(samples.shape[0] * stretch_ratio)))
        resampled = resampled.astype(np.float32, copy=False)
        if resampled.shape[0] > target_len:
            resampled = resampled[:target_len]
        elif resampled.shape[0] < target_len:
            pad = np.zeros(target_len - resampled.shape[0], dtype=np.float32)
            resampled = np.concatenate([resampled, pad])

        LOGGER.debug("soxr stretch applied stretch_ratio=%.8f target_len=%d", stretch_ratio, target_len)
        return resampled


class AudiostretchyStretcher:
    """Time-stretch backed by the ``audiostretchy-f32`` package.

    Uses ``audiostretchy.AudioStretch`` and applies ``AudioStretch.stretch`` to
    a whole-array mono buffer. This does not require an external executable
    dependency.
    """

    name: str = "audiostretchy"

    @staticmethod
    def _import_audiostretchy_f32():
        module = importlib.import_module("audiostretchy")

        if not hasattr(module, "AudioStretch"):
            raise RuntimeError("audiostretchy.AudioStretch is not available")

        return module

    def stretch_by_ratio(
        self,
        samples: np.ndarray,
        stretch_ratio: float,
        *,
        sample_rate: int | None = None,
    ) -> np.ndarray:
        if stretch_ratio <= 0:
            raise ValueError(f"stretch_ratio must be positive; got {stretch_ratio!r}")
        if samples.size == 0:
            return samples.astype(np.float32, copy=False)
        if sample_rate is None:
            raise ValueError(
                "AudiostretchyStretcher requires sample_rate to be specified explicitly. "
                "Pass the source audio sample rate (e.g. sample_rate=48000)."
            )
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(f"sample_rate must be a positive integer; got {sample_rate!r}")

        try:
            audiostretchy = self._import_audiostretchy_f32()
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                'audiostretchy stretch requires audiostretchy-f32 package. '
                'Install with: pip install "double-ender-sync[audiostretchy]"'
            ) from exc

        processor = audiostretchy.AudioStretch()
        processor.samples = np.ascontiguousarray(samples.astype(np.float32, copy=False)).reshape(1, -1)
        processor.samplerate = int(sample_rate)
        processor.num_channels = 1
        processor.stretch(ratio=float(stretch_ratio))

        stretched_2d = np.asarray(processor.samples, dtype=np.float32)
        if stretched_2d.ndim != 2 or stretched_2d.shape[0] != 1:
            raise RuntimeError(f"audiostretchy-f32 AudioStretch produced unexpected shape: {stretched_2d.shape!r}")
        stretched = np.ascontiguousarray(stretched_2d[0], dtype=np.float32)

        target_len = max(1, int(round(samples.shape[0] * stretch_ratio)))
        stretched = stretched.astype(np.float32, copy=False)
        if stretched.shape[0] > target_len:
            stretched = stretched[:target_len]
        elif stretched.shape[0] < target_len:
            pad = np.zeros(target_len - stretched.shape[0], dtype=np.float32)
            stretched = np.concatenate([stretched, pad])

        LOGGER.debug("audiostretchy-f32 stretch applied stretch_ratio=%.8f target_len=%d", stretch_ratio, target_len)
        return stretched


_STRETCHER_REGISTRY: dict[str, type[Stretcher]] = {
    "resample": NumpyInterpolationStretcher,
    "pitch_preserving": LibrosaStretcher,
    "rubberband": RubberbandStretcher,
    "soxr": SoxrStretcher,
    "audiostretchy": AudiostretchyStretcher,
}

VALID_STRETCH_METHODS: frozenset[str] = frozenset(_STRETCHER_REGISTRY)


def make_stretcher(method: str) -> Stretcher:
    """Return a ``Stretcher`` instance for the given method name.

    Raises:
        ValueError: if ``method`` is not a known stretch method name.
    """
    cls = _STRETCHER_REGISTRY.get(method)
    if cls is None:
        raise ValueError(
            f"stretch_method must be one of: {', '.join(sorted(_STRETCHER_REGISTRY))}; got {method!r}"
        )
    return cls()
