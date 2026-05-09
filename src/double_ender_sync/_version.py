"""Shared package version helpers."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib

_PACKAGE_NAME = "double-ender-sync"
_UNKNOWN_VERSION = "0.0.0+unknown"


def _find_version_in_pyproject(start: Path) -> str | None:
    for parent in start.parents:
        pyproject = parent / "pyproject.toml"
        if not pyproject.is_file():
            continue
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        candidate = data.get("project", {}).get("version")
        if isinstance(candidate, str) and candidate.strip():
            return candidate
        continue
    return None


def _resolve_version() -> str:
    try:
        return version(_PACKAGE_NAME)
    except PackageNotFoundError:
        source_version = _find_version_in_pyproject(Path(__file__).resolve())
        if source_version is not None:
            return source_version
        return _UNKNOWN_VERSION


__version__ = _resolve_version()


def get_version() -> str:
    """Return the public package version string."""

    return __version__


def get_cli_version_text() -> str:
    """Return the user-facing CLI version line."""

    return f"version {get_version()}"


def get_gui_version_text() -> str:
    """Return the compact GUI version label."""

    return f"v{get_version()}"
