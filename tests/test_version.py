from pathlib import Path
from importlib.metadata import PackageNotFoundError

import pytest

from double_ender_sync import _version


def test_find_version_in_pyproject_reads_project_version(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    module_path = project_root / "src" / "double_ender_sync" / "_version.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# mock", encoding="utf-8")
    (project_root / "pyproject.toml").write_text(
        '[project]\nname = "double-ender-sync"\nversion = "9.8.7"\n',
        encoding="utf-8",
    )

    assert _version._find_version_in_pyproject(module_path) == "9.8.7"


def test_find_version_in_pyproject_returns_none_without_version(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    module_path = project_root / "src" / "double_ender_sync" / "_version.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# mock", encoding="utf-8")
    (project_root / "pyproject.toml").write_text(
        '[project]\nname = "double-ender-sync"\n',
        encoding="utf-8",
    )

    assert _version._find_version_in_pyproject(module_path) is None


def test_resolve_version_uses_pyproject_when_metadata_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    module_path = project_root / "src" / "double_ender_sync" / "_version.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# mock", encoding="utf-8")
    (project_root / "pyproject.toml").write_text(
        '[project]\nname = "double-ender-sync"\nversion = "1.2.3"\n',
        encoding="utf-8",
    )

    def _raise_not_found(_: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr(_version, "version", _raise_not_found)
    monkeypatch.setattr(_version, "__file__", str(module_path))

    assert _version._resolve_version() == "1.2.3"
