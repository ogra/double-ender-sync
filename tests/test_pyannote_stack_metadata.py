import tomllib
from pathlib import Path


PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _optional_dependencies(extra_name: str) -> list[str]:
    pyproject = tomllib.loads(PYPROJECT.read_text())
    return pyproject["project"]["optional-dependencies"][extra_name]


def test_pyannote_extra_targets_python314_latest_stack() -> None:
    dependencies = _optional_dependencies("vad-pyannote")

    assert "pyannote.audio>=4.0.4,<5" in dependencies
    assert "torch==2.11.0" in dependencies
    assert "torchaudio==2.11.0" in dependencies
    assert "torchcodec>=0.11.1,<0.12" in dependencies


def test_all_extra_keeps_pyannote_stack_in_sync() -> None:
    pyannote_dependencies = set(_optional_dependencies("vad-pyannote"))
    all_dependencies = set(_optional_dependencies("all"))

    assert pyannote_dependencies <= all_dependencies
