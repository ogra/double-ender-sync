import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
_DEMO_PACK_SCRIPT = _REPO_ROOT / "examples" / "calibration" / "generate_synthetic_demo_pack.py"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


def _pythonpath_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    src_path = str(_SRC_DIR)
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    return env


def test_demo_pack_generator_is_registered_as_console_script() -> None:
    pyproject = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))

    assert (
        pyproject["project"]["scripts"]["double-ender-sync-generate-demo-pack"]
        == "double_ender_sync.calibration.demo_pack:main"
    )


def test_legacy_demo_pack_script_wrapper_displays_help() -> None:
    result = subprocess.run(
        [sys.executable, str(_DEMO_PACK_SCRIPT), "--help"],
        check=True,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert "Output directory for generated WAV demo pack" in result.stdout


def test_demo_pack_manifest_uses_relative_paths_and_default_anchorable_audio(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "demo-pack"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "double_ender_sync.calibration.demo_pack",
            "--out",
            str(pack_dir),
            "--sample-rate",
            "4000",
            "--duration",
            "8",
        ],
        check=True,
        cwd=_REPO_ROOT,
        env=_pythonpath_env(),
    )

    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    first_case = manifest["cases"][0]
    assert first_case["master"] == "known-offset/master.wav"
    assert first_case["track"] == "known-offset/speaker-a.wav"
    assert not Path(first_case["master"]).is_absolute()
    assert not Path(first_case["track"]).is_absolute()

    report_dir = tmp_path / "known-offset-output"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "double_ender_sync.cli",
            "--master",
            str(pack_dir / first_case["master"]),
            "--track",
            str(pack_dir / first_case["track"]),
            "--out",
            str(report_dir),
            "--allow-nonlinear-drift",
            "--verbose-report",
            "--stretch-ratio-auto-continue",
        ],
        check=True,
        cwd=_REPO_ROOT,
        env=_pythonpath_env(),
    )

    report = json.loads((report_dir / "sync-report.json").read_text(encoding="utf-8"))
    track = report["tracks"][0]
    assert track["anchor_candidate_summary"]["count"] > 0
    assert track["initial_offset"] is not None
    assert track["anchor_count"] > 0
