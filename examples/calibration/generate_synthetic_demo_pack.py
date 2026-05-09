#!/usr/bin/env python3
"""Backward-compatible wrapper for the synthetic calibration demo-pack CLI."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.exists():
    sys.path.insert(0, str(_REPO_SRC))

from double_ender_sync.calibration.demo_pack import main


if __name__ == "__main__":
    main()
