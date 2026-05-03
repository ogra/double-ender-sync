from __future__ import annotations

import importlib.util
import sys


def main() -> int:
    if importlib.util.find_spec("PySide6") is None:
        print(
            'error: GUI dependencies are not installed. Install with: pip install "double-ender-sync[gui]"',
            file=sys.stderr,
        )
        return 1

    from double_ender_sync.gui import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
