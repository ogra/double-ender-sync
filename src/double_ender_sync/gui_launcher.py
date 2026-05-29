from __future__ import annotations

import importlib.util
import sys

from double_ender_sync.i18n.catalog import TranslationCatalog
from double_ender_sync.i18n.resolver import resolve_language


def main() -> int:
    if importlib.util.find_spec("PySide6") is None:
        lang = resolve_language()
        catalog = TranslationCatalog(lang)
        print(
            catalog.t("cli.error.gui_dependencies_missing"),
            file=sys.stderr,
        )
        return 1

    from double_ender_sync.gui import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
