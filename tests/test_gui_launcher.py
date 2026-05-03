from __future__ import annotations

import sys
import types

from double_ender_sync import gui_launcher


def test_main_returns_error_when_pyside6_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(gui_launcher.importlib.util, "find_spec", lambda _: None)

    exit_code = gui_launcher.main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert 'pip install "double-ender-sync[gui]"' in captured.err


def test_main_delegates_to_gui_module_when_pyside6_is_available(monkeypatch):
    monkeypatch.setattr(gui_launcher.importlib.util, "find_spec", lambda _: object())

    stub_module = types.ModuleType("double_ender_sync.gui")

    def stub_main() -> int:
        return 7

    stub_module.main = stub_main
    monkeypatch.setitem(sys.modules, "double_ender_sync.gui", stub_module)

    assert gui_launcher.main() == 7
