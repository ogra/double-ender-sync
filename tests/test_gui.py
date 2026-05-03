import pytest
from pathlib import Path

pytest.importorskip("PySide6")

from PySide6.QtCore import QMimeData, QUrl
from PySide6.QtWidgets import QApplication, QMessageBox

from double_ender_sync.api import AlignmentOptions
from double_ender_sync.gui import MainWindow, extract_audio_paths


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _options(out_dir: Path) -> AlignmentOptions:
    return AlignmentOptions(
        master=Path("master.wav"),
        tracks=[Path("speaker.wav")],
        out=out_dir,
        analysis_sample_rate=16000,
    )


def test_success_dialog_adds_open_output_button(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    captured: dict[str, object] = {}

    class FakeDialog:
        Information = QMessageBox.Information
        Ok = QMessageBox.Ok
        ActionRole = QMessageBox.ActionRole

        def __init__(self, parent):
            captured["parent"] = parent
            self._open_button = None
            self._clicked = None

        def setIcon(self, icon):
            captured["icon"] = icon

        def setWindowTitle(self, title):
            captured["title"] = title

        def setText(self, text):
            captured["text"] = text

        def addButton(self, button_or_text, role=None):
            if role is None:
                button = object()
                captured["ok_button"] = button
                return button
            button = object()
            captured["open_text"] = button_or_text
            captured["open_role"] = role
            self._open_button = button
            return button

        def setDefaultButton(self, button):
            captured["default_button"] = button

        def exec(self):
            self._clicked = captured.get("ok_button")

        def clickedButton(self):
            return self._clicked

    monkeypatch.setattr("double_ender_sync.gui.QMessageBox", FakeDialog)

    window._on_finished(0, _options(tmp_path))

    assert captured["title"] == "double-ender-sync"
    assert captured["text"] == "Alignment completed successfully."
    assert captured["open_text"] == "Open output folder"
    assert captured["open_role"] == QMessageBox.ActionRole
    assert captured["default_button"] == captured["ok_button"]


def test_open_output_directory_uses_desktop_services(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    opened: dict[str, str] = {}

    def _open_url(url):
        opened["path"] = url.toLocalFile()
        return True

    monkeypatch.setattr("double_ender_sync.gui.QDesktopServices.openUrl", _open_url)
    monkeypatch.setattr(window, "_show_error", lambda message: (_ for _ in ()).throw(AssertionError(message)))

    window._open_output_directory(tmp_path)

    assert opened["path"] == str(tmp_path.resolve())


def test_open_output_directory_shows_error_when_open_fails(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    errors: list[str] = []

    monkeypatch.setattr("double_ender_sync.gui.QDesktopServices.openUrl", lambda _url: False)
    monkeypatch.setattr(window, "_show_error", errors.append)

    window._open_output_directory(tmp_path)

    assert len(errors) == 1
    assert str(tmp_path) in errors[0]


def test_extract_audio_paths_accepts_wav_and_aiff(tmp_path) -> None:
    wav = tmp_path / "a.wav"
    aiff = tmp_path / "b.aiff"
    aif = tmp_path / "c.aif"
    txt = tmp_path / "d.txt"
    for path in (wav, aiff, aif, txt):
        path.write_text("x")

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(wav)), QUrl.fromLocalFile(str(aiff)), QUrl.fromLocalFile(str(aif)), QUrl.fromLocalFile(str(txt))])

    paths = extract_audio_paths(mime)

    assert str(wav) in paths
    assert str(aiff) in paths
    assert str(aif) in paths
    assert str(txt) not in paths
