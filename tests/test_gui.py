import pytest
from pathlib import Path

pytest.importorskip("PySide6")

from PySide6.QtCore import QMimeData, QUrl
from PySide6.QtWidgets import QApplication, QMessageBox

from double_ender_sync.api import AlignmentOptions
from double_ender_sync.analysis.vad import DEFAULT_PYANNOTE_MODEL, MODERN_PYANNOTE_SEGMENTATION_MODEL
from double_ender_sync.config import DEFAULT_ANCHOR_SELECTION_CONFIG
from double_ender_sync.gui import AlignmentWorker, MainWindow, extract_audio_paths


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


def test_gui_vad_selector_includes_supported_backends() -> None:
    _app()
    window = MainWindow(lang="en")
    values = [window.vad_strategy_input.itemData(i) for i in range(window.vad_strategy_input.count())]
    assert values == ["adaptive_rms", "rms", "silero", "webrtc", "pyannote"]


def test_alignment_worker_catches_exceptions_and_emits_finished(monkeypatch, tmp_path) -> None:
    _app()
    options = _options(tmp_path)
    worker = AlignmentWorker(options)
    emitted: dict[str, object] = {}

    def _boom(*args, **kwargs):
        raise RuntimeError("backend failed")

    monkeypatch.setattr("double_ender_sync.gui.run_alignment", _boom)
    worker.log_line.connect(lambda message: emitted.setdefault("log", message))
    worker.finished.connect(lambda code, _opts: emitted.setdefault("code", code))

    worker.run()

    assert emitted["code"] == 1
    assert "backend failed" in str(emitted["log"])


def test_gui_pyannote_model_input_defaults_to_community_model() -> None:
    _app()
    window = MainWindow(lang="en")
    assert window.pyannote_model_input.text() == DEFAULT_PYANNOTE_MODEL


def test_gui_pyannote_model_input_is_only_enabled_for_pyannote() -> None:
    _app()
    window = MainWindow(lang="en")

    assert window.vad_strategy_input.currentData() == "adaptive_rms"
    assert not window.pyannote_model_input.isEnabled()

    pyannote_index = window.vad_strategy_input.findData("pyannote")
    window.vad_strategy_input.setCurrentIndex(pyannote_index)

    assert window.pyannote_model_input.isEnabled()


def test_gui_ignores_pyannote_model_when_non_pyannote_strategy(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    captured: dict[str, AlignmentOptions] = {}

    window.master_input.setText("master.wav")
    window.track_list.addItem("speaker.wav")
    window.output_input.setText(str(tmp_path))
    window.pyannote_model_input.setText(MODERN_PYANNOTE_SEGMENTATION_MODEL)
    monkeypatch.setattr(window, "_start_worker", lambda options: captured.setdefault("options", options))

    window.run()

    assert captured["options"].vad_strategy == "adaptive_rms"
    assert captured["options"].pyannote_model == DEFAULT_PYANNOTE_MODEL


def test_gui_forwards_custom_pyannote_model_for_pyannote_strategy(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    captured: dict[str, AlignmentOptions] = {}

    window.master_input.setText("master.wav")
    window.track_list.addItem("speaker.wav")
    window.output_input.setText(str(tmp_path))
    pyannote_index = window.vad_strategy_input.findData("pyannote")
    window.vad_strategy_input.setCurrentIndex(pyannote_index)
    window.pyannote_model_input.setText(MODERN_PYANNOTE_SEGMENTATION_MODEL)
    monkeypatch.setattr(window, "_start_worker", lambda options: captured.setdefault("options", options))

    window.run()

    assert captured["options"].vad_strategy == "pyannote"
    assert captured["options"].pyannote_model == MODERN_PYANNOTE_SEGMENTATION_MODEL


def test_gui_uses_shared_anchor_selection_defaults(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    captured: dict[str, AlignmentOptions] = {}

    window.master_input.setText("master.wav")
    window.track_list.addItem("speaker.wav")
    window.output_input.setText(str(tmp_path))
    monkeypatch.setattr(window, "_start_worker", lambda options: captured.setdefault("options", options))

    window.run()

    assert captured["options"].anchor_selection == DEFAULT_ANCHOR_SELECTION_CONFIG
