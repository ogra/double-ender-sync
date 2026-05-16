import pytest
from pathlib import Path

pytest.importorskip("PySide6")

from PySide6.QtCore import QMimeData, Qt, QUrl
from PySide6.QtWidgets import QApplication, QMessageBox

from double_ender_sync.api import AlignmentOptions, build_cli_argv
from double_ender_sync.analysis.vad import DEFAULT_PYANNOTE_MODEL, MODERN_PYANNOTE_SEGMENTATION_MODEL
from double_ender_sync.config import DEFAULT_ANCHOR_SELECTION_CONFIG, DEFAULT_DRIFT_MODEL_CONFIG
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


def test_gui_displays_package_version_in_corner() -> None:
    _app()
    window = MainWindow(lang="en")

    assert window.version_label.text() == "v0.2.4"
    assert window.version_label.alignment() & Qt.AlignRight
    assert window.version_label.alignment() & Qt.AlignBottom


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


def test_gui_drift_gate_default_matches_shared_config() -> None:
    _app()
    window = MainWindow(lang="en")

    assert window.allow_nonlinear_drift_checkbox.isChecked() is DEFAULT_DRIFT_MODEL_CONFIG.allow_nonlinear_drift


def test_gui_auto_enables_nonlinear_gate_for_explicit_nonlinear_model(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    captured: dict[str, AlignmentOptions] = {}

    window.master_input.setText("master.wav")
    window.track_list.addItem("speaker.wav")
    window.output_input.setText(str(tmp_path))
    spline_index = window.drift_model_input.findData("spline")
    window.drift_model_input.setCurrentIndex(spline_index)
    monkeypatch.setattr(window, "_start_worker", lambda options: captured.setdefault("options", options))

    window.run()

    assert window.allow_nonlinear_drift_checkbox.isChecked() is True
    assert captured["options"].drift_model == "spline"
    assert captured["options"].allow_nonlinear_drift is True
    assert "--allow-nonlinear-drift" in build_cli_argv(captured["options"])


def test_gui_resets_explicit_nonlinear_model_when_gate_is_disabled(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    captured: dict[str, AlignmentOptions] = {}

    window.master_input.setText("master.wav")
    window.track_list.addItem("speaker.wav")
    window.output_input.setText(str(tmp_path))
    kalman_index = window.drift_model_input.findData("kalman")
    window.drift_model_input.setCurrentIndex(kalman_index)
    assert window.allow_nonlinear_drift_checkbox.isChecked() is True

    window.allow_nonlinear_drift_checkbox.setChecked(False)
    monkeypatch.setattr(window, "_start_worker", lambda options: captured.setdefault("options", options))

    window.run()

    assert window.drift_model_input.currentData() == "auto"
    assert captured["options"].drift_model == "auto"
    assert captured["options"].allow_nonlinear_drift is False
    assert "--allow-nonlinear-drift" not in build_cli_argv(captured["options"])


def test_gui_rejects_invalid_shared_options_before_starting_worker(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    errors: list[str] = []
    started: list[AlignmentOptions] = []

    window.master_input.setText("master.wav")
    window.track_list.addItem("speaker.wav")
    window.output_input.setText(str(tmp_path))

    def _raise_invalid_options(_options):
        raise ValueError("invalid shared options")

    monkeypatch.setattr("double_ender_sync.gui.build_cli_argv", _raise_invalid_options)
    monkeypatch.setattr(window, "_show_error", errors.append)
    monkeypatch.setattr(window, "_start_worker", started.append)

    window.run()

    assert errors == ["invalid shared options"]
    assert started == []


def test_gui_max_breakpoints_accepts_shared_non_negative_values(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    captured: dict[str, AlignmentOptions] = {}

    window.master_input.setText("master.wav")
    window.track_list.addItem("speaker.wav")
    window.output_input.setText(str(tmp_path))
    window.allow_nonlinear_drift_checkbox.setChecked(True)
    window.max_breakpoints_input.setValue(12)
    monkeypatch.setattr(window, "_start_worker", lambda options: captured.setdefault("options", options))

    window.run()

    assert captured["options"].max_breakpoints == 12
    assert "--max-breakpoints" in build_cli_argv(captured["options"])


def test_gui_anchor_minimums_accept_large_shared_values(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    captured: dict[str, AlignmentOptions] = {}

    window.master_input.setText("master.wav")
    window.track_list.addItem("speaker.wav")
    window.output_input.setText(str(tmp_path))
    window.allow_nonlinear_drift_checkbox.setChecked(True)
    window.min_anchors_for_piecewise_input.setValue(202)
    window.min_anchors_per_segment_input.setValue(101)
    monkeypatch.setattr(window, "_start_worker", lambda options: captured.setdefault("options", options))

    window.run()

    assert captured["options"].min_anchors_for_piecewise == 202
    assert captured["options"].min_anchors_per_segment == 101
    argv = build_cli_argv(captured["options"])
    assert "--min-anchors-for-piecewise" in argv
    assert "202" in argv
    assert "--min-anchors-per-segment" in argv
    assert "101" in argv


def test_gui_english_kalman_label_is_clear() -> None:
    _app()
    window = MainWindow(lang="en")

    kalman_index = window.drift_model_input.findData("kalman")

    assert window.drift_model_input.itemText(kalman_index) == "Kalman (research/experimental)"


def test_gui_max_anchor_gap_unit_is_in_label_not_suffix() -> None:
    _app()
    window = MainWindow(lang="en")

    assert window.t("gui.max_anchor_gap_seconds") == "Max trusted anchor gap (s)"
    assert window.max_anchor_gap_input.suffix() == ""
    assert window.max_anchor_gap_input.specialValueText() == "Disabled"


def test_gui_japanese_max_anchor_gap_unit_is_in_label_not_suffix() -> None:
    _app()
    window = MainWindow(lang="ja")

    assert window.t("gui.max_anchor_gap_seconds") == "信頼済みアンカー最大間隔（秒）"
    assert window.max_anchor_gap_input.suffix() == ""
    assert window.max_anchor_gap_input.specialValueText() == "無効"


def test_gui_forwards_shared_drift_model_options(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    captured: dict[str, AlignmentOptions] = {}

    window.master_input.setText("master.wav")
    window.track_list.addItem("speaker.wav")
    window.output_input.setText(str(tmp_path))
    piecewise_index = window.drift_model_input.findData("piecewise_linear")
    window.drift_model_input.setCurrentIndex(piecewise_index)
    assert window.drift_model_input.findData("spline") >= 0
    assert window.drift_model_input.findData("kalman") >= 0
    window.allow_nonlinear_drift_checkbox.setChecked(True)
    window.max_breakpoints_input.setValue(2)
    window.min_anchors_per_segment_input.setValue(4)
    window.max_anchor_gap_input.setValue(123.5)
    window.verbose_report_checkbox.setChecked(True)
    monkeypatch.setattr(window, "_start_worker", lambda options: captured.setdefault("options", options))

    window.run()

    assert captured["options"].drift_model == "piecewise_linear"
    assert captured["options"].allow_nonlinear_drift is True
    assert captured["options"].max_breakpoints == 2
    assert captured["options"].min_anchors_for_piecewise == 8
    assert captured["options"].min_anchors_per_segment == 4
    assert captured["options"].max_anchor_gap_seconds == 123.5
    assert captured["options"].verbose_report is True


def test_gui_stretch_method_combo_shows_all_methods() -> None:
    _app()
    window = MainWindow(lang="en")
    values = [window.stretch_method_combo.itemData(i) for i in range(window.stretch_method_combo.count())]
    assert set(values) == {"resample", "pitch_preserving", "rubberband", "soxr", "audiostretchy"}


def test_gui_stretch_method_combo_defaults_to_resample() -> None:
    _app()
    window = MainWindow(lang="en")
    assert window.stretch_method_combo.currentData() == "resample"


def test_gui_pitch_preserving_checkbox_unchecked_by_default() -> None:
    _app()
    window = MainWindow(lang="en")
    assert window.pitch_preserving_checkbox.isChecked() is False


def test_gui_selecting_rubberband_checks_pitch_preserving() -> None:
    _app()
    window = MainWindow(lang="en")
    rubberband_index = window.stretch_method_combo.findData("rubberband")
    window.stretch_method_combo.setCurrentIndex(rubberband_index)
    assert window.pitch_preserving_checkbox.isChecked() is True


def test_gui_selecting_pitch_preserving_checks_pitch_preserving() -> None:
    _app()
    window = MainWindow(lang="en")
    pp_index = window.stretch_method_combo.findData("pitch_preserving")
    window.stretch_method_combo.setCurrentIndex(pp_index)
    assert window.pitch_preserving_checkbox.isChecked() is True


def test_gui_selecting_audiostretchy_checks_pitch_preserving() -> None:
    _app()
    window = MainWindow(lang="en")
    audiostretchy_index = window.stretch_method_combo.findData("audiostretchy")
    window.stretch_method_combo.setCurrentIndex(audiostretchy_index)
    assert window.pitch_preserving_checkbox.isChecked() is True


def test_gui_selecting_soxr_unchecks_pitch_preserving() -> None:
    _app()
    window = MainWindow(lang="en")
    pp_index = window.stretch_method_combo.findData("pitch_preserving")
    window.stretch_method_combo.setCurrentIndex(pp_index)
    assert window.pitch_preserving_checkbox.isChecked() is True

    soxr_index = window.stretch_method_combo.findData("soxr")
    window.stretch_method_combo.setCurrentIndex(soxr_index)
    assert window.pitch_preserving_checkbox.isChecked() is False


def test_gui_checking_pitch_preserving_sets_combo_to_pitch_preserving() -> None:
    _app()
    window = MainWindow(lang="en")
    assert window.stretch_method_combo.currentData() == "resample"
    window.pitch_preserving_checkbox.setChecked(True)
    assert window.stretch_method_combo.currentData() == "pitch_preserving"


def test_gui_unchecking_pitch_preserving_resets_combo_to_resample() -> None:
    _app()
    window = MainWindow(lang="en")
    rubberband_index = window.stretch_method_combo.findData("rubberband")
    window.stretch_method_combo.setCurrentIndex(rubberband_index)
    assert window.pitch_preserving_checkbox.isChecked() is True

    window.pitch_preserving_checkbox.setChecked(False)
    assert window.stretch_method_combo.currentData() == "resample"
    assert window.pitch_preserving_checkbox.isChecked() is False


def test_gui_run_forwards_stretch_method_from_combo(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    captured: dict[str, AlignmentOptions] = {}

    window.master_input.setText("master.wav")
    window.track_list.addItem("speaker.wav")
    window.output_input.setText(str(tmp_path))
    soxr_index = window.stretch_method_combo.findData("soxr")
    window.stretch_method_combo.setCurrentIndex(soxr_index)
    monkeypatch.setattr(window, "_start_worker", lambda options: captured.setdefault("options", options))

    window.run()

    assert captured["options"].stretch_method == "soxr"
