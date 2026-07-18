import pytest
from pathlib import Path

pytest.importorskip("PySide6")

from PySide6.QtCore import QMimeData, Qt, QUrl
from PySide6.QtWidgets import QApplication, QMessageBox

from double_ender_sync.api import AlignmentOptions, build_cli_argv
from double_ender_sync.analysis.vad import DEFAULT_PYANNOTE_MODEL, MODERN_PYANNOTE_SEGMENTATION_MODEL
from double_ender_sync.config import (
    DEFAULT_ANCHOR_MATCHING_CONFIG,
    DEFAULT_ANCHOR_SELECTION_CONFIG,
    DEFAULT_DRIFT_MODEL_CONFIG,
    DEFAULT_INITIAL_OFFSET_SAFETY_CONFIG,
)
from double_ender_sync.gui import AlignmentWorker, MainWindow, extract_audio_paths


def _capture_alignment_options_from_window(window, tmp_path, monkeypatch):
    captured: dict[str, AlignmentOptions] = {}
    window.master_input.setText("master.wav")
    window.track_list.addItem("speaker.wav")
    window.output_input.setText(str(tmp_path))
    monkeypatch.setattr(window, "_start_worker", lambda options: captured.setdefault("options", options))
    window.run()
    return captured["options"]


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

    assert window.version_label.text() == "v0.2.7"
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


def test_gui_advanced_tabs_present_in_expected_order() -> None:
    _app()
    window = MainWindow(lang="en")
    labels = [window.advanced_tabs.tabText(i) for i in range(window.advanced_tabs.count())]
    assert labels == [
        "Basic / Output",
        "VAD",
        "Drift model",
        "Anchor selection",
        "Anchor matching",
        "Initial offset safety",
    ]


def test_gui_advanced_settings_open_in_modal_dialog() -> None:
    _app()
    window = MainWindow(lang="en")
    # The advanced tabs are not hosted directly on the main window anymore;
    # they live inside a reusable modal dialog opened via the button.
    assert window.advanced_button.text() == window.catalog.t("gui.advanced_settings_open")
    window.open_advanced_settings()
    dialog = window._advanced_dialog
    assert dialog is not None
    assert dialog.isModal()
    assert window.advanced_tabs.parent() is not None
    # Re-opening reuses the same dialog instance (state is preserved).
    window.open_advanced_settings()
    assert window._advanced_dialog is dialog
    dialog.close()


def test_gui_pyannote_selection_emphasizes_advanced_button() -> None:
    _app()
    window = MainWindow(lang="en")
    base_text = window.catalog.t("gui.advanced_settings_open")
    attention_text = window.catalog.t("gui.advanced_settings_attention")
    assert window.advanced_button.text() == base_text
    pyannote_index = window.vad_strategy_input.findData("pyannote")
    window.vad_strategy_input.setCurrentIndex(pyannote_index)
    # The button is emphasized and the VAD tab is preselected so the pyannote
    # model field is visible the moment the dialog opens.
    assert window.advanced_button.text() == attention_text
    assert window.advanced_tabs.currentIndex() == 1
    adaptive_index = window.vad_strategy_input.findData("adaptive_rms")
    window.vad_strategy_input.setCurrentIndex(adaptive_index)
    assert window.advanced_button.text() == base_text


def test_gui_anchor_selection_defaults_match_shared_config(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    options = _capture_alignment_options_from_window(window, tmp_path, monkeypatch)
    assert options.anchor_selection == DEFAULT_ANCHOR_SELECTION_CONFIG


def test_gui_anchor_matching_defaults_match_shared_config(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    options = _capture_alignment_options_from_window(window, tmp_path, monkeypatch)
    assert options.anchor_matching == DEFAULT_ANCHOR_MATCHING_CONFIG


def test_gui_nullable_anchor_selection_special_values_become_none(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    window.max_anchor_count_input.setValue(0)
    window.stratified_bin_count_input.setValue(0)
    window.anchors_per_bin_input.setValue(0)
    # min_snr_db and spectral_flatness default to their disabled sentinels.
    options = _capture_alignment_options_from_window(window, tmp_path, monkeypatch)
    selection = options.anchor_selection
    assert selection.max_anchor_count is None
    assert selection.stratified_bin_count is None
    assert selection.anchors_per_bin is None
    assert selection.min_snr_db is None
    assert selection.spectral_flatness_threshold is None


def test_gui_forwards_custom_anchor_selection_values(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    window.max_anchor_count_input.setValue(40)
    window.stratified_bin_count_input.setValue(6)
    window.anchors_per_bin_input.setValue(3)
    window.min_snr_db_input.setValue(12.5)
    window.spectral_flatness_input.setValue(0.25)
    options = _capture_alignment_options_from_window(window, tmp_path, monkeypatch)
    selection = options.anchor_selection
    assert selection.max_anchor_count == 40
    assert selection.stratified_bin_count == 6
    assert selection.anchors_per_bin == 3
    assert selection.min_snr_db == 12.5
    assert selection.spectral_flatness_threshold == 0.25
    argv = build_cli_argv(options)
    assert "--max-anchor-count" in argv and "40" in argv
    assert "--stratified-bin-count" in argv
    assert "--min-snr-db" in argv


def test_gui_anchor_density_bounds_clamp(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    # Density cannot exceed the max density bound.
    window.max_anchor_density_input.setValue(1.5)
    window.anchor_density_input.setValue(5.0)
    assert window.anchor_density_input.value() <= window.max_anchor_density_input.value()


def test_gui_anchor_durations_keep_ordering(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    window.min_anchor_duration_input.setValue(6.0)
    assert window.min_anchor_duration_input.value() <= window.base_anchor_duration_input.value()
    assert window.base_anchor_duration_input.value() <= window.max_anchor_duration_input.value()


def test_gui_max_anchor_count_clamped_to_min(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    window.max_anchor_count_input.setValue(10)
    window.min_anchor_count_input.setValue(30)
    # Capped max must not fall below the configured minimum.
    assert window.max_anchor_count_input.value() >= window.min_anchor_count_input.value()


def test_gui_ncc_low_high_pairs_stay_ordered() -> None:
    _app()
    window = MainWindow(lang="en")
    window.ncc_margin_high_input.setValue(0.1)
    window.ncc_margin_low_input.setValue(0.9)
    assert window.ncc_margin_low_input.value() < window.ncc_margin_high_input.value()
    window.ncc_prominence_high_input.setValue(0.1)
    window.ncc_prominence_low_input.setValue(0.9)
    assert window.ncc_prominence_low_input.value() < window.ncc_prominence_high_input.value()
    window.ncc_good_width_input.setValue(0.9)
    assert window.ncc_good_width_input.value() < window.ncc_bad_width_input.value()


def test_gui_gcc_phat_gate_disables_dependents() -> None:
    _app()
    window = MainWindow(lang="en")
    window.gcc_phat_enabled_checkbox.setChecked(False)
    assert not window.gcc_phat_only_when_ambiguous_checkbox.isEnabled()
    assert not window.gcc_phat_tolerance_input.isEnabled()
    window.gcc_phat_enabled_checkbox.setChecked(True)
    assert window.gcc_phat_only_when_ambiguous_checkbox.isEnabled()
    assert window.gcc_phat_tolerance_input.isEnabled()


def test_gui_forwards_custom_anchor_matching_values(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    window.ncc_min_score_input.setValue(0.6)
    window.min_confidence_for_fit_input.setValue(0.2)
    window.gcc_phat_enabled_checkbox.setChecked(False)
    options = _capture_alignment_options_from_window(window, tmp_path, monkeypatch)
    matching = options.anchor_matching
    assert matching.ncc_min_score == 0.6
    assert matching.min_confidence_for_fit == 0.2
    assert matching.gcc_phat_enabled is False
    argv = build_cli_argv(options)
    assert "--ncc-min-score" in argv
    assert "--no-gcc-phat" in argv


# --- Initial offset safety GUI tests ---


def test_gui_initial_offset_safety_tab_present() -> None:
    _app()
    window = MainWindow(lang="en")
    last_index = window.advanced_tabs.count() - 1
    assert window.advanced_tabs.tabText(last_index) == window.catalog.t("gui.tab.initial_offset_safety")


def test_gui_initial_offset_safety_defaults_match_shared_config(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    options = _capture_alignment_options_from_window(window, tmp_path, monkeypatch)
    assert options.initial_offset_safety == DEFAULT_INITIAL_OFFSET_SAFETY_CONFIG


def test_gui_coarse_fallback_enabled_gates_dependent_inputs() -> None:
    _app()
    window = MainWindow(lang="en")

    assert window.coarse_fallback_enabled_checkbox.isChecked() is True
    assert window.coarse_fallback_sample_rate_input.isEnabled() is True
    assert window.coarse_fallback_min_peak_margin_input.isEnabled() is True
    assert window.coarse_fallback_max_duration_input.isEnabled() is True
    assert window.coarse_fallback_max_memory_input.isEnabled() is True
    assert window.coarse_fallback_min_confidence_input.isEnabled() is True
    assert window.coarse_fallback_confidence_margin_input.isEnabled() is True

    window.coarse_fallback_enabled_checkbox.setChecked(False)

    assert window.coarse_fallback_sample_rate_input.isEnabled() is False
    assert window.coarse_fallback_min_peak_margin_input.isEnabled() is False
    assert window.coarse_fallback_max_duration_input.isEnabled() is False
    assert window.coarse_fallback_max_memory_input.isEnabled() is False
    assert window.coarse_fallback_min_confidence_input.isEnabled() is False
    assert window.coarse_fallback_confidence_margin_input.isEnabled() is False

    window.coarse_fallback_enabled_checkbox.setChecked(True)

    assert window.coarse_fallback_sample_rate_input.isEnabled() is True
    assert window.coarse_fallback_min_peak_margin_input.isEnabled() is True
    assert window.coarse_fallback_max_duration_input.isEnabled() is True
    assert window.coarse_fallback_max_memory_input.isEnabled() is True
    assert window.coarse_fallback_min_confidence_input.isEnabled() is True
    assert window.coarse_fallback_confidence_margin_input.isEnabled() is True


def test_gui_master_vad_filter_enabled_gates_dependent_inputs() -> None:
    _app()
    window = MainWindow(lang="en")

    assert window.master_vad_filter_enabled_checkbox.isChecked() is True
    assert window.master_vad_min_overlap_ratio_input.isEnabled() is True
    assert window.master_vad_padding_input.isEnabled() is True
    assert window.master_vad_uncertain_policy_combo.isEnabled() is True

    window.master_vad_filter_enabled_checkbox.setChecked(False)

    assert window.master_vad_min_overlap_ratio_input.isEnabled() is False
    assert window.master_vad_padding_input.isEnabled() is False
    assert window.master_vad_uncertain_policy_combo.isEnabled() is False

    window.master_vad_filter_enabled_checkbox.setChecked(True)

    assert window.master_vad_min_overlap_ratio_input.isEnabled() is True
    assert window.master_vad_padding_input.isEnabled() is True
    assert window.master_vad_uncertain_policy_combo.isEnabled() is True


def test_gui_initial_offset_safety_config_roundtrip(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")

    window.initial_offset_min_confidence_input.setValue(0.55)
    window.high_confidence_threshold_input.setValue(0.80)
    window.medium_confidence_threshold_input.setValue(0.55)
    window.low_confidence_threshold_input.setValue(0.30)
    window.coarse_fallback_enabled_checkbox.setChecked(True)
    window.coarse_fallback_sample_rate_input.setValue(16000)
    window.coarse_fallback_min_peak_margin_input.setValue(0.15)
    window.coarse_fallback_max_duration_input.setValue(0.0)
    window.coarse_fallback_max_memory_input.setValue(2048.0)
    window.coarse_fallback_min_confidence_input.setValue(0.60)
    window.coarse_fallback_confidence_margin_input.setValue(0.20)
    window.max_drift_search_radius_input.setValue(45.0)
    window.high_confidence_search_radius_input.setValue(8.0)
    window.medium_confidence_search_radius_input.setValue(15.0)
    window.low_confidence_search_radius_input.setValue(25.0)
    window.master_vad_filter_enabled_checkbox.setChecked(True)
    window.master_vad_min_overlap_ratio_input.setValue(0.30)
    window.master_vad_padding_input.setValue(0.50)
    window.master_vad_uncertain_policy_combo.setCurrentIndex(
        window.master_vad_uncertain_policy_combo.findData("skip")
    )

    options = _capture_alignment_options_from_window(window, tmp_path, monkeypatch)
    config = options.initial_offset_safety

    assert config.initial_offset_min_confidence == 0.55
    assert config.high_confidence_threshold == 0.80
    assert config.medium_confidence_threshold == 0.55
    assert config.low_confidence_threshold == 0.30
    assert config.coarse_fallback_enabled is True
    assert config.coarse_fallback_sample_rate == 16000
    assert config.coarse_fallback_min_peak_margin == 0.15
    assert config.coarse_fallback_max_duration_seconds is None
    assert config.coarse_fallback_max_memory_mb == 2048.0
    assert config.coarse_fallback_min_confidence == 0.60
    assert config.coarse_fallback_confidence_margin == 0.20
    assert config.max_drift_search_radius_seconds == 45.0
    assert config.high_confidence_search_radius_seconds == 8.0
    assert config.medium_confidence_search_radius_seconds == 15.0
    assert config.low_confidence_search_radius_seconds == 25.0
    assert config.master_vad_filter_enabled is True
    assert config.master_vad_min_overlap_ratio == 0.30
    assert config.master_vad_padding_seconds == 0.50
    assert config.master_vad_uncertain_policy == "skip"

    argv = build_cli_argv(options)
    assert "--initial-offset-min-confidence" in argv
    assert "--high-confidence-threshold" in argv
    assert "--coarse-fallback-sample-rate" in argv
    assert "--master-vad-filter-enabled" in argv
    assert "--master-vad-uncertain-policy" in argv
    assert "skip" in argv


def test_gui_coarse_fallback_max_duration_none_mapping() -> None:
    _app()
    window = MainWindow(lang="en")

    window.coarse_fallback_max_duration_input.setValue(0.0)
    assert window.coarse_fallback_max_duration_input.specialValueText() == "Unlimited"

    config = window._build_initial_offset_safety_config()
    assert config.coarse_fallback_max_duration_seconds is None

    window.coarse_fallback_max_duration_input.setValue(3600.0)
    config = window._build_initial_offset_safety_config()
    assert config.coarse_fallback_max_duration_seconds == 3600.0


def test_gui_confidence_threshold_ordering_enforced() -> None:
    _app()
    window = MainWindow(lang="en")

    window.low_confidence_threshold_input.setValue(0.30)
    window.medium_confidence_threshold_input.setValue(0.60)
    window.high_confidence_threshold_input.setValue(0.85)

    assert window.low_confidence_threshold_input.value() < window.medium_confidence_threshold_input.value()
    assert window.medium_confidence_threshold_input.value() < window.high_confidence_threshold_input.value()
    assert window.initial_offset_min_confidence_input.minimum() == window.low_confidence_threshold_input.value()


def test_gui_initial_offset_safety_vad_policy_combo_data() -> None:
    _app()
    window = MainWindow(lang="en")

    values = [
        window.master_vad_uncertain_policy_combo.itemData(i)
        for i in range(window.master_vad_uncertain_policy_combo.count())
    ]
    assert values == ["warn", "skip", "reject"]
    assert window.master_vad_uncertain_policy_combo.currentData() == "warn"


def test_gui_initial_offset_safety_verbose_report_roundtrip(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    window.verbose_report_checkbox.setChecked(True)

    options = _capture_alignment_options_from_window(window, tmp_path, monkeypatch)
    assert options.verbose_report is True
    argv = build_cli_argv(options)
    assert "--verbose-report" in argv


def test_gui_initial_offset_safety_disabled_coarse_fallback_roundtrip(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    window.coarse_fallback_enabled_checkbox.setChecked(False)

    options = _capture_alignment_options_from_window(window, tmp_path, monkeypatch)
    assert options.initial_offset_safety.coarse_fallback_enabled is False
    argv = build_cli_argv(options)
    assert "--no-coarse-fallback" in argv


def test_gui_initial_offset_safety_disabled_master_vad_roundtrip(monkeypatch, tmp_path) -> None:
    _app()
    window = MainWindow(lang="en")
    window.master_vad_filter_enabled_checkbox.setChecked(False)

    options = _capture_alignment_options_from_window(window, tmp_path, monkeypatch)
    assert options.initial_offset_safety.master_vad_filter_enabled is False
    argv = build_cli_argv(options)
    assert "--no-master-vad-filter" in argv
