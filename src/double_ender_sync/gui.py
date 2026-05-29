from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QMimeData, Qt, QObject, Signal, QThread, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from double_ender_sync._version import get_gui_version_text
from double_ender_sync.api import AlignmentOptions, build_cli_argv, run_alignment
from double_ender_sync.analysis.vad import DEFAULT_PYANNOTE_MODEL
from double_ender_sync.cli import EXIT_STRETCH_CONFIRMATION_REQUIRED
from double_ender_sync.config import (
    DEFAULT_ANCHOR_MATCHING_CONFIG,
    DEFAULT_ANCHOR_SELECTION_CONFIG,
    DEFAULT_DRIFT_MODEL_CONFIG,
    AnchorMatchingConfig,
    AnchorSelectionConfig,
)
from double_ender_sync.i18n import TranslationCatalog, resolve_language
from double_ender_sync.i18n.resolver import extract_explicit_lang

_PITCH_PRESERVING_STRETCH_METHODS: frozenset[str] = frozenset({"pitch_preserving", "rubberband", "audiostretchy"})

# Sentinel spin-box values used to represent "None" for nullable options.
# Each sits at the widget's minimum so Qt renders the special-value text.
_MAX_ANCHOR_COUNT_UNLIMITED = 0
_STRATIFIED_BIN_COUNT_AUTO = 0
_ANCHORS_PER_BIN_AUTO = 0
_MIN_SNR_DB_DISABLED = -200.0
_SPECTRAL_FLATNESS_DISABLED = 0.0
# Largest signed integer accepted by QSpinBox; used where the shared config
# applies minimum-only validation (mirrors the existing drift spin boxes).
_SHARED_NON_NEGATIVE_SPINBOX_MAX = 2_147_483_647


class DropListWidget(QListWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.ExtendedSelection)

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if _mime_has_files(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        if _mime_has_files(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        for file_path in extract_audio_paths(event.mimeData()):
            if not _contains_item(self, file_path):
                self.addItem(file_path)
        event.acceptProposedAction()


def _mime_has_files(mime_data: QMimeData) -> bool:
    return mime_data.hasUrls()


SUPPORTED_AUDIO_SUFFIXES = {".wav", ".aiff", ".aif"}
SUPPORTED_AUDIO_FILTER = "Audio Files (" + " ".join(f"*{suffix}" for suffix in sorted(SUPPORTED_AUDIO_SUFFIXES)) + ")"


def extract_audio_paths(mime_data: QMimeData) -> list[str]:
    paths: list[str] = []
    for url in mime_data.urls():
        path = Path(url.toLocalFile())
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES:
            paths.append(str(path))
    return paths


def _contains_item(widget: QListWidget, value: str) -> bool:
    for index in range(widget.count()):
        item = widget.item(index)
        if item is not None and item.text() == value:
            return True
    return False


class MainWindow(QMainWindow):
    def __init__(self, lang: str | None = None) -> None:
        super().__init__()
        self.catalog = TranslationCatalog(resolve_language(explicit_lang=lang))
        self.setWindowTitle(self.t("gui.window_title"))
        self.resize(900, 620)

        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        form_layout = QFormLayout()
        form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignTop)
        form_layout.setHorizontalSpacing(16)
        form_layout.setVerticalSpacing(10)

        self.master_input = QLineEdit()
        self._configure_path_input(self.master_input)
        master_browse = QPushButton(self.t("gui.browse"))
        master_browse.clicked.connect(self.select_master)
        master_row = QHBoxLayout()
        master_row.setContentsMargins(0, 0, 0, 0)
        master_row.addWidget(self.master_input)
        master_row.addWidget(master_browse)
        master_row.setStretch(0, 1)
        form_layout.addRow(QLabel(self.t("gui.master_wav")), _wrap_layout(master_row))

        self.track_list = DropListWidget()
        self._configure_path_input(self.track_list)
        track_list_policy = self.track_list.sizePolicy()
        track_list_policy.setVerticalPolicy(QSizePolicy.Expanding)
        self.track_list.setSizePolicy(track_list_policy)
        self.track_list.setMinimumHeight(160)
        tracks_buttons = QHBoxLayout()
        add_tracks = QPushButton(self.t("gui.add_tracks"))
        add_tracks.clicked.connect(self.select_tracks)
        remove_selected = QPushButton(self.t("gui.remove_selected"))
        remove_selected.clicked.connect(self.remove_selected_tracks)
        tracks_buttons.addWidget(add_tracks)
        tracks_buttons.addWidget(remove_selected)

        tracks_col = QVBoxLayout()
        tracks_col.addWidget(self.track_list)
        tracks_col.addLayout(tracks_buttons)
        form_layout.addRow(QLabel(self.t("gui.speaker_tracks")), _wrap_layout(tracks_col))

        self.output_input = QLineEdit()
        self._configure_path_input(self.output_input)
        output_browse = QPushButton(self.t("gui.browse"))
        output_browse.clicked.connect(self.select_output)
        output_row = QHBoxLayout()
        output_row.setContentsMargins(0, 0, 0, 0)
        output_row.addWidget(self.output_input)
        output_row.addWidget(output_browse)
        output_row.setStretch(0, 1)
        form_layout.addRow(QLabel(self.t("gui.output_directory")), _wrap_layout(output_row))

        self.sample_rate_input = QSpinBox()
        self.sample_rate_input.setRange(8000, 192000)
        self.sample_rate_input.setValue(16000)
        self.sample_rate_input.setAlignment(Qt.AlignRight)
        self.sample_rate_input.setMinimumWidth(140)

        self.normalize_checkbox = QCheckBox(self.t("gui.normalize_output"))
        self.local_adjust_checkbox = QCheckBox(self.t("gui.enable_local_adjustment"))
        self.pitch_preserving_checkbox = QCheckBox(self.t("gui.use_pitch_preserving"))
        self.stretch_method_combo = QComboBox()
        self.stretch_method_combo.addItem(self.t("gui.stretch_method.resample"), "resample")
        self.stretch_method_combo.addItem(self.t("gui.stretch_method.pitch_preserving"), "pitch_preserving")
        self.stretch_method_combo.addItem(self.t("gui.stretch_method.rubberband"), "rubberband")
        self.stretch_method_combo.addItem(self.t("gui.stretch_method.soxr"), "soxr")
        self.stretch_method_combo.addItem(self.t("gui.stretch_method.audiostretchy"), "audiostretchy")
        self.stretch_threshold_input = QDoubleSpinBox()
        self.stretch_threshold_input.setRange(0.0001, 0.1)
        self.stretch_threshold_input.setDecimals(4)
        self.stretch_threshold_input.setSingleStep(0.0005)
        self.stretch_threshold_input.setValue(0.0030)
        self.stretch_threshold_input.setMinimumWidth(140)
        self.vad_strategy_input = QComboBox()
        self.vad_strategy_input.addItem(self.t("gui.vad_strategy.adaptive_rms"), "adaptive_rms")
        self.vad_strategy_input.addItem(self.t("gui.vad_strategy.rms"), "rms")
        self.vad_strategy_input.addItem(self.t("gui.vad_strategy.silero"), "silero")
        self.vad_strategy_input.addItem(self.t("gui.vad_strategy.webrtc"), "webrtc")
        self.vad_strategy_input.addItem(self.t("gui.vad_strategy.pyannote"), "pyannote")
        self.drift_model_input = QComboBox()
        self.drift_model_input.addItem(self.t("gui.drift_model.auto"), "auto")
        self.drift_model_input.addItem(self.t("gui.drift_model.linear"), "linear")
        self.drift_model_input.addItem(self.t("gui.drift_model.piecewise_linear"), "piecewise_linear")
        self.drift_model_input.addItem(self.t("gui.drift_model.spline"), "spline")
        self.drift_model_input.addItem(self.t("gui.drift_model.kalman"), "kalman")
        self.drift_model_input.setCurrentIndex(self.drift_model_input.findData(DEFAULT_DRIFT_MODEL_CONFIG.drift_model))
        self.allow_nonlinear_drift_checkbox = QCheckBox(self.t("gui.allow_nonlinear_drift"))
        self.allow_nonlinear_drift_checkbox.setChecked(DEFAULT_DRIFT_MODEL_CONFIG.allow_nonlinear_drift)
        shared_non_negative_spinbox_max = 2_147_483_647
        # Match the shared CLI/API validation where Qt spinboxes allow it:
        # these values have minimum-only validation, so avoid small GUI-only
        # caps and use the largest signed integer supported by QSpinBox.
        self.max_breakpoints_input = QSpinBox()
        self.max_breakpoints_input.setRange(0, shared_non_negative_spinbox_max)
        self.max_breakpoints_input.setValue(DEFAULT_DRIFT_MODEL_CONFIG.max_breakpoints)
        self.max_breakpoints_input.setMinimumWidth(140)
        self.min_anchors_for_piecewise_input = QSpinBox()
        self.min_anchors_for_piecewise_input.setRange(2, shared_non_negative_spinbox_max)
        self.min_anchors_for_piecewise_input.setValue(DEFAULT_DRIFT_MODEL_CONFIG.min_anchors_for_piecewise)
        self.min_anchors_for_piecewise_input.setMinimumWidth(140)
        self.min_anchors_per_segment_input = QSpinBox()
        self.min_anchors_per_segment_input.setRange(2, shared_non_negative_spinbox_max // 2)
        self.min_anchors_per_segment_input.setValue(DEFAULT_DRIFT_MODEL_CONFIG.min_anchors_per_segment)
        self.min_anchors_per_segment_input.setMinimumWidth(140)
        self.max_anchor_gap_input = QDoubleSpinBox()
        self.max_anchor_gap_input.setRange(0.0, float(shared_non_negative_spinbox_max))
        self.max_anchor_gap_input.setDecimals(3)
        self.max_anchor_gap_input.setSingleStep(30.0)
        self.max_anchor_gap_input.setSpecialValueText(self.t("gui.max_anchor_gap_disabled"))
        self.max_anchor_gap_input.setValue(DEFAULT_DRIFT_MODEL_CONFIG.max_anchor_gap_seconds or 0.0)
        self.max_anchor_gap_input.setMinimumWidth(140)
        self.verbose_report_checkbox = QCheckBox(self.t("gui.verbose_report"))

        self.pyannote_model_input = QLineEdit(DEFAULT_PYANNOTE_MODEL)
        self.pyannote_model_input.setPlaceholderText(DEFAULT_PYANNOTE_MODEL)

        self._build_anchor_selection_inputs()
        self._build_anchor_matching_inputs()

        self.vad_strategy_input.currentIndexChanged.connect(self._sync_pyannote_model_input_enabled)
        self.drift_model_input.currentIndexChanged.connect(self._sync_drift_model_gate)
        self.allow_nonlinear_drift_checkbox.toggled.connect(self._sync_drift_model_gate)
        self.min_anchors_per_segment_input.valueChanged.connect(self._sync_piecewise_anchor_minimum)
        self.stretch_method_combo.currentIndexChanged.connect(self._sync_checkbox_from_combo)
        self.pitch_preserving_checkbox.toggled.connect(self._sync_combo_from_checkbox)
        self._sync_pyannote_model_input_enabled()
        self._sync_drift_model_gate()
        self._sync_piecewise_anchor_minimum()
        self._sync_checkbox_from_combo()
        self._sync_anchor_density_bounds()
        self._sync_anchor_durations()
        self._sync_max_anchor_count_minimum()
        self._sync_ncc_margin_range()
        self._sync_ncc_prominence_range()
        self._sync_ncc_width_range()
        self._sync_gcc_phat_gate()

        layout.addLayout(form_layout)

        self.advanced_tabs = QTabWidget()

        basic_tab, basic_form = self._new_form_tab()
        self._add_form_rows(
            basic_form,
            [
                ("gui.analysis_sample_rate", self.sample_rate_input),
                (None, self.normalize_checkbox),
                (None, self.local_adjust_checkbox),
                (None, self.pitch_preserving_checkbox),
                ("gui.stretch_method", self.stretch_method_combo),
                ("gui.stretch_threshold", self.stretch_threshold_input),
                (None, self.verbose_report_checkbox),
            ],
        )
        self.advanced_tabs.addTab(basic_tab, self.t("gui.tab.basic"))

        vad_tab, vad_form = self._new_form_tab()
        self._add_form_rows(
            vad_form,
            [
                ("gui.vad_strategy", self.vad_strategy_input),
                ("gui.pyannote_model", self.pyannote_model_input),
            ],
        )
        self.advanced_tabs.addTab(vad_tab, self.t("gui.tab.vad"))

        drift_tab, drift_form = self._new_form_tab()
        self._add_form_rows(
            drift_form,
            [
                ("gui.drift_model", self.drift_model_input),
                (None, self.allow_nonlinear_drift_checkbox),
                ("gui.max_breakpoints", self.max_breakpoints_input),
                ("gui.min_anchors_for_piecewise", self.min_anchors_for_piecewise_input),
                ("gui.min_anchors_per_segment", self.min_anchors_per_segment_input),
                ("gui.max_anchor_gap_seconds", self.max_anchor_gap_input),
            ],
        )
        self.advanced_tabs.addTab(drift_tab, self.t("gui.tab.drift"))

        anchor_selection_tab, anchor_selection_form = self._new_form_tab()
        self._add_form_rows(
            anchor_selection_form,
            [
                ("gui.anchor_selection.anchor_density_per_minute", self.anchor_density_input),
                ("gui.anchor_selection.max_anchor_density_per_minute", self.max_anchor_density_input),
                ("gui.anchor_selection.min_anchor_count", self.min_anchor_count_input),
                ("gui.anchor_selection.max_anchor_count", self.max_anchor_count_input),
                ("gui.anchor_selection.min_anchor_duration_seconds", self.min_anchor_duration_input),
                ("gui.anchor_selection.base_anchor_duration_seconds", self.base_anchor_duration_input),
                ("gui.anchor_selection.max_anchor_duration_seconds", self.max_anchor_duration_input),
                ("gui.anchor_selection.stratified_bin_count", self.stratified_bin_count_input),
                ("gui.anchor_selection.anchors_per_bin", self.anchors_per_bin_input),
                ("gui.anchor_selection.min_snr_db", self.min_snr_db_input),
                ("gui.anchor_selection.spectral_flatness_threshold", self.spectral_flatness_input),
            ],
        )
        self.advanced_tabs.addTab(anchor_selection_tab, self.t("gui.tab.anchor_selection"))

        anchor_matching_tab, anchor_matching_form = self._new_form_tab()
        self._add_form_rows(
            anchor_matching_form,
            [
                ("gui.anchor_matching.ncc_min_score", self.ncc_min_score_input),
                ("gui.anchor_matching.ncc_min_margin", self.ncc_min_margin_input),
                ("gui.anchor_matching.ncc_min_prominence", self.ncc_min_prominence_input),
                ("gui.anchor_matching.min_confidence_for_fit", self.min_confidence_for_fit_input),
                ("gui.anchor_matching.nms_exclusion_seconds", self.nms_exclusion_input),
                ("gui.anchor_matching.ncc_good_width_seconds", self.ncc_good_width_input),
                ("gui.anchor_matching.ncc_bad_width_seconds", self.ncc_bad_width_input),
                ("gui.anchor_matching.ncc_margin_low", self.ncc_margin_low_input),
                ("gui.anchor_matching.ncc_margin_high", self.ncc_margin_high_input),
                ("gui.anchor_matching.ncc_prominence_low", self.ncc_prominence_low_input),
                ("gui.anchor_matching.ncc_prominence_high", self.ncc_prominence_high_input),
                (None, self.gcc_phat_enabled_checkbox),
                (None, self.gcc_phat_only_when_ambiguous_checkbox),
                ("gui.anchor_matching.gcc_phat_agreement_tolerance_seconds", self.gcc_phat_tolerance_input),
            ],
        )
        self.advanced_tabs.addTab(anchor_matching_tab, self.t("gui.tab.anchor_matching"))

        advanced_button = QPushButton(self.t("gui.advanced_settings_open"))
        self.advanced_button = advanced_button
        advanced_button.setMinimumHeight(34)
        advanced_button.clicked.connect(self.open_advanced_settings)
        self._advanced_button_base_text = advanced_button.text()
        self._advanced_dialog: QDialog | None = None
        layout.addWidget(advanced_button)

        run_button = QPushButton(self.t("gui.run_alignment"))
        run_button.setMinimumHeight(34)
        run_button.clicked.connect(self.run)
        self.run_button = run_button
        layout.addWidget(run_button)

        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label = QLabel("0%")
        self.eta_label = QLabel(self.t("gui.progress_eta_idle"))
        progress_row.addWidget(self.progress_bar, 1)
        progress_row.addWidget(self.progress_label)
        progress_row.addWidget(self.eta_label)
        layout.addLayout(progress_row)

        self.current_task_label = QLabel(self.t("gui.current_task_idle"))
        layout.addWidget(self.current_task_label)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText(self.t("gui.log_placeholder"))
        layout.addWidget(self.log_view)

        footer_row = QHBoxLayout()
        footer_row.addStretch(1)
        self.version_label = QLabel(get_gui_version_text())
        self.version_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        footer_row.addWidget(self.version_label)
        layout.addLayout(footer_row)

        self.setCentralWidget(root)
        self._worker_thread: QThread | None = None
        self._worker: AlignmentWorker | None = None

    def append_log(self, message: str) -> None:
        self.log_view.append(message)

    def t(self, key: str, **kwargs: object) -> str:
        return self.catalog.t(key, **kwargs)

    def _new_form_tab(self) -> tuple[QWidget, QFormLayout]:
        tab = QWidget()
        form = QFormLayout(tab)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)
        return tab, form

    def _add_form_rows(self, form: QFormLayout, rows: list[tuple[str | None, QWidget]]) -> None:
        for label_key, widget in rows:
            if label_key is None:
                form.addRow(widget)
            else:
                form.addRow(QLabel(self.t(label_key)), widget)

    def _build_anchor_selection_inputs(self) -> None:
        defaults = DEFAULT_ANCHOR_SELECTION_CONFIG

        self.anchor_density_input = QDoubleSpinBox()
        self.anchor_density_input.setRange(0.01, float(_SHARED_NON_NEGATIVE_SPINBOX_MAX))
        self.anchor_density_input.setDecimals(2)
        self.anchor_density_input.setSingleStep(0.1)
        self.anchor_density_input.setValue(defaults.anchor_density_per_minute)
        self.anchor_density_input.setMinimumWidth(140)

        self.max_anchor_density_input = QDoubleSpinBox()
        self.max_anchor_density_input.setRange(0.01, float(_SHARED_NON_NEGATIVE_SPINBOX_MAX))
        self.max_anchor_density_input.setDecimals(2)
        self.max_anchor_density_input.setSingleStep(0.1)
        self.max_anchor_density_input.setValue(defaults.max_anchor_density_per_minute)
        self.max_anchor_density_input.setMinimumWidth(140)

        self.min_anchor_count_input = QSpinBox()
        self.min_anchor_count_input.setRange(0, _SHARED_NON_NEGATIVE_SPINBOX_MAX)
        self.min_anchor_count_input.setValue(defaults.min_anchor_count)
        self.min_anchor_count_input.setMinimumWidth(140)

        self.max_anchor_count_input = QSpinBox()
        self.max_anchor_count_input.setRange(_MAX_ANCHOR_COUNT_UNLIMITED, _SHARED_NON_NEGATIVE_SPINBOX_MAX)
        self.max_anchor_count_input.setSpecialValueText(self.t("gui.anchor_selection.max_anchor_count_unlimited"))
        self.max_anchor_count_input.setValue(
            _MAX_ANCHOR_COUNT_UNLIMITED if defaults.max_anchor_count is None else defaults.max_anchor_count
        )
        self.max_anchor_count_input.setMinimumWidth(140)

        self.min_anchor_duration_input = QDoubleSpinBox()
        self.min_anchor_duration_input.setRange(0.001, float(_SHARED_NON_NEGATIVE_SPINBOX_MAX))
        self.min_anchor_duration_input.setDecimals(3)
        self.min_anchor_duration_input.setSingleStep(0.5)
        self.min_anchor_duration_input.setValue(defaults.min_anchor_duration_seconds)
        self.min_anchor_duration_input.setMinimumWidth(140)

        self.base_anchor_duration_input = QDoubleSpinBox()
        self.base_anchor_duration_input.setRange(0.001, float(_SHARED_NON_NEGATIVE_SPINBOX_MAX))
        self.base_anchor_duration_input.setDecimals(3)
        self.base_anchor_duration_input.setSingleStep(0.5)
        self.base_anchor_duration_input.setValue(defaults.base_anchor_duration_seconds)
        self.base_anchor_duration_input.setMinimumWidth(140)

        self.max_anchor_duration_input = QDoubleSpinBox()
        self.max_anchor_duration_input.setRange(0.001, float(_SHARED_NON_NEGATIVE_SPINBOX_MAX))
        self.max_anchor_duration_input.setDecimals(3)
        self.max_anchor_duration_input.setSingleStep(0.5)
        self.max_anchor_duration_input.setValue(defaults.max_anchor_duration_seconds)
        self.max_anchor_duration_input.setMinimumWidth(140)

        self.stratified_bin_count_input = QSpinBox()
        self.stratified_bin_count_input.setRange(_STRATIFIED_BIN_COUNT_AUTO, _SHARED_NON_NEGATIVE_SPINBOX_MAX)
        self.stratified_bin_count_input.setSpecialValueText(self.t("gui.anchor_selection.auto"))
        self.stratified_bin_count_input.setValue(
            _STRATIFIED_BIN_COUNT_AUTO if defaults.stratified_bin_count is None else defaults.stratified_bin_count
        )
        self.stratified_bin_count_input.setMinimumWidth(140)

        self.anchors_per_bin_input = QSpinBox()
        self.anchors_per_bin_input.setRange(_ANCHORS_PER_BIN_AUTO, _SHARED_NON_NEGATIVE_SPINBOX_MAX)
        self.anchors_per_bin_input.setSpecialValueText(self.t("gui.anchor_selection.auto"))
        self.anchors_per_bin_input.setValue(
            _ANCHORS_PER_BIN_AUTO if defaults.anchors_per_bin is None else defaults.anchors_per_bin
        )
        self.anchors_per_bin_input.setMinimumWidth(140)

        self.min_snr_db_input = QDoubleSpinBox()
        self.min_snr_db_input.setRange(_MIN_SNR_DB_DISABLED, 200.0)
        self.min_snr_db_input.setDecimals(1)
        self.min_snr_db_input.setSingleStep(1.0)
        self.min_snr_db_input.setSpecialValueText(self.t("gui.anchor_selection.disabled"))
        self.min_snr_db_input.setValue(_MIN_SNR_DB_DISABLED if defaults.min_snr_db is None else defaults.min_snr_db)
        self.min_snr_db_input.setMinimumWidth(140)

        self.spectral_flatness_input = QDoubleSpinBox()
        self.spectral_flatness_input.setRange(_SPECTRAL_FLATNESS_DISABLED, 1.0)
        self.spectral_flatness_input.setDecimals(3)
        self.spectral_flatness_input.setSingleStep(0.01)
        self.spectral_flatness_input.setSpecialValueText(self.t("gui.anchor_selection.disabled"))
        self.spectral_flatness_input.setValue(
            _SPECTRAL_FLATNESS_DISABLED
            if defaults.spectral_flatness_threshold is None
            else defaults.spectral_flatness_threshold
        )
        self.spectral_flatness_input.setMinimumWidth(140)

        self.anchor_density_input.valueChanged.connect(self._sync_anchor_density_bounds)
        self.max_anchor_density_input.valueChanged.connect(self._sync_anchor_density_bounds)
        self.min_anchor_duration_input.valueChanged.connect(self._sync_anchor_durations)
        self.base_anchor_duration_input.valueChanged.connect(self._sync_anchor_durations)
        self.max_anchor_duration_input.valueChanged.connect(self._sync_anchor_durations)
        self.min_anchor_count_input.valueChanged.connect(self._sync_max_anchor_count_minimum)

    def _build_anchor_matching_inputs(self) -> None:
        defaults = DEFAULT_ANCHOR_MATCHING_CONFIG

        self.ncc_min_score_input = QDoubleSpinBox()
        self.ncc_min_score_input.setRange(-1.0, 0.999)
        self.ncc_min_score_input.setDecimals(3)
        self.ncc_min_score_input.setSingleStep(0.01)
        self.ncc_min_score_input.setValue(defaults.ncc_min_score)
        self.ncc_min_score_input.setMinimumWidth(140)

        self.ncc_min_margin_input = QDoubleSpinBox()
        self.ncc_min_margin_input.setRange(0.0, 1.0)
        self.ncc_min_margin_input.setDecimals(3)
        self.ncc_min_margin_input.setSingleStep(0.01)
        self.ncc_min_margin_input.setValue(defaults.ncc_min_margin)
        self.ncc_min_margin_input.setMinimumWidth(140)

        self.ncc_min_prominence_input = QDoubleSpinBox()
        self.ncc_min_prominence_input.setRange(0.0, 1.0)
        self.ncc_min_prominence_input.setDecimals(3)
        self.ncc_min_prominence_input.setSingleStep(0.01)
        self.ncc_min_prominence_input.setValue(defaults.ncc_min_prominence)
        self.ncc_min_prominence_input.setMinimumWidth(140)

        self.min_confidence_for_fit_input = QDoubleSpinBox()
        self.min_confidence_for_fit_input.setRange(0.0, 1.0)
        self.min_confidence_for_fit_input.setDecimals(3)
        self.min_confidence_for_fit_input.setSingleStep(0.01)
        self.min_confidence_for_fit_input.setValue(defaults.min_confidence_for_fit)
        self.min_confidence_for_fit_input.setMinimumWidth(140)

        self.nms_exclusion_input = QDoubleSpinBox()
        self.nms_exclusion_input.setRange(0.001, 0.5)
        self.nms_exclusion_input.setDecimals(3)
        self.nms_exclusion_input.setSingleStep(0.005)
        self.nms_exclusion_input.setValue(defaults.nms_exclusion_seconds)
        self.nms_exclusion_input.setMinimumWidth(140)

        self.ncc_good_width_input = QDoubleSpinBox()
        self.ncc_good_width_input.setRange(0.0001, 1.0)
        self.ncc_good_width_input.setDecimals(4)
        self.ncc_good_width_input.setSingleStep(0.001)
        self.ncc_good_width_input.setValue(defaults.ncc_good_width_seconds)
        self.ncc_good_width_input.setMinimumWidth(140)

        self.ncc_bad_width_input = QDoubleSpinBox()
        self.ncc_bad_width_input.setRange(0.0001, 1.0)
        self.ncc_bad_width_input.setDecimals(4)
        self.ncc_bad_width_input.setSingleStep(0.001)
        self.ncc_bad_width_input.setValue(defaults.ncc_bad_width_seconds)
        self.ncc_bad_width_input.setMinimumWidth(140)

        self.ncc_margin_low_input = QDoubleSpinBox()
        self.ncc_margin_low_input.setRange(0.0, 1.0)
        self.ncc_margin_low_input.setDecimals(3)
        self.ncc_margin_low_input.setSingleStep(0.01)
        self.ncc_margin_low_input.setValue(defaults.ncc_margin_low)
        self.ncc_margin_low_input.setMinimumWidth(140)

        self.ncc_margin_high_input = QDoubleSpinBox()
        self.ncc_margin_high_input.setRange(0.0, 1.0)
        self.ncc_margin_high_input.setDecimals(3)
        self.ncc_margin_high_input.setSingleStep(0.01)
        self.ncc_margin_high_input.setValue(defaults.ncc_margin_high)
        self.ncc_margin_high_input.setMinimumWidth(140)

        self.ncc_prominence_low_input = QDoubleSpinBox()
        self.ncc_prominence_low_input.setRange(0.0, 1.0)
        self.ncc_prominence_low_input.setDecimals(3)
        self.ncc_prominence_low_input.setSingleStep(0.01)
        self.ncc_prominence_low_input.setValue(defaults.ncc_prominence_low)
        self.ncc_prominence_low_input.setMinimumWidth(140)

        self.ncc_prominence_high_input = QDoubleSpinBox()
        self.ncc_prominence_high_input.setRange(0.0, 1.0)
        self.ncc_prominence_high_input.setDecimals(3)
        self.ncc_prominence_high_input.setSingleStep(0.01)
        self.ncc_prominence_high_input.setValue(defaults.ncc_prominence_high)
        self.ncc_prominence_high_input.setMinimumWidth(140)

        self.gcc_phat_enabled_checkbox = QCheckBox(self.t("gui.anchor_matching.gcc_phat_enabled"))
        self.gcc_phat_enabled_checkbox.setChecked(defaults.gcc_phat_enabled)
        self.gcc_phat_only_when_ambiguous_checkbox = QCheckBox(
            self.t("gui.anchor_matching.gcc_phat_only_when_ambiguous")
        )
        self.gcc_phat_only_when_ambiguous_checkbox.setChecked(defaults.gcc_phat_only_when_ambiguous)

        self.gcc_phat_tolerance_input = QDoubleSpinBox()
        self.gcc_phat_tolerance_input.setRange(0.001, 1.0)
        self.gcc_phat_tolerance_input.setDecimals(3)
        self.gcc_phat_tolerance_input.setSingleStep(0.005)
        self.gcc_phat_tolerance_input.setValue(defaults.gcc_phat_agreement_tolerance_seconds)
        self.gcc_phat_tolerance_input.setMinimumWidth(140)

        self.ncc_margin_low_input.valueChanged.connect(self._sync_ncc_margin_range)
        self.ncc_margin_high_input.valueChanged.connect(self._sync_ncc_margin_range)
        self.ncc_prominence_low_input.valueChanged.connect(self._sync_ncc_prominence_range)
        self.ncc_prominence_high_input.valueChanged.connect(self._sync_ncc_prominence_range)
        self.ncc_good_width_input.valueChanged.connect(self._sync_ncc_width_range)
        self.ncc_bad_width_input.valueChanged.connect(self._sync_ncc_width_range)
        self.gcc_phat_enabled_checkbox.toggled.connect(self._sync_gcc_phat_gate)

    def _configure_path_input(self, widget: QWidget) -> None:
        """Make path fields wide and resizable with the window width.

        QLineEdit natively supports horizontal scrolling for long paths.
        """
        widget.setMinimumWidth(480)
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def _selected_vad_strategy(self) -> str:
        return str(self.vad_strategy_input.currentData())

    def _selected_pyannote_model(self, vad_strategy: str) -> str:
        if vad_strategy != "pyannote":
            return DEFAULT_PYANNOTE_MODEL
        return self.pyannote_model_input.text().strip() or DEFAULT_PYANNOTE_MODEL

    def open_advanced_settings(self) -> None:
        dialog = self._ensure_advanced_dialog()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _ensure_advanced_dialog(self) -> QDialog:
        if self._advanced_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle(self.t("gui.advanced_settings"))
            dialog.setModal(True)
            dialog_layout = QVBoxLayout(dialog)
            dialog_layout.setContentsMargins(12, 12, 12, 12)
            dialog_layout.addWidget(self.advanced_tabs)
            button_box = QDialogButtonBox(QDialogButtonBox.Close)
            button_box.rejected.connect(dialog.reject)
            button_box.accepted.connect(dialog.accept)
            dialog_layout.addWidget(button_box)
            self._advanced_dialog = dialog
        return self._advanced_dialog

    def _sync_pyannote_model_input_enabled(self, *_args: object) -> None:
        pyannote_selected = self._selected_vad_strategy() == "pyannote"
        if pyannote_selected and hasattr(self, "advanced_button"):
            # The pyannote model field lives on the VAD tab inside the
            # advanced-settings dialog. Surface that the relevant settings are
            # behind the button and preselect the VAD tab so it is visible the
            # moment the dialog opens.
            self.advanced_tabs.setCurrentWidget(self.advanced_tabs.widget(1))
            self.advanced_button.setText(self.t("gui.advanced_settings_attention"))
        elif hasattr(self, "advanced_button"):
            self.advanced_button.setText(self._advanced_button_base_text)
        self.pyannote_model_input.setEnabled(pyannote_selected)

    def _sync_drift_model_gate(self, *_args: object) -> None:
        selected_model = str(self.drift_model_input.currentData())
        nonlinear_selected = selected_model in {"piecewise_linear", "spline", "kalman"}
        nonlinear_allowed = self.allow_nonlinear_drift_checkbox.isChecked()
        if nonlinear_selected and not nonlinear_allowed:
            sender = self.sender()
            if sender is self.allow_nonlinear_drift_checkbox:
                auto_index = self.drift_model_input.findData("auto")
                if auto_index >= 0:
                    self.drift_model_input.setCurrentIndex(auto_index)
            else:
                self.allow_nonlinear_drift_checkbox.setChecked(True)

    def _sync_piecewise_anchor_minimum(self, *_args: object) -> None:
        required_minimum = int(self.min_anchors_per_segment_input.value()) * 2
        self.min_anchors_for_piecewise_input.setMinimum(required_minimum)
        if self.min_anchors_for_piecewise_input.value() < required_minimum:
            self.min_anchors_for_piecewise_input.setValue(required_minimum)

    def _sync_anchor_density_bounds(self, *_args: object) -> None:
        # Enforce anchor_density_per_minute <= max_anchor_density_per_minute.
        self.anchor_density_input.setMaximum(self.max_anchor_density_input.value())
        self.max_anchor_density_input.setMinimum(self.anchor_density_input.value())

    def _sync_anchor_durations(self, *_args: object) -> None:
        # Enforce min <= base <= max for anchor durations.
        self.base_anchor_duration_input.setMinimum(self.min_anchor_duration_input.value())
        self.base_anchor_duration_input.setMaximum(self.max_anchor_duration_input.value())
        self.min_anchor_duration_input.setMaximum(self.base_anchor_duration_input.value())
        self.max_anchor_duration_input.setMinimum(self.base_anchor_duration_input.value())

    def _sync_max_anchor_count_minimum(self, *_args: object) -> None:
        # max_anchor_count (when capped) must stay >= min_anchor_count. The
        # special "unlimited" value sits at the spin-box minimum, so clamp via
        # value rather than the range to preserve the sentinel.
        required_minimum = int(self.min_anchor_count_input.value())
        current = int(self.max_anchor_count_input.value())
        if current != _MAX_ANCHOR_COUNT_UNLIMITED and current < required_minimum:
            self.max_anchor_count_input.setValue(required_minimum)

    def _sync_bounded_pair(self, low_input: QDoubleSpinBox, high_input: QDoubleSpinBox) -> None:
        # Keep low < high for a paired (low, high) threshold using the low
        # input's single step as the minimum separation.
        gap = low_input.singleStep()
        high_input.setMinimum(low_input.value() + gap)
        low_input.setMaximum(high_input.value() - gap)

    def _sync_ncc_margin_range(self, *_args: object) -> None:
        self._sync_bounded_pair(self.ncc_margin_low_input, self.ncc_margin_high_input)

    def _sync_ncc_prominence_range(self, *_args: object) -> None:
        self._sync_bounded_pair(self.ncc_prominence_low_input, self.ncc_prominence_high_input)

    def _sync_ncc_width_range(self, *_args: object) -> None:
        self._sync_bounded_pair(self.ncc_good_width_input, self.ncc_bad_width_input)

    def _sync_gcc_phat_gate(self, *_args: object) -> None:
        enabled = self.gcc_phat_enabled_checkbox.isChecked()
        self.gcc_phat_only_when_ambiguous_checkbox.setEnabled(enabled)
        self.gcc_phat_tolerance_input.setEnabled(enabled)

    def _sync_checkbox_from_combo(self, *_args: object) -> None:
        method = str(self.stretch_method_combo.currentData())
        is_pitch_preserving = method in _PITCH_PRESERVING_STRETCH_METHODS
        self.pitch_preserving_checkbox.blockSignals(True)
        self.pitch_preserving_checkbox.setChecked(is_pitch_preserving)
        self.pitch_preserving_checkbox.blockSignals(False)

    def _sync_combo_from_checkbox(self, checked: bool, *_args: object) -> None:
        current = str(self.stretch_method_combo.currentData())
        if checked and current not in _PITCH_PRESERVING_STRETCH_METHODS:
            target_index = self.stretch_method_combo.findData("pitch_preserving")
        elif not checked and current in _PITCH_PRESERVING_STRETCH_METHODS:
            target_index = self.stretch_method_combo.findData("resample")
        else:
            return
        if target_index >= 0:
            self.stretch_method_combo.blockSignals(True)
            self.stretch_method_combo.setCurrentIndex(target_index)
            self.stretch_method_combo.blockSignals(False)

    def select_master(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, self.t("gui.dialog.select_master"), filter=SUPPORTED_AUDIO_FILTER)
        if file_path:
            self.master_input.setText(file_path)

    def select_tracks(self) -> None:
        file_paths, _ = QFileDialog.getOpenFileNames(self, self.t("gui.dialog.select_tracks"), filter=SUPPORTED_AUDIO_FILTER)
        for file_path in file_paths:
            if not _contains_item(self.track_list, file_path):
                self.track_list.addItem(QListWidgetItem(file_path))

    def remove_selected_tracks(self) -> None:
        for item in self.track_list.selectedItems():
            self.track_list.takeItem(self.track_list.row(item))

    def select_output(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, self.t("gui.dialog.select_output"))
        if directory:
            self.output_input.setText(directory)

    def _build_anchor_selection_config(self) -> AnchorSelectionConfig:
        max_anchor_count = int(self.max_anchor_count_input.value())
        stratified_bin_count = int(self.stratified_bin_count_input.value())
        anchors_per_bin = int(self.anchors_per_bin_input.value())
        min_snr_db = float(self.min_snr_db_input.value())
        spectral_flatness = float(self.spectral_flatness_input.value())
        return AnchorSelectionConfig(
            anchor_density_per_minute=float(self.anchor_density_input.value()),
            max_anchor_density_per_minute=float(self.max_anchor_density_input.value()),
            min_anchor_count=int(self.min_anchor_count_input.value()),
            max_anchor_count=(None if max_anchor_count == _MAX_ANCHOR_COUNT_UNLIMITED else max_anchor_count),
            stratified_bin_count=(None if stratified_bin_count == _STRATIFIED_BIN_COUNT_AUTO else stratified_bin_count),
            anchors_per_bin=(None if anchors_per_bin == _ANCHORS_PER_BIN_AUTO else anchors_per_bin),
            min_anchor_duration_seconds=float(self.min_anchor_duration_input.value()),
            base_anchor_duration_seconds=float(self.base_anchor_duration_input.value()),
            max_anchor_duration_seconds=float(self.max_anchor_duration_input.value()),
            min_snr_db=(None if min_snr_db <= _MIN_SNR_DB_DISABLED else min_snr_db),
            spectral_flatness_threshold=(
                None if spectral_flatness <= _SPECTRAL_FLATNESS_DISABLED else spectral_flatness
            ),
        )

    def _build_anchor_matching_config(self) -> AnchorMatchingConfig:
        return AnchorMatchingConfig(
            nms_exclusion_seconds=float(self.nms_exclusion_input.value()),
            ncc_min_score=float(self.ncc_min_score_input.value()),
            ncc_min_margin=float(self.ncc_min_margin_input.value()),
            ncc_min_prominence=float(self.ncc_min_prominence_input.value()),
            ncc_good_width_seconds=float(self.ncc_good_width_input.value()),
            ncc_bad_width_seconds=float(self.ncc_bad_width_input.value()),
            ncc_margin_low=float(self.ncc_margin_low_input.value()),
            ncc_margin_high=float(self.ncc_margin_high_input.value()),
            ncc_prominence_low=float(self.ncc_prominence_low_input.value()),
            ncc_prominence_high=float(self.ncc_prominence_high_input.value()),
            gcc_phat_enabled=self.gcc_phat_enabled_checkbox.isChecked(),
            gcc_phat_only_when_ambiguous=self.gcc_phat_only_when_ambiguous_checkbox.isChecked(),
            gcc_phat_agreement_tolerance_seconds=float(self.gcc_phat_tolerance_input.value()),
            min_confidence_for_fit=float(self.min_confidence_for_fit_input.value()),
        )

    def run(self) -> None:
        master_path = self.master_input.text().strip()
        output_dir = self.output_input.text().strip()
        tracks = [self.track_list.item(i).text() for i in range(self.track_list.count())]

        if not master_path:
            self._show_error(self.t("gui.error.master_required"))
            return
        if not tracks:
            self._show_error(self.t("gui.error.track_required"))
            return
        if not output_dir:
            self._show_error(self.t("gui.error.output_required"))
            return

        vad_strategy = self._selected_vad_strategy()
        try:
            anchor_selection = self._build_anchor_selection_config()
            anchor_matching = self._build_anchor_matching_config()
        except ValueError as exc:
            self._show_error(str(exc))
            return
        options = AlignmentOptions(
            master=Path(master_path),
            tracks=[Path(track) for track in tracks],
            out=Path(output_dir),
            analysis_sample_rate=int(self.sample_rate_input.value()),
            normalize_output=self.normalize_checkbox.isChecked(),
            local_adjust_enabled=self.local_adjust_checkbox.isChecked(),
            stretch_ratio_warning_threshold=float(self.stretch_threshold_input.value()),
            stretch_ratio_auto_continue=False,
            stretch_method=str(self.stretch_method_combo.currentData()),
            vad_strategy=vad_strategy,
            pyannote_model=self._selected_pyannote_model(vad_strategy),
            drift_model=str(self.drift_model_input.currentData()),
            allow_nonlinear_drift=self.allow_nonlinear_drift_checkbox.isChecked(),
            max_breakpoints=int(self.max_breakpoints_input.value()),
            min_anchors_for_piecewise=int(self.min_anchors_for_piecewise_input.value()),
            min_anchors_per_segment=int(self.min_anchors_per_segment_input.value()),
            max_anchor_gap_seconds=(
                None if self.max_anchor_gap_input.value() <= 0.0 else float(self.max_anchor_gap_input.value())
            ),
            anchor_selection=anchor_selection,
            anchor_matching=anchor_matching,
            verbose_report=self.verbose_report_checkbox.isChecked(),
        )

        try:
            build_cli_argv(options)
        except ValueError as exc:
            self._show_error(str(exc))
            return

        self._reset_progress_state()
        self.append_log(self.t("gui.log.starting"))
        self._start_worker(options)

    def _reset_progress_state(self) -> None:
        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("0%")
        self.eta_label.setText(self.t("gui.progress_eta_unknown"))
        self.current_task_label.setText(self.t("gui.current_task_starting"))

    def _start_worker(self, options: AlignmentOptions) -> None:
        self._worker_thread = QThread(self)
        self._worker = AlignmentWorker(options)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.log_line.connect(self.append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.start()

    def _on_progress(self, percent: float, eta_seconds: float, task_message: str) -> None:
        clamped_percent = max(0.0, min(100.0, percent))
        self.progress_bar.setValue(int(round(clamped_percent)))
        self.progress_label.setText(f"{clamped_percent:.1f}%")
        self.eta_label.setText(self.t("gui.progress_eta", value=_format_eta(eta_seconds)))
        self.current_task_label.setText(self.t("gui.current_task", task=task_message))

    def _on_finished(self, exit_code: int, options: AlignmentOptions) -> None:
        self.run_button.setEnabled(True)
        if exit_code == EXIT_STRETCH_CONFIRMATION_REQUIRED:
            reply = QMessageBox.warning(
                self,
                self.t("gui.alert.stretch_title"),
                self.t("gui.alert.stretch_continue"),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._reset_progress_state()
                self.append_log(self.t("gui.log.starting"))
                self._start_worker(replace(options, stretch_ratio_auto_continue=True))
            return
        if exit_code == 0:
            self.progress_bar.setValue(100)
            self.progress_label.setText(self.t("gui.progress_done_percent"))
            self.eta_label.setText(self.t("gui.progress_eta_done"))
            self.current_task_label.setText(self.t("gui.current_task_done"))
            self.append_log(self.t("gui.log.success"))
            dialog = QMessageBox(self)
            dialog.setIcon(QMessageBox.Information)
            dialog.setWindowTitle(self.t("gui.dialog.app_name"))
            dialog.setText(self.t("gui.log.success"))
            ok_button = dialog.addButton(QMessageBox.Ok)
            open_output_button = dialog.addButton(self.t("gui.dialog.open_output"), QMessageBox.ActionRole)
            dialog.setDefaultButton(ok_button)
            dialog.exec()
            if dialog.clickedButton() == open_output_button:
                self._open_output_directory(options.out)
            return

        self.append_log(self.t("gui.log.failed", exit_code=exit_code))
        self._show_error(self.t("gui.error.failed", exit_code=exit_code))

    def _open_output_directory(self, output_dir: Path | str) -> None:
        path = Path(output_dir)
        if not path.exists() or not path.is_dir():
            self._show_error(self.t("gui.error.output_not_found", path=str(path)))
            return
        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
        if not opened:
            self._show_error(self.t("gui.error.output_open_failed", path=str(path)))

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, self.t("gui.dialog.app_name"), message)


class AlignmentWorker(QObject):
    finished = Signal(int, object)
    log_line = Signal(str)
    progress = Signal(float, float, str)

    def __init__(self, options: AlignmentOptions) -> None:
        super().__init__()
        self.options = options

    def run(self) -> None:
        try:
            exit_code = run_alignment(
                self.options,
                progress_callback=self._on_progress,
                event_callback=self._on_event,
            )
        except Exception as exc:  # pragma: no cover - GUI runtime guard
            self.log_line.emit(f"error: {exc}")
            exit_code = 1
        self.finished.emit(exit_code, self.options)

    def _on_progress(self, percent: float, eta_seconds: float, task_message: str) -> None:
        self.progress.emit(percent, eta_seconds, task_message)

    def _on_event(self, message: str) -> None:
        self.log_line.emit(message)


def _format_eta(eta_seconds: float) -> str:
    rounded = max(0, int(round(eta_seconds)))
    minutes, seconds = divmod(rounded, 60)
    return f"{minutes:02d}:{seconds:02d}"


def _wrap_layout(layout) -> QWidget:
    wrapper = QWidget()
    wrapper.setLayout(layout)
    wrapper.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return wrapper



def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow(lang=extract_explicit_lang(sys.argv[1:]))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
