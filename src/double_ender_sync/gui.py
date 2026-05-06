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
    QFileDialog,
    QFormLayout,
    QGroupBox,
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from double_ender_sync.api import AlignmentOptions, run_alignment
from double_ender_sync.analysis.vad import DEFAULT_PYANNOTE_MODEL
from double_ender_sync.cli import EXIT_STRETCH_CONFIRMATION_REQUIRED
from double_ender_sync.i18n import TranslationCatalog, resolve_language
from double_ender_sync.i18n.resolver import extract_explicit_lang


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
        self.pyannote_model_input = QLineEdit(DEFAULT_PYANNOTE_MODEL)
        self.pyannote_model_input.setPlaceholderText(DEFAULT_PYANNOTE_MODEL)
        self.vad_strategy_input.currentIndexChanged.connect(self._sync_pyannote_model_input_enabled)
        self._sync_pyannote_model_input_enabled()

        layout.addLayout(form_layout)

        advanced_panel = QWidget()
        advanced_layout = QFormLayout(advanced_panel)
        advanced_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        advanced_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignTop)
        advanced_layout.setHorizontalSpacing(16)
        advanced_layout.setVerticalSpacing(10)
        advanced_layout.addRow(QLabel(self.t("gui.analysis_sample_rate")), self.sample_rate_input)
        advanced_layout.addRow(self.normalize_checkbox)
        advanced_layout.addRow(self.local_adjust_checkbox)
        advanced_layout.addRow(self.pitch_preserving_checkbox)
        advanced_layout.addRow(QLabel(self.t("gui.stretch_threshold")), self.stretch_threshold_input)
        advanced_layout.addRow(QLabel(self.t("gui.vad_strategy")), self.vad_strategy_input)
        advanced_layout.addRow(QLabel(self.t("gui.pyannote_model")), self.pyannote_model_input)

        advanced_box = QGroupBox(self.t("gui.advanced_settings"))
        advanced_box.setCheckable(True)
        advanced_box.setChecked(False)
        advanced_box_layout = QVBoxLayout(advanced_box)
        advanced_box_layout.setContentsMargins(8, 8, 8, 8)
        advanced_box_layout.addWidget(advanced_panel)
        advanced_box.toggled.connect(advanced_panel.setVisible)
        advanced_panel.setVisible(False)
        layout.addWidget(advanced_box)

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

        self.setCentralWidget(root)
        self._worker_thread: QThread | None = None
        self._worker: AlignmentWorker | None = None

    def append_log(self, message: str) -> None:
        self.log_view.append(message)

    def t(self, key: str, **kwargs: object) -> str:
        return self.catalog.t(key, **kwargs)

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

    def _sync_pyannote_model_input_enabled(self, *_args: object) -> None:
        self.pyannote_model_input.setEnabled(self._selected_vad_strategy() == "pyannote")

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
        options = AlignmentOptions(
            master=Path(master_path),
            tracks=[Path(track) for track in tracks],
            out=Path(output_dir),
            analysis_sample_rate=int(self.sample_rate_input.value()),
            normalize_output=self.normalize_checkbox.isChecked(),
            local_adjust_enabled=self.local_adjust_checkbox.isChecked(),
            stretch_ratio_warning_threshold=float(self.stretch_threshold_input.value()),
            stretch_ratio_auto_continue=False,
            stretch_method="pitch_preserving" if self.pitch_preserving_checkbox.isChecked() else "resample",
            vad_strategy=vad_strategy,
            pyannote_model=self._selected_pyannote_model(vad_strategy),
        )

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
            self.progress_label.setText("100.0%")
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
