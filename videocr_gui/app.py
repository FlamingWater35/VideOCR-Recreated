"""Main window: tabs, state, event wiring for VideOCR Recreated."""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
from typing import Any

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import args as argsmod
from . import config, i18n, resources
from . import constants as C
from .about_tab import AboutTab
from .dialogs import CountdownDialog, ask_yes_no, info
from .progress import ProgressUpdate
from .queue_tab import QueueTab
from .settings_tab import SettingsTab
from .video_preview import VideoPreview
from .widgets import ClickableSlider, OutlinedProgressBar, WheelGuardComboBox
from .workers import VIDEOCR_PATH, CLIWorker

try:  # pragma: no cover
    import psutil
except Exception:  # pragma: no cover
    psutil = None

if sys.platform == "win32":
    try:
        import PyTaskbar  # type: ignore
    except Exception:  # pragma: no cover
        PyTaskbar = None
    try:
        from winotify import Notification, audio  # type: ignore
    except Exception:  # pragma: no cover
        Notification = None
        audio = None
else:
    Notification = None
    audio = None
    try:
        from plyer import notification  # type: ignore
    except Exception:  # pragma: no cover
        notification = None


class MainWindow(QMainWindow):
    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.setWindowTitle("VideOCR Recreated")

        icon = resources.icon_path()
        if icon:
            self.setWindowIcon(QIcon(icon))

        self._settings = settings if settings is not None else config.load_settings()
        self._video_path: str | None = None
        self._current_ms = 0.0
        self._duration_ms = 0
        self._batch_queue: list[dict[str, Any]] = []
        self._worker: CLIWorker | None = None
        self._is_processing = False
        self._cancelled_by_user = False
        self._paused = False
        self._current_pid: int | None = None
        self._wake_lock: Any = None
        self._taskbar: Any = None
        self._graph_size = (720, 405)

        self._crop_save_timer = QTimer(self)
        self._crop_save_timer.setSingleShot(True)
        self._crop_save_timer.setInterval(300)
        self._crop_save_timer.timeout.connect(self._flush_crop_boxes)

        # Load the saved UI language BEFORE building widgets so the initial
        # render is localized (i18n.tr() falls back to English defaults while
        # the language dict is empty).
        i18n.load_language(str(self._settings.get("--language", "en")))

        self._build_ui()
        self._init_taskbar()
        self._apply_settings_to_ui()
        self._resize_to_work_area()

    # --- UI construction ----------------------------------------------------
    def _build_ui(self) -> None:
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.preview = VideoPreview()
        self.preview.frame_loaded.connect(self._on_frame_loaded)
        self.preview.video_error.connect(self._on_video_error)
        self.preview.crop_changed.connect(self._on_crop_changed)

        self.tabs.addTab(
            self._build_process_tab(), i18n.tr("tab_video", "Process Video")
        )
        self.queue_tab = QueueTab()
        self.tabs.addTab(self.queue_tab, i18n.tr("tab_batch", "Queue"))
        self.settings_tab = SettingsTab()
        self.tabs.addTab(
            self.settings_tab, i18n.tr("tab_advanced", "Advanced Settings")
        )
        self.about_tab = AboutTab(
            config.__version__ if hasattr(config, "__version__") else "1.6.0"
        )
        self.tabs.addTab(self.about_tab, i18n.tr("tab_about", "About"))

        # wire queue signals
        self.queue_tab.start_requested.connect(self._on_queue_start)
        self.queue_tab.stop_requested.connect(self._on_cancel)
        self.queue_tab.pause_requested.connect(self._on_pause)
        self.queue_tab.resume_requested.connect(self._on_resume)
        self.queue_tab.move_up_requested.connect(self._on_move_up)
        self.queue_tab.move_down_requested.connect(self._on_move_down)
        self.queue_tab.reset_requested.connect(self._on_reset)
        self.queue_tab.edit_requested.connect(self._on_edit)
        self.queue_tab.remove_requested.connect(self._on_remove)
        self.queue_tab.clear_requested.connect(self._on_clear)

        # wire settings signals
        self.settings_tab.settings_changed.connect(self._on_settings_changed)
        self.settings_tab.ui_language_changed.connect(self._on_ui_language)
        self.settings_tab.scaling_changed.connect(self._on_scaling_changed)
        self.settings_tab.directml_gpu_changed.connect(self._on_directml_gpu)
        self.settings_tab.restart_requested.connect(self._restart)
        self.settings_tab.info_requested.connect(self._show_engine_info)
        self.settings_tab.help_requested.connect(self._show_help)

    def _build_process_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(8)

        # source / output row
        row1 = QHBoxLayout()
        self.source_combo = WheelGuardComboBox()
        self.source_combo.setMinimumWidth(300)
        self.source_combo.setEditable(False)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        self.source_lbl = QLabel(i18n.tr("lbl_source", "Source:"))
        row1.addWidget(self.source_lbl)
        row1.addWidget(self.source_combo, 1)
        self.open_btn = QPushButton(i18n.tr("btn_browse", "Open File..."))
        self.folder_btn = QPushButton(i18n.tr("btn_browse_folder", "Open Folder..."))
        self.open_btn.clicked.connect(self._open_file)
        self.folder_btn.clicked.connect(self._open_folder)
        row1.addWidget(self.open_btn)
        row1.addWidget(self.folder_btn)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        self.output_lbl = QLabel(i18n.tr("lbl_output_srt", "Output SRT:"))
        row2.addWidget(self.output_lbl)
        row2.addWidget(self.output_edit, 1)
        self.save_as_btn = QPushButton(i18n.tr("btn_save_as", "Save As..."))
        self.save_as_btn.clicked.connect(self._save_as)
        self.info_btn = QPushButton(i18n.tr("btn_info", "Info"))
        self.info_btn.clicked.connect(self._show_engine_info)
        self.help_btn = QPushButton(i18n.tr("btn_how_to_use", "How to Use"))
        self.help_btn.clicked.connect(self._show_help)
        row2.addWidget(self.save_as_btn)
        row2.addWidget(self.info_btn)
        row2.addWidget(self.help_btn)
        layout.addLayout(row2)

        # engine / language / position row
        row3 = QHBoxLayout()
        self.engine_lbl = QLabel(i18n.tr("lbl_ocr_engine", "OCR Engine:"))
        row3.addWidget(self.engine_lbl)
        self.engine_combo = WheelGuardComboBox()
        self.engine_combo.setMinimumWidth(240)
        row3.addWidget(self.engine_combo, 1)
        self.lang_lbl = QLabel(i18n.tr("lbl_sub_lang", "Subtitle Language:"))
        row3.addWidget(self.lang_lbl)
        self.lang_combo = WheelGuardComboBox()
        self.lang_combo.setMinimumWidth(150)
        row3.addWidget(self.lang_combo, 1)
        self.pos_lbl = QLabel(i18n.tr("lbl_sub_pos", "Subtitle Position:"))
        row3.addWidget(self.pos_lbl)
        self.pos_combo = WheelGuardComboBox()
        self.pos_combo.setMinimumWidth(120)
        row3.addWidget(self.pos_combo)
        layout.addLayout(row3)
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        self.lang_combo.currentIndexChanged.connect(self._on_lang_changed)
        self.pos_combo.currentIndexChanged.connect(self._on_pos_changed)

        layout.addWidget(self.preview, 1)

        # seek row
        seek_row = QHBoxLayout()
        self.seek_lbl = QLabel(i18n.tr("lbl_seek", "Seek:"))
        seek_row.addWidget(self.seek_lbl)
        self.seek_slider = ClickableSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setEnabled(False)
        self.seek_slider.valueChanged.connect(self._on_seek)
        seek_row.addWidget(self.seek_slider, 1)
        self.time_label = QLabel(i18n.tr("time_text_empty", "Time: -/-"))
        seek_row.addWidget(self.time_label)
        layout.addLayout(seek_row)

        # crop coords row
        crop_row = QHBoxLayout()
        self.crop_lbl = QLabel(i18n.tr("lbl_crop_box", "Crop Box (X, Y, W, H):"))
        crop_row.addWidget(self.crop_lbl)
        self.crop_label = QLabel(i18n.tr("crop_not_set", "Not Set"))
        self.crop_label.setStyleSheet("font-family: monospace;")
        crop_row.addWidget(self.crop_label, 1)
        self.center_crop_btn = QPushButton()
        self.center_crop_btn.setIcon(resources.center_crop_icon())
        self.center_crop_btn.setIconSize(QSize(16, 16))
        self.center_crop_btn.setFixedSize(30, 30)
        self.center_crop_btn.setToolTip(i18n.tr("btn_center_crop", "Center Crop Box"))
        self.center_crop_btn.setEnabled(False)
        self.center_crop_btn.clicked.connect(self.preview.center_last_crop_box)
        self.center_h_btn = QPushButton()
        self.center_h_btn.setIcon(resources.center_horizontal_icon())
        self.center_h_btn.setIconSize(QSize(16, 16))
        self.center_h_btn.setFixedSize(30, 30)
        self.center_h_btn.setToolTip(i18n.tr("btn_center_crop_h", "Center Horizontally"))
        self.center_h_btn.setEnabled(False)
        self.center_h_btn.clicked.connect(self.preview.center_last_crop_box_horizontal)
        self.center_v_btn = QPushButton()
        self.center_v_btn.setIcon(resources.center_vertical_icon())
        self.center_v_btn.setIconSize(QSize(16, 16))
        self.center_v_btn.setFixedSize(30, 30)
        self.center_v_btn.setToolTip(i18n.tr("btn_center_crop_v", "Center Vertically"))
        self.center_v_btn.setEnabled(False)
        self.center_v_btn.clicked.connect(self.preview.center_last_crop_box_vertical)
        self.clear_crop_btn = QPushButton(i18n.tr("btn_clear_crop", "Clear Crop"))
        self.clear_crop_btn.clicked.connect(self.preview.clear_crop)
        crop_row.addWidget(self.center_crop_btn)
        crop_row.addWidget(self.center_h_btn)
        crop_row.addWidget(self.center_v_btn)
        crop_row.addWidget(self.clear_crop_btn)
        layout.addLayout(crop_row)

        # run row
        run_row = QHBoxLayout()
        self.run_btn = QPushButton(i18n.tr("btn_run", "Run"))
        self.run_btn.setObjectName("primaryButton")
        self.pause_btn = QPushButton(i18n.tr("btn_pause", "Pause"))
        self.cancel_btn = QPushButton(i18n.tr("btn_cancel", "Cancel"))
        self.cancel_btn.setObjectName("dangerButton")
        self.pause_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._on_run)
        self.pause_btn.clicked.connect(self._on_pause)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.add_btn = QPushButton(i18n.tr("btn_add_to_queue", "Add to Queue"))
        self.add_all_btn = QPushButton(
            i18n.tr("btn_add_all_to_queue", "Add All to Queue")
        )
        self.add_btn.clicked.connect(self._on_add_to_queue)
        self.add_all_btn.clicked.connect(self._on_add_all)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.pause_btn)
        run_row.addWidget(self.cancel_btn)
        run_row.addStretch(1)
        run_row.addWidget(self.add_btn)
        run_row.addWidget(self.add_all_btn)
        layout.addLayout(run_row)

        # progress row
        progress_row = QHBoxLayout()
        self.status_label = QLabel("")
        self.eta_label = QLabel("")
        self.eta_label.setObjectName("etaLabel")
        self.progress_bar = OutlinedProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_row.addWidget(self.status_label, 1)
        progress_row.addWidget(self.eta_label)
        layout.addLayout(progress_row)
        layout.addWidget(self.progress_bar)

        # post-action row
        post_row = QHBoxLayout()
        post_row.addStretch(1)
        self.when_ready_lbl = QLabel(i18n.tr("lbl_when_ready", "When ready:"))
        post_row.addWidget(self.when_ready_lbl)
        self.post_action_combo = WheelGuardComboBox()
        self._refresh_post_action_combo()
        self.post_action_combo.currentIndexChanged.connect(self._on_post_action_changed)
        post_row.addWidget(self.post_action_combo)
        layout.addLayout(post_row)

        # log pane
        self.log_lbl = QLabel(i18n.tr("lbl_log", "Log:"))
        layout.addWidget(self.log_lbl)
        self.log_pane = QPlainTextEdit()
        self.log_pane.setReadOnly(True)
        self.log_pane.setMaximumBlockCount(5000)
        layout.addWidget(self.log_pane, 1)

        # keyboard seek
        seek_action_left = QAction(self)
        seek_action_left.setShortcut(QKeySequence(Qt.Key.Key_Left))
        seek_action_left.triggered.connect(lambda: self._seek_step(-1))
        self.addAction(seek_action_left)
        seek_action_right = QAction(self)
        seek_action_right.setShortcut(QKeySequence(Qt.Key.Key_Right))
        seek_action_right.triggered.connect(lambda: self._seek_step(1))
        self.addAction(seek_action_right)
        return page

    def _refresh_post_action_combo(self) -> None:
        self.post_action_combo.clear()
        for key in C.POST_ACTION_KEYS:
            self.post_action_combo.addItem(i18n.tr(key, C.DEFAULT_ACTION_TEXTS[key]))
        idx = int(self._settings.get("post_action", 0))
        if 0 <= idx < self.post_action_combo.count():
            self.post_action_combo.setCurrentIndex(idx)

    # --- taskbar / wake ------------------------------------------------------
    def _init_taskbar(self) -> None:
        if sys.platform == "win32" and PyTaskbar is not None:
            # Defer until the window is realized: winId() may be invalid/0
            # before show(), which makes TaskbarProgress fail or crash.
            QTimer.singleShot(0, self._setup_taskbar)

    def _setup_taskbar(self) -> None:
        if self._taskbar is not None or PyTaskbar is None:
            return
        try:
            self._taskbar = PyTaskbar.TaskbarProgress(int(self.winId()))
            self._taskbar.set_progress_type(PyTaskbar.ProgressType.NOPROGRESS)
        except Exception as e:
            config.log_error(f"Taskbar init failed: {e}")
            self._taskbar = None

    def _update_taskbar(
        self, state: str | None = None, progress: int | None = None
    ) -> None:
        if self._taskbar is None:
            return
        try:
            if state == "paused":
                self._taskbar.set_progress_type(PyTaskbar.ProgressType.PAUSED)
            elif state == "normal":
                self._taskbar.set_progress_type(PyTaskbar.ProgressType.NORMAL)
            if progress is not None:
                self._taskbar.set_progress(int(progress), 100)
        except Exception:
            pass

    def _set_system_awake(self, should_be_awake: bool) -> None:
        if should_be_awake:
            if self._wake_lock is None:
                try:
                    from wakepy import keep

                    self._wake_lock = keep.running()
                    self._wake_lock.__enter__()
                except Exception as e:
                    config.log_error(f"Failed to acquire wake lock: {e}")
        else:
            if self._wake_lock is not None:
                try:
                    self._wake_lock.__exit__(None, None, None)
                except Exception as e:
                    config.log_error(f"Failed to release wake lock: {e}")
                finally:
                    self._wake_lock = None

    # --- settings <-> UI ------------------------------------------------------
    def _apply_settings_to_ui(self) -> None:
        i18n.load_language(str(self._settings.get("--language", "en")))
        self.settings_tab.populate(
            self._settings, sorted(i18n.get_available_languages().keys())
        )
        self._populate_engine_lang_pos()
        self._retranslate_all()
        self.source_combo.clear()
        self.output_edit.setText("")
        self.post_action_combo.setCurrentIndex(
            int(self._settings.get("post_action", 0))
        )
        self._refresh_directml_combo()
        self.preview.set_dual_zone(bool(self._settings.get("--use_dual_zone", False)))

    def _populate_engine_lang_pos(self) -> None:
        """Populates the OCR Engine / Subtitle Language / Position combos from settings."""
        self.engine_combo.blockSignals(True)
        self.lang_combo.blockSignals(True)
        self.pos_combo.blockSignals(True)
        try:
            engine = self._settings.get("ocr_engine", C.DEFAULT_OCR_ENGINE)
            # Old configs may still store a display name from an engine that
            # was removed from the list (e.g. EasyOCR DirectML); map it to the
            # engine that still uses it internally.
            engine = C.LEGACY_OCR_ENGINE_MAP.get(engine, engine)
            self.engine_combo.clear()
            self.engine_combo.addItems(C.OCR_ENGINES)
            idx = self.engine_combo.findText(engine)
            if idx >= 0:
                self.engine_combo.setCurrentIndex(idx)

            # language list depends on engine
            self._update_lang_list(engine, keep_selection=True)

            pos_internal = self._settings.get(
                "subtitle_position", C.DEFAULT_INTERNAL_SUBTITLE_POSITION
            )
            display_to_internal = {
                i18n.tr(key, key): internal
                for key, internal in C.SUBTITLE_POSITIONS_LIST
            }
            internal_to_display = {v: k for k, v in display_to_internal.items()}
            self.pos_combo.clear()
            self.pos_combo.addItems(list(display_to_internal.keys()))
            display = internal_to_display.get(
                str(pos_internal), list(display_to_internal.keys())[0]
            )
            idx = self.pos_combo.findText(display)
            if idx >= 0:
                self.pos_combo.setCurrentIndex(idx)
        finally:
            self.engine_combo.blockSignals(False)
            self.lang_combo.blockSignals(False)
            self.pos_combo.blockSignals(False)

    def _update_lang_list(self, engine: str, keep_selection: bool = False) -> None:
        """Refreshes the language combo for the given engine; preserves selection if possible."""
        if "Google Lens" in engine:
            names = C.lens_display_names
        elif "EasyOCR DirectML" in engine or "ONNX Runtime DirectML" in engine:
            names = C.easyocr_display_names
        else:
            names = C.paddle_display_names

        current = self.lang_combo.currentText()
        self.lang_combo.clear()
        self.lang_combo.addItems(names)
        if keep_selection:
            saved = self._settings.get("subtitle_language", C.DEFAULT_SUBTITLE_LANGUAGE)
            if saved in names:
                self.lang_combo.setCurrentText(saved)
            elif current in names:
                self.lang_combo.setCurrentText(current)
            else:
                default_idx = (
                    names.index(C.DEFAULT_SUBTITLE_LANGUAGE)
                    if C.DEFAULT_SUBTITLE_LANGUAGE in names
                    else 0
                )
                self.lang_combo.setCurrentIndex(default_idx)
        else:
            default_idx = (
                names.index(C.DEFAULT_SUBTITLE_LANGUAGE)
                if C.DEFAULT_SUBTITLE_LANGUAGE in names
                else 0
            )
            self.lang_combo.setCurrentIndex(default_idx)

    def _on_engine_changed(self) -> None:
        if self.engine_combo.signalsBlocked():
            return
        engine = self.engine_combo.currentText()
        self._settings["ocr_engine"] = engine
        self._update_lang_list(engine, keep_selection=True)
        self._settings["subtitle_language"] = self.lang_combo.currentText()
        config.save_settings(self._settings)
        self._update_output_path()

    def _current_settings(self) -> dict[str, Any]:
        """Merges settings-tab widget values with the process-tab engine /
        language / position selections (which live in self._settings)."""
        merged = self.settings_tab.read_settings()
        for key in ("ocr_engine", "subtitle_language", "subtitle_position"):
            if key in self._settings:
                merged[key] = self._settings[key]
        return merged

    def _on_lang_changed(self) -> None:
        if self.lang_combo.signalsBlocked():
            return
        self._settings["subtitle_language"] = self.lang_combo.currentText()
        config.save_settings(self._settings)
        self._update_output_path()

    def _on_pos_changed(self) -> None:
        if self.pos_combo.signalsBlocked():
            return
        display_to_internal = {
            i18n.tr(key, key): internal for key, internal in C.SUBTITLE_POSITIONS_LIST
        }
        internal = display_to_internal.get(
            self.pos_combo.currentText(), C.DEFAULT_INTERNAL_SUBTITLE_POSITION
        )
        self._settings["subtitle_position"] = internal
        config.save_settings(self._settings)

    def _retranslate_all(self) -> None:
        self.tabs.setTabText(0, i18n.tr("tab_video", "Process Video"))
        self.tabs.setTabText(1, i18n.tr("tab_batch", "Queue"))
        self.tabs.setTabText(2, i18n.tr("tab_advanced", "Advanced Settings"))
        self.tabs.setTabText(3, i18n.tr("tab_about", "About"))
        self.queue_tab.retranslate()
        self.about_tab.retranslate(self._version())
        self.settings_tab.retranslate()
        self._refresh_post_action_combo()
        self._translate_process_tab()

    def _translate_process_tab(self) -> None:
        """Re-translates every process-tab widget that has translatable text."""
        self.source_lbl.setText(i18n.tr("lbl_source", "Source:"))
        self.output_lbl.setText(i18n.tr("lbl_output_srt", "Output SRT:"))
        self.engine_lbl.setText(i18n.tr("lbl_ocr_engine", "OCR Engine:"))
        self.lang_lbl.setText(i18n.tr("lbl_sub_lang", "Subtitle Language:"))
        self.pos_lbl.setText(i18n.tr("lbl_sub_pos", "Subtitle Position:"))
        self.seek_lbl.setText(i18n.tr("lbl_seek", "Seek:"))
        self.crop_lbl.setText(i18n.tr("lbl_crop_box", "Crop Box (X, Y, W, H):"))
        self.crop_label.setText(i18n.tr("crop_not_set", "Not Set"))
        self.when_ready_lbl.setText(i18n.tr("lbl_when_ready", "When ready:"))
        self.log_lbl.setText(i18n.tr("lbl_log", "Log:"))
        self.time_label.setText(i18n.tr("time_text_empty", "Time: -/-"))
        self.open_btn.setText(i18n.tr("btn_browse", "Open File..."))
        self.folder_btn.setText(i18n.tr("btn_browse_folder", "Open Folder..."))
        self.save_as_btn.setText(i18n.tr("btn_save_as", "Save As..."))
        self.info_btn.setText(i18n.tr("btn_info", "Info"))
        self.help_btn.setText(i18n.tr("btn_how_to_use", "How to Use"))
        self.center_crop_btn.setToolTip(i18n.tr("btn_center_crop", "Center Crop Box"))
        self.center_h_btn.setToolTip(i18n.tr("btn_center_crop_h", "Center Horizontally"))
        self.center_v_btn.setToolTip(i18n.tr("btn_center_crop_v", "Center Vertically"))
        self.clear_crop_btn.setText(i18n.tr("btn_clear_crop", "Clear Crop"))
        self.run_btn.setText(i18n.tr("btn_run", "Run"))
        self.pause_btn.setText(i18n.tr("btn_pause", "Pause"))
        self.cancel_btn.setText(i18n.tr("btn_cancel", "Cancel"))
        self.add_btn.setText(i18n.tr("btn_add_to_queue", "Add to Queue"))
        self.add_all_btn.setText(i18n.tr("btn_add_all_to_queue", "Add All to Queue"))
        self._populate_engine_lang_pos()

    def _version(self) -> str:
        return getattr(config, "__version__", "1.6.0")

    def _resize_to_work_area(self) -> None:
        screen = self.screen()
        if screen is None:
            self.resize(1000, 720)
            return
        avail = screen.availableGeometry()
        self.resize(min(1000, avail.width() - 40), min(760, avail.height() - 40))

    # --- source / output ------------------------------------------------------
    def _open_file(self) -> None:
        filetypes = i18n.tr("video_file_types", "Video Files")
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "",
            f"{filetypes} (*.mp4 *.avi *.mkv *.mov *.webm *.flv *.wmv *.ts *.m2ts);;"
            f"{i18n.tr('all_file_types', 'All Files')} (*.*)",
        )
        if filename:
            self.source_combo.blockSignals(True)
            self.source_combo.clear()
            self.source_combo.addItem(filename)
            self.source_combo.blockSignals(False)
            self._load_video(filename)

    def _open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open Folder")
        if not folder:
            return
        videos = self._scan_video_folder(folder)
        if not videos:
            info(self, "No Videos", "No supported videos found in folder.")
            return
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItems(videos)
        self.source_combo.setCurrentIndex(0)
        self.source_combo.blockSignals(False)
        self._load_video(videos[0])

    @staticmethod
    def _scan_video_folder(folder: str) -> list[str]:
        exts = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".ts", ".m2ts"}
        if not os.path.isdir(folder):
            return []
        videos = []
        for entry in os.listdir(folder):
            full = os.path.join(folder, entry)
            if os.path.isfile(full) and os.path.splitext(entry)[1].lower() in exts:
                videos.append(full)
        return sorted(videos)

    def _on_source_changed(self, index: int) -> None:
        """Loads the video selected in the source combo."""
        if index < 0 or self.source_combo.signalsBlocked():
            return
        path = self.source_combo.itemText(index)
        if path and path != self._video_path:
            self._load_video(path)

    def _load_video(self, path: str) -> None:
        self._video_path = path
        self.run_btn.setEnabled(False)
        self.seek_slider.setEnabled(False)
        self.output_edit.setText("")
        if not self.preview.load_video(path):
            self._video_path = None
            return

    def _on_frame_loaded(self, path: str, duration_ms: int) -> None:
        self._duration_ms = duration_ms
        self._current_ms = 0.0
        self.seek_slider.setRange(0, duration_ms)
        self.seek_slider.setValue(0)
        self.seek_slider.setEnabled(True)
        self._update_time_display()
        self.run_btn.setEnabled(True)
        self.save_as_btn.setEnabled(True)
        # output path
        self._settings["_video_duration_ms"] = duration_ms
        self._update_output_path()
        # restore the saved crop box selection once the first frame has been
        # rendered (the preview's resized dimensions are set by show_frame)
        QTimer.singleShot(0, self._restore_crop_boxes)

    def _restore_crop_boxes(self) -> None:
        """Restores the saved crop boxes (relative coords) for the current video."""
        if not self._settings.get("--save_crop_box", False):
            return
        w, h = self.preview.original_size
        if w <= 0 or h <= 0:
            return
        # Ensure the preview knows the current zone limit before restoring.
        self.preview.set_dual_zone(bool(self._settings.get("--use_dual_zone", False)))
        from .crop import absolute_from_relative

        saved = config.parse_saved_crop_boxes(
            str(self._settings.get("--saved_crop_boxes", "[]"))
        )
        boxes = []
        limit = 2 if self._settings.get("--use_dual_zone", False) else 1
        for rel in saved[:limit]:
            abs_coords = absolute_from_relative(rel, w, h)
            if abs_coords["crop_width"] <= 0 or abs_coords["crop_height"] <= 0:
                continue
            boxes.append({"coords": abs_coords})
        if boxes:
            self.preview.restore_crop_boxes(boxes)
            self.crop_label.setText(self.preview.crop_coords_text())

    def _on_video_error(self, path: str) -> None:
        info(
            self,
            i18n.tr("error_invalid_video_title", "Invalid or Empty Video File"),
            i18n.tr(
                "error_invalid_video_msg",
                "Could not load video, video has no frames, or FPS is zero:\n{}",
            ).format(path),
        )
        self._video_path = None

    def _update_output_path(self) -> None:
        if not self._video_path:
            return
        try:
            out = argsmod.generate_output_path(
                self._video_path, self._current_settings()
            )
            self.output_edit.setText(str(out))
            self._settings["--output"] = str(out)
        except Exception as e:
            config.log_error(f"Could not generate output path: {e}")
            self.output_edit.setText("")

    def _save_as(self) -> None:
        current = self.output_edit.text()
        if self._current_settings().get("enable_label_detection"):
            path, _ = QFileDialog.getSaveFileName(
                self,
                i18n.tr("save_as_title", "Save As"),
                current,
                f"{i18n.tr('save_as_ass_filter_name', 'Advanced SubStation Alpha Subtitle')} (*.ass);;{i18n.tr('all_file_types', 'All Files')} (*.*)",
            )
        else:
            path, _ = QFileDialog.getSaveFileName(
                self,
                i18n.tr("save_as_title", "Save As"),
                current,
                f"{i18n.tr('save_as_filter_name', 'SubRip Subtitle')} (*.srt);;{i18n.tr('all_file_types', 'All Files')} (*.*)",
            )
        if path:
            self.output_edit.setText(path)

    # --- seek / preview --------------------------------------------------------
    def _on_seek(self, value: int) -> None:
        if abs(value - self._current_ms) > 50:
            self._current_ms = float(value)
            self.preview.seek_to(self._current_ms)
            self._update_time_display()

    def _seek_step(self, direction: int) -> None:
        if not self._video_path or self._duration_ms <= 0:
            return
        try:
            step = float(
                self.settings_tab.read_settings().get("--keyboard_seek_step", 1)
            )
        except (ValueError, TypeError):
            step = 1
        new_time = self._current_ms + direction * step * 1000.0
        new_time = max(0.0, min(float(self._duration_ms), new_time))
        if new_time != self._current_ms:
            self._current_ms = new_time
            self.seek_slider.setValue(int(new_time))
            self.preview.seek_to(new_time)
            self._update_time_display()

    def _update_time_display(self) -> None:
        if self._duration_ms > 0:
            fmt = i18n.tr("time_text_format", "Time: {}")
            self.time_label.setText(
                fmt.format(
                    self._fmt_time(self._current_ms / 1000.0)
                    + " / "
                    + self._fmt_time(self._duration_ms / 1000.0)
                )
            )
        else:
            self.time_label.setText(i18n.tr("time_text_empty", "Time: -/-"))

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        seconds = int(seconds)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    # --- crop ----------------------------------------------------------------
    def _on_crop_changed(self, boxes: list[dict[str, Any]]) -> None:
        # boxes live in the preview; refresh the label + button state, and
        # persist the crop selection when "Save Crop Box Selection" is enabled.
        self.crop_label.setText(self.preview.crop_coords_text())
        self.clear_crop_btn.setEnabled(bool(boxes))
        self.center_crop_btn.setEnabled(bool(boxes))
        self.center_h_btn.setEnabled(bool(boxes))
        self.center_v_btn.setEnabled(bool(boxes))
        if not self._settings.get("--save_crop_box", False):
            return
        self._save_crop_boxes()

    def _save_crop_boxes(self) -> None:
        """Persists the current crop boxes as relative (0..1) coords.

        Debounced: dragging/resizing a box emits crop_changed on every pixel,
        so the config write is deferred ~300 ms to coalesce bursts.
        """
        if self._crop_save_timer.isActive():
            self._crop_save_timer.stop()
        self._crop_save_timer.start()

    def _flush_crop_boxes(self) -> None:
        from .crop import relative_from_absolute

        w, h = self.preview.original_size
        if w <= 0 or h <= 0:
            return
        boxes = []
        for box in self.preview.crop_boxes:
            coords = box.get("coords", {})
            if not coords:
                continue
            boxes.append(relative_from_absolute(coords, w, h))
        self._settings["--saved_crop_boxes"] = repr(boxes)
        config.save_settings(self._settings)

    def _on_crop_clear(self) -> None:
        self.preview.clear_crop()

    # --- settings changes ------------------------------------------------------
    def _on_settings_changed(self, keys: list[str]) -> None:
        # The settings tab only owns its own widgets; the process-tab OCR
        # engine / language / position selections live in self._settings and
        # must NOT be overwritten by the settings-tab snapshot (which is
        # taken at boot and would clobber a fresh engine selection back to
        # the boot default).
        tab_settings = self.settings_tab.read_settings()
        for key in (
            "ocr_engine",
            "subtitle_language",
            "subtitle_position",
        ):
            tab_settings.pop(key, None)
        self._settings.update(tab_settings)
        for key in keys:
            # Output extension/format changes when the subtitle crop, save dir,
            # or label detection setting changes.
            if key in ("--save_in_video_dir", "enable_label_detection"):
                self._update_output_path()
        # Propagate dual-zone toggle to the preview so the user can draw 2 boxes.
        if "--use_dual_zone" in keys:
            self.preview.set_dual_zone(
                bool(self._settings.get("--use_dual_zone", False))
            )
        config.save_settings(self._settings)

    def _on_directml_gpu(self, index: str) -> None:
        self._settings["--directml_device_index"] = index
        config.save_settings(self._settings)

    def _on_post_action_changed(self, index: int) -> None:
        self._settings["post_action"] = index
        config.save_settings(self._settings)

    def _on_ui_language(self, code_or_native: str) -> None:
        # The signal now carries the language code; accept a native name too
        # for robustness.
        code = code_or_native
        if code not in i18n.get_available_languages().values():
            code = i18n.get_available_languages().get(code_or_native)
        if not code:
            return
        self._settings["--language"] = code
        i18n.load_language(code)
        self._retranslate_all()
        self.settings_tab.populate(
            self._settings, sorted(i18n.get_available_languages().keys())
        )
        config.save_settings(self._settings)

    def _on_scaling_changed(self, value: str) -> None:
        self._settings["gui_scaling"] = value
        config.save_settings(self._settings)
        if ask_yes_no(
            self,
            i18n.tr("title_restart", "Restart Required"),
            i18n.tr(
                "msg_restart_scaling",
                "The scaling factor has been updated.\nWould you like to restart the application now to apply this change?",
            ),
        ):
            self._restart()

    def _restart(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        if sys.argv[0].endswith((".py", ".pyw")):
            cmd = [sys.executable] + sys.argv
        else:
            cmd = sys.argv
        subprocess.Popen(cmd)
        self.close()

    def _refresh_directml_combo(self) -> None:
        from . import directml

        combo = self.settings_tab._widgets["-DML_ADAPTER_COMBO-"]
        options = directml.get_adapter_options()
        idx = self._settings.get("--directml_device_index", "0")
        selected = directml.index_to_option(options, idx)
        if selected not in options:
            options.append(selected)
        self.settings_tab._block_signals = True
        try:
            combo.clear()
            combo.addItems(options)
            combo.setCurrentText(selected)
        finally:
            self.settings_tab._block_signals = False

    # --- info/help -------------------------------------------------------------
    def _show_engine_info(self) -> None:
        info(
            self,
            i18n.tr("engine_info", "OCR Engine Information"),
            i18n.tr(
                "engine_message",
                "PaddleOCR (Det. + Rec.):\n"
                "• 100% local processing.\n"
                "• Both text detection and recognition are done locally.\n\n"
                "PaddleOCR (Det.) + Google Lens (Rec.):\n"
                "• Hybrid processing.\n"
                "• PaddleOCR handles text detection locally.\n"
                "• Google Lens (online) handles text recognition.\n"
                "• Requires an active internet connection.\n\n"
                "ONNX Runtime DirectML (AMD GPU Experimental):\n"
                "• Experimental local ONNX Runtime DirectML backend for Windows AMD GPUs.\n"
                "• Uses RapidOCR with the DirectML execution provider.\n"
                "• Falls back to EasyOCR DirectML Hybrid when the ONNX/DirectML stack is unavailable.",
            ),
        )

    def _show_help(self) -> None:
        info(
            self,
            i18n.tr("help_title", "Cropping Info"),
            i18n.tr(
                "help_message",
                "Draw a crop box over the subtitle region in the video.\n"
                "Use click+drag to select.\n"
                "In 'Dual Zone' mode, you can draw two crop boxes.\n"
                "If no crop box is selected, the bottom third of the video\n"
                "will be used for OCR by default.",
            ),
        )

    # --- queue ----------------------------------------------------------------
    def _jobs(self) -> list[dict[str, Any]]:
        return self._batch_queue

    def _refresh_queue_ui(self) -> None:
        self.queue_tab.set_jobs(self._batch_queue)
        pending = any(j["status"] == "Pending" for j in self._batch_queue)
        if pending:
            self.run_btn.setText(i18n.tr("btn_start_queue", "Start Queue"))
        else:
            self.run_btn.setText(i18n.tr("btn_run", "Run"))
        self._update_queue_tab_title()

    def _update_queue_tab_title(self) -> None:
        active = len(
            [
                j
                for j in self._batch_queue
                if j["status"] in ("Pending", "Processing", "Cancelled", "Paused")
            ]
        )
        base = i18n.tr("tab_batch", "Queue")
        self.tabs.setTabText(1, f"{base} ({active})" if active else base)

    def _on_add_to_queue(self) -> None:
        if not self._video_path:
            return
        settings = self._current_settings()
        settings["_video_duration_ms"] = self._duration_ms
        args, errors = argsmod.build_args(
            self._video_path,
            settings,
            self.preview.crop_boxes,
            self.output_edit.text() or None,
        )
        if errors or args is None:
            info(self, "Validation Error", "\n".join(errors))
            return
        target = args["output"]
        existing = next(
            (
                i
                for i, j in enumerate(self._batch_queue)
                if j["args"]["output"] == target
            ),
            None,
        )
        if existing is not None:
            status = self._batch_queue[existing]["status"]
            if status in ("Processing", "Paused"):
                info(
                    self,
                    i18n.tr("title_duplicate", "Duplicate"),
                    i18n.tr(
                        "msg_duplicate_queue_running",
                        "A job for '{}' is currently active (Status: {}).\nPlease change the output path or wait for it to finish.",
                    ).format(os.path.basename(target), status),
                )
                return
            if ask_yes_no(
                self,
                i18n.tr("title_duplicate_job", "Duplicate Job"),
                i18n.tr(
                    "popup_duplicate_msg",
                    "A job for this output file already exists (Status: {}).\nDo you want to update/restart it with current settings?",
                ).format(status),
            ):
                self._batch_queue[existing]["args"] = args
                self._batch_queue[existing]["status"] = "Pending"
            else:
                return
        else:
            self._batch_queue.append(
                {
                    "filename": os.path.basename(args["video_path"]),
                    "output": os.path.basename(target),
                    "status": "Pending",
                    "args": args,
                }
            )
        self._refresh_queue_ui()

    def _on_add_all(self) -> None:
        videos = [
            self.source_combo.itemText(i) for i in range(self.source_combo.count())
        ]
        if not videos:
            return
        added = 0
        skipped: list[str] = []
        existing_outputs = {j["args"]["output"] for j in self._batch_queue}
        for v in videos:
            out = str(argsmod.generate_output_path(v, self._current_settings()))
            if out in existing_outputs:
                skipped.append(
                    f"{os.path.basename(v)} ({i18n.tr('reason_dup_path', 'Duplicate Output path')})"
                )
                continue
            self._batch_queue.append(
                {
                    "filename": os.path.basename(v),
                    "output": os.path.basename(out),
                    "status": "Pending",
                    "args": {"video_path": v, "output": out},
                }
            )
            existing_outputs.add(out)
            added += 1
        self._refresh_queue_ui()
        if skipped:
            msg = i18n.tr(
                "msg_batch_report_summary", "Added {} videos.\nSkipped {} video(s):\n"
            ).format(added, len(skipped))
            msg += "\n".join(skipped[:10])
            if len(skipped) > 10:
                msg += "\n" + i18n.tr("msg_and_others", "...and others.")
            info(self, i18n.tr("title_batch_report", "Batch Report"), msg)

    def _on_queue_start(self) -> None:
        self._start_processing()

    def _on_run(self) -> None:
        if self.run_btn.text() == i18n.tr("btn_start_queue", "Start Queue"):
            self._start_processing()
            return
        if not self._video_path:
            return
        if self._worker is not None and self._worker.isRunning():
            self._append_log(
                i18n.tr("error_already_running", "Process is already running.\n")
            )
            return
        settings = self._current_settings()
        settings["_video_duration_ms"] = self._duration_ms
        args, errors = argsmod.build_args(
            self._video_path,
            settings,
            self.preview.crop_boxes,
            self.output_edit.text() or None,
        )
        if errors or args is None:
            self._append_log(i18n.tr("val_err_header", "Validation Errors:\n"))
            for err in errors:
                self._append_log(f"- {err}\n")
            return
        # add to queue then start
        target = args["output"]
        existing = next(
            (
                i
                for i, j in enumerate(self._batch_queue)
                if j["args"]["output"] == target
            ),
            None,
        )
        if existing is not None and self._batch_queue[existing]["status"] in (
            "Cancelled",
            "Error",
            "Completed",
        ):
            if ask_yes_no(
                self,
                i18n.tr("title_duplicate_job", "Duplicate Job"),
                i18n.tr(
                    "popup_duplicate_msg",
                    "A job for this output file already exists (Status: {}).\nDo you want to restart it with current settings?",
                ).format(self._batch_queue[existing]["status"]),
            ):
                self._batch_queue[existing]["args"] = args
                self._batch_queue[existing]["status"] = "Pending"
        elif existing is None:
            self._batch_queue.append(
                {
                    "filename": os.path.basename(args["video_path"]),
                    "output": os.path.basename(args["output"]),
                    "status": "Pending",
                    "args": args,
                }
            )
        self._refresh_queue_ui()
        self._start_processing()

    def _start_processing(self) -> None:
        pending = [j for j in self._batch_queue if j["status"] == "Pending"]
        if not pending:
            return
        if self._settings.get("prevent_system_sleep", True):
            self._set_system_awake(True)
        self._is_processing = True
        self._cancelled_by_user = False
        self._paused = False
        self.queue_tab.set_processing_state(True)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText(i18n.tr("btn_pause", "Pause"))
        self._run_next_job()

    def _run_next_job(self) -> None:
        job = next((j for j in self._batch_queue if j["status"] == "Pending"), None)
        if job is None:
            self._finish_processing()
            return
        job["status"] = "Processing"
        self._refresh_queue_ui()
        header = f"{'=' * 10} {i18n.tr('batch_processing_file', 'Processing')}: {os.path.basename(job['args']['video_path'])} {'=' * 10}\n"
        self._append_log("\n" + header)
        self._start_worker(job["args"])

    def _start_worker(self, args: dict[str, Any]) -> None:
        if not VIDEOCR_PATH:
            self._append_log(
                "\n"
                + i18n.tr(
                    "error_cli_not_found",
                    "Error: videocr-cli not found. Please check the path.\n",
                )
                + "\n"
            )
            self._finish_processing()
            return
        self._worker = CLIWorker(args)
        self._worker.signals.output.connect(self._append_log)
        self._worker.signals.progress.connect(self._on_progress)
        self._worker.signals.repacking.connect(self._on_repacking)
        self._worker.signals.process_started.connect(self._on_process_started)
        self._worker.signals.process_finished.connect(self._on_process_finished)
        self._worker.signals.fatal.connect(self._on_fatal)
        self._worker.signals.warning.connect(self._on_warning)
        self.progress_bar.setValue(0)
        self.status_label.setText(
            i18n.tr("status_starting", "Starting subtitle extraction...")
        )
        self._worker.start()

    def _on_process_started(self, pid: int) -> None:
        self._current_pid = pid
        self._update_taskbar(state="normal", progress=0)

    def _on_progress(self, upd: ProgressUpdate) -> None:
        if upd.text:
            self.status_label.setText(upd.text)
        if upd.percent is not None:
            self.progress_bar.setValue(int(upd.percent))
        if upd.eta:
            self.eta_label.setText(upd.eta)
        if upd.percent is not None:
            self._update_taskbar(progress=max(1, int(upd.percent)))

    def _on_repacking(self, line: str) -> None:
        self._append_log(line + "\n")

    def _on_fatal(self, message: str) -> None:
        self._append_log(
            f"\n--- FATAL ERROR ---\n"
            f"{i18n.tr('fatal_error_reason_1', 'Your system does not meet the hardware requirements.')}\n"
            f"{i18n.tr('fatal_error_reason_2', 'Reason:')} {message}\n"
        )

    def _on_warning(self, message: str) -> None:
        self._append_log(f"\n{i18n.tr('warning_header', 'WARNING:')} {message}\n")

    def _on_process_finished(self, success: bool, exit_code: int) -> None:
        job = next((j for j in self._batch_queue if j["status"] == "Processing"), None)
        if self._cancelled_by_user and job is not None:
            job["status"] = "Cancelled"
        elif job is not None:
            if success:
                job["status"] = "Completed"
                self._append_log(
                    "\n"
                    + i18n.tr(
                        "status_success", "Successfully generated subtitle file!\n"
                    )
                )
            else:
                job["status"] = "Error"
        self._refresh_queue_ui()
        self._worker = None
        self._update_taskbar(state="normal", progress=0)
        self.progress_bar.setValue(0)
        self.status_label.setText("")
        self.eta_label.setText("")
        if self._cancelled_by_user:
            self._finish_processing()
            return
        QTimer.singleShot(100, self._run_next_job)

    def _finish_processing(self) -> None:
        self._is_processing = False
        self._paused = False
        self._set_system_awake(False)
        self.queue_tab.set_processing_state(False)
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText(i18n.tr("btn_pause", "Pause"))
        msg = (
            i18n.tr("status_queue_cancelled", "Queue Cancelled")
            if self._cancelled_by_user
            else i18n.tr("status_queue_finished", "Queue Finished")
        )
        self._append_log("\n" + msg + "\n")
        self._update_taskbar(state="normal", progress=0)
        self._refresh_queue_ui()
        if not self._cancelled_by_user:
            self._maybe_send_notification()
            self._execute_post_action()

    def _maybe_send_notification(self) -> None:
        if not self._settings.get("--send_notification", True):
            return
        completed = [j for j in self._batch_queue if j["status"] == "Completed"]
        if not completed:
            return
        title = i18n.tr("notification_title", "Your Subtitle generation is done!")
        if len(completed) == 1:
            message = os.path.basename(completed[0]["args"]["output"])
        else:
            message = i18n.tr(
                "batch_finished_count", "Batch finished: {} files processed."
            ).format(len(completed))
        self._send_notification(title, message)

    def _send_notification(self, title: str, message: str) -> None:
        try:
            if sys.platform == "win32" and Notification is not None:
                toast = Notification(
                    app_id="VideOCR Recreated",
                    title=title,
                    msg=message,
                    icon=resources.notification_icon_path(),
                )
                toast.set_audio(audio.Default, loop=False)
                toast.show()
            elif notification is not None:
                notification.notify(
                    title=title,
                    message=message,
                    app_name="VideOCR Recreated",
                    app_icon=resources.notification_icon_path(),
                )
        except Exception as e:
            config.log_error(f"Failed to send notification: {e}")

    def _execute_post_action(self) -> None:
        idx = self.post_action_combo.currentIndex()
        if idx <= 0:
            return
        action_key = C.POST_ACTION_KEYS[idx]
        display = i18n.tr(action_key, C.DEFAULT_ACTION_TEXTS[action_key])
        proceed = CountdownDialog.run(self, display)
        if not proceed:
            self._append_log(
                "\n"
                + i18n.tr(
                    "log_action_cancelled", "Post-completion action cancelled by user."
                )
                + "\n"
            )
            return
        self._append_log(
            "\n"
            + i18n.tr("log_post_action", "Executing post-completion action: {}").format(
                display
            )
            + "\n"
        )
        if action_key == "action_shutdown":
            os.system(
                "shutdown /s /t 0" if sys.platform == "win32" else "systemctl poweroff"
            )
        elif action_key == "action_sleep":
            if sys.platform == "win32":
                import ctypes

                ctypes.windll.powrprof.SetSuspendState(False, False, False)
            else:
                os.system("systemctl suspend")
        elif action_key == "action_hibernate" and sys.platform == "win32":
            import ctypes

            ctypes.windll.powrprof.SetSuspendState(True, False, False)
        elif action_key == "action_lock" and sys.platform == "win32":
            import ctypes

            ctypes.windll.user32.LockWorkStation()

    # --- pause / cancel ---------------------------------------------------------
    def _on_pause(self) -> None:
        if self._worker is None or not self._worker.isRunning():
            return
        if self._paused:
            if self._worker.resume():
                self._paused = False
                self.pause_btn.setText(i18n.tr("btn_pause", "Pause"))
                self.queue_tab.set_pause_text(False)
                self._append_log(
                    "\n" + i18n.tr("status_resuming", "Resuming process...\n")
                )
                self._update_taskbar(state="normal")
                for j in self._batch_queue:
                    if j["status"] == "Paused":
                        j["status"] = "Processing"
                        break
                if self._settings.get("prevent_system_sleep", True):
                    self._set_system_awake(True)
        else:
            if self._worker.pause():
                self._paused = True
                self.pause_btn.setText(i18n.tr("btn_resume", "Resume"))
                self.queue_tab.set_pause_text(True)
                self._append_log(
                    "\n" + i18n.tr("status_pausing", "Pausing process...\n")
                )
                self._update_taskbar(state="paused")
                for j in self._batch_queue:
                    if j["status"] == "Processing":
                        j["status"] = "Paused"
                        break
                self._set_system_awake(False)
        self._refresh_queue_ui()

    def _on_resume(self) -> None:
        self._on_pause()

    def _on_cancel(self) -> None:
        if self._worker is None or not self._worker.isRunning():
            self._append_log(
                "\n"
                + i18n.tr(
                    "error_no_process_to_cancel",
                    "No process is currently running to cancel.\n",
                )
            )
            return
        self._cancelled_by_user = True
        self._append_log("\n" + i18n.tr("status_cancelling", "Cancelling process...\n"))
        self._worker.cancel()

    # --- queue row ops -----------------------------------------------------------
    def _on_move_up(self) -> None:
        rows = self.queue_tab.selected_rows()
        if rows and rows[0] > 0:
            for idx in sorted(rows):
                self._batch_queue[idx], self._batch_queue[idx - 1] = (
                    self._batch_queue[idx - 1],
                    self._batch_queue[idx],
                )
            self._refresh_queue_ui()
            self.queue_tab.select_rows([r - 1 for r in rows])

    def _on_move_down(self) -> None:
        rows = self.queue_tab.selected_rows()
        if rows and rows[-1] < len(self._batch_queue) - 1:
            for idx in sorted(rows, reverse=True):
                self._batch_queue[idx], self._batch_queue[idx + 1] = (
                    self._batch_queue[idx + 1],
                    self._batch_queue[idx],
                )
            self._refresh_queue_ui()
            self.queue_tab.select_rows([r + 1 for r in rows])

    def _on_reset(self) -> None:
        rows = self.queue_tab.selected_rows()
        changed = False
        for idx in rows:
            if self._batch_queue[idx]["status"] in ("Cancelled", "Error", "Completed"):
                self._batch_queue[idx]["status"] = "Pending"
                changed = True
        if changed:
            self._refresh_queue_ui()

    def _on_remove(self) -> None:
        rows = self.queue_tab.selected_rows()
        if not rows:
            return
        if any(
            self._batch_queue[i]["status"] in ("Processing", "Paused") for i in rows
        ):
            info(
                self,
                i18n.tr("title_error", "Error"),
                i18n.tr(
                    "popup_cannot_remove_running",
                    "The currently running or paused job cannot be removed.\nPlease stop or cancel the process first.",
                ),
            )
            return
        for i in sorted(rows, reverse=True):
            del self._batch_queue[i]
        self._refresh_queue_ui()

    def _on_clear(self) -> None:
        active = [
            j for j in self._batch_queue if j["status"] in ("Processing", "Paused")
        ]
        if active:
            self._batch_queue[:] = active
        else:
            self._batch_queue.clear()
        self._refresh_queue_ui()

    def _on_edit(self) -> None:
        rows = self.queue_tab.selected_rows()
        if len(rows) != 1:
            return
        idx = rows[0]
        job = self._batch_queue[idx]
        if job["status"] in ("Processing", "Paused"):
            info(
                self,
                i18n.tr("title_error", "Error"),
                i18n.tr(
                    "popup_cannot_edit_running",
                    "A job that is currently {} cannot be edited.\nPlease stop or cancel the process first.",
                ).format(job["status"]),
            )
            return
        v_path = job["args"]["video_path"]
        if not os.path.exists(v_path):
            info(
                self,
                i18n.tr("title_error", "Error"),
                i18n.tr("error_video_not_found", "Video file not found:\n{}").format(
                    v_path
                ),
            )
            return
        self.tabs.setCurrentIndex(0)
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItem(v_path)
        self.source_combo.blockSignals(False)
        self._load_video(v_path)
        # restore args into settings
        self._restore_args_to_ui(job["args"])
        del self._batch_queue[idx]
        self._refresh_queue_ui()

    def _restore_args_to_ui(self, args: dict[str, Any]) -> None:
        settings = self.settings_tab.read_settings()
        # Old queue args may carry a removed display name (e.g. the EasyOCR
        # DirectML option); map it to the engine that still uses it internally
        # BEFORE the internal-code lookup so it never falls through to the
        # PaddleOCR default.
        raw_engine = C.LEGACY_OCR_ENGINE_MAP.get(
            args.get("ocr_engine", "paddleocr"), args.get("ocr_engine", "paddleocr")
        )
        engine_map = {
            "google_lens": C.OCR_ENGINES[1],
            "easyocr_directml": C.OCR_ENGINES[2],
            "onnx_directml": C.OCR_ENGINES[2],
            "paddleocr": C.OCR_ENGINES[0],
        }
        engine = engine_map.get(raw_engine)
        if engine is None and raw_engine in C.OCR_ENGINES:
            # raw_engine is already a display name (from the legacy map).
            engine = raw_engine
        settings["ocr_engine"] = engine or C.OCR_ENGINES[0]

        lang_lookup = {
            "google_lens": C.lens_abbr_lookup,
            "easyocr_directml": C.easyocr_abbr_lookup,
            "onnx_directml": C.easyocr_abbr_lookup,
            "paddleocr": C.paddle_abbr_lookup,
        }
        # Use the mapped engine key (raw_engine) so a legacy display-name arg
        # resolves to the same language list as its mapped engine.
        lookup = lang_lookup.get(raw_engine, C.paddle_abbr_lookup)
        disp = next(
            (k for k, v in lookup.items() if v == args.get("lang", "en")),
            C.DEFAULT_SUBTITLE_LANGUAGE,
        )
        settings["subtitle_language"] = disp

        for arg_key, arg_val in args.items():
            if arg_key in (
                "ocr_engine",
                "lang",
                "video_path",
                "output",
                "send_notification",
                "allow_system_sleep",
                "subtitle_position",
            ):
                continue
            # Crop coordinates are restored separately via restore_box below;
            # writing them into settings would pollute the config file.
            if arg_key in (
                "crop_x",
                "crop_y",
                "crop_width",
                "crop_height",
                "crop_x2",
                "crop_y2",
                "crop_width2",
                "crop_height2",
            ):
                continue
            gui_key = f"--{arg_key}"
            if arg_key == "enable_label_detection":
                settings["enable_label_detection"] = bool(arg_val)
            elif gui_key == "--directml_performance_preset":
                settings[gui_key] = C.DIRECTML_PERFORMANCE_FROM_CLI.get(
                    str(arg_val), "Balanced (recommended)"
                )
            elif gui_key == "--directml_recognition_mode":
                settings[gui_key] = C.DIRECTML_RECOGNITION_FROM_CLI.get(
                    str(arg_val), "Stable Hybrid (recommended)"
                )
            elif gui_key == "--directml_frame_scan_mode":
                settings[gui_key] = C.DIRECTML_FRAME_SCAN_FROM_CLI.get(
                    str(arg_val), "CPU SSIM (compatible)"
                )
            elif gui_key == "--onnx_directml_tuning":
                settings[gui_key] = C.ONNX_DIRECTML_TUNING_FROM_CLI.get(
                    str(arg_val), "Balanced ONNX (recommended)"
                )
            else:
                settings[gui_key] = arg_val
        if args.get("output"):
            self.output_edit.setText(args["output"])
        self.settings_tab.populate(
            settings, sorted(i18n.get_available_languages().keys())
        )
        self._settings.update(settings)
        # refresh the process-tab engine/language/position combos from the restored settings
        self._populate_engine_lang_pos()
        config.save_settings(self._settings)

        # restore crop boxes
        boxes = []
        from .crop import restore_box

        w, h = self.preview.original_size
        rw, rh = w, h
        if "crop_x" in args:
            boxes.append(
                restore_box(
                    args["crop_x"],
                    args["crop_y"],
                    args["crop_width"],
                    args["crop_height"],
                    w,
                    h,
                    rw,
                    rh,
                )
            )
        if args.get("use_dual_zone") and "crop_x2" in args:
            boxes.append(
                restore_box(
                    args["crop_x2"],
                    args["crop_y2"],
                    args["crop_width2"],
                    args["crop_height2"],
                    w,
                    h,
                    rw,
                    rh,
                )
            )
        if boxes:
            self.preview.restore_crop_boxes(boxes)
            self.crop_label.setText(self.preview.crop_coords_text())

    # --- misc --------------------------------------------------------------------
    def _append_log(self, text: str) -> None:
        if not text:
            return
        if text.strip():
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.log_pane.appendPlainText(f"[{timestamp}] {text.rstrip()}")
        else:
            self.log_pane.appendPlainText("")

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        self._set_system_awake(False)
        # flush any pending debounced crop-box save before writing settings
        if self._crop_save_timer.isActive():
            self._crop_save_timer.stop()
            self._flush_crop_boxes()
        self.preview.handler.close()
        config.save_settings(self._settings)
        super().closeEvent(event)
