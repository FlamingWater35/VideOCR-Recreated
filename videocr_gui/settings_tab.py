"""Advanced Settings tab: OCR settings + DirectML controls + VideOCR settings."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from . import constants as C
from . import i18n
from .widgets import WheelGuardComboBox

# Map DirectML combo widget key -> {English option name: i18n key}.
# Settings/CLI store the English canonical option name; the UI shows the
# localized text. Kept here (not in constants.py) so the CLI round-trip and
# settings persistence stay on the stable English strings.
# The i18n key per option is derived from constants' CLI value, so the two
# "Manual Grid Size" entries (presets vs onnx) get distinct keys.
DML_OPTION_KEYS: dict[str, dict[str, str]] = {
    "--directml_performance_preset": {
        name: f"dml_preset_{cli}" for name, cli in C.DIRECTML_PERFORMANCE_PRESETS
    },
    "--directml_recognition_mode": {
        name: f"dml_recognition_{cli}" for name, cli in C.DIRECTML_RECOGNITION_MODES
    },
    "--directml_frame_scan_mode": {
        name: f"dml_frame_scan_{cli}" for name, cli in C.DIRECTML_FRAME_SCAN_MODES
    },
    "--onnx_directml_tuning": {
        name: f"dml_onnx_{cli}" for name, cli in C.ONNX_DIRECTML_TUNING_MODES
    },
}

# Combo widget keys whose options are translatable via DML_OPTION_KEYS.
DML_OPTION_COMBOS = (
    "--directml_performance_preset",
    "--directml_recognition_mode",
    "--directml_frame_scan_mode",
    "--onnx_directml_tuning",
)

# Widgets disabled while "Enable GPU Usage" is unchecked.
DML_DEPENDENT_WIDGETS = (
    "-DML_ADAPTER_COMBO-",
    "--directml_performance_preset",
    "--directml_recognition_mode",
    "--directml_frame_scan_mode",
    "--onnx_directml_tuning",
    "--directml_grid_max_width",
    "--directml_grid_max_height",
)

# Widgets disabled while label detection is disabled.
LABEL_DEPENDENT_WIDGETS = (
    "--label_time_start",
    "--label_time_end",
    "--label_ocr_image_max_width",
    "--label_min_display_duration",
    "--label_min_confirmation_frames",
    "--label_reappear_merge_gap",
    "--label_filter_single_char",
)


def _translated_label(key: str, fallback: str) -> str:
    """Returns the translated text for a label key, falling back to the default."""
    # Use i18n.tr() so missing keys fall back to the English language file
    # first (consistent with the rest of the UI), then to the fallback text.
    return i18n.tr(key, fallback)


class SettingsTab(QWidget):
    """Advanced settings. Emits settings_changed(list_of_keys) on any change."""

    settings_changed = Signal(list)
    ui_language_changed = Signal(str)      # native name
    scaling_changed = Signal(str)
    directml_gpu_changed = Signal(str)     # numeric index
    restart_requested = Signal()
    info_requested = Signal()
    help_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._widgets: dict[str, QWidget] = {}
        self._labels: list[QLabel] = []
        self._tooltips: dict[QWidget, str] = {}  # widget -> tip key
        self._section_boxes: list[tuple[QGroupBox, str, str]] = []
        # combo key -> list of internal values parallel to the combo's items
        self._combo_internal: dict[str, list[str]] = {}
        self._block_signals = False
        self._settings: dict[str, Any] = {}
        self._build_ui()
        self._retranslate()

    # --- UI construction ----------------------------------------------------
    def _build_ui(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        container = QWidget()
        scroll.setWidget(container)

        root = QVBoxLayout(container)
        root.setSpacing(12)

        # OCR settings
        ocr_box = QGroupBox()
        ocr_box.setObjectName("sectionBox")
        self._section_boxes.append((ocr_box, "lbl_ocr_settings", "OCR Settings:"))
        form = QFormLayout(ocr_box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._add_line(form, "--time_start", "lbl_time_start", "Start Time (e.g., 0:00 or 1:23:45):", "0:00", "tip_time_start")
        self._add_line(form, "--time_end", "lbl_time_end", "End Time (e.g., 0:10 or 2:34:56):", "", "tip_time_end")
        self._add_line(form, "--conf_threshold", "lbl_conf_threshold", "Confidence Threshold (0-100):", "75", "tip_conf_threshold")
        self._add_line(form, "--sim_threshold", "lbl_sim_threshold", "Similarity Threshold (0-100):", "80", "tip_sim_threshold")
        self._add_line(form, "--max_merge_gap", "lbl_merge_gap", "Max Merge Gap (seconds):", "0.1", "tip_merge_gap")
        self._add_line(form, "--brightness_threshold", "lbl_brightness", "Brightness Threshold (0-255):", "", "tip_brightness")
        self._add_line(form, "--ssim_threshold", "lbl_ssim", "SSIM Threshold (0-100):", "92", "tip_ssim")
        self._add_line(form, "--ocr_image_max_width", "lbl_ocr_width", "Max OCR Image Width (pixel):", "720", "tip_ocr_width")
        self._add_line(form, "--frames_to_skip", "lbl_frames_skip", "Frames to Skip:", "1", "tip_frames_skip")
        self._add_line(form, "--min_subtitle_duration", "lbl_min_duration", "Minimum Subtitle Duration (seconds):", "0.2", "tip_min_duration")
        root.addWidget(ocr_box)

        # DirectML box
        dml_box = QGroupBox()
        dml_box.setObjectName("sectionBox")
        self._section_boxes.append((dml_box, "lbl_dml_section", "DirectML (AMD GPU):"))
        dml_form = QFormLayout(dml_box)
        dml_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        dml_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._add_check(dml_form, "--use_gpu", "chk_use_gpu", "Enable GPU Usage", "tip_use_gpu", True)

        gpu_row = QHBoxLayout()
        self._widgets["-DML_ADAPTER_COMBO-"] = WheelGuardComboBox()
        self._widgets["-DML_ADAPTER_COMBO-"].setMinimumWidth(300)
        self._widgets["-DML_ADAPTER_COMBO-"].setToolTip(i18n.tr("tip_dml_device", ""))
        self._tooltips[self._widgets["-DML_ADAPTER_COMBO-"]] = "tip_dml_device"
        self._widgets["-DML_ADAPTER_COMBO-"].currentIndexChanged.connect(
            lambda _i: self._on_changed(["-DML_ADAPTER_COMBO-"])
        )
        self.refresh_btn = QPushButton(i18n.tr("btn_refresh", "Refresh"))
        self.refresh_btn.clicked.connect(self._refresh_directml)
        gpu_row.addWidget(self._widgets["-DML_ADAPTER_COMBO-"])
        gpu_row.addWidget(self.refresh_btn)
        gpu_row.addStretch(1)
        dml_form.addRow(self._lbl("lbl_dml_device", "DirectML GPU:"), gpu_row)

        self._add_combo(dml_form, "--directml_performance_preset", "lbl_dml_preset",
                        "AMD Performance Preset:", C.DIRECTML_PERFORMANCE_DISPLAY, "tip_dml_preset")
        self._add_combo(dml_form, "--directml_recognition_mode", "lbl_dml_recognition",
                        "DirectML Recognition Mode:", C.DIRECTML_RECOGNITION_DISPLAY, "tip_dml_recognition")
        self._add_combo(dml_form, "--directml_frame_scan_mode", "lbl_dml_frame_scan",
                        "AMD Frame Scan Mode:", C.DIRECTML_FRAME_SCAN_DISPLAY, "tip_dml_frame_scan")
        self._add_combo(dml_form, "--onnx_directml_tuning", "lbl_onnx_tuning",
                        "ONNX DirectML Tuning:", C.ONNX_DIRECTML_TUNING_DISPLAY, "tip_onnx_tuning")
        self._add_line(dml_form, "--directml_grid_max_width", "lbl_dml_grid_w", "DirectML Grid Max Width:", "2400", "tip_dml_grid_w")
        self._add_line(dml_form, "--directml_grid_max_height", "lbl_dml_grid_h", "DirectML Grid Max Height:", "2400", "tip_dml_grid_h")
        root.addWidget(dml_box)

        # OCR mode toggles
        mode_box = QGroupBox()
        mode_box.setObjectName("sectionBox")
        self._section_boxes.append((mode_box, "lbl_ocr_mode", "OCR Mode:"))
        mode_form = QFormLayout(mode_box)
        mode_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        mode_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._add_check(mode_form, "--use_fullframe", "chk_full_frame", "Use Full Frame OCR", "tip_full_frame", False)
        self._add_check(mode_form, "--use_dual_zone", "chk_dual_zone", "Enable Dual Zone OCR", "tip_dual_zone", False)
        self._add_check(mode_form, "enable_subtitle_alignment", "chk_enable_subtitle_alignment",
                        "Enable Subtitle Alignment", "tip_enable_subtitle_alignment", False)
        self._add_combo(mode_form, "--subtitle_alignment", "lbl_subtitle_alignment1",
                        "Zone 1 Alignment:", C.SUBTITLE_ALIGNMENT_LIST, "tip_subtitle_alignment1",
                        internal_values=[v for _, v in C.SUBTITLE_ALIGNMENT_LIST])
        self._add_combo(mode_form, "--subtitle_alignment2", "lbl_subtitle_alignment2",
                        "Zone 2 Alignment:", C.SUBTITLE_ALIGNMENT_LIST, "tip_subtitle_alignment2",
                        internal_values=[v for _, v in C.SUBTITLE_ALIGNMENT_LIST])
        self._add_check(mode_form, "--use_angle_cls", "chk_angle_cls", "Enable Angle Classification", "tip_angle_cls", False)
        self._add_check(mode_form, "--post_processing", "chk_post_processing", "Enable Post Processing", "tip_post_processing", False)
        self._add_check(mode_form, "--normalize_to_simplified_chinese", "chk_normalize_chinese",
                        "Normalize Traditional to Simplified Chinese", "tip_normalize_chinese", True)
        self._add_check(mode_form, "--use_server_model", "chk_server_model", "Use Server Model", "tip_server_model", False)
        root.addWidget(mode_box)

        # Label detection settings (text outside the subtitle crop area, e.g. names)
        label_box = QGroupBox()
        label_box.setObjectName("sectionBox")
        self._section_boxes.append((label_box, "lbl_label_section", "Label Detection:"))
        label_form = QFormLayout(label_box)
        label_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        label_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._add_check(label_form, "enable_label_detection", "chk_label_detection",
                        "Enable Label Detection (detect text outside the subtitle crop)", "tip_label_detection", False)
        self._add_line(label_form, "--label_time_start", "lbl_label_time_start",
                       "Label Start Time (e.g., 0:00 or 1:23:45):", "", "tip_label_time_start")
        self._add_line(label_form, "--label_time_end", "lbl_label_time_end",
                       "Label End Time (e.g., 0:10 or 2:34:56):", "", "tip_label_time_end")
        self._add_line(label_form, "--label_ocr_image_max_width", "lbl_label_ocr_width",
                       "Label Max OCR Image Width (pixel):", "720", "tip_label_ocr_width")
        self._add_line(label_form, "--label_min_display_duration", "lbl_label_min_duration",
                       "Label Minimum Display Duration (seconds):", "1.0", "tip_label_min_duration")
        self._add_line(label_form, "--label_min_confirmation_frames", "lbl_label_conf_frames",
                       "Label Minimum Confirmation Frames:", "2", "tip_label_conf_frames")
        self._add_line(label_form, "--label_reappear_merge_gap", "lbl_label_reappear_gap",
                       "Label Reappear Merge Gap (seconds):", "2.0", "tip_label_reappear_gap")
        self._add_check(label_form, "--label_filter_single_char", "chk_label_filter_single",
                        "Filter Single-Character Label Noise", "tip_label_filter_single", True)
        root.addWidget(label_box)

        # VideOCR Recreated settings
        vo_box = QGroupBox()
        vo_box.setObjectName("sectionBox")
        self._section_boxes.append((vo_box, "lbl_videocr_settings", "VideOCR Recreated Settings:"))
        vo_form = QFormLayout(vo_box)
        vo_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        vo_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._widgets["-UI_LANG_COMBO-"] = WheelGuardComboBox()
        self._widgets["-UI_LANG_COMBO-"].currentIndexChanged.connect(
            lambda _i: self._on_changed(["-UI_LANG_COMBO-"])
        )
        vo_form.addRow(self._lbl("lbl_ui_lang", "UI Language:"), self._widgets["-UI_LANG_COMBO-"])

        self._widgets["gui_scaling"] = WheelGuardComboBox()
        self._widgets["gui_scaling"].currentIndexChanged.connect(
            lambda _i: self._on_changed(["gui_scaling"])
        )
        vo_form.addRow(self._lbl("lbl_gui_scaling", "GUI Scaling:"), self._widgets["gui_scaling"])

        self._add_check(vo_form, "--save_crop_box", "chk_save_crop_box", "Save Crop Box Selection", "tip_save_crop_box", True)
        self._add_check(vo_form, "--save_in_video_dir", "chk_save_in_video_dir", "Save SRT in Video Directory", "tip_save_in_video_dir", True)

        out_row = QHBoxLayout()
        self._widgets["--default_output_dir"] = QLineEdit()
        self.browse_output_btn = QPushButton(i18n.tr("btn_browse_folder", "Open Folder..."))
        out_row.addWidget(self._widgets["--default_output_dir"])
        out_row.addWidget(self.browse_output_btn)
        vo_form.addRow(self._lbl("lbl_output_dir", "Output Directory:"), out_row)

        self._add_line(vo_form, "--keyboard_seek_step", "lbl_seek_step", "Keyboard Seek Step (seconds):", "1", "tip_seek_step")
        self._add_check(vo_form, "--send_notification", "chk_send_notification", "Send Notification", "tip_send_notification", True)
        self._add_check(vo_form, "prevent_system_sleep", "chk_prevent_sleep", "Prevent System Sleep", "tip_prevent_sleep", True)
        root.addWidget(vo_box)

        root.addStretch(1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # wire browse
        self.browse_output_btn.clicked.connect(self._browse_output_dir)

    def _lbl(self, key: str, fallback: str) -> QLabel:
        label = QLabel(_translated_label(key, fallback))
        label.setObjectName("formLabel")
        label.setProperty("langKey", key)
        label.setProperty("fallbackText", fallback)
        self._labels.append(label)
        return label

    def _add_line(self, form: QFormLayout, key: str, label_key: str, fallback_label: str, default: str,
                  tooltip_key: str | None = None) -> QLineEdit:
        edit = QLineEdit(default)
        edit.setObjectName(key.lstrip("-").replace("_", "-"))
        self._widgets[key] = edit
        if tooltip_key:
            edit.setToolTip(i18n.tr(tooltip_key, ""))
            self._tooltips[edit] = tooltip_key
        form.addRow(self._lbl(label_key, fallback_label), edit)
        edit.textChanged.connect(lambda _t, k=key: self._on_changed([k]))
        return edit

    def _add_combo(self, form: QFormLayout, key: str, label_key: str, fallback_label: str,
                   options: list[tuple[str, str]] | list[str], tooltip_key: str | None = None,
                   internal_values: list[str] | None = None) -> QComboBox:
        combo = WheelGuardComboBox()
        if options and isinstance(options[0], tuple):
            display = [_translated_label(lang_key, name) for lang_key, name in options]
            combo.addItems(display)
        else:
            display = [self._localized_option(key, str(o)) for o in options]
            combo.addItems(display)
        self._widgets[key] = combo
        if internal_values is not None:
            self._combo_internal[key] = internal_values
        if tooltip_key:
            combo.setToolTip(i18n.tr(tooltip_key, ""))
            self._tooltips[combo] = tooltip_key
        form.addRow(self._lbl(label_key, fallback_label), combo)
        combo.currentIndexChanged.connect(lambda _i, k=key: self._on_changed([k]))
        return combo

    @staticmethod
    def _localized_option(combo_key: str, option: str) -> str:
        """Localizes a DirectML option's display text via its i18n key."""
        if combo_key in DML_OPTION_COMBOS:
            i18n_key = DML_OPTION_KEYS.get(combo_key, {}).get(option)
            if i18n_key:
                return i18n.tr(i18n_key, option)
        return option

    @staticmethod
    def _dml_english_options(combo_key: str) -> list[str]:
        """English canonical option names for a DirectML combo key."""
        source = {
            "--directml_performance_preset": C.DIRECTML_PERFORMANCE_PRESETS,
            "--directml_recognition_mode": C.DIRECTML_RECOGNITION_MODES,
            "--directml_frame_scan_mode": C.DIRECTML_FRAME_SCAN_MODES,
            "--onnx_directml_tuning": C.ONNX_DIRECTML_TUNING_MODES,
        }.get(combo_key, [])
        return [name for name, _ in source]

    def _add_check(self, form: QFormLayout, key: str, label_key: str, fallback_label: str,
                   tooltip_key: str, default: bool) -> QCheckBox:
        cb = QCheckBox(_translated_label(label_key, fallback_label))
        cb.setChecked(default)
        cb.setToolTip(i18n.tr(tooltip_key, ""))
        self._tooltips[cb] = tooltip_key
        cb.setProperty("langKey", label_key)
        cb.setProperty("fallbackText", fallback_label)
        self._widgets[key] = cb
        form.addRow(cb)
        cb.toggled.connect(lambda _v, k=key: self._on_changed([k]))
        return cb

    # --- public API ---------------------------------------------------------
    def populate(self, settings: dict[str, Any], ui_languages: list[str]) -> None:
        """Fills widgets from settings; called at startup and on language switch."""
        self._block_signals = True
        self._settings = dict(settings)
        try:
            for key, widget in self._widgets.items():
                if key not in settings:
                    continue
                value = settings[key]
                if isinstance(widget, QComboBox):
                    if key in self._combo_internal:
                        internal_list = self._combo_internal[key]
                        idx = internal_list.index(str(value)) if str(value) in internal_list else 0
                        widget.setCurrentIndex(idx)
                    elif key in DML_OPTION_COMBOS:
                        # settings store the English canonical value; the combo
                        # displays localized text, so match by option index
                        names = self._dml_english_options(key)
                        idx = names.index(str(value)) if str(value) in names else 0
                        widget.setCurrentIndex(idx)
                    else:
                        idx = widget.findText(str(value))
                        if idx >= 0:
                            widget.setCurrentIndex(idx)
                        elif isinstance(value, str) and value:
                            widget.addItem(value)
                            widget.setCurrentIndex(widget.count() - 1)
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(value))

            lang_combo = self._widgets["-UI_LANG_COMBO-"]
            lang_combo.clear()
            lang_combo.addItems(ui_languages)
            code_to_native = {v: k for k, v in i18n.get_available_languages().items()}
            native = code_to_native.get(settings.get("--language", "en"), "English")
            idx = lang_combo.findText(native)
            if idx >= 0:
                lang_combo.setCurrentIndex(idx)

            scale_combo = self._widgets["gui_scaling"]
            scale_combo.clear()
            scale_combo.addItems([i18n.tr(k, d) for k, d in C.GUI_SCALING_LIST])
            saved = str(settings.get("gui_scaling", C.DEFAULT_GUI_SCALING))
            # settings store the internal key (system_default/scale_1_0/...)
            internal_keys = [k for k, _ in C.GUI_SCALING_LIST]
            idx = internal_keys.index(saved) if saved in internal_keys else 0
            scale_combo.setCurrentIndex(idx)

            self._update_dependent_states()
        finally:
            self._block_signals = False

    def _update_dependent_states(self) -> None:
        """Enables/disables dependent widgets based on master toggles.

        DirectML controls are disabled while "Enable GPU Usage" is unchecked;
        label-detection controls are disabled while label detection is disabled.
        """
        use_gpu = self._widgets.get("--use_gpu")
        gpu_enabled = isinstance(use_gpu, QCheckBox) and use_gpu.isChecked()
        for key in DML_DEPENDENT_WIDGETS:
            widget = self._widgets.get(key)
            if isinstance(widget, (QComboBox, QLineEdit, QPushButton)):
                widget.setEnabled(gpu_enabled)
        # The refresh button next to the adapter combo follows the same state.
        refresh_btn = getattr(self, "refresh_btn", None)
        if isinstance(refresh_btn, QPushButton):
            refresh_btn.setEnabled(gpu_enabled)

        label_check = self._widgets.get("enable_label_detection")
        labels_enabled = isinstance(label_check, QCheckBox) and label_check.isChecked()
        for key in LABEL_DEPENDENT_WIDGETS:
            widget = self._widgets.get(key)
            if isinstance(widget, (QComboBox, QLineEdit)):
                widget.setEnabled(labels_enabled)

    def read_settings(self) -> dict[str, Any]:
        """Reads current widget values back into a settings dict.

        Combos with a parallel internal-value list (e.g. subtitle alignment)
        return the internal value; all other widgets return their raw value.
        """
        settings = dict(self._settings)
        for key, widget in self._widgets.items():
            if isinstance(widget, QComboBox):
                if key in self._combo_internal:
                    internal_list = self._combo_internal[key]
                    idx = widget.currentIndex()
                    settings[key] = internal_list[idx] if 0 <= idx < len(internal_list) else (internal_list[0] if internal_list else "")
                elif key == "gui_scaling":
                    # combo shows localized text; persist the internal key
                    internal_keys = [k for k, _ in C.GUI_SCALING_LIST]
                    idx = widget.currentIndex()
                    settings[key] = internal_keys[idx] if 0 <= idx < len(internal_keys) else (internal_keys[0] if internal_keys else "")
                elif key in DML_OPTION_COMBOS:
                    # combo shows localized text; persist the English canonical value
                    names = self._dml_english_options(key)
                    idx = widget.currentIndex()
                    settings[key] = names[idx] if 0 <= idx < len(names) else (names[0] if names else "")
                else:
                    settings[key] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                settings[key] = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                settings[key] = widget.text()
        return settings

    def _retranslate(self) -> None:
        """Re-translates labels, tooltips and combo entries after a language change."""
        # QLabels created by _lbl()
        for label in self._labels:
            key = label.property("langKey")
            fallback = label.property("fallbackText")
            if key:
                label.setText(_translated_label(key, fallback or label.text()))

        # Checkboxes (in _widgets, carry langKey/fallbackText)
        for _key, widget in self._widgets.items():
            if isinstance(widget, QCheckBox):
                prop = widget.property("langKey")
                fallback = widget.property("fallbackText")
                if prop:
                    widget.setText(_translated_label(prop, fallback or widget.text()))

        # Alignment combos: rebuild translated items preserving internal selection
        for key in ("--subtitle_alignment", "--subtitle_alignment2"):
            widget = self._widgets.get(key)
            if not isinstance(widget, QComboBox):
                continue
            internal_list = self._combo_internal.get(key, [])
            idx = widget.currentIndex()
            cur_internal = internal_list[idx] if 0 <= idx < len(internal_list) else (internal_list[0] if internal_list else "")
            display = [_translated_label(lang_key, name) for lang_key, name in C.SUBTITLE_ALIGNMENT_LIST]
            widget.blockSignals(True)
            widget.clear()
            widget.addItems(display)
            if cur_internal in internal_list:
                widget.setCurrentIndex(internal_list.index(cur_internal))
            widget.blockSignals(False)

        # DirectML option combos: rebuild localized items preserving selection
        for key in DML_OPTION_COMBOS:
            widget = self._widgets.get(key)
            if not isinstance(widget, QComboBox):
                continue
            idx = widget.currentIndex()
            display = [self._localized_option(key, name) for name in self._dml_english_options(key)]
            widget.blockSignals(True)
            widget.clear()
            widget.addItems(display)
            if 0 <= idx < widget.count():
                widget.setCurrentIndex(idx)
            widget.blockSignals(False)

        # GUI scaling combo: rebuild localized items preserving selection
        scale_combo = self._widgets.get("gui_scaling")
        if isinstance(scale_combo, QComboBox):
            idx = scale_combo.currentIndex()
            display = [i18n.tr(k, d) for k, d in C.GUI_SCALING_LIST]
            scale_combo.blockSignals(True)
            scale_combo.clear()
            scale_combo.addItems(display)
            if 0 <= idx < scale_combo.count():
                scale_combo.setCurrentIndex(idx)
            scale_combo.blockSignals(False)

        # Section titles
        for box, key, fallback in self._section_boxes:
            box.setTitle(i18n.tr(key, fallback))

        # DirectML refresh button
        self.refresh_btn.setText(i18n.tr("btn_refresh", "Refresh"))

        # Output directory browse button
        self.browse_output_btn.setText(i18n.tr("btn_browse_folder", "Open Folder..."))

        # Tooltips
        for widget, tip_key in self._tooltips.items():
            widget.setToolTip(i18n.tr(tip_key, ""))

    def retranslate(self) -> None:
        """Public re-translation hook (called by the main window on language change)."""
        self._retranslate()

    # --- slots --------------------------------------------------------------
    def _on_changed(self, keys: list[str]) -> None:
        if self._block_signals:
            return
        key = keys[0]
        if key == "--directml_performance_preset":
            self._apply_preset_grid()
        elif key == "--onnx_directml_tuning":
            self._apply_onnx_grid()
        elif key == "-UI_LANG_COMBO-":
            # emit the language code, not the localized native name, so the
            # main window can apply the switch unambiguously
            native = self._widgets[key].currentText()
            code = i18n.get_available_languages().get(native)
            self.ui_language_changed.emit(code or native)
            return
        elif key == "gui_scaling":
            # emit the internal key, not the localized display text
            internal_keys = [k for k, _ in C.GUI_SCALING_LIST]
            idx = self._widgets[key].currentIndex()
            internal = internal_keys[idx] if 0 <= idx < len(internal_keys) else (internal_keys[0] if internal_keys else "")
            self.scaling_changed.emit(internal)
            return
        elif key == "-DML_ADAPTER_COMBO-":
            self.directml_gpu_changed.emit(self._directml_index())
            return
        elif key == "--use_dual_zone":
            self.settings_changed.emit([key, "--use_dual_zone"])
            return
        elif key == "--use_gpu":
            # Re-evaluate DirectML controls' enabled state when the master
            # toggle flips, then emit the change.
            self._update_dependent_states()
        elif key == "enable_label_detection":
            # Re-evaluate label-detection controls' enabled state when the
            # master toggle flips, then emit the change.
            self._update_dependent_states()
        self.settings_changed.emit(keys)

    def _directml_index(self) -> str:
        from . import directml

        return directml.option_to_index(self._widgets["-DML_ADAPTER_COMBO-"].currentText())

    def _apply_preset_grid(self) -> None:
        preset = self._dml_english_options("--directml_performance_preset")[
            self._widgets["--directml_performance_preset"].currentIndex()
        ]
        cli = C.DIRECTML_PERFORMANCE_TO_CLI.get(preset, "balanced")
        grid = {"compatibility": "1600", "balanced": "2400", "max": "4096"}.get(cli)
        if grid:
            self._widgets["--directml_grid_max_width"].setText(grid)
            self._widgets["--directml_grid_max_height"].setText(grid)

    def _apply_onnx_grid(self) -> None:
        tuning = self._dml_english_options("--onnx_directml_tuning")[
            self._widgets["--onnx_directml_tuning"].currentIndex()
        ]
        cli = C.ONNX_DIRECTML_TUNING_TO_CLI.get(tuning, "balanced")
        engine = self._widgets.get("ocr_engine", None)
        engine_text = ""
        if isinstance(engine, QComboBox):
            engine_text = engine.currentText()
        if "ONNX Runtime DirectML" not in engine_text:
            return
        grid = {"low_vram": "1600", "balanced": "2048", "max": "3072"}.get(cli)
        if grid:
            self._widgets["--directml_grid_max_width"].setText(grid)
            self._widgets["--directml_grid_max_height"].setText(grid)

    def _refresh_directml(self) -> None:
        from . import directml

        combo = self._widgets["-DML_ADAPTER_COMBO-"]
        current = combo.currentText()
        options = directml.get_adapter_options()
        idx = directml.option_to_index(current)
        selected = directml.index_to_option(options, idx)
        if selected not in options:
            options.append(selected)
        self._block_signals = True
        try:
            combo.clear()
            combo.addItems(options)
            combo.setCurrentText(selected)
        finally:
            self._block_signals = False
        self.directml_gpu_changed.emit(idx)

    def _browse_output_dir(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        folder = QFileDialog.getExistingDirectory(self, i18n.tr("lbl_output_dir", "Output Directory:"))
        if folder:
            self._widgets["--default_output_dir"].setText(folder)
            self.settings_changed.emit(["--default_output_dir"])


class DirectMLTab(SettingsTab):
    """Alias retained for clarity; SettingsTab covers the DirectML controls."""
