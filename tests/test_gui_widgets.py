"""Tests for GUI widgets: style, queue tab, settings tab, dialogs, resources.

All Qt tests run offscreen via pytest-qt (QT_QPA_PLATFORM=offscreen is set in
conftest.py).
"""

from __future__ import annotations

import re

import pytest
from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit

from videocr_gui import dialogs, i18n, resources, style
from videocr_gui.config import get_default_settings
from videocr_gui.queue_tab import STATUS_TRANSLATIONS, QueueTab
from videocr_gui.settings_tab import (
    DML_DEPENDENT_WIDGETS,
    LABEL_DEPENDENT_WIDGETS,
    SettingsTab,
)
from videocr_gui.style import PALETTE


@pytest.fixture(autouse=True)
def _english():
    i18n.load_language("en")


class TestStyle:
    def test_palette_keys_complete(self):
        expected = {
            "bg_deep", "bg_surface", "bg_raised", "bg_hover", "border",
            "border_strong", "accent", "accent_hi", "accent_deep", "amber",
            "text_emphasis", "danger", "ok", "text", "text_dim", "text_faint",
        }
        assert set(PALETTE) == expected

    @pytest.mark.parametrize("key", sorted(PALETTE))
    def test_palette_values_are_hex(self, key):
        assert re.fullmatch(r"#[0-9a-fA-F]{6}", PALETTE[key]), key

    def test_qss_nonempty_and_has_named_buttons(self):
        assert len(style.QSS) > 1000
        assert "#primaryButton" in style.QSS
        assert "#dangerButton" in style.QSS

    def test_apply_sets_stylesheet(self, qapp):
        original = qapp.styleSheet()
        try:
            style.apply(qapp)
            assert qapp.styleSheet() == style.QSS
        finally:
            qapp.setStyleSheet(original)

    @pytest.mark.parametrize(
        ("ass", "html"),
        [
            ("&H00FFFFFF", "#FFFFFF"),
            ("&H00000000", "#000000"),
            ("&H00FF0000", "#0000FF"),  # ASS BBGGRR: blue=FF → html blue
            ("&H0000FF00", "#00FF00"),
            ("&H00123456", "#563412"),
        ],
    )
    def test_ass_color_to_html(self, ass, html):
        assert style.ass_color_to_html(ass).upper() == html.upper()

    @pytest.mark.parametrize(
        ("html", "ass"),
        [("#FFFFFF", "&H00FFFFFF"), ("#0000FF", "&H00FF0000"), ("#00FF00", "&H0000FF00")],
    )
    def test_html_to_ass_color(self, html, ass):
        assert style.html_to_ass_color(html) == ass

    def test_color_round_trip(self):
        for ass in ("&H00FFFFFF", "&H00FF0000", "&H00123456"):
            assert style.html_to_ass_color(style.ass_color_to_html(ass)) == ass

    @pytest.mark.parametrize("bad", ["", "red", "&H00", "#GGG"])
    def test_invalid_ass_color_falls_back_black(self, bad):
        assert style.ass_color_to_html(bad) == "#000000"
        assert style.html_to_ass_color(bad) == "&H00000000"


class TestQueueTab:
    @pytest.fixture
    def tab(self, qtbot):
        t = QueueTab()
        qtbot.addWidget(t)
        return t

    def test_initial_state(self, tab):
        assert tab.table.columnCount() == 3
        assert tab.table.rowCount() == 0
        assert tab.jobs() == []
        assert not tab.stop_btn.isEnabled()
        assert not tab.pause_btn.isEnabled()
        assert tab.start_btn.isEnabled()

    def test_set_jobs_round_trip(self, tab):
        jobs = [
            {"filename": "a.mp4", "output": "a.srt", "status": "Pending"},
            {"filename": "b.mkv", "output": "b.srt", "status": "Completed"},
        ]
        tab.set_jobs(jobs)
        assert tab.jobs() == jobs
        assert tab.table.rowCount() == 2
        assert tab.table.item(0, 0).text() == "a.mp4"
        assert tab.table.item(1, 1).text() == "b.srt"

    def test_status_stored_raw_and_colored(self, tab):
        from PySide6.QtCore import Qt

        tab.set_jobs([{"filename": "a.mp4", "output": "a.srt", "status": "Processing"}])
        item = tab.table.item(0, 2)
        assert item.data(Qt.ItemDataRole.UserRole) == "Processing"
        assert item.foreground().color().name() == PALETTE["accent"]

    @pytest.mark.parametrize(
        ("status", "color"),
        [
            ("Pending", PALETTE["text_dim"]),
            ("Processing", PALETTE["accent"]),
            ("Paused", PALETTE["amber"]),
            ("Completed", PALETTE["ok"]),
            ("Cancelled", PALETTE["danger"]),
            ("Error", PALETTE["danger"]),
            ("UnknownStatus", PALETTE["text_dim"]),
        ],
    )
    def test_status_colors(self, status, color):
        assert QueueTab._status_color(status) == color

    def test_status_translations_cover_all_statuses(self):
        assert set(STATUS_TRANSLATIONS) == {
            "Pending", "Processing", "Completed", "Cancelled", "Error", "Paused",
        }

    def test_select_and_selected_rows(self, tab):
        tab.set_jobs([
            {"filename": "a.mp4", "output": "a.srt", "status": "Pending"},
            {"filename": "b.mp4", "output": "b.srt", "status": "Pending"},
            {"filename": "c.mp4", "output": "c.srt", "status": "Pending"},
        ])
        tab.select_rows([2, 0])
        assert tab.selected_rows() == [0, 2]

    def test_processing_state_toggles_buttons(self, tab):
        tab.set_processing_state(True)
        assert not tab.start_btn.isEnabled()
        assert tab.stop_btn.isEnabled()
        assert tab.pause_btn.isEnabled()
        tab.set_processing_state(False)
        assert tab.start_btn.isEnabled()
        assert not tab.stop_btn.isEnabled()

    def test_pause_text_toggle_emits(self, tab, qtbot):
        tab.set_processing_state(True)  # pause button only active while processing
        tab.set_pause_text(paused=False)
        assert tab.pause_btn.text() == i18n.tr("btn_pause", "Pause")
        with qtbot.waitSignal(tab.pause_requested, timeout=1000):
            tab.pause_btn.click()
        tab.set_pause_text(paused=True)
        assert tab.pause_btn.text() == i18n.tr("btn_resume", "Resume")
        with qtbot.waitSignal(tab.resume_requested, timeout=1000):
            tab.pause_btn.click()

    def test_buttons_emit_signals(self, tab, qtbot):
        with qtbot.waitSignal(tab.start_requested, timeout=1000):
            tab.start_btn.click()
        with qtbot.waitSignal(tab.remove_requested, timeout=1000):
            tab.remove_btn.click()

    def test_retranslate_updates_headers(self, tab):
        tab.retranslate()
        assert tab.table.horizontalHeaderItem(0).text() == i18n.tr("col_video_file", "Video File")


class TestSettingsTab:
    @pytest.fixture
    def tab(self, qtbot):
        t = SettingsTab()
        qtbot.addWidget(t)
        return t

    @pytest.fixture
    def languages(self):
        return list(i18n.get_available_languages().keys())

    def test_defaults_round_trip(self, tab, languages):
        defaults = get_default_settings()
        tab.populate(defaults, languages)
        read = tab.read_settings()
        assert read["--conf_threshold"] == defaults["--conf_threshold"]
        assert read["--use_gpu"] is True
        assert read["--time_start"] == defaults["--time_start"]
        assert read["ocr_engine"] == defaults["ocr_engine"]

    def test_populate_applies_modified_values(self, tab, languages):
        settings = get_default_settings()
        settings.update({
            "--conf_threshold": "40",
            "--time_start": "0:05",
            "--time_end": "1:00",
            "--frames_to_skip": "3",
            "--use_gpu": False,
            "--use_angle_cls": True,
        })
        tab.populate(settings, languages)
        read = tab.read_settings()
        assert read["--conf_threshold"] == "40"
        assert read["--time_start"] == "0:05"
        assert read["--time_end"] == "1:00"
        assert read["--frames_to_skip"] == "3"
        assert read["--use_gpu"] is False
        assert read["--use_angle_cls"] is True

    def test_gui_scaling_internal_key_round_trip(self, tab, languages):
        settings = get_default_settings()
        settings["gui_scaling"] = "scale_1_5"
        tab.populate(settings, languages)
        assert tab.read_settings()["gui_scaling"] == "scale_1_5"

    def test_dml_preset_round_trip(self, tab, languages):
        settings = get_default_settings()
        settings["--directml_performance_preset"] = "Max AMD GPU Load"
        tab.populate(settings, languages)
        read = tab.read_settings()
        assert read["--directml_performance_preset"] == "Max AMD GPU Load"

    def test_dml_recognition_round_trip(self, tab, languages):
        settings = get_default_settings()
        settings["--directml_recognition_mode"] = "Experimental Full DirectML"
        tab.populate(settings, languages)
        assert tab.read_settings()["--directml_recognition_mode"] == "Experimental Full DirectML"

    def test_alignment_internal_value_round_trip(self, tab, languages):
        settings = get_default_settings()
        settings["--subtitle_alignment"] = "top-right"
        tab.populate(settings, languages)
        assert tab.read_settings()["--subtitle_alignment"] == "top-right"

    def test_unknown_combo_value_falls_back_to_index_zero(self, tab, languages):
        settings = get_default_settings()
        settings["--directml_frame_scan_mode"] = "not-a-real-mode"
        tab.populate(settings, languages)
        assert tab.read_settings()["--directml_frame_scan_mode"] == "CPU SSIM (compatible)"

    def test_dml_widgets_disabled_without_gpu(self, tab, languages):
        settings = get_default_settings()
        settings["--use_directml_gpu"] = False
        tab.populate(settings, languages)
        for key in DML_DEPENDENT_WIDGETS:
            widget = tab._widgets[key]
            assert not widget.isEnabled(), key
        assert not tab.refresh_btn.isEnabled()

    def test_dml_widgets_enabled_with_gpu(self, tab, languages):
        settings = get_default_settings()
        settings["--use_directml_gpu"] = True
        tab.populate(settings, languages)
        for key in DML_DEPENDENT_WIDGETS:
            widget = tab._widgets[key]
            assert widget.isEnabled(), key

    def test_label_widgets_disabled_without_label_detection(self, tab, languages):
        settings = get_default_settings()
        settings["enable_label_detection"] = False
        tab.populate(settings, languages)
        for key in LABEL_DEPENDENT_WIDGETS:
            widget = tab._widgets[key]
            assert not widget.isEnabled(), key

    def test_label_widgets_enabled_with_label_detection(self, tab, languages):
        settings = get_default_settings()
        settings["enable_label_detection"] = True
        tab.populate(settings, languages)
        for key in LABEL_DEPENDENT_WIDGETS:
            widget = tab._widgets[key]
            assert widget.isEnabled(), key

    def test_toggling_gpu_checkbox_updates_dependents(self, tab, languages):
        tab.populate(get_default_settings(), languages)
        gpu_check = tab._widgets["--use_directml_gpu"]
        assert isinstance(gpu_check, QCheckBox)
        gpu_check.setChecked(False)
        tab._update_dependent_states()
        assert not tab._widgets["--directml_performance_preset"].isEnabled()
        gpu_check.setChecked(True)
        tab._update_dependent_states()
        assert tab._widgets["--directml_performance_preset"].isEnabled()

    def test_output_dir_disabled_when_saving_in_video_dir(self, tab, languages):
        settings = get_default_settings()
        settings["--save_in_video_dir"] = True
        tab.populate(settings, languages)
        out_dir = tab._widgets["--default_output_dir"]
        assert not out_dir.isEnabled()
        assert not tab.browse_output_btn.isEnabled()

        save_check = tab._widgets["--save_in_video_dir"]
        assert isinstance(save_check, QCheckBox)
        save_check.setChecked(False)
        tab._update_dependent_states()
        assert out_dir.isEnabled()
        assert tab.browse_output_btn.isEnabled()

    @pytest.mark.parametrize(
        ("variant", "cuda_ok", "dml_ok"),
        [
            ("cpu", False, False),
            ("gpu-cuda11.8", True, False),
            ("gpu-cuda12.9", True, False),
            ("gpu-directml", False, True),
        ],
    )
    def test_gpu_toggles_follow_the_build_variant(
        self, tab, languages, _tmp_build_variant, variant, cuda_ok, dml_ok
    ):
        _tmp_build_variant.write_text(variant + "\n", encoding="utf-8")
        tab.populate(get_default_settings(), languages)
        assert tab._widgets["--use_gpu"].isEnabled() is cuda_ok
        assert tab._widgets["--use_directml_gpu"].isEnabled() is dml_ok
        # The DirectML controls follow both the toggle and the build variant.
        assert tab._widgets["--directml_performance_preset"].isEnabled() is dml_ok
        assert tab._widgets["-DML_ADAPTER_COMBO-"].isEnabled() is dml_ok
        assert tab.refresh_btn.isEnabled() is dml_ok

    def test_gpu_section_is_titled_for_both_backends(self, tab):
        titles = [box.title() for box, _key, _fb in tab._section_boxes]
        assert any("GPU Acceleration" in title for title in titles)

    def test_reset_button_is_present_and_styled(self, tab):
        assert tab.reset_btn.objectName() == "dangerButton"
        assert tab.reset_btn.text() == i18n.tr(
            "btn_reset_settings", "Reset to Defaults"
        )

    def test_reset_button_emits_reset_requested(self, tab, qtbot):
        with qtbot.waitSignal(tab.reset_requested, timeout=1000):
            tab.reset_btn.click()

    def test_reset_button_retranslates(self, tab):
        tab.reset_btn.setText("stale")
        tab.retranslate()
        assert tab.reset_btn.text() == i18n.tr(
            "btn_reset_settings", "Reset to Defaults"
        )

    def test_settings_changed_signal_emitted(self, tab, languages, qtbot):
        tab.populate(get_default_settings(), languages)
        conf_widget = tab._widgets["--conf_threshold"]
        assert isinstance(conf_widget, QLineEdit)
        with qtbot.waitSignal(tab.settings_changed, timeout=1000):
            conf_widget.setText("33")

    def test_unknown_settings_key_ignored_on_populate(self, tab, languages):
        settings = get_default_settings()
        settings["not_a_widget_key"] = "whatever"
        tab.populate(settings, languages)  # must not raise
        # Unknown keys pass through read_settings unchanged (they are not
        # widget-backed), but every widget-backed key must still be present.
        read = tab.read_settings()
        assert read["not_a_widget_key"] == "whatever"
        assert read["--conf_threshold"] == settings["--conf_threshold"]

    def test_ui_language_combo_populated(self, tab, languages):
        tab.populate(get_default_settings(), languages)
        lang_combo = tab._widgets["-UI_LANG_COMBO-"]
        assert isinstance(lang_combo, QComboBox)
        assert lang_combo.count() >= 2


class TestDialogs:
    def test_info_dialog(self):
        from PySide6.QtWidgets import QLabel

        dlg = dialogs.InfoDialog("Title here", "Some message body")
        assert dlg.windowTitle() == "Title here"
        texts = [lbl.text() for lbl in dlg.findChildren(QLabel)]
        assert "Some message body" in texts
        dlg.close()

    def test_yes_no_dialog_buttons(self):
        dlg = dialogs.YesNoDialog("Confirm", "Are you sure?")
        from PySide6.QtWidgets import QPushButton

        texts = [b.text() for b in dlg.findChildren(QPushButton)]
        assert i18n.tr("btn_yes", "Yes") in texts
        assert i18n.tr("btn_no", "No") in texts
        dlg.close()

    def test_yes_no_dialog_default_focus(self):
        yes_dlg = dialogs.YesNoDialog("t", "m", default_yes=True)
        no_dlg = dialogs.YesNoDialog("t", "m", default_yes=False)

        def focused_button(dlg):
            w = dlg.focusWidget()
            return w.text() if w is not None else None

        assert focused_button(yes_dlg) == i18n.tr("btn_yes", "Yes")
        assert focused_button(no_dlg) == i18n.tr("btn_no", "No")
        yes_dlg.close()
        no_dlg.close()

    def test_countdown_dialog_initial_state(self):
        dlg = dialogs.CountdownDialog(None, "Shutdown", timeout_seconds=42)
        assert dlg._counter == 42
        assert dlg._proceed is False
        assert "Shutdown" in dlg._label.text()
        assert "42" in dlg._label.text()
        dlg._timer.stop()
        dlg.close()

    def test_countdown_tick_updates_label(self):
        dlg = dialogs.CountdownDialog(None, "Sleep", timeout_seconds=5)
        dlg._tick()
        assert dlg._counter == 4
        assert "4" in dlg._label.text()
        dlg._timer.stop()
        dlg.close()

    def test_countdown_zero_accepts(self):
        dlg = dialogs.CountdownDialog(None, "Sleep", timeout_seconds=1)
        dlg._tick()  # counter → 0 → auto-accept
        assert dlg._counter == 0
        assert dlg._proceed is True
        dlg._timer.stop()
        dlg.close()

    def test_countdown_proceed_button(self):
        dlg = dialogs.CountdownDialog(None, "Hibernate", timeout_seconds=30)
        dlg._on_proceed()
        assert dlg._proceed is True
        dlg._timer.stop()
        dlg.close()

    def test_dialog_is_modal(self):
        dlg = dialogs.InfoDialog("t", "m")
        assert dlg.isModal()
        dlg.close()


class TestResources:
    def test_icon_path_is_string(self):
        path = resources.icon_path()
        assert isinstance(path, str)

    def test_notification_icon_path_is_string(self):
        path = resources.notification_icon_path()
        assert isinstance(path, str)


class TestWidgets:
    def test_clickable_slider_handle_width(self):
        from videocr_gui.widgets import ClickableSlider

        slider = ClickableSlider()
        assert isinstance(slider.handle_width(), int)
        assert slider.handle_width() >= 0
        slider.deleteLater()

    def test_outlined_progress_bar_constants(self):
        from videocr_gui.widgets import OutlinedProgressBar

        for attr in ("OUTLINE_COLOR", "TEXT_COLOR", "OUTLINE_WIDTH"):
            assert hasattr(OutlinedProgressBar, attr)
