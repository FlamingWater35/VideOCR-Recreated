"""Smoke tests for videocr_gui.app.MainWindow — construction, tabs, helpers.

The window is built with an injected settings dict (no config file read), shown
offscreen, and closed cleanly. No OCR worker is ever started.
"""

from __future__ import annotations

import inspect
import sys
from typing import Any

import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import QApplication

from videocr_gui import app as app_module
from videocr_gui import config, i18n, update_check
from videocr_gui.app import MainWindow


@pytest.fixture(autouse=True)
def _english():
    i18n.load_language("en")


@pytest.fixture
def settings() -> dict[str, Any]:
    return config.get_default_settings()


@pytest.fixture
def window(qtbot, settings):
    win = MainWindow(settings=settings)
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win, timeout=3000)
    yield win
    win.close()


class TestMainWindowConstruction:
    def test_window_constructs_and_shows(self, window):
        assert window.isVisible()

    def test_four_tabs_present(self, window):
        assert window.tabs.count() == 4

    def test_tab_titles_translated(self, window):
        titles = [window.tabs.tabText(i) for i in range(4)]
        assert all(titles)
        assert titles != ["", "", "", ""]

    def test_core_widgets_exist(self, window):
        assert window.preview is not None
        assert window.queue_tab is not None
        assert window.settings_tab is not None
        assert window.about_tab is not None

    def test_injected_settings_used(self, window, settings):
        assert window._settings.get("ocr_engine") == settings["ocr_engine"]

    def test_queue_starts_empty(self, window):
        assert window.queue_tab.jobs() == []

    def test_close_is_clean(self, qtbot, settings):
        win = MainWindow(settings=settings)
        qtbot.addWidget(win)
        win.show()
        win.close()
        assert not win.isVisible()

    def test_default_size_is_1000x850_on_large_screen(self, qtbot, settings):
        win = MainWindow(settings=settings)
        qtbot.addWidget(win)

        class _LargeScreen:
            def availableGeometry(self):
                return QRect(0, 0, 2560, 1440)

        win.screen = lambda: _LargeScreen()  # type: ignore[method-assign]
        win._resize_to_work_area()
        assert (win.width(), win.height()) == (1000, 850)


class TestWindowStatePersistence:
    """Fullscreen/maximized status is saved at exit and restored at start.

    The offscreen test platform has an 800x800 screen — smaller than the
    window's ~1136 px minimum width — so Qt drops a pre-show fullscreen
    request there (a real screen honors it). Fullscreen is therefore asserted
    on the state applied while hidden, and entered *after* show() for the
    save path.
    """

    @pytest.mark.parametrize("state", ["maximized", "normal"])
    def test_state_saved_on_close(self, qtbot, settings, state):
        settings["window_state"] = state
        win = MainWindow(settings=settings)
        qtbot.addWidget(win)
        win.show()
        win.close()
        assert config.load_settings()["window_state"] == state

    def test_fullscreen_survives_restart(self, qtbot, settings):
        # Session 1: the user closes the window while fullscreen.
        win = MainWindow(settings=settings)
        qtbot.addWidget(win)
        win.show()
        win.setWindowState(win.windowState() | Qt.WindowState.WindowFullScreen)
        win.close()
        assert config.load_settings()["window_state"] == "fullscreen"

        # Session 2: a fresh window built from the saved config is put back
        # into fullscreen before it is shown.
        win2 = MainWindow(settings=config.load_settings())
        qtbot.addWidget(win2)
        assert win2.windowState() & Qt.WindowState.WindowFullScreen
        win2.close()

    def test_maximized_restored_on_start(self, qtbot, settings):
        settings["window_state"] = "maximized"
        win = MainWindow(settings=settings)
        qtbot.addWidget(win)
        win.show()
        assert win.isMaximized()

    def test_normal_start_leaves_window_unmaximized(self, qtbot, settings):
        settings["window_state"] = "normal"
        win = MainWindow(settings=settings)
        qtbot.addWidget(win)
        win.show()
        assert not win.isMaximized()
        assert not win.isFullScreen()


class TestStaticHelpers:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("a.mp4", True), ("b.AVI", True), ("c.mkv", True), ("d.mov", True),
            ("e.webm", True), ("f.flv", True), ("g.wmv", True), ("h.ts", True),
            ("i.m2ts", True), ("j.txt", False), ("k.srt", False),
        ],
    )
    def test_scan_video_folder_extensions(self, tmp_path, name, expected):
        (tmp_path / name).write_bytes(b"")
        result = MainWindow._scan_video_folder(str(tmp_path))
        assert (name in [p.split("/")[-1].split("\\")[-1] for p in result]) is expected

    def test_scan_video_folder_sorted(self, tmp_path):
        for name in ("zz.mp4", "aa.mp4", "mm.mp4"):
            (tmp_path / name).write_bytes(b"")
        result = MainWindow._scan_video_folder(str(tmp_path))
        basenames = [p.replace("\\", "/").rsplit("/", 1)[-1] for p in result]
        assert basenames == sorted(basenames)

    def test_scan_video_folder_not_a_folder(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x", encoding="utf-8")
        assert MainWindow._scan_video_folder(str(f)) == []

    def test_scan_video_folder_ignores_subdirs(self, tmp_path):
        (tmp_path / "sub.mp4").mkdir()
        (tmp_path / "real.mp4").write_bytes(b"")
        result = MainWindow._scan_video_folder(str(tmp_path))
        assert len(result) == 1
        assert result[0].endswith("real.mp4")

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(0, "00:00"), (90, "01:30"), (3599, "59:59"), (3600, "01:00:00"), (7325, "02:02:05")],
    )
    def test_fmt_time(self, seconds, expected):
        assert MainWindow._fmt_time(seconds) == expected


class TestCurrentSettings:
    def test_returns_settings_with_ui_overrides(self, window):
        current = window._current_settings()
        assert isinstance(current, dict)
        # The three combo-backed keys come from the main window, not the tab.
        for key in ("ocr_engine", "subtitle_language", "subtitle_position"):
            assert key in current
            assert current[key] == window._settings[key]

    def test_thresholds_come_from_tab(self, window, settings):
        current = window._current_settings()
        assert current["--conf_threshold"] == settings["--conf_threshold"]


class TestLanguageList:
    def test_default_engine_language_list(self, window):
        # Default engine is Paddle → language list must be the paddle list.
        window._update_lang_list("PaddleOCR (Det. + Rec.)")
        combo = window.lang_combo
        assert combo.count() > 50
        assert "English" in [combo.itemText(i) for i in range(combo.count())]

    def test_lens_engine_language_list(self, window):
        window._update_lang_list("PaddleOCR (Det.) + Google Lens (Rec.)")
        combo = window.lang_combo
        texts = [combo.itemText(i) for i in range(combo.count())]
        assert "Japanese" in texts  # lens uses ja-style names
        assert "Chinese (Simplified)" in texts

    def test_onnx_engine_language_list(self, window):
        window._update_lang_list("ONNX Runtime DirectML (AMD GPU Experimental)")
        combo = window.lang_combo
        texts = [combo.itemText(i) for i in range(combo.count())]
        assert "Japanese + English" in texts  # easyocr-style names

    def test_engine_list_offers_exactly_three(self, window):
        combo = window.engine_combo
        texts = [combo.itemText(i) for i in range(combo.count())]
        from videocr_gui import constants as C

        assert texts == C.OCR_ENGINES


class TestLegacyEngineMigration:
    def test_legacy_engine_displayed_as_onnx(self, window):
        # Old configs with the removed EasyOCR display name must surface as the
        # ONNX Runtime DirectML engine (LEGACY_OCR_ENGINE_MAP).
        window._settings["ocr_engine"] = "EasyOCR DirectML (AMD GPU)"
        window._populate_engine_lang_pos()
        combo = window.engine_combo
        current = combo.currentText()
        from videocr_gui import constants as C

        assert current == C.LEGACY_OCR_ENGINE_MAP["EasyOCR DirectML (AMD GPU)"]
        assert current == C.OCR_ENGINES[2]


class TestBuildVariantEngineAvailability:
    """Engines the package cannot run are greyed out in the picker."""

    @staticmethod
    def _enabled_engines(combo):
        model = combo.model()
        enabled = []
        for i in range(combo.count()):
            item = model.item(i)
            if item is None or item.isEnabled():
                enabled.append(combo.itemText(i))
        return enabled

    def test_every_engine_enabled_without_a_marker(self, window):
        from videocr_gui import constants as C

        assert self._enabled_engines(window.engine_combo) == list(C.OCR_ENGINES)

    @pytest.mark.parametrize("variant", ["cpu", "gpu-cuda11.8", "gpu-cuda12.9"])
    def test_onnx_greyed_out_outside_the_directml_build(
        self, qtbot, settings, _tmp_build_variant, variant
    ):
        from videocr_gui import constants as C

        _tmp_build_variant.write_text(variant + "\n", encoding="utf-8")
        win = MainWindow(settings=settings)
        qtbot.addWidget(win)
        assert self._enabled_engines(win.engine_combo) == list(C.OCR_ENGINES[:2])

    def test_onnx_selectable_in_the_directml_build(self, qtbot, settings, _tmp_build_variant):
        from videocr_gui import constants as C

        _tmp_build_variant.write_text("gpu-directml\n", encoding="utf-8")
        win = MainWindow(settings=settings)
        qtbot.addWidget(win)
        assert self._enabled_engines(win.engine_combo) == list(C.OCR_ENGINES)

    def test_saved_onnx_selection_falls_back_when_greyed_out(
        self, qtbot, settings, _tmp_build_variant
    ):
        from videocr_gui import constants as C

        _tmp_build_variant.write_text("cpu\n", encoding="utf-8")
        settings["ocr_engine"] = C.OCR_ENGINES[2]
        win = MainWindow(settings=settings)
        qtbot.addWidget(win)
        assert win.engine_combo.currentText() == C.DEFAULT_OCR_ENGINE
        assert win._settings["ocr_engine"] == C.DEFAULT_OCR_ENGINE


class TestResetSettings:
    """The Advanced Settings tab can restore every stored preference to default."""

    @staticmethod
    def _window(qtbot, settings):
        win = MainWindow(settings=settings)
        qtbot.addWidget(win)
        return win

    def test_confirmed_reset_restores_defaults(self, qtbot, settings, monkeypatch):
        from videocr_gui import constants as C

        settings["--conf_threshold"] = "11"
        settings["ocr_engine"] = C.OCR_ENGINES[1]
        win = self._window(qtbot, settings)
        assert win.settings_tab.read_settings()["--conf_threshold"] == "11"

        monkeypatch.setattr(app_module, "ask_yes_no", lambda *a, **k: True)
        win.settings_tab.reset_btn.click()

        defaults = config.get_default_settings()
        assert win._settings["--conf_threshold"] == defaults["--conf_threshold"]
        assert win._settings["ocr_engine"] == defaults["ocr_engine"]
        assert win.settings_tab.read_settings()["--conf_threshold"] == (
            defaults["--conf_threshold"]
        )
        assert win.engine_combo.currentText() == defaults["ocr_engine"]

    def test_declined_reset_changes_nothing(self, qtbot, settings, monkeypatch):
        settings["--conf_threshold"] = "11"
        win = self._window(qtbot, settings)

        monkeypatch.setattr(app_module, "ask_yes_no", lambda *a, **k: False)
        win.settings_tab.reset_btn.click()

        assert win._settings["--conf_threshold"] == "11"
        assert win.settings_tab.read_settings()["--conf_threshold"] == "11"

    def test_reset_persists_to_the_config_file(
        self, qtbot, settings, monkeypatch, _tmp_config_file
    ):
        settings["--conf_threshold"] = "11"
        win = self._window(qtbot, settings)

        monkeypatch.setattr(app_module, "ask_yes_no", lambda *a, **k: True)
        win.settings_tab.reset_btn.click()

        assert config.load_settings()["--conf_threshold"] == (
            config.get_default_settings()["--conf_threshold"]
        )


class TestAppModuleImports:
    def test_video_path_checked_at_start(self, monkeypatch, tmp_path):
        # The module-level VIDEOCR_PATH binding controls the "CLI not found"
        # branch; just assert the name exists on both modules.
        assert hasattr(app_module, "VIDEOCR_PATH")
        from videocr_gui import workers

        assert hasattr(workers, "VIDEOCR_PATH")

    def test_qapplication_exists_for_gui_tests(self):
        assert QApplication.instance() is not None


class TestTaskbarIntegration:
    """Guards the PyTaskbar dependency contract used by app.py.

    Regression for the v1.6.6 CI break: the old git dependency
    (timminator/PyTaskbar) renamed its distribution and changed its API to
    `Progress`, which (a) made `pip install .[directml]` fail outright and
    (b) silently disabled taskbar progress because app.py calls
    `TaskbarProgress`/`ProgressType`.
    """

    def test_installed_package_exposes_app_api(self):
        PyTaskbar = pytest.importorskip("PyTaskbar")
        assert hasattr(PyTaskbar, "TaskbarProgress"), (
            "installed PyTaskbar lacks TaskbarProgress — the app.py API; "
            "the timminator git fork's 'Progress' API is not compatible"
        )
        assert hasattr(PyTaskbar, "ProgressType")
        for state in ("NOPROGRESS", "NORMAL", "PAUSED"):
            assert hasattr(PyTaskbar.ProgressType, state), state
        # app.py calls set_progress(value, 100) — the second arg must exist.
        params = inspect.signature(PyTaskbar.TaskbarProgress.set_progress).parameters
        assert "max" in params

    @pytest.mark.skipif(sys.platform != "win32", reason="taskbar progress is Windows-only")
    def test_taskbar_initializes_and_updates(self, qtbot, settings, _error_log):
        win = MainWindow(settings=settings)
        qtbot.addWidget(win)
        win.show()
        # _init_taskbar schedules _setup_taskbar via QTimer.singleShot(0, ...).
        qtbot.wait(200)
        assert win._taskbar is not None, (
            "MainWindow taskbar failed to initialize: "
            + "; ".join(m for m in _error_log if "Taskbar" in m)
        )
        win._update_taskbar(state="normal", progress=42)
        win._update_taskbar(state="paused", progress=42)
        win._update_taskbar(progress=100)
        win.close()


class TestUpdateCheck:
    """Boot scheduling, the dynamic Update tab, and the settings toggle."""

    def test_boot_check_scheduled_after_3_seconds(self, qtbot, settings):
        win = MainWindow(settings=settings)
        qtbot.addWidget(win)
        assert win._update_timer.isActive()
        assert win._update_timer.interval() == update_check.BOOT_CHECK_DELAY_MS
        assert update_check.BOOT_CHECK_DELAY_MS == 3000

    def test_boot_check_skipped_when_disabled(self, qtbot, settings):
        settings["check_updates"] = False
        win = MainWindow(settings=settings)
        qtbot.addWidget(win)
        assert not win._update_timer.isActive()

    def test_start_check_respects_disabled_setting(
        self, qtbot, settings, monkeypatch
    ):
        settings["check_updates"] = False
        calls: list[int] = []

        def fake_fetch(*args, **kwargs):
            calls.append(1)
            return None

        monkeypatch.setattr(update_check, "fetch_latest_version", fake_fetch)
        win = MainWindow(settings=settings)
        qtbot.addWidget(win)
        win._start_update_check()
        assert not win._update_check_started
        assert calls == []

    def test_start_check_runs_once_off_thread(self, qtbot, settings, monkeypatch):
        calls: list[int] = []

        def fake_fetch(*args, **kwargs):
            calls.append(1)
            return None

        monkeypatch.setattr(update_check, "fetch_latest_version", fake_fetch)
        win = MainWindow(settings=settings)
        qtbot.addWidget(win)
        win._start_update_check()
        win._start_update_check()  # guarded: one check per session
        assert win._update_check_started
        qtbot.waitUntil(lambda: len(calls) >= 1, timeout=3000)
        qtbot.wait(50)
        assert calls == [1]
        assert win.tabs.count() == 4  # no update info → no Update tab

    def test_no_update_tab_without_newer_version(self, qtbot, settings):
        win = MainWindow(settings=settings)
        qtbot.addWidget(win)
        assert win.tabs.count() == 4
        for latest in (None, "", "0.9.0", win._version()):
            win._on_update_check_finished(latest)
            assert win.tabs.count() == 4
        assert win._update_tab is None

    def test_update_tab_appears_only_for_newer_version(self, qtbot, settings):
        win = MainWindow(settings=settings)
        qtbot.addWidget(win)
        assert win.tabs.count() == 4
        win._on_update_check_finished("999.0.0")
        assert win.tabs.count() == 5
        assert win.tabs.tabText(4) == "Update Available"
        # Re-reporting a newer version refreshes the existing tab, not a second one.
        win._on_update_check_finished("999.0.1")
        assert win.tabs.count() == 5

    def test_update_tab_content_links_to_latest_release(self, qtbot, settings):
        win = MainWindow(settings=settings)
        qtbot.addWidget(win)
        win._on_update_check_finished("999.0.0")
        tab = win._update_tab
        assert tab is not None
        assert "999.0.0" in tab.available_lbl.text()
        assert win._version() in tab.current_lbl.text()
        assert tab.download_lbl.text()
        assert tab.link_lbl.text() == update_check.LATEST_RELEASE_URL

    def test_disabling_check_removes_update_tab(self, qtbot, settings):
        win = MainWindow(settings=settings)
        qtbot.addWidget(win)
        win._on_update_check_finished("999.0.0")
        assert win.tabs.count() == 5

        win.settings_tab._widgets["check_updates"].setChecked(False)

        assert win.tabs.count() == 4
        assert win._update_tab is None
        assert not win._update_timer.isActive()
        assert config.load_settings()["check_updates"] is False
