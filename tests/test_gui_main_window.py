"""Smoke tests for videocr_gui.app.MainWindow — construction, tabs, helpers.

The window is built with an injected settings dict (no config file read), shown
offscreen, and closed cleanly. No OCR worker is ever started.
"""

from __future__ import annotations

import inspect
import sys
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication

from videocr_gui import app as app_module
from videocr_gui import config, i18n
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
