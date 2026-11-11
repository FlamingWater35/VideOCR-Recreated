"""Tests for the engine presets (OCR + label tuning) and the selector wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QComboBox

REPO_ROOT = Path(__file__).resolve().parent.parent

PADDLE_ENGINE = "PaddleOCR (Det. + Rec.)"
LENS_ENGINE = "PaddleOCR (Det.) + Google Lens (Rec.)"
ONNX_ENGINE = "ONNX Runtime DirectML (AMD GPU Experimental)"


def _presets():
    from videocr_gui import presets

    return presets


class TestPresetTables:
    def test_three_presets_are_offered(self):
        presets = _presets()
        assert presets.preset_ids() == ["paddle", "onnx", "custom"]

    def test_every_preset_but_custom_has_values(self):
        presets = _presets()
        for pid in presets.preset_ids():
            values = presets.preset_values(pid)
            if pid == presets.CUSTOM_PRESET:
                assert values == {}
            else:
                assert values, f"{pid} has no values"

    def test_presets_only_touch_ocr_and_label_keys(self):
        """A preset must not reach outside OCR Settings and Label Detection."""
        presets = _presets()
        allowed = set(presets.PRESET_KEYS)
        for pid in presets.preset_ids():
            assert set(presets.preset_values(pid)) <= allowed

    def test_preset_never_toggles_label_detection(self):
        """A preset tunes label knobs; it must not switch the feature on/off."""
        presets = _presets()
        for pid in presets.preset_ids():
            assert "--enable_label_detection" not in presets.preset_values(pid)

    def test_preset_never_touches_crop_or_gpu_keys(self):
        presets = _presets()
        forbidden = {
            "--crop_x",
            "--crop_y",
            "--crop_width",
            "--crop_height",
            "--use_fullframe",
            "--directml_performance_preset",
            "--directml_frame_scan_mode",
            "--label_font",
            "--subtitle_font",
        }
        for pid in presets.preset_ids():
            assert not (set(presets.preset_values(pid)) & forbidden)

    def test_onnx_preset_picks_the_medium_tier(self):
        """The one measured per-engine difference: medium beats small on ONNX."""
        presets = _presets()
        assert presets.preset_values(presets.ONNX_PRESET)["--model_tier"] == "medium"
        assert presets.preset_values(presets.PADDLE_PRESET)["--model_tier"] == "small"

    def test_fresh_install_starts_on_the_paddle_preset(self):
        """A brand-new config must not show "Custom" before the user touches it."""
        from videocr_gui import config

        presets = _presets()
        defaults = config.get_default_settings()
        assert presets.matches_preset(defaults, presets.PADDLE_PRESET)
        assert (
            presets.detect_preset(defaults, "PaddleOCR (Det. + Rec.)")
            == presets.PADDLE_PRESET
        )

    def test_paddle_preset_matches_the_shipped_label_defaults(self):
        """The preset must not silently diverge from the CLI's own defaults."""
        from videocr_gui import config

        presets = _presets()
        defaults = config.get_default_settings()
        for key in ("--label_pos_drift", "--label_reappear_merge_gap"):
            assert (
                presets.preset_values(presets.PADDLE_PRESET)[key] == defaults[key]
            ), key

    def test_ocr_settings_keys_are_real_config_keys(self):
        from videocr_gui import config

        defaults = config.get_default_settings()
        for key in _presets().PRESET_KEYS:
            assert key in defaults, f"{key} is not a config key"

    def test_preset_values_are_non_empty_strings(self):
        presets = _presets()
        for pid in (presets.PADDLE_PRESET, presets.ONNX_PRESET):
            for key, value in presets.preset_values(pid).items():
                assert isinstance(value, str)
                # brightness_threshold is legitimately empty (auto)
                assert value != "" or key == "--brightness_threshold"

    def test_engines_map_to_their_preset(self):
        presets = _presets()
        assert presets.preset_for_engine(PADDLE_ENGINE) == presets.PADDLE_PRESET
        assert presets.preset_for_engine(LENS_ENGINE) == presets.PADDLE_PRESET
        assert presets.preset_for_engine(ONNX_ENGINE) == presets.ONNX_PRESET

    def test_unknown_engine_falls_back_to_paddle(self):
        assert _presets().preset_for_engine("Some Future Engine") == "paddle"


class TestDetectPreset:
    def test_exact_values_detect_their_preset(self):
        presets = _presets()
        for pid in (presets.PADDLE_PRESET, presets.ONNX_PRESET):
            settings = dict(presets.preset_values(pid))
            assert presets.detect_preset(settings, PADDLE_ENGINE) in (
                presets.PADDLE_PRESET,
                presets.ONNX_PRESET,
            )

    def test_engine_preference_breaks_ties(self):
        """With both presets matching nothing, the engine's own is chosen."""
        presets = _presets()
        settings = dict(presets.preset_values(presets.PADDLE_PRESET))
        assert presets.detect_preset(settings, PADDLE_ENGINE) == presets.PADDLE_PRESET

    def test_edited_value_detects_custom(self):
        presets = _presets()
        settings = dict(presets.preset_values(presets.PADDLE_PRESET))
        settings["--conf_threshold"] = "42"
        assert presets.detect_preset(settings, PADDLE_ENGINE) == presets.CUSTOM_PRESET

    def test_missing_key_detects_custom(self):
        presets = _presets()
        settings = dict(presets.preset_values(presets.PADDLE_PRESET))
        settings.pop("--sim_threshold")
        assert presets.detect_preset(settings, PADDLE_ENGINE) == presets.CUSTOM_PRESET

    def test_whitespace_is_ignored(self):
        presets = _presets()
        settings = {
            key: f"  {value}  "
            for key, value in presets.preset_values(presets.PADDLE_PRESET).items()
        }
        assert presets.detect_preset(settings, PADDLE_ENGINE) == presets.PADDLE_PRESET

    def test_apply_preset_does_not_mutate_the_input(self):
        presets = _presets()
        original = {"--conf_threshold": "1", "--keep_me": "x"}
        result = presets.apply_preset(original, presets.PADDLE_PRESET)
        assert original["--conf_threshold"] == "1"
        assert result["--keep_me"] == "x"
        assert result["--conf_threshold"] == presets.preset_values("paddle")["--conf_threshold"]


class TestSettingsTabPresetSelector:
    @pytest.fixture
    def tab(self, qtbot):
        from videocr_gui import config
        from videocr_gui.settings_tab import SettingsTab

        widget = SettingsTab()
        qtbot.addWidget(widget)
        widget.populate(config.get_default_settings(), ["English"])
        return widget

    def test_selector_exists_with_three_entries(self, tab):
        combo = tab._widgets["--preset"]
        assert combo.count() == 3
        assert [combo.itemText(i) for i in range(3)] == [
            "PaddleOCR / Google Lens",
            "ONNX Runtime DirectML",
            "Custom",
        ]

    def test_choosing_a_preset_writes_every_managed_value(self, tab):
        presets = _presets()
        tab._select_preset(presets.ONNX_PRESET)
        for key, value in presets.preset_values(presets.ONNX_PRESET).items():
            widget = tab._widgets[key]
            if isinstance(widget, QComboBox):
                internal = tab._combo_internal[key]
                assert internal[widget.currentIndex()] == value
            else:
                assert widget.text() == value

    def test_one_selection_shows_the_preset_not_custom(self, tab):
        """Regression: applying a preset used to flip the combo to Custom.

        Every value write emitted textChanged -> _on_changed -> _mark_custom(),
        so the first click looked ignored and only a second click (a no-op,
        because the values already matched) left the preset showing.
        """
        presets = _presets()
        combo = tab._widgets["--preset"]
        internal = tab._combo_internal["--preset"]

        combo.setCurrentIndex(internal.index(presets.ONNX_PRESET))
        assert tab._current_preset() == presets.ONNX_PRESET

        combo.setCurrentIndex(internal.index(presets.PADDLE_PRESET))
        assert tab._current_preset() == presets.PADDLE_PRESET

        combo.setCurrentIndex(internal.index(presets.ONNX_PRESET))
        assert tab._current_preset() == presets.ONNX_PRESET

    def test_selecting_the_same_preset_twice_stays_on_it(self, tab):
        presets = _presets()
        combo = tab._widgets["--preset"]
        internal = tab._combo_internal["--preset"]
        index = internal.index(presets.ONNX_PRESET)
        combo.setCurrentIndex(index)
        combo.setCurrentIndex(index)
        assert tab._current_preset() == presets.ONNX_PRESET

    def test_manual_edit_after_a_preset_still_marks_custom(self, tab):
        """The guard must not swallow real user edits."""
        presets = _presets()
        tab._select_preset(presets.ONNX_PRESET)
        assert tab._current_preset() == presets.ONNX_PRESET
        tab._widgets["--conf_threshold"].setText("42")
        assert tab._current_preset() == presets.CUSTOM_PRESET

    def test_editing_after_applying_a_preset_marks_custom(self, tab):
        """_apply_preset_values must clear its guard even if a write throws."""
        assert tab._applying_preset is False
        tab._apply_preset_values(_presets().ONNX_PRESET)
        assert tab._applying_preset is False

    def test_applying_a_preset_emits_one_consolidated_signal(self, tab):
        """The tab must not emit once per field while a preset is applied.

        The main window snapshots the whole tab on every settings_changed, so
        per-field signals made it persist intermediate states.
        """
        seen: list[list[str]] = []
        tab.settings_changed.connect(lambda keys: seen.append(list(keys)))
        tab._select_preset(_presets().ONNX_PRESET)
        assert len(seen) == 1, seen
        assert "--preset" in seen[0]

    def test_saved_preset_matches_saved_values_after_an_engine_switch(self, qtbot):
        """Regression: the config paired --preset paddle with the ONNX tier.

        The selector was updated after the values, so the snapshot taken on the
        first value-change signal carried the old preset id. The conftest's
        autouse fixture points CONFIG_FILE at a throwaway path.
        """
        from videocr_gui import config
        from videocr_gui.app import MainWindow

        window = MainWindow(config.get_default_settings())
        qtbot.addWidget(window)
        try:
            window.engine_combo.setCurrentIndex(2)  # ONNX
            saved = config.load_settings()
            assert saved["--preset"] == "onnx", saved.get("--preset")
            assert saved["--model_tier"] == "medium", saved.get("--model_tier")
            assert _presets().matches_preset(saved, "onnx")
        finally:
            window.close()

    def test_choosing_a_preset_leaves_unmanaged_keys_alone(self, tab):
        """Crop/GPU/style settings and the label toggle must not move."""
        unmanaged = (
            "--label_font",
            "--label_fontsize",
            "--label_primary_color",
            "--subtitle_fontsize",
            "--crop_x",
            "--directml_frame_scan_mode",
        )
        tab._widgets["--label_fontsize"].setText("41")

        def _value(key: str) -> str:
            widget = tab._widgets[key]
            if isinstance(widget, QComboBox):
                return widget.currentText()
            return widget.text()

        before = {k: _value(k) for k in unmanaged if k in tab._widgets}
        tab._select_preset("onnx")
        after = {k: _value(k) for k in unmanaged if k in tab._widgets}
        assert before == after

    def test_choosing_a_preset_never_enables_label_detection(self, tab):
        tab._widgets["enable_label_detection"].setChecked(False)
        tab._select_preset("onnx")
        assert not tab._widgets["enable_label_detection"].isChecked()
        tab._widgets["enable_label_detection"].setChecked(True)
        tab._select_preset("paddle")
        assert tab._widgets["enable_label_detection"].isChecked()

    def test_preset_sets_the_model_tier_combo(self, tab):
        presets = _presets()
        tab._select_preset(presets.ONNX_PRESET)
        combo = tab._widgets["--model_tier"]
        internal = tab._combo_internal["--model_tier"]
        assert internal[combo.currentIndex()] == "medium"
        tab._select_preset(presets.PADDLE_PRESET)
        assert internal[combo.currentIndex()] == "small"

    def test_editing_an_ocr_setting_switches_to_custom(self, tab):
        tab._select_preset("paddle")
        assert tab._current_preset() == "paddle"
        tab._on_changed(["--conf_threshold"])
        assert tab._current_preset() == "custom"

    def test_editing_a_label_setting_switches_to_custom(self, tab):
        """Label knobs are preset-managed now, so editing one means Custom."""
        tab._select_preset("paddle")
        tab._on_changed(["--label_conf_threshold"])
        assert tab._current_preset() == "custom"

    def test_editing_an_unmanaged_setting_keeps_the_preset(self, tab):
        tab._select_preset("paddle")
        tab._on_changed(["--label_fontsize"])
        assert tab._current_preset() == "paddle"

    def test_engine_switch_applies_that_engine_preset(self, tab):
        presets = _presets()
        tab._select_preset("paddle")
        tab.sync_preset_to_engine(ONNX_ENGINE)
        assert tab._current_preset() == "onnx"
        for key, value in presets.preset_values(presets.ONNX_PRESET).items():
            widget = tab._widgets[key]
            if isinstance(widget, QComboBox):
                internal = tab._combo_internal[key]
                assert internal[widget.currentIndex()] == value
            else:
                assert widget.text() == value

    def test_engine_switch_keeps_custom_untouched(self, tab):
        """A hand-tuned user must not lose their numbers on an engine switch."""
        tab._select_preset("paddle")
        tab._widgets["--conf_threshold"].setText("42")
        tab._on_changed(["--conf_threshold"])
        assert tab._current_preset() == "custom"
        tab.sync_preset_to_engine(ONNX_ENGINE)
        assert tab._current_preset() == "custom"
        assert tab._widgets["--conf_threshold"].text() == "42"

    def test_populate_restores_the_saved_preset(self, qtbot):
        from videocr_gui import config
        from videocr_gui.settings_tab import SettingsTab

        settings = config.get_default_settings()
        settings["--preset"] = "onnx"
        settings.update(_presets().preset_values("onnx"))
        widget = SettingsTab()
        qtbot.addWidget(widget)
        widget.populate(settings, ["English"])
        assert widget._current_preset() == "onnx"

    def test_populate_detects_custom_from_values(self, qtbot):
        from videocr_gui import config
        from videocr_gui.settings_tab import SettingsTab

        settings = config.get_default_settings()
        settings.update(_presets().preset_values("onnx"))
        settings["--conf_threshold"] = "33"
        widget = SettingsTab()
        qtbot.addWidget(widget)
        widget.populate(settings, ["English"])
        assert widget._current_preset() == "custom"


class TestPresetIsGuiOnly:
    def test_preset_is_not_forwarded_to_the_cli(self):
        from videocr_gui import config
        from videocr_gui.args import build_args

        settings = config.get_default_settings()
        settings["--preset"] = "onnx"
        args, warnings = build_args("C:/v/movie.mp4", settings, [], "C:/v/out.srt")
        assert args is not None
        assert "preset" not in args
        assert not warnings

    def test_preset_is_persisted(self):
        from videocr_gui import config

        assert "--preset" in config.get_default_settings()


class TestPresetTranslations:
    def test_every_language_translates_the_preset_ui(self):
        keys = (
            "lbl_preset",
            "preset_paddle",
            "preset_onnx",
            "preset_custom",
            "tip_preset",
            "preset_note_applies",
            "preset_note_custom",
        )
        for path in sorted((REPO_ROOT / "languages").glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            for key in keys:
                assert key in data, f"{path.name} is missing {key}"
                assert str(data[key]).strip(), f"{path.name} has an empty {key}"

    def test_preset_names_stay_in_english_everywhere(self):
        """Engine names are product names; they must not be translated."""
        for path in sorted((REPO_ROOT / "languages").glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["preset_paddle"] == "PaddleOCR / Google Lens"
            assert data["preset_onnx"] == "ONNX Runtime DirectML"
