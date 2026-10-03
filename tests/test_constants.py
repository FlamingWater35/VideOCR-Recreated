"""Tests for videocr_gui.constants — static tables, lookups and defaults."""

from __future__ import annotations

import sys

import pytest

from videocr_gui import constants as C


class TestEngineTables:
    def test_ocr_engines_unique(self):
        assert len(C.OCR_ENGINES) == len(set(C.OCR_ENGINES))

    def test_default_engine_is_offered(self):
        assert C.DEFAULT_OCR_ENGINE in C.OCR_ENGINES

    def test_easyocr_not_selectable(self):
        # EasyOCR DirectML is intentionally not a selectable GUI engine.
        assert not any("EasyOCR" in e for e in C.OCR_ENGINES)

    def test_legacy_map_targets_real_engine(self):
        for old, new in C.LEGACY_OCR_ENGINE_MAP.items():
            assert old not in C.OCR_ENGINES  # removed from the picker
            assert new in C.OCR_ENGINES      # remaps to a live engine

    def test_legacy_map_covers_removed_easyocr_name(self):
        assert C.LEGACY_OCR_ENGINE_MAP["EasyOCR DirectML (AMD GPU)"] in C.OCR_ENGINES


class TestLanguageLists:
    def test_paddle_languages_unique_names(self):
        names = [n for n, _ in C.PADDLEOCR_LANGUAGES_LIST]
        codes = [c for _, c in C.PADDLEOCR_LANGUAGES_LIST]
        assert len(names) == len(set(names))
        assert len(codes) == len(set(codes))

    def test_lens_languages_unique_names(self):
        names = [n for n, _ in C.GOOGLE_LENS_LANGUAGES_LIST]
        assert len(names) == len(set(names))

    def test_easyocr_languages_unique_names(self):
        names = [n for n, _ in C.EASYOCR_LANGUAGES_LIST]
        assert len(names) == len(set(names))

    def test_lookup_dicts_match_lists(self):
        assert C.paddle_abbr_lookup == dict(C.PADDLEOCR_LANGUAGES_LIST)
        assert C.lens_abbr_lookup == dict(C.GOOGLE_LENS_LANGUAGES_LIST)
        assert C.easyocr_abbr_lookup == dict(C.EASYOCR_LANGUAGES_LIST)

    def test_display_name_lists_match(self):
        assert C.paddle_display_names == [n for n, _ in C.PADDLEOCR_LANGUAGES_LIST]
        assert C.lens_display_names == [n for n, _ in C.GOOGLE_LENS_LANGUAGES_LIST]
        assert C.easyocr_display_names == [n for n, _ in C.EASYOCR_LANGUAGES_LIST]

    def test_common_languages_present_everywhere(self):
        for lookup in (C.paddle_abbr_lookup, C.lens_abbr_lookup, C.easyocr_abbr_lookup):
            assert "English" in lookup
        assert "Japanese" in C.paddle_abbr_lookup
        assert "Japanese" in C.lens_abbr_lookup
        assert "Japanese + English" in C.easyocr_abbr_lookup
        assert "German" in C.paddle_abbr_lookup
        assert "German" in C.lens_abbr_lookup

    def test_iso_maps_only_contain_known_codes(self):
        for code in C.EASY_TO_ISO_MAP:
            assert code in C.easyocr_abbr_lookup.values()
        # Codes shared with the GUI paddle list must be known; a couple of
        # legacy entries (e.g. 'mo') predate the trimmed GUI language list.
        for code in ("ch", "chinese_cht", "german", "japan", "korean", "rs_latin"):
            assert code in C.paddle_abbr_lookup.values()
            assert code in C.PADDLE_TO_ISO_MAP

    def test_subtitle_lists(self):
        internals = [internal for _, internal in C.SUBTITLE_POSITIONS_LIST]
        assert C.DEFAULT_SUBTITLE_POSITION in internals
        assert len(internals) == len(set(internals))
        aligns = [internal for _, internal in C.SUBTITLE_ALIGNMENT_LIST]
        assert C.DEFAULT_SUBTITLE_ALIGNMENT in aligns
        assert len(aligns) == len(set(aligns)) == 9


class TestDirectMLTables:
    @pytest.mark.parametrize(
        "table",
        [
            C.DIRECTML_PERFORMANCE_PRESETS,
            C.DIRECTML_RECOGNITION_MODES,
            C.DIRECTML_FRAME_SCAN_MODES,
            C.ONNX_DIRECTML_TUNING_MODES,
        ],
    )
    def test_to_cli_matches_preset_tuples(self, table):
        assert dict(table) == dict(table)  # sanity: constructable
        assert len(table) == len(dict(table))

    @pytest.mark.parametrize(
        ("display", "to_cli", "from_cli"),
        [
            (C.DIRECTML_PERFORMANCE_DISPLAY, C.DIRECTML_PERFORMANCE_TO_CLI, C.DIRECTML_PERFORMANCE_FROM_CLI),
            (C.DIRECTML_RECOGNITION_DISPLAY, C.DIRECTML_RECOGNITION_TO_CLI, C.DIRECTML_RECOGNITION_FROM_CLI),
            (C.DIRECTML_FRAME_SCAN_DISPLAY, C.DIRECTML_FRAME_SCAN_TO_CLI, C.DIRECTML_FRAME_SCAN_FROM_CLI),
            (C.ONNX_DIRECTML_TUNING_DISPLAY, C.ONNX_DIRECTML_TUNING_TO_CLI, C.ONNX_DIRECTML_TUNING_FROM_CLI),
        ],
    )
    def test_display_to_cli_round_trip(self, display, to_cli, from_cli):
        assert display == list(to_cli.keys())
        for name, cli in to_cli.items():
            assert from_cli[cli] == name

    def test_recognition_modes_keep_stable_default(self):
        assert C.DIRECTML_RECOGNITION_TO_CLI["Stable Hybrid (recommended)"] == "stable"

    def test_auto_option_is_stable(self):
        assert C.DIRECTML_AUTO_OPTION == "Auto (recommended)"


class TestMiscConstants:
    def test_post_actions_platform(self):
        if sys.platform == "win32":
            assert C.POST_ACTION_KEYS == C.POST_ACTION_KEYS_WINDOWS
        else:
            assert C.POST_ACTION_KEYS == C.POST_ACTION_KEYS_OTHER
        assert "action_none" in C.POST_ACTION_KEYS
        assert set(C.DEFAULT_ACTION_TEXTS) == set(C.POST_ACTION_KEYS_WINDOWS)

    def test_gui_scaling(self):
        internals = [internal for internal, _ in C.GUI_SCALING_LIST]
        assert C.DEFAULT_GUI_SCALING in internals
        assert len(internals) == len(set(internals))

    def test_default_thresholds_within_own_ranges(self):
        assert 0 <= C.DEFAULT_CONF_THRESHOLD <= 100
        assert 0 <= C.DEFAULT_SIM_THRESHOLD <= 100
        assert 0 <= C.DEFAULT_SSIM_THRESHOLD <= 100
        assert C.DEFAULT_FRAMES_TO_SKIP >= 0
        assert C.DEFAULT_MAX_MERGE_GAP >= 0
        assert C.DEFAULT_MIN_SUBTITLE_DURATION >= 0
        assert C.DEFAULT_OCR_IMAGE_MAX_WIDTH >= 0

    def test_defaults_stringified_for_config(self):
        # config.get_default_settings() stores numeric defaults as strings;
        # keep the constants ints/floats but assert the pairing exists.
        from videocr_gui import config

        defaults = config.get_default_settings()
        assert defaults["--conf_threshold"] == str(C.DEFAULT_CONF_THRESHOLD)
        assert defaults["--sim_threshold"] == str(C.DEFAULT_SIM_THRESHOLD)
        assert defaults["--frames_to_skip"] == str(C.DEFAULT_FRAMES_TO_SKIP)
