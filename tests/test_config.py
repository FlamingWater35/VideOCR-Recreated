"""Tests for videocr_gui.config — defaults, INI persistence, path logic, crop parsing."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from videocr_gui import config
from videocr_gui import constants as C


class TestDefaults:
    def test_default_settings_has_expected_core_keys(self):
        defaults = config.get_default_settings()
        for key in (
            "--language", "ocr_engine", "subtitle_language", "subtitle_position",
            "--time_start", "--time_end", "--conf_threshold", "--use_gpu",
            "--directml_device_index", "--saved_crop_boxes", "gui_scaling",
            "enable_label_detection", "prevent_system_sleep",
        ):
            assert key in defaults, key

    def test_defaults_are_fresh_dicts(self):
        a = config.get_default_settings()
        b = config.get_default_settings()
        assert a is not b
        a["--conf_threshold"] = "mutated"
        assert b["--conf_threshold"] != "mutated"

    def test_default_engine_is_valid(self):
        assert config.get_default_settings()["ocr_engine"] in C.OCR_ENGINES

    def test_no_update_or_benchmark_keys(self):
        defaults = config.get_default_settings()
        for key in defaults:
            assert "update" not in key.lower()
            assert "benchmark" not in key.lower()

    def test_directml_index_default_is_zero(self):
        assert config.get_default_settings()["--directml_device_index"] == "0"

    def test_label_settings_defaults_present(self):
        defaults = config.get_default_settings()
        expected = {
            "--label_ssim_dedup": True,
            "--label_hash_threshold": "4",
            "--label_min_zone_height": "32",
            "--label_font": "Arial",
            "--label_fontsize": "22",
            "--label_primary_color": "&H00FFFFFF",
            "--label_outline_color": "&H00000000",
            "--label_alignment": "top-left",
            "--label_lang": "",
        }
        for key, value in expected.items():
            assert defaults[key] == value, key
        # Style/lang keys are not booleans.
        assert "--label_fontsize" not in config._BOOL_KEYS
        assert "--label_lang" not in config._BOOL_KEYS


class TestNormalizeValue:
    @pytest.mark.parametrize("raw", ["1", "true", "True", "TRUE", "yes", "on", " On "])
    def test_truthy(self, raw):
        assert config._normalize_value("--use_gpu", raw) is True

    @pytest.mark.parametrize("raw", ["0", "false", "no", "off", "maybe", "", "2"])
    def test_falsy(self, raw):
        assert config._normalize_value("--use_gpu", raw) is False

    def test_non_bool_key_passes_through(self):
        assert config._normalize_value("--conf_threshold", " 75 ") == " 75 "

    def test_all_bool_keys_documented(self):
        # Every _BOOL_KEYS entry must exist in defaults (catches typos).
        defaults = config.get_default_settings()
        for key in config._BOOL_KEYS:
            assert key in defaults, key


class TestLoadSave:
    def test_missing_file_returns_defaults(self, _tmp_config_file):
        assert not Path(_tmp_config_file).exists()
        settings = config.load_settings()
        assert settings == config.get_default_settings()

    def test_round_trip(self, _tmp_config_file):
        settings = config.get_default_settings()
        settings["--conf_threshold"] = "42"
        settings["--use_gpu"] = False
        settings["ocr_engine"] = C.OCR_ENGINES[1]
        config.save_settings(settings)

        loaded = config.load_settings()
        assert loaded["--conf_threshold"] == "42"
        assert loaded["--use_gpu"] is False  # bool coerced on load
        assert loaded["ocr_engine"] == C.OCR_ENGINES[1]

    def test_bool_written_as_str_read_as_bool(self, _tmp_config_file):
        config.save_settings({"--save_in_video_dir": False})
        assert config.load_settings()["--save_in_video_dir"] is False

    def test_unicode_survives(self, _tmp_config_file):
        settings = config.get_default_settings()
        settings["--default_output_dir"] = "C:/ Videos /日本語"
        config.save_settings(settings)
        assert config.load_settings()["--default_output_dir"] == "C:/ Videos /日本語"

    def test_unknown_keys_ignored(self, _tmp_config_file):
        _tmp_config_file.write_text(
            "[Settings]\n--not_a_real_key = 1\n--conf_threshold = 33\n",
            encoding="utf-8",
        )
        loaded = config.load_settings()
        assert "--not_a_real_key" not in loaded
        assert loaded["--conf_threshold"] == "33"

    def test_missing_section_returns_defaults(self, _tmp_config_file):
        _tmp_config_file.write_text("[Other]\n--conf_threshold = 1\n", encoding="utf-8")
        assert config.load_settings() == config.get_default_settings()

    def test_corrupt_ini_returns_defaults(self, _tmp_config_file, _error_log):
        _tmp_config_file.write_text("not an ini at all {{{", encoding="utf-8")
        loaded = config.load_settings()
        assert loaded == config.get_default_settings()
        assert any("Error parsing config" in m for m in _error_log)

    def test_invalid_utf8_returns_defaults(self, _tmp_config_file, _error_log):
        _tmp_config_file.write_bytes(b"[Settings]\n--conf_threshold = \xff\xfe\xff\n")
        loaded = config.load_settings()
        assert loaded == config.get_default_settings()
        assert _error_log  # parse failure was logged

    def test_garbage_bool_loads_as_false(self, _tmp_config_file):
        _tmp_config_file.write_text(
            "[Settings]\n--use_gpu = maybe\n", encoding="utf-8"
        )
        assert config.load_settings()["--use_gpu"] is False


class TestPathLogic:
    def test_portable_mode_flag(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "APP_DIR", str(tmp_path))
        (tmp_path / "portable_mode.txt").write_text("", encoding="utf-8")
        path = config.get_config_file_path()
        assert path == str(tmp_path / "videocr_gui_config.ini")

    @pytest.mark.skipif(os.name != "nt", reason="Windows APPDATA branch")
    def test_windows_uses_appdata(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "APP_DIR", str(tmp_path / "app"))
        monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
        path = config.get_config_file_path()
        assert path == str(tmp_path / "roaming" / "VideOCR" / "videocr_gui_config.ini")

    @pytest.mark.skipif(os.name == "nt", reason="POSIX XDG branch")
    def test_posix_uses_xdg_config_home(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "APP_DIR", str(tmp_path / "app"))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        path = config.get_config_file_path()
        assert path == str(tmp_path / "xdg" / "VideOCR" / "videocr_gui_config.ini")

    def test_log_error_writes_file(self, monkeypatch, tmp_path, _error_log):
        real_log_error = _error_log.originals["config"]
        monkeypatch.setattr(config, "log_error", real_log_error)
        monkeypatch.setattr(config, "APP_DIR", str(tmp_path / "app"))
        # Not portable, so we control the state dir via env.
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        if os.name == "nt":
            monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
        log_path = real_log_error("test message", "my_log.txt")
        assert Path(log_path).exists()
        content = Path(log_path).read_text(encoding="utf-8")
        assert "test message" in content
        # Timestamped.
        assert "[" in content


class TestBuildVariant:
    @pytest.mark.parametrize(
        ("variant", "cuda", "directml"),
        [
            ("cpu", False, False),
            ("gpu-cuda11.8", True, False),
            ("gpu-cuda12.9", True, False),
            ("gpu-directml", False, True),
        ],
    )
    def test_packaged_variant_capabilities(self, _tmp_build_variant, variant, cuda, directml):
        _tmp_build_variant.write_text(variant + "\n", encoding="utf-8")
        assert config.get_build_variant() == variant
        assert config.supports_cuda() is cuda
        assert config.supports_directml() is directml

    @pytest.mark.parametrize("content", [None, "", "gpu-turbo\n"])
    def test_missing_or_unrecognised_marker_enables_everything(
        self, _tmp_build_variant, content
    ):
        if content is not None:
            _tmp_build_variant.write_text(content, encoding="utf-8")
        assert config.get_build_variant() == "unknown"
        assert config.supports_cuda() is True
        assert config.supports_directml() is True

    @pytest.mark.parametrize(
        ("variant", "onnx_selectable"),
        [
            ("cpu", False),
            ("gpu-cuda11.8", False),
            ("gpu-cuda12.9", False),
            ("gpu-directml", True),
            ("unknown", True),
        ],
    )
    def test_onnx_engine_follows_directml_support(
        self, _tmp_build_variant, variant, onnx_selectable
    ):
        if variant != "unknown":
            _tmp_build_variant.write_text(variant + "\n", encoding="utf-8")
        onnx = C.OCR_ENGINES[2]
        assert config.supports_engine(onnx) is onnx_selectable
        # The PaddleOCR engines are packaged in every target.
        assert config.supports_engine(C.OCR_ENGINES[0]) is True
        assert config.supports_engine(C.OCR_ENGINES[1]) is True

    def test_directml_only_set_names_the_onnx_engine(self):
        assert {C.OCR_ENGINES[2]} == C.DIRECTML_ONLY_ENGINES


class TestLegacyGpuFlagMigration:
    def test_combined_flag_carries_over_to_directml_toggle(self, _tmp_config_file):
        _tmp_config_file.write_text("[Settings]\n--use_gpu = false\n", encoding="utf-8")
        loaded = config.load_settings()
        assert loaded["--use_gpu"] is False
        assert loaded["--use_directml_gpu"] is False

    def test_split_flag_is_left_alone(self, _tmp_config_file):
        _tmp_config_file.write_text(
            "[Settings]\n--use_gpu = true\n--use_directml_gpu = false\n",
            encoding="utf-8",
        )
        loaded = config.load_settings()
        assert loaded["--use_gpu"] is True
        assert loaded["--use_directml_gpu"] is False

    def test_no_combined_flag_keeps_the_default(self, _tmp_config_file):
        _tmp_config_file.write_text(
            "[Settings]\n--conf_threshold = 40\n", encoding="utf-8"
        )
        loaded = config.load_settings()
        assert loaded["--use_directml_gpu"] is True


class TestParseSavedCropBoxes:
    def test_valid_list(self):
        raw = repr([{"crop_x": 0.1, "crop_y": 0.2, "crop_width": 0.5, "crop_height": 0.1}])
        boxes = config.parse_saved_crop_boxes(raw)
        assert len(boxes) == 1
        assert boxes[0]["crop_x"] == 0.1

    def test_empty_list(self):
        assert config.parse_saved_crop_boxes("[]") == []

    @pytest.mark.parametrize("raw", ["", "not a list", "(1 + 2)"])
    def test_invalid_returns_empty_and_logs(self, raw, _error_log):
        assert config.parse_saved_crop_boxes(raw) == []
        assert any("Could not parse saved_crop_boxes" in m for m in _error_log)

    @pytest.mark.parametrize("raw", ["{1: 2}", "({'a': 1})", "(1, 2)", "'string'", "42"])
    def test_valid_non_list_literal_returns_empty_silently(self, raw, _error_log):
        assert config.parse_saved_crop_boxes(raw) == []
        assert not any("Could not parse saved_crop_boxes" in m for m in _error_log)
