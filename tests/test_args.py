"""Tests for videocr_gui.args — time validation, output paths, build_args mapping."""

from __future__ import annotations

import pytest

from videocr_gui import args as args_mod
from videocr_gui import config, i18n
from videocr_gui import constants as C


@pytest.fixture(autouse=True)
def _english():
    i18n.load_language("en")


def _default_settings(**overrides):
    settings = config.get_default_settings()
    settings.update(overrides)
    return settings


class TestIsValidTime:
    @pytest.mark.parametrize(
        "value",
        ["", None, "0:00", "1:30", "10:59", "0:00:00", "1:30:00", "2:05:59", "0:00", "12:34"],
    )
    def test_valid(self, value):
        assert args_mod._is_valid_time(value) is True

    @pytest.mark.parametrize(
        "value",
        ["1", "a:b", "1:2:3:4", "0:75", "1:60:00", "-1:00", "x:30", "1:2:60", "::"],
    )
    def test_invalid(self, value):
        assert args_mod._is_valid_time(value) is False


class TestTimeToSeconds:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("0:00", 0), ("1:30", 90), ("0:01:30", 90), ("1:02:03", 3723), ("2:05", 125)],
    )
    def test_valid(self, value, expected):
        assert args_mod._time_to_seconds(value) == expected

    @pytest.mark.parametrize("value", ["", None, "bogus", "1:2:3:4", "a:b"])
    def test_invalid(self, value):
        assert args_mod._time_to_seconds(value) is None


class TestFmtTime:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(0, "00:00"), (59, "00:59"), (60, "01:00"), (3599, "59:59"), (3600, "01:00:00"), (7325, "02:02:05")],
    )
    def test_formats(self, seconds, expected):
        assert args_mod._fmt_time(seconds) == expected


class TestLanguageIsoCode:
    def test_paddle_english(self):
        assert args_mod._language_iso_code(_default_settings()) == "en"

    def test_paddle_german_maps_to_iso(self):
        s = _default_settings(subtitle_language="German")
        assert args_mod._language_iso_code(s) == "de"

    def test_paddle_japanese(self):
        s = _default_settings(subtitle_language="Japanese")
        assert args_mod._language_iso_code(s) == "ja"

    def test_lens_engine_uses_lens_lookup(self):
        s = _default_settings(
            ocr_engine="PaddleOCR (Det.) + Google Lens (Rec.)",
            subtitle_language="German",  # lens 'de' (no PADDLE remap)
        )
        assert args_mod._language_iso_code(s) == "de"

    def test_onnx_engine_uses_easyocr_lookup(self):
        s = _default_settings(
            ocr_engine=C.OCR_ENGINES[2], subtitle_language="Japanese + English"
        )
        assert args_mod._language_iso_code(s) == "ja"

    def test_unknown_language_defaults_to_en(self):
        s = _default_settings(subtitle_language="Klingon")
        assert args_mod._language_iso_code(s) == "en"


class TestGenerateOutputPath:
    def test_in_video_dir_with_srt(self, tmp_path):
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"")
        out = args_mod.generate_output_path(str(video), _default_settings())
        assert out == tmp_path / "movie.en.srt"

    def test_ass_when_label_detection_enabled(self, tmp_path):
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"")
        s = _default_settings(enable_label_detection=True)
        out = args_mod.generate_output_path(str(video), s)
        assert out.suffix == ".ass"

    def test_language_in_filename(self, tmp_path):
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"")
        s = _default_settings(subtitle_language="German")
        out = args_mod.generate_output_path(str(video), s)
        assert out.name == "movie.de.srt"

    def test_collision_appends_counter(self, tmp_path):
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"")
        (tmp_path / "movie.en.srt").write_text("x", encoding="utf-8")
        (tmp_path / "movie(1).en.srt").write_text("x", encoding="utf-8")
        out = args_mod.generate_output_path(str(video), _default_settings())
        assert out.name == "movie(2).en.srt"

    def test_respects_output_dir_setting(self, tmp_path):
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"")
        out_dir = tmp_path / "subs"
        out_dir.mkdir()
        s = _default_settings(**{"--save_in_video_dir": False, "--default_output_dir": str(out_dir)})
        out = args_mod.generate_output_path(str(video), s)
        assert out.parent == out_dir

    def test_blank_output_dir_falls_back_to_default_dir(self, tmp_path):
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"")
        s = _default_settings(**{"--save_in_video_dir": False, "--default_output_dir": "   "})
        out = args_mod.generate_output_path(str(video), s, default_dir=str(tmp_path))
        assert out.parent == tmp_path


class TestBuildArgsValidation:
    def test_defaults_pass(self):
        argd, errors = args_mod.build_args("video.mp4", _default_settings(), [], output_path="out.srt")
        assert errors == []
        assert argd is not None
        assert argd["video_path"] == "video.mp4"
        assert argd["output"] == "out.srt"

    @pytest.mark.parametrize("bad", ["1", "a:b", "0:75", "1:60:00"])
    def test_bad_start_time(self, bad):
        argd, errors = args_mod.build_args("v.mp4", _default_settings(**{"--time_start": bad}), [])
        assert argd is None
        assert any("Invalid Start Time" in e for e in errors)

    @pytest.mark.parametrize("bad", ["1", "a:b", "0:99"])
    def test_bad_end_time(self, bad):
        argd, errors = args_mod.build_args("v.mp4", _default_settings(**{"--time_end": bad}), [])
        assert argd is None
        assert any("Invalid End Time" in e for e in errors)

    def test_bad_label_times(self):
        s = _default_settings(**{"--label_time_start": "zz", "--label_time_end": "qq"})
        argd, errors = args_mod.build_args("v.mp4", s, [])
        assert argd is None
        assert any("Label Start Time" in e for e in errors)
        assert any("Label End Time" in e for e in errors)

    def test_start_after_end(self):
        s = _default_settings(**{"--time_start": "0:10", "--time_end": "0:05"})
        argd, errors = args_mod.build_args("v.mp4", s, [])
        assert argd is None
        assert any("Start Time cannot be after End Time" in e for e in errors)

    def test_start_exceeds_duration(self):
        s = _default_settings(**{"--time_start": "1:00", "_video_duration_ms": 30_000})
        argd, errors = args_mod.build_args("v.mp4", s, [])
        assert argd is None
        assert any("exceeds video duration" in e for e in errors)

    def test_end_exceeds_duration(self):
        s = _default_settings(**{"--time_end": "1:00", "_video_duration_ms": 30_000})
        argd, errors = args_mod.build_args("v.mp4", s, [])
        assert argd is None
        assert any("exceeds video duration" in e for e in errors)

    def test_label_start_after_end(self):
        s = _default_settings(**{"--label_time_start": "0:10", "--label_time_end": "0:05"})
        argd, errors = args_mod.build_args("v.mp4", s, [])
        assert argd is None
        assert any("Label Start Time cannot be after Label End Time" in e for e in errors)

    @pytest.mark.parametrize(
        ("key", "bad"),
        [
            ("--label_pos_drift", "abc"),
            ("--label_hash_threshold", "99"),
            ("--label_min_zone_height", "4"),
            ("--label_fontsize", "0"),
            ("--label_conf_threshold", "150"),
            ("--label_close_pos_length_ratio", "2.5"),
        ],
    )
    def test_label_numeric_validation(self, key, bad):
        s = _default_settings(**{key: bad})
        argd, errors = args_mod.build_args("v.mp4", s, [])
        assert argd is None, key
        assert any("Invalid value" in e for e in errors), errors

    def test_dual_zone_requires_two_boxes(self):
        s = _default_settings(**{"--use_dual_zone": True})
        box = {"coords": {"crop_x": 0, "crop_y": 0, "crop_width": 10, "crop_height": 10}}
        argd, errors = args_mod.build_args("v.mp4", s, [box])
        assert argd is None
        assert any("2 crop boxes" in e for e in errors)

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("--conf_threshold", "101"),
            ("--conf_threshold", "-1"),
            ("--conf_threshold", "abc"),
            ("--sim_threshold", "150"),
            ("--brightness_threshold", "300"),
            ("--ssim_threshold", "-5"),
            ("--directml_grid_max_width", "100"),
            ("--directml_grid_max_height", "511"),
            ("--frames_to_skip", "-1"),
            ("--max_merge_gap", "-0.5"),
            ("--min_subtitle_duration", "-1"),
            ("--directml_device_index", "abc"),
        ],
    )
    def test_numeric_out_of_range(self, key, value):
        s = _default_settings(**{key: value})
        argd, errors = args_mod.build_args("v.mp4", s, [])
        assert argd is None
        assert any("Invalid value" in e for e in errors)

    def test_empty_numeric_skipped(self):
        s = _default_settings(**{"--brightness_threshold": ""})
        argd, errors = args_mod.build_args("v.mp4", s, [], output_path="o.srt")
        assert errors == []
        assert "brightness_threshold" not in argd

    def test_multiple_errors_collected(self):
        s = _default_settings(**{"--time_start": "bad", "--conf_threshold": "200"})
        argd, errors = args_mod.build_args("v.mp4", s, [])
        assert argd is None
        assert len(errors) >= 2


class TestBuildArgsMapping:
    @pytest.mark.parametrize(
        ("engine", "expected"),
        [
            ("PaddleOCR (Det. + Rec.)", "paddleocr"),
            ("PaddleOCR (Det.) + Google Lens (Rec.)", "google_lens"),
            ("ONNX Runtime DirectML (AMD GPU Experimental)", "onnx_directml"),
            ("EasyOCR DirectML (AMD GPU)", "easyocr_directml"),  # legacy name still maps
        ],
    )
    def test_engine_mapping(self, engine, expected):
        s = _default_settings(ocr_engine=engine)
        argd, errors = args_mod.build_args("v.mp4", s, [], output_path="o.srt")
        assert errors == []
        assert argd["ocr_engine"] == expected

    def test_language_set_from_lookup(self):
        s = _default_settings(subtitle_language="German")
        argd, _ = args_mod.build_args("v.mp4", s, [], output_path="o.srt")
        assert argd["lang"] == "german"  # paddle abbreviation

    def test_position_internal_value(self):
        s = _default_settings(subtitle_position="left")
        argd, _ = args_mod.build_args("v.mp4", s, [], output_path="o.srt")
        assert argd["subtitle_position"] == "left"

    def test_position_localized_display_resolves(self):
        display = i18n.tr("pos_left")  # e.g. "Left"
        s = _default_settings(subtitle_position=display)
        argd, _ = args_mod.build_args("v.mp4", s, [], output_path="o.srt")
        assert argd["subtitle_position"] == "left"

    def test_unknown_position_falls_back_to_default(self):
        s = _default_settings(subtitle_position="diagonal")
        argd, _ = args_mod.build_args("v.mp4", s, [], output_path="o.srt")
        assert argd["subtitle_position"] == C.DEFAULT_INTERNAL_SUBTITLE_POSITION

    def test_generic_pass_through(self):
        s = _default_settings(**{"--conf_threshold": "60", "--time_start": "0:05"})
        argd, _ = args_mod.build_args("v.mp4", s, [], output_path="o.srt")
        assert argd["conf_threshold"] == "60"
        assert argd["time_start"] == "0:05"

    def test_gui_only_keys_excluded(self):
        argd, _ = args_mod.build_args("v.mp4", _default_settings(), [], output_path="o.srt")
        for key in (
            "keyboard_seek_step", "default_output_dir", "save_in_video_dir",
            "save_crop_box", "language", "use_dual_zone", "subtitle_alignment",
            "saved_crop_boxes", "use_dual_zone",
        ):
            assert key not in argd, key

    def test_bools_stay_bool(self):
        s = _default_settings(**{"--use_gpu": False, "--use_angle_cls": True})
        argd, _ = args_mod.build_args("v.mp4", s, [], output_path="o.srt")
        assert argd["use_gpu"] is False
        assert argd["use_angle_cls"] is True

    @pytest.mark.parametrize(
        ("display", "expected_cli"),
        [
            ("Compatibility / Low VRAM", "compatibility"),
            ("Balanced (recommended)", "balanced"),
            ("Max AMD GPU Load", "max"),
            ("Manual Grid Size", "manual"),
        ],
    )
    def test_directml_performance_to_cli(self, display, expected_cli):
        s = _default_settings(**{"--directml_performance_preset": display})
        argd, _ = args_mod.build_args("v.mp4", s, [], output_path="o.srt")
        assert argd["directml_performance_preset"] == expected_cli

    @pytest.mark.parametrize("display", ["Stable Hybrid (recommended)", "AMD Max Auto (try GPU recognition + fallback)", "Experimental Full DirectML"])
    def test_directml_recognition_to_cli(self, display):
        s = _default_settings(**{"--directml_recognition_mode": display})
        argd, _ = args_mod.build_args("v.mp4", s, [], output_path="o.srt")
        assert argd["directml_recognition_mode"] in ("stable", "auto", "experimental")

    def test_alignment_only_when_enabled(self):
        s = _default_settings(**{"--subtitle_alignment": "top-left"})
        argd, _ = args_mod.build_args("v.mp4", s, [], output_path="o.srt")
        assert "subtitle_alignment" not in argd

        s["enable_subtitle_alignment"] = True
        argd, _ = args_mod.build_args("v.mp4", s, [], output_path="o.srt")
        assert argd["subtitle_alignment"] == "top-left"
        assert "subtitle_alignment2" not in argd  # not dual-zone

    def test_alignment_localized_display_resolves(self):
        display = i18n.tr("align_top_right")
        s = _default_settings(
            enable_subtitle_alignment=True, **{"--subtitle_alignment": display}
        )
        argd, _ = args_mod.build_args("v.mp4", s, [], output_path="o.srt")
        assert argd["subtitle_alignment"] == "top-right"

    def test_dual_alignment_in_dual_zone(self):
        box = {"coords": {"crop_x": 0, "crop_y": 0, "crop_width": 10, "crop_height": 10}}
        s = _default_settings(
            **{"--use_dual_zone": True, "enable_subtitle_alignment": True,
               "--subtitle_alignment": "top-left", "--subtitle_alignment2": "bottom-right"}
        )
        argd, errors = args_mod.build_args("v.mp4", s, [box, box], output_path="o.srt")
        assert errors == []
        assert argd["subtitle_alignment"] == "top-left"
        assert argd["subtitle_alignment2"] == "bottom-right"

    def test_label_detection_flag(self):
        argd, _ = args_mod.build_args(
            "v.mp4", _default_settings(enable_label_detection=True), [], output_path="o.ass"
        )
        assert argd["enable_label_detection"] is True

    def test_send_notification_and_sleep(self):
        argd, _ = args_mod.build_args("v.mp4", _default_settings(), [], output_path="o.srt")
        assert argd["send_notification"] is True  # default
        assert argd["allow_system_sleep"] is True  # always

    def test_crop_box_applied(self):
        box = {"coords": {"crop_x": 10, "crop_y": 20, "crop_width": 300, "crop_height": 80}}
        argd, _ = args_mod.build_args("v.mp4", _default_settings(), [box], output_path="o.srt")
        assert argd["crop_x"] == 10
        assert argd["crop_height"] == 80

    def test_fullframe_ignores_crop_box(self):
        box = {"coords": {"crop_x": 10, "crop_y": 20, "crop_width": 300, "crop_height": 80}}
        s = _default_settings(**{"--use_fullframe": True})
        argd, _ = args_mod.build_args("v.mp4", s, [box], output_path="o.srt")
        assert "crop_x" not in argd

    def test_dual_zone_boxes(self):
        b1 = {"coords": {"crop_x": 1, "crop_y": 2, "crop_width": 3, "crop_height": 4}}
        b2 = {"coords": {"crop_x": 5, "crop_y": 6, "crop_width": 7, "crop_height": 8}}
        s = _default_settings(**{"--use_dual_zone": True})
        argd, errors = args_mod.build_args("v.mp4", s, [b1, b2], output_path="o.srt")
        assert errors == []
        assert argd["crop_x"] == 1
        assert argd["crop_x2"] == 5
        assert argd["crop_height2"] == 8

    def test_output_path_override_used(self):
        argd, _ = args_mod.build_args(
            "v.mp4", _default_settings(), [], output_path="custom path/out.srt"
        )
        assert argd["output"] == "custom path/out.srt"

    def test_generated_output_path_used_when_no_override(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"")
        argd, _ = args_mod.build_args(str(video), _default_settings(), [])
        assert argd["output"] == str(tmp_path / "clip.en.srt")

    def test_output_with_spaces_stays_single_value(self):
        argd, _ = args_mod.build_args(
            "v.mp4", _default_settings(), [], output_path="/tmp/my videos/out.srt"
        )
        assert argd["output"] == "/tmp/my videos/out.srt"


class TestUseGpuMapping:
    """The CLI keeps a single --use_gpu flag; the GUI tracks CUDA and DirectML apart.

    Which toggle feeds the flag depends on the engine (CUDA for PaddleOCR,
    DirectML for ONNX), and a package that does not ship the backend forces it
    off regardless of what a stale config still carries.
    """

    @staticmethod
    def _build(marker, variant, **overrides):
        marker.write_text(variant + "\n", encoding="utf-8")
        argd, errors = args_mod.build_args(
            "v.mp4", _default_settings(**overrides), [], output_path="o.srt"
        )
        assert errors == []
        return argd

    @pytest.mark.parametrize("variant", ["cpu", "gpu-cuda11.8", "gpu-cuda12.9", "gpu-directml"])
    def test_directml_toggle_never_reaches_the_cli(self, _tmp_build_variant, variant):
        argd = self._build(_tmp_build_variant, variant, ocr_engine=C.OCR_ENGINES[2])
        assert "use_directml_gpu" not in argd

    @pytest.mark.parametrize("variant", ["gpu-cuda11.8", "gpu-cuda12.9"])
    def test_paddle_engine_follows_the_cuda_toggle(self, _tmp_build_variant, variant):
        assert self._build(_tmp_build_variant, variant, **{"--use_gpu": True})["use_gpu"] is True
        assert self._build(_tmp_build_variant, variant, **{"--use_gpu": False})["use_gpu"] is False

    @pytest.mark.parametrize("variant", ["cpu", "gpu-directml"])
    def test_cuda_toggle_is_forced_off_by_a_non_cuda_build(self, _tmp_build_variant, variant):
        assert self._build(_tmp_build_variant, variant, **{"--use_gpu": True})["use_gpu"] is False

    @pytest.mark.parametrize("variant", ["cpu", "gpu-cuda11.8", "gpu-cuda12.9"])
    def test_directml_toggle_is_forced_off_outside_the_directml_build(
        self, _tmp_build_variant, variant
    ):
        argd = self._build(
            _tmp_build_variant,
            variant,
            ocr_engine=C.OCR_ENGINES[2],
            **{"--use_directml_gpu": True},
        )
        assert argd["use_gpu"] is False

    def test_onnx_engine_follows_the_directml_toggle(self, _tmp_build_variant):
        assert self._build(
            _tmp_build_variant,
            "gpu-directml",
            ocr_engine=C.OCR_ENGINES[2],
            **{"--use_directml_gpu": True},
        )["use_gpu"] is True
        assert self._build(
            _tmp_build_variant,
            "gpu-directml",
            ocr_engine=C.OCR_ENGINES[2],
            **{"--use_directml_gpu": False},
        )["use_gpu"] is False

    def test_google_lens_engine_follows_the_cuda_toggle(self, _tmp_build_variant):
        assert self._build(
            _tmp_build_variant,
            "gpu-cuda12.9",
            ocr_engine=C.OCR_ENGINES[1],
            **{"--use_gpu": True},
        )["use_gpu"] is True
        assert self._build(
            _tmp_build_variant,
            "cpu",
            ocr_engine=C.OCR_ENGINES[1],
            **{"--use_gpu": True},
        )["use_gpu"] is False
