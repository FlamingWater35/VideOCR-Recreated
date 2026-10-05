"""Tests for CLI/videocr_cli.py — argparse validators and main() flag mapping.

``main()`` reads ``sys.argv`` and calls ``save_subtitles_to_file`` (imported by
value), so tests replace both to exercise validation and flag→kwarg mapping
without any OCR.
"""

from __future__ import annotations

import argparse
import sys

import pytest
import videocr_cli

from videocr import utils

# --- validators (directly callable) -------------------------------------------


class TestValidVideoPath:
    def test_existing_file(self, tmp_path):
        video = tmp_path / "a.mp4"
        video.write_bytes(b"")
        assert videocr_cli.valid_video_path(str(video)) == str(video)

    def test_missing_file_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="does not exist"):
            videocr_cli.valid_video_path("/no/such/file.mp4")

    def test_directory_raises(self, tmp_path):
        with pytest.raises(argparse.ArgumentTypeError):
            videocr_cli.valid_video_path(str(tmp_path))


class TestValidOutputPath:
    def test_existing_dir(self, tmp_path):
        out = tmp_path / "out.srt"
        assert videocr_cli.valid_output_path(str(out)) == str(out)

    def test_relative_name_uses_cwd(self):
        assert videocr_cli.valid_output_path("subtitle.srt") == "subtitle.srt"

    def test_missing_dir_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="does not exist"):
            videocr_cli.valid_output_path("/no/such/dir/out.srt")


class TestRestrictedInt:
    def test_valid_value(self):
        assert videocr_cli.restricted_int(0, 100)("42") == 42

    def test_non_int_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="Must be an integer"):
            videocr_cli.restricted_int()("abc")

    def test_min_violation(self):
        with pytest.raises(argparse.ArgumentTypeError, match=">= 0"):
            videocr_cli.restricted_int(min_val=0)("-1")


class TestValidAssColor:
    def test_valid_normalized_uppercase(self):
        assert videocr_cli.valid_ass_color("&H00FFFFFF") == "&H00FFFFFF"
        assert videocr_cli.valid_ass_color("&h00ffffff") == "&H00FFFFFF"

    @pytest.mark.parametrize("value", ["#FFFFFF", "&H00FFF", "&H00GGGGGG", "red", ""])
    def test_invalid_raises(self, value):
        with pytest.raises(argparse.ArgumentTypeError, match="ASS color"):
            videocr_cli.valid_ass_color(value)

    def test_max_violation(self):
        with pytest.raises(argparse.ArgumentTypeError, match="<= 100"):
            videocr_cli.restricted_int(0, 100)("101")

    def test_bounds_inclusive(self):
        v = videocr_cli.restricted_int(0, 100)
        assert v("0") == 0
        assert v("100") == 100


class TestRestrictedFloat:
    def test_valid_value(self):
        assert videocr_cli.restricted_float(min_val=0.0)("0.5") == 0.5

    def test_non_float_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="decimal number"):
            videocr_cli.restricted_float()("abc")

    def test_min_violation(self):
        with pytest.raises(argparse.ArgumentTypeError):
            videocr_cli.restricted_float(min_val=0.0)("-0.1")


class TestValidTimeString:
    @pytest.mark.parametrize("value", ["", "0:00", "1:30", "01:02:03"])
    def test_valid(self, value):
        assert videocr_cli.valid_time_string(value) == value

    @pytest.mark.parametrize("value", ["bogus", "1", "1:2:3:4"])
    def test_invalid(self, value):
        with pytest.raises(argparse.ArgumentTypeError, match="Invalid time format"):
            videocr_cli.valid_time_string(value)


class TestValidAlignmentName:
    @pytest.mark.parametrize(
        ("name", "code"),
        [
            ("bottom-left", "an1"), ("bottom-center", "an2"), ("bottom-right", "an3"),
            ("middle-left", "an4"), ("middle-center", "an5"), ("middle-right", "an6"),
            ("top-left", "an7"), ("top-center", "an8"), ("top-right", "an9"),
        ],
    )
    def test_valid(self, name, code):
        assert videocr_cli.valid_alignment_name(name) == code

    def test_empty_returns_none(self):
        assert videocr_cli.valid_alignment_name("") is None

    def test_invalid_raises_with_allowed_list(self):
        with pytest.raises(argparse.ArgumentTypeError, match="Invalid alignment"):
            videocr_cli.valid_alignment_name("diagonal")


# --- main() integration with a fake save_subtitles_to_file --------------------

@pytest.fixture
def fake_save(monkeypatch):
    """Replace save_subtitles_to_file with a recorder; returns the recorder list."""
    calls: list[dict] = []

    def _record(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(videocr_cli, "save_subtitles_to_file", _record)
    return calls


@pytest.fixture(autouse=True)
def _clean_directml_env(monkeypatch):
    for key in (
        "VIDEOCR_DIRECTML_DEVICE_INDEX",
        "VIDEOCR_DIRECTML_RECOGNITION_MODE",
        "VIDEOCR_DIRECTML_AUTO_PREFER_HIGH_PERFORMANCE",
        "VIDEOCR_ONNX_DIRECTML_TUNING",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def video_file(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"")
    return video


def run_main(monkeypatch, *extra_args):
    monkeypatch.setattr(sys, "argv", ["videocr-cli", *extra_args])
    videocr_cli.main()


class TestMainParsing:
    def test_help_exits_zero(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            run_main(monkeypatch, "--help")
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "--video_path" in out
        assert "--ocr_engine" in out

    def test_missing_required_video_path(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            run_main(monkeypatch)
        assert exc.value.code == 2
        assert "--video_path" in capsys.readouterr().err

    def test_invalid_video_path_exits_two(self, monkeypatch):
        with pytest.raises(SystemExit) as exc:
            run_main(monkeypatch, "--video_path", "/does/not/exist.mp4",
                     "--allow_system_sleep", "true")
        assert exc.value.code == 2

    def test_invalid_conf_threshold_exits_two(self, monkeypatch, video_file):
        with pytest.raises(SystemExit) as exc:
            run_main(monkeypatch, "--video_path", str(video_file),
                     "--conf_threshold", "150", "--allow_system_sleep", "true")
        assert exc.value.code == 2

    def test_invalid_engine_choice(self, monkeypatch, video_file):
        with pytest.raises(SystemExit) as exc:
            run_main(monkeypatch, "--video_path", str(video_file),
                     "--ocr_engine", "tesseract", "--allow_system_sleep", "true")
        assert exc.value.code == 2

    def test_invalid_time_string_exits_two(self, monkeypatch, video_file):
        with pytest.raises(SystemExit) as exc:
            run_main(monkeypatch, "--video_path", str(video_file),
                     "--time_start", "garbage", "--allow_system_sleep", "true")
        assert exc.value.code == 2


class TestMainFlagMapping:
    def test_defaults_passed_through(self, monkeypatch, video_file, fake_save):
        run_main(monkeypatch, "--video_path", str(video_file),
                 "--allow_system_sleep", "true")
        assert len(fake_save) == 1
        kwargs = fake_save[0]
        assert kwargs["video_path"] == str(video_file)
        assert kwargs["file_path"] == "subtitle.srt"
        assert kwargs["ocr_engine"] == "paddleocr"
        assert kwargs["lang"] == "en"
        assert kwargs["conf_threshold"] == 75
        assert kwargs["sim_threshold"] == 80
        assert kwargs["frames_to_skip"] == 1
        assert kwargs["crop_zones"] == []
        assert kwargs["subtitle_alignments"] == [None, None]
        assert kwargs["use_gpu"] is False

    @pytest.mark.parametrize(
        "engine", ["paddleocr", "google_lens", "easyocr_directml", "onnx_directml"]
    )
    def test_engine_forwarded(self, monkeypatch, video_file, fake_save, engine):
        run_main(monkeypatch, "--video_path", str(video_file), "--ocr_engine", engine,
                 "--allow_system_sleep", "true")
        assert fake_save[0]["ocr_engine"] == engine

    def test_thresholds_and_ranges_forwarded(self, monkeypatch, video_file, fake_save):
        run_main(
            monkeypatch, "--video_path", str(video_file),
            "--conf_threshold", "40", "--sim_threshold", "90",
            "--max_merge_gap", "0.5", "--brightness_threshold", "120",
            "--ssim_threshold", "85", "--min_subtitle_duration", "1.5",
            "--ocr_image_max_width", "1000", "--frames_to_skip", "2",
            "--allow_system_sleep", "true",
        )
        kwargs = fake_save[0]
        assert kwargs["conf_threshold"] == 40
        assert kwargs["sim_threshold"] == 90
        assert kwargs["max_merge_gap_sec"] == 0.5
        assert kwargs["brightness_threshold"] == 120
        assert kwargs["ssim_threshold"] == 85
        assert kwargs["min_subtitle_duration_sec"] == 1.5
        assert kwargs["ocr_image_max_width"] == 1000
        assert kwargs["frames_to_skip"] == 2

    def test_bool_flag_true(self, monkeypatch, video_file, fake_save):
        run_main(monkeypatch, "--video_path", str(video_file),
                 "--use_gpu", "true", "--use_fullframe", "true",
                 "--allow_system_sleep", "true")
        kwargs = fake_save[0]
        assert kwargs["use_gpu"] is True
        assert kwargs["use_fullframe"] is True

    def test_bool_flag_only_exact_true_is_true(self, monkeypatch, video_file, fake_save):
        run_main(monkeypatch, "--video_path", str(video_file),
                 "--use_gpu", "yes", "--post_processing", "1",
                 "--allow_system_sleep", "true")
        kwargs = fake_save[0]
        assert kwargs["use_gpu"] is False      # only "true" counts
        assert kwargs["post_processing"] is False

    def test_crop_zones_built(self, monkeypatch, video_file, fake_save):
        run_main(monkeypatch, "--video_path", str(video_file),
                 "--crop_x", "10", "--crop_y", "20",
                 "--crop_width", "300", "--crop_height", "80",
                 "--crop_x2", "500", "--crop_y2", "600",
                 "--crop_width2", "700", "--crop_height2", "90",
                 "--allow_system_sleep", "true")
        assert fake_save[0]["crop_zones"] == [
            {"x": 10, "y": 20, "width": 300, "height": 80},
            {"x": 500, "y": 600, "width": 700, "height": 90},
        ]

    def test_alignment_converted_to_an_codes(self, monkeypatch, video_file, fake_save):
        run_main(monkeypatch, "--video_path", str(video_file),
                 "--subtitle_alignment", "top-left",
                 "--subtitle_alignment2", "bottom-right",
                 "--allow_system_sleep", "true")
        assert fake_save[0]["subtitle_alignments"] == ["an7", "an3"]

    def test_label_detection_flags(self, monkeypatch, video_file, fake_save):
        run_main(monkeypatch, "--video_path", str(video_file),
                 "--enable_label_detection", "true",
                 "--label_time_start", "0:05", "--label_time_end", "0:10",
                 "--label_conf_threshold", "50",
                 "--allow_system_sleep", "true")
        kwargs = fake_save[0]
        assert kwargs["enable_label_detection"] is True
        assert kwargs["label_time_start"] == "0:05"
        assert kwargs["label_time_end"] == "0:10"
        assert kwargs["label_conf_threshold"] == 50

    def test_label_style_flags(self, monkeypatch, video_file, fake_save):
        run_main(monkeypatch, "--video_path", str(video_file),
                 "--enable_label_detection", "true",
                 "--label_font", "Verdana",
                 "--label_fontsize", "30",
                 "--label_primary_color", "&h0000ff00",
                 "--label_outline_color", "&H000000FF",
                 "--label_alignment", "top-right",
                 "--label_min_zone_height", "48",
                 "--label_hash_threshold", "6",
                 "--allow_system_sleep", "true")
        kwargs = fake_save[0]
        assert kwargs["label_font"] == "Verdana"
        assert kwargs["label_fontsize"] == 30
        assert kwargs["label_primary_color"] == "&H0000FF00"  # normalized upper
        assert kwargs["label_outline_color"] == "&H000000FF"
        assert kwargs["label_alignment"] == "an9"
        assert kwargs["label_min_zone_height"] == 48
        assert kwargs["label_hash_threshold"] == 6

    def test_label_lang_flag(self, monkeypatch, video_file, fake_save):
        run_main(monkeypatch, "--video_path", str(video_file),
                 "--enable_label_detection", "true",
                 "--label_lang", "japan",
                 "--allow_system_sleep", "true")
        assert fake_save[0]["label_lang"] == "japan"

    def test_label_lang_empty_by_default(self, monkeypatch, video_file, fake_save):
        run_main(monkeypatch, "--video_path", str(video_file),
                 "--enable_label_detection", "true",
                 "--allow_system_sleep", "true")
        assert fake_save[0]["label_lang"] == ""


class TestMainValidationErrors:
    def _expect_exit(self, monkeypatch, video_file, capsys, *extra):
        with pytest.raises(SystemExit) as exc:
            run_main(monkeypatch, "--video_path", str(video_file),
                     "--allow_system_sleep", "true", *extra)
        assert exc.value.code == 1
        return capsys.readouterr().out

    def test_unsupported_paddle_lang(self, monkeypatch, video_file, capsys, fake_save):
        out = self._expect_exit(monkeypatch, video_file, capsys,
                                "--ocr_engine", "paddleocr", "--lang", "zzzz")
        assert "Unsupported language code 'zzzz' for PaddleOCR." in out
        assert not fake_save

    def test_unsupported_paddle_label_lang(self, monkeypatch, video_file, capsys, fake_save):
        out = self._expect_exit(monkeypatch, video_file, capsys,
                                "--ocr_engine", "paddleocr", "--label_lang", "zzzz")
        assert "Unsupported label language code 'zzzz' for PaddleOCR." in out
        assert not fake_save

    def test_unsupported_lens_label_lang(self, monkeypatch, video_file, capsys, fake_save):
        out = self._expect_exit(monkeypatch, video_file, capsys,
                                "--ocr_engine", "google_lens", "--label_lang", "zzzz")
        assert "Unsupported label language code 'zzzz' for Google Lens." in out
        assert not fake_save

    def test_onnx_label_lang_prints_note(self, monkeypatch, video_file, capsys, fake_save):
        run_main(monkeypatch, "--video_path", str(video_file),
                 "--ocr_engine", "onnx_directml",
                 "--enable_label_detection", "true",
                 "--label_lang", "japan",
                 "--allow_system_sleep", "true")
        assert "--label_lang has no effect for onnx_directml" in capsys.readouterr().out
        assert fake_save[0]["label_lang"] == "japan"

    def test_unsupported_lens_lang(self, monkeypatch, video_file, capsys, fake_save):
        out = self._expect_exit(monkeypatch, video_file, capsys,
                                "--ocr_engine", "google_lens", "--lang", "zzzz")
        assert "for Google Lens" in out
        assert not fake_save

    def test_lens_lang_is_case_sensitive(self, monkeypatch, video_file, capsys, fake_save):
        out = self._expect_exit(monkeypatch, video_file, capsys,
                                "--ocr_engine", "google_lens", "--lang", "EN")
        assert "for Google Lens" in out

    def test_paddle_lang_case_insensitive(self, monkeypatch, video_file, fake_save):
        # Paddle validation lowercases; "EN" must pass.
        run_main(monkeypatch, "--video_path", str(video_file),
                 "--ocr_engine", "paddleocr", "--lang", "EN",
                 "--allow_system_sleep", "true")
        assert fake_save and fake_save[0]["lang"] == "EN"

    def test_start_after_end(self, monkeypatch, video_file, capsys, fake_save):
        out = self._expect_exit(monkeypatch, video_file, capsys,
                                "--time_start", "0:10", "--time_end", "0:05")
        assert "cannot be after" in out
        assert not fake_save

    def test_label_start_after_end(self, monkeypatch, video_file, capsys, fake_save):
        out = self._expect_exit(monkeypatch, video_file, capsys,
                                "--label_time_start", "0:10", "--label_time_end", "0:05")
        assert "Label Start Time" in out
        assert not fake_save

    def test_partial_zone1(self, monkeypatch, video_file, capsys, fake_save):
        out = self._expect_exit(monkeypatch, video_file, capsys,
                                "--crop_x", "10", "--crop_y", "20")
        assert "Partial crop coordinates detected for Zone 1" in out
        assert not fake_save

    def test_partial_zone2(self, monkeypatch, video_file, capsys, fake_save):
        out = self._expect_exit(
            monkeypatch, video_file, capsys,
            "--crop_x", "1", "--crop_y", "2", "--crop_width", "3", "--crop_height", "4",
            "--crop_x2", "500",
        )
        assert "Partial crop coordinates detected for Zone 2" in out
        assert not fake_save

    def test_fullframe_skips_partial_zone_check(self, monkeypatch, video_file, fake_save):
        run_main(monkeypatch, "--video_path", str(video_file),
                 "--use_fullframe", "true", "--crop_x", "10",
                 "--allow_system_sleep", "true")
        assert fake_save and fake_save[0]["crop_zones"] == []


class TestDirectMLSideEffects:
    def test_onnx_engine_sets_env_and_prints(self, monkeypatch, video_file, fake_save, capsys):
        run_main(monkeypatch, "--video_path", str(video_file),
                 "--ocr_engine", "onnx_directml", "--directml_device_index", "1",
                 "--directml_recognition_mode", "auto",
                 "--onnx_directml_tuning", "low_vram",
                 "--allow_system_sleep", "true")
        import os

        assert os.environ["VIDEOCR_DIRECTML_DEVICE_INDEX"] == "1"
        assert os.environ["VIDEOCR_DIRECTML_RECOGNITION_MODE"] == "auto"
        assert os.environ["VIDEOCR_ONNX_DIRECTML_TUNING"] == "low_vram"
        out = capsys.readouterr().out
        assert "DirectML performance preset: balanced" in out
        assert "ONNX DirectML tuning mode: low_vram" in out
        assert fake_save

    def test_paddle_engine_does_not_touch_env(self, monkeypatch, video_file, fake_save):
        run_main(monkeypatch, "--video_path", str(video_file),
                 "--ocr_engine", "paddleocr", "--allow_system_sleep", "true")
        import os

        assert "VIDEOCR_DIRECTML_DEVICE_INDEX" not in os.environ
        assert "VIDEOCR_ONNX_DIRECTML_TUNING" not in os.environ

    def test_benchmark_print(self, monkeypatch, video_file, capsys, fake_save):
        run_main(monkeypatch, "--video_path", str(video_file),
                 "--ocr_engine", "onnx_directml",
                 "--benchmark_compare_engine", "true",
                 "--benchmark_compare_sample_grids", "5",
                 "--allow_system_sleep", "true")
        assert "Benchmark compare enabled: sample grids=5" in capsys.readouterr().out


def test_cli_import_reconfigures_stdio_safely():
    # Import already happened; just assert the helper is idempotent.
    videocr_cli._force_utf8_stdio()
    videocr_cli._force_utf8_stdio()


def test_time_string_used_by_validators_matches_utils():
    assert videocr_cli.valid_time_string("1:30") == "1:30"
    assert utils.get_ms_from_time_str("1:30") == 90_000.0
