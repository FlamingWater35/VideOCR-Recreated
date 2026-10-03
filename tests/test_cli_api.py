"""Tests for videocr.api.save_subtitles_to_file with a faked Video class.

No decoding or OCR happens: ``videocr.api.Video`` is replaced by a recorder,
helper-executable discovery is stubbed, and the written subtitle file is
asserted against the fake ``get_subtitles`` return value.
"""

from __future__ import annotations

import runpy

import pytest

from videocr import api


class FakeVideo:
    """Records constructor + method calls; returns a fixed SRT."""

    instances: list[FakeVideo] = []
    next_subtitles = "1\n00:00:00,000 --> 00:00:01,000\nHello\n"
    duration_ms = 0

    def __init__(self, *args):
        self.ctor_args = args
        self.ocr_args: tuple = ()
        self.sub_args: tuple = ()
        FakeVideo.instances.append(self)

    def run_ocr(self, *args):
        self.ocr_args = args

    def get_subtitles(self, *args):
        self.sub_args = args
        return self.next_subtitles


@pytest.fixture(autouse=True)
def _reset_fake(monkeypatch):
    FakeVideo.instances = []
    FakeVideo.next_subtitles = "1\n00:00:00,000 --> 00:00:01,000\nHello\n"
    FakeVideo.duration_ms = 0
    monkeypatch.setattr(api, "Video", FakeVideo)
    # Default engine is google_lens, which probes helper executables; stub the
    # whole helper layer so no real binaries are required. Individual tests
    # override these to exercise the discovery/error paths.
    monkeypatch.setattr(api.utils, "find_executable", lambda name: f"/fake/{name}")
    monkeypatch.setattr(api.utils, "perform_hardware_check", lambda *a: None)
    monkeypatch.setattr(
        api.utils, "resolve_model_dirs", lambda lang, server: ("/det", "/rec", "/cls")
    )
    yield
    FakeVideo.instances = []
    FakeVideo.duration_ms = 0


@pytest.fixture
def out_path(tmp_path):
    return tmp_path / "subtitle.srt"


def _no_helpers(name):
    raise FileNotFoundError(f"no {name}")


class TestHappyPath:
    def test_writes_subtitles_file(self, out_path):
        api.save_subtitles_to_file("video.mp4", file_path=str(out_path))
        assert out_path.exists()
        assert out_path.read_text(encoding="utf-8") == FakeVideo.next_subtitles

    def test_perf_prints(self, out_path, capsys):
        api.save_subtitles_to_file("video.mp4", file_path=str(out_path))
        out = capsys.readouterr().out
        assert "[Perf] Step 3 subtitle merge/write:" in out
        assert "[Perf] End-to-end runtime:" in out
        assert "[Bench]" not in out  # duration_ms is 0 → no bench line

    def test_bench_print_when_duration_known(self, out_path, capsys):
        FakeVideo.duration_ms = 10_000
        api.save_subtitles_to_file("video.mp4", file_path=str(out_path), ocr_engine="onnx_directml")
        out = capsys.readouterr().out
        assert "[Bench] Video duration: 10.00s" in out
        assert "engine: onnx_directml" in out

    def test_default_crop_and_alignment_normalization(self, out_path):
        # crop_zones=None → []; subtitle_alignments=None → [None, None].
        api.save_subtitles_to_file("video.mp4", file_path=str(out_path))
        v = FakeVideo.instances[0]
        assert v.ocr_args[12] == []                 # crop_zones
        assert v.sub_args[5] == [None, None]        # subtitle_alignments

    def test_single_alignment_gets_none_partner(self, out_path):
        api.save_subtitles_to_file(
            "video.mp4", file_path=str(out_path), subtitle_alignments=["an7"]
        )
        assert FakeVideo.instances[0].sub_args[5] == ["an7", None]

    def test_get_subtitles_arguments_forwarded(self, out_path):
        api.save_subtitles_to_file(
            "video.mp4", file_path=str(out_path),
            sim_threshold=90, max_merge_gap_sec=0.5, lang="german",
            post_processing=True, min_subtitle_duration_sec=1.5,
            subtitle_alignments=["an1", "an2"],
        )
        v = FakeVideo.instances[0]
        assert v.sub_args == (90, 0.5, "german", True, 1.5, ["an1", "an2"])

    def test_run_ocr_arguments_forwarded(self, out_path):
        api.save_subtitles_to_file(
            "video.mp4", file_path=str(out_path),
            ocr_engine="onnx_directml", lang="japan",
            time_start="0:05", time_end="0:20", conf_threshold=40,
            use_fullframe=True, crop_zones=[{"x": 0, "y": 0, "width": 10, "height": 10}],
            directml_performance_preset="max",
            directml_recognition_mode="experimental",
            onnx_directml_tuning="low_vram",
            enable_label_detection=False,
            label_time_start="0:01", label_time_end="0:02",
        )
        args = FakeVideo.instances[0].ocr_args
        assert args[0] is False              # use_gpu
        assert args[1] == "onnx_directml"
        assert args[2] == "japan"
        assert args[4] == "0:05"
        assert args[5] == "0:20"
        assert args[6] == 40
        assert args[7] is True               # use_fullframe
        assert args[12] == [{"x": 0, "y": 0, "width": 10, "height": 10}]
        assert args[17] == "max"             # performance preset
        assert args[18] == "experimental"    # recognition mode
        assert args[20] == "low_vram"        # onnx tuning
        assert args[29] == "0:01"            # label_time_start
        assert args[30] == "0:02"            # label_time_end

    def test_video_paths_forwarded_to_constructor(self, out_path):
        api.save_subtitles_to_file("some/video.mp4", file_path=str(out_path))
        assert FakeVideo.instances[0].ctor_args[0] == "some/video.mp4"


class TestEngineHelpers:
    def test_paddleocr_requires_helper(self, out_path, capsys, monkeypatch):
        monkeypatch.setattr(api.utils, "find_executable", _no_helpers)
        with pytest.raises(SystemExit) as exc:
            api.save_subtitles_to_file(
                "v.mp4", file_path=str(out_path), ocr_engine="paddleocr"
            )
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "PaddleOCR helper executable not found" in out
        assert not FakeVideo.instances  # never reached Video()

    def test_google_lens_requires_both_helpers(self, out_path, capsys, monkeypatch):
        calls = []

        def fake_find(name):
            calls.append(name)
            if name == "paddleocr":
                return "/fake/paddleocr.exe"
            raise FileNotFoundError("no chrome-lens")

        monkeypatch.setattr(api.utils, "find_executable", fake_find)
        monkeypatch.setattr(api.utils, "perform_hardware_check", lambda *a: None)
        with pytest.raises(SystemExit) as exc:
            api.save_subtitles_to_file(
                "v.mp4", file_path=str(out_path), ocr_engine="google_lens"
            )
        assert exc.value.code == 1
        assert calls == ["paddleocr", "chrome-lens"]
        assert "Chrome Lens helper executable not found" in capsys.readouterr().out

    def test_onnx_engine_needs_no_helpers(self, out_path, monkeypatch):
        monkeypatch.setattr(api.utils, "find_executable", _no_helpers)
        api.save_subtitles_to_file("v.mp4", file_path=str(out_path), ocr_engine="onnx_directml")
        assert FakeVideo.instances  # reached Video()

    def test_paddle_helpers_and_models_resolved(self, out_path, monkeypatch):
        hw_checks = []

        monkeypatch.setattr(api.utils, "find_executable", lambda name: f"/fake/{name}")
        monkeypatch.setattr(
            api.utils, "perform_hardware_check",
            lambda path, use_gpu: hw_checks.append((path, use_gpu)),
        )
        monkeypatch.setattr(
            api.utils, "resolve_model_dirs",
            lambda lang, server: ("/det/" + lang, "/rec/" + lang, "/cls"),
        )
        api.save_subtitles_to_file(
            "v.mp4", file_path=str(out_path), ocr_engine="paddleocr",
            lang="german", use_gpu=True,
        )
        v = FakeVideo.instances[0]
        assert v.ctor_args == ("v.mp4", "/fake/paddleocr", "/det/german", "/rec/german", "/cls", "")
        assert hw_checks == [("/fake/paddleocr", True)]

    def test_google_lens_uses_english_detection_models(self, out_path, monkeypatch):
        monkeypatch.setattr(api.utils, "find_executable", lambda name: f"/fake/{name}")
        monkeypatch.setattr(api.utils, "perform_hardware_check", lambda *a: None)
        monkeypatch.setattr(
            api.utils, "resolve_model_dirs",
            lambda lang, server: (f"/det/{lang}", f"/rec/{lang}", "/cls"),
        )
        api.save_subtitles_to_file(
            "v.mp4", file_path=str(out_path), ocr_engine="google_lens", lang="japan"
        )
        v = FakeVideo.instances[0]
        # Detection model dirs always resolve with "en"; lens path passed.
        assert v.ctor_args[2] == "/det/en"
        assert v.ctor_args[5] == "/fake/chrome-lens"

    def test_hardware_check_failure_exits(self, out_path, capsys, monkeypatch):
        monkeypatch.setattr(api.utils, "find_executable", lambda name: f"/fake/{name}")

        def fail_check(path, use_gpu):
            raise SystemExit("Unsupported Hardware Error: no AVX")

        monkeypatch.setattr(api.utils, "perform_hardware_check", fail_check)
        with pytest.raises(SystemExit) as exc:
            api.save_subtitles_to_file("v.mp4", file_path=str(out_path), ocr_engine="paddleocr")
        assert exc.value.code == 1
        assert "Unsupported Hardware Error: no AVX" in capsys.readouterr().out
        assert not FakeVideo.instances


class TestErrorPaths:
    def test_run_ocr_exception_exits_with_error(self, out_path, capsys, monkeypatch):
        class ExplodingVideo(FakeVideo):
            def run_ocr(self, *args):
                raise RuntimeError("decoder exploded")

        monkeypatch.setattr(api, "Video", ExplodingVideo)
        with pytest.raises(SystemExit) as exc:
            api.save_subtitles_to_file("v.mp4", file_path=str(out_path))
        assert exc.value.code == 1
        assert "Error: decoder exploded" in capsys.readouterr().out
        assert not out_path.exists()


class TestLabelDetectionOutput:
    def test_srt_rewritten_to_ass(self, out_path, capsys, monkeypatch):
        ass_out = out_path.with_suffix(".ass")
        monkeypatch.setattr(runpy, "run_path", lambda *a, **k: None)
        api.save_subtitles_to_file(
            "v.mp4", file_path=str(out_path), enable_label_detection=True
        )
        assert not out_path.exists()
        assert ass_out.exists()
        out = capsys.readouterr().out
        assert "output will be written as .ass subtitle file" in out
        assert "ass-qafix" in out

    def test_existing_ass_output_untouched_suffix(self, tmp_path, capsys, monkeypatch):
        target = tmp_path / "already.ass"
        target.write_text("old content", encoding="utf-8")  # pre-existing output
        monkeypatch.setattr(runpy, "run_path", lambda *a, **k: None)
        api.save_subtitles_to_file(
            "v.mp4", file_path=str(target), enable_label_detection=True
        )
        assert target.exists()
        assert "Warning: overwriting existing .ass file" in capsys.readouterr().out

    def test_qafix_receives_ass_path(self, tmp_path, monkeypatch):
        target = tmp_path / "out.srt"
        seen: dict = {}

        def fake_run_path(script, run_name=None):
            seen["script"] = script
            seen["run_name"] = run_name
            seen["argv"] = list(__import__("sys").argv)

        monkeypatch.setattr(runpy, "run_path", fake_run_path)
        api.save_subtitles_to_file("v.mp4", file_path=str(target), enable_label_detection=True)
        assert seen["run_name"] == "__main__"
        assert seen["argv"][:2] == ["ass_qafix", "--inplace"]
        assert seen["argv"][2] == str(tmp_path / "out.ass")
        assert seen["script"].endswith("ass_qafix.py")

    def test_qafix_nonzero_exit_warns(self, tmp_path, capsys, monkeypatch):
        target = tmp_path / "out.srt"

        def fake_run_path(script, run_name=None):
            raise SystemExit(3)

        monkeypatch.setattr(runpy, "run_path", fake_run_path)
        api.save_subtitles_to_file("v.mp4", file_path=str(target), enable_label_detection=True)
        assert "Warning: ass-qafix exited with code 3" in capsys.readouterr().out
        assert target.with_suffix(".ass").exists()  # output still written

    def test_qafix_missing_script_skips(self, tmp_path, capsys, monkeypatch):
        target = tmp_path / "out.srt"

        def no_script(*a, **k):  # pragma: no cover - guard
            raise AssertionError("run_path must not be called")

        monkeypatch.setattr(runpy, "run_path", no_script)
        # Point __file__ resolution at a temp tree without tools/ by patching
        # the script-existence check via os.path.isfile on ass_qafix only.
        real_isfile = __import__("os").path.isfile

        def fake_isfile(p):
            if str(p).endswith("ass_qafix.py"):
                return False
            return real_isfile(p)

        monkeypatch.setattr("os.path.isfile", fake_isfile)
        api.save_subtitles_to_file("v.mp4", file_path=str(target), enable_label_detection=True)
        assert "ass-qafix script not found; skipping" in capsys.readouterr().out
