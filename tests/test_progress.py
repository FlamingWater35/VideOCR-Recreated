"""Tests for videocr_gui.progress — CLI stdout classification and progress/ETA."""

from __future__ import annotations

import pytest

from videocr_gui import i18n, progress
from videocr_gui.progress import (
    ProgressState,
    ProgressUpdate,
    classify_line,
    format_seconds,
    handle_progress,
    parse_srt_time_to_seconds,
)


@pytest.fixture(autouse=True)
def _english():
    i18n.load_language("en")


class TestParseSrtTime:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("00:00:01,500", 1.5),
            ("00:01:00", 60.0),
            ("01:02:03", 3723.0),
            ("00:00:00", 0.0),
            ("1:30", 90.0),
            ("0:00:10,250", 10.25),
        ],
    )
    def test_valid(self, value, expected):
        assert parse_srt_time_to_seconds(value) == pytest.approx(expected)

    @pytest.mark.parametrize("value", ["", "bogus", "1", "a:b:c", "::", "00:xx:00"])
    def test_invalid_returns_zero(self, value):
        assert parse_srt_time_to_seconds(value) == 0.0


class TestFormatSeconds:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (None, "--:--"),
            (-1, "--:--"),
            (0, "00m 00s"),
            (59, "00m 59s"),
            (65, "01m 05s"),
            (3599, "59m 59s"),
            (3600, "1h 00m"),
            (3725, "1h 02m"),
            (7325, "2h 02m"),
        ],
    )
    def test_formats(self, seconds, expected):
        assert format_seconds(seconds) == expected


class TestClassifyLine:
    def test_step1_progress(self):
        line = "Step 1/3: Processing video... Current: 00:00:01 / 00:00:10, Frame: 5"
        kind, payload = classify_line(line)
        assert kind == "progress"
        assert payload == {
            "key": "progress_step1", "step": 1,
            "curr": "00:00:01", "total": "00:00:10", "extra": "5",
        }

    def test_step1_unknown_total(self):
        line = "Step 1/3: Processing video... Current: 00:00:01 / Unknown, Frame: 0"
        kind, payload = classify_line(line)
        assert kind == "progress"
        assert payload["total"] == "Unknown"

    @pytest.mark.parametrize(
        ("step", "key"),
        [(2, "progress_step2"), (3, "progress_step3")],
    )
    def test_step_image_progress(self, step, key):
        line = f"Step {step}/3: Performing Text-Detection on image 3 of 10"
        kind, payload = classify_line(line)
        assert kind == "progress"
        assert payload == {"key": key, "step": step, "curr": "3", "total": "10", "extra": None}

    def test_step3_ocr_variant(self):
        kind, payload = classify_line("Step 3/3: Performing OCR on image 1 of 4")
        assert kind == "progress"
        assert payload["key"] == "progress_step3"

    def test_step3_onnx_variant(self):
        kind, payload = classify_line("Step 3/3: Performing ONNX DirectML OCR on image 2 of 4")
        assert kind == "progress"
        assert payload["key"] == "progress_step3"

    def test_repacking(self):
        kind, payload = classify_line("Analyzing frame 12 of 30")
        assert kind == "repacking"
        assert payload == {"line": "Analyzing frame 12 of 30"}

    def test_fatal_hardware_error(self):
        kind, payload = classify_line("Unsupported Hardware Error: DirectML device lost")
        assert kind == "fatal"
        assert payload == {"message": "DirectML device lost"}

    def test_warning(self):
        kind, payload = classify_line("Hardware Check Warning: adapter index 9 out of range")
        assert kind == "warning"
        assert payload == {"message": "adapter index 9 out of range"}

    def test_process_error(self):
        kind, _ = classify_line("Error: Process failed.")
        assert kind == "process_error"

    @pytest.mark.parametrize(
        ("line", "key"),
        [
            ("Starting PaddleOCR...", "cli_starting_paddleocr"),
            ("Starting Google Lens CLI...", "cli_starting_lens"),
        ],
    )
    def test_starting_with_key(self, line, key):
        kind, payload = classify_line(line)
        assert kind == "starting"
        assert payload["key"] == key

    @pytest.mark.parametrize(
        "line",
        ["Starting EasyOCR...", "Starting EasyOCR DirectML...", "Starting ONNX Runtime DirectML OCR"],
    )
    def test_starting_without_key(self, line):
        kind, payload = classify_line(line)
        assert kind == "starting"
        assert payload["key"] is None

    def test_info_pass(self):
        line = "Running Text-Detection-Only pass on 42 filtered frame(s) stitched into 7 image grid(s)..."
        kind, payload = classify_line(line)
        assert kind == "info_pass"
        assert payload == {"frames": "42", "grids": "7"}

    def test_filtered(self):
        kind, payload = classify_line(
            "Filtered out 17 redundant frame(s) via Text-Detection and tight-box SSIM analysis."
        )
        assert kind == "filtered"
        assert payload == {"frames": "17"}

    def test_generating(self):
        kind, _ = classify_line("Generating subtitles...")
        assert kind == "generating"

    def test_reached_end(self):
        kind, _ = classify_line("Reached end time. Stopping.")
        assert kind == "reached_end"

    @pytest.mark.parametrize(
        "line",
        ["", "some plain log line", "INFO: worker started", "WARNING: low disk space"],
    )
    def test_unknown_lines_are_logs(self, line):
        kind, payload = classify_line(line)
        assert kind == "log"
        assert payload == {"line": line}


class TestHandleProgress:
    def test_step1_time_based_percent(self):
        update = handle_progress("progress_step1", 1, "00:00:05", "00:00:10", "25")
        assert update.percent == pytest.approx(50.0)
        assert update.step == 1
        assert "50.0" in update.text
        assert "00:00:05" in update.text

    def test_step1_clamped_to_100(self):
        update = handle_progress("progress_step1", 1, "00:00:20", "00:00:10", "99")
        assert update.percent == 100.0

    def test_step1_unknown_total_gives_zero_percent(self):
        update = handle_progress("progress_step1", 1, "00:00:05", "Unknown", "1")
        assert update.percent == 0.0

    def test_step2_count_based_percent(self):
        update = handle_progress("progress_step2", 2, "3", "10")
        assert update.percent == pytest.approx(30.0)
        assert update.step == 2
        assert "3 of 10" in update.text

    def test_step3_percent(self):
        update = handle_progress("progress_step3", 3, "4", "8")
        assert update.percent == pytest.approx(50.0)
        assert "4 of 8" in update.text

    def test_non_numeric_current_treated_as_zero(self):
        update = handle_progress("progress_step2", 2, "garbage", "10")
        assert update.percent == 0.0

    def test_zero_total_gives_zero_percent(self):
        update = handle_progress("progress_step2", 2, "3", "0")
        assert update.percent == 0.0

    def test_rate_limit_returns_empty_update(self):
        # First call updates last_update_time; an immediate second call on the
        # same key must be swallowed (<0.2s and percent < 99.9).
        first = handle_progress("progress_step2", 2, "1", "100")
        second = handle_progress("progress_step2", 2, "2", "100")
        assert first.text != ""
        assert second == ProgressUpdate()

    def test_full_percent_bypasses_rate_limit(self):
        handle_progress("progress_step3", 3, "99", "100")
        update = handle_progress("progress_step3", 3, "100", "100")
        assert update.percent == 100.0

    def test_eta_computed_on_advance(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr(progress.time, "time", lambda: now[0])
        handle_progress("progress_step2", 2, "1", "100")  # start key at 0%
        now[0] += 10.0
        update = handle_progress("progress_step2", 2, "51", "100")
        assert update.eta.startswith("ETA Step 2/3")
        # 50% done in 10s → 10% per second → 49% remaining ≈ 4.9s → "00m 04s"
        assert "00m 0" in update.eta

    def test_step_key_change_resets_baseline(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr(progress.time, "time", lambda: now[0])
        handle_progress("progress_step2", 2, "1", "100")
        now[0] += 5.0
        update = handle_progress("progress_step3", 3, "1", "100")
        # New step starts fresh at its own percent; no bogus ETA carried over.
        assert update.percent == pytest.approx(1.0)

    def test_state_defaults(self):
        state = ProgressState()
        assert state.last_key is None
        assert state.start_time == 0.0
        assert state.last_eta == ""
