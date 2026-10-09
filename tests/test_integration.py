"""Integration tests: real video decoding, real CLI subprocess, real ONNX OCR.

These are the "app works fully" tests. They are skipped automatically when the
local test video (gitignored) or the ONNX/DirectML runtime is unavailable.

The heavy OCR run is marked ``integration`` (registered in pyproject.toml);
run ``pytest -m "not integration"`` to deselect it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

SRT_TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$")


def _onnx_dml_available() -> bool:
    try:
        import onnxruntime as ort
    except Exception:
        return False
    return "DmlExecutionProvider" in ort.get_available_providers()


def _rapidocr_available() -> bool:
    try:
        import rapidocr  # noqa: F401
    except Exception:
        return False
    return True


needs_ocr = pytest.mark.skipif(
    not (_onnx_dml_available() and _rapidocr_available()),
    reason="ONNX Runtime DirectML + rapidocr not available",
)


def parse_srt(text: str) -> list[dict]:
    """Parses SRT content into [{'index', 'start', 'end', 'text'}, ...]."""
    blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]
    entries = []
    for block in blocks:
        lines = block.splitlines()
        assert len(lines) >= 3, f"malformed SRT block: {block!r}"
        assert lines[0].isdigit(), f"bad index: {lines[0]!r}"
        assert SRT_TIME_RE.match(lines[1]), f"bad timing: {lines[1]!r}"
        start, end = lines[1].split(" --> ")
        entries.append({
            "index": int(lines[0]),
            "start": start,
            "end": end,
            "text": "\n".join(lines[2:]).strip(),
        })
    return entries


class TestVideoDecoding:
    """The test video is decodable and reports sane properties."""

    def test_pyav_properties(self, test_video):
        from videocr.pyav_adapter import get_video_properties

        props = get_video_properties(str(test_video))
        assert props["width"] == 3840
        assert props["height"] == 1608
        assert props["fps"] == pytest.approx(25.0)
        assert 17_000 <= props["duration_ms"] <= 18_000

    def test_capture_reads_frames(self, test_video):
        from videocr.pyav_adapter import Capture

        with Capture(str(test_video)) as cap:
            ok, frame, ts = cap.read()
            assert ok
            assert frame is not None
            assert ts >= 0
            assert frame.width == 3840

    def test_gui_video_handler_opens(self, qtbot, test_video):
        from videocr_gui.video_preview import VideoHandler

        handler = VideoHandler()
        info = handler.open(str(test_video))
        assert info["width"] == 3840
        assert info["height"] == 1608
        handler.close()

    def test_gui_video_handler_get_frame(self, qtbot, test_video):
        from PySide6.QtGui import QImage

        from videocr_gui.video_preview import VideoHandler

        handler = VideoHandler()
        try:
            handler.open(str(test_video))
            image, w, h, off_x, off_y = handler.get_frame(0.0, (720, 405))
            assert isinstance(image, QImage)
            assert not image.isNull()
            assert (image.width(), image.height()) == (w, h)
            assert off_x >= 0 and off_y >= 0
        finally:
            handler.close()

    def test_gui_preview_displays_first_frame(self, qtbot, test_video):
        from videocr_gui.video_preview import VideoPreview

        preview = VideoPreview()
        try:
            assert preview.load_video(str(test_video))
            # First frame decodes on the background worker and must land on
            # the scene via the async display path.
            qtbot.waitUntil(
                lambda: preview._current_pixmap is not None, timeout=5000
            )
            assert preview._pixmap_item is not None
            assert preview.duration_ms > 0
        finally:
            preview.shutdown()


class TestCliHelp:
    def test_cli_help_via_source_script(self, test_video):
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "CLI" / "videocr_cli.py"), "--help"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=120, cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 0
        assert "--video_path" in proc.stdout
        assert "--ocr_engine" in proc.stdout

    def test_gui_cli_discovery_resolves(self):
        from videocr_gui import workers

        result = workers.find_videocr_program()
        assert result is not None
        if isinstance(result, list):
            assert Path(result[1]).name == "videocr_cli.py"
        else:
            assert "videocr-cli" in Path(result).name


@pytest.mark.integration
@needs_ocr
class TestFullOcrRun:
    """Full GUI-style run: args dict → CLIWorker → real CLI → ONNX OCR → SRT."""

    def test_end_to_end(self, test_video, tmp_path, monkeypatch):

        from videocr_gui import args as argsmod
        from videocr_gui import progress, workers
        from videocr_gui.config import get_default_settings

        out_srt = tmp_path / "e2e.srt"

        # Build the args dict exactly like the GUI does.
        settings = get_default_settings()
        settings.update({
            "ocr_engine": "ONNX Runtime DirectML (AMD GPU Experimental)",
            "subtitle_language": "English",
            "--use_gpu": True,
            "--use_fullframe": True,
            "--time_start": "0:00",
            "--time_end": "0:03",
            "--frames_to_skip": "5",
        })
        argd, errors = argsmod.build_args(
            str(test_video), settings, [], output_path=str(out_srt)
        )
        assert errors == []
        assert argd is not None
        assert argd["ocr_engine"] == "onnx_directml"
        assert argd["use_gpu"] is True
        assert argd["output"] == str(out_srt)

        # Run through the real worker with the source CLI script.
        monkeypatch.setattr(
            workers, "VIDEOCR_PATH",
            [sys.executable, str(REPO_ROOT / "CLI" / "videocr_cli.py")],
        )
        progress._STATE = progress.ProgressState()  # fresh rate-limit state

        worker = workers.CLIWorker(argd)
        seen: dict[str, list] = {
            "output": [], "progress": [], "fatal": [], "warning": [],
            "repacking": [], "finished": [],
        }
        worker.signals.output.connect(lambda s: seen["output"].append(s))
        worker.signals.progress.connect(lambda s: seen["progress"].append(s))
        worker.signals.fatal.connect(lambda s: seen["fatal"].append(s))
        worker.signals.warning.connect(lambda s: seen["warning"].append(s))
        worker.signals.repacking.connect(lambda s: seen["repacking"].append(s))
        worker.signals.process_finished.connect(
            lambda ok, code: seen["finished"].append((ok, code))
        )

        # worker.run() blocks; it emits synchronously in this thread.
        worker.run()

        # 1. Process succeeded.
        assert seen["finished"] == [(True, 0)], seen["output"][-10:]
        assert not seen["fatal"], seen["fatal"]

        # 2. The GUI saw real progress updates and status lines.
        assert seen["progress"], "no progress signals parsed from CLI output"
        assert any(p.percent is not None for p in seen["progress"])
        joined = "".join(seen["output"])
        assert "Generating subtitles" in joined
        assert "[Perf] End-to-end runtime" in joined
        # CLI emitted progress lines the GUI regexes can classify:
        raw_lines = [line for line in joined.splitlines() if "Step " in line]
        assert raw_lines or seen["progress"]

        # 3. A valid, non-empty SRT was written.
        assert out_srt.exists()
        text = out_srt.read_text(encoding="utf-8")
        entries = parse_srt(text)
        assert len(entries) >= 1
        assert all(e["text"] for e in entries), entries
        # Indices are sequential from 1.
        assert [e["index"] for e in entries] == list(range(1, len(entries) + 1))

    def test_stdout_lines_classify_cleanly(self, test_video, tmp_path, monkeypatch):
        """Raw CLI stdout must be fully classifiable by the GUI parser."""
        proc = subprocess.run(
            [
                sys.executable, str(REPO_ROOT / "CLI" / "videocr_cli.py"),
                "--video_path", str(test_video),
                "--output", str(tmp_path / "raw.srt"),
                "--ocr_engine", "onnx_directml",
                "--use_gpu", "true",
                "--use_fullframe", "true",
                "--time_start", "0:00", "--time_end", "0:02",
                "--frames_to_skip", "5",
                "--allow_system_sleep", "true",
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=600, cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]

        from videocr_gui.progress import classify_line

        kinds: dict[str, int] = {}
        for line in proc.stdout.splitlines():
            if not line.strip():
                continue
            kind, _ = classify_line(line)
            kinds[kind] = kinds.get(kind, 0) + 1

        # Real progress + status lines were seen and nothing was misclassified
        # into 'fatal'.
        assert kinds.get("fatal") is None, "CLI reported a fatal hardware error"
        assert kinds.get("progress", 0) >= 1, kinds
        assert kinds.get("generating", 0) >= 1, kinds

        # Output file is a parseable, non-empty SRT.
        entries = parse_srt((tmp_path / "raw.srt").read_text(encoding="utf-8"))
        assert len(entries) >= 1
