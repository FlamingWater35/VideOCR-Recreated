"""Tests for videocr_gui.workers — CLI discovery, command building, output handling.

The full ``CLIWorker.run()`` loop is exercised against a fake CLI (a tiny
``python -c`` script) so no OCR is ever performed.
"""

from __future__ import annotations

import sys

import pytest

from videocr_gui import workers
from videocr_gui.progress import ProgressUpdate


@pytest.fixture
def fake_cli(monkeypatch, tmp_path):
    """Install a fake videocr CLI that prints scripted stdout and exits 0."""
    script = tmp_path / "fake_cli.py"

    def install(body: str, exit_code: int = 0):
        script.write_text(
            "import sys\n"
            "argv = sys.argv[1:]\n"
            f"{body}\n"
            f"sys.exit({exit_code})\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(workers, "VIDEOCR_PATH", [sys.executable, str(script)])
        return script

    return install


class TestFindVideocrProgram:
    def test_prefers_app_dir_executable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(workers, "APP_DIR", str(tmp_path))
        monkeypatch.setattr(workers.sys, "platform", "win32")
        exe = tmp_path / "videocr-cli.exe"
        exe.write_bytes(b"")
        assert workers.find_videocr_program() == str(exe)

    def test_venv_executable_fallback(self, monkeypatch, tmp_path):
        monkeypatch.setattr(workers, "APP_DIR", str(tmp_path))
        monkeypatch.setattr(workers.sys, "platform", "win32")
        venv_dir = tmp_path / ".venv" / "Scripts"
        venv_dir.mkdir(parents=True)
        exe = venv_dir / "videocr-cli.exe"
        exe.write_bytes(b"")
        assert workers.find_videocr_program() == str(exe)

    def test_path_lookup(self, monkeypatch, tmp_path):
        monkeypatch.setattr(workers, "APP_DIR", str(tmp_path / "empty"))
        monkeypatch.setattr(workers.sys, "platform", "linux")
        monkeypatch.setattr(workers.shutil, "which", lambda name: "/usr/local/bin/videocr-cli")
        assert workers.find_videocr_program() == "/usr/local/bin/videocr-cli"

    def test_source_script_fallback(self, monkeypatch, tmp_path):
        # Mimic the repo layout: APP_DIR/CLI/videocr_cli.py + APP_DIR/CLI/videocr/.
        app = tmp_path / "app"
        (app / "CLI" / "videocr").mkdir(parents=True)
        script = app / "CLI" / "videocr_cli.py"
        script.write_text("# fake", encoding="utf-8")
        monkeypatch.setattr(workers, "APP_DIR", str(app))
        monkeypatch.setattr(workers.sys, "platform", "linux")
        monkeypatch.setattr(workers.shutil, "which", lambda name: None)
        assert workers.find_videocr_program() == [sys.executable, str(script)]

    def test_returns_none_when_nothing_found(self, monkeypatch, tmp_path):
        monkeypatch.setattr(workers, "APP_DIR", str(tmp_path / "empty"))
        monkeypatch.setattr(workers.sys, "platform", "linux")
        monkeypatch.setattr(workers.shutil, "which", lambda name: None)
        assert workers.find_videocr_program() is None

    def test_repo_source_layout_resolves(self, monkeypatch, repo_root):
        # The real repository must always resolve to the source CLI script
        # (unless a compiled exe happens to be present).
        monkeypatch.setattr(workers, "APP_DIR", str(repo_root))
        monkeypatch.setattr(workers.shutil, "which", lambda name: None)
        result = workers.find_videocr_program()
        if result is None:
            pytest.skip("compiled videocr-cli present or layout changed")
        if isinstance(result, list):
            assert result[1] == str(repo_root / "CLI" / "videocr_cli.py")


class TestBuildCommand:
    def test_empty_args_without_program(self, monkeypatch):
        monkeypatch.setattr(workers, "VIDEOCR_PATH", None)
        worker = workers.CLIWorker({})
        assert worker.build_command() == []

    def test_basic_command(self, monkeypatch):
        monkeypatch.setattr(workers, "VIDEOCR_PATH", "/bin/videocr-cli")
        worker = workers.CLIWorker({
            "video_path": "/videos/a.mp4",
            "ocr_engine": "paddleocr",
            "conf_threshold": "75",
        })
        assert worker.build_command() == [
            "/bin/videocr-cli",
            "--video_path", "/videos/a.mp4",
            "--ocr_engine", "paddleocr",
            "--conf_threshold", "75",
        ]

    def test_bools_lowercased(self, monkeypatch):
        monkeypatch.setattr(workers, "VIDEOCR_PATH", "cli")
        worker = workers.CLIWorker({"use_gpu": True, "use_angle_cls": False})
        cmd = worker.build_command()
        assert cmd[cmd.index("--use_gpu") + 1] == "true"
        assert cmd[cmd.index("--use_angle_cls") + 1] == "false"

    def test_none_and_empty_skipped(self, monkeypatch):
        monkeypatch.setattr(workers, "VIDEOCR_PATH", "cli")
        worker = workers.CLIWorker({"a": None, "b": "", "c": "x"})
        cmd = worker.build_command()
        assert "--a" not in cmd
        assert "--b" not in cmd
        assert "--c" in cmd

    def test_send_notification_not_forwarded(self, monkeypatch):
        monkeypatch.setattr(workers, "VIDEOCR_PATH", "cli")
        worker = workers.CLIWorker({"send_notification": True, "video_path": "v.mp4"})
        assert "--send_notification" not in worker.build_command()

    def test_paths_with_spaces_stay_single_argv_elements(self, monkeypatch):
        monkeypatch.setattr(workers, "VIDEOCR_PATH", "cli")
        spaced = "C:/My Videos/movie one.mp4"
        worker = workers.CLIWorker({"video_path": spaced, "output": "C:/My Subs/out one.srt"})
        cmd = worker.build_command()
        assert spaced in cmd
        assert "C:/My Subs/out one.srt" in cmd

    def test_list_program_preserved(self, monkeypatch):
        monkeypatch.setattr(workers, "VIDEOCR_PATH", [sys.executable, "script.py"])
        worker = workers.CLIWorker({"k": "v"})
        cmd = worker.build_command()
        assert cmd[:2] == [sys.executable, "script.py"]
        assert cmd[2:] == ["--k", "v"]


class TestHandleLine:
    @pytest.fixture
    def worker(self):
        return workers.CLIWorker({})

    @pytest.fixture
    def capture(self, worker):
        """Collect emissions from every WorkerSignals signal."""
        seen: dict[str, list] = {
            "output": [], "progress": [], "repacking": [],
            "fatal": [], "warning": [],
        }
        worker.signals.output.connect(lambda s: seen["output"].append(s))
        worker.signals.progress.connect(lambda s: seen["progress"].append(s))
        worker.signals.repacking.connect(lambda s: seen["repacking"].append(s))
        worker.signals.fatal.connect(lambda s: seen["fatal"].append(s))
        worker.signals.warning.connect(lambda s: seen["warning"].append(s))
        return seen

    def test_progress_line(self, worker, capture):
        worker._handle_line("Step 2/3: Performing Text-Detection on image 1 of 4")
        assert capture["progress"]
        assert isinstance(capture["progress"][0], ProgressUpdate)
        assert capture["progress"][0].step == 2

    def test_repacking_line(self, worker, capture):
        worker._handle_line("Analyzing frame 3 of 9")
        assert capture["repacking"] == ["Analyzing frame 3 of 9"]

    def test_fatal_line(self, worker, capture):
        worker._handle_line("Unsupported Hardware Error: device lost")
        assert capture["fatal"] == ["device lost"]

    def test_warning_line(self, worker, capture):
        worker._handle_line("Hardware Check Warning: adapter 9 missing")
        assert capture["warning"] == ["adapter 9 missing"]

    def test_plain_log_line(self, worker, capture):
        worker._handle_line("hello world")
        assert capture["output"] == ["hello world\n"]

    def test_starting_line_translated(self, worker, capture):
        worker._handle_line("Starting PaddleOCR...")
        assert capture["output"]
        assert capture["output"][0].endswith("\n")

    def test_generating_line(self, worker, capture):
        worker._handle_line("Generating subtitles...")
        assert capture["output"]


class TestRunLoop:
    """Drive CLIWorker.run() against a fake CLI process (no OCR, no network)."""

    def _run(self, worker):
        """Call run() synchronously and collect all signal emissions."""
        seen: dict[str, list] = {
            "output": [], "progress": [], "repacking": [],
            "finished": [], "started": [],
        }
        worker.signals.output.connect(lambda s: seen["output"].append(s))
        worker.signals.progress.connect(lambda s: seen["progress"].append(s))
        worker.signals.repacking.connect(lambda s: seen["repacking"].append(s))
        worker.signals.process_finished.connect(
            lambda ok, code: seen["finished"].append((ok, code))
        )
        worker.signals.process_started.connect(lambda pid: seen["started"].append(pid))
        worker.run()
        return seen

    def test_successful_run(self, fake_cli):
        fake_cli(
            "print('Step 2/3: Performing Text-Detection on image 1 of 2')\n"
            "print('Analyzing frame 1 of 2')\n"
            "print('plain log line')\n"
            "print('Generating subtitles...')"
        )
        worker = workers.CLIWorker({})
        seen = self._run(worker)
        assert seen["finished"] == [(True, 0)]
        assert len(seen["started"]) == 1
        assert seen["progress"]  # step-2 progress parsed
        assert seen["repacking"] == ["Analyzing frame 1 of 2"]
        assert any("plain log line" in line for line in seen["output"])
        assert any("Generating subtitles" in line for line in seen["output"])

    def test_failure_with_clean_error_not_logged_as_crash(self, fake_cli, _error_log):
        fake_cli("print('Error: Process failed.')", exit_code=1)
        worker = workers.CLIWorker({})
        seen = self._run(worker)
        assert seen["finished"] == [(False, 1)]
        assert not any("crashed with exit code" in m for m in _error_log)
        assert any("Error: Process failed." in line for line in seen["output"])

    def test_unexpected_crash_is_logged(self, fake_cli, _error_log):
        fake_cli("import sys; print('boom', file=sys.stderr)", exit_code=3)
        worker = workers.CLIWorker({})
        seen = self._run(worker)
        assert seen["finished"] == [(False, 3)]
        assert any("crashed with exit code 3" in m for m in _error_log)
        assert any("UNEXPECTED ERROR" in line for line in seen["output"])

    def test_cancelled_run_not_logged_as_crash(self, fake_cli, _error_log):
        fake_cli("print('partial output')", exit_code=1)
        worker = workers.CLIWorker({})
        worker._cancelled_by_user = True  # as cancel() would set
        seen = self._run(worker)
        assert seen["finished"] == [(False, 1)]
        assert not any("crashed with exit code" in m for m in _error_log)

    def test_missing_program_reports_failure(self, monkeypatch, _error_log):
        monkeypatch.setattr(workers, "VIDEOCR_PATH", None)
        worker = workers.CLIWorker({})
        seen = self._run(worker)
        assert seen["finished"] == [(False, -1)]
        assert any("videocr-cli not found" in line for line in seen["output"])

    def test_spawn_error_reports_failure(self, monkeypatch):
        monkeypatch.setattr(workers, "VIDEOCR_PATH", "/definitely/not/a/real/binary")
        worker = workers.CLIWorker({})
        seen = self._run(worker)
        assert seen["finished"] == [(False, -1)]
        assert any("An error occurred" in line for line in seen["output"])

    def test_command_arguments_forwarded_to_cli(self, fake_cli, tmp_path):
        # Fake CLI dumps argv; we assert the worker forwards args verbatim.
        fake_cli("print('argv:' + '|'.join(argv))")
        worker = workers.CLIWorker({"video_path": "/v/a b.mp4", "ocr_engine": "paddleocr"})
        seen = self._run(worker)
        joined = "".join(seen["output"])
        assert "argv:--video_path|/v/a b.mp4|--ocr_engine|paddleocr" in joined

    def test_pause_without_pid_returns_false(self):
        worker = workers.CLIWorker({})
        assert worker.pause() is False
        assert worker.resume() is False
