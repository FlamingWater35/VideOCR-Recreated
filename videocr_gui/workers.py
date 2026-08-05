"""CLI worker: runs videocr-cli as a subprocess on a QThread and parses output.

Ported from the legacy GUI's ``run_videocr`` / ``run_batch_thread`` logic:
- finds ``videocr-cli`` (compiled exe/bin, PATH, or ``python CLI/videocr_cli.py``)
- spawns with CREATE_NO_WINDOW on Windows
- reads stdout line-by-line, classifies progress/status lines
- supports pause/resume (psutil suspend) and cancel (taskkill tree / SIGTERM group)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from typing import IO, Any

from PySide6.QtCore import QObject, QThread, Signal

from . import i18n
from .config import APP_DIR, log_error
from .progress import classify_line, handle_progress

try:  # pragma: no cover - import guard for non-Windows
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def find_videocr_program() -> str | list[str] | None:
    """Determines how to start the videocr-cli worker (same logic as legacy GUI)."""
    program_name = "videocr-cli"
    extension = ".exe" if sys.platform == "win32" else ".bin"

    candidates = [os.path.join(APP_DIR, f"{program_name}{extension}")]
    if sys.platform == "win32":
        candidates.extend([
            os.path.join(APP_DIR, ".venv", "Scripts", f"{program_name}.exe"),
            os.path.join(APP_DIR, "venv", "Scripts", f"{program_name}.exe"),
        ])

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    path_candidate = shutil.which(f"{program_name}{extension}") or shutil.which(program_name)
    if path_candidate:
        return path_candidate

    cli_script = os.path.join(APP_DIR, "CLI", "videocr_cli.py")
    cli_package = os.path.join(APP_DIR, "CLI", "videocr")
    if os.path.isfile(cli_script) and os.path.isdir(cli_package):
        return [sys.executable, cli_script]

    return None


VIDEOCR_PATH = find_videocr_program()


def kill_process_tree(pid: int) -> None:
    """Kills the process with the given PID and its descendants."""
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                check=True, capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except Exception as e:
            log_error(f"Error terminating process tree {pid}: {e}")
    else:
        try:
            os.killpg(os.getpgid(pid), 15)
        except Exception as e:
            log_error(f"Error terminating process group {pid}: {e}")


def set_process_pause_state(pid: int, pause: bool = True) -> bool:
    """Pauses/resumes the process and its child tree via psutil."""
    if psutil is None:
        return False
    try:
        parent = psutil.Process(pid)
        if pause:
            parent.suspend()
            for child in parent.children(recursive=True):
                try:
                    child.suspend()
                except Exception:
                    pass
        else:
            for child in parent.children(recursive=True):
                try:
                    child.resume()
                except Exception:
                    pass
            parent.resume()
        return True
    except Exception as e:
        log_error(f"Failed to change pause state: {e}")
        return False


class WorkerSignals(QObject):
    """Signals emitted by the CLI worker thread."""

    output = Signal(str)            # log line to append (already processed)
    progress = Signal(object)       # ProgressUpdate
    repacking = Signal(str)         # "Analyzing frame X of Y" line
    process_started = Signal(int)   # pid
    process_finished = Signal(bool, int)  # success, exit_code
    fatal = Signal(str)
    warning = Signal(str)


class CLIWorker(QThread):
    """Runs one videocr-cli invocation, parsing stdout into signals."""

    def __init__(self, args: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.args = args
        self.signals = WorkerSignals()
        self._process: subprocess.Popen[bytes] | None = None
        self._pid: int | None = None
        self._cancelled_by_user = False

    def build_command(self) -> list[str]:
        if not VIDEOCR_PATH:
            return []
        command = VIDEOCR_PATH[:] if isinstance(VIDEOCR_PATH, list) else [VIDEOCR_PATH]
        for key, value in self.args.items():
            if value is None or value == "":
                continue
            if key == "send_notification":
                continue
            arg_name = f"--{key}"
            command.append(arg_name)
            if isinstance(value, bool):
                command.append(str(value).lower())
            else:
                command.append(str(value))
        return command

    def run(self) -> None:  # noqa: C901 - ported parsing loop
        if not VIDEOCR_PATH:
            self.signals.output.emit("\n" + i18n.tr("error_cli_not_found", "Error: videocr-cli not found. Please check the path.\n") + "\n")
            self.signals.process_finished.emit(False, -1)
            return

        command = self.build_command()
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
                start_new_session=(sys.platform != "win32"),
            )
        except Exception as e:
            self.signals.output.emit(f"\nAn error occurred: {e}\n")
            self.signals.process_finished.emit(False, -1)
            return

        self._pid = self._process.pid
        self.signals.process_started.emit(self._pid)

        stderr_thread = threading.Thread(target=self._read_pipe, args=(self._process.stderr, stderr_lines), daemon=True)
        stderr_thread.start()

        try:
            assert self._process.stdout is not None
            for line in iter(self._process.stdout.readline, ""):
                stdout_lines.append(line)
                if self._process.poll() is not None and line == "":
                    break
                line = line.rstrip("\r\n")
                self._handle_line(line)
        finally:
            stderr_thread.join(timeout=2)

        exit_code = self._process.wait()
        self._process = None

        if exit_code != 0 and not self._cancelled_by_user:
            full_stdout = "".join(stdout_lines)
            full_stderr = "".join(stderr_lines)
            if "Error: Process failed" not in full_stdout and "Unsupported Hardware Error:" not in full_stdout:
                log_message = (
                    f"The videocr-cli process crashed with exit code {exit_code}.\n\n"
                    f"--- COMMAND ---\n{' '.join(command)}\n\n"
                    f"--- STDOUT ---\n{full_stdout}\n\n"
                    f"--- STDERR ---\n{full_stderr}\n"
                )
                log_file_path = log_error(log_message, log_name="videocr-cli_crash.log")
                self.signals.output.emit(
                    f"\n--- UNEXPECTED ERROR ---\n"
                    f"{i18n.tr('unexpected_error_1', 'The subtitle extraction process failed unexpectedly.')}\n"
                    f"{i18n.tr('unexpected_error_2', 'A detailed crash report has been saved to:')}\n{log_file_path}\n"
                )

        self.signals.process_finished.emit(exit_code == 0, exit_code)

    def _handle_line(self, line: str) -> None:
        kind, payload = classify_line(line)

        if kind == "progress":
            upd = handle_progress(
                payload["key"], payload["step"], payload["curr"], payload["total"], payload.get("extra")
            )
            if upd.text:
                self.signals.progress.emit(upd)
        elif kind == "repacking":
            self.signals.repacking.emit(payload["line"])
        elif kind == "fatal":
            self.signals.fatal.emit(payload["message"])
        elif kind == "warning":
            self.signals.warning.emit(payload["message"])
        elif kind == "starting":
            key = payload["key"]
            if key:
                text = i18n.tr(key, payload["line"])
            else:
                text = payload["line"]
            self.signals.output.emit(text + "\n")
        elif kind == "info_pass":
            raw = i18n.tr("cli_info_pass", "Running Text-Detection-Only pass on {} filtered frame(s) stitched into {} image grid(s)...")
            self.signals.output.emit(raw.format(payload["frames"], payload["grids"]) + "\n")
        elif kind == "filtered":
            raw = i18n.tr("cli_filtered", "Filtered out {} redundant frame(s) via Text-Detection and tight-box SSIM analysis.")
            self.signals.output.emit(raw.format(payload["frames"]) + "\n")
        elif kind == "generating":
            self.signals.output.emit(i18n.tr("cli_generating_subs", payload["line"]) + "\n")
        elif kind == "reached_end":
            self.signals.output.emit(i18n.tr("log_reached_end", payload["line"]) + "\n")
        elif kind == "process_error":
            self.signals.output.emit(payload["line"] + "\n")
        else:
            self.signals.output.emit(line + "\n")

    @staticmethod
    def _read_pipe(pipe: IO[str] | None, output_list: list[str]) -> None:
        if pipe is None:
            return
        try:
            for line in iter(pipe.readline, ""):
                output_list.append(line)
        finally:
            pipe.close()

    def cancel(self) -> None:
        """Kills the process tree (user cancel)."""
        self._cancelled_by_user = True
        if self._pid:
            try:
                if self._process is not None and self._process.poll() is None:
                    kill_process_tree(self._pid)
            except Exception as e:
                log_error(f"Exception during cancel: {e}")

    def pause(self) -> bool:
        if self._pid:
            return set_process_pause_state(self._pid, pause=True)
        return False

    def resume(self) -> bool:
        if self._pid:
            return set_process_pause_state(self._pid, pause=False)
        return False
