"""Advanced-settings row for managing the FFmpeg binary.

The row answers three questions at a glance — is an FFmpeg available, which
build is it (bundled or system), and is there a newer one — and offers the two
actions that follow from that: *check* and *update*.

Updating never needs an application restart. The GUI does not run ffmpeg
itself; it spawns the CLI per job and the CLI resolves the binary at the start
of each run, so a file replaced on disk is picked up by the next job.

Threading rule: network and file work happens on a worker thread, but **no
worker ever touches a widget**. Workers emit signals and the slots below run on
the GUI thread, which is what makes the status updates safe.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import threading
from types import ModuleType
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import config, dialogs, i18n

#: Cached ``videocr.ffmpeg_tools`` module (see :func:`_tools`).
_TOOLS: ModuleType | None = None


def _tools() -> Any:
    """Imports the CLI's ``ffmpeg_tools`` module, from source or a build.

    ``ffmpeg_tools`` is a plain module with no package-relative imports, so it
    is bundled as a **data file** into the GUI build and loaded from its path
    here. A normal ``from videocr import ffmpeg_tools`` cannot work in a
    release: the GUI compile runs from the repository root with the CLI package
    off ``sys.path`` and deliberately does not follow it (that would pull torch
    and the OCR stack into the GUI). The package therefore does not exist in
    the shipped folder at all, and importing it raised
    ``ModuleNotFoundError: No module named 'videocr'`` in the settings row.

    The module is cached so that callers (and tests) can patch attributes on
    the returned object and see the change on later calls.
    """
    global _TOOLS
    if _TOOLS is not None:
        return _TOOLS

    # Prefer a real import: that is the from-source case, where the repository
    # root has ``CLI/`` on ``sys.path`` (see videocr_gui/main.py).
    ffmpeg_tools: ModuleType | None = None
    with contextlib.suppress(ImportError):
        from videocr import ffmpeg_tools  # type: ignore[import-not-found,no-redef]

    if ffmpeg_tools is None:
        bundled = os.path.join(config.APP_DIR, "CLI", "videocr", "ffmpeg_tools.py")
        if not os.path.isfile(bundled):
            raise ImportError(
                "ffmpeg_tools is unavailable: neither the videocr package nor "
                f"the bundled helper at {bundled} could be found."
            )
        spec = importlib.util.spec_from_file_location("videocr_ffmpeg_tools", bundled)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load the bundled helper at {bundled}")
        ffmpeg_tools = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = ffmpeg_tools
        spec.loader.exec_module(ffmpeg_tools)

    _TOOLS = ffmpeg_tools
    return ffmpeg_tools


class FFmpegRow(QWidget):
    """Status + check/update buttons for the managed FFmpeg."""

    #: Emitted after the row's status text changes, with the new text.
    status_changed = Signal(str)
    #: Emitted when an install finishes successfully, with the version.
    installed = Signal(str)
    #: Worker -> GUI thread: result of a version check ("" when unknown).
    _check_finished = Signal(str)
    #: Worker -> GUI thread: result of an install ("" on success).
    _install_finished = Signal(str)
    #: Worker -> GUI thread: download progress in percent, or -1 for unknown.
    _progress = Signal(int)
    #: Worker -> GUI thread: a completed background status read.
    _status_ready = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._busy = False
        self._latest: str | None = None
        self._version = ""
        self._status_seq = 0
        self._sentence_held = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self.status_label = QLabel()
        self.status_label.setObjectName("ffmpegStatus")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        top.addWidget(self.status_label, 1)

        self.check_btn = QPushButton(i18n.tr("btn_ffmpeg_check", "Check for Updates"))
        self.check_btn.setObjectName("ffmpegCheckButton")
        self.check_btn.clicked.connect(self.check_for_updates)
        top.addWidget(self.check_btn)

        self.update_btn = QPushButton(i18n.tr("btn_ffmpeg_update", "Update FFmpeg"))
        self.update_btn.setObjectName("ffmpegUpdateButton")
        self.update_btn.clicked.connect(self.install_latest)
        top.addWidget(self.update_btn)

        root.addLayout(top)

        self.progress = QProgressBar()
        self.progress.setObjectName("ffmpegProgress")
        self.progress.setTextVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        self._check_finished.connect(self._on_check_finished)
        self._install_finished.connect(self._on_install_finished)
        self._progress.connect(self._on_progress)
        self._status_ready.connect(self._on_status_ready)

        # Never probe synchronously from the constructor: the first
        # ``ffmpeg -version`` on a 100 MB binary can take seconds on a cold
        # cache, and this row is built during MainWindow() — it would freeze
        # the whole window before it is even on screen.
        self.status_label.setText(i18n.tr("ffmpeg_checking", "Checking for FFmpeg updates..."))
        self._request_status()

    # --- state -------------------------------------------------------------

    def refresh(self) -> None:
        """Re-reads the current binary and repaints the row (off the GUI thread)."""
        self._sentence_held = False
        self._request_status()

    def _request_status(self) -> None:
        """Starts a background read of the current binary."""
        self._status_seq += 1
        seq = self._status_seq

        def worker() -> None:
            try:
                status = _tools().managed_status(config.APP_DIR)
                payload = {str(k): str(v) for k, v in status.items()}
            except Exception as e:  # pragma: no cover - defensive
                payload = {"__error__": str(e)}
            payload["__seq__"] = str(seq)
            with contextlib.suppress(RuntimeError):
                self._status_ready.emit(payload)

        threading.Thread(target=worker, name="ffmpeg-status", daemon=True).start()

    def _on_status_ready(self, payload: dict[str, str]) -> None:
        """GUI thread: apply a background status read."""
        # A read started earlier may finish later; only the newest one may
        # write anything, so a slow cold probe cannot clobber a fresh result.
        if payload.get("__seq__") != str(self._status_seq):
            return
        error = payload.get("__error__")
        if error is not None:
            self._set_status(
                i18n.tr("ffmpeg_status_error", "FFmpeg status unavailable: {error}").replace(
                    "{error}", error
                )
            )
            self._set_buttons(check=False, update=False)
            return
        self._apply_status(payload)
        if not self._sentence_held:
            self._set_status(self._describe(payload))

    def _load_status(self) -> dict[str, str] | None:
        """Reads the current binary and updates buttons; ``None`` on failure.

        Kept separate from :meth:`refresh` so a caller that wants to show its
        own sentence (``"up to date"``, an error) can reload the state without
        having that sentence immediately overwritten.

        This is the *synchronous* read, used only where a caller already runs
        off the GUI thread (the check worker) or where a test wants a
        deterministic result. GUI code paths must use :meth:`refresh`.
        """
        try:
            status = _tools().managed_status(config.APP_DIR)
        except Exception as e:  # pragma: no cover - defensive
            self._set_status(
                i18n.tr("ffmpeg_status_error", "FFmpeg status unavailable: {error}").replace(
                    "{error}", str(e)
                )
            )
            self._set_buttons(check=False, update=False)
            return None
        self._apply_status(status)
        # ``managed_status`` returns a plain dict; typing it here keeps the
        # Any that leaks through the lazily imported module out of the caller.
        return {str(key): str(value) for key, value in status.items()}

    def _apply_status(self, status: dict[str, str]) -> None:
        """Applies a status dict to the buttons and cached version."""
        self._version = status.get("version", "")
        self._set_buttons(check=True, update=self._can_install(status))
        self.update_btn.setText(
            i18n.tr("btn_ffmpeg_update", "Update FFmpeg")
            if status.get("source") == "bundled"
            else i18n.tr("btn_ffmpeg_download", "Download FFmpeg")
        )

    def _describe(self, status: dict[str, str]) -> str:
        """Human summary: which build, which version, from where."""
        source = status.get("source", "")
        if source == "bundled":
            template = i18n.tr(
                "ffmpeg_status_bundled",
                "Bundled FFmpeg {version} (managed by the app).",
            )
        elif source == "system":
            template = i18n.tr(
                "ffmpeg_status_system",
                "System FFmpeg {version} (from PATH; no bundled copy).",
            )
        else:
            return i18n.tr(
                "ffmpeg_status_missing",
                "No FFmpeg found. The FFmpeg frame-scan mode will use PyAV instead.",
            )
        return template.replace("{version}", self._short(status.get("version", "")))

    def _short(self, version: str) -> str:
        """Trimmed version for display, tolerant of an unavailable helper."""
        try:
            return _tools().short_version(version) or "?"
        except Exception:  # pragma: no cover - defensive
            return version or "?"

    def _can_install(self, status: dict[str, str]) -> bool:
        """Whether a managed copy can be installed here.

        Only Windows has a downloadable archive. A *system* FFmpeg does not
        block this: installing a managed copy adds one next to the app, which
        then takes precedence — it never overwrites the user's own install.
        """
        return sys.platform == "win32"

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)
        self.status_changed.emit(text)

    def _set_buttons(self, check: bool, update: bool) -> None:
        enabled = not self._busy
        self.check_btn.setEnabled(check and enabled)
        self.update_btn.setEnabled(update and enabled)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.progress.setVisible(busy)
        if not busy:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
        self._set_buttons(check=True, update=True)

    # --- actions -----------------------------------------------------------

    def check_for_updates(self) -> None:
        """Asks the vendor for the newest release, off the UI thread."""
        if self._busy:
            return
        self._set_busy(True)
        self.progress.setRange(0, 0)  # indeterminate
        self._set_status(i18n.tr("ffmpeg_checking", "Checking for FFmpeg updates..."))
        self._set_buttons(check=False, update=False)

        def worker() -> None:
            latest = ""
            with contextlib.suppress(Exception):
                latest = _tools().latest_release_version() or ""
            with contextlib.suppress(RuntimeError):
                self._check_finished.emit(latest)

        threading.Thread(target=worker, name="ffmpeg-check", daemon=True).start()

    def _on_check_finished(self, latest: str) -> None:
        """GUI thread: report whether ``latest`` beats the installed build."""
        self._set_busy(False)
        self._latest = latest or None
        if not latest:
            self._hold_sentence(
                i18n.tr(
                    "ffmpeg_check_failed",
                    "Could not reach the FFmpeg release server. The current build is unchanged.",
                )
            )
            return
        tools = _tools()
        short = self._short(latest)
        if tools.is_newer(latest, self._version):
            self._hold_sentence(
                i18n.tr(
                    "ffmpeg_update_available",
                    "FFmpeg {version} is available (installed: {current}).",
                )
                .replace("{version}", short)
                .replace("{current}", self._short(self._version))
            )
        else:
            self._hold_sentence(
                i18n.tr("ffmpeg_up_to_date", "FFmpeg {version} is up to date.").replace(
                    "{version}", short
                )
            )

    def _hold_sentence(self, text: str) -> None:
        """Shows ``text`` and keeps it while the state is re-read in the background.

        The sentence is the answer to what the user just asked, so a status
        refresh must not overwrite it. The refresh still runs (the buttons need
        the new version), it just does not get to set the text.
        """
        self._sentence_held = True
        self._set_status(text)
        self._request_status()

    def install_latest(self) -> None:
        """Downloads and swaps in the newest FFmpeg, without a restart."""
        if self._busy:
            return
        target = self._latest
        self._set_busy(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._set_status(i18n.tr("ffmpeg_downloading", "Downloading FFmpeg..."))
        self._set_buttons(check=False, update=False)

        def on_progress(done: int, total: int) -> None:
            percent = int(done * 100 / total) if total > 0 else -1
            with contextlib.suppress(RuntimeError):
                self._progress.emit(percent)

        def worker() -> None:
            error = ""
            try:
                _tools().install_managed(
                    config.APP_DIR, version=target, progress=on_progress
                )
            except Exception as e:
                error = str(e)
            with contextlib.suppress(RuntimeError):
                self._install_finished.emit(error)

        threading.Thread(target=worker, name="ffmpeg-install", daemon=True).start()

    def _on_progress(self, percent: int) -> None:
        if percent < 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(percent)

    def _on_install_finished(self, error: str) -> None:
        """GUI thread: report the outcome of an install."""
        self._set_busy(False)
        if error:
            self._hold_sentence(
                i18n.tr("ffmpeg_install_failed", "FFmpeg update failed: {error}").replace(
                    "{error}", error
                )
            )
            dialogs.info(
                self,
                i18n.tr("ffmpeg_install_failed_title", "FFmpeg Update Failed"),
                error,
            )
            return
        # The new binary is on disk now. Read its version synchronously — this
        # slot runs on the GUI thread, but the read is a local probe that the
        # version cache makes cheap, and the dialog must name the version.
        self._sentence_held = False
        self._load_status()
        self._set_status(self._describe_installed())
        self._announce_installed()

    def _describe_installed(self) -> str:
        """The "installed" sentence, without a second status probe."""
        return i18n.tr("ffmpeg_up_to_date", "FFmpeg {version} is up to date.").replace(
            "{version}", self._short(self._version)
        )

    def _announce_installed(self) -> None:
        """GUI thread: tell the user the new build is live."""
        installed = self._short(self._version)
        self.installed.emit(installed)
        dialogs.info(
            self,
            i18n.tr("ffmpeg_install_done_title", "FFmpeg Updated"),
            i18n.tr(
                "ffmpeg_install_done",
                "FFmpeg {version} is installed. The next run uses it automatically; no restart is needed.",
            ).replace("{version}", installed),
        )

    # --- retranslation -----------------------------------------------------

    def retranslate(self) -> None:
        """Re-applies translations after a UI language switch."""
        self.check_btn.setText(i18n.tr("btn_ffmpeg_check", "Check for Updates"))
        self.update_btn.setText(i18n.tr("btn_ffmpeg_update", "Update FFmpeg"))
        self.refresh()
