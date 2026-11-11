"""Keeps child processes from flashing console windows in the windowed GUI.

The GUI is built as a *windowed* executable, so it has no console of its own.
On Windows every child process that is not explicitly marked otherwise gets a
brand-new console window, which appears as a black box that flashes on screen
and disappears — visible every time the app starts.

Startup used to spawn four such processes:

* ``ffmpeg -version`` (three times, from the FFmpeg status probe), and
* ``ver``, which ``platform.win32_ver()`` shells out to and which is reached
  from ``import torch_directml`` (torch checks the Windows version).

Rather than hunting down each spawn site — including ones inside third-party
packages we cannot edit — the default is inverted once, at start-up: every
``subprocess`` child is created with ``CREATE_NO_WINDOW`` unless the caller
asked for something specific.
"""

from __future__ import annotations

import subprocess
import sys

_APPLIED = False


def suppress_child_consoles() -> bool:
    """Makes every later ``subprocess`` spawn windowless on Windows.

    Idempotent, and a no-op on other platforms (which have no such flag).
    Returns True when the patch is active.
    """
    global _APPLIED
    if sys.platform != "win32":
        return False
    if _APPLIED:
        return True

    flag = subprocess.CREATE_NO_WINDOW
    original_init = subprocess.Popen.__init__

    def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Respect an explicit request: a caller that sets creationflags or
        # supplies a STARTUPINFO has already decided how the child should
        # appear, and overriding it could break e.g. a visible console helper.
        if not kwargs.get("creationflags") and kwargs.get("startupinfo") is None:
            kwargs["creationflags"] = flag
        return original_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = patched_init  # type: ignore[method-assign]
    _APPLIED = True
    return True
