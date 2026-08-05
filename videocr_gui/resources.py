"""Resources: icon paths and scaling helpers."""

from __future__ import annotations

import os
import sys

from .config import APP_DIR


def icon_path() -> str:
    """Returns the app icon path (.ico on Windows, .png elsewhere)."""
    if sys.platform == "win32":
        candidate = os.path.join(APP_DIR, "Installer", "VideOCR.ico")
        if os.path.exists(candidate):
            return candidate
    candidate = os.path.join(APP_DIR, "Installer", "VideOCR.png")
    if os.path.exists(candidate):
        return candidate
    candidate = os.path.join(APP_DIR, "VideOCR.png")
    return candidate if os.path.exists(candidate) else ""


def notification_icon_path() -> str:
    """Path used by winotify/plyer notifications (same icon, different fallbacks)."""
    return icon_path()
