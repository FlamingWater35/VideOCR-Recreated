"""Shared custom widgets for VideOCR Recreated."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox


class WheelGuardComboBox(QComboBox):
    """A QComboBox that ignores mouse-wheel events.

    Qt's default QComboBox changes the selected option when the user scrolls
    the mouse wheel over it. That is easy to trigger accidentally while
    scrolling a settings page, so this subclass swallows wheel events while
    keeping normal mouse/keyboard interaction intact.
    """

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API name
        event.ignore()
