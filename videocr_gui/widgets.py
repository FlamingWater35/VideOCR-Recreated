"""Shared custom widgets for VideOCR Recreated."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QSlider


class WheelGuardComboBox(QComboBox):
    """A QComboBox that ignores mouse-wheel events.

    Qt's default QComboBox changes the selected option when the user scrolls
    the mouse wheel over it. That is easy to trigger accidentally while
    scrolling a settings page, so this subclass swallows wheel events while
    keeping normal mouse/keyboard interaction intact.
    """

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API name
        event.ignore()


class ClickableSlider(QSlider):
    """A QSlider where clicking anywhere on the track seeks to that position.

    Qt's default QSlider only jumps to the clicked point when a ``pageStep``
    is configured; with pageStep 0 the click is ignored unless it lands on the
    handle. This subclass maps a left click on the groove to the corresponding
    value so the user can click anywhere on the seek bar.
    """

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.orientation() == Qt.Orientation.Horizontal
        ):
            self._seek_to_click(event.position().x())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API name
        # Keep seeking while dragging (left button held) anywhere on the track.
        if (
            event.buttons() & Qt.MouseButton.LeftButton
            and self.orientation() == Qt.Orientation.Horizontal
        ):
            self._seek_to_click(event.position().x())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def _seek_to_click(self, x: float) -> None:
        if self.maximum() <= self.minimum():
            return
        width = self.width()
        if width <= 0:
            return
        # Account for the handle width so the extremes map to the ends.
        handle_w = self.handle_width()
        usable = max(1, width - handle_w)
        ratio = (x - handle_w / 2.0) / usable
        ratio = max(0.0, min(1.0, ratio))
        value = self.minimum() + int(round(ratio * (self.maximum() - self.minimum())))
        self.setValue(value)

    def handle_width(self) -> int:
        """Best-effort handle width via the style's sub-control rect."""
        try:
            from PySide6.QtWidgets import QStyle

            opt = self._style_option()
            rect = self.style().subControlRect(
                QStyle.ComplexControl.CC_Slider,
                opt,
                QStyle.SubControl.SC_SliderHandle,
                self,
            )
            if rect.isValid() and rect.width() > 0:
                return rect.width()
        except Exception:
            pass
        return 20

    def _style_option(self) -> object:
        from PySide6.QtWidgets import QStyleOptionSlider

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        return opt
