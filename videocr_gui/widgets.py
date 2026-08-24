"""Shared custom widgets for VideOCR Recreated."""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QComboBox,
    QProgressBar,
    QSlider,
    QStyle,
    QStyleOptionProgressBar,
)

from .style import PALETTE


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


class OutlinedProgressBar(QProgressBar):
    """A QProgressBar that renders the percentage text with an outline.

    Qt Style Sheets cannot add an outline/shadow to the progress-bar label,
    so the text is painted manually on top of the style-rendered groove and
    chunk. The style draws everything *except* the label (``option.text`` is
    cleared before ``drawControl``), then the percentage is drawn as a
    ``QPainterPath`` in two passes so it stays readable whether it sits on the
    dark groove or the gold chunk:

    1. Stroke the glyph contour with the outline color (black ring).
    2. Fill the glyph interior with the text color (white) on top.

    Drawing the fill last guarantees the white interior is never covered by
    the centered stroke. The existing ``QProgressBar { ... }`` QSS rule in
    ``style.py`` still applies (Qt type selectors match subclasses), so the
    border, background and gold chunk styling are preserved.
    """

    #: Outline ring around the percentage glyphs.
    OUTLINE_COLOR = QColor(PALETTE["bg_deep"])
    #: Percentage glyph fill.
    TEXT_COLOR = QColor(PALETTE["text_emphasis"])
    #: Outline stroke width in pixels. Only the outer half of the centered
    #: stroke remains visible (the white fill covers the inner half), so use
    #: ~2x the desired visible outline thickness.
    OUTLINE_WIDTH = 3.0

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        option = QStyleOptionProgressBar()
        self.initStyleOption(option)
        text = option.text
        option.text = ""  # suppress the style-drawn label; we paint our own
        self.style().drawControl(
            QStyle.ControlElement.CE_ProgressBar, option, painter, self
        )

        if not text:
            return

        font = self.font()
        metrics = QFontMetrics(font)
        rect = option.rect
        text_width = metrics.horizontalAdvance(text)
        x = rect.x() + (rect.width() - text_width) / 2.0
        baseline_y = (
            rect.y() + (rect.height() - metrics.height()) / 2.0 + metrics.ascent()
        )

        path = QPainterPath()
        # OddEvenFill keeps the counters of glyphs like 0/6/8/9/% hollow.
        path.setFillRule(Qt.FillRule.OddEvenFill)
        path.addText(QPointF(x, baseline_y), font, text)

        # Pass 1: black outline. The stroke is centered on the glyph contour,
        # so it extends OUTLINE_WIDTH/2 outward (visible) and inward (covered
        # by the white fill in pass 2).
        painter.setPen(
            QPen(
                self.OUTLINE_COLOR,
                self.OUTLINE_WIDTH,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        # Pass 2: white glyph fill on top, so the percentage reads white with
        # a clean black ring around it.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.TEXT_COLOR))
        painter.drawPath(path)
