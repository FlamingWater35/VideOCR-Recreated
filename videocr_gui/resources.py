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


def _new_icon_pixmap(size: int = 16):
    """Creates a transparent HiDPI pixmap + painter for icon drawing."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPainter, QPixmap

    pm = QPixmap(size * 2, size * 2)
    pm.setDevicePixelRatio(2.0)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    return pm, p


def center_crop_icon():
    """'Center crop box' icon: focus brackets around a center dot."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QColor, QIcon, QPen

    from .style import PALETTE

    size = 16
    pm, p = _new_icon_pixmap(size)

    pen = QPen(QColor(PALETTE["text"]), 1.5)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)

    m = 1.5     # margin from the icon edge
    arm = 4.0   # focus-bracket arm length
    # corner focus brackets
    p.drawLine(QPointF(m, m + arm), QPointF(m, m))
    p.drawLine(QPointF(m, m), QPointF(m + arm, m))
    p.drawLine(QPointF(size - m - arm, m), QPointF(size - m, m))
    p.drawLine(QPointF(size - m, m), QPointF(size - m, m + arm))
    p.drawLine(QPointF(m, size - m - arm), QPointF(m, size - m))
    p.drawLine(QPointF(m, size - m), QPointF(m + arm, size - m))
    p.drawLine(QPointF(size - m - arm, size - m), QPointF(size - m, size - m))
    p.drawLine(QPointF(size - m, size - m), QPointF(size - m, size - m - arm))

    # center dot (gold accent)
    c = size / 2.0
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(PALETTE["accent"]))
    p.drawEllipse(QPointF(c, c), 1.7, 1.7)
    p.end()
    return QIcon(pm)


def center_horizontal_icon():
    """'Center horizontally' icon: a gold vertical axis with inward arrows."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QColor, QIcon, QPen

    from .style import PALETTE

    size = 16
    pm, p = _new_icon_pixmap(size)
    c = size / 2.0

    # vertical center axis (gold)
    axis_pen = QPen(QColor(PALETTE["accent"]), 1.5)
    axis_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(axis_pen)
    p.drawLine(QPointF(c, 2.0), QPointF(c, size - 2.0))

    # inward arrows (text color)
    pen = QPen(QColor(PALETTE["text"]), 1.5)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    # left arrow -> points right toward the axis
    p.drawLine(QPointF(2.0, c), QPointF(c - 2.5, c))
    p.drawLine(QPointF(c - 2.5, c), QPointF(c - 4.5, c - 1.8))
    p.drawLine(QPointF(c - 2.5, c), QPointF(c - 4.5, c + 1.8))
    # right arrow -> points left toward the axis
    p.drawLine(QPointF(size - 2.0, c), QPointF(c + 2.5, c))
    p.drawLine(QPointF(c + 2.5, c), QPointF(c + 4.5, c - 1.8))
    p.drawLine(QPointF(c + 2.5, c), QPointF(c + 4.5, c + 1.8))
    p.end()
    return QIcon(pm)


def center_vertical_icon():
    """'Center vertically' icon: a gold horizontal axis with inward arrows."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QColor, QIcon, QPen

    from .style import PALETTE

    size = 16
    pm, p = _new_icon_pixmap(size)
    c = size / 2.0

    # horizontal center axis (gold)
    axis_pen = QPen(QColor(PALETTE["accent"]), 1.5)
    axis_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(axis_pen)
    p.drawLine(QPointF(2.0, c), QPointF(size - 2.0, c))

    # inward arrows (text color)
    pen = QPen(QColor(PALETTE["text"]), 1.5)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    # top arrow -> points down toward the axis
    p.drawLine(QPointF(c, 2.0), QPointF(c, c - 2.5))
    p.drawLine(QPointF(c, c - 2.5), QPointF(c - 1.8, c - 4.5))
    p.drawLine(QPointF(c, c - 2.5), QPointF(c + 1.8, c - 4.5))
    # bottom arrow -> points up toward the axis
    p.drawLine(QPointF(c, size - 2.0), QPointF(c, c + 2.5))
    p.drawLine(QPointF(c, c + 2.5), QPointF(c - 1.8, c + 4.5))
    p.drawLine(QPointF(c, c + 2.5), QPointF(c + 1.8, c + 4.5))
    p.end()
    return QIcon(pm)
