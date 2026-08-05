"""About tab: version info and GitHub links."""

from __future__ import annotations

import webbrowser

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from . import i18n


class ClickableLabel(QLabel):
    clicked = Signal()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class AboutTab(QWidget):
    def __init__(self, version: str, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)

        title = QLabel("VideOCR Recreated")
        title.setObjectName("aboutTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        version_label = QLabel(i18n.tr("lbl_about_version", "Version: {}").replace("{}", "{version}").format(version=version))
        version_label.setObjectName("aboutVersion")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_label)

        layout.addSpacing(12)

        newest = QLabel(i18n.tr("lbl_get_newest", "Get the newest version here:"))
        newest.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(newest)

        releases = ClickableLabel("https://github.com/timminator/VideOCR/releases")
        releases.setObjectName("linkLabel")
        releases.setAlignment(Qt.AlignmentFlag.AlignCenter)
        releases.setCursor(Qt.CursorShape.PointingHandCursor)
        releases.clicked.connect(lambda: webbrowser.open("https://github.com/timminator/VideOCR/releases"))
        layout.addWidget(releases)

        layout.addSpacing(8)

        bug = QLabel(i18n.tr("lbl_bug_report", "Found a bug or have a suggestion? Feel free to open an issue at:"))
        bug.setWordWrap(True)
        bug.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(bug)

        issues = ClickableLabel("https://github.com/timminator/VideOCR/issues")
        issues.setObjectName("linkLabel")
        issues.setAlignment(Qt.AlignmentFlag.AlignCenter)
        issues.setCursor(Qt.CursorShape.PointingHandCursor)
        issues.clicked.connect(lambda: webbrowser.open("https://github.com/timminator/VideOCR/issues"))
        layout.addWidget(issues)

        layout.addStretch(1)

    def retranslate(self, version: str) -> None:
        for label in self.findChildren(QLabel):
            if label.objectName() == "aboutVersion":
                label.setText(i18n.tr("lbl_about_version", "Version: {}").replace("{}", "{version}").format(version=version))
            elif label.objectName() == "aboutTitle":
                label.setText("VideOCR Recreated")
