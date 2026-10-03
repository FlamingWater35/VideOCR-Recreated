"""About tab: version info and GitHub links."""

from __future__ import annotations

import webbrowser

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from . import i18n

RELEASES_REPO_URL = "https://github.com/FlamingWater35/VideOCR-Recreated"

# Related repositories, shown in the About tab: this fork, the AMD DirectML
# fork it builds on, and the original VideOCR.
REPO_LINKS = (
    "https://github.com/BaseCrunch/VideOCR-AMD-DirectML",
    "https://github.com/timminator/VideOCR",
)


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

        version_label = QLabel(
            i18n.tr("lbl_about_version", "Version: {}")
            .replace("{}", "{version}")
            .format(version=version)
        )
        version_label.setObjectName("aboutVersion")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_label)

        layout.addSpacing(12)

        # Fork notice: this project is a fork of the original VideOCR.
        self.fork_lbl = QLabel(
            i18n.tr(
                "lbl_about_fork",
                "This project is a fork of the original VideOCR by timminator.",
            )
        )
        self.fork_lbl.setWordWrap(True)
        self.fork_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.fork_lbl)

        self.repos_lbl = QLabel(
            i18n.tr("lbl_about_related_repos", "Related repositories:")
        )
        self.repos_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.repos_lbl)

        for url in REPO_LINKS:
            link = ClickableLabel(url)
            link.setObjectName("linkLabel")
            link.setAlignment(Qt.AlignmentFlag.AlignCenter)
            link.setCursor(Qt.CursorShape.PointingHandCursor)
            link.clicked.connect(lambda u=url: webbrowser.open(u))
            layout.addWidget(link)

        layout.addSpacing(12)

        self.newest_lbl = QLabel(
            i18n.tr("lbl_get_newest", "Get the newest version here:")
        )
        self.newest_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.newest_lbl)

        releases = ClickableLabel(RELEASES_REPO_URL)
        releases.setObjectName("linkLabel")
        releases.setAlignment(Qt.AlignmentFlag.AlignCenter)
        releases.setCursor(Qt.CursorShape.PointingHandCursor)
        releases.clicked.connect(lambda: webbrowser.open(RELEASES_REPO_URL))
        layout.addWidget(releases)

        layout.addStretch(1)

    def retranslate(self, version: str) -> None:
        for label in self.findChildren(QLabel):
            if label.objectName() == "aboutVersion":
                label.setText(
                    i18n.tr("lbl_about_version", "Version: {}")
                    .replace("{}", "{version}")
                    .format(version=version)
                )
            elif label.objectName() == "aboutTitle":
                label.setText("VideOCR Recreated")
        self.fork_lbl.setText(
            i18n.tr(
                "lbl_about_fork",
                "This project is a fork of the original VideOCR by timminator.",
            )
        )
        self.repos_lbl.setText(
            i18n.tr("lbl_about_related_repos", "Related repositories:")
        )
        self.newest_lbl.setText(
            i18n.tr("lbl_get_newest", "Get the newest version here:")
        )
