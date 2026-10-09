"""Update tab: shown only when a newer release is available on GitHub."""

from __future__ import annotations

import webbrowser

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from . import i18n
from .about_tab import ClickableLabel
from .update_check import LATEST_RELEASE_URL


class UpdateTab(QWidget):
    """Tells the user an update exists and links to the latest release."""

    def __init__(
        self, latest_version: str, current_version: str, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._latest = latest_version
        self._current = current_version

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)

        self.title_lbl = QLabel()
        self.title_lbl.setObjectName("aboutTitle")
        self.title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_lbl)

        self.available_lbl = QLabel()
        self.available_lbl.setWordWrap(True)
        self.available_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.available_lbl)

        self.current_lbl = QLabel()
        self.current_lbl.setWordWrap(True)
        self.current_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.current_lbl)

        self.download_lbl = QLabel()
        self.download_lbl.setWordWrap(True)
        self.download_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.download_lbl)

        layout.addSpacing(8)

        self.link_lbl = ClickableLabel(LATEST_RELEASE_URL)
        self.link_lbl.setObjectName("linkLabel")
        self.link_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.link_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        self.link_lbl.clicked.connect(lambda: webbrowser.open(LATEST_RELEASE_URL))
        layout.addWidget(self.link_lbl)

        layout.addStretch(1)
        self.retranslate()

    def set_versions(self, latest_version: str, current_version: str) -> None:
        self._latest = latest_version
        self._current = current_version
        self.retranslate()

    def retranslate(self) -> None:
        self.title_lbl.setText(i18n.tr("update_title", "Update Available"))
        self.available_lbl.setText(
            i18n.tr(
                "update_available_1",
                "A new version of VideOCR Recreated ({}) is available!",
            ).replace("{}", self._latest)
        )
        self.current_lbl.setText(
            i18n.tr("update_available_2", "You are currently using version {}.")
            .replace("{}", self._current)
        )
        self.download_lbl.setText(
            i18n.tr(
                "update_available_3",
                "Click the link below to visit the download page:",
            )
        )
