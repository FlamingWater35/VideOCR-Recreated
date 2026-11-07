"""About tab: hero header, update status, and project links.

The tab is the single place where version information lives. It owns the
update indicator: a card that stays hidden until the startup check has
something to say, then switches between "checking", "available" and "up to
date". When a newer release exists the card turns gold and grows a button
that opens the latest release page.
"""

from __future__ import annotations

import webbrowser

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import i18n, resources
from .update_check import LATEST_RELEASE_URL

RELEASES_REPO_URL = "https://github.com/FlamingWater35/VideOCR-Recreated"

# Related repositories, shown in the About tab: this fork, the AMD DirectML
# fork it builds on, and the original VideOCR.
REPO_LINKS = (
    "https://github.com/BaseCrunch/VideOCR-AMD-DirectML",
    "https://github.com/timminator/VideOCR",
)

#: Update-card states. ``idle`` hides the card entirely.
UPDATE_STATES = ("idle", "checking", "available", "uptodate")


def _fmt(template: str, **values: str) -> str:
    """Substitute ``{name}`` placeholders without raising on stray braces.

    Translations are data, not format strings: a missing or malformed
    placeholder must never crash the About tab, so this replaces literally
    instead of using :meth:`str.format`.
    """
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", value)
    return out


def _restyle(widget: QWidget) -> None:
    """Re-apply QSS property selectors after changing a dynamic property."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


class ClickableLabel(QLabel):
    clicked = Signal()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class AboutTab(QWidget):
    def __init__(self, version: str, parent=None) -> None:
        super().__init__(parent)
        self._version = version
        self._latest = ""
        self._state = "idle"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(16)

        layout.addLayout(self._build_hero())
        layout.addWidget(self._build_update_card())
        layout.addWidget(self._build_repos_card())
        layout.addWidget(self._build_issue_card())
        layout.addStretch(1)

        self.retranslate(version)

    # --- construction -------------------------------------------------------
    def _build_hero(self) -> QHBoxLayout:
        """Icon + name + version pill, side by side."""
        row = QHBoxLayout()
        row.setSpacing(16)

        self.icon_lbl = QLabel()
        self.icon_lbl.setObjectName("aboutIcon")
        self.icon_lbl.setFixedSize(64, 64)
        self.icon_lbl.setScaledContents(False)
        icon = resources.icon_path()
        if icon:
            pixmap = QPixmap(icon)
            if not pixmap.isNull():
                self.icon_lbl.setPixmap(
                    pixmap.scaled(
                        64,
                        64,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
        row.addWidget(self.icon_lbl, 0, Qt.AlignmentFlag.AlignVCenter)

        text_col = QVBoxLayout()
        text_col.setSpacing(4)

        self.title_lbl = QLabel("VideOCR Recreated")
        self.title_lbl.setObjectName("aboutTitle")
        text_col.addWidget(self.title_lbl)

        self.tagline_lbl = QLabel()
        self.tagline_lbl.setObjectName("aboutTagline")
        self.tagline_lbl.setWordWrap(True)
        text_col.addWidget(self.tagline_lbl)

        self.version_lbl = QLabel()
        self.version_lbl.setObjectName("aboutVersionPill")
        self.version_lbl.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed
        )
        text_col.addWidget(self.version_lbl, 0, Qt.AlignmentFlag.AlignLeft)

        row.addLayout(text_col, 1)
        return row

    def _build_update_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("aboutUpdateCard")
        card.setProperty("state", "idle")

        row = QHBoxLayout(card)
        row.setContentsMargins(18, 14, 18, 14)
        row.setSpacing(14)

        text_col = QVBoxLayout()
        text_col.setSpacing(4)

        self.update_badge = QLabel()
        self.update_badge.setObjectName("aboutBadge")
        self.update_badge.setProperty("state", "idle")
        self.update_badge.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed
        )
        text_col.addWidget(self.update_badge, 0, Qt.AlignmentFlag.AlignLeft)

        self.update_message = QLabel()
        self.update_message.setObjectName("aboutCardBody")
        self.update_message.setWordWrap(True)
        text_col.addWidget(self.update_message)

        row.addLayout(text_col, 1)

        self.update_button = QPushButton()
        self.update_button.setObjectName("primaryButton")
        self.update_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update_button.clicked.connect(self._open_latest_release)
        row.addWidget(self.update_button, 0, Qt.AlignmentFlag.AlignVCenter)

        self.update_card = card
        card.setVisible(False)
        return card

    def _build_repos_card(self) -> QFrame:
        """Fork notice, the repositories this project builds on, and releases."""
        card = QFrame()
        card.setObjectName("aboutCard")

        col = QVBoxLayout(card)
        col.setContentsMargins(18, 14, 18, 14)
        col.setSpacing(6)

        self.fork_lbl = QLabel()
        self.fork_lbl.setObjectName("aboutCardBody")
        self.fork_lbl.setWordWrap(True)
        col.addWidget(self.fork_lbl)

        col.addSpacing(14)
        self.repos_lbl = QLabel()
        self.repos_lbl.setObjectName("aboutSectionHeading")
        col.addWidget(self.repos_lbl)
        col.addSpacing(8)

        for url in REPO_LINKS:
            link = ClickableLabel(url)
            link.setObjectName("linkLabel")
            link.setCursor(Qt.CursorShape.PointingHandCursor)
            link.clicked.connect(lambda u=url: webbrowser.open(u))
            col.addWidget(link)

        # Manual navigation stays available even when the update check is off.
        col.addSpacing(18)
        self.newest_lbl = QLabel()
        self.newest_lbl.setObjectName("aboutSectionHeading")
        col.addWidget(self.newest_lbl)
        col.addSpacing(8)

        releases = ClickableLabel(RELEASES_REPO_URL)
        releases.setObjectName("linkLabel")
        releases.setCursor(Qt.CursorShape.PointingHandCursor)
        releases.clicked.connect(lambda: webbrowser.open(RELEASES_REPO_URL))
        col.addWidget(releases)

        return card

    def _build_issue_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("aboutCard")

        col = QVBoxLayout(card)
        col.setContentsMargins(18, 14, 18, 14)
        col.setSpacing(6)

        self.issue_lbl = QLabel()
        self.issue_lbl.setObjectName("aboutCardBody")
        self.issue_lbl.setWordWrap(True)
        col.addWidget(self.issue_lbl)

        self.issue_link = ClickableLabel(f"{RELEASES_REPO_URL}/issues")
        self.issue_link.setObjectName("linkLabel")
        self.issue_link.setCursor(Qt.CursorShape.PointingHandCursor)
        self.issue_link.clicked.connect(
            lambda: webbrowser.open(f"{RELEASES_REPO_URL}/issues")
        )
        col.addWidget(self.issue_link)

        return card

    # --- update indicator ---------------------------------------------------
    def set_update_state(self, state: str, latest: str = "") -> None:
        """Switch the update card between idle/checking/available/uptodate/error."""
        if state not in UPDATE_STATES:
            state = "idle"
        self._state = state
        if latest:
            self._latest = latest
        self._apply_update_state()

    def update_state(self) -> str:
        return self._state

    def has_update(self) -> bool:
        return self._state == "available"

    def _apply_update_state(self) -> None:
        state = self._state
        self.update_card.setVisible(state != "idle")

        self.update_badge.setProperty("state", state)
        self.update_card.setProperty("state", state)
        _restyle(self.update_badge)
        _restyle(self.update_card)

        self.update_badge.setText(self._badge_text())
        self.update_message.setText(self._message_text())
        self.update_message.setVisible(bool(self.update_message.text()))
        self.update_button.setVisible(state == "available")
        self.update_button.setText(i18n.tr("about_update_button", "Open Latest Release"))

    def _badge_text(self) -> str:
        return {
            "checking": i18n.tr("about_update_checking", "Checking for updates..."),
            "available": i18n.tr("update_title", "Update Available"),
            "uptodate": i18n.tr("update_title_uptodate", "Up to Date"),
        }.get(self._state, "")

    def _message_text(self) -> str:
        if self._state == "available":
            # The current version is already in the hero pill right above, so
            # this line only names the release that is waiting.
            return _fmt(
                i18n.tr("about_update_available", "Version {version} is available."),
                version=self._latest,
            )
        if self._state == "uptodate":
            return i18n.tr(
                "update_msg_uptodate",
                "You are running the latest version of VideOCR Recreated.",
            )
        return ""

    def _open_latest_release(self) -> None:
        webbrowser.open(LATEST_RELEASE_URL)

    # --- i18n ---------------------------------------------------------------
    def retranslate(self, version: str) -> None:
        self._version = version
        self.title_lbl.setText("VideOCR Recreated")
        self.tagline_lbl.setText(
            i18n.tr(
                "about_tagline",
                "Extract hardcoded subtitles from video with OCR.",
            )
        )
        self.version_lbl.setText(
            _fmt(
                i18n.tr("lbl_about_version", "Version: {version}"),
                version=version,
            )
        )
        self.issue_lbl.setText(
            i18n.tr(
                "lbl_bug_report",
                "Found a bug or have a suggestion? Feel free to open an issue at:",
            )
        )
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
        for label in self.findChildren(QLabel):
            key = label.property("i18n_key")
            if key:
                label.setText(i18n.tr(str(key), str(label.property("i18n_default"))))
        self._apply_update_state()
