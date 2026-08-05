"""Modal dialogs: info, yes/no, and the post-completion countdown dialog."""

from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from . import i18n


def _setup_dialog(dialog: QDialog, parent: object | None = None) -> None:
    dialog.setWindowModality(Qt.WindowModality.WindowModal)
    dialog.setModal(True)
    dialog.setMinimumWidth(420)


class InfoDialog(QDialog):
    """Simple centered info dialog with an OK button."""

    def __init__(self, title: str, message: str, parent=None) -> None:
        super().__init__(parent)
        _setup_dialog(self, parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        label = QLabel(message)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(label)

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        box.accepted.connect(self.accept)
        box.button(QDialogButtonBox.StandardButton.Ok).setText(i18n.tr("btn_ok", "OK"))
        layout.addWidget(box)


class YesNoDialog(QDialog):
    """Yes/No confirmation dialog; returns True if Yes was chosen."""

    def __init__(self, title: str, message: str, parent=None, default_yes: bool = False) -> None:
        super().__init__(parent)
        _setup_dialog(self, parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        label = QLabel(message)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(label)

        box = QDialogButtonBox()
        yes_btn = QPushButton(i18n.tr("btn_yes", "Yes"))
        no_btn = QPushButton(i18n.tr("btn_no", "No"))
        box.addButton(yes_btn, QDialogButtonBox.ButtonRole.AcceptRole)
        box.addButton(no_btn, QDialogButtonBox.ButtonRole.RejectRole)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        if default_yes:
            yes_btn.setFocus()
        else:
            no_btn.setFocus()


def info(parent, title: str, message: str) -> None:
    InfoDialog(title, message, parent).exec()


def ask_yes_no(parent, title: str, message: str, default_yes: bool = False) -> bool:
    dlg = YesNoDialog(title, message, parent, default_yes=default_yes)
    return dlg.exec() == QDialog.DialogCode.Accepted


class CountdownDialog(QDialog):
    """Post-completion action countdown (60 s) with Proceed/Cancel."""

    def __init__(self, parent, action_text: str, timeout_seconds: int = 60) -> None:
        super().__init__(parent)
        _setup_dialog(self, parent)
        self.setWindowTitle(i18n.tr("title_countdown", "Action Required"))
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._timeout = timeout_seconds
        self._counter = timeout_seconds
        self._proceed = False

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        title = QLabel(i18n.tr("title_countdown", "Action Required"))
        title.setObjectName("countdownTitle")
        layout.addWidget(title)

        self._label = QLabel(
            i18n.tr("lbl_action_countdown", "System will execute '{}' in {} seconds.").format(action_text, self._counter)
        )
        self._label.setWordWrap(True)
        layout.addWidget(self._label)

        row = QHBoxLayout()
        proceed_btn = QPushButton(i18n.tr("btn_proceed", "Proceed Now"))
        cancel_btn = QPushButton(i18n.tr("btn_cancel", "Cancel"))
        proceed_btn.clicked.connect(self._on_proceed)
        cancel_btn.clicked.connect(self.reject)
        row.addStretch(1)
        row.addWidget(proceed_btn)
        row.addWidget(cancel_btn)
        row.addStretch(1)
        layout.addLayout(row)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000)

    def _tick(self) -> None:
        self._counter -= 1
        if self._counter <= 0:
            self._proceed = True
            self.accept()
            return
        self._label.setText(
            i18n.tr("lbl_action_countdown", "System will execute '{}' in {} seconds.").format(
                self._action_text_placeholder(), self._counter
            )
        )

    def _action_text_placeholder(self) -> str:
        return self._label.text().split("'")[1] if "'" in self._label.text() else ""

    def _on_proceed(self) -> None:
        self._proceed = True
        self.accept()

    @staticmethod
    def run(parent, action_text: str, timeout_seconds: int = 60) -> bool:
        dlg = CountdownDialog(parent, action_text, timeout_seconds)
        dlg._timer.start(1000)
        dlg.exec()
        return dlg._proceed
