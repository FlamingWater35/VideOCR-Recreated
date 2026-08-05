"""QSS stylesheet for the VideOCR Recreated dark theme."""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

QSS = """
* {
    font-family: "Segoe UI", "Noto Sans", sans-serif;
    font-size: 13px;
}
QMainWindow, QDialog, QWidget {
    background-color: #1e1f24;
    color: #e8e8e8;
}
QTabWidget::pane {
    border: 1px solid #3a3d45;
    border-radius: 4px;
    top: -1px;
}
QTabBar::tab {
    background: #26282f;
    color: #b8bcc4;
    padding: 7px 18px;
    border: 1px solid #3a3d45;
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #32353d;
    color: #ffffff;
}
QTabBar::tab:hover:!selected {
    background: #2c2f36;
}
QPushButton {
    background-color: #3a3f4b;
    border: 1px solid #4a5060;
    border-radius: 5px;
    padding: 6px 14px;
    color: #ffffff;
}
QPushButton:hover {
    background-color: #454b59;
}
QPushButton:pressed {
    background-color: #2f333d;
}
QPushButton:disabled {
    background-color: #2a2c33;
    color: #6a6e76;
    border-color: #33363e;
}
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #26282f;
    border: 1px solid #3a3d45;
    border-radius: 4px;
    padding: 4px 8px;
    color: #e8e8e8;
    selection-background-color: #3d6de8;
}
QComboBox:disabled, QLineEdit:disabled {
    background-color: #212329;
    color: #6a6e76;
}
QComboBox::drop-down {
    border: none;
    width: 22px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #b8bcc4;
    margin-right: 6px;
}
QCheckBox {
    spacing: 7px;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid #4a5060;
    border-radius: 3px;
    background: #26282f;
}
QCheckBox::indicator:checked {
    background: #3d6de8;
    border-color: #3d6de8;
}
QSlider::groove:horizontal {
    height: 5px;
    background: #3a3d45;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: #3d6de8;
}
QSlider::sub-page:horizontal {
    background: #3d6de8;
    border-radius: 2px;
}
QProgressBar {
    border: 1px solid #3a3d45;
    border-radius: 4px;
    background: #26282f;
    text-align: center;
    color: #e8e8e8;
}
QProgressBar::chunk {
    background: #3d6de8;
    border-radius: 3px;
}
QPlainTextEdit {
    background-color: #141519;
    color: #d6d8de;
    border: 1px solid #3a3d45;
    border-radius: 4px;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 12px;
}
QGroupBox#sectionBox {
    border: 1px solid #3a3d45;
    border-radius: 6px;
    margin-top: 4px;
    padding-top: 8px;
}
QLabel#formLabel {
    color: #b8bcc4;
}
QLabel#aboutTitle {
    font-size: 26px;
    font-weight: 700;
    color: #ffffff;
}
QLabel#aboutVersion {
    color: #9aa0aa;
}
QLabel#linkLabel {
    color: #6da1ff;
    text-decoration: underline;
}
QLabel#etaLabel {
    color: #9aa0aa;
}
QLabel#countdownTitle {
    font-size: 15px;
    font-weight: 600;
}
QScrollArea {
    border: none;
}
QTableWidget {
    background-color: #1e1f24;
    alternate-background-color: #23252c;
    gridline-color: #3a3d45;
    border: 1px solid #3a3d45;
    border-radius: 4px;
}
QHeaderView::section {
    background-color: #2c2f36;
    color: #c8ccd4;
    border: none;
    border-right: 1px solid #3a3d45;
    padding: 5px;
}
QToolTip {
    background-color: #2c2f36;
    color: #e8e8e8;
    border: 1px solid #4a5060;
}
"""


def apply(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
