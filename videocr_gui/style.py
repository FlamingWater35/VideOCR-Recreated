"""QSS stylesheet for the VideOCR Recreated "neon dark" theme.

The theme uses a deep navy surface palette with a cyan/violet accent pair:

    --bg-deep        #0f1117   window / dialog background
    --bg-surface     #171a22   tabs, inputs, group boxes
    --bg-raised      #1f2430   hovered surfaces
    --border         #2a3140   default borders
    --border-strong  #3a4458   hovered / emphasized borders
    --accent         #22d3ee   cyan   (primary actions, selection, focus)
    --accent-2       #8b5cf6   violet (gradient partner, special states)
    --danger         #f87171   destructive actions
    --ok             #34d399   success states
    --text           #e6e9ef   primary text
    --text-dim       #9aa4b6   secondary text
    --text-faint     #6b7486   disabled text
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

QSS = """
* {
    font-family: "Segoe UI", "Noto Sans", sans-serif;
    font-size: 13px;
    color: #e6e9ef;
}

/* --- top-level surfaces -------------------------------------------------- */
QMainWindow, QDialog {
    background-color: #0f1117;
    color: #e6e9ef;
}
QWidget {
    background-color: transparent;
    color: #e6e9ef;
}
QWidget#qt_scrollarea_viewport {
    background-color: #0f1117;
}
QMessageBox, QInputDialog {
    background-color: #0f1117;
}

/* --- tabs ---------------------------------------------------------------- */
QTabWidget::pane {
    border: 1px solid #2a3140;
    border-radius: 8px;
    background-color: #14161d;
    top: -1px;
}
QTabBar::tab {
    background: #1b1f28;
    color: #9aa4b6;
    padding: 8px 20px;
    border: 1px solid transparent;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 4px;
}
QTabBar::tab:selected {
    background: #14161d;
    color: #22d3ee;
    font-weight: 600;
    border: 1px solid #2a3140;
    border-bottom: 1px solid #14161d;
}
QTabBar::tab:hover:!selected {
    background: #232836;
    color: #e6e9ef;
}

/* --- buttons ------------------------------------------------------------- */
QPushButton {
    background-color: #262c3b;
    border: 1px solid #39415a;
    border-radius: 6px;
    padding: 6px 14px;
    color: #e6e9ef;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #303850;
    border-color: #4b5678;
}
QPushButton:pressed {
    background-color: #1d2230;
}
QPushButton:disabled {
    background-color: #171b24;
    color: #6b7486;
    border-color: #242a37;
}
QPushButton:focus {
    border: 1px solid #22d3ee;
}

/* primary (Run / Start Queue / Proceed Now) */
QPushButton#primaryButton {
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                stop: 0 #0ea5e9, stop: 1 #8b5cf6);
    border: 1px solid #38bdf8;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#primaryButton:hover {
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                stop: 0 #0c96d8, stop: 1 #7c4def);
    border-color: #7dd3fc;
}
QPushButton#primaryButton:pressed {
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                stop: 0 #0a86c3, stop: 1 #6d3fe0);
}
QPushButton#primaryButton:disabled {
    background: #1d2230;
    border-color: #2a3140;
    color: #6b7486;
}

/* danger (Cancel / Stop / Remove) */
QPushButton#dangerButton {
    background-color: #46252b;
    border: 1px solid #7f2f3b;
    color: #fda4af;
}
QPushButton#dangerButton:hover {
    background-color: #5c2b34;
    border-color: #f87171;
    color: #ffffff;
}
QPushButton#dangerButton:pressed {
    background-color: #381d22;
}
QPushButton#dangerButton:disabled {
    background-color: #171b24;
    border-color: #242a37;
    color: #6b7486;
}

/* icon-only reorder buttons (▲/▼) */
QPushButton[reorder="true"] {
    padding: 4px 10px;
    font-size: 11px;
}

/* --- inputs -------------------------------------------------------------- */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #171a22;
    border: 1px solid #2a3140;
    border-radius: 6px;
    padding: 5px 9px;
    color: #e6e9ef;
    selection-background-color: #0ea5e9;
    selection-color: #ffffff;
}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #3a4458;
}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #22d3ee;
}
QComboBox:disabled, QLineEdit:disabled {
    background-color: #12151c;
    color: #6b7486;
    border-color: #222834;
}
QComboBox::drop-down {
    border: none;
    width: 26px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #9aa4b6;
    margin-right: 6px;
}
QComboBox::down-arrow:hover {
    border-top-color: #22d3ee;
}

/* combo popup */
QComboBox QAbstractItemView {
    background-color: #1b1f28;
    border: 1px solid #3a4458;
    border-radius: 6px;
    padding: 4px;
    selection-background-color: #0ea5e9;
    selection-color: #ffffff;
    outline: none;
}
QComboBox QAbstractItemView::item {
    padding: 5px 8px;
    border-radius: 4px;
}
QComboBox QAbstractItemView::item:hover {
    background-color: #2a3140;
}

/* generic popup menu (combos, context menus) */
QMenu {
    background-color: #1b1f28;
    border: 1px solid #3a4458;
    border-radius: 6px;
    padding: 4px;
}
QMenu::item {
    padding: 5px 18px 5px 8px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #0ea5e9;
    color: #ffffff;
}
QMenu::item:disabled {
    color: #6b7486;
}

/* --- checkboxes / radio -------------------------------------------------- */
QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #4b5678;
    border-radius: 4px;
    background: #171a22;
}
QCheckBox::indicator:hover {
    border-color: #22d3ee;
}
QCheckBox::indicator:checked {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #0ea5e9, stop: 1 #8b5cf6);
    border-color: #38bdf8;
    image: none;
}
QCheckBox::indicator:disabled {
    border-color: #2a3140;
    background: #12151c;
}
QRadioButton {
    spacing: 8px;
}
QRadioButton::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid #4b5678;
    border-radius: 8px;
    background: #171a22;
}
QRadioButton::indicator:checked {
    background: #22d3ee;
    border: 4px solid #171a22;
}

/* --- sliders ------------------------------------------------------------- */
QSlider::groove:horizontal {
    height: 6px;
    background: #262c3b;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 15px;
    height: 15px;
    margin: -5px 0;
    border-radius: 8px;
    background: #22d3ee;
    border: 2px solid #0f1117;
}
QSlider::handle:horizontal:hover {
    background: #67e8f9;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                stop: 0 #0ea5e9, stop: 1 #8b5cf6);
    border-radius: 3px;
}
QSlider::groove:vertical {
    width: 6px;
    background: #262c3b;
    border-radius: 3px;
}
QSlider::handle:vertical {
    width: 15px;
    height: 15px;
    margin: 0 -5px;
    border-radius: 8px;
    background: #22d3ee;
    border: 2px solid #0f1117;
}
QSlider::sub-page:vertical {
    background: qlineargradient(x1: 0, y1: 1, x2: 0, y2: 0,
                                stop: 0 #0ea5e9, stop: 1 #8b5cf6);
    border-radius: 3px;
}

/* --- progress ------------------------------------------------------------ */
QProgressBar {
    border: 1px solid #2a3140;
    border-radius: 6px;
    background: #171a22;
    text-align: center;
    color: #e6e9ef;
    font-weight: 600;
}
QProgressBar::chunk {
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                stop: 0 #0ea5e9, stop: 1 #22d3ee, stop: 0.85 #8b5cf6);
    border-radius: 5px;
    margin-right: 2px;
}

/* --- log pane ------------------------------------------------------------ */
QPlainTextEdit {
    background-color: #0c0e14;
    color: #cdd3de;
    border: 1px solid #2a3140;
    border-radius: 6px;
    padding: 4px;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 12px;
    selection-background-color: #0ea5e9;
    selection-color: #ffffff;
}
QPlainTextEdit:focus {
    border: 1px solid #22d3ee;
}

/* --- group boxes --------------------------------------------------------- */
QGroupBox#sectionBox {
    border: 1px solid #2a3140;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 14px;
    background-color: #14161d;
}
QGroupBox#sectionBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #22d3ee;
    font-weight: 600;
}
QGroupBox {
    border: 1px solid #2a3140;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 10px;
}

/* --- labels -------------------------------------------------------------- */
QLabel#formLabel {
    color: #9aa4b6;
}
QLabel#aboutTitle {
    font-size: 30px;
    font-weight: 800;
    color: #ffffff;
}
QLabel#aboutVersion {
    color: #9aa4b6;
}
QLabel#linkLabel {
    color: #22d3ee;
    text-decoration: underline;
}
QLabel#linkLabel:hover {
    color: #67e8f9;
}
QLabel#etaLabel {
    color: #9aa4b6;
}
QLabel#countdownTitle {
    font-size: 16px;
    font-weight: 700;
    color: #22d3ee;
}

/* --- tables -------------------------------------------------------------- */
QTableWidget {
    background-color: #14161d;
    alternate-background-color: #1a1e28;
    gridline-color: #2a3140;
    border: 1px solid #2a3140;
    border-radius: 6px;
    selection-background-color: #1e3a5f;
    selection-color: #ffffff;
}
QHeaderView::section {
    background-color: #1b1f28;
    color: #9aa4b6;
    border: none;
    border-right: 1px solid #2a3140;
    border-bottom: 1px solid #2a3140;
    padding: 7px;
    font-weight: 600;
}
QHeaderView::section:hover {
    color: #e6e9ef;
}
QTableCornerButton::section {
    background-color: #1b1f28;
    border: none;
}
QTableWidget::item {
    padding: 4px 6px;
}

/* --- scroll bars --------------------------------------------------------- */
QScrollBar:vertical {
    background: transparent;
    width: 12px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #2e3546;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #3a4458;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: transparent;
    height: 12px;
    margin: 2px;
}
QScrollBar::handle:horizontal {
    background: #2e3546;
    border-radius: 4px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background: #3a4458;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
QScrollBar::add-page, QScrollBar::sub-page {
    background: transparent;
}

/* --- tooltips ------------------------------------------------------------ */
QToolTip {
    background-color: #1b1f28;
    color: #e6e9ef;
    border: 1px solid #22d3ee;
    border-radius: 4px;
    padding: 5px 8px;
}

/* --- scroll area --------------------------------------------------------- */
QScrollArea {
    border: none;
    background: transparent;
}
"""


def apply(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
