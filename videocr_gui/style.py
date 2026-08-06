"""QSS stylesheet for the VideOCR Recreated "warm obsidian & gold" theme.

Design language (see DESIGN.md): a warm, near-black obsidian foundation lit
by a single imperial-gold signal (Shēng Luó) with a small vocabulary of
meaningful semantic states. There is no light theme and no cold "AI-blue"
accent — gold marks actions, focus, selection and live work.

Palette:

    --bg-deep        #0c0a07   window / dialog background (obsidian)
    --bg-surface     #16110a   tabs, inputs, group boxes (surface)
    --bg-raised      #201810   hovered surfaces (raised)
    --bg-hover       #2a2114   interactive hover wash
    --border         #2c2316   default borders (gold-tinted hairline)
    --border-strong  #3a2f1c   hovered / emphasized borders, scrollbar handles
    --accent         #e1a636   gold — primary actions, selection, focus
    --accent-hi      #ffd98a   gold highlight — hover, bloom core
    --accent-deep    #8f661f   gold deep — pressed, gradient foot
    --danger         #ff8a8a   coral — destructive actions, errors
    --ok             #9bd6a0   jade — success states
    --text           #ecdfc6   primary text
    --text-dim       #9c8b69   secondary text
    --text-faint     #6f6347   disabled text
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

# Shared palette (used by style.py and imported by other modules that draw
# colors directly, e.g. the queue table status colors).
PALETTE: dict[str, str] = {
    "bg_deep": "#0c0a07",
    "bg_surface": "#16110a",
    "bg_raised": "#201810",
    "bg_hover": "#2a2114",
    "border": "#2c2316",
    "border_strong": "#3a2f1c",
    "accent": "#e1a636",
    "accent_hi": "#ffd98a",
    "accent_deep": "#8f661f",
    "amber": "#f0b53e",
    "text_emphasis": "#fff4dc",
    "danger": "#ff8a8a",
    "ok": "#9bd6a0",
    "text": "#ecdfc6",
    "text_dim": "#9c8b69",
    "text_faint": "#6f6347",
}
QSS = """
* {
    font-family: "Segoe UI", "Noto Sans", sans-serif;
    font-size: 13px;
    color: #ecdfc6;
}

/* --- top-level surfaces -------------------------------------------------- */
QMainWindow, QDialog {
    background-color: #0c0a07;
    color: #ecdfc6;
}
QWidget {
    background-color: transparent;
    color: #ecdfc6;
}
QWidget#qt_scrollarea_viewport {
    background-color: #0c0a07;
}
QMessageBox, QInputDialog {
    background-color: #0c0a07;
}

/* --- tabs ---------------------------------------------------------------- */
QTabWidget::pane {
    border: 1px solid #2c2316;
    border-radius: 8px;
    background-color: #100c08;
    top: -1px;
}
QTabBar::tab {
    background: #16110a;
    color: #9c8b69;
    padding: 8px 20px;
    border: 1px solid transparent;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 4px;
}
QTabBar::tab:selected {
    background: #100c08;
    color: #e1a636;
    font-weight: 600;
    border: 1px solid #2c2316;
    border-bottom: 1px solid #100c08;
}
QTabBar::tab:hover:!selected {
    background: #201810;
    color: #ecdfc6;
}

/* --- buttons ------------------------------------------------------------- */
QPushButton {
    background-color: #201810;
    border: 1px solid #3a2f1c;
    border-radius: 6px;
    padding: 6px 14px;
    color: #ecdfc6;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #2a2114;
    border-color: #8f661f;
}
QPushButton:pressed {
    background-color: #16110a;
}
QPushButton:disabled {
    background-color: #100c08;
    color: #6f6347;
    border-color: #2c2316;
}
QPushButton:focus {
    border: 1px solid #e1a636;
}

/* primary (Run / Start Queue / Proceed Now) */
QPushButton#primaryButton {
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                stop: 0 #b8861f, stop: 1 #e1a636);
    border: 1px solid #e1a636;
    color: #0c0a07;
    font-weight: 600;
}
QPushButton#primaryButton:hover {
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                stop: 0 #c9932a, stop: 1 #ffd98a);
    border-color: #ffd98a;
}
QPushButton#primaryButton:pressed {
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                stop: 0 #8f661f, stop: 1 #c9932a);
}
QPushButton#primaryButton:disabled {
    background: #100c08;
    border-color: #2c2316;
    color: #6f6347;
}

/* danger (Cancel / Stop / Remove) */
QPushButton#dangerButton {
    background-color: #3a1715;
    border: 1px solid #7f2f2a;
    color: #ffb4a8;
}
QPushButton#dangerButton:hover {
    background-color: #4d1f1a;
    border-color: #ff8a8a;
    color: #ffd9c4;
}
QPushButton#dangerButton:pressed {
    background-color: #2a100e;
}
QPushButton#dangerButton:disabled {
    background-color: #100c08;
    border-color: #2c2316;
    color: #6f6347;
}

/* icon-only reorder buttons (▲/▼) */
QPushButton[reorder="true"] {
    padding: 4px 10px;
    font-size: 11px;
}

/* --- inputs -------------------------------------------------------------- */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #16110a;
    border: 1px solid #2c2316;
    border-radius: 6px;
    padding: 5px 9px;
    color: #ecdfc6;
    selection-background-color: #8f661f;
    selection-color: #fff4dc;
}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #3a2f1c;
}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #e1a636;
}
QComboBox:disabled, QLineEdit:disabled {
    background-color: #100c08;
    color: #6f6347;
    border-color: #241c10;
}
QComboBox::drop-down {
    border: none;
    width: 26px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #9c8b69;
    margin-right: 6px;
}
QComboBox::down-arrow:hover {
    border-top-color: #e1a636;
}

/* combo popup */
QComboBox QAbstractItemView {
    background-color: #16110a;
    border: 1px solid #3a2f1c;
    border-radius: 6px;
    padding: 4px;
    selection-background-color: #8f661f;
    selection-color: #fff4dc;
    outline: none;
}
QComboBox QAbstractItemView::item {
    padding: 5px 8px;
    border-radius: 4px;
}
QComboBox QAbstractItemView::item:hover {
    background-color: #201810;
}

/* generic popup menu (combos, context menus) */
QMenu {
    background-color: #16110a;
    border: 1px solid #3a2f1c;
    border-radius: 6px;
    padding: 4px;
}
QMenu::item {
    padding: 5px 18px 5px 8px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #8f661f;
    color: #fff4dc;
}
QMenu::item:disabled {
    color: #6f6347;
}

/* --- checkboxes / radio -------------------------------------------------- */
QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #3a2f1c;
    border-radius: 4px;
    background: #16110a;
}
QCheckBox::indicator:hover {
    border-color: #e1a636;
}
QCheckBox::indicator:checked {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #e1a636, stop: 1 #b8861f);
    border-color: #e1a636;
    image: none;
}
QCheckBox::indicator:disabled {
    border-color: #2c2316;
    background: #100c08;
}
QRadioButton {
    spacing: 8px;
}
QRadioButton::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid #3a2f1c;
    border-radius: 8px;
    background: #16110a;
}
QRadioButton::indicator:checked {
    background: #e1a636;
    border: 4px solid #16110a;
}

/* --- sliders ------------------------------------------------------------- */
QSlider::groove:horizontal {
    height: 6px;
    background: #201810;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 15px;
    height: 15px;
    margin: -5px 0;
    border-radius: 8px;
    background: #e1a636;
    border: 2px solid #0c0a07;
}
QSlider::handle:horizontal:hover {
    background: #ffd98a;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                stop: 0 #8f661f, stop: 1 #e1a636);
    border-radius: 3px;
}
QSlider::groove:vertical {
    width: 6px;
    background: #201810;
    border-radius: 3px;
}
QSlider::handle:vertical {
    width: 15px;
    height: 15px;
    margin: 0 -5px;
    border-radius: 8px;
    background: #e1a636;
    border: 2px solid #0c0a07;
}
QSlider::sub-page:vertical {
    background: qlineargradient(x1: 0, y1: 1, x2: 0, y2: 0,
                                stop: 0 #8f661f, stop: 1 #e1a636);
    border-radius: 3px;
}

/* --- progress ------------------------------------------------------------ */
QProgressBar {
    border: 1px solid #2c2316;
    border-radius: 6px;
    background: #16110a;
    text-align: center;
    color: #ecdfc6;
    font-weight: 600;
}
QProgressBar::chunk {
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                stop: 0 #8f661f, stop: 1 #e1a636, stop: 0.85 #ffd98a);
    border-radius: 5px;
    margin-right: 2px;
}

/* --- log pane ------------------------------------------------------------ */
QPlainTextEdit {
    background-color: #0a0805;
    color: #d8c9a8;
    border: 1px solid #2c2316;
    border-radius: 6px;
    padding: 4px;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 12px;
    selection-background-color: #8f661f;
    selection-color: #fff4dc;
}
QPlainTextEdit:focus {
    border: 1px solid #e1a636;
}

/* --- group boxes --------------------------------------------------------- */
QGroupBox#sectionBox {
    border: 1px solid #2c2316;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 14px;
    background-color: #100c08;
}
QGroupBox#sectionBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #e1a636;
    font-weight: 600;
}
QGroupBox {
    border: 1px solid #2c2316;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 10px;
}

/* --- labels -------------------------------------------------------------- */
QLabel#formLabel {
    color: #9c8b69;
}
QLabel#aboutTitle {
    font-size: 30px;
    font-weight: 800;
    color: #fff4dc;
}
QLabel#aboutVersion {
    color: #9c8b69;
}
QLabel#linkLabel {
    color: #e1a636;
    text-decoration: underline;
}
QLabel#linkLabel:hover {
    color: #ffd98a;
}
QLabel#etaLabel {
    color: #9c8b69;
}
QLabel#countdownTitle {
    font-size: 16px;
    font-weight: 700;
    color: #e1a636;
}

/* --- tables -------------------------------------------------------------- */
QTableWidget {
    background-color: #100c08;
    alternate-background-color: #16110a;
    gridline-color: #2c2316;
    border: 1px solid #2c2316;
    border-radius: 6px;
    selection-background-color: #3a2f1c;
    selection-color: #fff4dc;
}
QHeaderView::section {
    background-color: #16110a;
    color: #9c8b69;
    border: none;
    border-right: 1px solid #2c2316;
    border-bottom: 1px solid #2c2316;
    padding: 7px;
    font-weight: 600;
}
QHeaderView::section:hover {
    color: #ecdfc6;
}
QTableCornerButton::section {
    background-color: #16110a;
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
    background: #3a2f1c;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #4a3d24;
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
    background: #3a2f1c;
    border-radius: 4px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background: #4a3d24;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
QScrollBar::add-page, QScrollBar::sub-page {
    background: transparent;
}

/* --- tooltips ------------------------------------------------------------ */
QToolTip {
    background-color: #16110a;
    color: #ecdfc6;
    border: 1px solid #e1a636;
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
