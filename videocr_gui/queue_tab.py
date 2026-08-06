"""Queue tab: batch job table and controls."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import i18n

STATUS_TRANSLATIONS = {
    "Pending": "status_pending",
    "Processing": "status_processing",
    "Completed": "status_completed",
    "Cancelled": "status_cancelled_queue",
    "Error": "status_error",
    "Paused": "status_paused",
}


class QueueTab(QWidget):
    """Batch queue table + controls. Emits action signals for the main window."""

    add_requested = Signal()
    add_all_requested = Signal()
    start_requested = Signal()
    stop_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    move_up_requested = Signal()
    move_down_requested = Signal()
    reset_requested = Signal()
    edit_requested = Signal()
    remove_requested = Signal()
    clear_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._jobs: list[dict[str, Any]] = []
        self._anchor: int | None = None
        self._focus: int | None = None
        self._tooltip_map: dict[QWidget, str] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([
            i18n.tr("col_video_file", "Video File"),
            i18n.tr("col_output_file", "Output File"),
            i18n.tr("col_status", "Status"),
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, 1)

        # start/stop/pause row
        row1 = QHBoxLayout()
        self.start_btn = QPushButton(i18n.tr("btn_start_queue", "Start Queue"))
        self.start_btn.setObjectName("primaryButton")
        self.stop_btn = QPushButton(i18n.tr("btn_stop_queue", "Stop Queue"))
        self.stop_btn.setObjectName("dangerButton")
        self.pause_btn = QPushButton(i18n.tr("btn_pause", "Pause"))
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        row1.addWidget(self.start_btn)
        row1.addWidget(self.stop_btn)
        row1.addWidget(self.pause_btn)
        row1.addStretch(1)
        layout.addLayout(row1)

        # reorder/edit row
        row2 = QHBoxLayout()
        self.up_btn = QPushButton("▲")
        self.up_btn.setProperty("reorder", True)
        self.down_btn = QPushButton("▼")
        self.down_btn.setProperty("reorder", True)
        self._tooltip_map[self.up_btn] = "tip_batch_up"
        self._tooltip_map[self.down_btn] = "tip_batch_down"
        self.up_btn.setToolTip(i18n.tr("tip_batch_up", "Move up"))
        self.down_btn.setToolTip(i18n.tr("tip_batch_down", "Move down"))
        self.reset_btn = QPushButton(i18n.tr("btn_reset", "Reset"))
        self.reset_btn.setToolTip(i18n.tr("tip_batch_reset", "Reset selected jobs to Pending"))
        self._tooltip_map[self.reset_btn] = "tip_batch_reset"
        self.edit_btn = QPushButton(i18n.tr("btn_edit", "Edit"))
        self.edit_btn.setToolTip(i18n.tr("tip_batch_edit", "Edit selected job settings"))
        self._tooltip_map[self.edit_btn] = "tip_batch_edit"
        self.remove_btn = QPushButton(i18n.tr("btn_remove", "Remove"))
        self.remove_btn.setToolTip(i18n.tr("tip_batch_remove", "Remove selected jobs"))
        self._tooltip_map[self.remove_btn] = "tip_batch_remove"
        self.clear_btn = QPushButton(i18n.tr("btn_clear_queue", "Clear Queue"))
        self.clear_btn.setToolTip(i18n.tr("tip_batch_clear", "Clear finished/cancelled jobs"))
        self._tooltip_map[self.clear_btn] = "tip_batch_clear"
        for b in (self.up_btn, self.down_btn, self.reset_btn, self.edit_btn, self.remove_btn, self.clear_btn):
            row2.addWidget(b)
        row2.addStretch(1)
        layout.addLayout(row2)

        self.up_btn.clicked.connect(self.move_up_requested)
        self.down_btn.clicked.connect(self.move_down_requested)
        self.reset_btn.clicked.connect(self.reset_requested)
        self.edit_btn.clicked.connect(self.edit_requested)
        self.remove_btn.clicked.connect(self.remove_requested)
        self.clear_btn.clicked.connect(self.clear_requested)
        self.start_btn.clicked.connect(self.start_requested)
        self.stop_btn.clicked.connect(self.stop_requested)
        self.pause_btn.clicked.connect(self._pause_clicked)

    # --- public API ---------------------------------------------------------
    def set_jobs(self, jobs: list[dict[str, Any]]) -> None:
        self._jobs = jobs
        self.refresh()

    def jobs(self) -> list[dict[str, Any]]:
        return self._jobs

    def refresh(self) -> None:
        self.table.setRowCount(len(self._jobs))
        for row, job in enumerate(self._jobs):
            self.table.setItem(row, 0, QTableWidgetItem(job.get("filename", "")))
            self.table.setItem(row, 1, QTableWidgetItem(job.get("output", "")))
            status = job.get("status", "Pending")
            status_item = QTableWidgetItem(self._translated_status(status))
            status_item.setData(Qt.ItemDataRole.UserRole, status)
            status_item.setForeground(QBrush(QColor(self._status_color(status))))
            self.table.setItem(row, 2, status_item)

    def selected_rows(self) -> list[int]:
        return sorted({idx.row() for idx in self.table.selectedIndexes()})

    def select_rows(self, rows: list[int]) -> None:
        self.table.clearSelection()
        for row in rows:
            self.table.selectRow(row)

    def set_pause_text(self, paused: bool) -> None:
        self.pause_btn.setText(i18n.tr("btn_resume" if paused else "btn_pause", "Resume" if paused else "Pause"))

    def set_processing_state(self, processing: bool) -> None:
        self.start_btn.setEnabled(not processing)
        self.stop_btn.setEnabled(processing)
        self.pause_btn.setEnabled(processing)

    def _translated_status(self, status: str) -> str:
        key = STATUS_TRANSLATIONS.get(status, "status_pending")
        return i18n.tr(key, status)

    @staticmethod
    def _status_color(status: str) -> str:
        """Accent color for each queue status (matches the theme palette)."""
        return {
            "Pending": "#9aa4b6",
            "Processing": "#22d3ee",
            "Paused": "#fbbf24",
            "Completed": "#34d399",
            "Cancelled": "#f87171",
            "Error": "#f87171",
        }.get(status, "#9aa4b6")

    def _pause_clicked(self) -> None:
        if self.pause_btn.text() == i18n.tr("btn_resume", "Resume"):
            self.resume_requested.emit()
        else:
            self.pause_requested.emit()

    def retranslate(self) -> None:
        self.table.setHorizontalHeaderLabels([
            i18n.tr("col_video_file", "Video File"),
            i18n.tr("col_output_file", "Output File"),
            i18n.tr("col_status", "Status"),
        ])
        self.start_btn.setText(i18n.tr("btn_start_queue", "Start Queue"))
        self.stop_btn.setText(i18n.tr("btn_stop_queue", "Stop Queue"))
        self.pause_btn.setText(i18n.tr("btn_pause", "Pause"))
        self.reset_btn.setText(i18n.tr("btn_reset", "Reset"))
        self.edit_btn.setText(i18n.tr("btn_edit", "Edit"))
        self.remove_btn.setText(i18n.tr("btn_remove", "Remove"))
        self.clear_btn.setText(i18n.tr("btn_clear_queue", "Clear Queue"))
        for widget, tip_key in self._tooltip_map.items():
            widget.setToolTip(i18n.tr(tip_key, ""))
        self.refresh()
