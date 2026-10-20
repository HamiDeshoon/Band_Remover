"""
BatchTable — QTableWidget displaying batch processing results.
Columns: # | Filename | Method | Status | Bands | PSNR (dB) | Time (s) | Output Path
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)


_COLS = ["#", "Filename", "Method", "Status", "Bands", "PSNR (dB)", "Time (s)", "Output Path"]

_STATUS_COLORS = {
    "pending":    ("#64748B", "#0A0F1E"),
    "processing": ("#E9B840", "#0A0F1E"),
    "done":       ("#10B981", "#0A0F1E"),
    "error":      ("#EF4444", "#0A0F1E"),
}


class BatchTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, len(_COLS), parent)
        self.setHorizontalHeaderLabels(_COLS)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        hdr = self.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Interactive)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(7, QHeaderView.Stretch)
        self.setColumnWidth(1, 200)

    def add_file(self, path: str, method: str = "Deep Learning") -> int:
        """Add a file row with 'pending' status. Returns the new row index."""
        from pathlib import Path
        row = self.rowCount()
        self.insertRow(row)
        items = [
            str(row + 1),
            Path(path).name,
            method,
            "pending",
            "—",
            "—",
            "—",
            "",
        ]
        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            if col in (1, 7):
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.setItem(row, col, item)
        self._set_status_style(row, "pending")
        return row

    def update_status(
        self,
        row: int,
        status: str,
        bands: str = "",
        psnr: str = "",
        elapsed: str = "",
        out_path: str = "",
        method: str = "",
    ):
        if method:
            self.item(row, 2).setText(method)
        self.item(row, 3).setText(status)
        if bands:
            self.item(row, 4).setText(bands)
        if psnr:
            self.item(row, 5).setText(psnr)
        if elapsed:
            self.item(row, 6).setText(elapsed)
        if out_path:
            from pathlib import Path
            self.item(row, 7).setText(str(Path(out_path).name))
            self.item(row, 7).setToolTip(out_path)
        self._set_status_style(row, status)

    def _set_status_style(self, row: int, status: str):
        fg, bg = _STATUS_COLORS.get(status, ("#94A3B8", "#0A0F1E"))
        item = self.item(row, 3)
        if item:
            item.setForeground(QColor(fg))
            item.setBackground(QColor(bg))

    def clear_all(self):
        self.setRowCount(0)
