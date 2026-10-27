"""
FileQueue — scrollable list of queued files for sequential processing.
Supports Add / Remove selected / Clear all.
"""
from __future__ import annotations
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal as Signal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
    QListWidgetItem, QPushButton, QFileDialog, QLabel,
    QSizePolicy,
)

from gui.io_utils import is_supported_file, VALID_EXTS


class FileQueue(QWidget):
    """
    Scrollable file queue.
    Emits `queue_changed(list[str])` whenever files are added or removed.
    """

    queue_changed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paths: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Counter label
        self._counter = QLabel("0 files in queue")
        self._counter.setObjectName("lbl_section")
        layout.addWidget(self._counter)

        # List
        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.ExtendedSelection)
        self._list.setSortingEnabled(False)
        layout.addWidget(self._list)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._btn_add   = QPushButton("＋  Add Files")
        self._btn_add.setFixedHeight(28)
        self._btn_add.setFixedWidth(110)
        self._btn_add.setToolTip("Add image or DICOM files to the queue")

        self._btn_rm    = QPushButton("✕  Remove Selected")
        self._btn_rm.setObjectName("btn_danger")
        self._btn_rm.setFixedHeight(28)
        self._btn_rm.setFixedWidth(140)
        self._btn_rm.setToolTip("Remove highlighted files from the queue")

        self._btn_clear = QPushButton("⊘  Clear All")
        self._btn_clear.setObjectName("btn_danger")
        self._btn_clear.setFixedHeight(28)
        self._btn_clear.setFixedWidth(90)
        self._btn_clear.setToolTip("Clear the entire queue")

        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_rm)
        btn_row.addWidget(self._btn_clear)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Connections
        self._btn_add.clicked.connect(self._add_files)
        self._btn_rm.clicked.connect(self._remove_selected)
        self._btn_clear.clicked.connect(self.clear_all)

    # ── Public API ────────────────────────────────────────────────
    def add_paths(self, paths: list[str]):
        for p in paths:
            if p not in self._paths and is_supported_file(p):
                self._paths.append(p)
                item = QListWidgetItem(Path(p).name)
                item.setToolTip(p)
                self._list.addItem(item)
        self._refresh_counter()
        self.queue_changed.emit(list(self._paths))

    def get_paths(self) -> list[str]:
        return list(self._paths)

    def clear_all(self):
        self._paths.clear()
        self._list.clear()
        self._refresh_counter()
        self.queue_changed.emit([])

    def remove_path(self, path: str):
        if path in self._paths:
            idx = self._paths.index(path)
            self._paths.pop(idx)
            self._list.takeItem(idx)
        self._refresh_counter()
        self.queue_changed.emit(list(self._paths))

    # ── Internal ─────────────────────────────────────────────────
    def _add_files(self):
        ext_filter = (
            "All Supported Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.dcm *.dicom);;"
            "All Files (*);;"
            "DICOM (*.dcm *.dicom);;"
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        paths, _ = QFileDialog.getOpenFileNames(self, "Add Files to Queue", "", ext_filter)
        if paths:
            self.add_paths(paths)

    def _remove_selected(self):
        rows = sorted({idx.row() for idx in self._list.selectedIndexes()}, reverse=True)
        for r in rows:
            self._paths.pop(r)
            self._list.takeItem(r)
        self._refresh_counter()
        self.queue_changed.emit(list(self._paths))

    def _refresh_counter(self):
        n = len(self._paths)
        self._counter.setText(f"{n} file{'s' if n != 1 else ''} in queue")
