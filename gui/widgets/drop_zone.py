"""
DropZone — animated drag-and-drop file zone.
Shows a dashed border that glows cyan on hover.
Clicking opens a file picker.
"""
from __future__ import annotations

<<<<<<< HEAD
from pathlib import Path
from PySide6.QtCore import (
    Qt, Signal, QPropertyAnimation, QEasingCurve, Property,
=======
from PyQt5.QtCore import (
    Qt, pyqtSignal as Signal, QPropertyAnimation, QEasingCurve, pyqtProperty as Property,
>>>>>>> 95cdcbf7c9426c71a39a09d660b75d5cfba86570
    QPoint, QSize,
)
from PyQt5.QtGui import (
    QColor, QPainter, QPen, QFont, QDragEnterEvent,
    QDropEvent, QMouseEvent, QPainterPath,
)
from PyQt5.QtWidgets import QWidget, QFileDialog, QSizePolicy

from gui.io_utils import is_supported_file, VALID_EXTS


class DropZone(QWidget):
    """
    Animated drag-and-drop zone.
    Emits `files_dropped(list[str])` when files are accepted.
    """

    files_dropped = Signal(list)   # list of file paths

    def __init__(self, multi: bool = False, parent=None):
        super().__init__(parent)
        self._multi = multi
        self._hover = False
        self._glow  = 0.0          # 0.0 → dim, 1.0 → full teal glow
        self._anim  = QPropertyAnimation(self, b"glow_level", self)
        self._anim.setDuration(200)
        self._anim.setEasingCurve(QEasingCurve.InOutCubic)

        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(120)
        self.setCursor(Qt.PointingHandCursor)

    # ── Property for animation ────────────────────────────────────
    def _get_glow(self) -> float:
        return self._glow

    def _set_glow(self, v: float):
        self._glow = v
        self.update()

    glow_level = Property(float, _get_glow, _set_glow)

    def _animate_glow(self, on: bool):
        self._anim.stop()
        self._anim.setStartValue(self._glow)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    # ── Drag/drop events ─────────────────────────────────────────
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(is_supported_file(url.toLocalFile()) for url in urls):
                event.acceptProposedAction()
                self._animate_glow(True)
                return
        event.ignore()

    def dragLeaveEvent(self, _):
        self._animate_glow(False)

    def dropEvent(self, event: QDropEvent):
        self._animate_glow(False)
        paths = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if is_supported_file(url.toLocalFile())
        ]
        if paths:
            self.files_dropped.emit(paths if self._multi else paths[:1])

    # ── Click → file picker ──────────────────────────────────────
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._open_picker()

    def _open_picker(self):
        ext_filter = (
            "All Supported Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.dcm *.dicom);;"
            "All Files (*);;"
            "DICOM (*.dcm *.dicom);;"
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if self._multi:
            paths, _ = QFileDialog.getOpenFileNames(self, "Open Images", "", ext_filter)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", ext_filter)
            paths = [path] if path else []

        if paths:
            self.files_dropped.emit(paths)

    # ── Paint ────────────────────────────────────────────────────
    def enterEvent(self, _):
        self._hover = True
        self.update()

    def leaveEvent(self, _):
        self._hover = False
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        g = self._glow
        bg_alpha  = int(4  + g * 12)
        brd_alpha = int(35 + g * 165)

        # Background fill
        p.fillRect(self.rect(), QColor(0, 212, 216, bg_alpha))

        # Dashed border
        pen = QPen(QColor(0, 212, 216, brd_alpha))
        pen.setWidth(2)
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        r = self.rect().adjusted(4, 4, -4, -4)
        p.drawRoundedRect(r, 12, 12)

        # Clean non-overlapping vertical layout centered in widget
        h = self.height()
        w = self.width()
        mid_y = h // 2
        cx = w // 2

        # 1. Icon arrow ↓ (height 22px, from mid_y - 32 to mid_y - 10)
        pen2 = QPen(QColor(0, 212, 216, int(130 + g * 125)), 2.2)
        p.setPen(pen2)
        top_arrow = mid_y - 32
        bot_arrow = mid_y - 10
        p.drawLine(cx, top_arrow, cx, bot_arrow)
        # Arrow head
        p.drawLine(cx - 7, bot_arrow - 7, cx, bot_arrow)
        p.drawLine(cx + 7, bot_arrow - 7, cx, bot_arrow)

        # 2. Main Text (y from mid_y - 2 to mid_y + 18)
        label_color = QColor(0, 212, 216, int(150 + g * 105))
        p.setFont(QFont("Segoe UI", 10, QFont.DemiBold))
        p.setPen(label_color)
        p.drawText(
            0, mid_y - 2, w, 20,
            Qt.AlignHCenter | Qt.AlignVCenter,
            "Drop image here  or  click to browse",
        )

        # 3. Subtitle (y from mid_y + 20 to mid_y + 36)
        sub_color = QColor(100, 120, 150, 210)
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(sub_color)
        p.drawText(
            0, mid_y + 20, w, 18,
            Qt.AlignHCenter | Qt.AlignVCenter,
            "PNG · JPG · BMP · TIFF · DICOM",
        )
