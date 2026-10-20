"""
SplitPreview — synchronized side-by-side zoom/pan image viewer.
Left: noisy (original)  |  Right: denoised
Supports optional band artifact overlay highlights on the noisy canvas.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QPoint, QRectF
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont,
    QWheelEvent, QMouseEvent, QPainterPath,
)
from PySide6.QtWidgets import QWidget, QHBoxLayout, QSizePolicy, QFrame

from gui.classical_filter import ClassicalBand


class _ImageCanvas(QWidget):
    """Single-image canvas with zoom/pan support and band overlays."""

    def __init__(self, label: str, accent: str = "#00D4D8", parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._label = label
        self._accent = QColor(accent)
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        self._drag_start: QPoint | None = None
        self._drag_offset_start: QPoint | None = None
        self._bands: list[ClassicalBand] = []
        self._show_bands: bool = False
        self._img_shape: tuple[int, int] = (0, 0)

        self.setMinimumSize(300, 250)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.OpenHandCursor)

    def set_image(self, arr: np.ndarray | None):
        if arr is None:
            self._pixmap = None
            self._img_shape = (0, 0)
        else:
            h, w = arr.shape[:2]
            self._img_shape = (h, w)
            q = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
            self._pixmap = QPixmap.fromImage(q.copy())
        self._fit()
        self.update()

    def set_bands(self, bands: list[ClassicalBand], show: bool = True):
        self._bands = bands
        self._show_bands = show
        self.update()

    def toggle_bands(self, show: bool):
        self._show_bands = show
        self.update()

    def _fit(self):
        if self._pixmap is None:
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        cw, ch = self.width(), self.height()
        self._scale = min(cw / pw, ch / ph, 1.0)
        self._offset = QPoint(
            (cw - int(pw * self._scale)) // 2,
            (ch - int(ph * self._scale)) // 2,
        )

    def set_view(self, scale: float, offset: QPoint):
        self._scale = scale
        self._offset = QPoint(offset)
        self.update()

    def zoom(self, factor: float, anchor: QPoint):
        old_scale = self._scale
        new_scale = max(0.05, min(30.0, self._scale * factor))
        self._scale = new_scale
        ratio = new_scale / old_scale
        self._offset = anchor - (anchor - self._offset) * ratio
        self.update()

    def reset_view(self):
        self._fit()
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        if self._pixmap is not None:
            pw = int(self._pixmap.width() * self._scale)
            ph = int(self._pixmap.height() * self._scale)
            p.drawPixmap(self._offset.x(), self._offset.y(), pw, ph, self._pixmap)

            # Draw band overlays if enabled
            if self._show_bands and self._bands:
                p.save()
                for band in self._bands:
                    if not band.enabled:
                        continue
                    # Golden amber highlight for detected bands
                    overlay_color = QColor(255, 170, 0, 85)
                    pen_color = QColor(255, 200, 50, 200)

                    if band.orientation == "vertical":
                        bx = self._offset.x() + int(band.start_px * self._scale)
                        bw = max(2, int((band.end_px - band.start_px + 1) * self._scale))
                        by = self._offset.y()
                        bh = ph
                        rect = QRectF(bx, by, bw, bh)
                    else:
                        bx = self._offset.x()
                        bw = pw
                        by = self._offset.y() + int(band.start_px * self._scale)
                        bh = max(2, int((band.end_px - band.start_px + 1) * self._scale))
                        rect = QRectF(bx, by, bw, bh)

                    p.fillRect(rect, overlay_color)
                    p.setPen(QPen(pen_color, 1.0, Qt.DashLine))
                    p.drawRect(rect)
                p.restore()
        else:
            p.setPen(QColor(60, 80, 110))
            p.setFont(QFont("Segoe UI", 11))
            p.drawText(self.rect(), Qt.AlignCenter, "No image loaded")

        # Label badge
        font = QFont("Segoe UI", 9, QFont.Bold)
        p.setFont(font)
        fm = p.fontMetrics()
        text_w = fm.horizontalAdvance(self._label) + 16
        text_h = fm.height() + 8
        badge = QRectF(10, 10, text_w, text_h)
        path = QPainterPath()
        path.addRoundedRect(badge, 5, 5)
        p.fillPath(path, QColor(self._accent.red(), self._accent.green(), self._accent.blue(), 210))
        p.setPen(QColor("#0A0F1E"))
        p.drawText(badge.toRect(), Qt.AlignCenter, self._label)

        # Border
        pen = QPen(QColor(0, 212, 216, 40))
        pen.setWidth(1)
        p.setPen(pen)
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.zoom(factor, event.position().toPoint())
        if hasattr(self.parent(), "_sync_view"):
            self.parent()._sync_view(source=self)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._drag_start = event.pos()
            self._drag_offset_start = QPoint(self._offset)
            self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, _):
        self._drag_start = None
        self.setCursor(Qt.OpenHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_start is not None:
            delta = event.pos() - self._drag_start
            self._offset = self._drag_offset_start + delta
            self.update()
            if hasattr(self.parent(), "_sync_view"):
                self.parent()._sync_view(source=self)

    def resizeEvent(self, _):
        self._fit()


class SplitPreview(QFrame):
    """Two synchronized image canvases: noisy (left) | denoised (right)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._syncing = False

        self._left = _ImageCanvas("ORIGINAL", "#64748B", self)
        self._right = _ImageCanvas("RESTORED", "#00D4D8", self)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._left)
        layout.addWidget(self._right)

    def _sync_view(self, source: _ImageCanvas):
        if self._syncing:
            return
        self._syncing = True
        target = self._right if source is self._left else self._left
        target.set_view(source._scale, source._offset)
        self._syncing = False

    def set_noisy(self, arr: np.ndarray | None):
        self._left.set_image(arr)

    def set_denoised(self, arr: np.ndarray | None):
        self._right.set_image(arr)

    def set_bands(self, bands: list[ClassicalBand], show: bool = True):
        self._left.set_bands(bands, show)

    def toggle_bands(self, show: bool):
        self._left.toggle_bands(show)

    def reset_view(self):
        self._left.reset_view()
        self._right.reset_view()

    def clear(self):
        self._left.set_image(None)
        self._right.set_image(None)
        self._left.set_bands([])
