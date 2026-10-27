"""
DeStripe GUI — Main Application Window
PyQt5 · Dark Teal Medical Imaging Theme
======================================================
Features:
  1. Deep Learning SMP U-Net / EfficientNet-b0 (`destripe_best_eff.ckpt`)
  2. Non-Deep Learning / Classical Profile Band Filter (from Dicom_Band_Fixer:
     Median baseline profiling, statistical deviation, PCHIP/Linear interpolation,
     Gaussian smoothing)
  3. Hybrid Two-Stage Mode (Classical Interpolation + Neural Polish)
  4. Real-time band detection overlay, PSNR & timing metrics
  5. Single Image, Batch Folder, and File Queue tabs
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional, List

if sys.platform == "win32":
    import ctypes
    _sys_icu = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "icuuc.dll")
    if os.path.exists(_sys_icu):
        try:
            ctypes.windll.kernel32.LoadLibraryW(_sys_icu)
        except Exception:
            pass

# ── PyQt5 ─────────────────────────────────────────────────────────────────
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal as Signal, QTimer, QSize, QPropertyAnimation,
    QEasingCurve, QPoint,
)
from PyQt5.QtGui import (
    QColor, QPainter, QFont, QIcon, QPixmap,
    QLinearGradient, QGradient,
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QProgressBar,
    QFileDialog, QStatusBar, QFrame, QSplitter,
    QGroupBox, QSizePolicy, QSpacerItem, QScrollArea,
    QMessageBox, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
)

try:
    from PyQt5.QtWidgets import QAction
except ImportError:
    try:
        from PySide6.QtWidgets import QAction
    except ImportError:
        from PySide6.QtGui import QAction

# ── Local modules ────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_HERE, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gui.styles.theme import get_qss
from gui.config import load as load_cfg, save as save_cfg
from gui.io_utils import read_image, save_image, is_supported_file, is_dicom_file, VALID_EXTS
from gui.inference_engine import (
    load_model, select_device, device_label,
    InferenceWorker,
)
from gui.classical_filter import (
    ClassicalSettings, ClassicalBand,
    detect_artifact_bands, correct_bands,
)
from gui.widgets.split_preview import SplitPreview
from gui.widgets.drop_zone import DropZone
from gui.widgets.batch_table import BatchTable
from gui.widgets.file_queue import FileQueue

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# UI Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _card(parent=None) -> QFrame:
    f = QFrame(parent)
    f.setObjectName("card")
    return f


def _section_label(text: str, parent=None) -> QLabel:
    lbl = QLabel(text, parent)
    lbl.setObjectName("lbl_section")
    return lbl


def _metric_label(text: str, parent=None) -> QLabel:
    lbl = QLabel(text, parent)
    lbl.setObjectName("lbl_metric")
    return lbl


# ─────────────────────────────────────────────────────────────────────────────
# Settings Tab
# ─────────────────────────────────────────────────────────────────────────────

class SettingsTab(QWidget):
    model_loaded = Signal(object, object)   # (model, device)
    settings_changed = Signal(object)      # (ClassicalSettings)
    status_message = Signal(str)

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._model: Optional[nn.Module] = None
        self._device: Optional[torch.device] = None
        self._classical_settings = ClassicalSettings()
        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(28, 28, 28, 28)
        outer.setSpacing(20)

        # Title
        title = QLabel("Settings & Algorithm Tuning")
        title.setObjectName("lbl_title")
        outer.addWidget(title)

        # ── Device card ───────────────────────────────────────────
        dev_card = _card()
        dev_lay = QVBoxLayout(dev_card)
        dev_lay.setContentsMargins(20, 16, 20, 16)
        dev_lay.setSpacing(10)
        dev_lay.addWidget(_section_label("COMPUTE DEVICE"))

        row = QHBoxLayout()
        row.setSpacing(10)
        dev_icon = QLabel("⚡")
        dev_icon.setFont(QFont("Segoe UI Emoji", 18))
        row.addWidget(dev_icon)

        self._device_label = QLabel(device_label())
        self._device_label.setFont(QFont("Segoe UI", 13, QFont.DemiBold))
        row.addWidget(self._device_label)
        row.addStretch()

        badge_id = "lbl_badge_gpu" if torch.cuda.is_available() else "lbl_badge_cpu"
        self._badge = QLabel("GPU" if torch.cuda.is_available() else "CPU")
        self._badge.setObjectName(badge_id)
        row.addWidget(self._badge)
        dev_lay.addLayout(row)
        outer.addWidget(dev_card)

        # ── Checkpoint card ───────────────────────────────────────
        ckpt_card = _card()
        ckpt_lay = QVBoxLayout(ckpt_card)
        ckpt_lay.setContentsMargins(20, 16, 20, 16)
        ckpt_lay.setSpacing(12)
        ckpt_lay.addWidget(_section_label("DEEP LEARNING MODEL CHECKPOINT (SMP / DESTRIPE)"))

        ckpt_row = QHBoxLayout()
        ckpt_row.setSpacing(10)
        self._ckpt_edit = QLineEdit()
        self._ckpt_edit.setPlaceholderText("Path to destripe_best_eff.ckpt or .pt file …")
        
        # Default to local destripe_best_eff.ckpt if existing
        local_ckpt = _ROOT / "destripe_best_eff.ckpt"
        cfg_ckpt = self._cfg.get("model_path", "")
        if not cfg_ckpt and local_ckpt.exists():
            cfg_ckpt = str(local_ckpt)
        self._ckpt_edit.setText(cfg_ckpt)
        ckpt_row.addWidget(self._ckpt_edit)

        browse_btn = QPushButton("Browse …")
        browse_btn.setFixedWidth(100)
        browse_btn.clicked.connect(self._browse_checkpoint)
        ckpt_row.addWidget(browse_btn)
        ckpt_lay.addLayout(ckpt_row)

        load_row = QHBoxLayout()
        self._load_btn = QPushButton("Load Checkpoint")
        self._load_btn.setObjectName("btn_primary")
        self._load_btn.setFixedHeight(30)
        self._load_btn.setFixedWidth(140)
        self._load_btn.clicked.connect(self._load_checkpoint)
        load_row.addWidget(self._load_btn)
        load_row.addStretch()
        ckpt_lay.addLayout(load_row)

        self._ckpt_status = QLabel("")
        self._ckpt_status.setWordWrap(True)
        ckpt_lay.addWidget(self._ckpt_status)
        outer.addWidget(ckpt_card)

        # ── Non-Deep Learning / Classical Filter Tuning Card ───────
        c_card = _card()
        c_lay = QVBoxLayout(c_card)
        c_lay.setContentsMargins(20, 16, 20, 16)
        c_lay.setSpacing(12)
        c_lay.addWidget(_section_label("CLASSICAL BAND FILTER SETTINGS (DICOM_BAND_FIXER)"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)

        # Interpolation method
        grid.addWidget(QLabel("Interpolation Method:"), 0, 0)
        self._interp_combo = QComboBox()
        self._interp_combo.addItems([
            "Edge-Preserving + Texture (Clinical)",
            "Linear Context",
            "PCHIP Spline",
        ])
        self._interp_combo.currentIndexChanged.connect(self._update_classical_params)
        grid.addWidget(self._interp_combo, 0, 1)

        # Artifact Orientation
        grid.addWidget(QLabel("Artifact Orientation:"), 1, 0)
        self._orient_combo = QComboBox()
        self._orient_combo.addItems([
            "Auto (Detect Dominance)",
            "Horizontal Lines Only (Rows)",
            "Vertical Lines Only (Columns)",
            "Both (Cross-Directional)",
        ])
        self._orient_combo.currentIndexChanged.connect(self._update_classical_params)
        grid.addWidget(self._orient_combo, 1, 1)

        # Sensitivity
        grid.addWidget(QLabel("Detection Sensitivity:"), 2, 0)
        self._sens_spin = QDoubleSpinBox()
        self._sens_spin.setRange(0.5, 10.0)
        self._sens_spin.setSingleStep(0.5)
        self._sens_spin.setValue(2.5)
        self._sens_spin.valueChanged.connect(self._update_classical_params)
        grid.addWidget(self._sens_spin, 2, 1)

        # Max Band Width
        grid.addWidget(QLabel("Estimated Band Width (px):"), 3, 0)
        self._width_spin = QSpinBox()
        self._width_spin.setRange(1, 50)
        self._width_spin.setValue(5)
        self._width_spin.valueChanged.connect(self._update_classical_params)
        grid.addWidget(self._width_spin, 3, 1)

        # Neighbor Context
        grid.addWidget(QLabel("Neighbor Context (px):"), 0, 2)
        self._ctx_spin = QSpinBox()
        self._ctx_spin.setRange(5, 100)
        self._ctx_spin.setValue(15)
        self._ctx_spin.valueChanged.connect(self._update_classical_params)
        grid.addWidget(self._ctx_spin, 0, 3)

        # Min Line Coverage (%)
        grid.addWidget(QLabel("Min Line Coverage (%):"), 1, 2)
        self._coverage_spin = QSpinBox()
        self._coverage_spin.setRange(10, 95)
        self._coverage_spin.setSingleStep(5)
        self._coverage_spin.setValue(35)
        self._coverage_spin.setSuffix("%")
        self._coverage_spin.valueChanged.connect(self._update_classical_params)
        grid.addWidget(self._coverage_spin, 1, 3)

        # Gaussian Smoothing Sigma
        grid.addWidget(QLabel("Gaussian Smoothing (σ):"), 2, 2)
        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setRange(0.0, 5.0)
        self._sigma_spin.setSingleStep(0.2)
        self._sigma_spin.setValue(0.0)
        self._sigma_spin.valueChanged.connect(self._update_classical_params)
        grid.addWidget(self._sigma_spin, 2, 3)

        # Border Exclude Ratio
        grid.addWidget(QLabel("Border Exclude Ratio:"), 3, 2)
        self._border_spin = QDoubleSpinBox()
        self._border_spin.setRange(0.0, 0.2)
        self._border_spin.setSingleStep(0.01)
        self._border_spin.setValue(0.03)
        self._border_spin.valueChanged.connect(self._update_classical_params)
        grid.addWidget(self._border_spin, 3, 3)

        c_lay.addLayout(grid)
        outer.addWidget(c_card)

        # ── Save config ───────────────────────────────────────────
        save_row = QHBoxLayout()
        save_btn = QPushButton("💾  Save Configuration")
        save_btn.setObjectName("btn_primary")
        save_btn.setFixedHeight(30)
        save_btn.setFixedWidth(160)
        save_btn.clicked.connect(self._save_config)
        save_row.addWidget(save_btn)
        save_row.addStretch()
        outer.addLayout(save_row)

        outer.addStretch()

        scroll_layout = QVBoxLayout(self)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(container)
        scroll_layout.addWidget(scroll)

        # Auto-load if path is saved
        if self._ckpt_edit.text():
            QTimer.singleShot(400, self._load_checkpoint)

    def _update_classical_params(self):
        interp_idx = self._interp_combo.currentIndex()
        method_map = {0: "smoothstep_texture", 1: "linear", 2: "pchip"}
        method_str = method_map.get(interp_idx, "smoothstep_texture")

        orient_idx = self._orient_combo.currentIndex()
        orient_map = {0: "auto", 1: "horizontal", 2: "vertical", 3: "both"}
        orient_str = orient_map.get(orient_idx, "auto")

        self._classical_settings = ClassicalSettings(
            band_width_estimate=self._width_spin.value(),
            neighbor_pixels=self._ctx_spin.value(),
            detection_sensitivity=self._sens_spin.value(),
            border_exclude_ratio=self._border_spin.value(),
            orientation=orient_str,
            interpolation_method=method_str,
            gaussian_sigma=self._sigma_spin.value(),
            min_coverage_ratio=float(self._coverage_spin.value()) / 100.0,
            texture_matching=True,
        )
        self.settings_changed.emit(self._classical_settings)

    def _browse_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Checkpoint", "",
            "Checkpoints (*.ckpt *.pt);;All Files (*)",
        )
        if path:
            self._ckpt_edit.setText(path)

    def _load_checkpoint(self):
        path = self._ckpt_edit.text().strip()
        if not path or not Path(path).exists():
            self._ckpt_status.setObjectName("lbl_status_err")
            self._ckpt_status.setText("✗  File not found")
            self._ckpt_status.style().unpolish(self._ckpt_status)
            self._ckpt_status.style().polish(self._ckpt_status)
            return

        self._load_btn.setEnabled(False)
        self._ckpt_status.setObjectName("")
        self._ckpt_status.setText("Loading …")

        try:
            device = select_device()
            model = load_model(path, device)
            self._model = model
            self._device = device
            self._cfg["model_path"] = path
            save_cfg(self._cfg)

            arch_type = "SMP UNet (EfficientNet-b0)" if hasattr(model, "unet") else "DestripePlus"
            self._ckpt_status.setObjectName("lbl_status_ok")
            self._ckpt_status.setText(f"✓  Loaded [{arch_type}]  ·  {Path(path).name}")
            self.model_loaded.emit(model, device)
            self.status_message.emit(f"Model loaded: {Path(path).name} ({arch_type})  ·  {device_label()}")
        except Exception as exc:
            self._ckpt_status.setObjectName("lbl_status_err")
            self._ckpt_status.setText(f"✗  {exc}")
        finally:
            self._ckpt_status.style().unpolish(self._ckpt_status)
            self._ckpt_status.style().polish(self._ckpt_status)
            self._load_btn.setEnabled(True)

    def _save_config(self):
        self._cfg["model_path"] = self._ckpt_edit.text().strip()
        save_cfg(self._cfg)
        self.status_message.emit("Configuration saved.")

    @property
    def model(self): return self._model

    @property
    def device(self): return self._device

    @property
    def classical_settings(self): return self._classical_settings


# ─────────────────────────────────────────────────────────────────────────────
# Single Image Tab
# ─────────────────────────────────────────────────────────────────────────────

class SingleImageTab(QWidget):
    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model: Optional[nn.Module] = None
        self._device: Optional[torch.device] = None
        self._noisy: Optional[np.ndarray] = None
        self._denoised: Optional[np.ndarray] = None
        self._meta: dict = {}
        self._src_path: str = ""
        self._worker: Optional[InferenceWorker] = None
        self._classical_settings = ClassicalSettings()
        self._detected_bands: List[ClassicalBand] = []
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 28, 28, 28)
        outer.setSpacing(14)

        # Title + Mode Selector Row
        top_bar = QHBoxLayout()
        title = QLabel("Single Image Denoising")
        title.setObjectName("lbl_title")
        top_bar.addWidget(title)
        top_bar.addStretch()

        # Method selector
        top_bar.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        self._method_combo.addItems([
            "Classical Filter (Non-DL · Fast)",
            "Deep Learning (SMP U-Net)",
            "Hybrid (Classical + Neural)",
        ])
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        top_bar.addWidget(self._method_combo)

        # Overlay checkbox
        self._overlay_cb = QCheckBox("Show Detected Bands Overlay")
        self._overlay_cb.setChecked(True)
        self._overlay_cb.toggled.connect(self._toggle_overlay)
        top_bar.addWidget(self._overlay_cb)

        outer.addLayout(top_bar)

        # Drop zone
        self._drop = DropZone(multi=False)
        self._drop.setMaximumHeight(115)
        self._drop.files_dropped.connect(self._on_files_dropped)
        outer.addWidget(self._drop)

        # File label & detected bands badge
        info_row = QHBoxLayout()
        self._file_lbl = QLabel("No file loaded")
        self._file_lbl.setObjectName("lbl_section")
        info_row.addWidget(self._file_lbl)
        info_row.addStretch()
        self._band_badge = QLabel("Bands: —")
        self._band_badge.setObjectName("lbl_badge_gpu")
        info_row.addWidget(self._band_badge)
        outer.addLayout(info_row)

        # Split Preview Canvas
        self._preview = SplitPreview()
        outer.addWidget(self._preview, stretch=1)

        # Metrics row
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(14)
        self._psnr_lbl = _metric_label("PSNR: —")
        self._time_lbl = _metric_label("Time: —")
        metrics_row.addWidget(self._psnr_lbl)
        metrics_row.addWidget(self._time_lbl)
        metrics_row.addStretch()
        outer.addLayout(metrics_row)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setFixedHeight(6)
        outer.addWidget(self._progress)

        # Action Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch()

        self._reset_btn = QPushButton("↺  Reset View")
        self._reset_btn.setFixedHeight(30)
        self._reset_btn.setFixedWidth(110)
        self._reset_btn.clicked.connect(self._preview.reset_view)

        self._save_btn = QPushButton("💾  Save Denoised")
        self._save_btn.setFixedHeight(30)
        self._save_btn.setFixedWidth(135)
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_result)

        self._denoise_btn = QPushButton("⚡  Denoise")
        self._denoise_btn.setObjectName("btn_primary")
        self._denoise_btn.setFixedHeight(30)
        self._denoise_btn.setFixedWidth(120)
        self._denoise_btn.setEnabled(False)
        self._denoise_btn.clicked.connect(self._run_inference)

        btn_row.addWidget(self._reset_btn)
        btn_row.addWidget(self._save_btn)
        btn_row.addWidget(self._denoise_btn)
        outer.addLayout(btn_row)

    def set_model(self, model: nn.Module, device: torch.device):
        self._model = model
        self._device = device
        self._refresh_buttons()

    def set_classical_settings(self, settings: ClassicalSettings):
        self._classical_settings = settings
        if self._noisy is not None:
            self._run_fast_band_detection()

    def _on_method_changed(self, idx: int):
        self._refresh_buttons()

    def _toggle_overlay(self, checked: bool):
        self._preview.toggle_bands(checked)

    def _on_files_dropped(self, paths: list[str]):
        self._load_image(paths[0])

    def _load_image(self, path: str):
        try:
            arr, meta = read_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return
        self._noisy = arr
        self._meta = meta
        self._src_path = path
        self._denoised = None
        self._preview.set_noisy(arr)
        self._preview.set_denoised(None)
        self._file_lbl.setText(Path(path).name)
        self._psnr_lbl.setText("PSNR: —")
        self._time_lbl.setText("Time: —")
        self._save_btn.setEnabled(False)
        self._run_fast_band_detection()
        self._refresh_buttons()
        self.status_message.emit(f"Loaded: {Path(path).name} ({arr.shape[1]}×{arr.shape[0]})")

    def _run_fast_band_detection(self):
        if self._noisy is None:
            return
        self._detected_bands = detect_artifact_bands(self._noisy, self._classical_settings)
        count = len(self._detected_bands)
        self._band_badge.setText(f"Detected: {count} band{'s' if count != 1 else ''}")
        self._preview.set_bands(self._detected_bands, self._overlay_cb.isChecked())

    def _refresh_buttons(self):
        method_idx = self._method_combo.currentIndex()
        # Classical mode does NOT require a loaded DL model
        if method_idx == 0:
            ready = self._noisy is not None
        else:
            ready = self._noisy is not None and self._model is not None
        self._denoise_btn.setEnabled(ready)

    def _run_inference(self):
        if self._noisy is None:
            return
        method_idx = self._method_combo.currentIndex()
        method_key = "classical" if method_idx == 0 else ("dl" if method_idx == 1 else "both")

        self._denoise_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)

        self._worker = InferenceWorker(
            self._noisy,
            method=method_key,
            model=self._model,
            device=self._device,
            classical_settings=self._classical_settings,
        )
        self._worker.progress.connect(self._progress.setValue)
        self._worker.result.connect(self._on_result)
        self._worker.finished_with_stats.connect(self._on_stats)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_result(self, arr: np.ndarray):
        self._denoised = arr
        self._preview.set_denoised(arr)
        self._save_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._refresh_buttons()

    def _on_stats(self, elapsed: float, p: float, band_count: int):
        psnr_str = f"{p:.2f} dB" if p < 1000 else "∞"
        self._psnr_lbl.setText(f"PSNR: {psnr_str}")
        self._time_lbl.setText(f"Time: {elapsed:.2f}s")
        if band_count > 0:
            self._band_badge.setText(f"Removed: {band_count} bands")
        self.status_message.emit(f"Done — PSNR {psnr_str} · {elapsed:.2f}s")

    def _on_error(self, msg: str):
        self._progress.setVisible(False)
        self._refresh_buttons()
        QMessageBox.critical(self, "Processing Error", msg)

    def _save_result(self):
        if self._denoised is None:
            return
        try:
            out = save_image(self._denoised, self._src_path, self._meta, bands=self._detected_bands, settings=self._classical_settings)
            self.status_message.emit(f"Saved: {Path(out).name}")
            QMessageBox.information(self, "Saved", f"Denoised image saved to:\n{out}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Batch Folder Tab
# ─────────────────────────────────────────────────────────────────────────────

class BatchFolderTab(QWidget):
    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model: Optional[nn.Module] = None
        self._device: Optional[torch.device] = None
        self._folder: str = ""
        self._files: list[str] = []
        self._running = False
        self._current_row = 0
        self._worker: Optional[InferenceWorker] = None
        self._metas: list[dict] = []
        self._noisies: list[Optional[np.ndarray]] = []
        self._classical_settings = ClassicalSettings()
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 28, 28, 28)
        outer.setSpacing(16)

        # Header bar
        header_row = QHBoxLayout()
        title = QLabel("Batch Folder Processing")
        title.setObjectName("lbl_title")
        header_row.addWidget(title)
        header_row.addStretch()

        header_row.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        self._method_combo.addItems([
            "Classical Filter (Non-DL · Fast)",
            "Deep Learning (SMP U-Net)",
            "Hybrid (Classical + Neural)",
        ])
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        header_row.addWidget(self._method_combo)
        outer.addLayout(header_row)

        # Folder picker row
        folder_card = _card()
        fl = QHBoxLayout(folder_card)
        fl.setContentsMargins(16, 12, 16, 12)
        fl.setSpacing(10)

        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Select a folder containing images / DICOM files …")
        self._folder_edit.setReadOnly(True)
        fl.addWidget(self._folder_edit)

        browse_btn = QPushButton("Browse Folder")
        browse_btn.setFixedWidth(130)
        browse_btn.clicked.connect(self._browse_folder)
        fl.addWidget(browse_btn)
        outer.addWidget(folder_card)

        # Stats row
        stats_row = QHBoxLayout()
        self._count_lbl = _metric_label("0 files found")
        stats_row.addWidget(self._count_lbl)
        stats_row.addStretch()
        outer.addLayout(stats_row)

        # Table
        self._table = BatchTable()
        outer.addWidget(self._table, stretch=1)

        # Global progress
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setFixedHeight(6)
        outer.addWidget(self._progress)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch()

        self._clear_btn = QPushButton("⊘  Clear")
        self._clear_btn.setObjectName("btn_danger")
        self._clear_btn.setFixedHeight(30)
        self._clear_btn.setFixedWidth(90)
        self._clear_btn.clicked.connect(self._clear_table)

        self._run_btn = QPushButton("⚡  Process All")
        self._run_btn.setObjectName("btn_primary")
        self._run_btn.setFixedHeight(30)
        self._run_btn.setFixedWidth(130)
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run_all)

        btn_row.addWidget(self._clear_btn)
        btn_row.addWidget(self._run_btn)
        outer.addLayout(btn_row)

    def set_model(self, model: nn.Module, device: torch.device):
        self._model = model
        self._device = device
        self._refresh_run_btn()

    def set_classical_settings(self, settings: ClassicalSettings):
        self._classical_settings = settings

    def _on_method_changed(self, idx: int):
        self._refresh_run_btn()

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self._folder = folder
            self._folder_edit.setText(folder)
            self._scan_folder()

    def _scan_folder(self):
        self._files = [
            str(p) for p in Path(self._folder).rglob("*")
            if p.is_file() and is_supported_file(p)
        ]
        self._table.clear_all()
        self._metas.clear()
        self._noisies.clear()
        method_name = self._method_combo.currentText().split("(")[0].strip()
        for f in self._files:
            self._table.add_file(f, method=method_name)
            self._metas.append({})
            self._noisies.append(None)
        self._count_lbl.setText(f"{len(self._files)} file{'s' if len(self._files) != 1 else ''} found")
        self._refresh_run_btn()

    def _refresh_run_btn(self):
        method_idx = self._method_combo.currentIndex()
        has_files = bool(self._files) and not self._running
        if method_idx == 0:
            self._run_btn.setEnabled(has_files)
        else:
            self._run_btn.setEnabled(has_files and self._model is not None)

    def _run_all(self):
        if not self._files:
            return
        self._running = True
        self._current_row = 0
        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._process_next()

    def _process_next(self):
        if self._current_row >= len(self._files):
            self._running = False
            self._progress.setVisible(False)
            self._refresh_run_btn()
            self.status_message.emit("Batch complete.")
            return

        row = self._current_row
        path = self._files[row]
        self._table.update_status(row, "processing")

        try:
            arr, meta = read_image(path)
            self._noisies[row] = arr
            self._metas[row] = meta
        except Exception as exc:
            self._table.update_status(row, "error")
            self.status_message.emit(f"Load error: {Path(path).name} — {exc}")
            self._current_row += 1
            self._update_global_progress()
            self._process_next()
            return

        method_idx = self._method_combo.currentIndex()
        method_key = "classical" if method_idx == 0 else ("dl" if method_idx == 1 else "both")

        self._worker = InferenceWorker(
            arr,
            method=method_key,
            model=self._model,
            device=self._device,
            classical_settings=self._classical_settings,
        )
        self._worker.result.connect(lambda out, r=row, p=path, m=meta: self._on_batch_result(r, p, m, out))
        self._worker.finished_with_stats.connect(lambda e, psnr_v, bc, r=row: self._on_batch_stats(r, e, psnr_v, bc))
        self._worker.error.connect(lambda msg, r=row: self._on_batch_error(r, msg))
        self._worker.start()

    def _on_batch_result(self, row: int, path: str, meta: dict, out: np.ndarray):
        try:
            saved = save_image(out, path, meta, settings=self._classical_settings)
            self._table.update_status(row, "done", out_path=saved)
        except Exception as exc:
            self._table.update_status(row, "error")
            self.status_message.emit(f"Save error: {exc}")

    def _on_batch_stats(self, row: int, elapsed: float, p: float, band_count: int):
        psnr_str = f"{p:.2f}" if p < 1000 else "∞"
        bands_str = str(band_count) if band_count > 0 else "0"
        self._table.update_status(row, "done", bands=bands_str, psnr=psnr_str, elapsed=f"{elapsed:.2f}")
        self._current_row += 1
        self._update_global_progress()
        self.status_message.emit(
            f"[{self._current_row}/{len(self._files)}] {Path(self._files[row]).name}"
        )
        self._process_next()

    def _on_batch_error(self, row: int, msg: str):
        self._table.update_status(row, "error")
        self._current_row += 1
        self._update_global_progress()
        self.status_message.emit(f"Error on {Path(self._files[row]).name}: {msg}")
        self._process_next()

    def _update_global_progress(self):
        if self._files:
            self._progress.setValue(int(self._current_row * 100 / len(self._files)))

    def _clear_table(self):
        if self._running:
            return
        self._table.clear_all()
        self._files.clear()
        self._metas.clear()
        self._noisies.clear()
        self._count_lbl.setText("0 files found")
        self._refresh_run_btn()


# ─────────────────────────────────────────────────────────────────────────────
# Queue Tab
# ─────────────────────────────────────────────────────────────────────────────

class QueueTab(QWidget):
    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model: Optional[nn.Module] = None
        self._device: Optional[torch.device] = None
        self._running = False
        self._queue: list[str] = []
        self._current = 0
        self._worker: Optional[InferenceWorker] = None
        self._classical_settings = ClassicalSettings()
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 28, 28, 28)
        outer.setSpacing(16)

        # Header bar
        header_row = QHBoxLayout()
        title = QLabel("File Queue")
        title.setObjectName("lbl_title")
        header_row.addWidget(title)
        header_row.addStretch()

        header_row.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        self._method_combo.addItems([
            "Classical Filter (Non-DL · Fast)",
            "Deep Learning (SMP U-Net)",
            "Hybrid (Classical + Neural)",
        ])
        self._method_combo.currentIndexChanged.connect(self._refresh_btn)
        header_row.addWidget(self._method_combo)
        outer.addLayout(header_row)

        # Drop zone
        self._drop = DropZone(multi=True)
        self._drop.setMaximumHeight(110)
        self._drop.files_dropped.connect(self._on_drop)
        outer.addWidget(self._drop)

        # Queue list
        self._queue_widget = FileQueue()
        self._queue_widget.queue_changed.connect(self._on_queue_changed)
        outer.addWidget(self._queue_widget, stretch=1)

        # Results table
        results_lbl = _section_label("RESULTS")
        outer.addWidget(results_lbl)
        self._table = BatchTable()
        self._table.setMaximumHeight(220)
        outer.addWidget(self._table)

        # Progress
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setFixedHeight(6)
        outer.addWidget(self._progress)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch()

        self._run_btn = QPushButton("⚡  Process Queue")
        self._run_btn.setObjectName("btn_primary")
        self._run_btn.setFixedHeight(30)
        self._run_btn.setFixedWidth(140)
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run_queue)

        btn_row.addWidget(self._run_btn)
        outer.addLayout(btn_row)

    def set_model(self, model: nn.Module, device: torch.device):
        self._model = model
        self._device = device
        self._refresh_btn()

    def set_classical_settings(self, settings: ClassicalSettings):
        self._classical_settings = settings

    def _on_drop(self, paths: list[str]):
        self._queue_widget.add_paths(paths)

    def _on_queue_changed(self, paths: list[str]):
        self._queue = paths
        self._refresh_btn()

    def _refresh_btn(self):
        method_idx = self._method_combo.currentIndex()
        has_items = bool(self._queue) and not self._running
        if method_idx == 0:
            self._run_btn.setEnabled(has_items)
        else:
            self._run_btn.setEnabled(has_items and self._model is not None)

    def _run_queue(self):
        if not self._queue:
            return
        self._running = True
        self._current = 0
        self._run_btn.setEnabled(False)
        self._table.clear_all()
        method_name = self._method_combo.currentText().split("(")[0].strip()
        for p in self._queue:
            self._table.add_file(p, method=method_name)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._process_next()

    def _process_next(self):
        if self._current >= len(self._queue):
            self._running = False
            self._progress.setVisible(False)
            self._refresh_btn()
            self.status_message.emit("Queue complete.")
            return

        row = self._current
        path = self._queue[row]
        self._table.update_status(row, "processing")

        try:
            arr, meta = read_image(path)
        except Exception as exc:
            self._table.update_status(row, "error")
            self._current += 1
            self._progress.setValue(int(self._current * 100 / len(self._queue)))
            self._process_next()
            return

        method_idx = self._method_combo.currentIndex()
        method_key = "classical" if method_idx == 0 else ("dl" if method_idx == 1 else "both")

        self._worker = InferenceWorker(
            arr,
            method=method_key,
            model=self._model,
            device=self._device,
            classical_settings=self._classical_settings,
        )
        self._worker.result.connect(lambda out, r=row, p=path, m=meta: self._on_result(r, p, m, out))
        self._worker.finished_with_stats.connect(lambda e, pv, bc, r=row: self._on_stats(r, e, pv, bc))
        self._worker.error.connect(lambda msg, r=row: self._on_error(r, msg))
        self._worker.start()

    def _on_result(self, row: int, path: str, meta: dict, out: np.ndarray):
        try:
            saved = save_image(out, path, meta, settings=self._classical_settings)
            self._table.update_status(row, "done", out_path=saved)
        except Exception as exc:
            self._table.update_status(row, "error")

    def _on_stats(self, row: int, elapsed: float, p: float, band_count: int):
        psnr_str = f"{p:.2f}" if p < 1000 else "∞"
        bands_str = str(band_count) if band_count > 0 else "0"
        self._table.update_status(row, "done", bands=bands_str, psnr=psnr_str, elapsed=f"{elapsed:.2f}")
        self._current += 1
        self._progress.setValue(int(self._current * 100 / len(self._queue)))
        self.status_message.emit(
            f"[{self._current}/{len(self._queue)}] {Path(self._queue[row]).name}"
        )
        self._process_next()

    def _on_error(self, row: int, msg: str):
        self._table.update_status(row, "error")
        self._current += 1
        self._progress.setValue(int(self._current * 100 / len(self._queue)))
        self._process_next()


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, cfg: dict):
        super().__init__()
        self._cfg = cfg
        self.setWindowTitle("DeStripe — Band Artifact Remover")
        self.setMinimumSize(1100, 780)
        self.resize(1300, 840)

        icon_path = _HERE / "assets" / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(16, 12, 16, 12)
        root_layout.setSpacing(0)

        header = self._build_header()
        root_layout.addWidget(header)
        root_layout.addSpacing(10)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        root_layout.addWidget(self._tabs, stretch=1)

        self._single = SingleImageTab()
        self._batch = BatchFolderTab()
        self._queue = QueueTab()
        self._settings = SettingsTab(cfg)

        self._tabs.addTab(self._single, "  Single Image  ")
        self._tabs.addTab(self._batch, "  Batch Folder  ")
        self._tabs.addTab(self._queue, "  File Queue    ")
        self._tabs.addTab(self._settings, "  ⚙ Settings & Tuning   ")

        # Status bar
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status_lbl = QLabel("Ready · Classical filter active · DL checkpoint ready")
        self._status_lbl.setStyleSheet("color: #4A6080; font-size: 12px;")
        sb.addWidget(self._status_lbl)

        self._device_badge = QLabel(
            "  GPU  " if torch.cuda.is_available() else "  CPU  "
        )
        badge_id = "lbl_badge_gpu" if torch.cuda.is_available() else "lbl_badge_cpu"
        self._device_badge.setObjectName(badge_id)
        sb.addPermanentWidget(self._device_badge)

        # Signals
        self._settings.model_loaded.connect(self._on_model_loaded)
        self._settings.settings_changed.connect(self._on_settings_changed)
        self._settings.status_message.connect(self._show_status)
        self._single.status_message.connect(self._show_status)
        self._batch.status_message.connect(self._show_status)
        self._queue.status_message.connect(self._show_status)

    def _build_header(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(56)
        w.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 rgba(0,180,184,0.08), stop:1 rgba(14,165,233,0.04));"
            "border-radius: 10px;"
            "border: 1px solid rgba(0,212,216,0.12);"
        )
        row = QHBoxLayout(w)
        row.setContentsMargins(20, 0, 20, 0)
        row.setSpacing(14)

        logo = QLabel("◈")
        logo.setFont(QFont("Segoe UI", 20))
        logo.setStyleSheet("color: #00D4D8;")
        row.addWidget(logo)

        title = QLabel("DeStripe")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setStyleSheet(
            "background: transparent;"
            "color: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #00D4D8, stop:1 #0EA5E9);"
        )
        row.addWidget(title)

        sub = QLabel("Band Artifact Remover  ·  Classical + SMP U-Net  ·  v2.0")
        sub.setStyleSheet("color: #2D4A6A; font-size: 12px; background: transparent;")
        row.addWidget(sub)
        row.addStretch()
        return w

    def _on_model_loaded(self, model: nn.Module, device: torch.device):
        self._single.set_model(model, device)
        self._batch.set_model(model, device)
        self._queue.set_model(model, device)

    def _on_settings_changed(self, settings: ClassicalSettings):
        self._single.set_classical_settings(settings)
        self._batch.set_classical_settings(settings)
        self._queue.set_classical_settings(settings)

    def _show_status(self, msg: str):
        self._status_lbl.setText(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    app = QApplication(sys.argv)
    app.setApplicationName("DeStripe")
    app.setOrganizationName("BandRemover")
    app.setApplicationVersion("2.0")
    icon_path = _HERE / "assets" / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    app.setStyleSheet(get_qss())

    cfg = load_cfg()
    win = MainWindow(cfg)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
