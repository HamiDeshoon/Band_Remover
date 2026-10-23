"""
Inference engine supporting:
1. SMP U-Net with EfficientNet-b0 backbone (deep learning from Kaggle training)
2. DestripePlus architecture (legacy fallback)
3. Classical / Non-Deep Learning profile & interpolation band filter (from Dicom_Band_Fixer)
4. Tiled 512x512 inference with 64px Gaussian-blended overlap
"""
from __future__ import annotations

import sys
import os
import time
from pathlib import Path
from typing import Optional, Union, Tuple, List

if sys.platform == "win32":
    import ctypes
    _sys_icu = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "icuuc.dll")
    if os.path.exists(_sys_icu):
        try:
            ctypes.windll.kernel32.LoadLibraryW(_sys_icu)
        except Exception:
            pass

from PyQt5.QtCore import QThread, pyqtSignal as Signal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gui.classical_filter import (
    ClassicalSettings,
    ClassicalBand,
    detect_artifact_bands,
    correct_bands,
    classical_band_filter,
)


# ─────────────────────────── DEEP LEARNING MODELS ────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(c, c, 3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class AxialStripAttention(nn.Module):
    """Broadcast Axial Strip Attention."""
    def __init__(self, c: int):
        super().__init__()
        reduced = max(8, c // 4)
        self.row_pool = nn.Sequential(
            nn.Conv2d(c, reduced, 1, bias=False),
            nn.PReLU(),
            nn.Conv2d(reduced, c, 1, bias=False),
        )
        self.col_pool = nn.Sequential(
            nn.Conv2d(c, reduced, 1, bias=False),
            nn.PReLU(),
            nn.Conv2d(reduced, c, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.row_pool(x.mean(dim=3, keepdim=True))
        c = self.col_pool(x.mean(dim=2, keepdim=True))
        return x + x * torch.sigmoid(r + c)


class DestripePlus(nn.Module):
    def __init__(self, b: int = 32):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(1, b, 3, padding=1), nn.PReLU(), ResBlock(b))
        self.e2 = nn.Sequential(
            nn.Conv2d(b, 2 * b, 3, stride=2, padding=1), nn.PReLU(),
            ResBlock(2 * b), AxialStripAttention(2 * b),
        )
        self.e3 = nn.Sequential(
            nn.Conv2d(2 * b, 4 * b, 3, stride=2, padding=1), nn.PReLU(),
            ResBlock(4 * b), AxialStripAttention(4 * b),
        )
        self.mid = nn.Sequential(ResBlock(4 * b), AxialStripAttention(4 * b), ResBlock(4 * b))
        self.up2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2)
        self.d2 = nn.Sequential(
            nn.Conv2d(4 * b, 2 * b, 1), nn.PReLU(),
            ResBlock(2 * b), AxialStripAttention(2 * b),
        )
        self.up1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2)
        self.d1 = nn.Sequential(nn.Conv2d(2 * b, b, 1), nn.PReLU(), ResBlock(b))
        self.out_res = nn.Conv2d(b, 1, 3, padding=1)
        self.out_mask = nn.Conv2d(b, 1, 3, padding=1)

    def forward(self, x: torch.Tensor):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        mid = self.mid(e3)
        d2 = self.d2(torch.cat((self.up2(mid), e2), dim=1))
        d1 = self.d1(torch.cat((self.up1(d2), e1), dim=1))
        return x + self.out_res(d1), self.out_mask(d1)


class StripeNetSMP(nn.Module):
    """SMP U-Net with EfficientNet-b0 matching destripe_best_eff.ckpt training code."""
    def __init__(self, encoder: str = "efficientnet-b0"):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.unet = smp.Unet(encoder, encoder_weights=None, in_channels=1, classes=2)

    def forward(self, x: torch.Tensor):
        out = self.unet((x - 0.5) / 0.25).float()
        residual, logit = out[:, :1], out[:, 1:]
        return x + torch.sigmoid(logit) * residual, logit


# ──────────────────────────── MODEL LOADING ──────────────────────────────────

def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_label() -> str:
    if torch.cuda.is_available():
        return f"GPU · {torch.cuda.get_device_name(0)}"
    return "CPU"


def load_model(ckpt_path: str, device: torch.device) -> nn.Module:
    """
    Load model weights from .pt or .ckpt.
    Detects if weights belong to SMP U-Net or DestripePlus.
    """
    try:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        sd = torch.load(ckpt_path, map_location="cpu")

    if isinstance(sd, dict) and "state_dict" in sd:
        raw = sd["state_dict"]
    else:
        raw = sd

    # Determine architecture from key signatures
    is_smp = any("unet.encoder" in k or k.startswith("encoder.") for k in raw.keys())
    
    if is_smp:
        model = StripeNetSMP(encoder="efficientnet-b0")
        clean_sd = {}
        for k, v in raw.items():
            k_clean = k
            if k_clean.startswith("net."):
                k_clean = k_clean[4:]
            if k_clean.startswith("unet."):
                k_clean = k_clean[5:]
            if not k_clean.startswith("pos_w"):
                clean_sd[k_clean] = v
        model.unet.load_state_dict(clean_sd, strict=False)
    else:
        model = DestripePlus(b=32)
        clean_sd = {(k[4:] if k.startswith("net.") else k): v for k, v in raw.items() if not k.startswith("pos_w")}
        model.load_state_dict(clean_sd, strict=True)

    model.eval()
    model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


# ─────────────────────────── TILE INFERENCE ──────────────────────────────────

def _gaussian_kernel(size: int) -> np.ndarray:
    sigma = size / 6.0
    x = np.linspace(-size / 2, size / 2, size)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return np.outer(k, k).astype(np.float32)


def tile_infer(
    img_u8: np.ndarray,
    model: nn.Module,
    device: torch.device,
    tile: int = 512,
    overlap: int = 64,
    progress_cb=None,
) -> np.ndarray:
    """Tiled neural inference with Gaussian-blended overlaps."""
    H, W = img_u8.shape
    step = tile - overlap
    pad_h = max(0, tile - H)
    pad_w = max(0, tile - W)

    padded = np.pad(img_u8, ((0, pad_h), (0, pad_w)), mode="reflect")
    pH, pW = padded.shape

    accumulator = np.zeros((pH, pW), dtype=np.float32)
    weight_map = np.zeros((pH, pW), dtype=np.float32)
    gauss = _gaussian_kernel(tile)

    ys = list(range(0, max(1, pH - tile + 1), step)) or [0]
    xs = list(range(0, max(1, pW - tile + 1), step)) or [0]
    if ys[-1] + tile < pH:
        ys.append(pH - tile)
    if xs[-1] + tile < pW:
        xs.append(pW - tile)

    total = len(ys) * len(xs)
    done = 0

    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = padded[y:y + tile, x:x + tile].astype(np.float32) / 255.0
                t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                if device.type == "cuda":
                    t = t.to(memory_format=torch.channels_last)
                out, _ = model(t)
                out_np = out.squeeze().float().cpu().numpy()
                out_np = np.clip(out_np, 0.0, 1.0)

                accumulator[y:y + tile, x:x + tile] += out_np * gauss
                weight_map[y:y + tile, x:x + tile] += gauss

                done += 1
                if progress_cb:
                    progress_cb(int(done * 100 / total))

    result = accumulator[:H, :W] / np.maximum(weight_map[:H, :W], 1e-8)
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


# ─────────────────────────── QTHREAD WORKER ──────────────────────────────────

class InferenceWorker(QThread):
    """
    Runs either DL inference or Classical Non-DL filter in background thread.
    Emits progress (0-100), result image, stats, and detected bands count.
    """
    progress = Signal(int)
    result = Signal(object)
    error = Signal(str)
    finished_with_stats = Signal(float, float, int)  # elapsed, psnr, band_count

    def __init__(
        self,
        img_u8: np.ndarray,
        method: str = "dl",  # "dl", "classical", "both"
        model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        classical_settings: Optional[ClassicalSettings] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._img = img_u8
        self._method = method
        self._model = model
        self._device = device
        self._classical_settings = classical_settings or ClassicalSettings()

    def run(self):
        try:
            t0 = time.perf_counter()
            band_count = 0
            
            if self._method == "classical":
                self.progress.emit(25)
                corrected, bands = classical_band_filter(self._img, self._classical_settings)
                band_count = len(bands)
                self.progress.emit(100)
                out = corrected
                
            elif self._method == "dl":
                if self._model is None:
                    raise ValueError("Model is not loaded for Deep Learning mode.")
                out = tile_infer(
                    self._img,
                    self._model,
                    self._device,
                    progress_cb=lambda p: self.progress.emit(p),
                )
                
            elif self._method == "both":
                # First pass: Classical interpolation on detected bands
                self.progress.emit(15)
                stage1, bands = classical_band_filter(self._img, self._classical_settings)
                band_count = len(bands)
                self.progress.emit(30)
                
                # Second pass: Neural U-Net polish
                if self._model is not None:
                    out = tile_infer(
                        stage1,
                        self._model,
                        self._device,
                        progress_cb=lambda p: self.progress.emit(30 + int(p * 0.7)),
                    )
                else:
                    out = stage1
            else:
                out = self._img

            elapsed = time.perf_counter() - t0
            p = psnr(self._img, out)
            self.result.emit(out)
            self.finished_with_stats.emit(elapsed, p, band_count)
        except Exception as exc:
            self.error.emit(str(exc))
