"""
Classical / Non-Deep Learning Band & Streak Artifact Detection and Correction.
Ported and enhanced from Dicom_Band_Fixer algorithms:
  - 1D Profile analysis (horizontal & vertical)
  - Running median baseline extraction & robust MAD statistical deviation thresholding
  - Multi-column Laplacian second-derivative anomaly detection for detector lines
  - Contiguous band boundary segmentation (bright & dark lines)
  - Directional interpolation (PCHIP or Linear) using unaffected neighbor regions
  - Orthogonal 1D Gaussian smoothing across interpolated bands
  - Full native support for 8-bit, 16-bit (uint16/int16), and float images
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Literal, Tuple
import numpy as np


# ────────────────────────── FILTER PRIMITIVES ────────────────────────────────

def median_filter_1d(arr: np.ndarray, window_size: int) -> np.ndarray:
    """1D running median filter for profile baselines."""
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return arr.copy()

    half = window_size // 2
    res = np.empty(n, dtype=np.float64)

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        res[i] = np.median(arr[start:end])
    return res


def gaussian_filter_1d(arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """1D Gaussian smoothing filter."""
    arr = np.asarray(arr, dtype=np.float64)
    if sigma <= 0 or len(arr) < 2:
        return arr.copy()

    radius = int(math.ceil(sigma * 3.0))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    padded = np.pad(arr, radius, mode="reflect")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def pchip_interpolate(x_good: np.ndarray, y_good: np.ndarray, x_bad: np.ndarray) -> np.ndarray:
    """
    Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) in 1D.
    Preserves monotonicity and avoids polynomial overshoots/ringing.
    """
    x_good = np.asarray(x_good, dtype=np.float64)
    y_good = np.asarray(y_good, dtype=np.float64)
    x_bad = np.asarray(x_bad, dtype=np.float64)

    n = len(x_good)
    if n == 0:
        return np.zeros_like(x_bad)
    if n == 1:
        return np.full_like(x_bad, y_good[0])

    h = np.diff(x_good)
    h = np.where(h == 0, 1e-6, h)
    delta = np.diff(y_good) / h

    d = np.zeros(n, dtype=np.float64)
    d[0] = delta[0]
    d[-1] = delta[-1]

    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            d[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    out = np.empty_like(x_bad)
    for idx, x in enumerate(x_bad):
        if x <= x_good[0]:
            k = 0
        elif x >= x_good[-1]:
            k = n - 2
        else:
            k = int(np.searchsorted(x_good, x) - 1)
            k = min(max(0, k), n - 2)

        hk = x_good[k + 1] - x_good[k]
        hk = 1e-6 if hk == 0 else hk
        t = (x - x_good[k]) / hk
        t2 = t * t
        t3 = t2 * t

        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2

        out[idx] = (
            h00 * y_good[k]
            + h10 * hk * d[k]
            + h01 * y_good[k + 1]
            + h11 * hk * d[k + 1]
        )
    return out


def linear_interpolate(x_good: np.ndarray, y_good: np.ndarray, x_bad: np.ndarray) -> np.ndarray:
    """Standard 1D linear interpolation."""
    return np.interp(x_bad, x_good, y_good)


# ────────────────────────── DATA STRUCTURES ──────────────────────────────────

@dataclass
class ClassicalBand:
    id: str
    orientation: Literal["vertical", "horizontal"]
    start_px: int
    end_px: int
    score: float
    peak_dev: float
    enabled: bool = True


@dataclass
class ClassicalSettings:
    band_width_estimate: int = 5
    neighbor_pixels: int = 20
    detection_sensitivity: float = 1.5
    border_exclude_ratio: float = 0.01   # Default 1% to ensure lines near top/bottom are never missed
    interpolation_method: Literal["pchip", "linear"] = "pchip"
    gaussian_sigma: float = 1.0


# ────────────────────────── DETECTION & CORRECTION ───────────────────────────

def detect_artifact_bands(
    img: np.ndarray,
    settings: ClassicalSettings | None = None,
) -> List[ClassicalBand]:
    """
    Detect vertical and horizontal bands in grayscale image (2D numpy array: uint8, uint16, int16, or float).
    Uses a hybrid approach combining:
      1. Robust profile deviation (MAD-based thresholding, insensitive to foreground objects).
      2. Multi-column Laplacian second-derivative anomaly detection for detector lines.
    Returns list of ClassicalBand sorted by artifact prominence score.
    """
    if settings is None:
        settings = ClassicalSettings()

    img_data = img.astype(np.float64)
    rows, cols = img_data.shape[:2]
    bands: List[ClassicalBand] = []

    # Safe normalization to 0..255 for thresholding invariance across 8-bit and 16-bit
    i_min = float(np.min(img_data))
    i_max = float(np.max(img_data))
    span = max(i_max - i_min, 1e-5)
    img_norm = (img_data - i_min) / span * 255.0

    # ── Horizontal Analysis (mean across rows & row Laplacian) ────────────────
    h_start = max(0, int(math.floor(rows * settings.border_exclude_ratio)))
    h_end = min(rows, int(math.ceil(rows * (1.0 - settings.border_exclude_ratio))))

    if h_end > h_start + 6:
        h_profile = np.mean(img_norm, axis=1)[h_start:h_end]
        n_h = len(h_profile)
        h_win = max(5, min(51, n_h // 20))
        if h_win % 2 == 0:
            h_win += 1

        h_baseline = median_filter_1d(h_profile, h_win)
        h_dev = h_profile - h_baseline

        # Robust statistics via MAD to prevent foreground objects (e.g. key/bone) from raising threshold
        h_med = float(np.median(h_dev))
        h_mad = float(np.median(np.abs(h_dev - h_med)))
        h_robust_std = max(1.4826 * h_mad, 0.2)
        h_thresh = settings.detection_sensitivity * h_robust_std
        h_profile_flag = np.abs(h_dev - h_med) > h_thresh

        # Row Laplacian: measures row discontinuity across columns
        # Detector lines span the entire detector, so the column median/percentile of row differences spikes sharply
        if rows > 4:
            diff_h = np.abs(img_norm[1:-1, :] - 0.5 * (img_norm[:-2, :] + img_norm[2:, :]))
            row_lap = np.percentile(diff_h, 40, axis=1)
            row_lap = np.pad(row_lap, (1, 1), mode="edge")[h_start:h_end]
            lap_base = median_filter_1d(row_lap, h_win)
            lap_dev = row_lap - lap_base
            lap_med = float(np.median(lap_dev))
            lap_mad = float(np.median(np.abs(lap_dev - lap_med)))
            lap_std = max(1.4826 * lap_mad, 0.2)
            h_lap_flag = (lap_dev - lap_med) > (settings.detection_sensitivity * lap_std)
        else:
            h_lap_flag = np.zeros(n_h, dtype=bool)
            lap_dev = np.zeros(n_h, dtype=np.float64)

        h_flag = h_profile_flag | h_lap_flag

        # Group contiguous artifact rows
        in_reg = False
        reg_start = 0
        h_regions: List[Tuple[int, int]] = []
        for i, val in enumerate(h_flag):
            if val and not in_reg:
                in_reg = True
                reg_start = i
            elif not val and in_reg:
                in_reg = False
                h_regions.append((reg_start, i - 1))
        if in_reg:
            h_regions.append((reg_start, len(h_flag) - 1))

        max_allowed_w = max(settings.band_width_estimate * 6, 40)
        for s_idx, e_idx in h_regions:
            w = e_idx - s_idx + 1
            if w <= max_allowed_w:
                seg_dev = np.abs(h_dev[s_idx:e_idx + 1])
                sum_dev = float(np.sum(seg_dev))
                max_dev = float(np.max(seg_dev))
                lap_s = float(np.sum(lap_dev[s_idx:e_idx + 1])) if rows > 4 else 0.0
                score = max(sum_dev, lap_s * 2.0) * (1.0 + max_dev / 10.0)
                if score > 0:
                    bands.append(ClassicalBand(
                        id=f"h_{s_idx + h_start}_{e_idx + h_start}",
                        orientation="horizontal",
                        start_px=s_idx + h_start,
                        end_px=e_idx + h_start,
                        score=round(score, 2),
                        peak_dev=round(max_dev, 2),
                        enabled=True,
                    ))

    # ── Vertical Analysis (mean down columns & column Laplacian) ──────────────
    v_start = max(0, int(math.floor(cols * settings.border_exclude_ratio)))
    v_end = min(cols, int(math.ceil(cols * (1.0 - settings.border_exclude_ratio))))

    if v_end > v_start + 6:
        v_profile = np.mean(img_norm, axis=0)[v_start:v_end]
        n_v = len(v_profile)
        v_win = max(5, min(51, n_v // 20))
        if v_win % 2 == 0:
            v_win += 1

        v_baseline = median_filter_1d(v_profile, v_win)
        v_dev = v_profile - v_baseline
        v_med = float(np.median(v_dev))
        v_mad = float(np.median(np.abs(v_dev - v_med)))
        v_robust_std = max(1.4826 * v_mad, 0.2)
        v_thresh = settings.detection_sensitivity * v_robust_std
        v_profile_flag = np.abs(v_dev - v_med) > v_thresh

        if cols > 4:
            diff_v = np.abs(img_norm[:, 1:-1] - 0.5 * (img_norm[:, :-2] + img_norm[:, 2:]))
            col_lap = np.percentile(diff_v, 40, axis=0)
            col_lap = np.pad(col_lap, (1, 1), mode="edge")[v_start:v_end]
            lap_base_v = median_filter_1d(col_lap, v_win)
            lap_dev_v = col_lap - lap_base_v
            lap_med_v = float(np.median(lap_dev_v))
            lap_mad_v = float(np.median(np.abs(lap_dev_v - lap_med_v)))
            lap_std_v = max(1.4826 * lap_mad_v, 0.2)
            v_lap_flag = (lap_dev_v - lap_med_v) > (settings.detection_sensitivity * lap_std_v)
        else:
            v_lap_flag = np.zeros(n_v, dtype=bool)
            lap_dev_v = np.zeros(n_v, dtype=np.float64)

        v_flag = v_profile_flag | v_lap_flag

        in_reg = False
        reg_start = 0
        v_regions: List[Tuple[int, int]] = []
        for i, val in enumerate(v_flag):
            if val and not in_reg:
                in_reg = True
                reg_start = i
            elif not val and in_reg:
                in_reg = False
                v_regions.append((reg_start, i - 1))
        if in_reg:
            v_regions.append((reg_start, len(v_flag) - 1))

        max_allowed_w = max(settings.band_width_estimate * 6, 40)
        for s_idx, e_idx in v_regions:
            w = e_idx - s_idx + 1
            if w <= max_allowed_w:
                seg_dev = np.abs(v_dev[s_idx:e_idx + 1])
                sum_dev = float(np.sum(seg_dev))
                max_dev = float(np.max(seg_dev))
                lap_s = float(np.sum(lap_dev_v[s_idx:e_idx + 1])) if cols > 4 else 0.0
                score = max(sum_dev, lap_s * 2.0) * (1.0 + max_dev / 10.0)
                if score > 0:
                    bands.append(ClassicalBand(
                        id=f"v_{s_idx + v_start}_{e_idx + v_start}",
                        orientation="vertical",
                        start_px=s_idx + v_start,
                        end_px=e_idx + v_start,
                        score=round(score, 2),
                        peak_dev=round(max_dev, 2),
                        enabled=True,
                    ))

    bands.sort(key=lambda b: b.score, reverse=True)
    return bands


def correct_bands(
    img: np.ndarray,
    bands: List[ClassicalBand],
    settings: ClassicalSettings | None = None,
    expand_px: int = 2,
) -> np.ndarray:
    """
    Interpolates through the band regions using surrounding context.
    Works directly on 2D arrays (preserves input dtype: uint8, uint16, int16, float).
    """
    if settings is None:
        settings = ClassicalSettings()

    orig_dtype = img.dtype
    corrected = img.astype(np.float64).copy()
    rows, cols = corrected.shape[:2]

    active_bands = [b for b in bands if b.enabled]
    interp_neighbors = max(settings.neighbor_pixels, 15)

    for band in active_bands:
        start_px = max(0, band.start_px - expand_px)

        if band.orientation == "vertical":
            end_px = min(cols - 1, band.end_px + expand_px)
            left_start = max(0, start_px - interp_neighbors)
            right_end = min(cols - 1, end_px + interp_neighbors)

            x_left = np.arange(left_start, start_px)
            x_right = np.arange(end_px + 1, right_end + 1)
            x_good = np.concatenate((x_left, x_right))
            x_bad = np.arange(start_px, end_px + 1)

            if len(x_good) >= 2 and len(x_bad) > 0:
                for r in range(rows):
                    y_good = corrected[r, x_good]
                    if settings.interpolation_method == "linear":
                        interp_vals = linear_interpolate(x_good, y_good, x_bad)
                    else:
                        interp_vals = pchip_interpolate(x_good, y_good, x_bad)
                    corrected[r, x_bad] = interp_vals

            # Orthogonal 1D Gaussian smoothing along columns
            if settings.gaussian_sigma > 0:
                for c in range(start_px, end_px + 1):
                    corrected[:, c] = gaussian_filter_1d(corrected[:, c], settings.gaussian_sigma)

        else:
            # Horizontal band
            end_px = min(rows - 1, band.end_px + expand_px)
            top_start = max(0, start_px - interp_neighbors)
            bottom_end = min(rows - 1, end_px + interp_neighbors)

            y_top = np.arange(top_start, start_px)
            y_bottom = np.arange(end_px + 1, bottom_end + 1)
            y_good = np.concatenate((y_top, y_bottom))
            y_bad = np.arange(start_px, end_px + 1)

            if len(y_good) >= 2 and len(y_bad) > 0:
                for c in range(cols):
                    vals_good = corrected[y_good, c]
                    if settings.interpolation_method == "linear":
                        interp_vals = linear_interpolate(y_good, vals_good, y_bad)
                    else:
                        interp_vals = pchip_interpolate(y_good, vals_good, y_bad)
                    corrected[y_bad, c] = interp_vals

            # Orthogonal 1D Gaussian smoothing along rows
            if settings.gaussian_sigma > 0:
                for r in range(start_px, end_px + 1):
                    corrected[r, :] = gaussian_filter_1d(corrected[r, :], settings.gaussian_sigma)

    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        return np.clip(np.round(corrected), info.min, info.max).astype(orig_dtype)
    return corrected.astype(orig_dtype)


def classical_band_filter(
    img: np.ndarray,
    settings: ClassicalSettings | None = None,
) -> Tuple[np.ndarray, List[ClassicalBand]]:
    """Complete one-shot non-deep-learning band detection + correction pipeline."""
    if settings is None:
        settings = ClassicalSettings()
    bands = detect_artifact_bands(img, settings)
    corrected = correct_bands(img, bands, settings)
    return corrected, bands
