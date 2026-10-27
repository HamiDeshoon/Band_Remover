"""
Classical / Non-Deep Learning Band & Streak Artifact Detection and Correction.
Engineered for medical X-ray and CT imaging (Dicom_Band_Fixer enhanced):
  - Robust multi-scale second-derivative Laplacian & profile baseline analysis
  - Detector line coverage verification (rejects local anatomy & OSD annotations)
  - Auto-orientation disambiguation (horizontal vs vertical dominance)
  - Prominence score filtering (eliminates spurious background noise lines)
  - Edge-preserving smoothstep / monotonic boundary interpolation
  - Coherent quantum mottle texture synthesis for imperceptible repairs
  - Full native 16-bit (uint16/int16) and 8-bit dynamic range preservation
"""
from __future__ import annotations

import math
from dataclasses import dataclass
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
    """Piecewise Cubic Hermite Interpolating Polynomial in 1D."""
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
    neighbor_pixels: int = 15
    detection_sensitivity: float = 2.5
    border_exclude_ratio: float = 0.03     # 3% excludes OSD text overlays at borders
    orientation: Literal["auto", "horizontal", "vertical", "both"] = "auto"
    interpolation_method: Literal["smoothstep_texture", "linear", "pchip"] = "smoothstep_texture"
    gaussian_sigma: float = 0.0            # Default 0.0: avoid blurring sharp anatomical edges
    min_coverage_ratio: float = 0.35       # True detector line must span >= 35% of the detector
    texture_matching: bool = True          # Inject coherent quantum mottle texture for invisible blending


# ────────────────────────── DETECTION & CORRECTION ───────────────────────────

def detect_artifact_bands(
    img: np.ndarray,
    settings: ClassicalSettings | None = None,
) -> List[ClassicalBand]:
    """
    Detect vertical and horizontal bands in grayscale images (uint8, uint16, int16, float).
    Applies multi-scale Laplacian second-derivative analysis with line coverage validation.
    """
    if settings is None:
        settings = ClassicalSettings()

    img_data = img.astype(np.float64)
    rows, cols = img_data.shape[:2]

    # Safe normalization to 0..255 for thresholding invariance across 8-bit and 16-bit
    i_min = float(np.min(img_data))
    i_max = float(np.max(img_data))
    span = max(i_max - i_min, 1e-5)
    img_norm = (img_data - i_min) / span * 255.0

    orient = getattr(settings, "orientation", "auto").lower()
    h_candidates: List[ClassicalBand] = []
    v_candidates: List[ClassicalBand] = []

    # ── Horizontal Analysis (Rows across Columns) ───────────────────────────
    if orient in ("auto", "horizontal", "both") and rows > 6:
        h_start = max(1, int(rows * settings.border_exclude_ratio))
        h_end = min(rows - 1, int(rows * (1.0 - settings.border_exclude_ratio)))

        if h_end > h_start + 4:
            # 2nd derivative across vertical axis (H-2, W)
            lap_h = np.abs(img_norm[1:-1, :] - 0.5 * (img_norm[:-2, :] + img_norm[2:, :]))
            med_lap = float(np.median(lap_h))
            mad_lap = float(np.median(np.abs(lap_h - med_lap)))
            noise_sigma_h = max(1.4826 * mad_lap, 1.0)
            pixel_thresh_h = max(noise_sigma_h * settings.detection_sensitivity, 6.0)

            # Measure coverage: fraction of columns in row r with abnormal deviation
            coverage_h = np.pad((lap_h > pixel_thresh_h).mean(axis=1), (1, 1), mode="constant")
            row_med_dev = np.pad(np.median(lap_h, axis=1), (1, 1), mode="constant")

            # 1D row profile baseline deviation (for broad soft bands)
            h_profile = np.mean(img_norm, axis=1)
            h_win = max(5, min(51, len(h_profile) // 20 | 1))
            h_dev = np.abs(h_profile - median_filter_1d(h_profile, h_win))
            mad_p = float(np.median(np.abs(h_dev - np.median(h_dev))))
            prof_thresh = max(1.4826 * mad_p * settings.detection_sensitivity, 4.0)

            # Flag rows that meet detector line criteria:
            # High line coverage across columns OR strong profile deviation with moderate coverage
            flagged_h = (coverage_h >= settings.min_coverage_ratio) | (
                (coverage_h >= 0.20) & (h_dev >= prof_thresh)
            )

            # Group contiguous rows
            in_b = False
            s = 0
            for r in range(h_start, h_end):
                if flagged_h[r] and not in_b:
                    in_b = True
                    s = r
                elif not flagged_h[r] and in_b:
                    in_b = False
                    e = r - 1
                    w = e - s + 1
                    if w <= max(settings.band_width_estimate * 4, 30):
                        score = float(np.sum(row_med_dev[s:e+1]) * (1.0 + np.max(coverage_h[s:e+1]) * 2.0))
                        peak_d = float(np.max(row_med_dev[s:e+1]))
                        h_candidates.append(ClassicalBand(
                            id=f"h_{s}_{e}",
                            orientation="horizontal",
                            start_px=s,
                            end_px=e,
                            score=round(score, 2),
                            peak_dev=round(peak_d, 2),
                            enabled=True,
                        ))
            if in_b:
                e = h_end - 1
                w = e - s + 1
                if w <= max(settings.band_width_estimate * 4, 30):
                    score = float(np.sum(row_med_dev[s:e+1]) * (1.0 + np.max(coverage_h[s:e+1]) * 2.0))
                    peak_d = float(np.max(row_med_dev[s:e+1]))
                    h_candidates.append(ClassicalBand(
                        id=f"h_{s}_{e}",
                        orientation="horizontal",
                        start_px=s,
                        end_px=e,
                        score=round(score, 2),
                        peak_dev=round(peak_d, 2),
                        enabled=True,
                    ))

    # ── Vertical Analysis (Columns down Rows) ─────────────────────────────
    if orient in ("auto", "vertical", "both") and cols > 6:
        v_start = max(1, int(cols * settings.border_exclude_ratio))
        v_end = min(cols - 1, int(cols * (1.0 - settings.border_exclude_ratio)))

        if v_end > v_start + 4:
            # 2nd derivative across horizontal axis (H, W-2)
            lap_v = np.abs(img_norm[:, 1:-1] - 0.5 * (img_norm[:, :-2] + img_norm[:, 2:]))
            med_lap_v = float(np.median(lap_v))
            mad_lap_v = float(np.median(np.abs(lap_v - med_lap_v)))
            noise_sigma_v = max(1.4826 * mad_lap_v, 1.0)
            pixel_thresh_v = max(noise_sigma_v * settings.detection_sensitivity, 6.0)

            # Measure coverage: fraction of rows in column c with abnormal deviation
            coverage_v = np.pad((lap_v > pixel_thresh_v).mean(axis=0), (1, 1), mode="constant")
            col_med_dev = np.pad(np.median(lap_v, axis=0), (1, 1), mode="constant")

            v_profile = np.mean(img_norm, axis=0)
            v_win = max(5, min(51, len(v_profile) // 20 | 1))
            v_dev = np.abs(v_profile - median_filter_1d(v_profile, v_win))
            mad_pv = float(np.median(np.abs(v_dev - np.median(v_dev))))
            prof_thresh_v = max(1.4826 * mad_pv * settings.detection_sensitivity, 4.0)

            flagged_v = (coverage_v >= settings.min_coverage_ratio) | (
                (coverage_v >= 0.20) & (v_dev >= prof_thresh_v)
            )

            in_b = False
            s = 0
            for c in range(v_start, v_end):
                if flagged_v[c] and not in_b:
                    in_b = True
                    s = c
                elif not flagged_v[c] and in_b:
                    in_b = False
                    e = c - 1
                    w = e - s + 1
                    if w <= max(settings.band_width_estimate * 4, 30):
                        score = float(np.sum(col_med_dev[s:e+1]) * (1.0 + np.max(coverage_v[s:e+1]) * 2.0))
                        peak_d = float(np.max(col_med_dev[s:e+1]))
                        v_candidates.append(ClassicalBand(
                            id=f"v_{s}_{e}",
                            orientation="vertical",
                            start_px=s,
                            end_px=e,
                            score=round(score, 2),
                            peak_dev=round(peak_d, 2),
                            enabled=True,
                        ))
            if in_b:
                e = v_end - 1
                w = e - s + 1
                if w <= max(settings.band_width_estimate * 4, 30):
                    score = float(np.sum(col_med_dev[s:e+1]) * (1.0 + np.max(coverage_v[s:e+1]) * 2.0))
                    peak_d = float(np.max(col_med_dev[s:e+1]))
                    v_candidates.append(ClassicalBand(
                        id=f"v_{s}_{e}",
                        orientation="vertical",
                        start_px=s,
                        end_px=e,
                        score=round(score, 2),
                        peak_dev=round(peak_d, 2),
                        enabled=True,
                    ))

    # ── Orientation Disambiguation & Prominence Filtering ────────────────
    if orient == "auto":
        sum_h = sum(b.score for b in h_candidates)
        sum_v = sum(b.score for b in v_candidates)
        if sum_h > 2.5 * sum_v:
            bands = h_candidates
        elif sum_v > 2.5 * sum_h:
            bands = v_candidates
        else:
            bands = h_candidates + v_candidates
    elif orient == "horizontal":
        bands = h_candidates
    elif orient == "vertical":
        bands = v_candidates
    else:
        bands = h_candidates + v_candidates

    # Filter out weak spurious candidates relative to the dominant artifact peak
    if bands:
        max_score = max(b.score for b in bands)
        prominence_floor = max(max_score * 0.05, 10.0)
        bands = [b for b in bands if b.score >= prominence_floor]

    bands.sort(key=lambda b: b.score, reverse=True)
    return bands


def correct_bands(
    img: np.ndarray,
    bands: List[ClassicalBand],
    settings: ClassicalSettings | None = None,
    expand_px: int = 1,
) -> np.ndarray:
    """
    Interpolates through artifact band regions using surrounding context.
    Features:
      1. Edge-preserving smoothstep / monotonic boundary interpolation
      2. Coherent texture synthesis (preserves medical quantum mottle)
      3. No collateral orthogonal blurring
      4. Preserves input dtype (uint8, uint16, int16, float32/64)
    """
    if settings is None:
        settings = ClassicalSettings()

    orig_dtype = img.dtype
    corrected = img.astype(np.float64).copy()
    rows, cols = corrected.shape[:2]

    active_bands = [b for b in bands if b.enabled]
    interp_method = getattr(settings, "interpolation_method", "smoothstep_texture")

    for band in active_bands:
        start_px = max(0, band.start_px - expand_px)

        if band.orientation == "vertical":
            end_px = min(cols - 1, band.end_px + expand_px)
            c_left = max(0, start_px - 1)
            c_right = min(cols - 1, end_px + 1)
            n_cols = end_px - start_px + 1

            left_col = corrected[:, c_left]
            right_col = corrected[:, c_right]

            # Texture matching noise scale from clean adjacent columns
            c_outer = max(0, c_left - 1)
            noise_scale = float(np.std(corrected[:, c_left] - corrected[:, c_outer])) / 1.4142

            if interp_method in ("smoothstep_texture", "linear_texture", "smoothstep"):
                for idx, c in enumerate(range(start_px, end_px + 1)):
                    t = (idx + 1.0) / (n_cols + 1.0)
                    w = 3.0 * t * t - 2.0 * t * t * t  # Smoothstep
                    interp_col = (1.0 - w) * left_col + w * right_col

                    if getattr(settings, "texture_matching", True) and noise_scale > 0.5:
                        edge_mag = np.abs(np.gradient(interp_col))
                        flatness = np.exp(-(edge_mag / 10.0) ** 2)
                        np.random.seed(c * 1000 + 42)
                        noise = np.random.normal(0, noise_scale * 0.7, rows) * flatness
                        corrected[:, c] = interp_col + noise
                    else:
                        corrected[:, c] = interp_col

            elif interp_method == "linear":
                for idx, c in enumerate(range(start_px, end_px + 1)):
                    t = (idx + 1.0) / (n_cols + 1.0)
                    corrected[:, c] = (1.0 - t) * left_col + t * right_col

            elif interp_method == "pchip":
                interp_neighbors = max(settings.neighbor_pixels, 10)
                left_start = max(0, start_px - interp_neighbors)
                right_end = min(cols - 1, end_px + interp_neighbors)
                x_left = np.arange(left_start, start_px)
                x_right = np.arange(end_px + 1, right_end + 1)
                x_good = np.concatenate((x_left, x_right))
                x_bad = np.arange(start_px, end_px + 1)

                if len(x_good) >= 2 and len(x_bad) > 0:
                    for r in range(rows):
                        corrected[r, x_bad] = pchip_interpolate(x_good, corrected[r, x_good], x_bad)

            # Optional smoothing along the column axis only
            if settings.gaussian_sigma > 0:
                for c in range(start_px, end_px + 1):
                    corrected[:, c] = gaussian_filter_1d(corrected[:, c], settings.gaussian_sigma)

        else:
            # Horizontal band
            end_px = min(rows - 1, band.end_px + expand_px)
            r_top = max(0, start_px - 1)
            r_bot = min(rows - 1, end_px + 1)
            n_rows = end_px - start_px + 1

            top_row = corrected[r_top, :]
            bot_row = corrected[r_bot, :]

            # Texture matching noise scale from clean adjacent rows
            r_outer = max(0, r_top - 1)
            noise_scale = float(np.std(corrected[r_top, :] - corrected[r_outer, :])) / 1.4142

            if interp_method in ("smoothstep_texture", "linear_texture", "smoothstep"):
                for idx, r in enumerate(range(start_px, end_px + 1)):
                    t = (idx + 1.0) / (n_rows + 1.0)
                    w = 3.0 * t * t - 2.0 * t * t * t  # Smoothstep
                    interp_row = (1.0 - w) * top_row + w * bot_row

                    if getattr(settings, "texture_matching", True) and noise_scale > 0.5:
                        edge_mag = np.abs(np.gradient(interp_row))
                        flatness = np.exp(-(edge_mag / 10.0) ** 2)
                        np.random.seed(r * 1000 + 42)
                        noise = np.random.normal(0, noise_scale * 0.7, cols) * flatness
                        corrected[r, :] = interp_row + noise
                    else:
                        corrected[r, :] = interp_row

            elif interp_method == "linear":
                for idx, r in enumerate(range(start_px, end_px + 1)):
                    t = (idx + 1.0) / (n_rows + 1.0)
                    corrected[r, :] = (1.0 - t) * top_row + t * bot_row

            elif interp_method == "pchip":
                interp_neighbors = max(settings.neighbor_pixels, 10)
                top_start = max(0, start_px - interp_neighbors)
                bottom_end = min(rows - 1, end_px + interp_neighbors)
                y_top = np.arange(top_start, start_px)
                y_bottom = np.arange(end_px + 1, bottom_end + 1)
                y_good = np.concatenate((y_top, y_bottom))
                y_bad = np.arange(start_px, end_px + 1)

                if len(y_good) >= 2 and len(y_bad) > 0:
                    for c in range(cols):
                        corrected[y_bad, c] = pchip_interpolate(y_good, corrected[y_good, c], y_bad)

            # Optional smoothing along the row axis only
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
