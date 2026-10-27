"""
Pure Python Core Engine for Band Artifact Removal.
No third-party dependencies required (uses only Python standard library).
Supports PGM (P2/P5), PPM (P3/P6), BMP (grayscale/24-bit RGB), and CSV image I/O.
"""
from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Literal, Optional


class Image2D:
    def __init__(self, width: int, height: int, pixels: List[List[float]], max_val: int = 255):
        self.width = width
        self.height = height
        self.pixels = pixels
        self.max_val = max_val

    def copy(self) -> Image2D:
        new_pixels = [row[:] for row in self.pixels]
        return Image2D(self.width, self.height, new_pixels, self.max_val)

    def to_uint8_pixels(self) -> List[List[int]]:
        return [
            [min(255, max(0, int(round(val)))) for val in row]
            for row in self.pixels
        ]


def read_pgm(filepath: str) -> Image2D:
    with open(filepath, 'rb') as f:
        data = f.read()

    tokens = []
    idx = 0
    n = len(data)
    while idx < n and len(tokens) < 4:
        # Skip whitespace and comments
        while idx < n and (data[idx] in (32, 9, 10, 13) or data[idx] == 35):
            if data[idx] == 35: # '#'
                while idx < n and data[idx] not in (10, 13):
                    idx += 1
            else:
                idx += 1
        if idx >= n:
            break
        start = idx
        while idx < n and data[idx] not in (32, 9, 10, 13) and data[idx] != 35:
            idx += 1
        tokens.append(data[start:idx].decode('latin1'))

    if len(tokens) < 4:
        raise ValueError(f"Invalid PGM header in {filepath}")

    magic, width_str, height_str, max_val_str = tokens[:4]
    width, height, max_val = int(width_str), int(height_str), int(max_val_str)

    if magic not in ("P2", "P5"):
        raise ValueError(f"Unsupported PGM magic number '{magic}'")

    # Skip 1 whitespace byte after max_val
    if idx < n and data[idx] in (32, 9, 10, 13):
        idx += 1

    raster_bytes = data[idx:]
    pixels = []
    if magic == "P5":
        pos = 0
        for r in range(height):
            pixels.append([float(b) for b in raster_bytes[pos : pos + width]])
            pos += width
    else:
        vals = [float(x) for x in raster_bytes.decode('latin1').split()]
        pos = 0
        for r in range(height):
            pixels.append(vals[pos : pos + width])
            pos += width

    return Image2D(width, height, pixels, max_val)


def write_pgm(image: Image2D, filepath: str) -> None:
    uint8_rows = image.to_uint8_pixels()
    header = f"P5\n{image.width} {image.height}\n255\n".encode('ascii')
    data = bytearray(header)
    for row in uint8_rows:
        data.extend(row)
    with open(filepath, 'wb') as f:
        f.write(data)


def read_bmp(filepath: str) -> Image2D:
    with open(filepath, 'rb') as f:
        data = f.read()

    if len(data) < 54 or data[:2] != b'BM':
        raise ValueError("Not a valid BMP file")

    data_offset = struct.unpack('<I', data[10:14])[0]
    header_size, width, height, planes, bpp = struct.unpack('<IiiHH', data[14:30])
    width = abs(width)
    is_top_down = height < 0
    height = abs(height)

    row_stride = ((width * (bpp // 8) + 3) // 4) * 4
    pixels = [[0.0] * width for _ in range(height)]

    for r in range(height):
        row_idx = r if is_top_down else (height - 1 - r)
        offset = data_offset + r * row_stride
        row_data = data[offset : offset + row_stride]

        for c in range(width):
            if bpp == 8:
                val = float(row_data[c])
            elif bpp in (24, 32):
                bytes_per_pixel = bpp // 8
                b = row_data[c * bytes_per_pixel]
                g = row_data[c * bytes_per_pixel + 1]
                r_val = row_data[c * bytes_per_pixel + 2]
                val = 0.299 * r_val + 0.587 * g + 0.114 * b
            else:
                val = 0.0
            pixels[row_idx][c] = val

    return Image2D(width, height, pixels)


def write_bmp(image: Image2D, filepath: str) -> None:
    width = image.width
    height = image.height
    uint8_rows = image.to_uint8_pixels()

    row_padding = (4 - (width * 3) % 4) % 4
    row_stride = width * 3 + row_padding
    image_size = row_stride * height
    file_size = 54 + image_size

    header = struct.pack(
        '<2sIHHIIIIHHIIIIII',
        b'BM',
        file_size,
        0, 0,
        54,
        40,
        width, height,
        1, 24,
        0, image_size,
        2835, 2835,
        0, 0
    )

    pixel_bytes = bytearray()
    for r in reversed(range(height)):
        row = uint8_rows[r]
        for c in range(width):
            val = row[c]
            pixel_bytes.extend([val, val, val])  # B, G, R
        pixel_bytes.extend([0] * row_padding)

    with open(filepath, 'wb') as f:
        f.write(header)
        f.write(pixel_bytes)


def read_csv(filepath: str) -> Image2D:
    pixels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                row = [float(x.strip()) for x in line.split(',') if x.strip()]
                if row:
                    pixels.append(row)
    height = len(pixels)
    width = len(pixels[0]) if height > 0 else 0
    return Image2D(width, height, pixels)


def write_csv(image: Image2D, filepath: str) -> None:
    uint8_rows = image.to_uint8_pixels()
    with open(filepath, 'w', encoding='utf-8') as f:
        for row in uint8_rows:
            f.write(','.join(str(val) for val in row) + '\n')


def load_image(filepath: str) -> Image2D:
    ext = Path(filepath).suffix.lower()
    if ext in ('.pgm', '.pnm'):
        return read_pgm(filepath)
    elif ext == '.bmp':
        return read_bmp(filepath)
    elif ext in ('.csv', '.txt'):
        return read_csv(filepath)
    else:
        try:
            return read_pgm(filepath)
        except Exception:
            return read_bmp(filepath)


def save_image(image: Image2D, filepath: str) -> None:
    ext = Path(filepath).suffix.lower()
    if ext in ('.pgm', '.pnm'):
        write_pgm(image, filepath)
    elif ext in ('.csv', '.txt'):
        write_csv(image, filepath)
    else:
        write_bmp(image, filepath)


def median_1d(lst: List[float]) -> float:
    s = sorted(lst)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def median_filter_1d(arr: List[float], window_size: int) -> List[float]:
    n = len(arr)
    if n == 0:
        return []
    half = window_size // 2
    res = [0.0] * n
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        res[i] = median_1d(arr[start:end])
    return res


def gaussian_filter_1d(arr: List[float], sigma: float = 1.0) -> List[float]:
    n = len(arr)
    if sigma <= 0 or n < 2:
        return arr[:]

    radius = int(math.ceil(sigma * 3.0))
    kernel = []
    weight_sum = 0.0
    for x in range(-radius, radius + 1):
        w = math.exp(-0.5 * (x / sigma) ** 2)
        kernel.append(w)
        weight_sum += w

    kernel = [w / weight_sum for w in kernel]

    padded = [0.0] * (n + 2 * radius)
    for i in range(radius):
        padded[i] = arr[radius - i]
        padded[n + radius + i] = arr[n - 2 - i]
    for i in range(n):
        padded[radius + i] = arr[i]

    res = [0.0] * n
    for i in range(n):
        s = 0.0
        for k in range(len(kernel)):
            s += padded[i + k] * kernel[k]
        res[i] = s
    return res


def linear_interpolate_1d(x_good: List[float], y_good: List[float], x_bad: List[float]) -> List[float]:
    if not x_good or not y_good:
        return [0.0] * len(x_bad)
    if len(x_good) == 1:
        return [y_good[0]] * len(x_bad)

    out = []
    n = len(x_good)
    for x in x_bad:
        if x <= x_good[0]:
            out.append(y_good[0])
        elif x >= x_good[-1]:
            out.append(y_good[-1])
        else:
            idx = 0
            while idx < n - 1 and x_good[idx + 1] < x:
                idx += 1
            x0, x1 = x_good[idx], x_good[idx + 1]
            y0, y1 = y_good[idx], y_good[idx + 1]
            t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
            out.append(y0 + t * (y1 - y0))
    return out


def pchip_interpolate_1d(x_good: List[float], y_good: List[float], x_bad: List[float]) -> List[float]:
    n = len(x_good)
    if n == 0:
        return [0.0] * len(x_bad)
    if n == 1:
        return [y_good[0]] * len(x_bad)

    h = [x_good[i + 1] - x_good[i] for i in range(n - 1)]
    for i in range(len(h)):
        if h[i] == 0:
            h[i] = 1e-6

    delta = [(y_good[i + 1] - y_good[i]) / h[i] for i in range(n - 1)]

    d = [0.0] * n
    d[0] = delta[0]
    d[-1] = delta[-1]

    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            d[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    out = []
    for x in x_bad:
        if x <= x_good[0]:
            out.append(y_good[0])
        elif x >= x_good[-1]:
            out.append(y_good[-1])
        else:
            k = 0
            while k < n - 2 and x_good[k + 1] <= x:
                k += 1
            hk = x_good[k + 1] - x_good[k]
            if hk == 0:
                hk = 1e-6
            t = (x - x_good[k]) / hk
            t2 = t * t
            t3 = t2 * t

            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2

            val = (
                h00 * y_good[k]
                + h10 * hk * d[k]
                + h01 * y_good[k + 1]
                + h11 * hk * d[k + 1]
            )
            out.append(val)
    return out


@dataclass
class Band:
    id: str
    orientation: Literal["vertical", "horizontal"]
    start_px: int
    end_px: int
    score: float
    peak_dev: float
    enabled: bool = True


@dataclass
class FilterSettings:
    band_width_estimate: int = 5
    neighbor_pixels: int = 20
    detection_sensitivity: float = 1.5
    border_exclude_ratio: float = 0.15
    interpolation_method: Literal["pchip", "linear"] = "pchip"
    gaussian_sigma: float = 1.0


def detect_bands(img: Image2D, settings: Optional[FilterSettings] = None) -> List[Band]:
    if settings is None:
        settings = FilterSettings()

    width = img.width
    height = img.height
    bands: List[Band] = []

    v_profile = [0.0] * width
    for c in range(width):
        s = sum(img.pixels[r][c] for r in range(height))
        v_profile[c] = s / height

    v_start = max(0, int(math.floor(width * settings.border_exclude_ratio)))
    v_end = min(width, int(math.floor(width * (1.0 - settings.border_exclude_ratio))))

    if v_end > v_start + 10:
        v_region = v_profile[v_start:v_end]
        v_win = max(5, len(v_region) // 20)
        if v_win % 2 == 0:
            v_win += 1

        v_baseline = median_filter_1d(v_region, v_win)
        v_dev = [v_region[i] - v_baseline[i] for i in range(len(v_region))]

        v_mean = sum(v_dev) / len(v_dev)
        v_variance = sum((x - v_mean) ** 2 for x in v_dev) / len(v_dev)
        v_std = math.sqrt(v_variance)
        v_thresh = v_mean + settings.detection_sensitivity * v_std

        v_regions = []
        in_reg = False
        reg_start = 0
        for i, val in enumerate(v_dev):
            if val > v_thresh and not in_reg:
                in_reg = True
                reg_start = i
            elif val <= v_thresh and in_reg:
                in_reg = False
                v_regions.append((reg_start, i - 1))
        if in_reg:
            v_regions.append((reg_start, len(v_dev) - 1))

        for s_idx, e_idx in v_regions:
            w = e_idx - s_idx + 1
            if w <= settings.band_width_estimate * 4:
                seg_dev = v_dev[s_idx:e_idx + 1]
                sum_dev = sum(seg_dev)
                max_dev = max(seg_dev)
                score = sum_dev * (1.0 + max(0.0, max_dev) / 10.0)
                if score > 0:
                    bands.append(Band(
                        id=f"v_{s_idx + v_start}_{e_idx + v_start}",
                        orientation="vertical",
                        start_px=s_idx + v_start,
                        end_px=e_idx + v_start,
                        score=round(score, 2),
                        peak_dev=round(max_dev, 2),
                    ))

    h_profile = [sum(img.pixels[r]) / width for r in range(height)]
    h_start = max(0, int(math.floor(height * settings.border_exclude_ratio)))
    h_end = min(height, int(math.floor(height * (1.0 - settings.border_exclude_ratio))))

    if h_end > h_start + 10:
        h_region = h_profile[h_start:h_end]
        h_win = max(5, len(h_region) // 20)
        if h_win % 2 == 0:
            h_win += 1

        h_baseline = median_filter_1d(h_region, h_win)
        h_dev = [h_region[i] - h_baseline[i] for i in range(len(h_region))]

        h_mean = sum(h_dev) / len(h_dev)
        h_variance = sum((x - h_mean) ** 2 for x in h_dev) / len(h_dev)
        h_std = math.sqrt(h_variance)
        h_thresh = h_mean + settings.detection_sensitivity * h_std

        h_regions = []
        in_reg = False
        reg_start = 0
        for i, val in enumerate(h_dev):
            if val > h_thresh and not in_reg:
                in_reg = True
                reg_start = i
            elif val <= h_thresh and in_reg:
                in_reg = False
                h_regions.append((reg_start, i - 1))
        if in_reg:
            h_regions.append((reg_start, len(h_dev) - 1))

        for s_idx, e_idx in h_regions:
            w = e_idx - s_idx + 1
            if w <= settings.band_width_estimate * 4:
                seg_dev = h_dev[s_idx:e_idx + 1]
                sum_dev = sum(seg_dev)
                max_dev = max(seg_dev)
                score = sum_dev * (1.0 + max(0.0, max_dev) / 10.0)
                if score > 0:
                    bands.append(Band(
                        id=f"h_{s_idx + h_start}_{e_idx + h_start}",
                        orientation="horizontal",
                        start_px=s_idx + h_start,
                        end_px=e_idx + h_start,
                        score=round(score, 2),
                        peak_dev=round(max_dev, 2),
                    ))

    bands.sort(key=lambda b: b.score, reverse=True)
    return bands


def correct_bands(img: Image2D, bands: List[Band], settings: Optional[FilterSettings] = None, expand_px: int = 3) -> Image2D:
    if settings is None:
        settings = FilterSettings()

    res = img.copy()
    width = res.width
    height = res.height
    interp_neighbors = max(settings.neighbor_pixels, 15)

    for band in bands:
        if not band.enabled:
            continue

        start_px = max(0, band.start_px - expand_px)

        if band.orientation == "vertical":
            end_px = min(width - 1, band.end_px + expand_px)
            left_start = max(0, start_px - interp_neighbors)
            right_end = min(width - 1, end_px + interp_neighbors)

            x_left = list(range(left_start, start_px))
            x_right = list(range(end_px + 1, right_end + 1))
            x_good = [float(x) for x in x_left + x_right]
            x_bad = [float(x) for x in range(start_px, end_px + 1)]

            if len(x_good) >= 2 and len(x_bad) > 0:
                for r in range(height):
                    y_good = [res.pixels[r][int(c)] for c in x_good]
                    if settings.interpolation_method == "linear":
                        interp = linear_interpolate_1d(x_good, y_good, x_bad)
                    else:
                        interp = pchip_interpolate_1d(x_good, y_good, x_bad)
                    for idx, c in enumerate(range(start_px, end_px + 1)):
                        res.pixels[r][c] = interp[idx]

            if settings.gaussian_sigma > 0:
                for c in range(start_px, end_px + 1):
                    col_vals = [res.pixels[r][c] for r in range(height)]
                    smoothed = gaussian_filter_1d(col_vals, settings.gaussian_sigma)
                    for r in range(height):
                        res.pixels[r][c] = smoothed[r]

        else:
            end_px = min(height - 1, band.end_px + expand_px)
            top_start = max(0, start_px - interp_neighbors)
            bottom_end = min(height - 1, end_px + interp_neighbors)

            y_top = list(range(top_start, start_px))
            y_bottom = list(range(end_px + 1, bottom_end + 1))
            y_good = [float(y) for y in y_top + y_bottom]
            y_bad = [float(y) for y in range(start_px, end_px + 1)]

            if len(y_good) >= 2 and len(y_bad) > 0:
                for c in range(width):
                    vals_good = [res.pixels[int(r)][c] for r in y_good]
                    if settings.interpolation_method == "linear":
                        interp = linear_interpolate_1d(y_good, vals_good, y_bad)
                    else:
                        interp = pchip_interpolate_1d(y_good, vals_good, y_bad)
                    for idx, r in enumerate(range(start_px, end_px + 1)):
                        res.pixels[r][c] = interp[idx]

            if settings.gaussian_sigma > 0:
                for r in range(start_px, end_px + 1):
                    row_vals = res.pixels[r][:]
                    smoothed = gaussian_filter_1d(row_vals, settings.gaussian_sigma)
                    res.pixels[r] = smoothed

    return res


def denoise_image(img: Image2D, settings: Optional[FilterSettings] = None) -> Tuple[Image2D, List[Band]]:
    if settings is None:
        settings = FilterSettings()
    bands = detect_bands(img, settings)
    denoised = correct_bands(img, bands, settings)
    return denoised, bands


def process_file(input_path: str, output_path: Optional[str] = None, settings: Optional[FilterSettings] = None) -> Tuple[str, int]:
    img = load_image(input_path)
    denoised, bands = denoise_image(img, settings)

    if not output_path:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_denoised{p.suffix}")

    save_image(denoised, output_path)
    return output_path, len(bands)
