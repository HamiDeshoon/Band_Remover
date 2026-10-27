"""
Image I/O utilities for DeStripe GUI.
Supports PNG/JPG/BMP/TIFF (via PIL) and DICOM (via pydicom).
Handles 8-bit and 16-bit images with full dynamic range preservation.
"""
from __future__ import annotations
import os
import copy
from pathlib import Path
from typing import Any, Optional, List

import numpy as np

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VALID_EXTS = VALID_IMAGE_EXTS | {".dcm", ".dicom"}


def is_dicom_file(path: str | Path) -> bool:
    """Return True if the file is a valid DICOM file, even without a .dcm extension."""
    try:
        p = Path(path)
        if not p.is_file():
            return False
        # DICOM Part 10 has a 128-byte preamble followed by "DICM"
        if p.stat().st_size > 132:
            with open(p, "rb") as f:
                f.seek(128)
                if f.read(4) == b"DICM":
                    return True
        # DICOM files without preamble (raw ACR-NEMA or forced header)
        import pydicom
        ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        return hasattr(ds, "SOPClassUID") or hasattr(ds, "Modality") or hasattr(ds, "PixelData")
    except Exception:
        return False


def is_supported_file(path: str | Path) -> bool:
    """Return True if path is a supported image or DICOM file."""
    p = Path(path)
    if not p.is_file():
        return False
    if p.suffix.lower() in VALID_EXTS:
        return True
    return is_dicom_file(p)


def _read_pil(path: str) -> tuple[np.ndarray, dict]:
    from PIL import Image
    with Image.open(path) as im:
        raw = np.array(im)
        if raw.ndim == 3:
            raw = np.array(im.convert("L"))
        if raw.dtype in (np.uint16, np.int16, np.int32):
            # 16-bit image: preserve raw 16-bit and create normalized 8-bit display
            p1, p99 = np.percentile(raw, (1, 99))
            span = max(float(p99 - p1), 1.0)
            display = np.clip((raw.astype(np.float64) - p1) / span * 255.0, 0, 255).astype(np.uint8)
            return display, {"type": "pil", "path": path, "pixel_array": raw}
        else:
            arr = raw.astype(np.uint8) if raw.dtype == np.uint8 else np.array(im.convert("L"), dtype=np.uint8)
            return arr, {"type": "pil", "path": path, "pixel_array": arr}


def _apply_window(pixel_array: np.ndarray, ds: Any) -> np.ndarray:
    """Apply DICOM window center/width to produce uint8 display image."""
    arr = pixel_array.astype(np.float64)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    arr = arr * slope + intercept

    wc = None
    ww = None
    try:
        raw_wc = getattr(ds, "WindowCenter", None)
        raw_ww = getattr(ds, "WindowWidth", None)
        if raw_wc is not None:
            wc = float(raw_wc[0] if hasattr(raw_wc, "__iter__") and not isinstance(raw_wc, (str, bytes)) else raw_wc)
        if raw_ww is not None:
            ww = float(raw_ww[0] if hasattr(raw_ww, "__iter__") and not isinstance(raw_ww, (str, bytes)) else raw_ww)
    except Exception:
        pass

    if wc is None or ww is None or ww <= 0:
        p1 = float(np.percentile(arr, 1))
        p99 = float(np.percentile(arr, 99))
        if p99 > p1:
            lo, hi = p1, p99
        else:
            lo, hi = float(arr.min()), float(arr.max() if arr.max() > arr.min() else arr.min() + 1)
    else:
        lo = wc - ww / 2.0
        hi = wc + ww / 2.0

    denom = max(hi - lo, 1e-5)
    disp = np.clip((arr - lo) / denom * 255.0, 0, 255).astype(np.uint8)

    photo = getattr(ds, "PhotometricInterpretation", "").strip().upper()
    if photo == "MONOCHROME1":
        disp = (255 - disp).astype(np.uint8)

    return disp


def _read_dicom(path: str) -> tuple[np.ndarray, dict]:
    import pydicom
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=False)
    except Exception:
        ds = pydicom.dcmread(path, stop_before_pixels=False, force=True)

    px = ds.pixel_array
    if px.ndim == 3:                  # multi-frame or RGB
        px = px[px.shape[0] // 2] if px.ndim == 3 else px
    display = _apply_window(px, ds)
    meta = {
        "type": "dicom",
        "path": path,
        "dataset": ds,
        "pixel_array": px,
    }
    return display, meta


def read_image(path: str) -> tuple[np.ndarray, dict]:
    """Return (uint8 HxW grayscale array, metadata dict).
    Correctly recognizes DICOM files even without a .dcm extension.
    """
    ext = Path(path).suffix.lower()
    if ext in {".dcm", ".dicom"} or is_dicom_file(path):
        return _read_dicom(path)
    return _read_pil(path)


def save_image(
    denoised: np.ndarray,
    src_path: str,
    meta: dict,
    bands: Optional[List[Any]] = None,
    settings: Optional[Any] = None,
) -> str:
    """Save denoised array with _v1 suffix in the same directory / format as source.
    Ensures that exported files always have valid file extensions (.dcm for DICOM).
    Preserves full 16-bit dynamic range when the original image is 16-bit.
    Returns the absolute path of the saved file.
    """
    src = Path(src_path)
    if meta.get("type") == "dicom":
        ext = src.suffix if src.suffix.lower() in {".dcm", ".dicom"} else ".dcm"
        stem = src.stem if src.suffix else src.name
        out_path = src.parent / f"{stem}_v1{ext}"
        _save_dicom(denoised, out_path, meta, bands=bands, settings=settings)
    else:
        ext = src.suffix if src.suffix.lower() in VALID_IMAGE_EXTS else ".png"
        stem = src.stem if src.suffix else src.name
        out_path = src.parent / f"{stem}_v1{ext}"
        _save_pil(denoised, out_path, meta, bands=bands, settings=settings)

    return str(out_path)


def _save_dicom(
    denoised_display: np.ndarray,
    out_path: Path,
    meta: dict,
    bands: Optional[List[Any]] = None,
    settings: Optional[Any] = None,
) -> None:
    """Write denoised pixel data back into a copy of the DICOM dataset, preserving 16-bit depth."""
    import pydicom
    from gui.classical_filter import correct_bands

    ds = copy.deepcopy(meta["dataset"])
    orig_px = meta["pixel_array"]

    # If classical bands are identified and original is 16-bit, interpolate directly on 16-bit raw array
    if bands and len(bands) > 0 and orig_px.dtype in (np.uint16, np.int16, np.int32, np.float32, np.float64):
        restored_raw = correct_bands(orig_px, bands, settings)
    else:
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        wc = None
        ww = None
        try:
            raw_wc = getattr(ds, "WindowCenter", None)
            raw_ww = getattr(ds, "WindowWidth", None)
            if raw_wc is not None:
                wc = float(raw_wc[0] if hasattr(raw_wc, "__iter__") and not isinstance(raw_wc, (str, bytes)) else raw_wc)
            if raw_ww is not None:
                ww = float(raw_ww[0] if hasattr(raw_ww, "__iter__") and not isinstance(raw_ww, (str, bytes)) else raw_ww)
        except Exception:
            pass

        if wc is None or ww is None or ww <= 0:
            lo = float(orig_px.min())
            hi = float(orig_px.max() if orig_px.max() > orig_px.min() else orig_px.min() + 1)
        else:
            lo = wc - ww / 2.0
            hi = wc + ww / 2.0

        orig_f = orig_px.astype(np.float64)
        denom = max(hi - lo, 1e-5)
        orig_disp = np.clip((orig_f * slope + intercept - lo) / denom * 255.0, 0, 255)
        # Apply display delta so unaffected pixels remain 100% untouched at 16-bit fidelity
        delta_disp = denoised_display.astype(np.float64) - orig_disp
        delta_raw = (delta_disp / 255.0 * denom) / (slope if slope != 0 else 1.0)
        restored_raw = orig_f + delta_raw

    orig_dtype = orig_px.dtype if orig_px.dtype in (np.int16, np.uint16, np.int32) else np.int16
    info = np.iinfo(orig_dtype)
    ds.PixelData = np.clip(np.round(restored_raw), info.min, info.max).astype(orig_dtype).tobytes()
    ds.Rows, ds.Columns = denoised_display.shape[:2]
    try:
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.DerivationDescription = "DeStripe v1 — band artifact removal"
    except Exception:
        pass
    ds.save_as(str(out_path))


def _save_pil(
    denoised_display: np.ndarray,
    out_path: Path,
    meta: dict,
    bands: Optional[List[Any]] = None,
    settings: Optional[Any] = None,
) -> None:
    from PIL import Image
    from gui.classical_filter import correct_bands

    orig_px = meta.get("pixel_array")
    if orig_px is not None and orig_px.dtype in (np.uint16, np.int16) and bands and len(bands) > 0:
        corrected_16 = correct_bands(orig_px, bands, settings)
        Image.fromarray(corrected_16).save(str(out_path))
    else:
        Image.fromarray(denoised_display.astype(np.uint8), mode="L").save(str(out_path))
