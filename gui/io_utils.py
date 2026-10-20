"""
Image I/O utilities for DeStripe GUI.
Supports PNG/JPG/BMP/TIFF (via PIL) and DICOM (via pydicom).
All operations return / accept uint8 HxW grayscale numpy arrays.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

import numpy as np

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VALID_EXTS = VALID_IMAGE_EXTS | {".dcm", ".dicom"}


def _read_pil(path: str) -> tuple[np.ndarray, dict]:
    from PIL import Image
    with Image.open(path) as im:
        arr = np.array(im.convert("L"), dtype=np.uint8)
    return arr, {"type": "pil", "path": path}


def _apply_window(pixel_array: np.ndarray, ds: Any) -> np.ndarray:
    """Apply DICOM window center/width to produce uint8 display image."""
    arr = pixel_array.astype(np.float64)
    # Rescale slope/intercept
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    arr = arr * slope + intercept

    # Try window center/width from dataset
    try:
        wc = float(ds.WindowCenter) if not hasattr(ds.WindowCenter, "__iter__") else float(ds.WindowCenter[0])
        ww = float(ds.WindowWidth)  if not hasattr(ds.WindowWidth,  "__iter__") else float(ds.WindowWidth[0])
    except Exception:
        wc = float(arr.mean())
        ww = float(arr.std() * 4 + 1)

    lo = wc - ww / 2
    hi = wc + ww / 2
    arr = np.clip((arr - lo) / (hi - lo) * 255, 0, 255)
    return arr.astype(np.uint8)


def _read_dicom(path: str) -> tuple[np.ndarray, dict]:
    import pydicom
    ds = pydicom.dcmread(path, stop_before_pixels=False)
    px = ds.pixel_array
    if px.ndim == 3:                  # multi-frame or RGB
        px = px[px.shape[0] // 2] if px.ndim == 3 else px
    display = _apply_window(px, ds)
    meta = {"type": "dicom", "path": path, "dataset": ds, "pixel_array": px}
    return display, meta


def read_image(path: str) -> tuple[np.ndarray, dict]:
    """Return (uint8 HxW grayscale array, metadata dict).
    metadata always contains at least 'type' and 'path'.
    """
    ext = Path(path).suffix.lower()
    if ext in {".dcm", ".dicom"}:
        return _read_dicom(path)
    return _read_pil(path)


def save_image(denoised: np.ndarray, src_path: str, meta: dict) -> str:
    """Save denoised array with _v1 suffix in the same directory / format as source.
    Returns the absolute path of the saved file.
    """
    src = Path(src_path)
    out_path = src.parent / f"{src.stem}_v1{src.suffix}"

    if meta.get("type") == "dicom":
        _save_dicom(denoised, out_path, meta)
    else:
        from PIL import Image
        Image.fromarray(denoised.astype(np.uint8), mode="L").save(str(out_path))

    return str(out_path)


def _save_dicom(denoised_display: np.ndarray, out_path: Path, meta: dict) -> None:
    """Write denoised pixel data back into a copy of the DICOM dataset."""
    import pydicom, copy
    ds = copy.deepcopy(meta["dataset"])
    orig_px = meta["pixel_array"].astype(np.float64)

    # Re-map uint8 display back to the original HU domain linearly
    slope  = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    try:
        wc = float(ds.WindowCenter) if not hasattr(ds.WindowCenter, "__iter__") else float(ds.WindowCenter[0])
        ww = float(ds.WindowWidth)  if not hasattr(ds.WindowWidth,  "__iter__") else float(ds.WindowWidth[0])
    except Exception:
        wc = float(orig_px.mean())
        ww = float(orig_px.std() * 4 + 1)

    lo = wc - ww / 2
    hi = wc + ww / 2
    # denoised_display is float array 0-255 from the model; convert back
    restored = denoised_display.astype(np.float64) / 255.0 * (hi - lo) + lo
    # Undo rescale to raw stored pixel values
    restored_raw = (restored - intercept) / slope
    # Round and clip to original dtype range
    orig_dtype = orig_px.dtype if orig_px.dtype in (np.int16, np.uint16, np.int32) else np.int16
    info = np.iinfo(orig_dtype)
    ds.PixelData = np.clip(np.round(restored_raw), info.min, info.max).astype(orig_dtype).tobytes()
    ds.Rows, ds.Columns = denoised_display.shape[:2]
    # Update UID to mark as derived
    try:
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.DerivationDescription = "DeStripe v1 — band artifact removal"
    except Exception:
        pass
    ds.save_as(str(out_path))
