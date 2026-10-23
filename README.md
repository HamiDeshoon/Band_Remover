# Band_Remover
Band Artifact remover from X-Ray images

## Compatibility & System Requirements
- **Target OS**: Windows 7 / 8 / 10 / 11 (64-bit) & Linux / macOS
- **Target Python Version**: Python 3.8
- **GUI Engine**: PyQt5 (Qt5 framework compatible with Windows 7)

## Installation & Environment Setup

To run DeStripe with Python 3.8:

```bash
# Install dependencies from requirements.txt:
pip install -r requirements.txt
```

## Offline / Self-Contained Run Instructions

### Overview
This project is fully self-contained and operates completely offline without needing internet access, remote models, external network APIs, or additional package installations. It includes both a pure-Python zero-dependency engine (for portable image formats like PGM, BMP, CSV) and full GUI support powered by PyQt5 and PyTorch.

### Direct Executable Execution
A pre-compiled standalone executable is provided in the repository under `dist/DeStripe/DeStripe` (or `dist/DeStripe/DeStripe.exe` on Windows). You can launch the application directly without needing Python or external dependencies installed:

```bash
# Launch GUI application:
./dist/DeStripe/DeStripe

# Run via CLI:
./dist/DeStripe/DeStripe --cli -i samples/sample_xray.bmp -o output_clean.bmp
```

### Python Script Entry Points
Alternatively, the application can be run directly using Python 3.8 via `main.py`, `run.py`, `app.py`, or `run_gui.bat`:

```bash
# Launch GUI (if PyQt5 is installed):
python main.py

# Launch GUI via Windows Batch Script:
run_gui.bat

# Run CLI mode (works on standard Python with zero dependencies):
python main.py -i samples/sample_xray.pgm -o output_clean.pgm
```

### Expected Input Format
- **PGM** (P2 ASCII or P5 Binary grayscale)
- **BMP** (8-bit grayscale or 24-bit RGB)
- **CSV / TXT** (Comma-separated 2D numeric grid representing pixel intensities)
- **PNG / JPG / TIFF / DICOM** (Supported when running with PyQt5 / PIL / pydicom)

### Output Format
- Processed grayscale image matching the input format with band artifacts removed.

### Local Dependencies Included
- `core.py`: Pure Python 1D profile baseline analysis, statistical deviation thresholding, PCHIP/linear interpolation, and Gaussian smoothing.
- `destripe_best_eff.ckpt`: Bundled PyTorch model checkpoint for deep learning SMP U-Net / EfficientNet-b0 inference.
- `requirements.txt`: Environment package dependencies for Python 3.8 and Windows 7 compatibility.
- `dist/DeStripe/`: Pre-built standalone offline executable distribution.

### How the App Works Offline
1. The app reads the input image into memory.
2. It calculates 1D row and column baseline profiles using median filtering.
3. Statistical deviation analysis detects contiguous artifact band regions.
4. Neighboring unaffected pixel contexts are interpolated through the band region using PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) or linear interpolation.
5. Orthogonal Gaussian filtering smooths the corrected region.
6. The cleaned image is written back to disk.

### Limitations
- Deep learning neural inference requires PyTorch and the bundled model checkpoint (`destripe_best_eff.ckpt`). If PyTorch is not available, the app falls back to the pure-Python classical filter engine automatically.
