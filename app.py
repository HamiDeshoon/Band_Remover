"""
DeStripe Application Main Entry Point.
Automatically dispatches to GUI (if PyQt5/Tkinter display available) or CLI mode.
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

def run_cli(args):
    print("DeStripe - Band Artifact Remover (Offline CLI)")
    import core

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    settings = core.FilterSettings(
        detection_sensitivity=args.sensitivity,
        border_exclude_ratio=args.border_exclude,
        interpolation_method=args.interp,
        gaussian_sigma=args.sigma,
    )

    print(f"Processing image: {input_path} ...")
    out_path, count = core.process_file(input_path, args.output, settings)
    print(f"Done! Detected & corrected {count} band(s).")
    print(f"Saved output to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="DeStripe: Band Artifact Remover from X-Ray images")
    parser.add_argument("--cli", action="store_true", help="Force CLI mode")
    parser.add_argument("-i", "--input", type=str, help="Path to input image (PGM, BMP, CSV)")
    parser.add_argument("-o", "--output", type=str, help="Path to output image")
    parser.add_argument("--sensitivity", type=float, default=1.5, help="Detection sensitivity (default: 1.5)")
    parser.add_argument("--border-exclude", type=float, default=0.15, help="Border exclude ratio (default: 0.15)")
    parser.add_argument("--interp", choices=["pchip", "linear"], default="pchip", help="Interpolation method (pchip/linear)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian smoothing sigma (default: 1.0)")

    args = parser.parse_args()

    if args.cli or args.input:
        if not args.input:
            parser.error("--input / -i is required in CLI mode")
        run_cli(args)
        return

    # GUI Mode
    try:
        from gui.destripe_app import main as gui_main
        gui_main()
    except Exception as e:
        print(f"PyQt5 GUI unavailable ({e}). Attempting pure-Python CLI/GUI...")
        if args.input:
            run_cli(args)
        else:
            print("Usage: python main.py --input <image_path> [options]")
            sys.exit(1)

if __name__ == "__main__":
    main()
