#!/usr/bin/env python3
"""
Stack frames.png (top) and raster.png (bottom) into a single header image,
matching the layout of fig_data_overview panel (a). Saves to fig/header.png.

Run from repo root:
  python data_loading/combine_header_images.py

Expects data_loading/frames.png and data_loading/raster.png (or pass paths as args).
"""
import argparse
import sys
from pathlib import Path

# Project root
SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Combine frames.png and raster.png into header.png")
    parser.add_argument("--frames", type=Path, default=SCRIPT_DIR / "frames.png", help="Path to frames image")
    parser.add_argument("--raster", type=Path, default=SCRIPT_DIR / "raster.png", help="Path to raster image")
    parser.add_argument("--out", type=Path, default=SCRIPT_DIR / "fig" / "header.png", help="Output path")
    parser.add_argument("--max-width", type=int, default=500, help="Max width in px (default: 500 for README); use 0 to keep full size")
    args = parser.parse_args()

    try:
        from PIL import Image
    except ImportError:
        sys.exit("Install Pillow: pip install Pillow")

    if not args.frames.exists():
        sys.exit(f"Frames image not found: {args.frames}")
    if not args.raster.exists():
        sys.exit(f"Raster image not found: {args.raster}")

    top = Image.open(args.frames).convert("RGB")
    bottom = Image.open(args.raster).convert("RGB")

    # Same width: use the wider of the two and optionally resize the other to match
    w = max(top.width, bottom.width)
    if top.width != w:
        top = top.resize((w, int(top.height * w / top.width)), Image.Resampling.LANCZOS)
    if bottom.width != w:
        bottom = bottom.resize((w, int(bottom.height * w / bottom.width)), Image.Resampling.LANCZOS)

    # Stack: frames on top, raster below
    combined = Image.new("RGB", (w, top.height + bottom.height))
    combined.paste(top, (0, 0))
    combined.paste(bottom, (0, top.height))

    # Scale down for README if requested (use --max-width 0 to keep full size)
    if args.max_width > 0 and w > args.max_width:
        scale = args.max_width / w
        new_h = int(combined.height * scale)
        combined = combined.resize((args.max_width, new_h), Image.Resampling.LANCZOS)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
