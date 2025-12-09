import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def show_rgb(path: Path, title: str):
    """Display a MultiSenGE S2 patch as RGB (B04,B03,B02) with simple contrast stretch."""
    with rasterio.open(path) as src:
        b2 = src.read(1)   # B02
        b3 = src.read(2)   # B03
        b4 = src.read(3)   # B04
    rgb = np.stack([b4, b3, b2], axis=-1).astype(float)

    # simple contrast stretch (2–98th percentiles)
    lo, hi = np.percentile(rgb[~np.isnan(rgb)], (2, 98))
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)

    plt.figure(figsize=(5, 5))
    plt.imshow(rgb)
    plt.axis("off")
    plt.title(title)
    plt.show()


def parse_args():
    ap = argparse.ArgumentParser(description="Quick RGB preview of a MultiSenGE S2 TIFF (band order B02,B03,B04).")
    ap.add_argument("tiff_path", type=Path, help="Path to a MultiSenGE S2 TIFF.")
    ap.add_argument("title", nargs="?", default=None, help="Optional title for the preview window.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    title = args.title or args.tiff_path.name
    show_rgb(args.tiff_path, title)
