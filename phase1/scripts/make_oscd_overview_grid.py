import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


# Script lives in phase1/scripts; jump to phase1 root for data paths.
PHASE1_ROOT = Path(__file__).resolve().parents[1]

OSCD_IMAGES_ROOT = PHASE1_ROOT / "data" / "raw" / "OSCD" / "onera_satellite_change_detection dataset__images"
OSCD_TRAIN_LABELS_ROOT = PHASE1_ROOT / "data" / "raw" / "OSCD" / "onera_satellite_change_detection dataset__train_labels"
OSCD_TEST_LABELS_ROOT = PHASE1_ROOT / "data" / "raw" / "OSCD" / "onera_satellite_change_detection dataset__test_labels"


def _load_rgb(city: str, which: str):
    """Load pre or post RGB image (B04,B03,B02) for a given OSCD city.

    which: 'pre' or 'post'.
    """
    if which == "pre":
        subdir = "imgs_1_rect"
    elif which == "post":
        subdir = "imgs_2_rect"
    else:
        raise ValueError("which must be 'pre' or 'post'")

    city_dir = OSCD_IMAGES_ROOT / city / subdir
    # OSCD band names
    b2_path = city_dir / "B02.tif"
    b3_path = city_dir / "B03.tif"
    b4_path = city_dir / "B04.tif"

    with rasterio.open(b4_path) as src4, rasterio.open(b3_path) as src3, rasterio.open(
        b2_path
    ) as src2:
        b4 = src4.read(1).astype(np.float32)
        b3 = src3.read(1).astype(np.float32)
        b2 = src2.read(1).astype(np.float32)

    rgb = np.stack([b4, b3, b2], axis=-1)

    # Simple percentile-based contrast stretch per tile
    valid = np.isfinite(rgb)
    if not np.any(valid):
        return np.zeros_like(rgb, dtype=np.float32)

    lo, hi = np.percentile(rgb[valid], (2, 98))
    if hi <= lo:
        hi = lo + 1.0
    rgb = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)
    return rgb


def _load_gt(city: str):
    """Load binary GT mask for a given OSCD city as 0/1 float array."""
    # Train vs test labels live in different roots; pick the one that exists.
    train_dir = OSCD_TRAIN_LABELS_ROOT / city / "cm"
    test_dir = OSCD_TEST_LABELS_ROOT / city / "cm"

    label_dir = train_dir if train_dir.exists() else test_dir

    png_path = label_dir / "cm.png"
    tif_path = next(label_dir.glob("*.tif"), None)

    if png_path.exists():
        path = png_path
    elif tif_path is not None:
        path = tif_path
    else:
        raise FileNotFoundError(f"No GT mask found for city {city}")

    with rasterio.open(path) as src:
        mask = src.read(1)

    # OSCD masks use 0 for background and 255 (or >0) for change.
    mask_bin = (mask > 0).astype(np.float32)
    return mask_bin


def _load_all_cities():
    """Load ordered list of OSCD city names from all.txt."""
    all_txt = OSCD_IMAGES_ROOT / "all.txt"
    with open(all_txt, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    return [c.strip() for c in line.split(",") if c.strip()]


def make_oscd_overview_grid(cities, output_path: Path):
    n_rows = len(cities)
    n_cols = 3  # pre, post, GT

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 3.0, n_rows * 2.0),
        squeeze=False,
    )

    for i, city in enumerate(cities):
        pre_rgb = _load_rgb(city, "pre")
        post_rgb = _load_rgb(city, "post")
        gt = _load_gt(city)

        # Pre
        ax = axes[i, 0]
        ax.imshow(pre_rgb)
        ax.set_axis_off()
        if i == 0:
            ax.set_title("Pre RGB", fontsize=10)
        ax.set_ylabel(city, fontsize=8)

        # Post
        ax = axes[i, 1]
        ax.imshow(post_rgb)
        ax.set_axis_off()
        if i == 0:
            ax.set_title("Post RGB", fontsize=10)

        # GT mask
        ax = axes[i, 2]
        ax.imshow(gt, cmap="gray")
        ax.set_axis_off()
        if i == 0:
            ax.set_title("GT change mask", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    all_cities = _load_all_cities()

    # Full 24x3 grid
    out_full = PHASE1_ROOT / "docs" / "figs" / "oscd_24x3_overview.png"
    make_oscd_overview_grid(all_cities, out_full)
    print(f"Saved OSCD overview grid to {out_full}")

    # Last 5 cities only
    last5 = all_cities[-5:]
    out_last5 = PHASE1_ROOT / "docs" / "figs" / "oscd_last5_overview.png"
    make_oscd_overview_grid(last5, out_last5)
    print(f"Saved OSCD last-5 overview grid to {out_last5}")
