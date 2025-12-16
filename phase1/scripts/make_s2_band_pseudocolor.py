from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import rasterio


# Script lives in repo_root/phase1/phase1; PHASE1_ROOT is repo_root/phase1.
PHASE1_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE1_ROOT.parent
OSCD_IMAGES_ROOT = (
    REPO_ROOT
    / "data"
    / "OSCD"
    / "onera_satellite_change_detection dataset__images"
)

# OSCD canonical band order
BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B10",
    "B11",
    "B12",
    "B8A",
]


def _load_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    return arr


def _load_band_stats():
    stats_path = PHASE1_ROOT / "data" / "oscd_band_stats.json"
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    means = np.array(stats["mean"], dtype=np.float32)
    stds = np.array(stats["std"], dtype=np.float32)
    return means, stds


def _normalize_with_stats(arr: np.ndarray, band_idx: int, means, stds) -> np.ndarray:
    """Standardize using global stats, then map roughly [-3,3] -> [0,1]."""
    mean = means[band_idx]
    std = stds[band_idx]
    if std <= 0:
        std = 1.0
    z = (arr - mean) / std
    z = np.clip(z, -3.0, 3.0)
    norm = (z + 3.0) / 6.0
    return np.clip(norm, 0.0, 1.0)


def _band_base_color(band: str) -> np.ndarray:
    """Return base RGB color (0–1) reflecting band wavelength."""
    base_colors = {
        "B01": np.array([0.2, 0.4, 1.0]),  # coastal, deep blue
        "B02": np.array([0.3, 0.6, 1.0]),  # blue
        "B03": np.array([0.0, 0.9, 0.0]),  # green
        "B04": np.array([1.0, 0.0, 0.0]),  # red
        "B05": np.array([1.0, 0.6, 0.0]),  # orange (red edge)
        "B06": np.array([1.0, 0.3, 0.0]),  # orange-red (red edge)
        "B07": np.array([0.8, 0.0, 0.8]),  # purple (red edge)
        "B08": np.array([1.0, 0.0, 1.0]),  # magenta (NIR)
        "B8A": np.array([0.7, 0.0, 1.0]),  # purple (NIR)
        "B09": np.array([0.4, 0.0, 0.6]),  # deep violet (water vapor)
        "B10": np.array([0.5, 0.0, 0.7]),  # violet (cirrus)
        "B11": np.array([0.8, 0.4, 0.1]),  # warm brown (SWIR)
        "B12": np.array([0.6, 0.3, 0.1]),  # darker brown (SWIR)
    }
    return base_colors.get(band, np.array([0.5, 0.5, 0.5]))


def _read_dates(city: str):
    dates_path = OSCD_IMAGES_ROOT / city / "dates.txt"
    date1 = "date1"
    date2 = "date2"
    if dates_path.exists():
        with open(dates_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
        for line in lines:
            if line.startswith("date_1:"):
                date1 = line.split(":", 1)[1].strip()
            elif line.startswith("date_2:"):
                date2 = line.split(":", 1)[1].strip()
    return date1, date2


def make_pseudocolor_bands(city: str, which: str, output_dir: Path):
    """Generate wavelength-aware pseudocolor PNGs for one OSCD city and split (pre/post)."""
    assert which in {"pre", "post"}

    subdir = "imgs_1_rect" if which == "pre" else "imgs_2_rect"
    date1, date2 = _read_dates(city)
    date_str = date1 if which == "pre" else date2

    means, stds = _load_band_stats()

    city_dir = OSCD_IMAGES_ROOT / city / subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    for band_idx, band in enumerate(BANDS):
        tif_path = city_dir / f"{band}.tif"
        if not tif_path.exists():
            print(f"[WARN] Missing {tif_path}, skipping")
            continue

        arr = _load_band(tif_path)
        norm = _normalize_with_stats(arr, band_idx, means, stds)
        base = _band_base_color(band).reshape(1, 1, 3)
        rgb = norm[..., None] * base

        out_path = output_dir / f"sentinel2_{city}_{date_str}_{band}_pseudocolor.png"
        plt.imsave(out_path, rgb)
        print(f"Saved {out_path}")


def main():
    city = "brasilia"  # change to another OSCD city if needed
    out_root = PHASE1_ROOT / "docs" / "figs" / "oscd_band_pseudocolor_scientific"

    for which in ["pre", "post"]:
        make_pseudocolor_bands(city, which, out_root / which)


if __name__ == "__main__":
    main()
