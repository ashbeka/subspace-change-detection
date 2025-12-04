from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
import rasterio


# Script lives in repo_root/phase1/phase1; PHASE1_ROOT is repo_root/phase1.
PHASE1_ROOT = Path(__file__).resolve().parents[1]
OSCD_IMAGES_ROOT = (
    PHASE1_ROOT
    / "data"
    / "raw"
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
    """Load global mean/std per band from Phase-1 OSCD stats."""
    stats_path = PHASE1_ROOT / "data" / "oscd_band_stats.json"
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    means = np.array(stats["mean"], dtype=np.float32)
    stds = np.array(stats["std"], dtype=np.float32)
    return means, stds


def _normalize_with_stats(arr: np.ndarray, band_idx: int, means, stds) -> np.ndarray:
    """Z-score using global OSCD band stats, then map [-3,3] -> [0,1]."""
    mean = means[band_idx]
    std = stds[band_idx]
    if std <= 0:
        std = 1.0
    z = (arr - mean) / std
    z = np.clip(z, -3.0, 3.0)
    out = (z + 3.0) / 6.0
    return out


def save_city_bands(city: str):
    means, stds = _load_band_stats()

    city_dir = OSCD_IMAGES_ROOT / city
    out_dir = PHASE1_ROOT / "docs" / "figs" / "oscd_band_pngs" / city
    out_dir.mkdir(parents=True, exist_ok=True)

    for which, subdir in [("pre", "imgs_1_rect"), ("post", "imgs_2_rect")]:
        band_dir = city_dir / subdir
        for band_idx, band in enumerate(BANDS):
            tif_path = band_dir / f"{band}.tif"
            if not tif_path.exists():
                print(f"[WARN] Missing {tif_path}, skipping")
                continue
            arr = _load_band(tif_path)
            norm = _normalize_with_stats(arr, band_idx, means, stds)
            out_path = out_dir / f"{city}_{which}_{band}.png"
            plt.imsave(out_path, norm, cmap="gray")
            print(f"Saved {out_path}")


def main():
    for city in ["brasilia", "chongqing"]:
        print(f"Processing {city}...")
        save_city_bands(city)


if __name__ == "__main__":
    main()
