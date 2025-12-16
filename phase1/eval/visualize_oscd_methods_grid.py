"""
Visualize OSCD tiles with Phase 1 unsupervised change detectors (score maps) side-by-side.

This script is meant for seminar-style qualitative comparison:
- Pre RGB
- Post RGB
- GT overlay (on pre)
- Score maps for key Phase 1 methods (loaded from a saved Phase 1 run):
  - pixel_diff (CVA is identical in this repo)
  - pca_diff
  - ds_projection
  - ds_cross_residual
  - celik
  - ir_mad

Typical usage (from repo root):
  .\\.venv\\Scripts\\python.exe -m phase1.eval.visualize_oscd_methods_grid --config phase1/configs/oscd_default.yaml --oscd_root data/OSCD --change_maps_root phase1/outputs/oscd_saved_full/oscd_change_maps --output_dir phase1/outputs/oscd_figs_all_methods --cities test
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

from phase1.data.oscd_dataset import OSCDEvaluatorDataset
from phase1.eval.utils import suppress_rasterio_warnings

suppress_rasterio_warnings()

PHASE1_ROOT = Path(__file__).resolve().parents[1]


def resolve_phase1_path(p: Path) -> Path:
  return p if p.is_absolute() else (PHASE1_ROOT / p)


def parse_args():
  ap = argparse.ArgumentParser()
  ap.add_argument("--config", required=True, type=Path, help="OSCD config YAML (for band order + nodata settings)")
  ap.add_argument("--oscd_root", required=True, type=Path, help="Path to OSCD root")
  ap.add_argument(
    "--change_maps_root",
    required=True,
    type=Path,
    help="Path to Phase 1 saved change maps root (e.g., phase1/outputs/oscd_saved_full/oscd_change_maps)",
  )
  ap.add_argument("--output_dir", required=True, type=Path, help="Where to save figures")
  ap.add_argument("--cities", default="test", help="Comma list of cities or one of: train|val|test|all")
  ap.add_argument(
    "--methods",
    default="pixel_diff,pca_diff,ds_projection,ds_cross_residual,celik,ir_mad",
    help="Comma list of methods to include (must match folder names under change_maps_root/<split>/)",
  )
  ap.add_argument("--dpi", type=int, default=150)
  return ap.parse_args()


def load_config(path: Path) -> dict:
  with path.open("r", encoding="utf-8") as f:
    return yaml.safe_load(f)


def robust_percentile_scale(arr: np.ndarray, valid_mask: Optional[np.ndarray] = None, p_low=2, p_high=98) -> np.ndarray:
  if valid_mask is not None:
    vals = arr[valid_mask]
  else:
    vals = arr.flatten()
  if vals.size == 0:
    return np.zeros_like(arr, dtype=np.float32)
  lo, hi = np.percentile(vals, [p_low, p_high])
  scaled = (arr - lo) / max(hi - lo, 1e-6)
  return np.clip(scaled, 0, 1).astype(np.float32)


def get_rgb(img: np.ndarray, band_order: List[str], valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
  idx = {b: i for i, b in enumerate(band_order)}
  r = img[idx["B04"]]
  g = img[idx["B03"]]
  b = img[idx["B02"]]
  rgb = np.stack([r, g, b], axis=-1)
  rgb = robust_percentile_scale(rgb, valid_mask=valid_mask)
  if valid_mask is not None:
    rgb = np.where(valid_mask[..., None], rgb, np.nan)
  return rgb


def normalize_map(m: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
  return robust_percentile_scale(m.astype(np.float32), valid_mask=valid_mask)


def load_score_map(change_maps_root: Path, split: str, method: str, city: str) -> Optional[np.ndarray]:
  p = change_maps_root / split / method / f"{city}_score.npy"
  if not p.exists():
    return None
  return np.load(p).astype(np.float32)


def method_title(method: str) -> str:
  if method == "pixel_diff":
    return "pixel_diff (CVA)"
  if method == "pca_diff":
    return "PCA-diff"
  if method == "ds_projection":
    return "DS projection"
  if method == "ds_cross_residual":
    return "DS cross-residual"
  if method == "celik":
    return "Celik (PCA+kmeans)"
  if method == "ir_mad":
    return "IR-MAD"
  return method


def resolve_city_list(cities_arg: str, split_map: Dict[str, OSCDEvaluatorDataset]) -> List[str]:
  key = cities_arg.lower()
  if key in ["train", "val", "test", "all"]:
    if key == "all":
      out: List[str] = []
      for d in split_map.values():
        out.extend(d.cities)
      return out
    return split_map[key].cities
  return [c.strip() for c in cities_arg.split(",") if c.strip()]


def main():
  args = parse_args()
  cfg = load_config(args.config)
  outdir = args.output_dir
  outdir.mkdir(parents=True, exist_ok=True)

  band_order = cfg["dataset"]["band_order"]
  nodata_value = cfg["dataset"].get("nodata_value", 0.0)
  min_valid_bands = cfg["dataset"].get("min_valid_bands", 3)

  # Datasets for split discovery + loading pre/post/GT
  ds_train = OSCDEvaluatorDataset(args.oscd_root, "train", band_order, nodata_value=nodata_value, min_valid_bands=min_valid_bands)
  ds_val = OSCDEvaluatorDataset(args.oscd_root, "val", band_order, nodata_value=nodata_value, min_valid_bands=min_valid_bands)
  ds_test = OSCDEvaluatorDataset(args.oscd_root, "test", band_order, nodata_value=nodata_value, min_valid_bands=min_valid_bands)
  split_map: Dict[str, OSCDEvaluatorDataset] = {"train": ds_train, "val": ds_val, "test": ds_test}

  target_cities = resolve_city_list(args.cities, split_map)

  # Build city -> split lookup
  city_to_split: Dict[str, Tuple[str, OSCDEvaluatorDataset]] = {}
  for split_name, dset in split_map.items():
    for c in dset.cities:
      city_to_split[c] = (split_name, dset)

  methods = [m.strip() for m in args.methods.split(",") if m.strip()]

  for city in target_cities:
    if city not in city_to_split:
      print(f"Skipping unknown city: {city}")
      continue
    split_name, dset = city_to_split[city]
    sample = dset.load_city(city)

    valid_mask = sample.valid_mask
    rgb_pre = get_rgb(sample.x_pre, band_order, valid_mask=valid_mask)
    rgb_post = get_rgb(sample.x_post, band_order, valid_mask=valid_mask)
    gt = (sample.y[0] > 0) if sample.y is not None else None

    # Layout: 3x3
    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    fig.suptitle(f"{city} ({split_name}) - Phase 1 unsupervised change detectors", fontsize=12)

    def imshow(ax, img, title, cmap=None, vmin=None, vmax=None):
      ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
      ax.set_title(title)
      ax.axis("off")

    # Row 1: pre/post/GT overlay
    imshow(axes[0, 0], rgb_pre, "Pre RGB")
    imshow(axes[0, 1], rgb_post, "Post RGB")
    if gt is not None:
      axes[0, 2].imshow(rgb_pre)
      axes[0, 2].imshow(gt, cmap="Reds", alpha=0.4)
      axes[0, 2].set_title("GT overlay")
      axes[0, 2].axis("off")
    else:
      imshow(axes[0, 2], rgb_pre, "GT overlay (none)")

    # Remaining 6 panels: methods (fill row-major order)
    panel_axes = [axes[1, 0], axes[1, 1], axes[1, 2], axes[2, 0], axes[2, 1], axes[2, 2]]
    for ax, method in zip(panel_axes, methods[:6]):
      score = load_score_map(args.change_maps_root, split_name, method, city)
      if score is None:
        ax.axis("off")
        ax.set_title(f"{method_title(method)} (missing)")
        continue
      score_viz = normalize_map(score, valid_mask=valid_mask)
      score_viz = np.ma.array(score_viz, mask=(~valid_mask))
      imshow(ax, score_viz, method_title(method), cmap="magma", vmin=0, vmax=1)

    # If more than 6 methods requested, warn (keeps figure readable)
    if len(methods) > 6:
      print(f"Note: requested {len(methods)} methods; only first 6 are shown in the 3x3 grid.")

    fig.tight_layout()
    out_path = outdir / f"{split_name}_{city}_methods.png"
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
  main()

