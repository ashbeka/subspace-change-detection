"""
Generate OSCD geodesic prior maps (Phase 1b).

This exports a per-pixel geodesic-distance map computed from local PCA subspaces
on sliding windows, saved in the same `oscd_change_maps/{split}/{method}/...`
structure that Phase 2 uses as extra channels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from phase1.data.oscd_dataset import OSCDEvaluatorDataset
from phase1.data.preprocessing import apply_normalization, load_band_stats
from phase1.eval.thresholding import apply_threshold, otsu_threshold
from phase1.priors.geodesic_priors import GeodesicPriorConfig, sliding_window_geodesic_prior


def parse_args():
  ap = argparse.ArgumentParser()
  ap.add_argument("--config", type=Path, required=True)
  ap.add_argument("--oscd_root", type=Path, required=True)
  ap.add_argument("--output_dir", type=Path, required=True)
  ap.add_argument("--max_cities", type=int, default=0, help="Optional cap per split (0 = all).")
  return ap.parse_args()


def load_config(path: Path) -> dict:
  return yaml.safe_load(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> None:
  path.mkdir(parents=True, exist_ok=True)


def _save_change_map(base_dir: Path, split: str, method: str, city: str, score: np.ndarray, valid_mask: np.ndarray) -> None:
  out_dir = base_dir / split / method
  _ensure_dir(out_dir)
  np.save(out_dir / f"{city}_score.npy", score.astype(np.float32))
  thr = otsu_threshold(score, valid_mask)
  mask = apply_threshold(score, thr)
  img = Image.fromarray((mask.astype(np.uint8) * 255))
  img.save(out_dir / f"{city}_mask.png")


def main():
  args = parse_args()
  cfg = load_config(args.config)

  band_order = cfg["dataset"].get("band_order") or [
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

  stats = load_band_stats(Path("phase1") / Path(cfg["normalization"]["stats_path"]))
  base_maps = Path(args.output_dir) / "oscd_change_maps"
  method_name = cfg.get("geodesic", {}).get("method_name", "geodesic_dist")

  window_cfg = cfg.get("geodesic", {}).get("window", {}) or {}
  window_size = int(window_cfg.get("size", 64))
  stride = int(window_cfg.get("stride", 32))
  aggregator = str(window_cfg.get("aggregator", "mean"))

  prior_cfg = GeodesicPriorConfig(
    rank_r=int(cfg["subspace"].get("rank_r", 6)),
    variance_threshold=cfg["subspace"].get("variance_threshold"),
    use_randomized_pca=bool(cfg["subspace"].get("use_randomized_pca", True)),
    random_state=int(cfg["subspace"].get("random_state", cfg.get("seed", 1234))),
    seed=int(cfg.get("seed", 1234)),
    pixel_sample=int(cfg["subspace"].get("pixel_sample", 0) or 0),
    nodata_value=float(cfg["dataset"].get("nodata_value", 0.0)),
    min_valid_bands=int(cfg["dataset"].get("min_valid_bands", 3)),
    score_normalization=cfg["geodesic"].get("score_normalization", "percentile_99"),
    percentile=float(cfg["geodesic"].get("percentile", 99.0)),
  )

  val_cities = cfg["dataset"].get("val_cities", [])
  val_from_train = int(cfg["dataset"].get("val_from_train", 0) or 0)
  for split in ("train", "val", "test"):
    ds = OSCDEvaluatorDataset(
      args.oscd_root,
      split,
      band_order,
      nodata_value=float(cfg["dataset"].get("nodata_value", 0.0)),
      min_valid_bands=int(cfg["dataset"].get("min_valid_bands", 3)),
      stats_path=None,
      val_cities=val_cities,
      val_from_train=val_from_train,
    )
    cities = ds.cities[: args.max_cities] if args.max_cities and args.max_cities > 0 else ds.cities
    for city in cities:
      sample = ds.load_city(city)
      x1n, vm = apply_normalization(sample.x_pre, stats, valid_mask=sample.valid_mask, nodata_value=prior_cfg.nodata_value)
      x2n, _ = apply_normalization(sample.x_post, stats, valid_mask=sample.valid_mask, nodata_value=prior_cfg.nodata_value)
      start = time.perf_counter()
      out = sliding_window_geodesic_prior(
        x1n,
        x2n,
        window_size=window_size,
        stride=stride,
        aggregator=aggregator,
        metric=str(cfg.get("geodesic", {}).get("metric", "geodesic_dist")),
        cfg=prior_cfg,
      )
      elapsed = time.perf_counter() - start
      _save_change_map(base_maps, split, method_name, city, out["score"], out["valid_mask"])
      print(f"[{split}] {city}: saved {method_name} in {elapsed:.2f}s")

  meta = {"config": str(args.config), "oscd_root": str(args.oscd_root), "output_dir": str(args.output_dir)}
  (Path(args.output_dir) / "run_metadata_geodesic.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
  main()

