"""
Run temporal/geodesic analysis on MultiSenGE S2 time series.

Outputs per-patch time-series plots for:
- geodesic "velocity" between consecutive subspaces
- 2nd-order DS "acceleration" magnitude over triples
- optional along/orth decomposition using geodesic projection ω
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml

from phase1.data.multisenge_manifest import load_manifest
from phase1.data.preprocessing import apply_normalization, build_valid_mask, load_band_stats, vectorize_cube
from phase1.ds.pca_utils import fit_pca_basis
from phase1.eval.utils import suppress_rasterio_warnings
from phase1.subspace.geodesic import grassmann_geodesic_distance, subspace_magnitude
from phase1.subspace.second_order_ds import second_order_difference_subspace

suppress_rasterio_warnings()

PHASE1_ROOT = Path(__file__).resolve().parents[1]


def resolve_phase1_path(p: Path) -> Path:
  return p if p.is_absolute() else (PHASE1_ROOT / p)


def parse_args():
  ap = argparse.ArgumentParser()
  ap.add_argument("--config", required=True, type=Path)
  ap.add_argument("--multisenge_root", required=True, type=Path, help="Path to data/MultiSenGE (contains s2/, labels/, ...).")
  ap.add_argument("--manifest", required=True, type=Path, help="Manifest JSON produced by phase1.scripts.build_multisenge_manifest.")
  ap.add_argument("--output_dir", required=True, type=Path)
  ap.add_argument("--max_patches", type=int, default=0, help="Optional cap on number of patches from the manifest.")
  return ap.parse_args()


def _load_config(path: Path) -> dict:
  return yaml.safe_load(path.read_text(encoding="utf-8"))


def _subset_stats(stats, stats_order, desired_order):
  idx_map = {b: i for i, b in enumerate(stats_order)}
  mean = []
  std = []
  for b in desired_order:
    if b not in idx_map:
      raise ValueError(f"Band {b} not found in stats_order")
    mean.append(stats.mean[idx_map[b]])
    std.append(stats.std[idx_map[b]])
  from phase1.data.preprocessing import BandStats

  return BandStats(mean=np.array(mean, dtype=np.float32), std=np.array(std, dtype=np.float32), eps=stats.eps)


def _load_tif(path: Path) -> np.ndarray:
  with rasterio.open(path) as src:
    return src.read().astype(np.float32)


def _load_ground_reference(multisenge_root: Path, tile: str, x: int, y: int) -> Optional[np.ndarray]:
  fp = Path(multisenge_root) / "ground_reference" / f"{tile}_GR_{x}_{y}.tif"
  if not fp.exists():
    return None
  with rasterio.open(fp) as src:
    return src.read(1)


def _majority_label(mask: np.ndarray) -> Optional[int]:
  if mask.size == 0:
    return None
  vals, counts = np.unique(mask, return_counts=True)
  if vals.size == 0:
    return None
  return int(vals[int(np.argmax(counts))])


def _parse_date(relpath: str) -> str:
  name = Path(relpath).name
  parts = name.split("_")
  if len(parts) < 2:
    return "unknown"
  return parts[1]


def _plot_timeseries(
  dates: List[str],
  vel_geo: List[float],
  vel_mag: List[float],
  acc_mag: List[float],
  acc_along: Optional[List[float]],
  acc_orth: Optional[List[float]],
  out_path: Path,
  title: str,
) -> None:
  out_path.parent.mkdir(parents=True, exist_ok=True)
  xs = [datetime.strptime(d, "%Y%m%d") for d in dates]

  fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
  ax0, ax1 = axes

  if vel_geo:
    x_mid = xs[1:]
    ax0.plot(x_mid, vel_geo, label="velocity: geodesic distance ρ", linewidth=2)
  if vel_mag:
    x_mid = xs[1:]
    ax0.plot(x_mid, vel_mag, label="velocity: magnitude Mag", linewidth=2, alpha=0.8)
  ax0.set_ylabel("Velocity")
  ax0.grid(True, alpha=0.3)
  ax0.legend(loc="best")

  if acc_mag:
    x_acc = xs[1:-1]
    ax1.plot(x_acc, acc_mag, label="acceleration: Mag(D2)", linewidth=2, color="tab:red")
    if acc_along is not None:
      ax1.plot(x_acc, acc_along, label="along-geodesic", linewidth=1.5, linestyle="--", color="tab:orange")
    if acc_orth is not None:
      ax1.plot(x_acc, acc_orth, label="orth-to-geodesic", linewidth=1.5, linestyle="--", color="tab:purple")
  ax1.set_ylabel("Acceleration")
  ax1.grid(True, alpha=0.3)
  ax1.legend(loc="best")

  fig.suptitle(title)
  fig.autofmt_xdate()
  fig.tight_layout()
  fig.savefig(out_path, dpi=150)
  plt.close(fig)


def main():
  args = parse_args()
  cfg = _load_config(args.config)
  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  multisenge_root = Path(args.multisenge_root)
  manifest = load_manifest(args.manifest)
  patches = list(manifest.get("patches", []))
  if args.max_patches and args.max_patches > 0:
    patches = patches[: args.max_patches]

  stats = load_band_stats(resolve_phase1_path(Path(cfg["normalization"]["stats_path"])))
  desired_order = cfg["dataset"].get("band_order")
  stats_order = cfg["normalization"].get("stats_band_order", desired_order)
  if desired_order is not None and stats_order is not None and stats_order != desired_order:
    stats = _subset_stats(stats, stats_order, desired_order)

  rank_r = int(cfg["subspace"].get("rank_r", 6))
  variance_threshold = cfg["subspace"].get("variance_threshold")
  use_randomized = bool(cfg["subspace"].get("use_randomized_pca", True))
  pixel_sample = int(cfg["subspace"].get("pixel_sample", 0) or 0)
  seed = int(cfg.get("seed", 1234))
  decompose = bool(cfg.get("geodesic", {}).get("decompose", True))
  nodata_value = float(cfg["dataset"].get("nodata_value", 0.0))
  min_valid_bands = int(cfg["dataset"].get("min_valid_bands", 3))

  rng = np.random.default_rng(seed)

  summary_rows = []
  for entry in patches:
    patch_id = entry["patch_id"]
    tile = entry.get("tile", "")
    x = int(entry.get("x", 0))
    y = int(entry.get("y", 0))
    s2_list = entry.get("s2", [])
    if not s2_list:
      continue

    dates = [_parse_date(z["relpath"]) for z in s2_list]
    paths = [multisenge_root / z["relpath"] for z in s2_list]
    if any(not p.exists() for p in paths):
      print(f"[WARN] Missing files for {patch_id}; skipping.")
      continue

    bases: List[np.ndarray] = []
    for p in paths:
      arr = _load_tif(p)
      vm = build_valid_mask(arr, nodata_value=nodata_value, min_valid_bands=min_valid_bands)
      arr_norm, vm = apply_normalization(arr, stats, valid_mask=vm, nodata_value=nodata_value)
      mat, _ = vectorize_cube(arr_norm, vm)
      if mat.size == 0:
        raise RuntimeError(f"No valid pixels for {patch_id} at {p}")
      if pixel_sample > 0 and mat.shape[1] > pixel_sample:
        idx = rng.choice(mat.shape[1], size=pixel_sample, replace=False)
        mat = mat[:, idx]
      basis = fit_pca_basis(
        mat,
        rank=rank_r,
        variance_threshold=variance_threshold,
        random_state=seed,
        use_randomized=use_randomized,
      ).basis
      bases.append(basis.astype(np.float32))

    vel_geo = []
    vel_mag = []
    for i in range(len(bases) - 1):
      vel_geo.append(grassmann_geodesic_distance(bases[i], bases[i + 1]))
      vel_mag.append(subspace_magnitude(bases[i], bases[i + 1]))

    acc_mag = []
    acc_along = [] if decompose else None
    acc_orth = [] if decompose else None
    for i in range(1, len(bases) - 1):
      res = second_order_difference_subspace(bases[i - 1], bases[i], bases[i + 1], decompose=decompose)
      acc_mag.append(res.mag_total)
      if decompose:
        acc_along.append(float(res.mag_along or 0.0))
        acc_orth.append(float(res.mag_orth or 0.0))

    gr = _load_ground_reference(multisenge_root, tile, x, y)
    majority = _majority_label(gr) if gr is not None else None

    patch_out = {
      "patch_id": patch_id,
      "tile": tile,
      "x": x,
      "y": y,
      "n_dates": len(dates),
      "dates": dates,
      "s2_relpaths": [z["relpath"] for z in s2_list],
      "labels_present": entry.get("labels_present", []),
      "ground_reference_majority": majority,
      "velocity_geodesic": vel_geo,
      "velocity_magnitude": vel_mag,
      "acceleration_magnitude": acc_mag,
      "acceleration_along": acc_along,
      "acceleration_orth": acc_orth,
    }

    (out_dir / "per_patch").mkdir(parents=True, exist_ok=True)
    (out_dir / "per_patch" / f"{patch_id}.json").write_text(json.dumps(patch_out, indent=2), encoding="utf-8")

    _plot_timeseries(
      dates,
      vel_geo,
      vel_mag,
      acc_mag,
      acc_along,
      acc_orth,
      out_dir / "plots" / f"{patch_id}.png",
      title=f"{patch_id} (majority GR={majority})",
    )

    summary_rows.append(
      {
        "patch_id": patch_id,
        "tile": tile,
        "x": x,
        "y": y,
        "n_dates": len(dates),
        "majority_gr": majority if majority is not None else "",
        "mean_vel_geo": float(np.mean(vel_geo)) if vel_geo else float("nan"),
        "mean_acc_mag": float(np.mean(acc_mag)) if acc_mag else float("nan"),
      }
    )

  if summary_rows:
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
      writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
      writer.writeheader()
      writer.writerows(summary_rows)

  print(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
  main()

