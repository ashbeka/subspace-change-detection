"""Controlled nuisance/change tests for temporal band-image subspaces.

This is a diagnostic benchmark, not evidence of real disaster detection.  It
uses real equal-gap MultiSenGE triples and modifies only the middle date:

- local multispectral pattern change (positive synthetic event),
- global radiometric gain/offset and one-pixel translation (negative nuisances).

The test asks whether centered/L2-normalized band-image first/second DS scores
respond more specifically to local spatial pattern changes than raw reflectance
differences.  The second-order and geodesic formulas are implemented in
``phase1.subspace.second_order_ds`` from Fukui et al. (2024).
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from sklearn.metrics import average_precision_score, roc_auc_score

from phase1.data.multisenge_manifest import load_manifest
from phase1.subspace.second_order_ds import second_order_difference_subspace
from phase1.subspace.temporal_band_images import build_band_image_subspace, sequence_common_valid_mask


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--multisenge_root", required=True, type=Path)
  parser.add_argument("--manifest", required=True, type=Path)
  parser.add_argument("--output_dir", required=True, type=Path)
  parser.add_argument("--target_date", default="20200909")
  parser.add_argument("--rank", type=int, default=6)
  parser.add_argument("--preprocessing", default="centered_band_l2")
  parser.add_argument("--repeats", type=int, default=12)
  parser.add_argument("--seed", type=int, default=1234)
  return parser.parse_args()


def _load(path: Path) -> np.ndarray:
  with rasterio.open(path) as source:
    return source.read().astype(np.float32)


def _find_local_window(mask: np.ndarray, size: int, rng: np.random.Generator) -> tuple[slice, slice]:
  height, width = mask.shape
  candidates = np.argwhere(mask)
  for _ in range(500):
    row, col = candidates[int(rng.integers(0, len(candidates)))]
    top = int(np.clip(row - size // 2, 0, height - size))
    left = int(np.clip(col - size // 2, 0, width - size))
    window = mask[top : top + size, left : left + size]
    if float(np.mean(window)) >= 0.8:
      return slice(top, top + size), slice(left, left + size)
  # Deterministic fallback: choose the valid pixel nearest the image center.
  center = np.array([height / 2.0, width / 2.0])
  row, col = candidates[int(np.argmin(np.sum((candidates - center) ** 2, axis=1)))]
  top = int(np.clip(row - size // 2, 0, height - size))
  left = int(np.clip(col - size // 2, 0, width - size))
  return slice(top, top + size), slice(left, left + size)


def _inject_local(cube: np.ndarray, window: tuple[slice, slice], strength: float) -> np.ndarray:
  changed = cube.copy()
  rows, cols = window
  # MultiSenGE order: B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12.
  changed[2, rows, cols] *= 1.0 + 0.15 * strength  # red
  changed[6, rows, cols] *= 1.0 - strength         # NIR
  changed[7, rows, cols] *= 1.0 - 0.8 * strength   # narrow NIR
  changed[8, rows, cols] *= 1.0 + 0.5 * strength   # SWIR1
  changed[9, rows, cols] *= 1.0 + 0.7 * strength   # SWIR2
  return changed


def _inject(cube: np.ndarray, kind: str, strength: float, window: tuple[slice, slice]) -> np.ndarray:
  if kind == "local_pattern_change":
    return _inject_local(cube, window, strength)
  if kind == "global_gain":
    return cube * (1.0 + strength)
  if kind == "global_offset":
    return cube + 10000.0 * strength
  if kind == "translation_1px":
    return np.roll(cube, shift=(0, 1), axis=(1, 2))
  if kind == "clean":
    return cube.copy()
  raise ValueError(kind)


def _nd(cube: np.ndarray, first: int, second: int, mask: np.ndarray) -> np.ndarray:
  a = cube[first, mask].astype(np.float64)
  b = cube[second, mask].astype(np.float64)
  return (a - b) / np.maximum(np.abs(a + b), 1e-6)


def _scores(cubes: list[np.ndarray], mask: np.ndarray, rank: int, preprocessing: str) -> dict[str, float]:
  fitted = [
    build_band_image_subspace(cube / 10000.0, mask, rank=rank, preprocessing=preprocessing)
    for cube in cubes
  ]
  result = second_order_difference_subspace(*(item.basis for item in fitted), decompose=True)
  values = [cube[:, mask].astype(np.float64) / 10000.0 for cube in cubes]
  raw_second = float(np.sqrt(np.mean((values[2] - 2.0 * values[1] + values[0]) ** 2)))
  ndvi = [_nd(cube, 6, 2, mask) for cube in cubes]
  nbr = [_nd(cube, 6, 9, mask) for cube in cubes]
  return {
    "second_magnitude": float(result.mag_total) / float(rank),
    "second_along": float(result.mag_along or 0.0) / float(rank),
    "second_orthogonal": float(result.mag_orth or 0.0) / float(rank),
    "raw_second_rmse": raw_second,
    "ndvi_second_mae": float(np.mean(np.abs(ndvi[2] - 2.0 * ndvi[1] + ndvi[0]))),
    "nbr_second_mae": float(np.mean(np.abs(nbr[2] - 2.0 * nbr[1] + nbr[0]))),
  }


def _plot_distributions(rows: list[dict], score_names: list[str], output: Path) -> None:
  kinds = ["clean", "global_gain", "global_offset", "translation_1px", "local_pattern_change"]
  figure, axes = plt.subplots(2, 3, figsize=(15, 8))
  for axis, score in zip(axes.flat, score_names):
    values = [[float(row[score]) for row in rows if row["injection"] == kind] for kind in kinds]
    axis.boxplot(values, tick_labels=["clean", "gain", "offset", "shift", "local"], showfliers=False)
    axis.set_title(score.replace("_", " "))
    axis.tick_params(axis="x", rotation=25)
    axis.grid(True, axis="y", alpha=0.25)
  figure.suptitle("Equal-gap real triples with controlled middle-date interventions")
  figure.tight_layout()
  figure.savefig(output, dpi=180)
  plt.close(figure)


def main() -> None:
  args = parse_args()
  output = args.output_dir
  output.mkdir(parents=True, exist_ok=True)
  root = args.multisenge_root
  rng = np.random.default_rng(args.seed)
  rows: list[dict] = []
  kinds = ["clean", "global_gain", "global_offset", "translation_1px", "local_pattern_change"]
  strengths = [0.1, 0.25, 0.5]
  sizes = [16, 32, 64]

  for entry in load_manifest(args.manifest).get("patches", []):
    ordered = sorted(entry.get("s2", []), key=lambda item: item["date"])
    date_to_index = {item["date"]: index for index, item in enumerate(ordered)}
    if args.target_date not in date_to_index:
      continue
    center = date_to_index[args.target_date]
    if center == 0 or center == len(ordered) - 1:
      continue
    triple_entries = ordered[center - 1 : center + 2]
    dates = [item["date"] for item in triple_entries]
    parsed = [datetime.strptime(date, "%Y%m%d") for date in dates]
    left_gap = (parsed[1] - parsed[0]).days
    right_gap = (parsed[2] - parsed[1]).days
    cubes = [_load(root / item["relpath"]) for item in triple_entries]
    mask = sequence_common_valid_mask(cubes, nodata_value=0.0)
    clean_scores = _scores(cubes, mask, args.rank, args.preprocessing)

    for repeat in range(args.repeats):
      for size in sizes:
        window = _find_local_window(mask, size, rng)
        for strength in strengths:
          for kind in kinds:
            # Clean and one-pixel translation do not depend on nominal strength;
            # keep one copy per repeat/window to avoid artificial weighting.
            if kind in {"clean", "translation_1px"} and strength != strengths[0]:
              continue
            modified = _inject(cubes[1], kind, strength, window)
            current = [cubes[0], modified, cubes[2]]
            scores = _scores(current, mask, args.rank, args.preprocessing)
            row = {
              "patch_id": entry["patch_id"],
              "date_left": dates[0],
              "date_middle": dates[1],
              "date_right": dates[2],
              "gap_left_days": int(left_gap),
              "gap_right_days": int(right_gap),
              "repeat": repeat,
              "window_size": size,
              "strength": strength,
              "injection": kind,
              "positive_local_change": int(kind == "local_pattern_change"),
            }
            for name, value in scores.items():
              row[name] = value
              row[f"delta_{name}"] = value - clean_scores[name]
            rows.append(row)
    print(f"[OK] {entry['patch_id']}: {dates}, common pixels={int(mask.sum())}")

  if not rows:
    raise RuntimeError("No target-date triples were found.")
  with (output / "injection_scores.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  score_names = [
    "delta_second_magnitude",
    "delta_second_along",
    "delta_second_orthogonal",
    "delta_raw_second_rmse",
    "delta_ndvi_second_mae",
    "delta_nbr_second_mae",
  ]
  labels = np.asarray([int(row["positive_local_change"]) for row in rows], dtype=int)
  summary = []
  for score in score_names:
    values = np.asarray([float(row[score]) for row in rows], dtype=float)
    summary.append({
      "score": score,
      "auroc_local_vs_nuisance": float(roc_auc_score(labels, values)),
      "average_precision_local_vs_nuisance": float(average_precision_score(labels, values)),
      "positive_mean": float(np.mean(values[labels == 1])),
      "negative_mean": float(np.mean(values[labels == 0])),
    })
  with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)
  _plot_distributions(rows, score_names, output / "score_distributions.png")
  (output / "run_metadata.json").write_text(
    json.dumps({
      "target_date": args.target_date,
      "rank": args.rank,
      "preprocessing": args.preprocessing,
      "repeats": args.repeats,
      "interpretation": "Synthetic diagnostic only; not real change-detection performance.",
    }, indent=2),
    encoding="utf-8",
  )
  print(f"Done. Outputs in: {output}")


if __name__ == "__main__":
  main()
