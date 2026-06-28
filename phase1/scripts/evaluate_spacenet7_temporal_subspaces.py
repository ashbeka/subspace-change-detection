"""Evaluate rolling local temporal subspaces on labeled SpaceNet 7 growth.

Research object and source trail
--------------------------------
SpaceNet 7 provides co-registered monthly RGB mosaics and persistent building
IDs (Van Etten et al., CVPR 2021).  For each fixed spatial cell, this script
forms a rolling window matrix

``X_t = [x_(t-W+1), ..., x_t] in R^((3*N) x W)``

where one column is one flattened monthly RGB observation over ``N`` common
valid pixels.  PCA produces a local temporal subspace ``S_t``.  The script
then measures first DS/Grassmann change between ``S_(t-1)`` and ``S_t`` and
the Fukui et al. (2024) second-order/geodesic decomposition of
``(S_(t-1), S_t, S_(t+1))``.

The rolling-window satellite construction and the construction-localization
evaluation are project adaptations.  They are not formulas claimed by Fukui,
Kanai, or the SpaceNet authors.  First appearance of a matched SpaceNet 7
building ID is used as independently annotated construction evidence.

Primary claim boundary: this is an exploratory RGB temporal-label test on one
or more AOIs, not proof of multispectral Sentinel-2 performance and not an
implementation of the official SCOT object-tracking metric.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from rasterio.transform import Affine
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from phase1.data.spacenet7_dataset import (
  first_appearance_features,
  load_spacenet7_image,
  rasterize_feature_mask,
  read_geojson_features,
  scan_spacenet7_aoi,
)
from phase1.subspace.geodesic import grassmann_geodesic_distance, subspace_magnitude
from phase1.subspace.second_order_ds import second_order_difference_subspace
from phase1.subspace.temporal_trajectory import (
  build_temporal_representation_subspace,
  representation_covariance_change,
  representation_energy_change,
  representation_spectrum_change,
)


FIRST_SCORES = (
  "raw_pair_rms",
  "window_mean_rms",
  "robust_temporal_z_rms",
  "first_ds_magnitude",
  "first_geodesic",
  "representation_energy_change",
  "representation_spectrum_change",
  "representation_covariance_change",
)
SECOND_SCORES = (
  "raw_second_rms",
  "window_second_rms",
  "second_total",
  "second_along",
  "second_orthogonal",
  "second_orthogonal_fraction",
)
ALL_SCORES = FIRST_SCORES + SECOND_SCORES


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--aoi_root", required=True, type=Path)
  parser.add_argument("--output_dir", required=True, type=Path)
  parser.add_argument("--window", type=int, default=4)
  parser.add_argument("--rank", type=int, default=2)
  parser.add_argument("--grids", default="8,16")
  parser.add_argument("--representations", default="unordered,difference,trajectory2")
  parser.add_argument("--preprocessing", default="feature_centered")
  parser.add_argument(
    "--radiometric_normalization",
    default="none",
    choices=["none", "per_date_channel_standardize", "per_date_channel_quantile"],
  )
  parser.add_argument("--min_valid_pixels", type=int, default=16)
  parser.add_argument("--min_new_building_pixels", type=int, default=2)
  parser.add_argument("--bootstrap", type=int, default=300)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--all_touched_labels", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument(
    "--controls_only",
    action="store_true",
    help="Compute radiometric first/second controls without fitting subspaces.",
  )
  return parser.parse_args()


def _parse_representation(value: str) -> tuple[str, int, str]:
  key = value.strip().lower().replace("-", "_")
  if key.startswith("trajectory"):
    suffix = key.removeprefix("trajectory")
    lag = int(suffix) if suffix else 2
    return "trajectory", lag, f"trajectory{lag}"
  if key in {"difference", "first_difference"}:
    return "difference", 1, "difference"
  if key == "unordered":
    return "unordered", 1, "unordered"
  raise ValueError(f"Unknown representation: {value}")


def _grid_cells(height: int, width: int, grid: int) -> Iterable[tuple[int, int, slice, slice]]:
  if grid < 1 or grid > min(height, width):
    raise ValueError(f"grid must be in [1, {min(height, width)}], got {grid}")
  row_edges = np.linspace(0, height, grid + 1, dtype=int)
  col_edges = np.linspace(0, width, grid + 1, dtype=int)
  for row in range(grid):
    for col in range(grid):
      yield row, col, slice(row_edges[row], row_edges[row + 1]), slice(col_edges[col], col_edges[col + 1])


def _normalize_date(cube: np.ndarray, valid: np.ndarray, mode: str) -> np.ndarray:
  if mode == "none":
    return cube.astype(np.float32, copy=False)
  output = cube.astype(np.float64, copy=True)
  for band in range(output.shape[0]):
    values = output[band][valid]
    if mode == "per_date_channel_standardize":
      center = float(np.median(values))
      mad = 1.4826 * float(np.median(np.abs(values - center)))
      output[band] = (output[band] - center) / max(mad, 1e-3)
    elif mode == "per_date_channel_quantile":
      low, high = np.quantile(values, [0.02, 0.98])
      output[band] = (output[band] - low) / max(float(high - low), 1e-3)
      output[band] = np.clip(output[band], 0.0, 1.0)
  return output.astype(np.float32)


def _raw_controls(
  cubes: list[np.ndarray],
  end_index: int,
  window: int,
  valid: np.ndarray,
) -> dict[str, float]:
  previous = cubes[end_index - 1][:, valid].astype(np.float64)
  current = cubes[end_index][:, valid].astype(np.float64)
  previous_window = np.stack(
    [cube[:, valid] for cube in cubes[end_index - window : end_index]],
    axis=0,
  ).astype(np.float64)
  current_window = np.stack(
    [cube[:, valid] for cube in cubes[end_index - window + 1 : end_index + 1]],
    axis=0,
  ).astype(np.float64)
  temporal_center = np.median(previous_window, axis=0)
  temporal_mad = 1.4826 * np.median(np.abs(previous_window - temporal_center), axis=0)
  standardized = (current - temporal_center) / np.maximum(temporal_mad, 0.02)
  return {
    "raw_pair_rms": float(np.sqrt(np.mean((current - previous) ** 2))),
    "window_mean_rms": float(np.sqrt(np.mean((np.mean(current_window, axis=0) - np.mean(previous_window, axis=0)) ** 2))),
    "robust_temporal_z_rms": float(np.sqrt(np.mean(standardized * standardized))),
  }


def _raw_second_controls(
  cubes: list[np.ndarray],
  end_index: int,
  window: int,
  valid: np.ndarray,
) -> dict[str, float]:
  previous = cubes[end_index - 1][:, valid].astype(np.float64)
  current = cubes[end_index][:, valid].astype(np.float64)
  following = cubes[end_index + 1][:, valid].astype(np.float64)
  previous_window = np.mean(
    np.stack([cube[:, valid] for cube in cubes[end_index - window : end_index]], axis=0),
    axis=0,
  ).astype(np.float64)
  current_window = np.mean(
    np.stack([cube[:, valid] for cube in cubes[end_index - window + 1 : end_index + 1]], axis=0),
    axis=0,
  ).astype(np.float64)
  following_window = np.mean(
    np.stack([cube[:, valid] for cube in cubes[end_index - window + 2 : end_index + 2]], axis=0),
    axis=0,
  ).astype(np.float64)
  return {
    "raw_second_rms": float(np.sqrt(np.mean((following - 2.0 * current + previous) ** 2))),
    "window_second_rms": float(
      np.sqrt(np.mean((following_window - 2.0 * current_window + previous_window) ** 2))
    ),
  }


def transition_valid_mask(
  valid_masks: list[np.ndarray],
  *,
  end_index: int,
  window: int,
) -> np.ndarray:
  """Intersect only dates required by one first/second rolling comparison."""
  support_stop = end_index + 2 if end_index + 1 < len(valid_masks) else end_index + 1
  required = valid_masks[end_index - window : support_stop]
  if len(required) < window + 1:
    raise ValueError("A transition comparison requires previous and current rolling windows.")
  return np.logical_and.reduce(required)


def _best_threshold_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
  precision, recall, _ = precision_recall_curve(labels, scores)
  f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
  iou = precision * recall / np.maximum(precision + recall - precision * recall, 1e-12)
  return float(np.max(f1)), float(np.max(iou))


def _weighted_arrays(rows: list[dict], score: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  labels, scores, weights = [], [], []
  for row in rows:
    value = float(row[score])
    if not np.isfinite(value):
      continue
    positive = int(row["positive_pixels"])
    negative = int(row["negative_pixels"])
    if positive:
      labels.append(1)
      scores.append(value)
      weights.append(positive)
    if negative:
      labels.append(0)
      scores.append(value)
      weights.append(negative)
  return np.asarray(labels), np.asarray(scores), np.asarray(weights, dtype=np.float64)


def _cell_arrays(rows: list[dict], score: str, min_new_pixels: int) -> tuple[np.ndarray, np.ndarray]:
  selected = [row for row in rows if np.isfinite(float(row[score]))]
  labels = np.asarray([int(row["positive_pixels"] >= min_new_pixels) for row in selected])
  scores = np.asarray([float(row[score]) for row in selected], dtype=np.float64)
  return labels, scores


def _summarize(
  rows: list[dict],
  *,
  min_new_pixels: int,
  bootstrap: int,
  rng: np.random.Generator,
) -> list[dict]:
  output = []
  groups = sorted({(row["representation"], int(row["grid"]), int(row["rank"])) for row in rows})
  for representation, grid, rank in groups:
    group = [
      row for row in rows
      if row["representation"] == representation and int(row["grid"]) == grid and int(row["rank"]) == rank
    ]
    for score in ALL_SCORES:
      labels, values = _cell_arrays(group, score, min_new_pixels)
      pixel_labels, pixel_values, pixel_weights = _weighted_arrays(group, score)
      if labels.size == 0 or np.unique(labels).size < 2 or np.unique(pixel_labels).size < 2:
        continue
      dates = sorted({row["date"] for row in group if np.isfinite(float(row[score]))})
      date_ap = []
      for date in dates:
        date_rows = [row for row in group if row["date"] == date]
        date_labels, date_values = _cell_arrays(date_rows, score, min_new_pixels)
        if np.unique(date_labels).size == 2:
          date_ap.append(float(average_precision_score(date_labels, date_values)))
      boot_ap = []
      for _ in range(max(int(bootstrap), 0)):
        sampled_dates = rng.choice(dates, size=len(dates), replace=True)
        sampled = [row for date in sampled_dates for row in group if row["date"] == date]
        sampled_labels, sampled_values = _cell_arrays(sampled, score, min_new_pixels)
        if np.unique(sampled_labels).size == 2:
          boot_ap.append(float(average_precision_score(sampled_labels, sampled_values)))
      best_f1, best_iou = _best_threshold_metrics(labels, values)
      output.append({
        "representation": representation,
        "grid": grid,
        "rank": rank,
        "score": score,
        "cells": int(labels.size),
        "positive_cells": int(np.sum(labels)),
        "cell_positive_rate": float(np.mean(labels)),
        "cell_auroc": float(roc_auc_score(labels, values)),
        "cell_average_precision": float(average_precision_score(labels, values)),
        "cell_best_f1": best_f1,
        "cell_best_iou": best_iou,
        "macro_date_average_precision": float(np.mean(date_ap)) if date_ap else float("nan"),
        "pixel_weighted_auroc": float(roc_auc_score(pixel_labels, pixel_values, sample_weight=pixel_weights)),
        "pixel_weighted_average_precision": float(average_precision_score(pixel_labels, pixel_values, sample_weight=pixel_weights)),
        "cell_average_precision_ci_low": float(np.quantile(boot_ap, 0.025)) if boot_ap else float("nan"),
        "cell_average_precision_ci_high": float(np.quantile(boot_ap, 0.975)) if boot_ap else float("nan"),
      })
  return output


def _paired_score_comparisons(
  rows: list[dict],
  *,
  min_new_pixels: int,
  bootstrap: int,
  rng: np.random.Generator,
) -> list[dict]:
  pairs = (
    ("first_ds_magnitude", "raw_pair_rms"),
    ("first_ds_magnitude", "robust_temporal_z_rms"),
    ("representation_spectrum_change", "raw_pair_rms"),
    ("representation_covariance_change", "raw_pair_rms"),
    ("second_orthogonal", "raw_pair_rms"),
    ("second_orthogonal", "robust_temporal_z_rms"),
    ("second_along", "raw_pair_rms"),
    ("second_along", "robust_temporal_z_rms"),
    ("second_along", "raw_second_rms"),
    ("second_along", "window_second_rms"),
    ("second_orthogonal", "raw_second_rms"),
    ("second_orthogonal", "window_second_rms"),
    ("second_orthogonal", "second_along"),
  )
  output = []
  groups = sorted({(row["representation"], int(row["grid"]), int(row["rank"])) for row in rows})
  for representation, grid, rank in groups:
    group = [
      row for row in rows
      if row["representation"] == representation and int(row["grid"]) == grid and int(row["rank"]) == rank
    ]
    for first_score, second_score in pairs:
      selected = [
        row for row in group
        if np.isfinite(float(row[first_score])) and np.isfinite(float(row[second_score]))
      ]
      labels = np.asarray([int(row["positive_pixels"] >= min_new_pixels) for row in selected])
      if labels.size == 0 or np.unique(labels).size < 2:
        continue
      first_values = np.asarray([float(row[first_score]) for row in selected])
      second_values = np.asarray([float(row[second_score]) for row in selected])
      first_ap = float(average_precision_score(labels, first_values))
      second_ap = float(average_precision_score(labels, second_values))
      dates = sorted({row["date"] for row in selected})
      differences = []
      for _ in range(max(int(bootstrap), 0)):
        sampled_dates = rng.choice(dates, size=len(dates), replace=True)
        sampled = [row for date in sampled_dates for row in selected if row["date"] == date]
        sampled_labels = np.asarray([int(row["positive_pixels"] >= min_new_pixels) for row in sampled])
        if np.unique(sampled_labels).size < 2:
          continue
        first_sample = np.asarray([float(row[first_score]) for row in sampled])
        second_sample = np.asarray([float(row[second_score]) for row in sampled])
        differences.append(
          float(average_precision_score(sampled_labels, first_sample))
          - float(average_precision_score(sampled_labels, second_sample))
        )
      output.append({
        "representation": representation,
        "grid": grid,
        "rank": rank,
        "first_score": first_score,
        "second_score": second_score,
        "first_average_precision": first_ap,
        "second_average_precision": second_ap,
        "delta_average_precision": first_ap - second_ap,
        "delta_ci_low": float(np.quantile(differences, 0.025)) if differences else float("nan"),
        "delta_ci_high": float(np.quantile(differences, 0.975)) if differences else float("nan"),
        "bootstrap_probability_delta_positive": float(np.mean(np.asarray(differences) > 0.0)) if differences else float("nan"),
        "bootstrap_unit": "monthly transition",
      })
  return output


def _write_csv(path: Path, rows: list[dict]) -> None:
  if not rows:
    return
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def _fill_map(rows: list[dict], score: str, shape: tuple[int, int]) -> np.ndarray:
  output = np.full(shape, np.nan, dtype=np.float32)
  for row in rows:
    output[
      int(row["row_start"]) : int(row["row_stop"]),
      int(row["col_start"]) : int(row["col_stop"]),
    ] = float(row[score])
  return output


def _rgb_for_display(cube: np.ndarray, valid: np.ndarray) -> np.ndarray:
  rgb = np.moveaxis(cube, 0, -1).astype(np.float64)
  low, high = np.quantile(rgb[valid], [0.01, 0.99])
  return np.clip((rgb - low) / max(float(high - low), 1e-6), 0.0, 1.0)


def _save_example(
  *,
  rows: list[dict],
  summary: list[dict],
  cubes: list[np.ndarray],
  valid_masks: list[np.ndarray],
  new_masks: list[np.ndarray],
  dates: list[str],
  output_dir: Path,
) -> None:
  date_counts = [
    (date, int(np.sum(mask & valid)))
    for date, mask, valid in zip(dates, new_masks, valid_masks)
  ]
  evaluated_dates = {row["date"] for row in rows}
  eligible = [(date, count) for date, count in date_counts if date in evaluated_dates]
  selected_date = max(eligible, key=lambda item: item[1])[0]
  date_index = dates.index(selected_date)
  ds_rows = [
    item for item in summary
    if item["score"] in {"first_ds_magnitude", "representation_spectrum_change", "representation_covariance_change"}
  ]
  if ds_rows:
    best = max(ds_rows, key=lambda item: float(item["cell_average_precision"]))
  else:
    first = rows[0]
    best = {
      "representation": first["representation"],
      "grid": first["grid"],
      "rank": first["rank"],
      "score": "raw_second_rms" if np.isfinite(float(first["raw_second_rms"])) else "raw_pair_rms",
    }
  selected = [
    row for row in rows
    if row["date"] == selected_date
    and row["representation"] == best["representation"]
    and int(row["grid"]) == int(best["grid"])
    and int(row["rank"]) == int(best["rank"])
  ]
  display_valid = valid_masks[date_index - 1] & valid_masks[date_index]
  panels = [
    (_rgb_for_display(cubes[date_index - 1], display_valid), f"Previous month: {dates[date_index - 1]}", None),
    (_rgb_for_display(cubes[date_index], display_valid), f"Current month: {selected_date}", None),
    (new_masks[date_index] & display_valid, f"First-appearance buildings ({date_counts[date_index][1]} px)", "gray"),
    (_fill_map(selected, "raw_pair_rms", display_valid.shape), "Raw pair RMS", "magma"),
    (_fill_map(selected, "robust_temporal_z_rms", display_valid.shape), "Robust temporal z RMS", "magma"),
    (_fill_map(selected, str(best["score"]), display_valid.shape), f"Best subspace diagnostic: {best['score']}", "magma"),
    (_fill_map(selected, "second_along", display_valid.shape), "Second DS: along", "magma"),
    (_fill_map(selected, "second_orthogonal", display_valid.shape), "Second DS: orthogonal", "magma"),
  ]
  figure, axes = plt.subplots(2, 4, figsize=(16, 8))
  for axis, (image, title, cmap) in zip(axes.flat, panels):
    axis.imshow(image, cmap=cmap)
    axis.set_title(title, fontsize=10)
    axis.axis("off")
  figure.suptitle(
    f"SpaceNet 7 exploratory construction evidence | {best['representation']} | grid={best['grid']} | rank={best['rank']}"
  )
  figure.tight_layout()
  figure.savefig(output_dir / "highest_growth_month_comparison.png", dpi=180)
  plt.close(figure)


def _save_rankings(summary: list[dict], output_dir: Path) -> None:
  if not summary:
    return
  ordered = sorted(summary, key=lambda row: float(row["cell_average_precision"]), reverse=True)[:24]
  labels = [f"{row['score']} | {row['representation']} | g{row['grid']}" for row in reversed(ordered)]
  values = [float(row["cell_average_precision"]) for row in reversed(ordered)]
  figure, axis = plt.subplots(figsize=(11, 9))
  axis.barh(np.arange(len(values)), values, color="#2878B5")
  axis.set_yticks(np.arange(len(values)), labels)
  axis.set_xlabel("Cell-level average precision")
  axis.set_title("SpaceNet 7 rolling temporal-subspace score ranking")
  axis.grid(axis="x", alpha=0.25)
  figure.tight_layout()
  figure.savefig(output_dir / "cell_average_precision_ranking.png", dpi=180)
  plt.close(figure)


def main() -> None:
  started = time.perf_counter()
  args = parse_args()
  if args.window < 3:
    raise ValueError("window must be >= 3 so centered PCA has at least two possible components.")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  grids = sorted({int(value) for value in args.grids.split(",") if value.strip()})
  representations = [_parse_representation(value) for value in args.representations.split(",") if value.strip()]
  observations = scan_spacenet7_aoi(args.aoi_root)
  loaded = [load_spacenet7_image(observation) for observation in observations]
  reference_shape = loaded[0].valid_mask.shape
  if any(image.valid_mask.shape != reference_shape for image in loaded):
    raise ValueError("SpaceNet 7 observations do not share one image shape.")
  valid_masks = [image.valid_mask for image in loaded]
  sequence_common_valid = np.logical_and.reduce(valid_masks)
  cubes = [
    _normalize_date(image.rgb, image.valid_mask, args.radiometric_normalization)
    for image in loaded
  ]
  feature_sequences = [read_geojson_features(observation.matched_label_path) for observation in observations]
  new_features = first_appearance_features(feature_sequences)
  new_masks = [
    rasterize_feature_mask(
      features,
      out_shape=reference_shape,
      # labels_match_pix polygons are already expressed in image pixel
      # coordinates; applying the GeoTIFF map transform would move them out of
      # the raster.  UDM/map-coordinate labels are handled separately by the
      # dataset loader.
      transform=Affine.identity(),
      all_touched=args.all_touched_labels,
    )
    for index, features in enumerate(new_features)
  ]
  dates = [observation.date for observation in observations]
  rows: list[dict] = []

  for grid in grids:
    for cell_row, cell_col, row_slice, col_slice in _grid_cells(*reference_shape, grid):
      cell_cubes = [cube[:, row_slice, col_slice] for cube in cubes]
      cell_valid_masks = [mask[row_slice, col_slice] for mask in valid_masks]
      if args.controls_only:
        for end_index in range(args.window, len(cubes)):
          cell_valid = transition_valid_mask(
            cell_valid_masks,
            end_index=end_index,
            window=args.window,
          )
          valid_pixels = int(np.sum(cell_valid))
          if valid_pixels < args.min_valid_pixels:
            continue
          second_scores = {score: float("nan") for score in SECOND_SCORES}
          if end_index + 1 < len(cubes):
            second_scores.update(_raw_second_controls(cell_cubes, end_index, args.window, cell_valid))
          target = new_masks[end_index][row_slice, col_slice] & cell_valid
          positive_pixels = int(np.sum(target))
          rows.append({
            "date": dates[end_index],
            "date_index": end_index,
            "representation": "controls",
            "lag": 1,
            "preprocessing": "not_applicable",
            "radiometric_normalization": args.radiometric_normalization,
            "window": args.window,
            "rank": 0,
            "grid": grid,
            "cell_row": cell_row,
            "cell_col": cell_col,
            "row_start": row_slice.start,
            "row_stop": row_slice.stop,
            "col_start": col_slice.start,
            "col_stop": col_slice.stop,
            "positive_pixels": positive_pixels,
            "valid_pixels": valid_pixels,
            "negative_pixels": valid_pixels - positive_pixels,
            "first_ds_magnitude": float("nan"),
            "first_geodesic": float("nan"),
            "representation_energy_change": float("nan"),
            "representation_spectrum_change": float("nan"),
            "representation_covariance_change": float("nan"),
            **_raw_controls(cell_cubes, end_index, args.window, cell_valid),
            **second_scores,
          })
        continue
      for representation, lag, representation_name in representations:
        if representation == "trajectory" and lag >= args.window:
          continue
        for end_index in range(args.window, len(cubes)):
          cell_valid = transition_valid_mask(
            cell_valid_masks,
            end_index=end_index,
            window=args.window,
          )
          valid_pixels = int(np.sum(cell_valid))
          if valid_pixels < args.min_valid_pixels:
            continue
          previous_fit = build_temporal_representation_subspace(
            cell_cubes[end_index - args.window : end_index],
            cell_valid,
            rank=args.rank,
            representation=representation,
            lag=lag,
            preprocessing=args.preprocessing,
          )
          current_fit = build_temporal_representation_subspace(
            cell_cubes[end_index - args.window + 1 : end_index + 1],
            cell_valid,
            rank=args.rank,
            representation=representation,
            lag=lag,
            preprocessing=args.preprocessing,
          )
          effective_rank = min(previous_fit.rank, current_fit.rank)
          if effective_rank < 1:
            continue
          previous_basis = previous_fit.basis[:, :effective_rank]
          current_basis = current_fit.basis[:, :effective_rank]
          second_scores = {score: float("nan") for score in SECOND_SCORES}
          if end_index + 1 < len(cubes):
            next_fit = build_temporal_representation_subspace(
              cell_cubes[end_index - args.window + 2 : end_index + 2],
              cell_valid,
              rank=args.rank,
              representation=representation,
              lag=lag,
              preprocessing=args.preprocessing,
            )
            second_rank = min(effective_rank, next_fit.rank)
            if second_rank >= 1:
              second = second_order_difference_subspace(
                previous_basis[:, :second_rank],
                current_basis[:, :second_rank],
                next_fit.basis[:, :second_rank],
                decompose=True,
              )
              total = float(second.mag_total) / second_rank
              along = float(second.mag_along if second.mag_along is not None else np.nan) / second_rank
              orthogonal = float(second.mag_orth if second.mag_orth is not None else np.nan) / second_rank
              second_scores = {
                **_raw_second_controls(cell_cubes, end_index, args.window, cell_valid),
                "second_total": total,
                "second_along": along,
                "second_orthogonal": orthogonal,
                "second_orthogonal_fraction": orthogonal / max(along + orthogonal, 1e-12),
              }
          target = new_masks[end_index][row_slice, col_slice] & cell_valid
          positive_pixels = int(np.sum(target))
          controls = _raw_controls(cell_cubes, end_index, args.window, cell_valid)
          rows.append({
            "date": dates[end_index],
            "date_index": end_index,
            "representation": representation_name,
            "lag": lag,
            "preprocessing": args.preprocessing,
            "radiometric_normalization": args.radiometric_normalization,
            "window": args.window,
            "rank": effective_rank,
            "grid": grid,
            "cell_row": cell_row,
            "cell_col": cell_col,
            "row_start": row_slice.start,
            "row_stop": row_slice.stop,
            "col_start": col_slice.start,
            "col_stop": col_slice.stop,
            "positive_pixels": positive_pixels,
            "valid_pixels": valid_pixels,
            "negative_pixels": valid_pixels - positive_pixels,
            "first_ds_magnitude": subspace_magnitude(previous_basis, current_basis) / effective_rank,
            "first_geodesic": grassmann_geodesic_distance(previous_basis, current_basis) / np.sqrt(effective_rank),
            "representation_energy_change": representation_energy_change(previous_fit, current_fit),
            "representation_spectrum_change": representation_spectrum_change(previous_fit, current_fit),
            "representation_covariance_change": representation_covariance_change(previous_fit, current_fit),
            **controls,
            **second_scores,
          })
    print(f"[OK] grid={grid}: accumulated_rows={len(rows)}", flush=True)

  if not rows:
    raise RuntimeError("No SpaceNet 7 cells were evaluated.")
  summary = _summarize(
    rows,
    min_new_pixels=args.min_new_building_pixels,
    bootstrap=args.bootstrap,
    rng=np.random.default_rng(args.seed),
  )
  paired = _paired_score_comparisons(
    rows,
    min_new_pixels=args.min_new_building_pixels,
    bootstrap=args.bootstrap,
    rng=np.random.default_rng(args.seed + 1),
  )
  _write_csv(args.output_dir / "cell_transition_scores.csv", rows)
  _write_csv(args.output_dir / "score_summary.csv", summary)
  _write_csv(args.output_dir / "paired_score_comparisons.csv", paired)
  _save_rankings(summary, args.output_dir)
  _save_example(
    rows=rows,
    summary=summary,
    cubes=cubes,
    valid_masks=valid_masks,
    new_masks=new_masks,
    dates=dates,
    output_dir=args.output_dir,
  )
  metadata = {
    "study": "rolling local temporal subspaces on labeled SpaceNet 7 construction",
    "dataset_source": "Van Etten et al. (2021), The Multi-Temporal Urban Development SpaceNet Dataset, DOI 10.1109/CVPR46437.2021.00633",
    "dataset_url": "s3://spacenet-dataset/spacenet/SN7_buildings/",
    "aoi_root": str(args.aoi_root),
    "dates": dates,
    "observations": len(observations),
    "sequence_intersection_valid_pixels": int(np.sum(sequence_common_valid)),
    "validity_support": "transition-local intersection over dates required by the compared rolling windows",
    "new_building_features_by_date": {date: len(features) for date, features in zip(dates, new_features)},
    "new_building_pixels_by_date": {
      date: int(np.sum(mask & valid))
      for date, mask, valid in zip(dates, new_masks, valid_masks)
    },
    "window": args.window,
    "rank": args.rank,
    "grids": grids,
    "representations": [name for _, _, name in representations],
    "preprocessing": args.preprocessing,
    "radiometric_normalization": args.radiometric_normalization,
    "all_touched_labels": args.all_touched_labels,
    "minimum_new_building_pixels_for_positive_cell": args.min_new_building_pixels,
    "controls_only": args.controls_only,
    "bootstrap_unit": "monthly transition within one AOI",
    "runtime_seconds": time.perf_counter() - started,
    "claim_boundary": "Exploratory RGB temporal-label evidence on one AOI; not SCOT, not held-out geography, and not multispectral Sentinel-2 validation.",
  }
  (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
  best = sorted(summary, key=lambda row: float(row["cell_average_precision"]), reverse=True)[:10]
  print(json.dumps({"output_dir": str(args.output_dir), "rows": len(rows), "top": best}, indent=2))


if __name__ == "__main__":
  main()
