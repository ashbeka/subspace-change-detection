"""Local/multiscale first- and second-order DS on registered image sequences.

Source/provenance:
- Fukui et al. (2024), *Second-order Difference Subspace*, supplies the
  first/second DS, PCS mean, and geodesic decomposition.
- Dagobert et al. (IPOL 2022), *Detection and Interpretation of Change in
  Registered Satellite Image Time Series*, motivates spatial tiling as a way
  to avoid hiding spatially sparse change inside a full-scene temporal model.

Satellite adaptation:
For each date and each fixed grid cell, the B aligned band images are flattened
into columns of X_(t,g) in R^(N_g x B).  Its leading left singular vectors form
a local band-image subspace.  Corresponding cells are compared across dates.
This is not an implementation of the IPOL novelty-filter/NFA algorithm.

Verification status:
Formula/shape tests cover the local construction and contribution identities.
Real-sequence outputs are exploratory because the bundled IPOL sequence has no
pixel ground truth; its paper supplies only qualitative relevant-change frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from phase1.subspace.temporal_band_images import (
  build_tiled_band_image_subspaces,
  sequence_common_valid_mask,
  tiled_temporal_decomposition,
)

DATE_PATTERN = re.compile(r"(?P<year>\d{4})[-_]?(?P<month>\d{2})[-_]?(?P<day>\d{2})")
COMPONENTS = ("paper_total", "orthogonal", "along", "time_aware")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence_dir", required=True, type=Path)
  parser.add_argument("--glob", default="*.tif")
  parser.add_argument("--output_dir", required=True, type=Path)
  parser.add_argument("--grids", default="1,2,4,8", help="Comma-separated grid side counts.")
  parser.add_argument("--rank", type=int, default=0, help="0 keeps the full band-image span.")
  parser.add_argument(
    "--preprocessing",
    default="centered",
    choices=["uncentered", "centered", "band_l2", "centered_band_l2"],
  )
  parser.add_argument("--nodata", type=float, default=0.0)
  parser.add_argument("--min-valid-locations", type=int, default=64)
  parser.add_argument(
    "--reference-event-frames",
    default="",
    help="Optional 1-based frames reported as relevant change by an external source.",
  )
  parser.add_argument(
    "--evaluation-frame-count",
    type=int,
    default=0,
    help="Restrict illustrative event-frame agreement to the first N frames.",
  )
  parser.add_argument("--figure-frame-count", type=int, default=5)
  parser.add_argument(
    "--reference-map-dir",
    type=Path,
    default=None,
    help="Optional directory containing n<frame>_nfa.png published detector maps.",
  )
  parser.add_argument(
    "--reference-crop",
    default="",
    help="Source-image crop x,y,width,height corresponding to the published maps.",
  )
  parser.add_argument(
    "--reference-lognfa-dir",
    type=Path,
    default=None,
    help="Optional directory of lognfa_<zero-based-frame>.tif outputs from the IPOL code.",
  )
  parser.add_argument("--reference-logepsilon", type=float, default=1.0)
  return parser.parse_args()


def _date(path: Path) -> str:
  match = DATE_PATTERN.search(path.name)
  if not match:
    raise ValueError(f"Could not parse a date from {path.name}")
  return "".join(match.group(name) for name in ("year", "month", "day"))


def _load(path: Path) -> np.ndarray:
  with rasterio.open(path) as source:
    return source.read().astype(np.float32)


def _rgb(cube: np.ndarray) -> np.ndarray:
  image = np.moveaxis(cube[:3], 0, -1).astype(np.float64)
  low, high = np.percentile(image, [2, 98])
  return np.clip((image - low) / max(float(high - low), 1e-12), 0.0, 1.0)


def _parse_ints(text: str) -> list[int]:
  return [int(value.strip()) for value in str(text).split(",") if value.strip()]


def _parse_crop(text: str) -> tuple[int, int, int, int] | None:
  if not str(text).strip():
    return None
  values = _parse_ints(text)
  if len(values) != 4 or values[2] < 1 or values[3] < 1:
    raise ValueError("--reference-crop must be x,y,width,height")
  return values[0], values[1], values[2], values[3]


def _map_stats(score: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
  values = np.asarray(score[valid_mask], dtype=np.float64)
  values = values[np.isfinite(values)]
  if values.size == 0:
    return {key: 0.0 for key in ("sum", "mean", "p95", "top1_mean", "top1_fraction", "entropy")}
  positive = np.maximum(values, 0.0)
  count = max(1, int(np.ceil(0.01 * positive.size)))
  top = np.partition(positive, positive.size - count)[-count:]
  total = float(np.sum(positive))
  if total > 0.0:
    probabilities = positive[positive > 0.0] / total
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    entropy /= max(float(np.log(positive.size)), 1.0)
  else:
    entropy = 0.0
  return {
    "sum": total,
    "mean": float(np.mean(positive)),
    "p95": float(np.percentile(positive, 95)),
    "top1_mean": float(np.mean(top)),
    "top1_fraction": float(np.sum(top) / total) if total > 0.0 else 0.0,
    "entropy": entropy,
  }


def _raw_time_interpolation_residual(
  left: np.ndarray,
  middle: np.ndarray,
  right: np.ndarray,
  time_fraction: float,
) -> np.ndarray:
  expected = (1.0 - float(time_fraction)) * left + float(time_fraction) * right
  return np.sqrt(np.mean((middle - expected) ** 2, axis=0)).astype(np.float32)


def _write_csv(path: Path, rows: list[dict]) -> None:
  if not rows:
    return
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def _robust_scale(score: np.ndarray, mask: np.ndarray) -> np.ndarray:
  values = score[mask]
  upper = float(np.percentile(values[values > 0], 99.5)) if np.any(values > 0) else 1.0
  return np.clip(score / max(upper, 1e-12), 0.0, 1.0)


def _save_frame_figure(
  cube: np.ndarray,
  raw_map: np.ndarray,
  maps_by_grid: dict[int, dict[str, np.ndarray]],
  mask: np.ndarray,
  path: Path,
  title: str,
) -> None:
  grids = sorted(maps_by_grid)
  rows = 2
  cols = len(grids) + 1
  figure, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 8.3))
  axes[0, 0].imshow(_rgb(cube))
  axes[0, 0].set_title("Middle-date RGB")
  axes[1, 0].imshow(_robust_scale(raw_map, mask), cmap="magma", vmin=0, vmax=1)
  axes[1, 0].set_title("Raw time-interpolation residual")
  for column, grid in enumerate(grids, start=1):
    axes[0, column].imshow(
      _robust_scale(maps_by_grid[grid]["paper_total"], mask),
      cmap="magma",
      vmin=0,
      vmax=1,
    )
    axes[0, column].set_title(f"Paper second DS, {grid}x{grid}")
    axes[1, column].imshow(
      _robust_scale(maps_by_grid[grid]["time_aware"], mask),
      cmap="magma",
      vmin=0,
      vmax=1,
    )
    axes[1, column].set_title(f"Time-aware DS, {grid}x{grid}")
  for axis in axes.reshape(-1):
    axis.axis("off")
  figure.suptitle(title)
  figure.tight_layout()
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=180, bbox_inches="tight")
  plt.close(figure)


def _event_agreement(
  score_rows: list[dict],
  reference_frames: set[int],
  evaluation_frame_count: int,
) -> list[dict]:
  if not reference_frames or evaluation_frame_count < 3:
    return []
  eligible = [row for row in score_rows if int(row["frame_number"]) <= evaluation_frame_count]
  labels = np.asarray(
    [int(int(row["frame_number"]) in reference_frames) for row in eligible], dtype=int
  )
  if labels.size < 2 or np.unique(labels).size < 2:
    return []
  feature_names = [
    "raw_top1_mean",
    *[
      f"{component}_{stat}"
      for component in COMPONENTS
      for stat in ("tile_max_per_rank", "tile_p90_per_rank", "map_top1_mean")
    ],
  ]
  outputs: list[dict] = []
  for grid in sorted({int(row["grid_size"]) for row in eligible}):
    subset = [row for row in eligible if int(row["grid_size"]) == grid]
    subset_labels = np.asarray(
      [int(int(row["frame_number"]) in reference_frames) for row in subset], dtype=int
    )
    for feature in feature_names:
      scores = np.asarray([float(row[feature]) for row in subset], dtype=float)
      outputs.append({
        "grid_size": grid,
        "feature": feature,
        "n_frames": int(scores.size),
        "n_reference_events": int(np.sum(subset_labels)),
        "auroc": float(roc_auc_score(subset_labels, scores)),
        "average_precision": float(average_precision_score(subset_labels, scores)),
      })
  return outputs


def _ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
  labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
  scores = np.asarray(scores, dtype=np.float64).reshape(-1)
  if np.unique(labels).size < 2:
    return {"auroc": float("nan"), "average_precision": float("nan"), "best_f1": float("nan"), "best_iou": float("nan")}
  precision, recall, _ = precision_recall_curve(labels, scores)
  denominator = np.maximum(precision + recall, 1e-12)
  f1 = 2.0 * precision * recall / denominator
  iou = precision * recall / np.maximum(precision + recall - precision * recall, 1e-12)
  return {
    "auroc": float(roc_auc_score(labels, scores)),
    "average_precision": float(average_precision_score(labels, scores)),
    "best_f1": float(np.max(f1)),
    "best_iou": float(np.max(iou)),
  }


def _load_reference_mask(path: Path, width: int, height: int) -> np.ndarray:
  mask = Image.open(path).convert("L").resize((width, height), Image.Resampling.NEAREST)
  return np.asarray(mask, dtype=np.uint8) > 127


def _published_map_agreement(
  *,
  frame_number: int,
  raw_map: np.ndarray,
  maps_by_grid: dict[int, dict[str, np.ndarray]],
  reference_map_dir: Path | None,
  reference_crop: tuple[int, int, int, int] | None,
) -> list[dict]:
  """Compare maps with a rasterized published detector output, not ground truth."""
  if reference_map_dir is None or reference_crop is None:
    return []
  path = reference_map_dir / f"n{frame_number}_nfa.png"
  if not path.exists():
    return []
  x, y, width, height = reference_crop
  reference = _load_reference_mask(path, width, height)
  variants = {
    "exact_rasterized": reference,
    "dilated_3px_tolerance": ndimage.binary_dilation(reference, iterations=3),
  }
  source_maps: dict[tuple[int, str], np.ndarray] = {}
  for grid, components in maps_by_grid.items():
    source_maps[(grid, "raw_time_interpolation_residual")] = raw_map
    for component, score in components.items():
      source_maps[(grid, component)] = score
  outputs: list[dict] = []
  for (grid, method), score in source_maps.items():
    crop = score[y:y + height, x:x + width]
    if crop.shape != reference.shape:
      raise ValueError(
        f"Reference crop {reference_crop} gives {crop.shape}, expected {reference.shape}."
      )
    for variant_name, labels in variants.items():
      metrics = _ranking_metrics(labels, crop)
      outputs.append({
        "frame_number": frame_number,
        "grid_size": grid,
        "method": method,
        "reference_variant": variant_name,
        "reference_positive_fraction": float(np.mean(labels)),
        **metrics,
      })
  return outputs


def _reference_nfa_agreement(
  *,
  frame_index: int,
  raw_map: np.ndarray,
  maps_by_grid: dict[int, dict[str, np.ndarray]],
  valid_mask: np.ndarray,
  reference_lognfa_dir: Path | None,
  logepsilon: float,
) -> tuple[list[dict], dict[str, float]]:
  """Compare with actual IPOL log-NFA output; this is method agreement, not accuracy."""
  if reference_lognfa_dir is None:
    return [], {}
  path = reference_lognfa_dir / f"lognfa_{frame_index:03d}.tif"
  if not path.exists():
    return [], {}
  with rasterio.open(path) as source:
    lognfa = source.read(1).astype(np.float64)
  if lognfa.shape != valid_mask.shape:
    raise ValueError(f"Reference log-NFA shape {lognfa.shape} does not match {valid_mask.shape}.")
  finite = valid_mask & np.isfinite(lognfa)
  labels = lognfa[finite] <= float(logepsilon)
  reference_score = -lognfa[finite]
  significance = np.maximum(float(logepsilon) - lognfa, 0.0)
  reference_stats = _map_stats(significance.astype(np.float32), finite)
  outputs: list[dict] = []
  for grid, components in maps_by_grid.items():
    methods = {"raw_time_interpolation_residual": raw_map, **components}
    for method, score in methods.items():
      values = np.asarray(score[finite], dtype=np.float64)
      metrics = _ranking_metrics(labels, values)
      pearson = float(np.corrcoef(values, reference_score)[0, 1])
      spearman = float(spearmanr(values, reference_score).statistic)
      outputs.append({
        "frame_number": frame_index + 1,
        "grid_size": grid,
        "method": method,
        "nfa_positive_fraction": float(np.mean(labels)),
        "pearson_with_negative_lognfa": pearson,
        "spearman_with_negative_lognfa": spearman,
        **metrics,
      })
  return outputs, {
    "nfa_change_fraction": float(np.mean(labels)),
    "nfa_significance_top1_mean": reference_stats["top1_mean"],
  }


def _reference_nfa_temporal_agreement(score_rows: list[dict]) -> list[dict]:
  if not score_rows or "nfa_change_fraction" not in score_rows[0]:
    return []
  features = [
    "raw_top1_mean",
    *[
      f"{component}_{stat}"
      for component in COMPONENTS
      for stat in ("tile_max_per_rank", "tile_p90_per_rank", "map_top1_mean")
    ],
  ]
  outputs: list[dict] = []
  for grid in sorted({int(row["grid_size"]) for row in score_rows}):
    subset = [row for row in score_rows if int(row["grid_size"]) == grid]
    for target in ("nfa_change_fraction", "nfa_significance_top1_mean"):
      target_values = np.asarray([float(row[target]) for row in subset], dtype=float)
      for feature in features:
        values = np.asarray([float(row[feature]) for row in subset], dtype=float)
        outputs.append({
          "grid_size": grid,
          "feature": feature,
          "reference_target": target,
          "n_frames": len(subset),
          "pearson": float(np.corrcoef(values, target_values)[0, 1]),
          "spearman": float(spearmanr(values, target_values).statistic),
        })
  return outputs


def main() -> None:
  args = parse_args()
  files = sorted(args.sequence_dir.glob(args.glob), key=_date)
  if len(files) < 3:
    raise ValueError("At least three dated TIFFs are required.")
  dates = [_date(path) for path in files]
  parsed_dates = [datetime.strptime(value, "%Y%m%d") for value in dates]
  cubes = [_load(path) for path in files]
  if any(cube.shape != cubes[0].shape for cube in cubes):
    raise ValueError("All sequence images must have the same cube shape.")
  mask = sequence_common_valid_mask(cubes, nodata_value=args.nodata)
  scaled_cubes = [cube / 10000.0 for cube in cubes]
  grids = sorted(set(_parse_ints(args.grids)))
  rank = int(args.rank or cubes[0].shape[0])
  output = args.output_dir
  output.mkdir(parents=True, exist_ok=True)

  fitted_by_grid: dict[int, list[list]] = {}
  for grid in grids:
    fitted_by_grid[grid] = [
      build_tiled_band_image_subspaces(
        cube,
        mask,
        grid_size=grid,
        rank=rank,
        preprocessing=args.preprocessing,
        min_valid_locations=args.min_valid_locations,
      )
      for cube in scaled_cubes
    ]

  score_rows: list[dict] = []
  tile_rows: list[dict] = []
  spatial_agreement_rows: list[dict] = []
  nfa_spatial_agreement_rows: list[dict] = []
  maps_per_frame: dict[int, dict[int, dict[str, np.ndarray]]] = {}
  raw_per_frame: dict[int, np.ndarray] = {}
  for index in range(1, len(cubes) - 1):
    left_days = float((parsed_dates[index] - parsed_dates[index - 1]).days)
    right_days = float((parsed_dates[index + 1] - parsed_dates[index]).days)
    fraction = left_days / (left_days + right_days)
    raw_map = _raw_time_interpolation_residual(
      scaled_cubes[index - 1], scaled_cubes[index], scaled_cubes[index + 1], fraction
    )
    raw_per_frame[index] = raw_map
    maps_per_frame[index] = {}
    raw_stats = _map_stats(raw_map, mask)
    frame_rows: list[dict] = []
    for grid in grids:
      decomposition = tiled_temporal_decomposition(
        fitted_by_grid[grid][index - 1],
        fitted_by_grid[grid][index],
        fitted_by_grid[grid][index + 1],
        output_shape=mask.shape,
        time_fraction=fraction,
      )
      component_maps = {component: getattr(decomposition, component) for component in COMPONENTS}
      maps_per_frame[index][grid] = component_maps
      row: dict[str, float | int | str] = {
        "frame_number": index + 1,
        "date_middle": dates[index],
        "date_left": dates[index - 1],
        "date_right": dates[index + 1],
        "gap_left_days": left_days,
        "gap_right_days": right_days,
        "time_fraction": fraction,
        "grid_size": grid,
        "tile_count": len(decomposition.tile_rows),
        "rank": rank,
        "raw_top1_mean": raw_stats["top1_mean"],
        "raw_p95": raw_stats["p95"],
      }
      for component, component_map in component_maps.items():
        stats = _map_stats(component_map, mask)
        tile_values = np.asarray(
          [float(tile[component]) / max(int(tile["rank"]), 1) for tile in decomposition.tile_rows],
          dtype=float,
        )
        row[f"{component}_map_sum"] = stats["sum"]
        row[f"{component}_map_top1_mean"] = stats["top1_mean"]
        row[f"{component}_map_top1_fraction"] = stats["top1_fraction"]
        row[f"{component}_map_entropy"] = stats["entropy"]
        row[f"{component}_tile_mean_per_rank"] = float(np.mean(tile_values))
        row[f"{component}_tile_p90_per_rank"] = float(np.percentile(tile_values, 90))
        row[f"{component}_tile_max_per_rank"] = float(np.max(tile_values))
      frame_rows.append(row)
      for tile in decomposition.tile_rows:
        tile_rows.append({
          "frame_number": index + 1,
          "date_middle": dates[index],
          **tile,
        })
    spatial_agreement_rows.extend(_published_map_agreement(
      frame_number=index + 1,
      raw_map=raw_map,
      maps_by_grid=maps_per_frame[index],
      reference_map_dir=args.reference_map_dir,
      reference_crop=_parse_crop(args.reference_crop),
    ))
    nfa_rows, nfa_frame_stats = _reference_nfa_agreement(
      frame_index=index,
      raw_map=raw_map,
      maps_by_grid=maps_per_frame[index],
      valid_mask=mask,
      reference_lognfa_dir=args.reference_lognfa_dir,
      logepsilon=args.reference_logepsilon,
    )
    nfa_spatial_agreement_rows.extend(nfa_rows)
    for row in frame_rows:
      row.update(nfa_frame_stats)
      score_rows.append(row)

  reference_frames = set(_parse_ints(args.reference_event_frames))
  agreement = _event_agreement(score_rows, reference_frames, args.evaluation_frame_count)
  _write_csv(output / "multiscale_temporal_scores.csv", score_rows)
  _write_csv(output / "tile_temporal_scores.csv", tile_rows)
  _write_csv(output / "published_event_frame_agreement.csv", agreement)
  _write_csv(output / "published_map_agreement.csv", spatial_agreement_rows)
  _write_csv(output / "reference_nfa_spatial_agreement.csv", nfa_spatial_agreement_rows)
  _write_csv(
    output / "reference_nfa_temporal_agreement.csv",
    _reference_nfa_temporal_agreement(score_rows),
  )

  first_figure_frame = 2
  final_figure_frame = min(int(args.figure_frame_count), len(cubes) - 1)
  for frame_number in range(first_figure_frame, final_figure_frame + 1):
    index = frame_number - 1
    _save_frame_figure(
      cubes[index],
      raw_per_frame[index],
      maps_per_frame[index],
      mask,
      output / "frame_maps" / f"frame_{frame_number:02d}_{dates[index]}.png",
      (
        f"Frame {frame_number}: {dates[index]} | external relevant-change frame="
        f"{frame_number in reference_frames}"
      ),
    )

  metadata = {
    "source_directory": str(args.sequence_dir),
    "n_frames": len(files),
    "cube_shape": list(cubes[0].shape),
    "common_valid_locations": int(np.sum(mask)),
    "grids": grids,
    "rank": rank,
    "preprocessing": args.preprocessing,
    "reference_event_frames": sorted(reference_frames),
    "evaluation_frame_count": int(args.evaluation_frame_count),
    "reference_map_directory": str(args.reference_map_dir) if args.reference_map_dir else None,
    "reference_crop": list(_parse_crop(args.reference_crop)) if _parse_crop(args.reference_crop) else None,
    "reference_lognfa_directory": str(args.reference_lognfa_dir) if args.reference_lognfa_dir else None,
    "reference_logepsilon": float(args.reference_logepsilon),
    "construction": (
      "For each date and grid cell: X_(t,g) has rows=fixed spatial positions "
      "and columns=complete band images; the left singular vectors form S_(t,g)."
    ),
    "raw_control": "Per-pixel RMS residual from linear interpolation in acquisition time.",
    "warning": (
      "Published relevant-change frames provide illustrative temporal agreement only, "
      "not ground truth. Published NFA map agreement, when supplied, compares with a "
      "rasterized paper figure and is not accuracy. Per-map percentile scaling is visualization-only."
    ),
  }
  (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
  print(f"Done. Outputs in: {output}")


if __name__ == "__main__":
  main()
