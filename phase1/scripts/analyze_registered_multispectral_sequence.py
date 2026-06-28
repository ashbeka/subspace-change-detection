"""First/second DS analysis for one registered multispectral image sequence.

This generic runner complements the MultiSenGE manifest runner.  It accepts a
directory of date-prefixed TIFFs, constructs one full/low-rank band-image
subspace per date, measures temporal first/second DS quantities, and saves
canonical spatial contribution maps whose sums equal the corresponding DS
magnitudes.

Source: Fukui et al. (2024), *Second-order Difference Subspace*.  The satellite
adaptation is ``X_t in R^(N spatial locations x B band-image vectors)``.
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

from phase1.subspace.geodesic import (
  geodesic_projection,
  grassmann_geodesic_interpolate,
  sum_subspace,
)
from phase1.subspace.second_order_ds import second_order_difference_subspace
from phase1.subspace.temporal_band_images import (
  build_band_image_subspace,
  sequence_common_valid_mask,
  spatial_difference_contribution,
  temporal_difference_measurements,
)

DATE_PATTERN = re.compile(r"(?P<year>\d{4})[-_]?(?P<month>\d{2})[-_]?(?P<day>\d{2})")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence_dir", required=True, type=Path)
  parser.add_argument("--glob", default="*.tif")
  parser.add_argument("--output_dir", required=True, type=Path)
  parser.add_argument("--rank", type=int, default=0, help="0 keeps the full band-image span.")
  parser.add_argument("--preprocessing", default="centered", choices=["uncentered", "centered", "band_l2", "centered_band_l2"])
  parser.add_argument("--nodata", type=float, default=0.0)
  parser.add_argument("--balanced_gap_ratio", type=float, default=1.5)
  parser.add_argument("--top_k", type=int, default=5)
  return parser.parse_args()


def _date(path: Path) -> str:
  match = DATE_PATTERN.search(path.name)
  if not match:
    raise ValueError(f"Could not parse YYYY-MM-DD or YYYYMMDD date from {path.name}")
  return "".join(match.group(name) for name in ("year", "month", "day"))


def _load(path: Path) -> np.ndarray:
  with rasterio.open(path) as source:
    return source.read().astype(np.float32)


def _rgb(cube: np.ndarray) -> np.ndarray:
  data = np.moveaxis(cube[:3], 0, -1).astype(np.float64)
  low, high = np.percentile(data, [2, 98])
  return np.clip((data - low) / max(high - low, 1e-12), 0.0, 1.0)


def _save_map(cube: np.ndarray, contribution: np.ndarray, path: Path, title: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  rgb = _rgb(cube)
  upper = float(np.percentile(contribution[contribution > 0], 99.5)) if np.any(contribution > 0) else 1.0
  scaled = np.clip(contribution / max(upper, 1e-12), 0.0, 1.0)
  figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
  axes[0].imshow(rgb)
  axes[0].set_title("Reference RGB")
  heat = axes[1].imshow(scaled, cmap="magma", vmin=0, vmax=1)
  axes[1].set_title("DS contribution (robust scale)")
  axes[2].imshow(rgb)
  axes[2].imshow(scaled, cmap="magma", vmin=0, vmax=1, alpha=np.where(scaled > 0.1, 0.65, 0.0))
  axes[2].set_title("Contribution overlay")
  for axis in axes:
    axis.axis("off")
  figure.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04)
  figure.suptitle(title)
  figure.tight_layout()
  figure.savefig(path, dpi=180)
  plt.close(figure)


def _save_second_decomposition(
  cube: np.ndarray,
  maps: dict[str, np.ndarray],
  path: Path,
  title: str,
) -> None:
  """Save the four geometrical second-order views on a shared percentile scale."""
  path.parent.mkdir(parents=True, exist_ok=True)
  rgb = _rgb(cube)
  positive_blocks = [value[value > 0] for value in maps.values() if np.any(value > 0)]
  positives = np.concatenate(positive_blocks) if positive_blocks else np.zeros((0,), dtype=np.float32)
  upper = float(np.percentile(positives, 99.5)) if positives.size else 1.0
  figure, axes = plt.subplots(2, 3, figsize=(15, 9))
  axes[0, 0].imshow(rgb)
  axes[0, 0].set_title("Reference RGB")
  positions = [
    ("paper_total", "Paper second DS"),
    ("orthogonal", "Orthogonal to endpoint geodesic"),
    ("along", "Along endpoint geodesic"),
    ("time_aware", "Irregular-time geodesic deviation"),
  ]
  last_image = None
  for axis, (key, label) in zip(axes.reshape(-1)[1:5], positions):
    scaled = np.clip(maps[key] / max(upper, 1e-12), 0.0, 1.0)
    last_image = axis.imshow(scaled, cmap="magma", vmin=0, vmax=1)
    axis.set_title(label)
  scaled = np.clip(maps["time_aware"] / max(upper, 1e-12), 0.0, 1.0)
  axes[1, 2].imshow(rgb)
  axes[1, 2].imshow(scaled, cmap="magma", vmin=0, vmax=1, alpha=np.where(scaled > 0.1, 0.65, 0.0))
  axes[1, 2].set_title("Time-aware overlay")
  for axis in axes.reshape(-1):
    axis.axis("off")
  if last_image is not None:
    figure.colorbar(last_image, ax=axes[:, 1:].reshape(-1).tolist(), fraction=0.025, pad=0.02)
  figure.suptitle(title)
  figure.subplots_adjust(top=0.91, wspace=0.06, hspace=0.12)
  figure.savefig(path, dpi=180, bbox_inches="tight")
  plt.close(figure)


def _write_csv(path: Path, rows: list[dict]) -> None:
  if not rows:
    return
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
  args = parse_args()
  files = sorted(args.sequence_dir.glob(args.glob), key=_date)
  if len(files) < 3:
    raise ValueError("At least three dated TIFFs are required.")
  dates = [_date(path) for path in files]
  cubes = [_load(path) for path in files]
  shape = cubes[0].shape
  if any(cube.shape != shape for cube in cubes):
    raise ValueError("All registered sequence cubes must have the same shape.")
  mask = sequence_common_valid_mask(cubes, nodata_value=args.nodata)
  rank = int(args.rank or shape[0])
  fitted = [
    build_band_image_subspace(cube / 10000.0, mask, rank=rank, preprocessing=args.preprocessing)
    for cube in cubes
  ]
  bases = [item.basis for item in fitted]
  metrics = temporal_difference_measurements(bases, dates)
  output = args.output_dir
  output.mkdir(parents=True, exist_ok=True)

  first_rows = []
  first_maps = []
  for index in range(len(files) - 1):
    contribution = spatial_difference_contribution(bases[index], bases[index + 1], mask)
    raw_rmse = float(np.sqrt(np.mean(((cubes[index + 1][:, mask] - cubes[index][:, mask]) / 10000.0) ** 2)))
    first_rows.append({
      "pair_index": index + 1,
      "date_left": dates[index],
      "date_right": dates[index + 1],
      "gap_days": metrics["adjacent_gap_days"][index],
      "first_magnitude": metrics["first_adjacent_magnitude"][index],
      "first_magnitude_per_rank": metrics["first_adjacent_magnitude"][index] / rank,
      "geodesic_distance": metrics["first_adjacent_geodesic"][index],
      "geodesic_distance_per_day": metrics["first_geodesic_per_day"][index],
      "raw_reflectance_rmse": raw_rmse,
      "contribution_sum": float(np.sum(contribution)),
    })
    first_maps.append(contribution)

  second_rows = []
  second_maps: list[dict[str, np.ndarray]] = []
  for index in range(1, len(files) - 1):
    result = second_order_difference_subspace(bases[index - 1], bases[index], bases[index + 1], decompose=True)
    total_map = spatial_difference_contribution(bases[index], result.mean_subspace, mask)
    endpoint_sum = sum_subspace(bases[index - 1], bases[index + 1])
    projected_middle = geodesic_projection(endpoint_sum, bases[index])
    orth_map = spatial_difference_contribution(bases[index], projected_middle, mask)
    along_map = spatial_difference_contribution(projected_middle, result.mean_subspace, mask)
    time_fraction = metrics["second_time_fraction"][index - 1]
    expected_middle = grassmann_geodesic_interpolate(
      bases[index - 1], bases[index + 1], time_fraction
    )
    time_aware_map = spatial_difference_contribution(bases[index], expected_middle, mask)
    ratio = metrics["second_gap_ratio"][index - 1]
    second_rows.append({
      "triple_index": index,
      "date_left": dates[index - 1],
      "date_middle": dates[index],
      "date_right": dates[index + 1],
      "gap_left_days": metrics["second_gap_left_days"][index - 1],
      "gap_right_days": metrics["second_gap_right_days"][index - 1],
      "gap_ratio": ratio,
      "balanced_gap": int(ratio <= args.balanced_gap_ratio),
      "second_magnitude": result.mag_total,
      "second_magnitude_per_rank": result.mag_total / rank,
      "second_along": float(result.mag_along or 0.0),
      "second_orthogonal": float(result.mag_orth or 0.0),
      "first_speed_change": metrics["first_speed_change"][index - 1],
      "time_fraction": metrics["second_time_fraction"][index - 1],
      "time_aware_geodesic_deviation_magnitude": metrics[
        "time_aware_geodesic_deviation_magnitude"
      ][index - 1],
      "time_aware_geodesic_deviation": metrics["time_aware_geodesic_deviation"][index - 1],
      "time_aware_acceleration_proxy": metrics["time_aware_acceleration_proxy"][index - 1],
      "total_contribution_sum": float(np.sum(total_map)),
      "orthogonal_contribution_sum": float(np.sum(orth_map)),
      "along_contribution_sum": float(np.sum(along_map)),
    })
    second_maps.append({
      "paper_total": total_map,
      "orthogonal": orth_map,
      "along": along_map,
      "time_aware": time_aware_map,
    })

  _write_csv(output / "first_order.csv", first_rows)
  _write_csv(output / "second_order.csv", second_rows)

  figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
  axes[0].plot([row["date_right"] for row in first_rows], [row["first_magnitude_per_rank"] for row in first_rows], marker="o")
  axes[0].set_ylabel("First magnitude / rank")
  axes[0].tick_params(axis="x", rotation=60)
  axes[1].plot([row["date_middle"] for row in second_rows], [row["second_magnitude_per_rank"] for row in second_rows], marker="o", label="Paper second DS")
  axes[1].plot([row["date_middle"] for row in second_rows], [row["second_orthogonal"] / rank for row in second_rows], marker=".", label="Orthogonal component")
  axes[1].set_ylabel("Magnitude / rank")
  axes[1].legend()
  axes[1].tick_params(axis="x", rotation=60)
  axes[2].plot([row["date_middle"] for row in second_rows], [row["time_aware_geodesic_deviation"] for row in second_rows], marker="o", color="tab:green")
  axes[2].set_ylabel("Time-aware geodesic deviation")
  axes[2].tick_params(axis="x", rotation=60)
  figure.suptitle("Temporal subspace dynamics")
  figure.tight_layout()
  figure.savefig(output / "temporal_dynamics.png", dpi=180)
  plt.close(figure)
  top_first = sorted(range(len(first_rows)), key=lambda i: first_rows[i]["first_magnitude"], reverse=True)[: args.top_k]
  top_second = sorted(range(len(second_rows)), key=lambda i: second_rows[i]["second_magnitude"], reverse=True)[: args.top_k]
  top_time_aware = sorted(
    range(len(second_rows)),
    key=lambda i: second_rows[i]["time_aware_geodesic_deviation"],
    reverse=True,
  )[: args.top_k]
  for rank_index, index in enumerate(top_first, start=1):
    row = first_rows[index]
    _save_map(
      cubes[index + 1],
      first_maps[index],
      output / "top_first" / f"{rank_index:02d}_{row['date_right']}.png",
      f"First DS: {row['date_left']} -> {row['date_right']}; Mag={row['first_magnitude']:.3f}",
    )
  for rank_index, index in enumerate(top_second, start=1):
    row = second_rows[index]
    _save_map(
      cubes[index + 1],
      second_maps[index]["paper_total"],
      output / "top_second" / f"{rank_index:02d}_{row['date_middle']}.png",
      f"Second DS at {row['date_middle']}; Mag={row['second_magnitude']:.3f}; gap ratio={row['gap_ratio']:.2f}",
    )
  for rank_index, index in enumerate(top_time_aware, start=1):
    row = second_rows[index]
    _save_second_decomposition(
      cubes[index + 1],
      second_maps[index],
      output / "top_time_aware" / f"{rank_index:02d}_{row['date_middle']}.png",
      (
        f"Temporal geometry at {row['date_middle']}; "
        f"paper Mag={row['second_magnitude']:.3f}; "
        f"time-aware rho={row['time_aware_geodesic_deviation']:.3f}; "
        f"gap ratio={row['gap_ratio']:.2f}"
      ),
    )
  metadata = {
    "source_directory": str(args.sequence_dir),
    "n_frames": len(files),
    "cube_shape": list(shape),
    "common_valid_pixels": int(np.sum(mask)),
    "rank": rank,
    "preprocessing": args.preprocessing,
    "dates": dates,
    "warning": "No pixel ground truth is assumed; compare maps with source-paper visual evidence or external annotations.",
    "time_aware_metric": (
      "For irregular acquisitions, compare the observed middle subspace with the "
      "constant-speed Grassmann geodesic at the observed time fraction. This is "
      "a project diagnostic, not Fukui et al.'s equal-spacing second-order DS."
    ),
  }
  (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
  print(f"Done. Outputs in: {output}")


if __name__ == "__main__":
  main()
