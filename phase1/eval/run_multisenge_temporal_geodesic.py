"""Analyze first/second Difference Subspaces in MultiSenGE sequences.

Primary construction (``band_image``): for each aligned date, flatten every
Sentinel-2 band over one common spatial mask.  The resulting matrix is
``X_t in R^(N x B)`` (spatial locations x bands), and its leading left singular
vectors form ``S_t in Gr(r, N)``.  This preserves spatial coordinate identity
inside the subspace instead of fitting a global distribution of pixel spectra.

Paper-to-code trail:
- Fukui et al. (2024), Definitions 2.6, 3.1, 3.2, and 4.1: first magnitude,
  second-order DS, and geodesic decomposition.
- ``phase1.subspace.temporal_band_images``: satellite sample adaptation.
- ``phase1.subspace.second_order_ds``: PCS mean and second-order quantities.

The paper assumes equal temporal intervals.  MultiSenGE is irregularly sampled,
so every triple records its two gaps and gap ratio.  Only approximately balanced
triples should support a central-difference interpretation.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml

from phase1.data.multisenge_manifest import load_manifest
from phase1.data.preprocessing import apply_normalization, build_valid_mask, load_band_stats, vectorize_cube
from phase1.ds.pca_utils import fit_pca_basis
from phase1.eval.utils import suppress_rasterio_warnings
from phase1.subspace.temporal_band_images import (
  build_band_image_subspace,
  sequence_common_valid_mask,
  temporal_difference_measurements,
)

suppress_rasterio_warnings()

PHASE1_ROOT = Path(__file__).resolve().parents[1]


def resolve_phase1_path(path: Path) -> Path:
  return path if path.is_absolute() else PHASE1_ROOT / path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", required=True, type=Path)
  parser.add_argument("--multisenge_root", required=True, type=Path)
  parser.add_argument("--manifest", required=True, type=Path)
  parser.add_argument("--output_dir", required=True, type=Path)
  parser.add_argument("--max_patches", type=int, default=0)
  parser.add_argument("--patch_ids", default="", help="Optional comma-separated patch IDs.")
  parser.add_argument("--rank", type=int, default=0, help="Override config subspace rank.")
  parser.add_argument("--preprocessing", default="", help="Override band-image preprocessing.")
  return parser.parse_args()


def _load_config(path: Path) -> dict:
  return yaml.safe_load(path.read_text(encoding="utf-8"))


def _subset_stats(stats, stats_order, desired_order):
  index = {band: i for i, band in enumerate(stats_order)}
  from phase1.data.preprocessing import BandStats

  return BandStats(
    mean=np.array([stats.mean[index[band]] for band in desired_order], dtype=np.float32),
    std=np.array([stats.std[index[band]] for band in desired_order], dtype=np.float32),
    eps=stats.eps,
  )


def _load_tif(path: Path) -> np.ndarray:
  with rasterio.open(path) as source:
    return source.read().astype(np.float32)


def _load_ground_reference(root: Path, tile: str, x: int, y: int) -> Optional[np.ndarray]:
  path = root / "ground_reference" / f"{tile}_GR_{x}_{y}.tif"
  if not path.exists():
    return None
  with rasterio.open(path) as source:
    return source.read(1)


def _majority_label(mask: np.ndarray) -> Optional[int]:
  if mask.size == 0:
    return None
  values, counts = np.unique(mask, return_counts=True)
  return int(values[int(np.argmax(counts))]) if values.size else None


def _safe_corr(left: list[float], right: list[float]) -> float:
  if len(left) != len(right) or len(left) < 3:
    return float("nan")
  a = np.asarray(left, dtype=np.float64)
  b = np.asarray(right, dtype=np.float64)
  if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
    return float("nan")
  return float(np.corrcoef(a, b)[0, 1])


def _normalized_difference(cube: np.ndarray, numerator: int, denominator: int, mask: np.ndarray) -> np.ndarray:
  first = cube[numerator, mask].astype(np.float64)
  second = cube[denominator, mask].astype(np.float64)
  return (first - second) / np.maximum(np.abs(first + second), 1e-6)


def _raw_controls(cubes: list[np.ndarray], mask: np.ndarray, band_order: list[str]) -> dict[str, list[float]]:
  band_index = {name: index for index, name in enumerate(band_order)}
  required = {"B04", "B08", "B12"}
  have_indices = required.issubset(band_index)
  adjacent_rmse: list[float] = []
  adjacent_ndvi_mae: list[float] = []
  adjacent_nbr_mae: list[float] = []
  second_rmse: list[float] = []
  second_ndvi_mae: list[float] = []
  second_nbr_mae: list[float] = []

  values = [cube[:, mask].astype(np.float64) / 10000.0 for cube in cubes]
  ndvi = []
  nbr = []
  if have_indices:
    for cube in cubes:
      ndvi.append(_normalized_difference(cube, band_index["B08"], band_index["B04"], mask))
      nbr.append(_normalized_difference(cube, band_index["B08"], band_index["B12"], mask))

  for index in range(len(cubes) - 1):
    adjacent_rmse.append(float(np.sqrt(np.mean((values[index + 1] - values[index]) ** 2))))
    if have_indices:
      adjacent_ndvi_mae.append(float(np.mean(np.abs(ndvi[index + 1] - ndvi[index]))))
      adjacent_nbr_mae.append(float(np.mean(np.abs(nbr[index + 1] - nbr[index]))))
  for index in range(1, len(cubes) - 1):
    second_rmse.append(float(np.sqrt(np.mean((values[index + 1] - 2.0 * values[index] + values[index - 1]) ** 2))))
    if have_indices:
      second_ndvi_mae.append(float(np.mean(np.abs(ndvi[index + 1] - 2.0 * ndvi[index] + ndvi[index - 1]))))
      second_nbr_mae.append(float(np.mean(np.abs(nbr[index + 1] - 2.0 * nbr[index] + nbr[index - 1]))))
  return {
    "raw_adjacent_rmse": adjacent_rmse,
    "ndvi_adjacent_mae": adjacent_ndvi_mae,
    "nbr_adjacent_mae": adjacent_nbr_mae,
    "raw_second_rmse": second_rmse,
    "ndvi_second_mae": second_ndvi_mae,
    "nbr_second_mae": second_nbr_mae,
  }


def _plot_patch(dates: list[str], metrics: dict[str, list[float]], controls: dict[str, list[float]], path: Path, title: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  times = [datetime.strptime(date, "%Y%m%d") for date in dates]
  pair_times = times[1:]
  triple_times = times[1:-1]
  figure, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

  axes[0].plot(pair_times, metrics["first_adjacent_magnitude"], marker="o", label="1st DS magnitude")
  axes[0].plot(pair_times, metrics["first_geodesic_per_day"], marker=".", label="geodesic distance/day")
  axes[0].set_ylabel("First order")
  axes[0].legend(loc="best")

  axes[1].plot(triple_times, metrics["second_magnitude"], marker="o", label="2nd DS total")
  axes[1].plot(triple_times, metrics["second_along_geodesic"], linestyle="--", label="along geodesic")
  axes[1].plot(triple_times, metrics["second_orthogonal_geodesic"], linestyle="--", label="orthogonal")
  axes[1].set_ylabel("Second order")
  axes[1].legend(loc="best")

  axes[2].plot(
    triple_times,
    metrics["time_aware_geodesic_deviation"],
    marker="o",
    color="tab:green",
    label="time-aware geodesic deviation",
  )
  axes[2].set_ylabel("Irregular-cadence\ndiagnostic")
  axes[2].legend(loc="best")

  axes[3].plot(pair_times, controls["raw_adjacent_rmse"], marker="o", label="raw reflectance RMSE")
  if controls["ndvi_adjacent_mae"]:
    axes[3].plot(pair_times, controls["ndvi_adjacent_mae"], label="NDVI MAE")
    axes[3].plot(pair_times, controls["nbr_adjacent_mae"], label="NBR MAE")
  axes[3].set_ylabel("Classical controls")
  axes[3].legend(loc="best")
  for axis in axes:
    axis.grid(True, alpha=0.3)
  figure.suptitle(title)
  figure.autofmt_xdate()
  figure.tight_layout()
  figure.savefig(path, dpi=170)
  plt.close(figure)


def _plot_rgb_montage(cubes: list[np.ndarray], dates: list[str], valid_fractions: list[float], path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  columns = 5
  rows = int(np.ceil(len(cubes) / columns))
  figure, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 3.0 * rows))
  axes = np.asarray(axes).reshape(-1)
  for axis, cube, date, valid_fraction in zip(axes, cubes, dates, valid_fractions):
    rgb = np.moveaxis(cube[[2, 1, 0]], 0, -1).astype(np.float64)
    valid = np.all(rgb > 0, axis=-1)
    if np.any(valid):
      low, high = np.percentile(rgb[valid], [2, 98])
      rgb = np.clip((rgb - low) / max(high - low, 1e-6), 0.0, 1.0)
    axis.imshow(rgb)
    axis.set_title(f"{date}\nvalid={valid_fraction:.2f}", fontsize=8)
    axis.axis("off")
  for axis in axes[len(cubes):]:
    axis.axis("off")
  figure.tight_layout()
  figure.savefig(path, dpi=150)
  plt.close(figure)


def _legacy_pixel_bases(cubes: list[np.ndarray], cfg: dict, seed: int) -> list[np.ndarray]:
  desired_order = cfg["dataset"].get("band_order")
  stats = load_band_stats(resolve_phase1_path(Path(cfg["normalization"]["stats_path"])))
  stats_order = cfg["normalization"].get("stats_band_order", desired_order)
  if desired_order is not None and stats_order != desired_order:
    stats = _subset_stats(stats, stats_order, desired_order)
  nodata = float(cfg["dataset"].get("nodata_value", 0.0))
  minimum = int(cfg["dataset"].get("min_valid_bands", 3))
  rank = int(cfg["subspace"].get("rank_r", 6))
  bases = []
  for cube in cubes:
    valid = build_valid_mask(cube, nodata_value=nodata, min_valid_bands=minimum)
    normalized, valid = apply_normalization(cube, stats, valid_mask=valid, nodata_value=nodata)
    matrix, _ = vectorize_cube(normalized, valid)
    bases.append(fit_pca_basis(matrix, rank=rank, variance_threshold=None, random_state=seed).basis)
  return bases


def main() -> None:
  args = parse_args()
  cfg = _load_config(args.config)
  output = Path(args.output_dir)
  output.mkdir(parents=True, exist_ok=True)
  root = Path(args.multisenge_root)
  patches = list(load_manifest(args.manifest).get("patches", []))
  requested = {value.strip() for value in args.patch_ids.split(",") if value.strip()}
  if requested:
    patches = [entry for entry in patches if entry.get("patch_id") in requested]
  if args.max_patches > 0:
    patches = patches[: args.max_patches]

  construction = str(cfg.get("subspace", {}).get("construction", "band_image")).strip().lower()
  preprocessing = str(args.preprocessing or cfg.get("subspace", {}).get("preprocessing", "centered_band_l2"))
  rank = int(args.rank or cfg.get("subspace", {}).get("rank_r", 6))
  nodata = float(cfg.get("dataset", {}).get("nodata_value", 0.0))
  band_order = list(cfg.get("dataset", {}).get("band_order", []))
  min_common = int(cfg.get("subspace", {}).get("min_common_pixels", 1024))
  balanced_ratio = float(cfg.get("temporal", {}).get("balanced_gap_ratio", 1.5))
  seed = int(cfg.get("seed", 1234))

  summary_rows: list[dict] = []
  triple_rows: list[dict] = []
  for entry in patches:
    patch_id = entry["patch_id"]
    s2_entries = sorted(entry.get("s2", []), key=lambda item: item.get("date", ""))
    dates = [str(item.get("date") or Path(item["relpath"]).name.split("_")[1]) for item in s2_entries]
    paths = [root / item["relpath"] for item in s2_entries]
    if len(paths) < 3 or any(not path.exists() for path in paths):
      print(f"[WARN] {patch_id}: fewer than three dates or missing files; skipped")
      continue
    cubes = [_load_tif(path) for path in paths]
    valid_fractions = [float(np.mean(np.all((cube != nodata) & np.isfinite(cube), axis=0))) for cube in cubes]
    common_mask = sequence_common_valid_mask(cubes, nodata_value=nodata, require_all_bands=True)
    common_count = int(np.sum(common_mask))
    if common_count < min_common:
      print(f"[WARN] {patch_id}: only {common_count} common valid pixels; skipped")
      continue

    explained = []
    if construction == "band_image":
      fitted = [
        build_band_image_subspace(cube / 10000.0, common_mask, rank=rank, preprocessing=preprocessing)
        for cube in cubes
      ]
      bases = [item.basis for item in fitted]
      explained = [float(np.sum(item.explained_energy_ratio)) for item in fitted]
      basis_shape = list(fitted[0].basis.shape)
    elif construction == "legacy_pixel_spectral":
      bases = _legacy_pixel_bases(cubes, cfg, seed)
      basis_shape = list(bases[0].shape)
    else:
      raise ValueError(f"Unknown subspace construction: {construction}")

    metrics = temporal_difference_measurements(bases, dates)
    controls = _raw_controls(cubes, common_mask, band_order)
    balanced = [ratio <= balanced_ratio for ratio in metrics["second_gap_ratio"]]
    gr = _load_ground_reference(root, entry.get("tile", ""), int(entry.get("x", 0)), int(entry.get("y", 0)))
    majority = _majority_label(gr) if gr is not None else None
    orth_fraction = [
      orth / max(along + orth, 1e-12)
      for orth, along in zip(metrics["second_orthogonal_geodesic"], metrics["second_along_geodesic"])
    ]

    payload = {
      "patch_id": patch_id,
      "dates": dates,
      "labels_present": entry.get("labels_present", []),
      "ground_reference_majority": majority,
      "construction_card": {
        "variant": construction,
        "sample_unit": "one complete aligned band image per date" if construction == "band_image" else "one pixel spectrum",
        "date_matrix_shape": [common_count, len(band_order)] if construction == "band_image" else [len(band_order), common_count],
        "basis_shape": basis_shape,
        "preprocessing": preprocessing if construction == "band_image" else "legacy centered spectral PCA",
        "spatial_information": "fixed valid pixel coordinates retained as ambient dimensions" if construction == "band_image" else "pixel positions discarded during fitting",
      },
      "common_valid_pixels": common_count,
      "common_valid_fraction": float(np.mean(common_mask)),
      "per_date_valid_fraction": valid_fractions,
      "explained_energy": explained,
      "balanced_gap_ratio_threshold": balanced_ratio,
      "balanced_triples": balanced,
      "measurements": metrics,
      "controls": controls,
      "orthogonal_fraction": orth_fraction,
    }
    (output / "per_patch").mkdir(parents=True, exist_ok=True)
    (output / "per_patch" / f"{patch_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _plot_patch(dates, metrics, controls, output / "plots" / f"{patch_id}.png", f"{patch_id}: {construction}, rank={rank}")
    _plot_rgb_montage(cubes, dates, valid_fractions, output / "montages" / f"{patch_id}.png")

    for index, date in enumerate(dates[1:-1]):
      triple_rows.append({
        "patch_id": patch_id,
        "date": date,
        "gap_left_days": metrics["second_gap_left_days"][index],
        "gap_right_days": metrics["second_gap_right_days"][index],
        "gap_ratio": metrics["second_gap_ratio"][index],
        "balanced_gap": int(balanced[index]),
        "first_endpoint_span_magnitude": metrics["first_endpoint_span_magnitude"][index],
        "first_speed_change": metrics["first_speed_change"][index],
        "second_magnitude": metrics["second_magnitude"][index],
        "second_along": metrics["second_along_geodesic"][index],
        "second_orthogonal": metrics["second_orthogonal_geodesic"][index],
        "time_fraction": metrics["second_time_fraction"][index],
        "time_aware_geodesic_deviation_magnitude": metrics[
          "time_aware_geodesic_deviation_magnitude"
        ][index],
        "time_aware_geodesic_deviation": metrics["time_aware_geodesic_deviation"][index],
        "time_aware_acceleration_proxy": metrics["time_aware_acceleration_proxy"][index],
        "orthogonal_fraction": orth_fraction[index],
        "decomposition_residual": metrics["second_decomposition_residual"][index],
        "raw_second_rmse": controls["raw_second_rmse"][index],
        "ndvi_second_mae": controls["ndvi_second_mae"][index] if controls["ndvi_second_mae"] else "",
        "nbr_second_mae": controls["nbr_second_mae"][index] if controls["nbr_second_mae"] else "",
      })

    summary_rows.append({
      "patch_id": patch_id,
      "n_dates": len(dates),
      "common_valid_pixels": common_count,
      "common_valid_fraction": float(np.mean(common_mask)),
      "majority_gr": majority if majority is not None else "",
      "mean_first_magnitude": float(np.mean(metrics["first_adjacent_magnitude"])),
      "mean_second_magnitude": float(np.mean(metrics["second_magnitude"])),
      "mean_second_orthogonal": float(np.mean(metrics["second_orthogonal_geodesic"])),
      "mean_time_aware_geodesic_deviation": float(
        np.mean(metrics["time_aware_geodesic_deviation"])
      ),
      "mean_orthogonal_fraction": float(np.mean(orth_fraction)),
      "balanced_triple_count": int(np.sum(balanced)),
      "corr_first_vs_raw": _safe_corr(metrics["first_adjacent_magnitude"], controls["raw_adjacent_rmse"]),
      "corr_second_vs_speed_change": _safe_corr(metrics["second_magnitude"], metrics["first_speed_change"]),
      "corr_second_vs_raw_second": _safe_corr(metrics["second_magnitude"], controls["raw_second_rmse"]),
      "corr_time_aware_vs_raw_second": _safe_corr(
        metrics["time_aware_geodesic_deviation"], controls["raw_second_rmse"]
      ),
    })
    print(f"[OK] {patch_id}: {len(dates)} dates, {common_count} common pixels, {sum(balanced)} balanced triples")

  for filename, rows in (("summary.csv", summary_rows), ("triples.csv", triple_rows)):
    if rows:
      with (output / filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
  metadata = {
    "config": str(args.config),
    "construction": construction,
    "preprocessing": preprocessing,
    "rank": rank,
    "balanced_gap_ratio": balanced_ratio,
    "patches_completed": len(summary_rows),
    "interpretation_warning": "MultiSenGE land-cover labels are static; temporal curves are descriptive, not supervised change accuracy.",
    "time_aware_metric": (
      "The irregular-cadence metric compares the observed middle subspace with "
      "the constant-speed endpoint geodesic at the observed acquisition fraction. "
      "It is a project diagnostic, not the paper-defined equal-spacing second-order DS."
    ),
  }
  (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
  print(f"Done. Outputs in: {output}")


if __name__ == "__main__":
  main()
