"""Evaluate local order-aware subspaces on controlled MultiSenGE interventions.

Source/provenance:
- Kanai et al. (2023) supplies the scalar SSA trajectory-matrix idea.
- ``phase1.subspace.temporal_trajectory`` implements the explicitly labeled
  multivariate satellite adaptation.
- Fukui et al. (2024) supplies first/second Difference Subspace quantities.

This script adds a project-designed spatial grid. Each cell independently
constructs a temporal representation subspace, so a localized seasonal-mode
intervention is not diluted into one whole-crop basis. Exact injected masks
provide pixel-weighted localization labels; stable and nuisance interventions
provide negatives. This remains controlled real-background evidence, not
accuracy on naturally occurring change labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score

from phase1.scripts.evaluate_multisenge_order_aware_interventions import (
  NUISANCES,
  _find_crop,
  _jitter,
  _load_cube,
  _load_reference,
  _parse_representation,
  _raw_controls,
  _scenario_spec,
  _transform,
)
from phase1.data.multisenge_manifest import load_manifest
from phase1.subspace.geodesic import grassmann_geodesic_distance, subspace_magnitude
from phase1.subspace.second_order_ds import second_order_difference_subspace
from phase1.subspace.temporal_band_images import sequence_common_valid_mask
from phase1.subspace.temporal_trajectory import (
  build_temporal_representation_subspace,
  representation_covariance_change,
  representation_energy_change,
  representation_spectrum_change,
)


SCENARIOS = (
  "stable_jitter",
  "global_gain_offset",
  "phase_shift",
  "missing_composites",
  "translation_1px",
  "seasonal_amplitude_change",
  "localized_seasonal_mode",
)
SCORES = (
  "first_ds_magnitude",
  "first_geodesic",
  "second_along",
  "second_orthogonal",
  "representation_energy_change",
  "representation_spectrum_change",
  "representation_covariance_change",
  "raw_sequence_rms",
  "raw_difference_rms",
  "ndvi_curve_rms",
  "ndvi_amplitude_change",
  "ndmi_amplitude_change",
  "nbr_amplitude_change",
  "band_mean_curve_rms",
  "band_amplitude_change",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--multisenge_root", required=True, type=Path)
  parser.add_argument("--manifest", required=True, type=Path)
  parser.add_argument("--output_dir", required=True, type=Path)
  parser.add_argument("--crop_size", type=int, default=32)
  parser.add_argument("--repeats", type=int, default=4)
  parser.add_argument("--grids", default="2,4")
  parser.add_argument("--representations", default="unordered,trajectory2")
  parser.add_argument("--rank", type=int, default=1)
  parser.add_argument("--preprocessing", default="feature_centered_observation_l2")
  parser.add_argument("--spatial_smoothing_sigma", type=float, default=0.0)
  parser.add_argument("--bootstrap", type=int, default=300)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--max_patches", type=int, default=5)
  return parser.parse_args()


def _cells(height: int, width: int, grid: int):
  row_edges = np.linspace(0, height, grid + 1, dtype=int)
  col_edges = np.linspace(0, width, grid + 1, dtype=int)
  for row in range(grid):
    for col in range(grid):
      yield row, col, slice(row_edges[row], row_edges[row + 1]), slice(col_edges[col], col_edges[col + 1])


def _random_local_mask(
  reference: np.ndarray | None,
  valid: np.ndarray,
  rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
  """Create an off-grid event support, optionally restricted to one class."""
  height, width = valid.shape
  side = max(min(height, width) // 2, 4)
  top = int(rng.integers(0, height - side + 1))
  left = int(rng.integers(0, width - side + 1))
  support = np.zeros_like(valid, dtype=bool)
  support[top : top + side, left : left + side] = True
  support &= valid
  if reference is None:
    return support, -1
  values = reference[support]
  values = values[values > 0]
  if values.size == 0:
    return support, -1
  labels, counts = np.unique(values, return_counts=True)
  dominant = int(labels[int(np.argmax(counts))])
  semantic = support & (reference == dominant)
  if int(np.sum(semantic)) < 16:
    return support, dominant
  return semantic, dominant


def _smooth(sequence: list[np.ndarray], sigma: float) -> list[np.ndarray]:
  if sigma <= 0.0:
    return sequence
  return [
    ndimage.gaussian_filter(cube, sigma=(0.0, sigma, sigma), mode="nearest").astype(np.float32)
    for cube in sequence
  ]


def _metric_arrays(rows: list[dict], score: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  labels, values, weights = [], [], []
  for row in rows:
    positive = int(row["positive_pixels"])
    negative = int(row["negative_pixels"])
    value = float(row[score])
    if positive:
      labels.append(1)
      values.append(value)
      weights.append(positive)
    if negative:
      labels.append(0)
      values.append(value)
      weights.append(negative)
  return np.asarray(labels), np.asarray(values), np.asarray(weights, dtype=np.float64)


def _tasks(rows: list[dict]) -> dict[str, list[dict]]:
  negatives = NUISANCES | {"stable_jitter"}
  return {
    "localized_event_pixels_vs_stable_and_nuisance": [
      row for row in rows if row["scenario"] in negatives | {"localized_seasonal_mode"}
    ],
    "global_amplitude_pixels_vs_stable_and_nuisance": [
      row for row in rows if row["scenario"] in negatives | {"seasonal_amplitude_change"}
    ],
    "all_event_pixels_vs_stable_and_nuisance": list(rows),
  }


def _summaries(rows: list[dict], bootstrap: int, rng: np.random.Generator) -> list[dict]:
  output = []
  groups = sorted({(row["representation"], int(row["grid"]), int(row["rank"])) for row in rows})
  for representation, grid, rank in groups:
    grouped = [row for row in rows if row["representation"] == representation and int(row["grid"]) == grid and int(row["rank"]) == rank]
    for task, selected in _tasks(grouped).items():
      patch_ids = sorted({row["patch_id"] for row in selected})
      for score in SCORES:
        labels, values, weights = _metric_arrays(selected, score)
        if np.unique(labels).size < 2:
          continue
        boot_auc, boot_ap = [], []
        for _ in range(max(int(bootstrap), 0)):
          sampled = rng.choice(patch_ids, size=len(patch_ids), replace=True)
          sample_rows = [row for patch in sampled for row in selected if row["patch_id"] == patch]
          y, x, w = _metric_arrays(sample_rows, score)
          if np.unique(y).size < 2:
            continue
          boot_auc.append(float(roc_auc_score(y, x, sample_weight=w)))
          boot_ap.append(float(average_precision_score(y, x, sample_weight=w)))
        output.append({
          "representation": representation,
          "grid": grid,
          "rank": rank,
          "preprocessing": grouped[0]["preprocessing"],
          "task": task,
          "score": score,
          "positive_pixels": int(np.sum(weights[labels == 1])),
          "negative_pixels": int(np.sum(weights[labels == 0])),
          "auroc": float(roc_auc_score(labels, values, sample_weight=weights)),
          "average_precision": float(average_precision_score(labels, values, sample_weight=weights)),
          "auroc_ci_low": float(np.quantile(boot_auc, 0.025)) if boot_auc else float("nan"),
          "auroc_ci_high": float(np.quantile(boot_auc, 0.975)) if boot_auc else float("nan"),
          "average_precision_ci_low": float(np.quantile(boot_ap, 0.025)) if boot_ap else float("nan"),
          "average_precision_ci_high": float(np.quantile(boot_ap, 0.975)) if boot_ap else float("nan"),
        })
  return output


def _false_alarm_summary(rows: list[dict]) -> list[dict]:
  output = []
  groups = sorted({(row["representation"], int(row["grid"]), int(row["rank"])) for row in rows})
  for representation, grid, rank in groups:
    selected = [row for row in rows if row["representation"] == representation and int(row["grid"]) == grid and int(row["rank"]) == rank]
    for score in SCORES:
      stable = np.asarray([float(row[score]) for row in selected if row["scenario"] == "stable_jitter"])
      threshold = float(np.quantile(stable, 0.95))
      for scenario in sorted(NUISANCES):
        nuisance = [row for row in selected if row["scenario"] == scenario]
        total = sum(int(row["positive_pixels"]) + int(row["negative_pixels"]) for row in nuisance)
        alarms = sum(
          (int(row["positive_pixels"]) + int(row["negative_pixels"]))
          for row in nuisance
          if float(row[score]) > threshold
        )
        output.append({
          "representation": representation,
          "grid": grid,
          "rank": rank,
          "score": score,
          "scenario": scenario,
          "stable_95pct_threshold": threshold,
          "false_alarm_pixel_rate": float(alarms / max(total, 1)),
        })
  return output


def _paired_comparisons(
  rows: list[dict],
  bootstrap: int,
  rng: np.random.Generator,
) -> list[dict]:
  pairs = (
    ("representation_spectrum_change", "ndmi_amplitude_change"),
    ("representation_spectrum_change", "first_ds_magnitude"),
    ("first_ds_magnitude", "ndmi_amplitude_change"),
  )
  output = []
  groups = sorted({(row["representation"], int(row["grid"]), int(row["rank"])) for row in rows})
  for representation, grid, rank in groups:
    grouped = [row for row in rows if row["representation"] == representation and int(row["grid"]) == grid and int(row["rank"]) == rank]
    selected = _tasks(grouped)["localized_event_pixels_vs_stable_and_nuisance"]
    patch_ids = sorted({row["patch_id"] for row in selected})
    for first_score, second_score in pairs:
      labels, first_values, weights = _metric_arrays(selected, first_score)
      _, second_values, _ = _metric_arrays(selected, second_score)
      first_ap = float(average_precision_score(labels, first_values, sample_weight=weights))
      second_ap = float(average_precision_score(labels, second_values, sample_weight=weights))
      differences = []
      for _ in range(max(int(bootstrap), 0)):
        sampled = rng.choice(patch_ids, size=len(patch_ids), replace=True)
        sample_rows = [row for patch in sampled for row in selected if row["patch_id"] == patch]
        y, first_x, w = _metric_arrays(sample_rows, first_score)
        _, second_x, _ = _metric_arrays(sample_rows, second_score)
        differences.append(
          float(average_precision_score(y, first_x, sample_weight=w))
          - float(average_precision_score(y, second_x, sample_weight=w))
        )
      output.append({
        "representation": representation,
        "grid": grid,
        "rank": rank,
        "task": "localized_event_pixels_vs_stable_and_nuisance",
        "first_score": first_score,
        "second_score": second_score,
        "first_average_precision": first_ap,
        "second_average_precision": second_ap,
        "delta_average_precision": first_ap - second_ap,
        "delta_ci_low": float(np.quantile(differences, 0.025)) if differences else float("nan"),
        "delta_ci_high": float(np.quantile(differences, 0.975)) if differences else float("nan"),
        "bootstrap_probability_delta_positive": float(np.mean(np.asarray(differences) > 0.0)) if differences else float("nan"),
      })
  return output


def _write_csv(path: Path, rows: list[dict]) -> None:
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def _fill_map(cell_rows: list[dict], score: str, size: int) -> np.ndarray:
  output = np.full((size, size), np.nan, dtype=np.float32)
  for row in cell_rows:
    output[int(row["row_start"]):int(row["row_stop"]), int(row["col_start"]):int(row["col_stop"])] = float(row[score])
  return output


def _save_example(example: dict, rows: list[dict], output: Path) -> None:
  representations = sorted({row["representation"] for row in rows})
  representation = "trajectory2" if "trajectory2" in representations else representations[0]
  grid = max(int(row["grid"]) for row in rows if row["representation"] == representation)
  selected = [
    row for row in rows
    if row["sample_id"] == example["sample_id"]
    and row["representation"] == representation
    and int(row["grid"]) == grid
  ]
  if not selected:
    return
  local = [row for row in selected if row["scenario"] == "localized_seasonal_mode"]
  translation = [row for row in selected if row["scenario"] == "translation_1px"]
  panels = [
    (example["rgb"], "Temporal-mean RGB", None),
    (example["local_mask"], "Injected local support", "gray"),
    (_fill_map(local, "first_ds_magnitude", example["size"]), f"Local {representation} DS", "magma"),
    (_fill_map(local, "representation_spectrum_change", example["size"]), "Local spectrum change", "magma"),
    (_fill_map(local, "ndvi_amplitude_change", example["size"]), "Local NDVI amplitude", "magma"),
    (_fill_map(translation, "first_ds_magnitude", example["size"]), "Translation nuisance DS", "magma"),
  ]
  figure, axes = plt.subplots(2, 3, figsize=(12, 8))
  for axis, (image, title, cmap) in zip(axes.flat, panels):
    axis.imshow(image, cmap=cmap)
    axis.set_title(title)
    axis.axis("off")
  figure.suptitle(f"Controlled local temporal example: {example['sample_id']} ({grid}x{grid})")
  figure.tight_layout()
  figure.savefig(output, dpi=180)
  plt.close(figure)


def main() -> None:
  started = time.perf_counter()
  args = parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  rng = np.random.default_rng(args.seed)
  grids = sorted({int(value) for value in args.grids.split(",") if value.strip()})
  representations = [_parse_representation(value) for value in args.representations.split(",") if value.strip()]
  rows: list[dict] = []
  patch_count = 0
  example = None

  for entry in load_manifest(args.manifest).get("patches", []):
    ordered = sorted(entry.get("s2", []), key=lambda item: item["date"])
    if len(ordered) < 8:
      continue
    full_sequence = [_load_cube(args.multisenge_root / item["relpath"]) for item in ordered]
    full_mask = sequence_common_valid_mask(full_sequence, nodata_value=0.0)
    reference = _load_reference(args.multisenge_root, entry["patch_id"])
    for repeat in range(args.repeats):
      crop = _find_crop(full_mask, args.crop_size, rng)
      sequence = [cube[:, crop[0], crop[1]] for cube in full_sequence]
      mask = full_mask[crop[0], crop[1]]
      crop_reference = reference[crop[0], crop[1]] if reference is not None else None
      local_mask, dominant_class = _random_local_mask(crop_reference, mask, rng)
      first = _jitter(sequence, rng)
      second_base = _jitter(sequence, rng)
      third_base = _jitter(sequence, rng)
      sample_id = f"{entry['patch_id']}:{repeat}"
      if example is None:
        mean_cube = np.mean(np.stack(first), axis=0)
        rgb = np.moveaxis(mean_cube[[2, 1, 0]], 0, -1)
        low, high = np.quantile(rgb[mask], [0.02, 0.98])
        example = {"sample_id": sample_id, "rgb": np.clip((rgb - low) / max(high - low, 1e-6), 0, 1), "local_mask": local_mask, "size": args.crop_size}

      transformed = {}
      for scenario in SCENARIOS:
        spec = _scenario_spec(scenario, len(sequence), rng)
        transformed[scenario] = tuple(_smooth(candidate, args.spatial_smoothing_sigma) for candidate in (
          _transform(second_base, kind=scenario, spec=spec, local_mask=local_mask),
          _transform(third_base, kind=scenario, spec=spec, local_mask=local_mask),
        ))
      first = _smooth(first, args.spatial_smoothing_sigma)

      for grid in grids:
        for row_index, col_index, row_slice, col_slice in _cells(args.crop_size, args.crop_size, grid):
          cell_mask = mask[row_slice, col_slice]
          valid_pixels = int(np.sum(cell_mask))
          if valid_pixels < 8:
            continue
          first_cell = [cube[:, row_slice, col_slice] for cube in first]
          local_cell = local_mask[row_slice, col_slice] & cell_mask
          for representation, lag, representation_name in representations:
            fitted_first = build_temporal_representation_subspace(
              first_cell, cell_mask, rank=args.rank, representation=representation,
              lag=lag, preprocessing=args.preprocessing,
            )
            for scenario, (second, third) in transformed.items():
              second_cell = [cube[:, row_slice, col_slice] for cube in second]
              third_cell = [cube[:, row_slice, col_slice] for cube in third]
              fitted_second = build_temporal_representation_subspace(
                second_cell, cell_mask, rank=args.rank, representation=representation,
                lag=lag, preprocessing=args.preprocessing,
              )
              fitted_third = build_temporal_representation_subspace(
                third_cell, cell_mask, rank=args.rank, representation=representation,
                lag=lag, preprocessing=args.preprocessing,
              )
              effective_rank = min(fitted_first.rank, fitted_second.rank, fitted_third.rank)
              if effective_rank < 1:
                continue
              first_basis = fitted_first.basis[:, :effective_rank]
              second_basis = fitted_second.basis[:, :effective_rank]
              third_basis = fitted_third.basis[:, :effective_rank]
              second_order = second_order_difference_subspace(first_basis, second_basis, third_basis, decompose=True)
              controls = _raw_controls(first_cell, second_cell, cell_mask)
              if scenario == "localized_seasonal_mode":
                positive_pixels = int(np.sum(local_cell))
              elif scenario == "seasonal_amplitude_change":
                positive_pixels = valid_pixels
              else:
                positive_pixels = 0
              rows.append({
                "sample_id": sample_id,
                "patch_id": entry["patch_id"],
                "repeat": repeat,
                "scenario": scenario,
                "dominant_reference_class": dominant_class,
                "representation": representation_name,
                "lag": lag,
                "preprocessing": args.preprocessing,
                "rank": effective_rank,
                "grid": grid,
                "cell_row": row_index,
                "cell_col": col_index,
                "row_start": row_slice.start,
                "row_stop": row_slice.stop,
                "col_start": col_slice.start,
                "col_stop": col_slice.stop,
                "positive_pixels": positive_pixels,
                "negative_pixels": valid_pixels - positive_pixels,
                "first_ds_magnitude": subspace_magnitude(first_basis, second_basis) / effective_rank,
                "first_geodesic": grassmann_geodesic_distance(first_basis, second_basis) / np.sqrt(effective_rank),
                "second_along": float(second_order.mag_along or 0.0) / effective_rank,
                "second_orthogonal": float(second_order.mag_orth or 0.0) / effective_rank,
                "representation_energy_change": representation_energy_change(fitted_first, fitted_second),
                "representation_spectrum_change": representation_spectrum_change(fitted_first, fitted_second),
                "representation_covariance_change": representation_covariance_change(fitted_first, fitted_second),
                **controls,
              })
    patch_count += 1
    print(f"[OK] {entry['patch_id']}: dates={len(ordered)}, repeats={args.repeats}")
    if args.max_patches and patch_count >= args.max_patches:
      break

  if not rows:
    raise RuntimeError("No multiscale temporal cells were evaluated.")
  summary = _summaries(rows, args.bootstrap, rng)
  false_alarms = _false_alarm_summary(rows)
  paired = _paired_comparisons(rows, args.bootstrap, np.random.default_rng(args.seed + 1))
  _write_csv(args.output_dir / "cell_scores.csv", rows)
  _write_csv(args.output_dir / "pixel_weighted_summary.csv", summary)
  _write_csv(args.output_dir / "nuisance_false_alarm_summary.csv", false_alarms)
  _write_csv(args.output_dir / "paired_score_comparisons.csv", paired)
  if example is not None:
    _save_example(example, rows, args.output_dir / "localized_event_example.png")
  metadata = {
    "study": "multiscale order-aware subspaces on controlled MultiSenGE interventions",
    "patches": patch_count,
    "repeats_per_patch": args.repeats,
    "grids": grids,
    "representations": [name for _, _, name in representations],
    "rank": args.rank,
    "crop_size": args.crop_size,
    "preprocessing": args.preprocessing,
    "spatial_smoothing_sigma": args.spatial_smoothing_sigma,
    "runtime_seconds": time.perf_counter() - started,
    "scenarios": list(SCENARIOS),
    "bootstrap_unit": "MultiSenGE patch",
    "claim_boundary": "Controlled real-background localization, not natural change accuracy.",
  }
  (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
  print(json.dumps({"output_dir": str(args.output_dir), "rows": len(rows), "patches": patch_count}, indent=2))


if __name__ == "__main__":
  main()
