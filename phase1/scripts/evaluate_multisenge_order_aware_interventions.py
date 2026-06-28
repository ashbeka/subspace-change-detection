"""Order-aware seasonal-subspace stress test on real MultiSenGE backgrounds.

This is a controlled intervention study, not natural change-detection accuracy.
Real 23-date Sentinel-2 sequences provide spatial texture, spectral covariance,
no-data patterns, and land-cover context.  Known transformations then test:

- stable acquisition jitter;
- global gain/offset, phase shift, missing composites, and translation nuisances;
- seasonal-amplitude and localized seasonal-mode events;
- date permutation as an explicit temporal-order diagnostic.

The compared representations are unordered observations, temporal first
differences, and multivariate block-Hankel trajectory matrices.  The latter is
a project adaptation of the scalar SSA trajectory matrix in Kanai et al.
(2023), Eq. (1).  First/second Difference Subspace quantities follow the
project's paper-checked Fukui implementations.  Raw reflectance and NDVI
controls are retained to prevent a geometry-versus-itself evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

# Repeated small SVDs are faster and more stable without nested BLAS workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score

from phase1.data.multisenge_manifest import load_manifest
from phase1.subspace.geodesic import (
  grassmann_geodesic_distance,
  principal_angles,
  subspace_magnitude,
)
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
  "date_permutation",
)
EVENTS = {"seasonal_amplitude_change", "localized_seasonal_mode"}
NUISANCES = {
  "global_gain_offset",
  "phase_shift",
  "missing_composites",
  "translation_1px",
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--multisenge_root", required=True, type=Path)
  parser.add_argument("--manifest", required=True, type=Path)
  parser.add_argument("--output_dir", required=True, type=Path)
  parser.add_argument("--crop_size", type=int, default=32)
  parser.add_argument("--repeats", type=int, default=8)
  parser.add_argument("--ranks", default="1,2")
  parser.add_argument(
    "--representations",
    default="unordered,difference,trajectory2,trajectory3",
  )
  parser.add_argument(
    "--preprocessing",
    default="feature_centered_observation_l2",
  )
  parser.add_argument("--bootstrap", type=int, default=300)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--max_patches", type=int, default=5)
  return parser.parse_args()


def _load_cube(path: Path) -> np.ndarray:
  with rasterio.open(path) as source:
    return source.read().astype(np.float32) / 10000.0


def _load_reference(root: Path, patch_id: str) -> np.ndarray | None:
  tile, x, y = patch_id.split("_")
  path = root / "ground_reference" / f"{tile}_GR_{x}_{y}.tif"
  if not path.exists():
    return None
  with rasterio.open(path) as source:
    return source.read(1)


def _find_crop(mask: np.ndarray, size: int, rng: np.random.Generator) -> tuple[slice, slice]:
  height, width = mask.shape
  if size > height or size > width:
    raise ValueError(f"crop_size={size} exceeds image shape={mask.shape}")
  candidates = np.argwhere(mask)
  for _ in range(1000):
    row, col = candidates[int(rng.integers(0, len(candidates)))]
    top = int(np.clip(row - size // 2, 0, height - size))
    left = int(np.clip(col - size // 2, 0, width - size))
    if float(np.mean(mask[top : top + size, left : left + size])) >= 0.95:
      return slice(top, top + size), slice(left, left + size)
  raise RuntimeError(f"No >=95% valid {size}x{size} crop found.")


def _jitter(sequence: list[np.ndarray], rng: np.random.Generator) -> list[np.ndarray]:
  output = []
  for cube in sequence:
    gain = float(rng.normal(1.0, 0.006))
    offset = float(rng.normal(0.0, 0.0008))
    noise = rng.normal(0.0, 0.0012, size=cube.shape)
    output.append(np.clip(gain * cube + offset + noise, 0.0001, 1.5).astype(np.float32))
  return output


def _dominant_local_mask(reference: np.ndarray | None, valid: np.ndarray) -> tuple[np.ndarray, int]:
  height, width = valid.shape
  support = np.zeros_like(valid, dtype=bool)
  margin_y = max(height // 4, 1)
  margin_x = max(width // 4, 1)
  support[margin_y : height - margin_y, margin_x : width - margin_x] = True
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


def _scenario_spec(kind: str, n_dates: int, rng: np.random.Generator) -> dict:
  if kind == "missing_composites":
    return {"indices": sorted(int(v) for v in rng.choice(np.arange(1, n_dates - 1), 2, replace=False))}
  if kind == "date_permutation":
    order = np.arange(n_dates)
    rng.shuffle(order)
    return {"order": order.tolist()}
  return {}


def _transform(
  sequence: list[np.ndarray],
  *,
  kind: str,
  spec: dict,
  local_mask: np.ndarray,
) -> list[np.ndarray]:
  output = [cube.copy() for cube in sequence]
  if kind == "stable_jitter":
    return output
  if kind == "global_gain_offset":
    return [np.clip(1.08 * cube + 0.008, 0.0001, 1.5).astype(np.float32) for cube in output]
  if kind == "phase_shift":
    return output[2:] + output[:2]
  if kind == "missing_composites":
    for index in spec["indices"]:
      output[index] = 0.5 * (output[index - 1] + output[index + 1])
    return output
  if kind == "translation_1px":
    return [
      ndimage.shift(cube, shift=(0, 0, 1), order=1, mode="nearest").astype(np.float32)
      for cube in output
    ]
  if kind == "seasonal_amplitude_change":
    mean = np.mean(np.stack(output), axis=0)
    return [np.clip(mean + 1.45 * (cube - mean), 0.0001, 1.5).astype(np.float32) for cube in output]
  if kind == "localized_seasonal_mode":
    start = max(len(output) // 4, 1)
    stop = min(3 * len(output) // 4, len(output))
    for index in range(start, stop):
      phase = np.sin(np.pi * (index - start + 1) / (stop - start + 1))
      cube = output[index]
      cube[2, local_mask] *= 1.0 + 0.12 * phase  # red
      cube[6, local_mask] *= 1.0 - 0.35 * phase  # broad NIR
      cube[7, local_mask] *= 1.0 - 0.25 * phase  # narrow NIR
      cube[8, local_mask] *= 1.0 + 0.20 * phase  # SWIR1
      cube[9, local_mask] *= 1.0 + 0.28 * phase  # SWIR2
      output[index] = np.clip(cube, 0.0001, 1.5).astype(np.float32)
    return output
  if kind == "date_permutation":
    return [output[index] for index in spec["order"]]
  raise ValueError(kind)


def _parse_representation(value: str) -> tuple[str, int, str]:
  key = value.strip().lower().replace("-", "_")
  if key == "unordered":
    return "unordered", 1, "unordered"
  if key in {"difference", "first_difference"}:
    return "difference", 1, "difference"
  if key.startswith("trajectory"):
    suffix = key.removeprefix("trajectory").lstrip("_")
    lag = int(suffix) if suffix else 3
    return "trajectory", lag, f"trajectory{lag}"
  raise ValueError(value)


def _ndvi_curve(sequence: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
  values = []
  for cube in sequence:
    nir = cube[6, mask].astype(np.float64)
    red = cube[2, mask].astype(np.float64)
    values.append(float(np.mean((nir - red) / np.maximum(nir + red, 1e-6))))
  return np.asarray(values)


def _normalized_difference_curve(
  sequence: list[np.ndarray],
  mask: np.ndarray,
  first_band: int,
  second_band: int,
) -> np.ndarray:
  values = []
  for cube in sequence:
    first = cube[first_band, mask].astype(np.float64)
    second = cube[second_band, mask].astype(np.float64)
    values.append(float(np.mean((first - second) / np.maximum(first + second, 1e-6))))
  return np.asarray(values)


def _raw_controls(first: list[np.ndarray], second: list[np.ndarray], mask: np.ndarray) -> dict[str, float]:
  left = np.stack([cube[:, mask] for cube in first]).astype(np.float64)
  right = np.stack([cube[:, mask] for cube in second]).astype(np.float64)
  ndvi_left = _ndvi_curve(first, mask)
  ndvi_right = _ndvi_curve(second, mask)
  ndmi_left = _normalized_difference_curve(first, mask, 6, 8)
  ndmi_right = _normalized_difference_curve(second, mask, 6, 8)
  nbr_left = _normalized_difference_curve(first, mask, 6, 9)
  nbr_right = _normalized_difference_curve(second, mask, 6, 9)
  mean_left = np.mean(left, axis=2)
  mean_right = np.mean(right, axis=2)
  return {
    "raw_sequence_rms": float(np.sqrt(np.mean((right - left) ** 2))),
    "raw_difference_rms": float(np.sqrt(np.mean((np.diff(right, axis=0) - np.diff(left, axis=0)) ** 2))),
    "ndvi_curve_rms": float(np.sqrt(np.mean((ndvi_right - ndvi_left) ** 2))),
    "ndvi_amplitude_change": abs(float(np.ptp(ndvi_right) - np.ptp(ndvi_left))),
    "ndmi_amplitude_change": abs(float(np.ptp(ndmi_right) - np.ptp(ndmi_left))),
    "nbr_amplitude_change": abs(float(np.ptp(nbr_right) - np.ptp(nbr_left))),
    "band_mean_curve_rms": float(np.sqrt(np.mean((mean_right - mean_left) ** 2))),
    "band_amplitude_change": float(
      np.linalg.norm(np.ptp(mean_right, axis=0) - np.ptp(mean_left, axis=0))
      / np.sqrt(mean_left.shape[1])
    ),
  }


def _raw_second(first: list[np.ndarray], second: list[np.ndarray], third: list[np.ndarray], mask: np.ndarray) -> dict[str, float]:
  values = [np.stack([cube[:, mask] for cube in sequence]).astype(np.float64) for sequence in (first, second, third)]
  ndvi = [_ndvi_curve(sequence, mask) for sequence in (first, second, third)]
  return {
    "raw_second_rms": float(np.sqrt(np.mean((values[2] - 2.0 * values[1] + values[0]) ** 2))),
    "ndvi_second_rms": float(np.sqrt(np.mean((ndvi[2] - 2.0 * ndvi[1] + ndvi[0]) ** 2))),
  }


def _write_csv(path: Path, rows: list[dict]) -> None:
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def _summaries(rows: list[dict], score_names: list[str], bootstrap: int, rng: np.random.Generator) -> list[dict]:
  evaluation = [row for row in rows if row["scenario"] != "date_permutation"]
  groups = sorted({(row["representation"], row["preprocessing"], int(row["rank"])) for row in evaluation})
  output = []
  for representation, preprocessing, rank in groups:
    selected = [row for row in evaluation if row["representation"] == representation and row["preprocessing"] == preprocessing and int(row["rank"]) == rank]
    labels = np.asarray([int(row["is_event"]) for row in selected])
    patch_ids = sorted({row["patch_id"] for row in selected})
    for score in score_names:
      values = np.asarray([float(row[score]) for row in selected])
      boot_auc, boot_ap = [], []
      for _ in range(max(int(bootstrap), 0)):
        sampled = rng.choice(patch_ids, size=len(patch_ids), replace=True)
        indices = [index for patch in sampled for index, row in enumerate(selected) if row["patch_id"] == patch]
        y = labels[indices]
        if np.unique(y).size < 2:
          continue
        x = values[indices]
        boot_auc.append(float(roc_auc_score(y, x)))
        boot_ap.append(float(average_precision_score(y, x)))
      output.append({
        "representation": representation,
        "preprocessing": preprocessing,
        "rank": rank,
        "score": score,
        "auroc_event_vs_nuisance": float(roc_auc_score(labels, values)),
        "average_precision_event_vs_nuisance": float(average_precision_score(labels, values)),
        "auroc_ci_low": float(np.quantile(boot_auc, 0.025)) if boot_auc else float("nan"),
        "auroc_ci_high": float(np.quantile(boot_auc, 0.975)) if boot_auc else float("nan"),
        "average_precision_ci_low": float(np.quantile(boot_ap, 0.025)) if boot_ap else float("nan"),
        "average_precision_ci_high": float(np.quantile(boot_ap, 0.975)) if boot_ap else float("nan"),
      })
  return output


def _false_alarms(rows: list[dict], score_names: list[str]) -> list[dict]:
  output = []
  groups = sorted({(row["representation"], row["preprocessing"], int(row["rank"])) for row in rows})
  for representation, preprocessing, rank in groups:
    selected = [row for row in rows if row["representation"] == representation and row["preprocessing"] == preprocessing and int(row["rank"]) == rank]
    for score in score_names:
      stable = np.asarray([float(row[score]) for row in selected if row["scenario"] == "stable_jitter"])
      threshold = float(np.quantile(stable, 0.95))
      for scenario in sorted(NUISANCES | {"date_permutation"}):
        values = np.asarray([float(row[score]) for row in selected if row["scenario"] == scenario])
        output.append({
          "representation": representation,
          "preprocessing": preprocessing,
          "rank": rank,
          "score": score,
          "scenario": scenario,
          "stable_95pct_threshold": threshold,
          "false_alarm_or_response_rate": float(np.mean(values > threshold)),
        })
  return output


def _plot(summary: list[dict], output: Path) -> None:
  geometry = [row for row in summary if row["score"] in {"first_ds_magnitude", "second_along", "second_orthogonal"}]
  labels = [f"{row['representation']} r{row['rank']} {row['score'].replace('_', ' ')}" for row in geometry]
  values = [float(row["average_precision_event_vs_nuisance"]) for row in geometry]
  figure, axis = plt.subplots(figsize=(12, max(5, 0.35 * len(labels))))
  positions = np.arange(len(labels))
  axis.barh(positions, values, color="#267365")
  axis.set_yticks(positions, labels)
  axis.set_xlim(0.0, 1.0)
  axis.set_xlabel("Average precision: event vs nuisance")
  axis.set_title("Order-aware temporal subspaces on real MultiSenGE backgrounds")
  axis.grid(True, axis="x", alpha=0.25)
  figure.tight_layout()
  figure.savefig(output, dpi=180)
  plt.close(figure)


def main() -> None:
  started = time.perf_counter()
  args = parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  rng = np.random.default_rng(args.seed)
  ranks = sorted({int(value) for value in args.ranks.split(",") if value.strip()})
  representations = [_parse_representation(value) for value in args.representations.split(",") if value.strip()]
  rows: list[dict] = []
  patch_count = 0

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
      local_mask, dominant_class = _dominant_local_mask(crop_reference, mask)
      first = _jitter(sequence, rng)
      second_base = _jitter(sequence, rng)
      third_base = _jitter(sequence, rng)
      sample_id = f"{entry['patch_id']}:{repeat}"

      for scenario in SCENARIOS:
        spec = _scenario_spec(scenario, len(sequence), rng)
        second = _transform(second_base, kind=scenario, spec=spec, local_mask=local_mask)
        third = _transform(third_base, kind=scenario, spec=spec, local_mask=local_mask)
        controls = _raw_controls(first, second, mask)
        second_controls = _raw_second(first, second, third, mask)
        for representation, lag, representation_name in representations:
          fitted_first = build_temporal_representation_subspace(first, mask, rank=max(ranks), representation=representation, lag=lag, preprocessing=args.preprocessing)
          fitted_second = build_temporal_representation_subspace(second, mask, rank=max(ranks), representation=representation, lag=lag, preprocessing=args.preprocessing)
          fitted_third = build_temporal_representation_subspace(third, mask, rank=max(ranks), representation=representation, lag=lag, preprocessing=args.preprocessing)
          energy_change = representation_energy_change(fitted_first, fitted_second)
          spectrum_change = representation_spectrum_change(fitted_first, fitted_second)
          covariance_change = representation_covariance_change(fitted_first, fitted_second)
          available = min(fitted_first.rank, fitted_second.rank, fitted_third.rank)
          evaluated_ranks: set[int] = set()
          for requested_rank in ranks:
            effective_rank = min(requested_rank, available)
            if effective_rank < 1:
              continue
            if effective_rank in evaluated_ranks:
              continue
            evaluated_ranks.add(effective_rank)
            first_basis = fitted_first.basis[:, :effective_rank]
            second_basis = fitted_second.basis[:, :effective_rank]
            third_basis = fitted_third.basis[:, :effective_rank]
            angles = principal_angles(first_basis, second_basis)
            second_order = second_order_difference_subspace(first_basis, second_basis, third_basis, decompose=True)
            rows.append({
              "sample_id": sample_id,
              "patch_id": entry["patch_id"],
              "repeat": repeat,
              "scenario": scenario,
              "is_event": int(scenario in EVENTS),
              "dominant_reference_class": dominant_class,
              "local_support_pixels": int(np.sum(local_mask)),
              "representation": representation_name,
              "lag": lag,
              "preprocessing": args.preprocessing,
              "rank": effective_rank,
              "first_ds_magnitude": subspace_magnitude(first_basis, second_basis) / effective_rank,
              "first_geodesic": grassmann_geodesic_distance(first_basis, second_basis) / np.sqrt(effective_rank),
              "minimum_principal_angle": float(np.min(angles)) if angles.size else 0.0,
              "representation_energy_change": energy_change,
              "representation_spectrum_change": spectrum_change,
              "representation_covariance_change": covariance_change,
              "second_magnitude": float(second_order.mag_total) / effective_rank,
              "second_along": float(second_order.mag_along or 0.0) / effective_rank,
              "second_orthogonal": float(second_order.mag_orth or 0.0) / effective_rank,
              **controls,
              **second_controls,
            })
    patch_count += 1
    print(f"[OK] {entry['patch_id']}: dates={len(ordered)}, repeats={args.repeats}")
    if args.max_patches and patch_count >= args.max_patches:
      break

  if not rows:
    raise RuntimeError("No MultiSenGE samples were evaluated.")
  score_names = [
    "first_ds_magnitude",
    "first_geodesic",
    "minimum_principal_angle",
    "representation_energy_change",
    "representation_spectrum_change",
    "representation_covariance_change",
    "second_magnitude",
    "second_along",
    "second_orthogonal",
    "raw_sequence_rms",
    "raw_difference_rms",
    "ndvi_curve_rms",
    "ndvi_amplitude_change",
    "ndmi_amplitude_change",
    "nbr_amplitude_change",
    "band_mean_curve_rms",
    "band_amplitude_change",
    "raw_second_rms",
    "ndvi_second_rms",
  ]
  summary = _summaries(rows, score_names, args.bootstrap, rng)
  false_alarm = _false_alarms(rows, score_names)
  _write_csv(args.output_dir / "intervention_scores.csv", rows)
  _write_csv(args.output_dir / "event_vs_nuisance_summary.csv", summary)
  _write_csv(args.output_dir / "nuisance_response_summary.csv", false_alarm)
  _plot(summary, args.output_dir / "geometry_event_vs_nuisance_ap.png")
  metadata = {
    "study": "order-aware temporal subspaces on real MultiSenGE backgrounds",
    "patches": patch_count,
    "repeats_per_patch": args.repeats,
    "scenarios": list(SCENARIOS),
    "events": sorted(EVENTS),
    "nuisances": sorted(NUISANCES),
    "ranks": ranks,
    "representations": [name for _, _, name in representations],
    "preprocessing": args.preprocessing,
    "crop_size": args.crop_size,
    "runtime_seconds": time.perf_counter() - started,
    "bootstrap_unit": "MultiSenGE patch",
    "claim_boundary": (
      "Controlled interventions on real satellite backgrounds. This measures method behavior "
      "against known transformations, not accuracy on naturally occurring change labels."
    ),
  }
  (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
  print(json.dumps({"output_dir": str(args.output_dir), "rows": len(rows), "patches": patch_count}, indent=2))


if __name__ == "__main__":
  main()
