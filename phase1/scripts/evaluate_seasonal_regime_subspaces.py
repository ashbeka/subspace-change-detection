"""Adversarial validation of seasonal observation Difference Subspaces.

This script is a controlled method test, not real irrigation performance.  It
constructs five-year multispectral seasonal sequences with known changes and
nuisances, builds one temporal subspace per year, and compares:

- first DS magnitude and Grassmann distance;
- minimum principal angle;
- singular-energy and normalized-spectrum changes;
- raw monthly-cube RMS and NDVI-curve/amplitude controls;
- paper-faithful second DS along/orthogonal decomposition.

Source trail:
- Kanai et al. (2023) supplies the scalar SSA trajectory-matrix construction;
- Fukui et al. (2024) defines first/second DS and the geodesic decomposition;
- the block-Hankel satellite representation, first-difference control,
  synthetic seasonal generator, and evaluation protocol are project-designed
  adaptations intended to expose both capability and failure boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score

from phase1.subspace.geodesic import (
  grassmann_geodesic_distance,
  principal_angles,
  subspace_magnitude,
)
from phase1.subspace.temporal_trajectory import (
  TemporalRepresentationSubspace,
  build_temporal_representation_subspace,
  representation_energy_change,
  representation_spectrum_change,
)
from phase1.subspace.second_order_ds import second_order_difference_subspace


SCENARIOS = (
  "stable_dry",
  "stable_irrigated",
  "global_gain_offset",
  "phase_shift",
  "missing_composites",
  "spatial_translation",
  "gradual_shape_drift",
  "abrupt_shape_on",
  "abrupt_shape_off",
  "abrupt_amplitude_only",
)
EVENT_SCENARIOS = {"abrupt_shape_on", "abrupt_shape_off", "abrupt_amplitude_only"}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output_dir", required=True, type=Path)
  parser.add_argument("--repeats", type=int, default=80)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--ranks", default="1,2,4,8")
  parser.add_argument(
    "--preprocessing",
    default="uncentered,feature_centered,feature_centered_observation_l2",
  )
  parser.add_argument(
    "--representations",
    default="unordered",
    help="Comma-separated unordered,difference,trajectory2,trajectory3,...",
  )
  parser.add_argument("--height", type=int, default=16)
  parser.add_argument("--width", type=int, default=16)
  parser.add_argument("--noise", type=float, default=0.008)
  parser.add_argument("--bootstrap", type=int, default=200)
  return parser.parse_args()


def _smooth_pattern(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
  pattern = ndimage.gaussian_filter(rng.normal(size=shape), sigma=sigma)
  pattern -= np.mean(pattern)
  pattern /= max(float(np.linalg.norm(pattern)), 1e-12)
  return pattern.astype(np.float32)


def _regime_cube_sequence(
  rng: np.random.Generator,
  *,
  pattern1: np.ndarray,
  pattern2: np.ndarray,
  amplitude: float,
  shape_strength: float,
  phase: float,
  noise: float,
  spatial_shift: tuple[float, float] = (0.0, 0.0),
) -> list[np.ndarray]:
  """Generate 12 monthly six-band composites with two seasonal modes."""
  shape = pattern1.shape
  if pattern2.shape != shape:
    raise ValueError("The two spatial modes must have the same shape.")
  if spatial_shift != (0.0, 0.0):
    pattern1 = ndimage.shift(pattern1, spatial_shift, order=1, mode="nearest")
    pattern2 = ndimage.shift(pattern2, spatial_shift, order=1, mode="nearest")
  base = np.asarray([0.11, 0.14, 0.18, 0.32, 0.23, 0.19], dtype=np.float32)
  spectral1 = np.asarray([0.05, 0.03, -0.08, 0.24, -0.05, -0.07], dtype=np.float32)
  spectral2 = np.asarray([-0.02, 0.01, -0.05, 0.12, 0.13, 0.17], dtype=np.float32)
  rows, cols = np.indices(shape)
  broad = (0.85 + 0.15 * rows / max(shape[0] - 1, 1) + 0.10 * cols / max(shape[1] - 1, 1))
  cubes: list[np.ndarray] = []
  for month in range(12):
    angle = 2.0 * np.pi * month / 12.0 + float(phase)
    first = np.sin(angle)
    second = np.sin(2.0 * angle + 0.35)
    cube = base[:, None, None] * broad[None, :, :]
    cube = cube + float(amplitude) * first * spectral1[:, None, None] * (1.0 + 5.0 * pattern1)[None, :, :]
    cube = cube + float(shape_strength) * second * spectral2[:, None, None] * (1.0 + 4.0 * pattern2)[None, :, :]
    cube = cube + rng.normal(scale=float(noise), size=cube.shape)
    cubes.append(np.clip(cube, 0.001, 1.0).astype(np.float32))
  return cubes


def _replace_missing_months(cubes: list[np.ndarray], rng: np.random.Generator) -> list[np.ndarray]:
  output = [cube.copy() for cube in cubes]
  missing = rng.choice(np.arange(1, 11), size=2, replace=False)
  for index in missing:
    output[int(index)] = 0.5 * (output[int(index) - 1] + output[int(index) + 1])
  return output


def generate_sequence(
  scenario: str,
  rng: np.random.Generator,
  *,
  shape: tuple[int, int],
  noise: float,
) -> tuple[list[list[np.ndarray]], int | None]:
  """Return five annual seasons and the true abrupt boundary, if any."""
  if scenario not in SCENARIOS:
    raise ValueError(scenario)
  event_boundary = int(rng.integers(0, 3)) if scenario in EVENT_SCENARIOS else None
  pattern1 = _smooth_pattern(rng, shape, sigma=2.2)
  pattern2 = _smooth_pattern(rng, shape, sigma=0.9)
  years: list[list[np.ndarray]] = []
  for year in range(5):
    amplitude = 0.45
    shape_strength = 0.10
    phase = float(rng.normal(scale=0.025))
    shift = (0.0, 0.0)

    if scenario == "stable_irrigated":
      amplitude, shape_strength = 1.0, 0.80
    elif scenario == "global_gain_offset":
      amplitude, shape_strength = 1.0, 0.80
    elif scenario == "phase_shift":
      amplitude, shape_strength = 1.0, 0.80
      phase += 0.18 * year
    elif scenario == "spatial_translation" and year == 2:
      shift = (0.0, 1.0)
    elif scenario == "gradual_shape_drift":
      amplitude = 0.65
      shape_strength = 0.10 + 0.16 * year
    elif scenario == "abrupt_shape_on":
      if year > int(event_boundary):
        amplitude, shape_strength = 1.0, 0.80
    elif scenario == "abrupt_shape_off":
      amplitude, shape_strength = 1.0, 0.80
      if year > int(event_boundary):
        amplitude, shape_strength = 0.45, 0.10
    elif scenario == "abrupt_amplitude_only":
      shape_strength = 0.25
      amplitude = 0.45 if year <= int(event_boundary) else 1.15

    season = _regime_cube_sequence(
      rng,
      pattern1=pattern1,
      pattern2=pattern2,
      amplitude=amplitude,
      shape_strength=shape_strength,
      phase=phase,
      noise=noise,
      spatial_shift=shift,
    )
    if scenario == "global_gain_offset":
      gain = float(rng.uniform(0.75, 1.30))
      offset = float(rng.uniform(-0.05, 0.05))
      season = [np.clip(gain * cube + offset, 0.001, 1.0).astype(np.float32) for cube in season]
    if scenario == "missing_composites":
      season = _replace_missing_months(season, rng)
    years.append(season)
  return years, event_boundary


def _ndvi_curve(season: list[np.ndarray]) -> np.ndarray:
  values = []
  for cube in season:
    nir = cube[3].astype(np.float64)
    red = cube[2].astype(np.float64)
    values.append(float(np.mean((nir - red) / np.maximum(nir + red, 1e-6))))
  return np.asarray(values, dtype=np.float64)


def _raw_boundary_scores(first: list[np.ndarray], second: list[np.ndarray]) -> dict[str, float]:
  left = np.stack(first).astype(np.float64)
  right = np.stack(second).astype(np.float64)
  ndvi_left = _ndvi_curve(first)
  ndvi_right = _ndvi_curve(second)
  return {
    "raw_monthly_cube_rms": float(np.sqrt(np.mean((right - left) ** 2))),
    "ndvi_curve_rms": float(np.sqrt(np.mean((ndvi_right - ndvi_left) ** 2))),
    "ndvi_amplitude_change": abs(float(np.ptp(ndvi_right) - np.ptp(ndvi_left))),
  }


def _raw_second_scores(
  left: list[np.ndarray], middle: list[np.ndarray], right: list[np.ndarray]
) -> dict[str, float]:
  values = [np.stack(season).astype(np.float64) for season in (left, middle, right)]
  ndvi = [_ndvi_curve(season) for season in (left, middle, right)]
  return {
    "raw_second_rms": float(np.sqrt(np.mean((values[2] - 2.0 * values[1] + values[0]) ** 2))),
    "ndvi_second_rms": float(np.sqrt(np.mean((ndvi[2] - 2.0 * ndvi[1] + ndvi[0]) ** 2))),
  }


def _fit_years(
  years: list[list[np.ndarray]],
  *,
  rank: int,
  preprocessing: str,
  representation: str,
  lag: int,
) -> list[TemporalRepresentationSubspace]:
  mask = np.ones(years[0][0].shape[1:], dtype=bool)
  return [
    build_temporal_representation_subspace(
      season,
      mask,
      rank=rank,
      preprocessing=preprocessing,
      representation=representation,
      lag=lag,
    )
    for season in years
  ]


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
  if key.startswith("hankel"):
    suffix = key.removeprefix("hankel").lstrip("_")
    lag = int(suffix) if suffix else 3
    return "trajectory", lag, f"trajectory{lag}"
  raise ValueError(f"Unsupported representation specification: {value!r}")


def _metric_summary(
  rows: list[dict],
  score_names: list[str],
  *,
  label_name: str,
  position_name: str,
  bootstrap: int,
  rng: np.random.Generator,
) -> list[dict]:
  output: list[dict] = []
  groups = sorted({(row["representation"], row["preprocessing"], int(row["rank"])) for row in rows})
  for representation, preprocessing, rank in groups:
    selected = [
      row for row in rows
      if row["representation"] == representation
      and row["preprocessing"] == preprocessing
      and int(row["rank"]) == rank
    ]
    labels = np.asarray([int(row[label_name]) for row in selected], dtype=int)
    if np.unique(labels).size < 2:
      continue
    for score_name in score_names:
      scores = np.asarray([float(row[score_name]) for row in selected], dtype=float)
      by_sequence: dict[int, list[int]] = {}
      for index, row in enumerate(selected):
        by_sequence.setdefault(int(row["sequence_id"]), []).append(index)
      sequence_ids = np.asarray(sorted(by_sequence), dtype=int)
      boot_auc: list[float] = []
      boot_ap: list[float] = []
      for _ in range(max(int(bootstrap), 0)):
        sampled_ids = rng.choice(sequence_ids, size=sequence_ids.size, replace=True)
        sampled_indices = [index for sequence_id in sampled_ids for index in by_sequence[int(sequence_id)]]
        sampled_labels = labels[sampled_indices]
        if np.unique(sampled_labels).size < 2:
          continue
        sampled_scores = scores[sampled_indices]
        boot_auc.append(float(roc_auc_score(sampled_labels, sampled_scores)))
        boot_ap.append(float(average_precision_score(sampled_labels, sampled_scores)))

      positive_sequences = [
        sequence_id
        for sequence_id, indices in by_sequence.items()
        if any(labels[index] == 1 for index in indices)
      ]
      top1 = []
      for sequence_id in positive_sequences:
        indices = by_sequence[sequence_id]
        predicted = indices[int(np.argmax(scores[indices]))]
        top1.append(int(labels[predicted] == 1))
      output.append({
        "representation": representation,
        "preprocessing": preprocessing,
        "rank": rank,
        "score": score_name,
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "positive_mean": float(np.mean(scores[labels == 1])),
        "negative_mean": float(np.mean(scores[labels == 0])),
        "auroc_ci_low": float(np.quantile(boot_auc, 0.025)) if boot_auc else float("nan"),
        "auroc_ci_high": float(np.quantile(boot_auc, 0.975)) if boot_auc else float("nan"),
        "average_precision_ci_low": float(np.quantile(boot_ap, 0.025)) if boot_ap else float("nan"),
        "average_precision_ci_high": float(np.quantile(boot_ap, 0.975)) if boot_ap else float("nan"),
        "top1_event_position_accuracy": float(np.mean(top1)) if top1 else float("nan"),
        "position_name": position_name,
      })
  return output


def _nuisance_false_alarm_summary(rows: list[dict], score_names: list[str]) -> list[dict]:
  output: list[dict] = []
  stable = {"stable_dry", "stable_irrigated"}
  nuisances = {
    "global_gain_offset",
    "phase_shift",
    "missing_composites",
    "spatial_translation",
    "gradual_shape_drift",
  }
  groups = sorted({(row["representation"], row["preprocessing"], int(row["rank"])) for row in rows})
  for representation, preprocessing, rank in groups:
    selected = [
      row for row in rows
      if row["representation"] == representation
      and row["preprocessing"] == preprocessing
      and int(row["rank"]) == rank
    ]
    for score_name in score_names:
      stable_values = np.asarray(
        [float(row[score_name]) for row in selected if row["scenario"] in stable], dtype=float
      )
      threshold = float(np.quantile(stable_values, 0.95))
      for scenario in sorted(nuisances):
        values = np.asarray(
          [float(row[score_name]) for row in selected if row["scenario"] == scenario], dtype=float
        )
        output.append({
          "representation": representation,
          "preprocessing": preprocessing,
          "rank": rank,
          "score": score_name,
          "threshold_from_stable_95pct": threshold,
          "nuisance_scenario": scenario,
          "false_alarm_rate": float(np.mean(values > threshold)),
          "nuisance_mean": float(np.mean(values)),
        })
  return output


def _write_csv(path: Path, rows: list[dict]) -> None:
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def _plot_summary(summary: list[dict], output: Path) -> None:
  methods = [
    "first_ds_magnitude",
    "first_geodesic",
    "minimum_principal_angle",
    "seasonal_energy_change",
    "singular_spectrum_change",
    "raw_monthly_cube_rms",
    "ndvi_curve_rms",
    "ndvi_amplitude_change",
  ]
  configs = sorted({
    (row["representation"], row["preprocessing"], int(row["rank"]))
    for row in summary
  })
  matrix = np.full((len(methods), len(configs)), np.nan, dtype=float)
  index = {
    (row["score"], row["representation"], row["preprocessing"], int(row["rank"])): row
    for row in summary
  }
  for row_index, method in enumerate(methods):
    for col_index, config in enumerate(configs):
      item = index.get((method, config[0], config[1], config[2]))
      if item:
        matrix[row_index, col_index] = float(item["average_precision"])
  figure, axis = plt.subplots(figsize=(max(10, 1.25 * len(configs)), 6.5))
  image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
  axis.set_yticks(range(len(methods)), [name.replace("_", " ") for name in methods])
  axis.set_xticks(
    range(len(configs)),
    [
      f"{representation}\n{name.replace('_', ' ')}\nr={rank}"
      for representation, name, rank in configs
    ],
    rotation=45,
    ha="right",
  )
  axis.set_title("Abrupt seasonal-regime boundary average precision")
  for row in range(matrix.shape[0]):
    for col in range(matrix.shape[1]):
      if np.isfinite(matrix[row, col]):
        axis.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color="white" if matrix[row, col] < 0.55 else "black", fontsize=8)
  figure.colorbar(image, ax=axis, label="Average precision")
  figure.tight_layout()
  figure.savefig(output, dpi=180)
  plt.close(figure)


def main() -> None:
  args = parse_args()
  output = args.output_dir
  output.mkdir(parents=True, exist_ok=True)
  ranks = sorted({int(value) for value in args.ranks.split(",") if value.strip()})
  preprocessing_modes = [value.strip() for value in args.preprocessing.split(",") if value.strip()]
  representations = [
    _parse_representation(value)
    for value in args.representations.split(",")
    if value.strip()
  ]
  rng = np.random.default_rng(args.seed)
  boundary_rows: list[dict] = []
  triple_rows: list[dict] = []

  sequence_id = 0
  for scenario in SCENARIOS:
    for repeat in range(args.repeats):
      local_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
      years, event_boundary = generate_sequence(
        scenario,
        local_rng,
        shape=(args.height, args.width),
        noise=args.noise,
      )
      raw_boundary = [_raw_boundary_scores(years[index], years[index + 1]) for index in range(4)]
      raw_second = [_raw_second_scores(years[index - 1], years[index], years[index + 1]) for index in range(1, 4)]
      for representation, lag, representation_name in representations:
        for preprocessing in preprocessing_modes:
          fitted = _fit_years(
            years,
            rank=max(ranks),
            preprocessing=preprocessing,
            representation=representation,
            lag=lag,
          )
          available_rank = min(item.rank for item in fitted)
          evaluated_ranks: set[int] = set()
          for rank in ranks:
            effective_rank = min(int(rank), available_rank)
            if effective_rank < 1 or effective_rank in evaluated_ranks:
              continue
            evaluated_ranks.add(effective_rank)
            bases = [item.basis[:, :effective_rank] for item in fitted]
            for boundary in range(4):
              angles = principal_angles(bases[boundary], bases[boundary + 1])
              row = {
                "sequence_id": sequence_id,
                "scenario": scenario,
                "repeat": repeat,
                "representation": representation_name,
                "lag": lag,
                "preprocessing": preprocessing,
                "rank": effective_rank,
                "boundary": boundary,
                "event_boundary": -1 if event_boundary is None else event_boundary,
                "is_abrupt_event_boundary": int(event_boundary == boundary),
                "is_shape_event_boundary": int(
                  event_boundary == boundary and scenario in {"abrupt_shape_on", "abrupt_shape_off"}
                ),
                "is_amplitude_event_boundary": int(
                  event_boundary == boundary and scenario == "abrupt_amplitude_only"
                ),
                "first_ds_magnitude": subspace_magnitude(bases[boundary], bases[boundary + 1]) / effective_rank,
                "first_geodesic": grassmann_geodesic_distance(bases[boundary], bases[boundary + 1]) / np.sqrt(effective_rank),
                "minimum_principal_angle": float(np.min(angles)) if angles.size else 0.0,
                "seasonal_energy_change": representation_energy_change(fitted[boundary], fitted[boundary + 1]),
                "singular_spectrum_change": representation_spectrum_change(fitted[boundary], fitted[boundary + 1]),
                **raw_boundary[boundary],
              }
              boundary_rows.append(row)
            for center in range(1, 4):
              result = second_order_difference_subspace(
                bases[center - 1], bases[center], bases[center + 1], decompose=True
              )
              triple_rows.append({
                "sequence_id": sequence_id,
                "scenario": scenario,
                "repeat": repeat,
                "representation": representation_name,
                "lag": lag,
                "preprocessing": preprocessing,
                "rank": effective_rank,
                "center_year": center,
                "event_boundary": -1 if event_boundary is None else event_boundary,
                "is_post_event_year": int(event_boundary is not None and center == event_boundary + 1),
                "second_magnitude": float(result.mag_total) / effective_rank,
                "second_along": float(result.mag_along or 0.0) / effective_rank,
                "second_orthogonal": float(result.mag_orth or 0.0) / effective_rank,
                **raw_second[center - 1],
              })
      sequence_id += 1

  _write_csv(output / "boundary_scores.csv", boundary_rows)
  _write_csv(output / "second_order_scores.csv", triple_rows)
  boundary_score_names = [
    "first_ds_magnitude",
    "first_geodesic",
    "minimum_principal_angle",
    "seasonal_energy_change",
    "singular_spectrum_change",
    "raw_monthly_cube_rms",
    "ndvi_curve_rms",
    "ndvi_amplitude_change",
  ]
  second_score_names = [
    "second_magnitude",
    "second_along",
    "second_orthogonal",
    "raw_second_rms",
    "ndvi_second_rms",
  ]
  boundary_summary = _metric_summary(
    boundary_rows,
    boundary_score_names,
    label_name="is_abrupt_event_boundary",
    position_name="boundary",
    bootstrap=args.bootstrap,
    rng=rng,
  )
  shape_rows = [row for row in boundary_rows if row["scenario"] != "abrupt_amplitude_only"]
  amplitude_rows = [
    row
    for row in boundary_rows
    if row["scenario"] not in {"abrupt_shape_on", "abrupt_shape_off"}
  ]
  shape_summary = _metric_summary(
    shape_rows,
    boundary_score_names,
    label_name="is_shape_event_boundary",
    position_name="boundary",
    bootstrap=args.bootstrap,
    rng=rng,
  )
  amplitude_summary = _metric_summary(
    amplitude_rows,
    boundary_score_names,
    label_name="is_amplitude_event_boundary",
    position_name="boundary",
    bootstrap=args.bootstrap,
    rng=rng,
  )
  second_summary = _metric_summary(
    triple_rows,
    second_score_names,
    label_name="is_post_event_year",
    position_name="center_year",
    bootstrap=args.bootstrap,
    rng=rng,
  )
  false_alarm_summary = _nuisance_false_alarm_summary(boundary_rows, boundary_score_names)
  for rows, name in (
    (boundary_summary, "boundary_summary.csv"),
    (shape_summary, "shape_event_vs_no_event_summary.csv"),
    (amplitude_summary, "amplitude_event_vs_no_event_summary.csv"),
    (second_summary, "second_order_summary.csv"),
    (false_alarm_summary, "nuisance_false_alarm_summary.csv"),
  ):
    _write_csv(output / name, rows)
  _plot_summary(boundary_summary, output / "boundary_average_precision.png")

  best = sorted(boundary_summary, key=lambda row: float(row["average_precision"]), reverse=True)[:20]
  metadata = {
    "study": "seasonal observation subspace synthetic stress test",
    "sequences": sequence_id,
    "scenarios": list(SCENARIOS),
    "event_scenarios": sorted(EVENT_SCENARIOS),
    "ranks": ranks,
    "preprocessing": preprocessing_modes,
    "representations": [name for _, _, name in representations],
    "seed": args.seed,
    "noise": args.noise,
    "bootstrap_resamples": args.bootstrap,
    "best_boundary_scores": best,
    "claim_boundary": (
      "Synthetic method diagnostic only. It can falsify invariance/selectivity assumptions, "
      "but it cannot establish irrigation or satellite change-detection performance."
    ),
  }
  (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
  print(json.dumps({"output_dir": str(output), "best_boundary_scores": best[:8]}, indent=2))


if __name__ == "__main__":
  main()
