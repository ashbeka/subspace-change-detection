"""Analyze a frozen geometry+radiometry fusion across SpaceNet 7 AOIs.

The fusion was fixed before the confirmation AOIs were scored:

1. unnormalized trajectory-window second-DS orthogonal magnitude;
2. per-date-channel-standardized raw pair RMS;
3. per-date-channel-standardized raw second-difference RMS.

Each score is converted to a percentile rank independently within one AOI and
monthly transition.  The three ranks are averaged with equal weights.  This
requires no learned parameters, but it is a project-designed hybrid rather
than an operation from the DS literature.  The script pairs runs by the AOI
recorded in ``run_metadata.json`` and never selects configurations by labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


METHODS = (
  "geometry_second_orthogonal",
  "standardized_raw_pair",
  "standardized_raw_second",
  "standardized_two_raw_rank_fusion",
  "fixed_three_score_rank_fusion",
)

GEOMETRY_DIAGNOSTICS = (
  "first_ds_magnitude",
  "first_geodesic",
  "second_total",
  "second_along",
  "second_orthogonal",
  "second_orthogonal_fraction",
  "representation_energy_change",
  "representation_spectrum_change",
  "representation_covariance_change",
  "raw_pair_rms",
  "raw_second_rms",
)

DIAGNOSTIC_CORRELATIONS = (
  ("first_ds_magnitude", "first_geodesic"),
  ("second_total", "second_along"),
  ("second_total", "second_orthogonal"),
  ("first_ds_magnitude", "raw_pair_rms"),
  ("second_orthogonal", "raw_second_rms"),
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_root", type=Path, default=Path("phase1/outputs"))
  parser.add_argument("--geometry_glob", required=True)
  parser.add_argument("--controls_glob", required=True)
  parser.add_argument("--output_dir", required=True, type=Path)
  parser.add_argument("--min_new_building_pixels", type=int, default=2)
  parser.add_argument("--bootstrap", type=int, default=2000)
  parser.add_argument("--seed", type=int, default=90210)
  return parser.parse_args()


def _read_csv(path: Path) -> list[dict]:
  with path.open(newline="", encoding="utf-8") as handle:
    return list(csv.DictReader(handle))


def _aoi_id(run_dir: Path) -> str:
  metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
  return Path(metadata["aoi_root"]).name


def _row_key(row: dict) -> tuple[str, str, str]:
  return row["date"], row["cell_row"], row["cell_col"]


def build_fusion_records(
  geometry_rows: list[dict],
  control_rows: list[dict],
  *,
  min_new_pixels: int,
) -> list[dict]:
  """Join matched cells and compute the fixed within-date rank fusion."""
  geometry = {_row_key(row): row for row in geometry_rows}
  controls = {_row_key(row): row for row in control_rows}
  records = []
  for key in sorted(set(geometry) & set(controls)):
    left = geometry[key]
    right = controls[key]
    geometric = float(left["second_orthogonal"])
    raw_pair = float(right["raw_pair_rms"])
    raw_second = float(right["raw_second_rms"])
    if not all(np.isfinite(value) for value in (geometric, raw_pair, raw_second)):
      continue
    records.append({
      "date": key[0],
      "cell_row": int(key[1]),
      "cell_col": int(key[2]),
      "label": int(int(left["positive_pixels"]) >= int(min_new_pixels)),
      "positive_pixels": int(left["positive_pixels"]),
      "geometry_second_orthogonal": geometric,
      "standardized_raw_pair": raw_pair,
      "standardized_raw_second": raw_second,
    })
  if not records:
    raise ValueError("Geometry/control runs have no finite matched second-order cells.")
  for date in sorted({row["date"] for row in records}):
    group = [row for row in records if row["date"] == date]
    for score in METHODS[:3]:
      ranks = rankdata([row[score] for row in group], method="average") / len(group)
      for row, rank in zip(group, ranks):
        row[f"{score}_rank"] = float(rank)
    for row in group:
      row["standardized_two_raw_rank_fusion"] = float(np.mean([
        row["standardized_raw_pair_rank"],
        row["standardized_raw_second_rank"],
      ]))
      row["fixed_three_score_rank_fusion"] = float(np.mean([
        row["geometry_second_orthogonal_rank"],
        row["standardized_raw_pair_rank"],
        row["standardized_raw_second_rank"],
      ]))
  return records


def _metrics(records: list[dict], method: str) -> dict[str, float]:
  labels = np.asarray([row["label"] for row in records])
  values = np.asarray([row[method] for row in records], dtype=np.float64)
  if np.unique(labels).size < 2:
    return {"average_precision": float("nan"), "auroc": float("nan")}
  return {
    "average_precision": float(average_precision_score(labels, values)),
    "auroc": float(roc_auc_score(labels, values)),
  }


def _geometry_diagnostics(aoi: str, rows: list[dict], min_new_pixels: int) -> list[dict]:
  output = []
  for score in GEOMETRY_DIAGNOSTICS:
    selected = [row for row in rows if np.isfinite(float(row[score]))]
    labels = np.asarray([int(int(row["positive_pixels"]) >= min_new_pixels) for row in selected])
    values = np.asarray([float(row[score]) for row in selected], dtype=np.float64)
    if values.size == 0 or np.unique(labels).size < 2:
      continue
    negative = values[labels == 0]
    positive = values[labels == 1]
    output.append({
      "aoi": aoi,
      "score": score,
      "cells": int(values.size),
      "positive_cells": int(np.sum(labels)),
      "average_precision": float(average_precision_score(labels, values)),
      "auroc": float(roc_auc_score(labels, values)),
      "negative_median": float(np.median(negative)),
      "positive_median": float(np.median(positive)),
      "positive_to_negative_median_ratio": float(
        np.median(positive) / max(float(np.median(negative)), 1e-12)
      ),
    })
  return output


def _geometry_correlations(aoi: str, rows: list[dict]) -> list[dict]:
  output = []
  for first, second in DIAGNOSTIC_CORRELATIONS:
    selected = [
      row for row in rows
      if np.isfinite(float(row[first])) and np.isfinite(float(row[second]))
    ]
    if len(selected) < 3:
      continue
    statistic = spearmanr(
      [float(row[first]) for row in selected],
      [float(row[second]) for row in selected],
    ).statistic
    output.append({
      "aoi": aoi,
      "first_score": first,
      "second_score": second,
      "cells": len(selected),
      "spearman_rho": float(statistic),
    })
  return output


def _bootstrap_delta(
  records: list[dict],
  first: str,
  second: str,
  *,
  bootstrap: int,
  rng: np.random.Generator,
) -> tuple[float, float, float]:
  dates = sorted({row["date"] for row in records})
  differences = []
  for _ in range(max(int(bootstrap), 0)):
    sampled_dates = rng.choice(dates, size=len(dates), replace=True)
    sample = [row for date in sampled_dates for row in records if row["date"] == date]
    labels = [row["label"] for row in sample]
    if len(set(labels)) < 2:
      continue
    differences.append(
      float(average_precision_score(labels, [row[first] for row in sample]))
      - float(average_precision_score(labels, [row[second] for row in sample]))
    )
  if not differences:
    return float("nan"), float("nan"), float("nan")
  return (
    float(np.quantile(differences, 0.025)),
    float(np.quantile(differences, 0.975)),
    float(np.mean(np.asarray(differences) > 0.0)),
  )


def _hierarchical_macro_delta(
  records_by_aoi: dict[str, list[dict]],
  first: str,
  second: str,
  *,
  bootstrap: int,
  rng: np.random.Generator,
) -> tuple[float, float, float]:
  aois = sorted(records_by_aoi)
  differences = []
  for _ in range(max(int(bootstrap), 0)):
    sampled_aois = rng.choice(aois, size=len(aois), replace=True)
    per_aoi = []
    for aoi in sampled_aois:
      records = records_by_aoi[str(aoi)]
      dates = sorted({row["date"] for row in records})
      sampled_dates = rng.choice(dates, size=len(dates), replace=True)
      sample = [row for date in sampled_dates for row in records if row["date"] == date]
      labels = [row["label"] for row in sample]
      if len(set(labels)) < 2:
        continue
      per_aoi.append(
        float(average_precision_score(labels, [row[first] for row in sample]))
        - float(average_precision_score(labels, [row[second] for row in sample]))
      )
    if per_aoi:
      differences.append(float(np.mean(per_aoi)))
  if not differences:
    return float("nan"), float("nan"), float("nan")
  return (
    float(np.quantile(differences, 0.025)),
    float(np.quantile(differences, 0.975)),
    float(np.mean(np.asarray(differences) > 0.0)),
  )


def _write_csv(path: Path, rows: list[dict]) -> None:
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def _plot(per_aoi: list[dict], output_dir: Path) -> None:
  aois = sorted({row["aoi"] for row in per_aoi})
  x = np.arange(len(aois))
  width = 0.16
  figure, axis = plt.subplots(figsize=(13, 6))
  for index, method in enumerate(METHODS):
    values = [
      next(float(row["average_precision"]) for row in per_aoi if row["aoi"] == aoi and row["method"] == method)
      for aoi in aois
    ]
    offset = (index - (len(METHODS) - 1) / 2.0) * width
    axis.bar(x + offset, values, width=width, label=method.replace("_", " "))
  axis.set_xticks(x, aois, rotation=20, ha="right")
  axis.set_ylabel("Cell-level average precision")
  axis.set_title("Frozen SpaceNet 7 geometry+radiometry confirmation")
  axis.grid(axis="y", alpha=0.25)
  axis.legend(fontsize=8)
  figure.tight_layout()
  figure.savefig(output_dir / "aoi_average_precision_comparison.png", dpi=180)
  plt.close(figure)


def main() -> None:
  args = parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  geometry_dirs = sorted(
    path for path in args.input_root.glob(args.geometry_glob)
    if (path / "run_metadata.json").exists() and (path / "cell_transition_scores.csv").exists()
  )
  control_dirs = sorted(
    path for path in args.input_root.glob(args.controls_glob)
    if (path / "run_metadata.json").exists() and (path / "cell_transition_scores.csv").exists()
  )
  geometry = {_aoi_id(path): path for path in geometry_dirs}
  controls = {_aoi_id(path): path for path in control_dirs}
  if set(geometry) != set(controls):
    raise ValueError(f"AOI mismatch: geometry={sorted(geometry)}, controls={sorted(controls)}")
  if not geometry:
    raise FileNotFoundError("No geometry/control run pairs matched the supplied globs.")

  records_by_aoi = {}
  per_aoi = []
  paired = []
  geometry_diagnostics = []
  geometry_correlations = []
  for index, aoi in enumerate(sorted(geometry)):
    geometry_rows = _read_csv(geometry[aoi] / "cell_transition_scores.csv")
    control_rows = _read_csv(controls[aoi] / "cell_transition_scores.csv")
    records = build_fusion_records(
      geometry_rows,
      control_rows,
      min_new_pixels=args.min_new_building_pixels,
    )
    records_by_aoi[aoi] = records
    geometry_diagnostics.extend(
      _geometry_diagnostics(aoi, geometry_rows, args.min_new_building_pixels)
    )
    geometry_correlations.extend(_geometry_correlations(aoi, geometry_rows))
    for method in METHODS:
      metrics = _metrics(records, method)
      per_aoi.append({
        "aoi": aoi,
        "method": method,
        "cells": len(records),
        "positive_cells": int(sum(row["label"] for row in records)),
        "positive_rate": float(np.mean([row["label"] for row in records])),
        **metrics,
      })
    for baseline in METHODS[:-1]:
      fusion_ap = _metrics(records, METHODS[-1])["average_precision"]
      baseline_ap = _metrics(records, baseline)["average_precision"]
      low, high, probability = _bootstrap_delta(
        records,
        METHODS[-1],
        baseline,
        bootstrap=args.bootstrap,
        rng=np.random.default_rng(args.seed + index * 10 + METHODS.index(baseline)),
      )
      paired.append({
        "aoi": aoi,
        "first_method": METHODS[-1],
        "second_method": baseline,
        "first_average_precision": fusion_ap,
        "second_average_precision": baseline_ap,
        "delta_average_precision": fusion_ap - baseline_ap,
        "delta_ci_low": low,
        "delta_ci_high": high,
        "bootstrap_probability_delta_positive": probability,
        "bootstrap_unit": "monthly transition",
      })

  macro = []
  for method in METHODS:
    values = [row["average_precision"] for row in per_aoi if row["method"] == method]
    macro.append({
      "method": method,
      "aois": len(values),
      "macro_average_precision": float(np.mean(values)),
      "median_average_precision": float(np.median(values)),
      "wins": int(sum(
        row["method"] == method
        and row["average_precision"] == max(
          item["average_precision"] for item in per_aoi if item["aoi"] == row["aoi"]
        )
        for row in per_aoi
      )),
    })
  macro_paired = []
  fusion_macro = next(row["macro_average_precision"] for row in macro if row["method"] == METHODS[-1])
  for baseline in METHODS[:-1]:
    baseline_macro = next(row["macro_average_precision"] for row in macro if row["method"] == baseline)
    low, high, probability = _hierarchical_macro_delta(
      records_by_aoi,
      METHODS[-1],
      baseline,
      bootstrap=args.bootstrap,
      rng=np.random.default_rng(args.seed + 100 + METHODS.index(baseline)),
    )
    macro_paired.append({
      "first_method": METHODS[-1],
      "second_method": baseline,
      "first_macro_average_precision": fusion_macro,
      "second_macro_average_precision": baseline_macro,
      "delta_macro_average_precision": fusion_macro - baseline_macro,
      "hierarchical_ci_low": low,
      "hierarchical_ci_high": high,
      "bootstrap_probability_delta_positive": probability,
      "bootstrap_units": "AOI then monthly transition",
    })

  _write_csv(args.output_dir / "aoi_method_metrics.csv", per_aoi)
  _write_csv(args.output_dir / "aoi_paired_fusion_comparisons.csv", paired)
  _write_csv(args.output_dir / "macro_method_metrics.csv", macro)
  _write_csv(args.output_dir / "macro_paired_fusion_comparisons.csv", macro_paired)
  _write_csv(args.output_dir / "geometry_component_metrics.csv", geometry_diagnostics)
  _write_csv(args.output_dir / "geometry_component_correlations.csv", geometry_correlations)
  geometry_macro = []
  for score in GEOMETRY_DIAGNOSTICS:
    selected = [row for row in geometry_diagnostics if row["score"] == score]
    if not selected:
      continue
    geometry_macro.append({
      "score": score,
      "aois": len(selected),
      "macro_average_precision": float(np.mean([row["average_precision"] for row in selected])),
      "macro_auroc": float(np.mean([row["auroc"] for row in selected])),
      "median_positive_to_negative_ratio": float(
        np.median([row["positive_to_negative_median_ratio"] for row in selected])
      ),
    })
  _write_csv(args.output_dir / "geometry_component_macro_metrics.csv", geometry_macro)
  _plot(per_aoi, args.output_dir)
  metadata = {
    "analysis": "frozen three-score SpaceNet 7 rank fusion",
    "geometry_glob": args.geometry_glob,
    "controls_glob": args.controls_glob,
    "aois": sorted(records_by_aoi),
    "minimum_new_building_pixels": args.min_new_building_pixels,
    "bootstrap": args.bootstrap,
    "seed": args.seed,
    "geometry_diagnostics": list(GEOMETRY_DIAGNOSTICS),
    "claim_boundary": "Confirmation AOIs only; fixed cell-level offline fusion, not SCOT and not Sentinel-2 validation.",
  }
  (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
  print(json.dumps({"output_dir": str(args.output_dir), "macro": macro, "paired": macro_paired}, indent=2))


if __name__ == "__main__":
  main()
