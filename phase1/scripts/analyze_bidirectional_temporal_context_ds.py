"""Benchmark bidirectional temporal-context DS on a registered sequence.

This runner evaluates a research hypothesis, not a finished detector.  It
compares backward/forward temporal-context DS, a symmetric linear projection
novelty control, and raw adjacent spectral difference.  Optional IPOL log-NFA
maps are treated as external method-agreement targets, never as ground truth.

Method provenance and the exact satellite adaptation are documented in
``phase1.subspace.temporal_context``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from phase1.subspace.temporal_band_images import sequence_common_valid_mask
from phase1.subspace.temporal_context import bidirectional_temporal_context_sequence

DATE_PATTERN = re.compile(r"(?P<year>\d{4})[-_]?(?P<month>\d{2})[-_]?(?P<day>\d{2})")
METHODS = ("temporal_context_ds", "linear_projection_novelty", "raw_adjacent_rms")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence_dir", required=True, type=Path)
  parser.add_argument("--glob", default="*.tif")
  parser.add_argument("--output_dir", required=True, type=Path)
  parser.add_argument("--context_sizes", default="3,5,7")
  parser.add_argument("--ranks", default="1,2,3")
  parser.add_argument("--factorizations", default="per_band,joint")
  parser.add_argument(
    "--preprocessing",
    default="centered_column_l2",
    choices=["uncentered", "centered", "column_l2", "centered_column_l2"],
  )
  parser.add_argument("--nodata", type=float, default=0.0)
  parser.add_argument("--scale_divisor", type=float, default=1.0)
  parser.add_argument("--reference_lognfa_dir", type=Path, default=None)
  parser.add_argument("--reference_logepsilon", type=float, default=1.0)
  parser.add_argument(
    "--figure_config",
    default="",
    help="Optional context:rank:factorization config to visualize, e.g. 5:2:per_band.",
  )
  parser.add_argument("--figure_count", type=int, default=4)
  return parser.parse_args()


def _split_ints(text: str) -> list[int]:
  return [int(value.strip()) for value in str(text).split(",") if value.strip()]


def _split_strings(text: str) -> list[str]:
  return [value.strip() for value in str(text).split(",") if value.strip()]


def _date(path: Path) -> str:
  match = DATE_PATTERN.search(path.name)
  if not match:
    raise ValueError(f"Could not parse a date from {path.name}")
  return "".join(match.group(name) for name in ("year", "month", "day"))


def _load(path: Path) -> np.ndarray:
  with rasterio.open(path) as source:
    return source.read().astype(np.float32)


def _write_csv(path: Path, rows: list[dict]) -> None:
  if not rows:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  fields: list[str] = []
  for row in rows:
    for key in row:
      if key not in fields:
        fields.append(key)
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def _map_stats(score: np.ndarray, mask: np.ndarray) -> dict[str, float]:
  values = np.maximum(np.asarray(score[mask], dtype=np.float64), 0.0)
  if not values.size:
    return {"mean": 0.0, "p95": 0.0, "top1_mean": 0.0, "top1_fraction": 0.0}
  count = max(1, int(np.ceil(0.01 * values.size)))
  top = np.partition(values, values.size - count)[-count:]
  total = float(np.sum(values))
  return {
    "mean": float(np.mean(values)),
    "p95": float(np.percentile(values, 95)),
    "top1_mean": float(np.mean(top)),
    "top1_fraction": float(np.sum(top) / total) if total > 0.0 else 0.0,
  }


def _ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
  labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
  scores = np.nan_to_num(np.asarray(scores, dtype=np.float64).reshape(-1))
  if np.unique(labels).size < 2:
    return {"auroc": float("nan"), "average_precision": float("nan"), "best_f1": float("nan"), "best_iou": float("nan")}
  precision, recall, _ = precision_recall_curve(labels, scores)
  denominator = precision + recall
  f1 = np.divide(2.0 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0)
  prevalence = float(np.mean(labels))
  iou_denominator = precision + recall - precision * recall
  iou = np.divide(
    precision * recall,
    iou_denominator,
    out=np.full_like(iou_denominator, prevalence),
    where=iou_denominator > 0,
  )
  return {
    "auroc": float(roc_auc_score(labels, scores)),
    "average_precision": float(average_precision_score(labels, scores)),
    "best_f1": float(np.max(f1)),
    "best_iou": float(np.max(iou)),
  }


def _safe_correlation(left: np.ndarray, right: np.ndarray, method: str) -> float:
  left = np.asarray(left, dtype=np.float64)
  right = np.asarray(right, dtype=np.float64)
  if left.size < 3 or np.std(left) <= 1e-15 or np.std(right) <= 1e-15:
    return float("nan")
  if method == "spearman":
    return float(spearmanr(left, right).statistic)
  return float(np.corrcoef(left, right)[0, 1])


def _reference(path: Path | None, boundary: int, shape: tuple[int, int]) -> np.ndarray | None:
  if path is None:
    return None
  candidate = path / f"lognfa_{boundary:03d}.tif"
  if not candidate.exists():
    return None
  with rasterio.open(candidate) as source:
    score = source.read(1).astype(np.float64)
  if score.shape != shape:
    raise ValueError(f"Reference map {candidate} has shape {score.shape}, expected {shape}")
  return score


def _robust_scale(score: np.ndarray, mask: np.ndarray) -> np.ndarray:
  values = np.asarray(score[mask], dtype=np.float64)
  upper = float(np.percentile(values[values > 0], 99.5)) if np.any(values > 0) else 1.0
  return np.clip(score / max(upper, 1e-15), 0.0, 1.0)


def _rgb(cube: np.ndarray) -> np.ndarray:
  image = np.moveaxis(cube[:3], 0, -1).astype(np.float64)
  low, high = np.percentile(image, [2, 98])
  return np.clip((image - low) / max(float(high - low), 1e-12), 0.0, 1.0)


def _save_figure(
  cube: np.ndarray,
  result,
  mask: np.ndarray,
  reference_lognfa: np.ndarray | None,
  logepsilon: float,
  path: Path,
  title: str,
) -> None:
  columns = 5 if reference_lognfa is not None else 4
  figure, axes = plt.subplots(1, columns, figsize=(4.1 * columns, 4.4))
  axes[0].imshow(_rgb(cube))
  axes[0].set_title("First post-date RGB")
  maps = (
    (result.ds_map, "Temporal-context DS"),
    (result.projection_novelty_map, "Linear context novelty"),
    (result.raw_boundary_difference_map, "Raw adjacent RMS"),
  )
  for axis, (score, label) in zip(axes[1:4], maps):
    axis.imshow(_robust_scale(score, mask), cmap="magma", vmin=0, vmax=1)
    axis.set_title(label)
  if reference_lognfa is not None:
    axes[4].imshow(reference_lognfa <= float(logepsilon), cmap="gray", vmin=0, vmax=1)
    axes[4].set_title("IPOL NFA decision (agreement only)")
  for axis in axes:
    axis.axis("off")
  figure.suptitle(title)
  figure.tight_layout()
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=180, bbox_inches="tight")
  plt.close(figure)


def main() -> None:
  args = parse_args()
  files = sorted(args.sequence_dir.glob(args.glob), key=_date)
  if len(files) < 4:
    raise ValueError("At least four dated TIFFs are required for temporal contexts.")
  dates = [_date(path) for path in files]
  cubes = [_load(path) / float(args.scale_divisor) for path in files]
  if any(cube.shape != cubes[0].shape for cube in cubes):
    raise ValueError("All sequence cubes must have identical dimensions.")
  mask = sequence_common_valid_mask(cubes, nodata_value=args.nodata)
  output = args.output_dir
  output.mkdir(parents=True, exist_ok=True)

  contexts = sorted(set(_split_ints(args.context_sizes)))
  ranks = sorted(set(_split_ints(args.ranks)))
  factorizations = _split_strings(args.factorizations)
  frame_rows: list[dict] = []
  spatial_rows: list[dict] = []
  result_cache: dict[tuple[int, int, str], list] = {}
  reference_fractions: dict[int, float] = {}
  for context_size in contexts:
    for rank in ranks:
      for factorization in factorizations:
        config = (context_size, rank, factorization)
        results = bidirectional_temporal_context_sequence(
          cubes,
          mask,
          context_size=context_size,
          rank=rank,
          factorization=factorization,
          preprocessing=args.preprocessing,
        )
        result_cache[config] = results
        for result in results:
          boundary = result.boundary_index
          maps = {
            "temporal_context_ds": result.ds_map,
            "linear_projection_novelty": result.projection_novelty_map,
            "raw_adjacent_rms": result.raw_boundary_difference_map,
          }
          reference = _reference(args.reference_lognfa_dir, boundary, mask.shape)
          row: dict[str, float | int | str] = {
            "context_size": context_size,
            "rank": rank,
            "factorization": factorization,
            "boundary_index": boundary,
            "date_before": dates[boundary - 1],
            "date_after": dates[boundary],
            "backward_indices": ";".join(str(value) for value in result.backward_indices),
            "forward_indices": ";".join(str(value) for value in result.forward_indices),
            "ds_magnitude": result.ds_magnitude,
            "product_geodesic_distance": result.geodesic_distance,
          }
          for method, score in maps.items():
            stats = _map_stats(score, mask)
            for name, value in stats.items():
              row[f"{method}_{name}"] = value
          if reference is not None:
            finite = mask & np.isfinite(reference)
            labels = reference[finite] <= float(args.reference_logepsilon)
            reference_score = -reference[finite]
            reference_fractions[boundary] = float(np.mean(labels))
            row["reference_nfa_positive_fraction"] = reference_fractions[boundary]
            for method, score in maps.items():
              metrics = _ranking_metrics(labels, score[finite])
              spatial_rows.append({
                "context_size": context_size,
                "rank": rank,
                "factorization": factorization,
                "boundary_index": boundary,
                "date_after": dates[boundary],
                "method": method,
                "reference_nfa_positive_fraction": reference_fractions[boundary],
                "pearson_with_negative_lognfa": _safe_correlation(score[finite], reference_score, "pearson"),
                "spearman_with_negative_lognfa": _safe_correlation(score[finite], reference_score, "spearman"),
                **metrics,
              })
          frame_rows.append(row)

  temporal_rows: list[dict] = []
  summary_rows: list[dict] = []
  for context_size in contexts:
    for rank in ranks:
      for factorization in factorizations:
        subset = [
          row for row in frame_rows
          if row["context_size"] == context_size
          and row["rank"] == rank
          and row["factorization"] == factorization
          and "reference_nfa_positive_fraction" in row
        ]
        spatial_subset = [
          row for row in spatial_rows
          if row["context_size"] == context_size
          and row["rank"] == rank
          and row["factorization"] == factorization
        ]
        for method in METHODS:
          method_spatial = [row for row in spatial_subset if row["method"] == method]
          if method_spatial:
            summary_rows.append({
              "context_size": context_size,
              "rank": rank,
              "factorization": factorization,
              "method": method,
              "n_boundaries": len(method_spatial),
              **{
                f"mean_{metric}": float(np.nanmean([float(row[metric]) for row in method_spatial]))
                for metric in ("auroc", "average_precision", "best_f1", "best_iou", "spearman_with_negative_lognfa")
              },
            })
          if subset:
            values = np.asarray([float(row[f"{method}_top1_mean"]) for row in subset])
            targets = np.asarray([float(row["reference_nfa_positive_fraction"]) for row in subset])
            temporal_rows.append({
              "context_size": context_size,
              "rank": rank,
              "factorization": factorization,
              "method": method,
              "n_boundaries": len(subset),
              "target": "reference_nfa_positive_fraction",
              "pearson": _safe_correlation(values, targets, "pearson"),
              "spearman": _safe_correlation(values, targets, "spearman"),
            })

  _write_csv(output / "temporal_context_frame_scores.csv", frame_rows)
  _write_csv(output / "reference_nfa_spatial_agreement.csv", spatial_rows)
  _write_csv(output / "reference_nfa_temporal_agreement.csv", temporal_rows)
  _write_csv(output / "configuration_summary.csv", summary_rows)

  if args.figure_config:
    parts = args.figure_config.split(":")
    if len(parts) != 3:
      raise ValueError("--figure_config must be context:rank:factorization")
    figure_config = (int(parts[0]), int(parts[1]), parts[2])
  else:
    figure_config = (contexts[0], ranks[0], factorizations[0])
  if figure_config not in result_cache:
    raise ValueError(f"Figure config {figure_config} was not evaluated.")
  figure_results = result_cache[figure_config]
  ranked = sorted(
    figure_results,
    key=lambda result: reference_fractions.get(result.boundary_index, -1.0),
    reverse=True,
  )[: max(0, int(args.figure_count))]
  for result in ranked:
    reference = _reference(args.reference_lognfa_dir, result.boundary_index, mask.shape)
    _save_figure(
      cubes[result.boundary_index],
      result,
      mask,
      reference,
      args.reference_logepsilon,
      output / "boundary_maps" / f"boundary_{result.boundary_index:03d}_{dates[result.boundary_index]}.png",
      (
        f"Boundary {result.boundary_index}: {dates[result.boundary_index - 1]} -> "
        f"{dates[result.boundary_index]} | V={figure_config[0]}, r={figure_config[1]}, "
        f"{figure_config[2]}"
      ),
    )

  metadata = {
    "sequence_directory": str(args.sequence_dir),
    "n_dates": len(cubes),
    "cube_shape": list(cubes[0].shape),
    "common_valid_locations": int(np.sum(mask)),
    "context_sizes": contexts,
    "ranks": ranks,
    "factorizations": factorizations,
    "preprocessing": args.preprocessing,
    "reference_lognfa_directory": str(args.reference_lognfa_dir) if args.reference_lognfa_dir else None,
    "reference_role": "External method agreement only; IPOL log-NFA is not ground truth.",
    "construction": (
      "At each boundary, V previous dates and V following dates form backward/forward "
      "temporal contexts. Per-band uses N x V matrices; joint uses N x (B*V)."
    ),
    "controls": {
      "linear_projection_novelty": "Symmetric cross-context orthogonal residual; not NNLS/NFA.",
      "raw_adjacent_rms": "Per-pixel RMS spectral difference between adjacent dates.",
    },
  }
  (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
  print(f"Done. Outputs in: {output}")


if __name__ == "__main__":
  main()
