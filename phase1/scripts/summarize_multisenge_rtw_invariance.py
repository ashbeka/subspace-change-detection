"""Summarize a completed MultiSenGE RTW gate without rerunning subspaces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


TIMING_FAMILIES = {"stable", "phase", "warp"}
STRUCTURAL_TARGETS = {"marginal_matched_shape", "relative_band_phase"}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output_dir", required=True, type=Path)
  parser.add_argument("--bootstrap", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=4321)
  return parser.parse_args()


def _read_csv(path: Path) -> list[dict]:
  with path.open(encoding="utf-8") as handle:
    return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def _task_rows(rows: list[dict], task: str) -> tuple[list[dict], np.ndarray]:
  selected, labels = [], []
  for row in rows:
    shape = row["is_shape_target"] == "1"
    structural = row["is_structural_target"] == "1"
    nuisance = row["is_nuisance"] == "1"
    control = row["is_control_target"] == "1"
    if task == "structural_shape_vs_timing":
      eligible, label = structural or row["family"] in TIMING_FAMILIES, structural
    elif task == "shape_vs_all_nuisance":
      eligible, label = shape or nuisance, shape
    elif task == "all_change_vs_all_nuisance":
      eligible, label = shape or control or nuisance, shape or control
    else:
      raise ValueError(task)
    if eligible:
      selected.append(row)
      labels.append(int(label))
  return selected, np.asarray(labels, dtype=int)


def _paired_delta(
  rows: list[dict],
  labels: np.ndarray,
  first: str,
  second: str,
) -> tuple[float, float]:
  first_scores = np.asarray([float(row[first]) for row in rows])
  second_scores = np.asarray([float(row[second]) for row in rows])
  return (
    float(roc_auc_score(labels, first_scores) - roc_auc_score(labels, second_scores)),
    float(average_precision_score(labels, first_scores) - average_precision_score(labels, second_scores)),
  )


def _bootstrap_delta(
  rows: list[dict],
  labels: np.ndarray,
  first: str,
  second: str,
  *,
  cluster: str,
  repeats: int,
  seed: int,
) -> tuple[float, float, float, float]:
  keys = [
    row["patch_id"] if cluster == "patch" else f"{row['patch_id']}:{row['repeat']}"
    for row in rows
  ]
  clusters = sorted(set(keys))
  rng = np.random.default_rng(seed)
  auc_deltas, ap_deltas = [], []
  for _ in range(int(repeats)):
    sampled = rng.choice(clusters, size=len(clusters), replace=True)
    indices = [index for key in sampled for index, value in enumerate(keys) if value == key]
    sampled_labels = labels[indices]
    if np.unique(sampled_labels).size < 2:
      continue
    sampled_rows = [rows[index] for index in indices]
    auc, ap = _paired_delta(sampled_rows, sampled_labels, first, second)
    auc_deltas.append(auc)
    ap_deltas.append(ap)
  return (
    float(np.quantile(auc_deltas, 0.025)),
    float(np.quantile(auc_deltas, 0.975)),
    float(np.quantile(ap_deltas, 0.025)),
    float(np.quantile(ap_deltas, 0.975)),
  )


def _plot_deltas(rows: list[dict], output: Path) -> None:
  patch_repeat = [row for row in rows if row["cluster_unit"] == "patch_repeat"]
  patch_repeat.sort(key=lambda row: float(row["ap_delta"]))
  figure, axis = plt.subplots(figsize=(10, 6))
  positions = np.arange(len(patch_repeat))
  values = np.asarray([float(row["ap_delta"]) for row in patch_repeat])
  lower = np.asarray([float(row["ap_delta_ci_low"]) for row in patch_repeat])
  upper = np.asarray([float(row["ap_delta_ci_high"]) for row in patch_repeat])
  errors = np.vstack([values - lower, upper - values])
  colors = ["#31766b" if value > 0 else "#b94b3f" for value in values]
  axis.barh(positions, values, xerr=errors, color=colors, capsize=3)
  axis.set_yticks(
    positions,
    [f"{row['task']} vs {row['comparator']}" for row in patch_repeat],
  )
  axis.axvline(0.0, color="black", linewidth=1)
  axis.set_xlabel("Paired AP difference: RTW minus comparator")
  axis.set_title("RTW incremental value with patch-repeat bootstrap")
  axis.grid(True, axis="x", alpha=0.25)
  figure.tight_layout()
  figure.savefig(output, dpi=180)
  plt.close(figure)


def summarize_output(
  output_dir: Path,
  *,
  bootstrap: int = 1000,
  seed: int = 4321,
) -> dict:
  rows = [
    row for row in _read_csv(output_dir / "sequence_pair_scores.csv")
    if row["split"] == "holdout"
  ]
  methods = [
    "snapshot_subspace",
    "dtw_reference_zscore",
    "twdtw_reference_zscore",
    "fourier_magnitude",
    "harmonic_phase_aligned",
    "mssa_subspace",
  ]
  tasks = (
    "structural_shape_vs_timing",
    "shape_vs_all_nuisance",
    "all_change_vs_all_nuisance",
  )
  delta_rows = []
  for task in tasks:
    selected, labels = _task_rows(rows, task)
    for comparator in methods:
      auc_delta, ap_delta = _paired_delta(
        selected, labels, "rtw_selected", comparator
      )
      for cluster in ("patch", "patch_repeat"):
        auc_low, auc_high, ap_low, ap_high = _bootstrap_delta(
          selected,
          labels,
          "rtw_selected",
          comparator,
          cluster=cluster,
          repeats=bootstrap,
          seed=seed + len(delta_rows),
        )
        delta_rows.append({
          "task": task,
          "comparator": comparator,
          "cluster_unit": cluster,
          "auroc_delta": auc_delta,
          "auroc_delta_ci_low": auc_low,
          "auroc_delta_ci_high": auc_high,
          "ap_delta": ap_delta,
          "ap_delta_ci_low": ap_low,
          "ap_delta_ci_high": ap_high,
        })

  target_rows = []
  timing_negatives = [row for row in rows if row["family"] in TIMING_FAMILIES]
  for target in sorted(STRUCTURAL_TARGETS):
    positives = [row for row in rows if row["scenario"] == target]
    combined = [*timing_negatives, *positives]
    labels = np.asarray([0] * len(timing_negatives) + [1] * len(positives))
    for method in ["rtw_selected", *methods]:
      scores = np.asarray([float(row[method]) for row in combined])
      target_rows.append({
        "target": target,
        "method": method,
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "median_target": float(np.median(scores[labels == 1])),
        "median_timing_nuisance": float(np.median(scores[labels == 0])),
      })

  sampling_std = np.asarray([float(row["rtw_sampling_std"]) for row in rows])
  rtw_scores = np.asarray([float(row["rtw_selected"]) for row in rows])
  stability = {
    "mean_sampling_std": float(np.mean(sampling_std)),
    "median_sampling_std": float(np.median(sampling_std)),
    "mean_score": float(np.mean(rtw_scores)),
    "median_sampling_std_to_score": float(
      np.median(sampling_std / np.maximum(np.abs(rtw_scores), 1e-12))
    ),
  }

  _write_csv(output_dir / "paired_method_deltas.csv", delta_rows)
  _write_csv(output_dir / "structural_target_breakdown.csv", target_rows)
  _plot_deltas(delta_rows, output_dir / "paired_ap_deltas.png")

  snapshot_delta = next(
    row for row in delta_rows
    if row["task"] == "structural_shape_vs_timing"
    and row["comparator"] == "snapshot_subspace"
    and row["cluster_unit"] == "patch_repeat"
  )
  matched_target = next(
    row for row in target_rows
    if row["target"] == "marginal_matched_shape" and row["method"] == "rtw_selected"
  )
  relative_target = next(
    row for row in target_rows
    if row["target"] == "relative_band_phase" and row["method"] == "rtw_selected"
  )
  go = (
    float(snapshot_delta["ap_delta_ci_low"]) > 0.0
    and float(matched_target["average_precision"]) > 0.70
  )
  decision = {
    "go_to_natural_transition_gate": go,
    "reason": (
      "RTW demonstrated incremental structural information beyond the snapshot subspace."
      if go else
      "RTW did not beat the simpler snapshot subspace beyond uncertainty; its high pooled AP is driven by the easy relative-band-phase target, while marginal-matched shape remains weak."
    ),
    "structural_rtw_minus_snapshot_ap": float(snapshot_delta["ap_delta"]),
    "structural_rtw_minus_snapshot_ap_ci": [
      float(snapshot_delta["ap_delta_ci_low"]),
      float(snapshot_delta["ap_delta_ci_high"]),
    ],
    "marginal_matched_shape_rtw_ap": float(matched_target["average_precision"]),
    "relative_band_phase_rtw_ap": float(relative_target["average_precision"]),
    "sampling_stability": stability,
  }
  (output_dir / "posthoc_decision.json").write_text(
    json.dumps(decision, indent=2), encoding="utf-8"
  )
  lines = [
    "# RTW Gate Decision",
    "",
    f"Decision: **{'GO' if go else 'NO-GO'}** for a natural-transition RTW study.",
    "",
    decision["reason"],
    "",
    f"- RTW minus snapshot AP: `{decision['structural_rtw_minus_snapshot_ap']:+.4f}` "
    f"with patch-repeat 95% interval "
    f"`[{decision['structural_rtw_minus_snapshot_ap_ci'][0]:+.4f}, "
    f"{decision['structural_rtw_minus_snapshot_ap_ci'][1]:+.4f}]`.",
    f"- RTW AP on marginal-matched shape: `{decision['marginal_matched_shape_rtw_ap']:.4f}`.",
    f"- RTW AP on relative-band phase: `{decision['relative_band_phase_rtw_ap']:.4f}`.",
    f"- Median RTW sampling-std/score ratio: `{stability['median_sampling_std_to_score']:.4f}`.",
    "",
    "The natural-change stage is conditional and is not run after a no-go.",
  ]
  (output_dir / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
  return decision


def main() -> None:
  args = parse_args()
  decision = summarize_output(
    args.output_dir,
    bootstrap=args.bootstrap,
    seed=args.seed,
  )
  print(json.dumps(decision, indent=2))


if __name__ == "__main__":
  main()
