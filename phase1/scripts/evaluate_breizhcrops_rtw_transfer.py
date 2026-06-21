"""Independent natural-label transfer gate for Randomized Time Warping.

Research question
-----------------
Do RTW hypo-subspaces keep same-crop Sentinel-2 trajectories similar under
natural and controlled timing variation while separating different crop
phenologies better than simpler sequence, PCA, correlation, and warping
controls?

This is a pairwise crop-phenology discrimination and invariance experiment,
not changed-area segmentation. The RTW configuration is frozen from the prior
MultiSenGE mechanism gate (R=4, L=64, rank=5); BreizhCrops labels are never
used to tune it. FRH01 is development geography and FRH04 is untouched holdout.

Sources
-------
- Russwurm et al. (2020), BreizhCrops,
  https://doi.org/10.5194/isprs-archives-XLIII-B2-2020-1545-2020.
- Suryanto, Xue, and Fukui (2016), RTW,
  https://doi.org/10.1016/j.imavis.2016.07.003.
- Maus et al. (2016), TWDTW,
  https://doi.org/10.1109/JSTARS.2016.2517118.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from phase1.baselines.temporal_sequence import (
  aligned_rms_distance,
  bandwise_correlation_distance,
  best_circular_shift_correlation_distance,
  best_circular_shift_rms_distance,
  centered_snapshot_subspace_distance,
  dtw_distance,
  fourier_magnitude_distance,
  harmonic_phase_aligned_distance,
  mean_spectral_angle_distance,
  mean_spectrum_distance,
  mssa_subspace_distance,
  preprocess_pair,
  seasonal_amplitude_distance,
  seasonal_summary_distance,
  shift_orbit_subspace_distance,
  snapshot_subspace_distance,
  soft_dtw_divergence,
  symmetric_pca_reconstruction_error,
  time_weighted_dtw_distance,
)
from phase1.data.breizhcrops_sequences import (
  BreizhCropsSequence,
  sample_breizhcrops_region,
)
from phase1.subspace.randomized_time_warping import compare_rtw_sequences


HARD_GROUPS = {
  0: "winter_cereal",
  1: "winter_cereal",
  2: "annual_crop",
  3: "annual_crop",
  4: "annual_crop",
  5: "woody_perennial",
  6: "woody_perennial",
  7: "meadow",
  8: "meadow",
}


@dataclass(frozen=True)
class PhenologyPair:
  pair_id: str
  anchor_id: str
  split: str
  scenario: str
  label: int
  hard_pair: bool
  first_class: int
  second_class: int
  first_name: str
  second_name: str
  first: np.ndarray
  first_times: np.ndarray
  second: np.ndarray
  second_times: np.ndarray


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_root", type=Path, default=Path("data/BreizhCrops"))
  parser.add_argument("--output_dir", type=Path, required=True)
  parser.add_argument("--development_region", default="frh01")
  parser.add_argument("--holdout_region", default="frh04")
  parser.add_argument("--max_fields_per_class", type=int, default=80)
  parser.add_argument("--anchors_per_class", type=int, default=40)
  parser.add_argument("--min_steps", type=int, default=12)
  parser.add_argument("--quality_threshold", type=float, default=0.5)
  parser.add_argument("--rtw_replicates", type=int, default=3)
  parser.add_argument("--search_rtw", action="store_true")
  parser.add_argument("--rtw_search_anchors_per_class", type=int, default=8)
  parser.add_argument("--rtw_search_subsequence_lengths", default="2,4,8,12")
  parser.add_argument("--rtw_search_n_samples", default="32,64,128")
  parser.add_argument("--rtw_search_ranks", default="2,5")
  parser.add_argument(
    "--rtw_search_preprocessing", default="raw,per_sequence_zscore"
  )
  parser.add_argument("--rtw_finalists", type=int, default=4)
  parser.add_argument("--bootstrap", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=2718)
  return parser.parse_args()


def _stable_seed(*parts: object) -> int:
  digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
  return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def _parse_ints(value: str) -> list[int]:
  return sorted({int(item) for item in value.split(",") if item.strip()})


def _parse_strings(value: str) -> list[str]:
  return [item.strip() for item in value.split(",") if item.strip()]


def _periodic_transform(
  sequence: np.ndarray,
  times: np.ndarray,
  query: np.ndarray,
) -> np.ndarray:
  normalized = (times - times[0]) / max(float(times[-1] - times[0]), 1.0)
  extended_time = np.concatenate([normalized - 1.0, normalized, normalized + 1.0])
  return np.stack([
    np.interp(
      np.mod(query, 1.0),
      extended_time,
      np.concatenate([sequence[:, band]] * 3),
    )
    for band in range(sequence.shape[1])
  ], axis=1)


def _phase_shift(record: BreizhCropsSequence, fraction: float) -> np.ndarray:
  normalized = (record.times - record.times[0]) / max(
    float(record.times[-1] - record.times[0]), 1.0
  )
  return _periodic_transform(record.values, record.times, normalized - fraction)


def _nonlinear_warp(record: BreizhCropsSequence, exponent: float) -> np.ndarray:
  normalized = (record.times - record.times[0]) / max(
    float(record.times[-1] - record.times[0]), 1.0
  )
  query = np.clip(normalized ** float(exponent), 0.0, 1.0)
  return np.stack([
    np.interp(query, normalized, record.values[:, band])
    for band in range(record.values.shape[1])
  ], axis=1)


def _choose_other(
  records: list[BreizhCropsSequence],
  anchor: BreizhCropsSequence,
  rng: np.random.Generator,
  *,
  same_class: bool,
  hard: bool = False,
) -> BreizhCropsSequence | None:
  candidates = []
  for record in records:
    if record.field_id == anchor.field_id:
      continue
    if same_class and record.class_id != anchor.class_id:
      continue
    if not same_class and record.class_id == anchor.class_id:
      continue
    if hard and HARD_GROUPS.get(record.class_id) != HARD_GROUPS.get(anchor.class_id):
      continue
    candidates.append(record)
  if not candidates and hard:
    return None
  if not candidates:
    raise ValueError(f"No partner available for class {anchor.class_id}.")
  return candidates[int(rng.integers(0, len(candidates)))]


def _make_pair(
  anchor: BreizhCropsSequence,
  second: BreizhCropsSequence,
  *,
  split: str,
  scenario: str,
  second_values: np.ndarray | None = None,
  label: int,
  hard_pair: bool,
) -> PhenologyPair:
  return PhenologyPair(
    pair_id=f"{split}:{anchor.field_id}:{scenario}",
    anchor_id=f"{split}:{anchor.field_id}",
    split=split,
    scenario=scenario,
    label=int(label),
    hard_pair=bool(hard_pair),
    first_class=anchor.class_id,
    second_class=second.class_id,
    first_name=anchor.class_name,
    second_name=second.class_name,
    first=anchor.values,
    first_times=anchor.times,
    second=second.values if second_values is None else second_values,
    second_times=second.times,
  )


def _build_pairs(
  records: list[BreizhCropsSequence],
  *,
  split: str,
  anchors_per_class: int,
  seed: int,
) -> list[PhenologyPair]:
  rng = np.random.default_rng(int(seed))
  pairs: list[PhenologyPair] = []
  by_class = {
    class_id: [record for record in records if record.class_id == class_id]
    for class_id in sorted({record.class_id for record in records})
  }
  for class_id, class_records in by_class.items():
    if len(class_records) < 2:
      continue
    count = min(int(anchors_per_class), len(class_records))
    anchors = [class_records[index] for index in rng.choice(
      len(class_records), size=count, replace=False
    )]
    for anchor_index, anchor in enumerate(anchors):
      same = _choose_other(records, anchor, rng, same_class=True)
      hard = _choose_other(records, anchor, rng, same_class=False, hard=True)
      random_other = _choose_other(records, anchor, rng, same_class=False)
      if same is None or random_other is None:
        raise RuntimeError("Required natural partner could not be sampled.")
      phase_fraction = (0.10, 0.20, -0.15)[anchor_index % 3]
      warp_exponent = (0.72, 1.35)[anchor_index % 2]
      anchor_pairs = [
        _make_pair(
          anchor, same, split=split, scenario="same_class_natural",
          label=0, hard_pair=False,
        ),
        _make_pair(
          anchor, random_other, split=split, scenario="different_class_random",
          label=1,
          hard_pair=(HARD_GROUPS.get(anchor.class_id) == HARD_GROUPS.get(random_other.class_id)),
        ),
        _make_pair(
          anchor, anchor, split=split, scenario="same_field_phase_shift",
          second_values=_phase_shift(anchor, phase_fraction), label=0, hard_pair=False,
        ),
        _make_pair(
          anchor, anchor, split=split, scenario="same_field_nonlinear_warp",
          second_values=_nonlinear_warp(anchor, warp_exponent), label=0, hard_pair=False,
        ),
      ]
      if hard is not None:
        anchor_pairs.append(_make_pair(
          anchor, hard, split=split, scenario="different_class_hard",
          label=1, hard_pair=True,
        ))
      pairs.extend(anchor_pairs)
  return pairs


def _rtw_pair_score(
  pair: PhenologyPair,
  config: dict,
  *,
  replicates: int,
  seed: int,
) -> dict[str, float | list[float]]:
  left, right = preprocess_pair(
    pair.first, pair.second, str(config["preprocessing"])
  )
  return compare_rtw_sequences(
    left,
    right,
    subsequence_length=int(config["subsequence_length"]),
    n_samples=int(config["n_samples"]),
    rank=int(config["rank"]),
    seed=int(seed),
    replicates=int(replicates),
  )


def _rtw_search_pairs(
  development_pairs: list[PhenologyPair], anchors_per_class: int
) -> list[PhenologyPair]:
  anchor_classes = {}
  for pair in development_pairs:
    anchor_classes[pair.anchor_id] = pair.first_class
  selected_anchors = set()
  for class_id in sorted(set(anchor_classes.values())):
    anchors = sorted(
      anchor for anchor, value in anchor_classes.items() if value == class_id
    )
    selected_anchors.update(anchors[: int(anchors_per_class)])
  return [
    pair for pair in development_pairs
    if pair.anchor_id in selected_anchors
    and pair.scenario in TASK_SCENARIOS["combined"]
  ]


def _evaluate_rtw_config(
  pairs: list[PhenologyPair],
  config: dict,
  *,
  replicates: int,
  seed: int,
) -> dict:
  labels, scores = [], []
  for pair in pairs:
    result = _rtw_pair_score(
      pair,
      config,
      replicates=replicates,
      seed=_stable_seed(seed, config["config_id"], pair.pair_id),
    )
    labels.append(pair.label)
    scores.append(float(result["dissimilarity"]))
  return {
    **config,
    "pairs": len(pairs),
    "replicates": int(replicates),
    "auroc": float(roc_auc_score(labels, scores)),
    "average_precision": float(average_precision_score(labels, scores)),
  }


def _select_rtw_config(
  development_pairs: list[PhenologyPair],
  args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
  minimum_steps = min(
    min(len(pair.first), len(pair.second)) for pair in development_pairs
  )
  configs = []
  for subsequence_length in _parse_ints(args.rtw_search_subsequence_lengths):
    if subsequence_length > minimum_steps:
      continue
    for n_samples in _parse_ints(args.rtw_search_n_samples):
      for rank in _parse_ints(args.rtw_search_ranks):
        if rank > min(10 * subsequence_length, n_samples):
          continue
        for preprocessing in _parse_strings(args.rtw_search_preprocessing):
          configs.append({
            "config_id": (
              f"R{subsequence_length}_L{n_samples}_m{rank}_{preprocessing}"
            ),
            "subsequence_length": subsequence_length,
            "n_samples": n_samples,
            "rank": rank,
            "preprocessing": preprocessing,
          })
  screening_pairs = _rtw_search_pairs(
    development_pairs, args.rtw_search_anchors_per_class
  )
  print(f"[RTW SEARCH] configs={len(configs)}, screening_pairs={len(screening_pairs)}")
  screening = []
  for index, config in enumerate(configs, start=1):
    row = _evaluate_rtw_config(
      screening_pairs,
      config,
      replicates=1,
      seed=_stable_seed(args.seed, "rtw_screening"),
    )
    row["stage"] = "screening"
    screening.append(row)
    print(
      f"[RTW SEARCH {index:02d}/{len(configs):02d}] "
      f"{config['config_id']}: AP={row['average_precision']:.4f}"
    )
  screening.sort(
    key=lambda row: (-row["average_precision"], -row["auroc"], row["config_id"])
  )
  finalist_configs = [
    {key: row[key] for key in (
      "config_id", "subsequence_length", "n_samples", "rank", "preprocessing"
    )}
    for row in screening[: min(int(args.rtw_finalists), len(screening))]
  ]
  finalists = []
  for index, config in enumerate(finalist_configs, start=1):
    row = _evaluate_rtw_config(
      development_pairs,
      config,
      replicates=args.rtw_replicates,
      seed=_stable_seed(args.seed, "rtw_finalist"),
    )
    row["stage"] = "finalist"
    finalists.append(row)
    print(
      f"[RTW FINALIST {index:02d}/{len(finalist_configs):02d}] "
      f"{config['config_id']}: AP={row['average_precision']:.4f}"
    )
  finalists.sort(
    key=lambda row: (-row["average_precision"], -row["auroc"], row["config_id"])
  )
  selected = {key: finalists[0][key] for key in (
    "config_id", "subsequence_length", "n_samples", "rank", "preprocessing"
  )}
  print(f"[RTW SELECTED] {selected['config_id']}")
  return selected, [*screening, *finalists]


def _scores(
  pair: PhenologyPair,
  args: argparse.Namespace,
  selected_config: dict | None = None,
) -> dict[str, float]:
  raw_left, raw_right = pair.first, pair.second
  z_left, z_right = preprocess_pair(raw_left, raw_right, "per_sequence_zscore")
  frozen_raw_config = {
    "config_id": "frozen_raw",
    "subsequence_length": 4,
    "n_samples": 64,
    "rank": 5,
    "preprocessing": "raw",
  }
  frozen_zscore_config = {**frozen_raw_config, "config_id": "frozen_zscore", "preprocessing": "per_sequence_zscore"}
  rtw_raw = _rtw_pair_score(
    pair,
    frozen_raw_config,
    replicates=args.rtw_replicates,
    seed=_stable_seed(args.seed, pair.pair_id, "rtw_raw"),
  )
  rtw_zscore = _rtw_pair_score(
    pair,
    frozen_zscore_config,
    replicates=args.rtw_replicates,
    seed=_stable_seed(args.seed, pair.pair_id, "rtw_zscore"),
  )
  output = {
    "rtw_frozen_raw": float(rtw_raw["dissimilarity"]),
    "rtw_frozen_raw_sampling_std": float(rtw_raw["dissimilarity_std"]),
    "rtw_per_sequence_zscore": float(rtw_zscore["dissimilarity"]),
    "aligned_rms_raw": aligned_rms_distance(
      raw_left, pair.first_times, raw_right, pair.second_times
    ),
    "aligned_rms_zscore": aligned_rms_distance(
      z_left, pair.first_times, z_right, pair.second_times
    ),
    "global_shift_rms_zscore": best_circular_shift_rms_distance(
      z_left, pair.first_times, z_right, pair.second_times
    ),
    "global_shift_correlation": best_circular_shift_correlation_distance(
      raw_left, pair.first_times, raw_right, pair.second_times
    ),
    "bandwise_correlation": bandwise_correlation_distance(
      raw_left, pair.first_times, raw_right, pair.second_times
    ),
    "mean_spectrum": mean_spectrum_distance(raw_left, raw_right),
    "seasonal_amplitude": seasonal_amplitude_distance(raw_left, raw_right),
    "seasonal_summary": seasonal_summary_distance(raw_left, raw_right),
    "mean_spectral_angle": mean_spectral_angle_distance(
      raw_left, pair.first_times, raw_right, pair.second_times
    ),
    "fourier_magnitude": fourier_magnitude_distance(
      raw_left, pair.first_times, raw_right, pair.second_times
    ),
    "harmonic_phase_aligned": harmonic_phase_aligned_distance(
      raw_left, pair.first_times, raw_right, pair.second_times
    ),
    "dtw_raw": dtw_distance(raw_left, raw_right),
    "dtw_zscore": dtw_distance(z_left, z_right),
    "twdtw_zscore": time_weighted_dtw_distance(
      z_left, pair.first_times, z_right, pair.second_times
    ),
    "soft_dtw_zscore": soft_dtw_divergence(z_left, z_right),
    "snapshot_uncentered": snapshot_subspace_distance(raw_left, raw_right, rank=5),
    "snapshot_centered": centered_snapshot_subspace_distance(raw_left, raw_right, rank=5),
    "pca_cross_reconstruction": symmetric_pca_reconstruction_error(
      raw_left, raw_right, rank=5
    ),
    "shift_orbit_subspace": shift_orbit_subspace_distance(
      raw_left, pair.first_times, raw_right, pair.second_times
    ),
    "mssa_subspace": mssa_subspace_distance(raw_left, raw_right, lag=4, rank=3),
  }
  if selected_config is not None:
    selected = _rtw_pair_score(
      pair,
      selected_config,
      replicates=args.rtw_replicates,
      seed=_stable_seed(args.seed, pair.pair_id, "rtw_selected"),
    )
    output["rtw_selected"] = float(selected["dissimilarity"])
    output["rtw_selected_sampling_std"] = float(selected["dissimilarity_std"])
  return output


TASK_SCENARIOS = {
  "natural_semantic": {
    "same_class_natural", "different_class_hard", "different_class_random",
  },
  "hard_natural": {"same_class_natural", "different_class_hard"},
  "timing_invariance": {
    "same_field_phase_shift", "same_field_nonlinear_warp",
    "different_class_hard", "different_class_random",
  },
  "combined": {
    "same_class_natural", "same_field_phase_shift", "same_field_nonlinear_warp",
    "different_class_hard", "different_class_random",
  },
}


def _select_task_rows(rows: list[dict], split: str, task: str) -> list[dict]:
  selected = [
    row for row in rows
    if row["split"] == split and row["scenario"] in TASK_SCENARIOS[task]
  ]
  if task == "hard_natural":
    eligible_anchors = {
      row["anchor_id"] for row in selected
      if row["scenario"] == "different_class_hard"
    }
    selected = [row for row in selected if row["anchor_id"] in eligible_anchors]
  return selected


def _write_csv(path: Path, rows: list[dict]) -> None:
  if not rows:
    return
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def _metric_rows(score_rows: list[dict], methods: list[str]) -> list[dict]:
  output = []
  for split in ("development", "holdout"):
    for task in TASK_SCENARIOS:
      selected = _select_task_rows(score_rows, split, task)
      labels = np.asarray([row["label"] for row in selected], dtype=int)
      for method in methods:
        scores = np.asarray([row[method] for row in selected], dtype=float)
        output.append({
          "split": split,
          "task": task,
          "method": method,
          "n": len(selected),
          "positives": int(np.sum(labels)),
          "auroc": float(roc_auc_score(labels, scores)),
          "average_precision": float(average_precision_score(labels, scores)),
          "median_positive": float(np.median(scores[labels == 1])),
          "median_negative": float(np.median(scores[labels == 0])),
        })
  return output


def _bootstrap_delta(
  score_rows: list[dict],
  first_method: str,
  second_method: str,
  *,
  split: str,
  task: str,
  repeats: int,
  seed: int,
) -> dict:
  selected = _select_task_rows(score_rows, split, task)
  anchors = sorted({row["anchor_id"] for row in selected})
  rng = np.random.default_rng(int(seed))
  deltas = []
  for _ in range(int(repeats)):
    sampled = rng.choice(anchors, size=len(anchors), replace=True)
    labels, first_scores, second_scores = [], [], []
    for anchor in sampled:
      rows = [row for row in selected if row["anchor_id"] == anchor]
      labels.extend(row["label"] for row in rows)
      first_scores.extend(row[first_method] for row in rows)
      second_scores.extend(row[second_method] for row in rows)
    labels = np.asarray(labels, dtype=int)
    if np.unique(labels).size < 2:
      continue
    deltas.append(
      average_precision_score(labels, first_scores)
      - average_precision_score(labels, second_scores)
    )
  point_rows = selected
  labels = np.asarray([row["label"] for row in point_rows], dtype=int)
  point = float(
    average_precision_score(labels, [row[first_method] for row in point_rows])
    - average_precision_score(labels, [row[second_method] for row in point_rows])
  )
  return {
    "first_method": first_method,
    "second_method": second_method,
    "split": split,
    "task": task,
    "ap_delta": point,
    "ci_low": float(np.quantile(deltas, 0.025)),
    "ci_high": float(np.quantile(deltas, 0.975)),
    "bootstrap_repeats": len(deltas),
  }


def _development_thresholds(
  score_rows: list[dict], methods: list[str], target_tpr: float = 0.8
) -> dict[str, float]:
  selected = [
    row for row in score_rows
    if row["split"] == "development"
    and row["scenario"] in {"different_class_hard", "different_class_random"}
  ]
  return {
    method: float(np.quantile([row[method] for row in selected], 1.0 - target_tpr))
    for method in methods
  }


def _threshold_rows(
  score_rows: list[dict], methods: list[str], thresholds: dict[str, float]
) -> list[dict]:
  output = []
  holdout = [row for row in score_rows if row["split"] == "holdout"]
  for method in methods:
    threshold = thresholds[method]
    for scenario in sorted({row["scenario"] for row in holdout}):
      selected = [row for row in holdout if row["scenario"] == scenario]
      output.append({
        "method": method,
        "scenario": scenario,
        "development_threshold_80pct_positive_tpr": threshold,
        "holdout_positive_rate": float(np.mean([
          row[method] >= threshold for row in selected
        ])),
      })
  return output


def _plot_method_ap(metrics: list[dict], output: Path) -> None:
  rows = [
    row for row in metrics
    if row["split"] == "holdout" and row["task"] == "combined"
  ]
  rows.sort(key=lambda row: row["average_precision"])
  figure, axis = plt.subplots(figsize=(11, 9))
  colors = ["#c4553d" if row["method"].startswith("rtw") else "#4b6f8a" for row in rows]
  axis.barh(
    np.arange(len(rows)),
    [row["average_precision"] for row in rows],
    color=colors,
  )
  axis.set_yticks(np.arange(len(rows)), [row["method"] for row in rows])
  axis.set_xlim(0.0, 1.0)
  axis.set_xlabel("Held-out average precision: different crop class")
  axis.set_title("BreizhCrops RTW transfer gate (FRH04 holdout)")
  axis.grid(True, axis="x", alpha=0.25)
  figure.tight_layout()
  figure.savefig(output, dpi=180)
  plt.close(figure)


def _plot_scenario_responses(
  score_rows: list[dict], methods: list[str], output: Path
) -> None:
  holdout = [row for row in score_rows if row["split"] == "holdout"]
  scenarios = sorted({row["scenario"] for row in holdout})
  matrix = np.zeros((len(methods), len(scenarios)), dtype=float)
  for method_index, method in enumerate(methods):
    values = np.asarray([
      np.median([row[method] for row in holdout if row["scenario"] == scenario])
      for scenario in scenarios
    ])
    low, high = np.min(values), np.max(values)
    matrix[method_index] = (values - low) / max(float(high - low), 1e-12)
  figure, axis = plt.subplots(figsize=(12, max(6, len(methods) * 0.42)))
  image = axis.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
  axis.set_xticks(np.arange(len(scenarios)), scenarios, rotation=35, ha="right")
  axis.set_yticks(np.arange(len(methods)), methods)
  axis.set_title("Held-out median response by pair type (row-normalized)")
  figure.colorbar(image, ax=axis, label="Within-method normalized response")
  figure.tight_layout()
  figure.savefig(output, dpi=180)
  plt.close(figure)


def _summary_markdown(
  path: Path,
  metrics: list[dict],
  comparisons: list[dict],
  best_baseline: str,
  metadata: dict,
) -> None:
  holdout = [
    row for row in metrics
    if row["split"] == "holdout" and row["task"] == "combined"
  ]
  holdout.sort(key=lambda row: row["average_precision"], reverse=True)
  lines = [
    "# BreizhCrops RTW Natural-Label Transfer Gate",
    "",
    "Pairwise task: distinguish different-crop trajectories from natural same-crop and timing-transformed same-field trajectories.",
    "",
    "| Method | Holdout AUROC | Holdout AP |",
    "|---|---:|---:|",
  ]
  for row in holdout:
    lines.append(
      f"| {row['method']} | {row['auroc']:.4f} | {row['average_precision']:.4f} |"
    )
  lines.extend([
    "",
    "## Frozen Comparison",
    "",
    f"The strongest non-RTW baseline was selected on FRH01: `{best_baseline}`.",
    "",
    "| Task | RTW variant | Development-selected baseline | Holdout AP delta | 95% interval |",
    "|---|---|---|---:|---:|",
  ])
  for comparison in comparisons:
    lines.append(
      f"| {comparison['task']} | {comparison['first_method']} | "
      f"{comparison['second_method']} | {comparison['ap_delta']:+.4f} | "
      f"[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}] |"
    )
  lines.extend([
    "",
    "## Claim Boundary",
    "",
    "This evaluates natural crop-label discrimination and timing nuisance behavior. It is not binary changed-area segmentation and does not establish RTW novelty in remote sensing.",
    "",
    "## Data",
    "",
    f"- Development: `{metadata['development']['region']}` ({metadata['development']['loaded_sequences']} sequences).",
    f"- Holdout: `{metadata['holdout']['region']}` ({metadata['holdout']['loaded_sequences']} sequences).",
  ])
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
  args = parse_args()
  started = time.perf_counter()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  development, development_metadata = sample_breizhcrops_region(
    args.data_root,
    args.development_region,
    max_per_class=args.max_fields_per_class,
    seed=_stable_seed(args.seed, "development_data"),
    min_steps=args.min_steps,
    quality_threshold=args.quality_threshold,
  )
  holdout, holdout_metadata = sample_breizhcrops_region(
    args.data_root,
    args.holdout_region,
    max_per_class=args.max_fields_per_class,
    seed=_stable_seed(args.seed, "holdout_data"),
    min_steps=args.min_steps,
    quality_threshold=args.quality_threshold,
  )
  development_counts = {
    class_id: sum(record.class_id == class_id for record in development)
    for class_id in {record.class_id for record in development}
  }
  holdout_counts = {
    class_id: sum(record.class_id == class_id for record in holdout)
    for class_id in {record.class_id for record in holdout}
  }
  eligible_classes = sorted(
    class_id for class_id in set(development_counts) & set(holdout_counts)
    if development_counts[class_id] >= args.anchors_per_class
    and holdout_counts[class_id] >= args.anchors_per_class
  )
  if len(eligible_classes) < 2:
    raise RuntimeError("Fewer than two sufficiently represented common classes remain.")
  development = [
    record for record in development if record.class_id in eligible_classes
  ]
  holdout = [record for record in holdout if record.class_id in eligible_classes]
  development_metadata["eligible_class_ids"] = eligible_classes
  holdout_metadata["eligible_class_ids"] = eligible_classes
  pairs = [
    *_build_pairs(
      development,
      split="development",
      anchors_per_class=args.anchors_per_class,
      seed=_stable_seed(args.seed, "development_pairs"),
    ),
    *_build_pairs(
      holdout,
      split="holdout",
      anchors_per_class=args.anchors_per_class,
      seed=_stable_seed(args.seed, "holdout_pairs"),
    ),
  ]
  development_pairs = [pair for pair in pairs if pair.split == "development"]
  selected_rtw_config = None
  rtw_search_rows: list[dict] = []
  if args.search_rtw:
    selected_rtw_config, rtw_search_rows = _select_rtw_config(
      development_pairs, args
    )
  print(
    f"[DATA] development={len(development)}, holdout={len(holdout)}, "
    f"pairs={len(pairs)}"
  )
  score_rows = []
  method_names: list[str] | None = None
  for index, pair in enumerate(pairs, start=1):
    scores = _scores(pair, args, selected_rtw_config)
    if method_names is None:
      method_names = [
        name for name in scores if not name.endswith("sampling_std")
      ]
    score_rows.append({
      "pair_id": pair.pair_id,
      "anchor_id": pair.anchor_id,
      "split": pair.split,
      "scenario": pair.scenario,
      "label": pair.label,
      "hard_pair": int(pair.hard_pair),
      "first_class": pair.first_class,
      "second_class": pair.second_class,
      "first_name": pair.first_name,
      "second_name": pair.second_name,
      "first_steps": len(pair.first),
      "second_steps": len(pair.second),
      **scores,
    })
    if index % 100 == 0 or index == len(pairs):
      print(f"[SCORE] {index}/{len(pairs)}")
  methods = method_names or []
  metrics = _metric_rows(score_rows, methods)
  comparisons = []
  task_baselines = {}
  rtw_methods = ["rtw_frozen_raw", "rtw_per_sequence_zscore"]
  if selected_rtw_config is not None:
    rtw_methods.insert(0, "rtw_selected")
  for task in TASK_SCENARIOS:
    candidates = [
      row for row in metrics
      if row["split"] == "development"
      and row["task"] == task
      and not row["method"].startswith("rtw")
    ]
    task_baselines[task] = max(
      candidates, key=lambda row: row["average_precision"]
    )["method"]
    for rtw_method in rtw_methods:
      comparisons.append(_bootstrap_delta(
        score_rows,
        rtw_method,
        task_baselines[task],
        split="holdout",
        task=task,
        repeats=args.bootstrap,
        seed=_stable_seed(args.seed, "paired_bootstrap", task, rtw_method),
      ))
  best_baseline = task_baselines["combined"]
  primary_rtw_method = "rtw_selected" if selected_rtw_config is not None else "rtw_frozen_raw"
  delta = next(
    row for row in comparisons
    if row["task"] == "combined" and row["first_method"] == primary_rtw_method
  )
  thresholds = _development_thresholds(score_rows, methods)
  threshold_rows = _threshold_rows(score_rows, methods, thresholds)
  holdout_primary = {
    row["method"]: row for row in metrics
    if row["split"] == "holdout" and row["task"] == "combined"
  }
  timing_scenarios = {"same_field_phase_shift", "same_field_nonlinear_warp"}
  rtw_timing_fpr = np.mean([
    row["holdout_positive_rate"] for row in threshold_rows
    if row["method"] == primary_rtw_method and row["scenario"] in timing_scenarios
  ])
  baseline_timing_fpr = np.mean([
    row["holdout_positive_rate"] for row in threshold_rows
    if row["method"] == best_baseline and row["scenario"] in timing_scenarios
  ])
  go = bool(
    delta["ci_low"] > 0.0
    and rtw_timing_fpr <= baseline_timing_fpr + 0.05
  )
  decision = {
    "go": go,
    "claim": (
      "RTW has incremental natural-label phenology value beyond the development-selected simple baseline."
      if go else
      "RTW lacks verified incremental natural-label phenology value beyond the development-selected simple baseline."
    ),
    "development_selected_baseline": best_baseline,
    "primary_rtw_method": primary_rtw_method,
    "selected_rtw_config": selected_rtw_config,
    "rtw_holdout_ap": holdout_primary[primary_rtw_method]["average_precision"],
    "baseline_holdout_ap": holdout_primary[best_baseline]["average_precision"],
    "paired_delta": delta,
    "task_comparisons": comparisons,
    "rtw_timing_false_positive_rate": float(rtw_timing_fpr),
    "baseline_timing_false_positive_rate": float(baseline_timing_fpr),
  }
  metadata = {
    "dataset": "BreizhCrops 2017 L2A",
    "development": development_metadata,
    "holdout": holdout_metadata,
    "pairs": len(pairs),
    "anchors_per_class": args.anchors_per_class,
    "rtw_config": {
      "subsequence_length": 4,
      "n_samples": 64,
      "rank": 5,
      "replicates": args.rtw_replicates,
      "selection": "frozen from MultiSenGE; no BreizhCrops tuning",
    },
    "selected_rtw_config": selected_rtw_config,
    "rtw_search_enabled": bool(args.search_rtw),
    "runtime_seconds": time.perf_counter() - started,
    "seed": args.seed,
  }
  _write_csv(args.output_dir / "pair_scores.csv", score_rows)
  _write_csv(args.output_dir / "method_metrics.csv", metrics)
  _write_csv(args.output_dir / "rtw_configuration_search.csv", rtw_search_rows)
  _write_csv(args.output_dir / "holdout_threshold_behavior.csv", threshold_rows)
  _write_csv(args.output_dir / "paired_ap_deltas.csv", comparisons)
  (args.output_dir / "decision.json").write_text(
    json.dumps(decision, indent=2), encoding="utf-8"
  )
  (args.output_dir / "run_metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
  )
  _plot_method_ap(metrics, args.output_dir / "holdout_method_ap.png")
  _plot_scenario_responses(
    score_rows,
    [
      primary_rtw_method, "rtw_frozen_raw", "snapshot_uncentered", "global_shift_correlation",
      "shift_orbit_subspace", "dtw_zscore", "seasonal_summary",
      "pca_cross_reconstruction", "fourier_magnitude",
    ],
    args.output_dir / "holdout_scenario_responses.png",
  )
  _summary_markdown(
    args.output_dir / "result_summary.md",
    metrics,
    comparisons,
    best_baseline,
    metadata,
  )
  print(json.dumps({
    "output_dir": str(args.output_dir),
    "go": go,
    "best_baseline": best_baseline,
    "rtw_holdout_ap": decision["rtw_holdout_ap"],
    "baseline_holdout_ap": decision["baseline_holdout_ap"],
    "rtw_minus_baseline_ap": delta["ap_delta"],
    "rtw_minus_baseline_ci": [delta["ci_low"], delta["ci_high"]],
    "runtime_seconds": metadata["runtime_seconds"],
  }, indent=2))


if __name__ == "__main__":
  main()
