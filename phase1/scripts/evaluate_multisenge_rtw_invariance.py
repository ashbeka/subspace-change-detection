"""Falsifiable RTW gate on real MultiSenGE temporal backgrounds.

Research question
-----------------
Can a Randomized Time Warping (RTW) hypo-subspace reject harmless seasonal
timing/tempo variation while remaining sensitive to a changed seasonal-cycle
shape, beyond Fourier, harmonic, DTW/TWDTW, M-SSA, and raw spectral controls?

This runner uses real 10-band, 23-date Sentinel-2 sequences as backgrounds and
adds controlled transformations with exact labels. Three patches are used to
select RTW hyperparameters; the selected configuration is frozen before two
held-out patches are scored. The result is a mechanism gate, not natural-change
accuracy and not proof of satellite novelty.

Method sources
--------------
- Suryanto, Xue, and Fukui (2016), Randomized Time Warping for Motion
  Recognition, https://doi.org/10.1016/j.imavis.2016.07.003.
- Hiraoka et al. (2025), Attention Mechanism in Randomized Time Warping,
  arXiv:2508.16366, Sections 2.1--2.3.
- Maus et al. (2016), TWDTW for land-cover mapping,
  https://doi.org/10.1109/JSTARS.2016.2517118.
- Bundled MATLAB RTW code under
  ``references/reference_code/Subspace Toolbox/RTW``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from phase1.baselines.temporal_sequence import (
  aligned_rms_distance,
  dtw_distance,
  fourier_magnitude_distance,
  harmonic_phase_aligned_distance,
  mean_spectral_angle_distance,
  mean_spectrum_distance,
  mssa_subspace_distance,
  preprocess_pair,
  seasonal_amplitude_distance,
  snapshot_subspace_distance,
  soft_dtw_divergence,
  time_weighted_dtw_distance,
)
from phase1.data.multisenge_manifest import load_manifest
from phase1.scripts.summarize_multisenge_rtw_invariance import summarize_output
from phase1.subspace.randomized_time_warping import compare_rtw_sequences
from phase1.subspace.temporal_band_images import sequence_common_valid_mask


@dataclass(frozen=True)
class SequencePair:
  sample_id: str
  patch_id: str
  repeat: int
  split: str
  scenario: str
  family: str
  severity: float
  is_nuisance: bool
  is_shape_target: bool
  is_structural_target: bool
  is_control_target: bool
  is_adversarial: bool
  first: np.ndarray
  first_times: np.ndarray
  second: np.ndarray
  second_times: np.ndarray


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--multisenge_root", required=True, type=Path)
  parser.add_argument("--manifest", required=True, type=Path)
  parser.add_argument("--output_dir", required=True, type=Path)
  parser.add_argument("--crop_size", type=int, default=32)
  parser.add_argument("--repeats", type=int, default=6)
  parser.add_argument("--development_patches", type=int, default=3)
  parser.add_argument("--max_patches", type=int, default=5)
  parser.add_argument("--subsequence_lengths", default="4,8,12")
  parser.add_argument("--n_samples", default="64,256")
  parser.add_argument("--ranks", default="2,5")
  parser.add_argument(
    "--preprocessing",
    default="raw,reference_zscore,per_sequence_zscore",
  )
  parser.add_argument("--rtw_replicates", type=int, default=3)
  parser.add_argument("--screening_repeats", type=int, default=2)
  parser.add_argument("--screening_rtw_replicates", type=int, default=1)
  parser.add_argument("--finalists", type=int, default=6)
  parser.add_argument("--bootstrap", type=int, default=500)
  parser.add_argument("--seed", type=int, default=1234)
  return parser.parse_args()


def _stable_seed(*parts: object) -> int:
  digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
  return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def _parse_ints(value: str) -> list[int]:
  return sorted({int(item) for item in value.split(",") if item.strip()})


def _parse_strings(value: str) -> list[str]:
  return [item.strip() for item in value.split(",") if item.strip()]


def _load_cube(path: Path) -> np.ndarray:
  with rasterio.open(path) as source:
    return source.read().astype(np.float32) / 10000.0


def _find_crop(mask: np.ndarray, size: int, rng: np.random.Generator) -> tuple[slice, slice]:
  height, width = mask.shape
  if size > height or size > width:
    raise ValueError(f"crop_size={size} exceeds image shape={mask.shape}.")
  candidates = np.argwhere(mask)
  for _ in range(2000):
    row, col = candidates[int(rng.integers(0, len(candidates)))]
    top = int(np.clip(row - size // 2, 0, height - size))
    left = int(np.clip(col - size // 2, 0, width - size))
    region = mask[top : top + size, left : left + size]
    if float(np.mean(region)) >= 0.98:
      return slice(top, top + size), slice(left, left + size)
  raise RuntimeError(f"No >=98% valid {size}x{size} crop found.")


def _date_offsets(date_strings: list[str]) -> np.ndarray:
  dates = [datetime.strptime(value, "%Y%m%d") for value in date_strings]
  start = datetime(dates[0].year, 1, 1)
  return np.asarray([(date - start).days for date in dates], dtype=np.float64)


def _mean_sequence(cubes: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
  values = np.stack([
    np.mean(cube[:, mask].astype(np.float64), axis=1)
    for cube in cubes
  ])
  return np.clip(values, 1e-5, 1.5)


def _independent_jitter(
  sequence: np.ndarray,
  rng: np.random.Generator,
  *,
  noise: float = 0.0012,
) -> np.ndarray:
  gain = rng.normal(1.0, 0.004, size=(len(sequence), 1))
  offset = rng.normal(0.0, 0.0005, size=(len(sequence), 1))
  perturbation = rng.normal(0.0, noise, size=sequence.shape)
  return np.clip(gain * sequence + offset + perturbation, 1e-5, 1.5)


def _normalized_time(times: np.ndarray) -> np.ndarray:
  values = np.asarray(times, dtype=np.float64)
  return (values - values[0]) / max(float(values[-1] - values[0]), 1.0)


def _periodic_query(sequence: np.ndarray, times: np.ndarray, query: np.ndarray) -> np.ndarray:
  t = _normalized_time(times)
  extended_t = np.concatenate([t - 1.0, t, t + 1.0])
  return np.stack([
    np.interp(query, extended_t, np.tile(sequence[:, feature], 3))
    for feature in range(sequence.shape[1])
  ], axis=1)


def _phase_shift(sequence: np.ndarray, times: np.ndarray, composites: int) -> np.ndarray:
  shift = float(composites) / max(len(sequence), 1)
  query = np.mod(_normalized_time(times) - shift, 1.0)
  return _periodic_query(sequence, times, query)


def _nonlinear_warp(sequence: np.ndarray, times: np.ndarray, exponent: float) -> np.ndarray:
  normalized = _normalized_time(times)
  query = np.power(normalized, float(exponent))
  return np.stack([
    np.interp(query, normalized, sequence[:, feature])
    for feature in range(sequence.shape[1])
  ], axis=1)


def _ndvi(sequence: np.ndarray) -> np.ndarray:
  red = sequence[:, 2]
  nir = sequence[:, 6]
  return (nir - red) / np.maximum(nir + red, 1e-6)


def _phenology_shape_change(
  sequence: np.ndarray,
  times: np.ndarray,
  strength: float,
) -> np.ndarray:
  output = sequence.copy()
  normalized = _normalized_time(times)
  peak = float(normalized[int(np.argmax(_ndvi(sequence)))])
  distance = np.abs(normalized - peak)
  distance = np.minimum(distance, 1.0 - distance)
  envelope = np.exp(-0.5 * (distance / 0.16) ** 2)
  output[:, 2] *= 1.0 + float(strength) * envelope
  output[:, 6] *= 1.0 - 1.25 * float(strength) * envelope
  output[:, 7] *= 1.0 - 1.00 * float(strength) * envelope
  output[:, 8] *= 1.0 + 0.65 * float(strength) * envelope
  output[:, 9] *= 1.0 + 0.80 * float(strength) * envelope
  return np.clip(output, 1e-5, 1.5)


def _season_shortening(sequence: np.ndarray, strength: float) -> np.ndarray:
  ndvi = _ndvi(sequence)
  activity = (ndvi - np.min(ndvi)) / max(float(np.ptp(ndvi)), 1e-6)
  sharpened = np.power(activity, 1.0 + 4.0 * float(strength))
  blend = 0.20 + 0.80 * sharpened
  baseline = np.quantile(sequence, 0.20, axis=0, keepdims=True)
  return np.clip(baseline + blend[:, None] * (sequence - baseline), 1e-5, 1.5)


def _marginal_match(changed: np.ndarray, reference: np.ndarray) -> np.ndarray:
  """Replace per-band ranks so every marginal exactly matches ``reference``."""
  output = np.empty_like(changed)
  for band in range(changed.shape[1]):
    changed_order = np.argsort(changed[:, band], kind="stable")
    output[changed_order, band] = np.sort(reference[:, band])
  return output


def _relative_band_phase(
  sequence: np.ndarray,
  times: np.ndarray,
  composites: int,
) -> np.ndarray:
  """Shift only NIR/SWIR trajectories, preserving each band's marginal values."""
  output = sequence.copy()
  shifted = np.roll(sequence, int(composites), axis=0)
  output[:, [6, 7, 8, 9]] = shifted[:, [6, 7, 8, 9]]
  return output


def _transform(
  sequence: np.ndarray,
  times: np.ndarray,
  *,
  scenario: str,
  severity: float,
  rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
  output = sequence.copy()
  output_times = times.copy()
  if scenario == "stable_jitter":
    return output, output_times
  if scenario == "noise":
    return np.clip(output + rng.normal(0.0, severity, size=output.shape), 1e-5, 1.5), output_times
  if scenario == "phase_shift":
    return _phase_shift(output, output_times, int(severity)), output_times
  if scenario == "nonlinear_warp":
    return _nonlinear_warp(output, output_times, severity), output_times
  if scenario == "missing_composites":
    count = max(1, int(round(len(output) * severity)))
    removable = np.arange(1, len(output) - 1)
    removed = rng.choice(removable, size=min(count, len(removable)), replace=False)
    keep = np.ones(len(output), dtype=bool)
    keep[removed] = False
    return output[keep], output_times[keep]
  if scenario == "global_gain_offset":
    return np.clip((1.0 + severity) * output + 0.004, 1e-5, 1.5), output_times
  if scenario == "band_gain":
    pattern = np.linspace(-1.0, 1.0, output.shape[1])
    return np.clip(output * (1.0 + severity * pattern[None, :]), 1e-5, 1.5), output_times
  if scenario == "phenology_shape":
    return _phenology_shape_change(output, output_times, severity), output_times
  if scenario == "marginal_matched_shape":
    changed = _phenology_shape_change(output, output_times, severity)
    return _marginal_match(changed, output), output_times
  if scenario == "relative_band_phase":
    return _relative_band_phase(output, output_times, int(severity)), output_times
  if scenario == "season_shortening":
    return _season_shortening(output, severity), output_times
  if scenario == "amplitude_change":
    mean = np.mean(output, axis=0, keepdims=True)
    return np.clip(mean + severity * (output - mean), 1e-5, 1.5), output_times
  if scenario == "mean_shift":
    direction = np.asarray([0.2, 0.2, 0.6, 0.1, 0.1, 0.1, -0.8, -0.6, 0.7, 1.0])
    return np.clip(output + severity * direction[None, :], 1e-5, 1.5), output_times
  if scenario == "date_permutation":
    order = rng.permutation(len(output))
    return output[order], output_times
  raise ValueError(scenario)


SCENARIOS = (
  # scenario, family, severities, nuisance, shape, structural, control, adversarial
  ("stable_jitter", "stable", (0.0,), True, False, False, False, False),
  ("noise", "noise", (0.002, 0.006), True, False, False, False, False),
  ("phase_shift", "phase", (1.0, 2.0, 4.0), True, False, False, False, False),
  ("nonlinear_warp", "warp", (0.75, 1.25), True, False, False, False, False),
  ("missing_composites", "missing", (0.10, 0.20), True, False, False, False, False),
  ("global_gain_offset", "radiometric", (0.05, 0.10), True, False, False, False, False),
  ("band_gain", "radiometric", (0.05, 0.10), True, False, False, False, False),
  ("marginal_matched_shape", "structural_shape", (0.15, 0.30), False, True, True, False, False),
  ("relative_band_phase", "structural_shape", (2.0, 4.0), False, True, True, False, False),
  ("phenology_shape", "shape", (0.15, 0.30), False, True, False, False, False),
  ("season_shortening", "shape", (0.25, 0.50), False, True, False, False, False),
  ("amplitude_change", "control_change", (1.25, 1.50), False, False, False, True, False),
  ("mean_shift", "control_change", (0.01, 0.02), False, False, False, True, False),
  ("date_permutation", "adversarial", (1.0,), False, False, False, False, True),
)


def _build_pairs(args: argparse.Namespace) -> tuple[list[SequencePair], dict]:
  manifest = load_manifest(args.manifest)
  entries = manifest.get("patches", [])[: args.max_patches or None]
  if len(entries) < 2:
    raise ValueError("At least two MultiSenGE patches are required for a split.")
  if args.development_patches < 1 or args.development_patches >= len(entries):
    raise ValueError("development_patches must leave at least one held-out patch.")
  development_ids = {entry["patch_id"] for entry in entries[: args.development_patches]}
  pairs: list[SequencePair] = []
  patch_metadata: list[dict] = []
  for entry in entries:
    ordered = sorted(entry.get("s2", []), key=lambda item: item["date"])
    if len(ordered) < 16:
      continue
    cubes = [_load_cube(args.multisenge_root / item["relpath"]) for item in ordered]
    common_mask = sequence_common_valid_mask(cubes, nodata_value=0.0)
    times = _date_offsets([item["date"] for item in ordered])
    split = "development" if entry["patch_id"] in development_ids else "holdout"
    for repeat in range(args.repeats):
      rng = np.random.default_rng(_stable_seed(args.seed, entry["patch_id"], repeat, "crop"))
      crop = _find_crop(common_mask, args.crop_size, rng)
      cropped = [cube[:, crop[0], crop[1]] for cube in cubes]
      mask = common_mask[crop[0], crop[1]]
      base = _mean_sequence(cropped, mask)
      first = _independent_jitter(
        base,
        np.random.default_rng(_stable_seed(args.seed, entry["patch_id"], repeat, "first")),
      )
      second_base = _independent_jitter(
        base,
        np.random.default_rng(_stable_seed(args.seed, entry["patch_id"], repeat, "second")),
      )
      for scenario, family, severities, nuisance, shape, structural, control, adversarial in SCENARIOS:
        for severity in severities:
          scenario_rng = np.random.default_rng(
            _stable_seed(args.seed, entry["patch_id"], repeat, scenario, severity)
          )
          second, second_times = _transform(
            second_base,
            times,
            scenario=scenario,
            severity=float(severity),
            rng=scenario_rng,
          )
          pairs.append(SequencePair(
            sample_id=f"{entry['patch_id']}:{repeat}:{scenario}:{severity:g}",
            patch_id=entry["patch_id"],
            repeat=repeat,
            split=split,
            scenario=scenario,
            family=family,
            severity=float(severity),
            is_nuisance=nuisance,
            is_shape_target=shape,
            is_structural_target=structural,
            is_control_target=control,
            is_adversarial=adversarial,
            first=first,
            first_times=times,
            second=second,
            second_times=second_times,
          ))
    patch_metadata.append({
      "patch_id": entry["patch_id"],
      "split": split,
      "dates": len(ordered),
      "first_date": ordered[0]["date"],
      "last_date": ordered[-1]["date"],
    })
    print(f"[DATA] {entry['patch_id']}: split={split}, dates={len(ordered)}")
  return pairs, {
    "patches": patch_metadata,
    "development_patch_ids": sorted(development_ids),
    "holdout_patch_ids": sorted({pair.patch_id for pair in pairs if pair.split == "holdout"}),
  }


def _configurations(args: argparse.Namespace, pairs: list[SequencePair]) -> list[dict]:
  minimum_steps = min(min(len(pair.first), len(pair.second)) for pair in pairs)
  configs = []
  for length in _parse_ints(args.subsequence_lengths):
    if length > minimum_steps:
      continue
    for n_samples in _parse_ints(args.n_samples):
      for rank in _parse_ints(args.ranks):
        for preprocessing in _parse_strings(args.preprocessing):
          configs.append({
            "config_id": f"R{length}_L{n_samples}_m{rank}_{preprocessing}",
            "subsequence_length": length,
            "n_samples": n_samples,
            "rank": rank,
            "preprocessing": preprocessing,
          })
  if not configs:
    raise ValueError("No valid RTW configurations remain after sequence-length checks.")
  return configs


def _rtw_score(
  pair: SequencePair,
  config: dict,
  args: argparse.Namespace,
  *,
  replicates: int | None = None,
) -> dict:
  first, second = preprocess_pair(pair.first, pair.second, config["preprocessing"])
  result = compare_rtw_sequences(
    first,
    second,
    subsequence_length=config["subsequence_length"],
    n_samples=config["n_samples"],
    rank=config["rank"],
    seed=_stable_seed(args.seed, pair.sample_id, config["config_id"]),
    replicates=args.rtw_replicates if replicates is None else int(replicates),
  )
  return {
    "score": float(result["dissimilarity"]),
    "score_std": float(result["dissimilarity_std"]),
    "te_energy_change": float(result["te_energy_change"]),
  }


def _baseline_scores(pair: SequencePair) -> dict[str, float]:
  raw_first, raw_second = preprocess_pair(pair.first, pair.second, "raw")
  ref_first, ref_second = preprocess_pair(pair.first, pair.second, "reference_zscore")
  shape_first, shape_second = preprocess_pair(pair.first, pair.second, "per_sequence_zscore")
  first_ndvi = _ndvi(pair.first)[:, None]
  second_ndvi = _ndvi(pair.second)[:, None]
  return {
    "aligned_rms_raw": aligned_rms_distance(
      raw_first, pair.first_times, raw_second, pair.second_times
    ),
    "aligned_rms_reference_zscore": aligned_rms_distance(
      ref_first, pair.first_times, ref_second, pair.second_times
    ),
    "mean_spectrum_l2": mean_spectrum_distance(raw_first, raw_second),
    "seasonal_amplitude_l2": seasonal_amplitude_distance(raw_first, raw_second),
    "mean_spectral_angle": mean_spectral_angle_distance(
      raw_first, pair.first_times, raw_second, pair.second_times
    ),
    "ndvi_aligned_rms": aligned_rms_distance(
      first_ndvi, pair.first_times, second_ndvi, pair.second_times
    ),
    "fourier_magnitude": fourier_magnitude_distance(
      shape_first, pair.first_times, shape_second, pair.second_times
    ),
    "harmonic_phase_aligned": harmonic_phase_aligned_distance(
      shape_first, pair.first_times, shape_second, pair.second_times
    ),
    "dtw_reference_zscore": dtw_distance(ref_first, ref_second),
    "twdtw_reference_zscore": time_weighted_dtw_distance(
      ref_first,
      pair.first_times,
      ref_second,
      pair.second_times,
    ),
    "soft_dtw_reference_zscore": soft_dtw_divergence(
      ref_first, ref_second, gamma=0.5
    ),
    "snapshot_subspace": snapshot_subspace_distance(
      shape_first, shape_second, rank=3
    ),
    "mssa_subspace": mssa_subspace_distance(
      shape_first, shape_second, lag=4, rank=3
    ),
  }


def _evaluate_rtw_config(
  config: dict,
  pairs: list[SequencePair],
  args: argparse.Namespace,
  *,
  replicates: int,
  stage: str,
) -> dict:
  scored = []
  for pair in pairs:
    score = _rtw_score(pair, config, args, replicates=replicates)
    scored.append({**_row_metadata(pair), "rtw_score": score["score"]})
  selected = []
  for row in scored:
    eligible, label = _task_label(row, "structural_shape_vs_timing")
    if eligible:
      selected.append((label, float(row["rtw_score"])))
  labels = np.asarray([item[0] for item in selected], dtype=int)
  values = np.asarray([item[1] for item in selected], dtype=float)
  threshold, best_f1 = _best_f1_threshold(labels, values)
  positive = values[labels == 1]
  negative = values[labels == 0]
  return {
    **config,
    "stage": stage,
    "evaluated_pairs": len(pairs),
    "rtw_replicates": int(replicates),
    "auroc": float(roc_auc_score(labels, values)),
    "average_precision": float(average_precision_score(labels, values)),
    "best_f1": best_f1,
    "best_threshold": threshold,
    "median_margin": float(np.median(positive) - np.median(negative)),
    "complexity": int(config["subsequence_length"] * config["n_samples"] * config["rank"]),
  }


def _row_metadata(pair: SequencePair) -> dict:
  return {
    "sample_id": pair.sample_id,
    "patch_id": pair.patch_id,
    "repeat": pair.repeat,
    "split": pair.split,
    "scenario": pair.scenario,
    "family": pair.family,
    "severity": pair.severity,
    "is_nuisance": int(pair.is_nuisance),
    "is_shape_target": int(pair.is_shape_target),
    "is_structural_target": int(pair.is_structural_target),
    "is_control_target": int(pair.is_control_target),
    "is_adversarial": int(pair.is_adversarial),
  }


def _task_label(row: dict, task: str) -> tuple[bool, int]:
  if task == "structural_shape_vs_timing":
    eligible = bool(row["is_structural_target"]) or row["family"] in {"stable", "phase", "warp"}
    return eligible, int(row["is_structural_target"])
  if task == "shape_vs_timing":
    eligible = bool(row["is_shape_target"]) or row["family"] in {"stable", "phase", "warp"}
    return eligible, int(row["is_shape_target"])
  if task == "shape_vs_all_nuisance":
    eligible = bool(row["is_shape_target"]) or bool(row["is_nuisance"])
    return eligible, int(row["is_shape_target"])
  if task == "all_change_vs_all_nuisance":
    positive = bool(row["is_shape_target"]) or bool(row["is_control_target"])
    eligible = positive or bool(row["is_nuisance"])
    return eligible, int(positive)
  raise ValueError(task)


def _best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
  candidates = np.unique(scores)
  if candidates.size > 500:
    candidates = np.quantile(scores, np.linspace(0.0, 1.0, 500))
  best = (-1.0, float(candidates[0]))
  for threshold in candidates:
    value = float(f1_score(labels, scores >= threshold, zero_division=0))
    if value > best[0]:
      best = (value, float(threshold))
  return best[1], best[0]


def _cluster_bootstrap(
  rows: list[dict],
  score_key: str,
  task: str,
  repeats: int,
  seed: int,
) -> tuple[float, float, float, float]:
  selected = []
  for row in rows:
    eligible, label = _task_label(row, task)
    if eligible:
      selected.append((row, label))
  patch_ids = sorted({row["patch_id"] for row, _ in selected})
  rng = np.random.default_rng(seed)
  aucs, aps = [], []
  for _ in range(max(int(repeats), 0)):
    sampled = rng.choice(patch_ids, size=len(patch_ids), replace=True)
    labels, scores = [], []
    for patch in sampled:
      for row, label in selected:
        if row["patch_id"] == patch:
          labels.append(label)
          scores.append(float(row[score_key]))
    if np.unique(labels).size < 2:
      continue
    aucs.append(float(roc_auc_score(labels, scores)))
    aps.append(float(average_precision_score(labels, scores)))
  if not aucs:
    return (float("nan"),) * 4
  return (
    float(np.quantile(aucs, 0.025)),
    float(np.quantile(aucs, 0.975)),
    float(np.quantile(aps, 0.025)),
    float(np.quantile(aps, 0.975)),
  )


def _method_summary(
  rows: list[dict],
  methods: list[str],
  args: argparse.Namespace,
) -> list[dict]:
  output = []
  tasks = (
    "structural_shape_vs_timing",
    "shape_vs_timing",
    "shape_vs_all_nuisance",
    "all_change_vs_all_nuisance",
  )
  for task in tasks:
    development_thresholds = {}
    for method in methods:
      dev = []
      for row in rows:
        eligible, label = _task_label(row, task)
        if row["split"] == "development" and eligible:
          dev.append((label, float(row[method])))
      labels = np.asarray([item[0] for item in dev], dtype=int)
      scores = np.asarray([item[1] for item in dev], dtype=float)
      development_thresholds[method] = _best_f1_threshold(labels, scores)[0]
    for split in ("development", "holdout", "all"):
      split_rows = rows if split == "all" else [row for row in rows if row["split"] == split]
      for method in methods:
        selected = []
        for row in split_rows:
          eligible, label = _task_label(row, task)
          if eligible:
            selected.append((row, label))
        labels = np.asarray([item[1] for item in selected], dtype=int)
        scores = np.asarray([float(item[0][method]) for item in selected], dtype=float)
        if np.unique(labels).size < 2:
          continue
        threshold, best_f1 = _best_f1_threshold(labels, scores)
        fixed_threshold = development_thresholds[method]
        fixed_f1 = float(f1_score(labels, scores >= fixed_threshold, zero_division=0))
        negatives = scores[labels == 0]
        positives = scores[labels == 1]
        auc_low, auc_high, ap_low, ap_high = _cluster_bootstrap(
          split_rows,
          method,
          task,
          args.bootstrap,
          _stable_seed(args.seed, split, task, method, "bootstrap"),
        )
        output.append({
          "method": method,
          "split": split,
          "task": task,
          "n": int(labels.size),
          "positives": int(np.sum(labels)),
          "negatives": int(np.sum(labels == 0)),
          "auroc": float(roc_auc_score(labels, scores)),
          "auroc_ci_low": auc_low,
          "auroc_ci_high": auc_high,
          "average_precision": float(average_precision_score(labels, scores)),
          "average_precision_ci_low": ap_low,
          "average_precision_ci_high": ap_high,
          "best_f1": best_f1,
          "best_threshold": threshold,
          "development_threshold": fixed_threshold,
          "development_threshold_f1": fixed_f1,
          "median_positive": float(np.median(positives)),
          "median_negative": float(np.median(negatives)),
          "median_margin": float(np.median(positives) - np.median(negatives)),
          "median_ratio": float(np.median(positives) / max(abs(float(np.median(negatives))), 1e-12)),
        })
  return output


def _write_csv(path: Path, rows: list[dict]) -> None:
  if not rows:
    return
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def _plot_config_search(rows: list[dict], output: Path) -> None:
  top = sorted(rows, key=lambda row: float(row["average_precision"]), reverse=True)[:20]
  figure, axis = plt.subplots(figsize=(11, 7))
  positions = np.arange(len(top))
  axis.barh(positions, [row["average_precision"] for row in top], color="#3c7c6f")
  axis.set_yticks(positions, [row["config_id"] for row in top])
  axis.invert_yaxis()
  axis.set_xlim(0.0, 1.0)
  axis.set_xlabel("Development AP: shape change versus timing nuisance")
  axis.set_title("RTW configuration search (development patches only)")
  axis.grid(True, axis="x", alpha=0.25)
  figure.tight_layout()
  figure.savefig(output, dpi=180)
  plt.close(figure)


def _plot_holdout_methods(summary: list[dict], output: Path) -> None:
  selected = [
    row for row in summary
    if row["split"] == "holdout" and row["task"] == "structural_shape_vs_timing"
  ]
  selected.sort(key=lambda row: float(row["average_precision"]))
  figure, axis = plt.subplots(figsize=(11, 7))
  positions = np.arange(len(selected))
  values = np.asarray([row["average_precision"] for row in selected])
  lower = np.asarray([row["average_precision_ci_low"] for row in selected])
  upper = np.asarray([row["average_precision_ci_high"] for row in selected])
  errors = np.vstack([np.maximum(values - lower, 0.0), np.maximum(upper - values, 0.0)])
  colors = ["#c4553d" if row["method"] == "rtw_selected" else "#4b6f8a" for row in selected]
  axis.barh(positions, values, xerr=errors, color=colors, alpha=0.9, capsize=3)
  axis.set_yticks(positions, [row["method"] for row in selected])
  axis.set_xlim(0.0, 1.0)
  axis.set_xlabel("Held-out AP: marginal-matched structural change versus timing nuisance")
  axis.set_title("RTW gate against temporal and spectral controls")
  axis.grid(True, axis="x", alpha=0.25)
  figure.tight_layout()
  figure.savefig(output, dpi=180)
  plt.close(figure)


def _plot_scenario_heatmap(rows: list[dict], methods: list[str], output: Path) -> None:
  holdout = [row for row in rows if row["split"] == "holdout"]
  scenarios = sorted({f"{row['scenario']}:{row['severity']:g}" for row in holdout})
  matrix = np.zeros((len(methods), len(scenarios)), dtype=float)
  for method_index, method in enumerate(methods):
    values = []
    for scenario in scenarios:
      scenario_values = [
        float(row[method]) for row in holdout
        if f"{row['scenario']}:{row['severity']:g}" == scenario
      ]
      values.append(float(np.median(scenario_values)))
    values = np.asarray(values)
    low, high = np.quantile(values, [0.05, 0.95])
    matrix[method_index] = np.clip((values - low) / max(float(high - low), 1e-12), 0.0, 1.0)
  figure, axis = plt.subplots(figsize=(16, max(5, 0.55 * len(methods))))
  image = axis.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
  axis.set_xticks(np.arange(len(scenarios)), scenarios, rotation=55, ha="right")
  axis.set_yticks(np.arange(len(methods)), methods)
  axis.set_title("Held-out median response by intervention (row-normalized)")
  figure.colorbar(image, ax=axis, label="Within-method normalized response")
  figure.tight_layout()
  figure.savefig(output, dpi=180)
  plt.close(figure)


def _plot_example_curves(pairs: list[SequencePair], output: Path) -> None:
  candidates = [
    pair for pair in pairs
    if pair.split == "holdout" and pair.repeat == 0 and (
      (pair.scenario == "stable_jitter")
      or (pair.scenario == "phase_shift" and pair.severity == 4.0)
      or (pair.scenario == "nonlinear_warp" and pair.severity == 1.25)
      or (pair.scenario == "marginal_matched_shape" and pair.severity == 0.30)
      or (pair.scenario == "relative_band_phase" and pair.severity == 4.0)
    )
  ]
  if not candidates:
    return
  patch = candidates[0].patch_id
  candidates = [pair for pair in candidates if pair.patch_id == patch]
  figure, axes = plt.subplots(len(candidates), 1, figsize=(11, 2.7 * len(candidates)), sharex=True)
  axes = np.atleast_1d(axes)
  for axis, pair in zip(axes, candidates):
    axis.plot(pair.first_times, _ndvi(pair.first), marker="o", label="reference", color="#315f83")
    axis.plot(pair.second_times, _ndvi(pair.second), marker="o", label="transformed", color="#c4553d")
    axis.set_ylabel("NDVI")
    axis.set_title(f"{pair.scenario}, severity={pair.severity:g}")
    axis.grid(True, alpha=0.25)
  axes[0].legend(loc="best")
  axes[-1].set_xlabel("Day of year")
  figure.suptitle(f"Controlled temporal interventions on held-out patch {patch}")
  figure.tight_layout()
  figure.savefig(output, dpi=180)
  plt.close(figure)


def _write_markdown_summary(
  path: Path,
  selected_config: dict,
  summary: list[dict],
  split_metadata: dict,
) -> None:
  holdout = [
    row for row in summary
    if row["split"] == "holdout" and row["task"] == "structural_shape_vs_timing"
  ]
  holdout.sort(key=lambda row: float(row["average_precision"]), reverse=True)
  lines = [
    "# MultiSenGE RTW Invariance Gate",
    "",
    "This is a controlled mechanism test on real satellite backgrounds, not natural-change accuracy.",
    "",
    "## Frozen RTW Configuration",
    "",
    "```json",
    json.dumps(selected_config, indent=2),
    "```",
    "",
    "## Split",
    "",
    f"- Development: {', '.join(split_metadata['development_patch_ids'])}",
    f"- Holdout: {', '.join(split_metadata['holdout_patch_ids'])}",
    "",
    "## Held-Out Marginal-Matched Structural-Shape Results",
    "",
    "| Method | AUROC | AP | AP 95% interval | Dev-threshold F1 |",
    "|---|---:|---:|---:|---:|",
  ]
  for row in holdout:
    lines.append(
      f"| {row['method']} | {row['auroc']:.4f} | {row['average_precision']:.4f} | "
      f"[{row['average_precision_ci_low']:.4f}, {row['average_precision_ci_high']:.4f}] | "
      f"{row['development_threshold_f1']:.4f} |"
    )
  rtw = next((row for row in holdout if row["method"] == "rtw_selected"), None)
  best_null = next((row for row in holdout if row["method"] != "rtw_selected"), None)
  lines.extend(["", "## Automatic Decision", ""])
  if rtw is None or best_null is None:
    lines.append("Insufficient result rows for a decision.")
  else:
    delta = float(rtw["average_precision"] - best_null["average_precision"])
    lines.append(
      f"RTW AP minus best null ({best_null['method']}) = `{delta:+.4f}`. "
      "A real-data transition gate is allowed only if this controlled result is positive beyond uncertainty."
    )
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
  started = time.perf_counter()
  args = parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  pairs, split_metadata = _build_pairs(args)
  configs = _configurations(args, pairs)
  development_pairs = [pair for pair in pairs if pair.split == "development"]

  screening_pairs = [
    pair for pair in development_pairs if pair.repeat < int(args.screening_repeats)
  ]
  print(
    f"[SCREEN] configurations={len(configs)}, pairs={len(screening_pairs)}, "
    f"replicates={args.screening_rtw_replicates}"
  )
  screening_rows: list[dict] = []
  for config_index, config in enumerate(configs, start=1):
    screening_rows.append(_evaluate_rtw_config(
      config,
      screening_pairs,
      args,
      replicates=args.screening_rtw_replicates,
      stage="screening",
    ))
    print(
      f"[SCREEN {config_index:02d}/{len(configs):02d}] {config['config_id']}: "
      f"AP={screening_rows[-1]['average_precision']:.4f}"
    )

  screening_rows.sort(
    key=lambda row: (
      -float(row["average_precision"]),
      -float(row["median_margin"]),
      int(row["complexity"]),
      str(row["config_id"]),
    )
  )
  finalist_configs = [
    {key: row[key] for key in (
      "config_id", "subsequence_length", "n_samples", "rank", "preprocessing"
    )}
    for row in screening_rows[: max(1, min(int(args.finalists), len(screening_rows)))]
  ]
  print(
    f"[FINALISTS] count={len(finalist_configs)}, pairs={len(development_pairs)}, "
    f"replicates={args.rtw_replicates}"
  )
  finalist_rows = []
  for finalist_index, config in enumerate(finalist_configs, start=1):
    finalist_rows.append(_evaluate_rtw_config(
      config,
      development_pairs,
      args,
      replicates=args.rtw_replicates,
      stage="finalist",
    ))
    print(
      f"[FINALIST {finalist_index:02d}/{len(finalist_configs):02d}] "
      f"{config['config_id']}: AP={finalist_rows[-1]['average_precision']:.4f}"
    )
  finalist_rows.sort(
    key=lambda row: (
      -float(row["average_precision"]),
      -float(row["median_margin"]),
      int(row["complexity"]),
      str(row["config_id"]),
    )
  )
  config_rows = [*screening_rows, *finalist_rows]
  selected_config = {key: finalist_rows[0][key] for key in (
    "config_id", "subsequence_length", "n_samples", "rank", "preprocessing"
  )}
  print(f"[SELECTED] {selected_config['config_id']}")

  score_rows: list[dict] = []
  baseline_methods: list[str] | None = None
  for index, pair in enumerate(pairs, start=1):
    baselines = _baseline_scores(pair)
    if baseline_methods is None:
      baseline_methods = list(baselines)
    rtw = _rtw_score(pair, selected_config, args)
    score_rows.append({
      **_row_metadata(pair),
      **baselines,
      "rtw_selected": rtw["score"],
      "rtw_sampling_std": rtw["score_std"],
      "rtw_te_energy_change": rtw["te_energy_change"],
    })
    if index % 50 == 0 or index == len(pairs):
      print(f"[SCORE] {index}/{len(pairs)}")

  methods = ["rtw_selected", *(baseline_methods or [])]
  summary = _method_summary(score_rows, methods, args)
  nuisance_rows = []
  primary_dev = [
    row for row in summary
    if row["split"] == "development" and row["task"] == "structural_shape_vs_timing"
  ]
  thresholds = {row["method"]: float(row["development_threshold"]) for row in primary_dev}
  for method in methods:
    for split in ("development", "holdout"):
      for scenario in sorted({row["scenario"] for row in score_rows if row["is_nuisance"]}):
        selected = [
          float(row[method]) for row in score_rows
          if row["split"] == split and row["scenario"] == scenario
        ]
        if selected:
          nuisance_rows.append({
            "method": method,
            "split": split,
            "scenario": scenario,
            "development_threshold": thresholds[method],
            "false_positive_rate": float(np.mean(np.asarray(selected) >= thresholds[method])),
            "median_score": float(np.median(selected)),
          })

  response_rows = []
  for method in methods:
    for split in ("development", "holdout"):
      for scenario, family, severities, *_ in SCENARIOS:
        for severity in severities:
          values = [
            float(row[method]) for row in score_rows
            if row["split"] == split
            and row["scenario"] == scenario
            and float(row["severity"]) == float(severity)
          ]
          if values:
            response_rows.append({
              "method": method,
              "split": split,
              "scenario": scenario,
              "family": family,
              "severity": severity,
              "median_score": float(np.median(values)),
              "mean_score": float(np.mean(values)),
              "std_score": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            })

  _write_csv(args.output_dir / "rtw_configuration_search.csv", config_rows)
  _write_csv(args.output_dir / "sequence_pair_scores.csv", score_rows)
  _write_csv(args.output_dir / "method_summary.csv", summary)
  _write_csv(args.output_dir / "nuisance_false_positive_rates.csv", nuisance_rows)
  _write_csv(args.output_dir / "scenario_response_curves.csv", response_rows)
  _plot_config_search(finalist_rows, args.output_dir / "rtw_configuration_search.png")
  _plot_holdout_methods(summary, args.output_dir / "holdout_method_ap.png")
  _plot_scenario_heatmap(
    score_rows,
    [
      "rtw_selected",
      "fourier_magnitude",
      "harmonic_phase_aligned",
      "dtw_reference_zscore",
      "twdtw_reference_zscore",
      "mssa_subspace",
      "aligned_rms_reference_zscore",
    ],
    args.output_dir / "holdout_scenario_heatmap.png",
  )
  _plot_example_curves(pairs, args.output_dir / "heldout_intervention_examples.png")
  _write_markdown_summary(
    args.output_dir / "result_summary.md",
    selected_config,
    summary,
    split_metadata,
  )

  metadata = {
    "study": "RTW phase/tempo invariance versus seasonal-cycle shape change",
    "source_methods": {
      "rtw": "Suryanto et al. 2016; Hiraoka et al. 2025; bundled MATLAB TEfeatures.m",
      "twdtw": "Maus et al. 2016; vwmaus/twdtw logistic local cost",
      "mssa": "deterministic multichannel Hankel/SSA control",
    },
    "claim_boundary": (
      "Controlled transformations on real MultiSenGE Sentinel-2 backgrounds. "
      "Not natural-change accuracy and not evidence of first use in remote sensing."
    ),
    "split": split_metadata,
    "selected_rtw_config": selected_config,
    "rtw_config_count": len(configs),
    "screening_pairs": len(screening_pairs),
    "screening_rtw_replicates": args.screening_rtw_replicates,
    "finalist_count": len(finalist_configs),
    "pairs": len(pairs),
    "repeats_per_patch": args.repeats,
    "rtw_replicates": args.rtw_replicates,
    "bootstrap_repeats": args.bootstrap,
    "seed": args.seed,
    "runtime_seconds": time.perf_counter() - started,
  }
  (args.output_dir / "run_metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
  )
  decision = summarize_output(
    args.output_dir,
    bootstrap=int(args.bootstrap),
    seed=args.seed + 999,
  )
  print(json.dumps({
    "output_dir": str(args.output_dir),
    "selected_rtw_config": selected_config,
    "go_to_natural_transition_gate": decision["go_to_natural_transition_gate"],
    "pairs": len(pairs),
    "runtime_seconds": metadata["runtime_seconds"],
  }, indent=2))


if __name__ == "__main__":
  main()
