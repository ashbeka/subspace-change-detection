"""Registration-robustness curve for temporal-context subspace evidence.

The experiment applies known subpixel translations to every forward-context
date on real MultiSenGE backgrounds and compares their induced response with a
known persistent local change.  Native, Gaussian low-pass, 2x2 pooled, and
phase-correlation-aligned representations are evaluated on matched interior
masks.  This is a controlled diagnostic, not real change-detection accuracy.

Temporal-context method provenance is documented in
``phase1.subspace.temporal_context``.  Phase correlation is the classical
normalized cross-power-spectrum translation estimator; it is implemented here
only for global translation correction and formula-tested on synthetic shifts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score

from phase1.data.multisenge_manifest import load_manifest
from phase1.scripts.evaluate_temporal_context_injections import _find_window, _load, _local_change
from phase1.subspace.temporal_band_images import sequence_common_valid_mask
from phase1.subspace.temporal_context import bidirectional_temporal_context_boundary

METHODS = ("temporal_context_ds", "linear_projection_novelty", "raw_adjacent_rms")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--multisenge_root", required=True, type=Path)
  parser.add_argument("--manifest", required=True, type=Path)
  parser.add_argument("--output_dir", required=True, type=Path)
  parser.add_argument("--target_date", default="20200909")
  parser.add_argument("--context_size", type=int, default=3)
  parser.add_argument("--rank", type=int, default=2)
  parser.add_argument("--preprocessing", default="centered_column_l2")
  parser.add_argument("--shifts", default="0.25,0.5,1,2")
  parser.add_argument("--strategies", default="native,gaussian1,gaussian2,pool2,pool4,phase_align")
  parser.add_argument("--local_strength", type=float, default=0.25)
  parser.add_argument("--window_size", type=int, default=32)
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--max_patches", type=int, default=5)
  parser.add_argument("--seed", type=int, default=1234)
  return parser.parse_args()


def estimate_phase_translation(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float]:
  """Estimate the `(dy, dx)` shift to apply to ``moving`` to match ``reference``."""
  if reference.shape != moving.shape or reference.ndim != 2:
    raise ValueError("Phase-correlation inputs must be equally shaped 2D arrays.")
  ref = np.asarray(reference, dtype=np.float64) - float(np.mean(reference))
  mov = np.asarray(moving, dtype=np.float64) - float(np.mean(moving))
  window = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1]))
  cross = np.fft.fft2(ref * window) * np.conj(np.fft.fft2(mov * window))
  cross /= np.maximum(np.abs(cross), 1e-15)
  correlation = np.real(np.fft.ifft2(cross))
  peak_integer = np.array(np.unravel_index(np.argmax(correlation), correlation.shape), dtype=int)
  peak = peak_integer.astype(float)
  for axis, size in enumerate(correlation.shape):
    before_index = peak_integer.copy()
    after_index = peak_integer.copy()
    before_index[axis] = (before_index[axis] - 1) % size
    after_index[axis] = (after_index[axis] + 1) % size
    before = correlation[tuple(before_index)]
    center = correlation[tuple(peak_integer)]
    after = correlation[tuple(after_index)]
    denominator = before - 2.0 * center + after
    if abs(float(denominator)) > 1e-15:
      peak[axis] += 0.5 * float(before - after) / float(denominator)
    if peak[axis] > size / 2.0:
      peak[axis] -= size
  return float(peak[0]), float(peak[1])


def _context_indices(boundary: int, size: int, count: int) -> tuple[list[int], list[int]]:
  backward = [max(0, index) for index in range(boundary - size, boundary)]
  forward = [min(count - 1, index) for index in range(boundary, boundary + size)]
  return backward, forward


def _phase_align(cubes: list[np.ndarray], boundary: int, context_size: int) -> tuple[list[np.ndarray], tuple[float, float]]:
  backward, forward = _context_indices(boundary, context_size, len(cubes))
  reference = np.median(
    np.stack([np.mean(cubes[index], axis=0) for index in backward], axis=0), axis=0
  )
  moving = np.median(
    np.stack([np.mean(cubes[index], axis=0) for index in forward], axis=0), axis=0
  )
  shift = estimate_phase_translation(reference, moving)
  output = [cube.copy() for cube in cubes]
  for index in sorted(set(forward)):
    output[index] = ndimage.shift(
      output[index], shift=(0.0, shift[0], shift[1]), order=1, mode="nearest", prefilter=False
    )
  return output, shift


def _pool(cube: np.ndarray, factor: int) -> np.ndarray:
  bands, height, width = cube.shape
  height -= height % factor
  width -= width % factor
  return cube[:, :height, :width].reshape(
    bands, height // factor, factor, width // factor, factor
  ).mean(axis=(2, 4))


def _pool_mask(mask: np.ndarray, factor: int) -> np.ndarray:
  height, width = mask.shape
  height -= height % factor
  width -= width % factor
  return mask[:height, :width].reshape(
    height // factor, factor, width // factor, factor
  ).all(axis=(1, 3))


def _transform(
  cubes: list[np.ndarray],
  mask: np.ndarray,
  strategy: str,
  boundary: int,
  context_size: int,
) -> tuple[list[np.ndarray], np.ndarray, tuple[float, float]]:
  if strategy == "native":
    return [cube.copy() for cube in cubes], mask.copy(), (0.0, 0.0)
  if strategy.startswith("gaussian"):
    sigma = float(strategy.removeprefix("gaussian"))
    return [ndimage.gaussian_filter(cube, sigma=(0, sigma, sigma)) for cube in cubes], mask.copy(), (0.0, 0.0)
  if strategy.startswith("pool"):
    factor = int(strategy.removeprefix("pool"))
    return [_pool(cube, factor) for cube in cubes], _pool_mask(mask, factor), (0.0, 0.0)
  if strategy == "phase_align":
    aligned, shift = _phase_align(cubes, boundary, context_size)
    return aligned, mask.copy(), shift
  raise ValueError(strategy)


def _maps(result) -> dict[str, np.ndarray]:
  return {
    "temporal_context_ds": result.ds_map,
    "linear_projection_novelty": result.projection_novelty_map,
    "raw_adjacent_rms": result.raw_boundary_difference_map,
  }


def _fit(cubes: list[np.ndarray], mask: np.ndarray, args, boundary: int):
  return bidirectional_temporal_context_boundary(
    cubes,
    mask,
    boundary_index=boundary,
    context_size=args.context_size,
    rank=args.rank,
    factorization="per_band",
    preprocessing=args.preprocessing,
  )


def _write(path: Path, rows: list[dict]) -> None:
  if not rows:
    return
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
  args = parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  shifts = [float(value) for value in args.shifts.split(",") if value.strip()]
  strategies = [value.strip() for value in args.strategies.split(",") if value.strip()]
  rng = np.random.default_rng(args.seed)
  rows: list[dict] = []
  patch_count = 0
  for entry in load_manifest(args.manifest).get("patches", []):
    ordered = sorted(entry.get("s2", []), key=lambda item: item["date"])
    date_to_index = {item["date"]: index for index, item in enumerate(ordered)}
    if args.target_date not in date_to_index:
      continue
    boundary = date_to_index[args.target_date]
    cubes = [_load(args.multisenge_root / item["relpath"]) for item in ordered]
    base_mask = sequence_common_valid_mask(cubes, nodata_value=0.0)
    # Exclude interpolation-filled borders for every strategy and shift.
    margin = int(np.ceil(max(shifts))) + 2
    base_mask[:margin] = False
    base_mask[-margin:] = False
    base_mask[:, :margin] = False
    base_mask[:, -margin:] = False
    for repeat in range(args.repeats):
      window = _find_window(base_mask, args.window_size, rng)
      if window is None:
        continue
      local_cubes = [cube.copy() for cube in cubes]
      _, forward = _context_indices(boundary, args.context_size, len(cubes))
      for index in sorted(set(forward)):
        local_cubes[index] = _local_change(local_cubes[index], window, args.local_strength)
      native_label = np.zeros(base_mask.shape, dtype=np.uint8)
      native_label[window] = 1
      for strategy in strategies:
        clean_t, mask_t, clean_estimate = _transform(
          cubes, base_mask, strategy, boundary, args.context_size
        )
        local_t, local_mask, local_estimate = _transform(
          local_cubes, base_mask, strategy, boundary, args.context_size
        )
        if strategy.startswith("pool"):
          factor = int(strategy.removeprefix("pool"))
          label_t = _pool_mask(native_label.astype(bool), factor).astype(np.uint8)
        else:
          label_t = native_label
        mask_t &= local_mask
        clean_maps = _maps(_fit(clean_t, mask_t, args, boundary))
        local_maps = _maps(_fit(local_t, mask_t, args, boundary))
        local_delta = {
          method: np.maximum(local_maps[method] - clean_maps[method], 0.0)
          for method in METHODS
        }
        for shift_amount in shifts:
          translated = [cube.copy() for cube in cubes]
          for index in sorted(set(forward)):
            translated[index] = ndimage.shift(
              translated[index],
              shift=(0.0, 0.0, shift_amount),
              order=1,
              mode="nearest",
              prefilter=False,
            )
          translated_t, translated_mask, estimated = _transform(
            translated, base_mask, strategy, boundary, args.context_size
          )
          evaluation_mask = mask_t & translated_mask
          translated_maps = _maps(_fit(translated_t, evaluation_mask, args, boundary))
          for method in METHODS:
            translation_delta = np.maximum(
              translated_maps[method] - clean_maps[method], 0.0
            )
            local_values = local_delta[method][evaluation_mask]
            shift_values = translation_delta[evaluation_mask]
            labels = label_t[evaluation_mask]
            rows.append({
              "patch_id": entry["patch_id"],
              "repeat": repeat,
              "strategy": strategy,
              "shift_pixels": shift_amount,
              "method": method,
              "estimated_clean_dy": clean_estimate[0],
              "estimated_clean_dx": clean_estimate[1],
              "estimated_shifted_dy": estimated[0],
              "estimated_shifted_dx": estimated[1],
              "estimated_local_dy": local_estimate[0],
              "estimated_local_dx": local_estimate[1],
              "local_delta_sum": float(np.sum(local_values)),
              "translation_delta_sum": float(np.sum(shift_values)),
              "translation_to_local_ratio": float(np.sum(shift_values) / max(np.sum(local_values), 1e-15)),
              "localization_auroc": float(roc_auc_score(labels, local_values)),
              "localization_ap": float(average_precision_score(labels, local_values)),
            })
    patch_count += 1
    print(f"[OK] {entry['patch_id']}: valid={int(base_mask.sum())}")
    if args.max_patches and patch_count >= args.max_patches:
      break
  if not rows:
    raise RuntimeError("No registration-curve rows were produced.")
  _write(args.output_dir / "registration_curve_scores.csv", rows)
  summary: list[dict] = []
  for strategy in strategies:
    for shift_amount in shifts:
      for method in METHODS:
        subset = [
          row for row in rows
          if row["strategy"] == strategy
          and row["shift_pixels"] == shift_amount
          and row["method"] == method
        ]
        summary.append({
          "strategy": strategy,
          "shift_pixels": shift_amount,
          "method": method,
          "n": len(subset),
          "mean_translation_to_local_ratio": float(np.mean([row["translation_to_local_ratio"] for row in subset])),
          "mean_localization_ap": float(np.mean([row["localization_ap"] for row in subset])),
          "mean_estimated_dx": float(np.mean([row["estimated_shifted_dx"] for row in subset])),
        })
  _write(args.output_dir / "registration_curve_summary.csv", summary)
  (args.output_dir / "run_metadata.json").write_text(json.dumps({
    "context_size": args.context_size,
    "rank": args.rank,
    "shifts": shifts,
    "strategies": strategies,
    "local_strength": args.local_strength,
    "window_size": args.window_size,
    "patch_count": patch_count,
    "interpretation": "Controlled registration sensitivity relative to a known persistent injection.",
  }, indent=2), encoding="utf-8")
  print(f"Done. Outputs in: {args.output_dir}")


if __name__ == "__main__":
  main()
