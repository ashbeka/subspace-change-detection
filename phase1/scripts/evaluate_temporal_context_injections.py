"""Controlled intervention tests for bidirectional temporal-context DS.

This diagnostic uses real MultiSenGE Sentinel-2 sequences as backgrounds but
adds changes with known masks.  It does not claim real change-detection
accuracy.  Persistent and transient local changes are contrasted with global
gain/offset and one-pixel registration nuisances.

The tested temporal-context construction is implemented and sourced in
``phase1.subspace.temporal_context``.  Raw adjacent RMS is retained as a
control so benefits from temporal context are not attributed to DS by default.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from sklearn.metrics import average_precision_score, roc_auc_score

from phase1.data.multisenge_manifest import load_manifest
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
  parser.add_argument("--factorization", default="per_band", choices=["per_band", "joint"])
  parser.add_argument("--preprocessing", default="centered_column_l2")
  parser.add_argument("--repeats", type=int, default=4)
  parser.add_argument("--max_patches", type=int, default=5)
  parser.add_argument("--seed", type=int, default=1234)
  return parser.parse_args()


def _load(path: Path) -> np.ndarray:
  with rasterio.open(path) as source:
    return source.read().astype(np.float32)


def _find_window(
  mask: np.ndarray, size: int, rng: np.random.Generator
) -> tuple[slice, slice] | None:
  height, width = mask.shape
  candidates = np.argwhere(mask)
  for _ in range(500):
    row, col = candidates[int(rng.integers(0, len(candidates)))]
    top = int(np.clip(row - size // 2, 0, height - size))
    left = int(np.clip(col - size // 2, 0, width - size))
    if float(np.mean(mask[top:top + size, left:left + size])) >= 0.9:
      return slice(top, top + size), slice(left, left + size)
  return None


def _local_change(cube: np.ndarray, window: tuple[slice, slice], strength: float) -> np.ndarray:
  changed = cube.copy()
  rows, cols = window
  # MultiSenGE S2 order: B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12.
  changed[2, rows, cols] *= 1.0 + 0.15 * strength
  changed[6, rows, cols] *= 1.0 - strength
  changed[7, rows, cols] *= 1.0 - 0.8 * strength
  changed[8, rows, cols] *= 1.0 + 0.5 * strength
  changed[9, rows, cols] *= 1.0 + 0.7 * strength
  return changed


def _intervene(
  cubes: list[np.ndarray],
  *,
  boundary: int,
  context_size: int,
  kind: str,
  strength: float,
  window: tuple[slice, slice],
) -> list[np.ndarray]:
  output = [cube.copy() for cube in cubes]
  forward = list(range(boundary, min(len(cubes), boundary + context_size)))
  if kind == "clean":
    return output
  if kind == "local_persistent":
    for index in forward:
      output[index] = _local_change(output[index], window, strength)
    return output
  if kind == "local_transient":
    output[boundary] = _local_change(output[boundary], window, strength)
    return output
  if kind == "local_gradual":
    for step, index in enumerate(forward, start=1):
      output[index] = _local_change(output[index], window, strength * step / len(forward))
    return output
  if kind == "global_gain":
    for index in forward:
      output[index] *= 1.0 + strength
    return output
  if kind == "global_offset":
    for index in forward:
      output[index] += 10000.0 * strength
    return output
  if kind == "translation_1px":
    for index in forward:
      output[index] = np.roll(output[index], shift=(0, 1), axis=(1, 2))
    return output
  raise ValueError(kind)


def _maps(result) -> dict[str, np.ndarray]:
  return {
    "temporal_context_ds": result.ds_map,
    "linear_projection_novelty": result.projection_novelty_map,
    "raw_adjacent_rms": result.raw_boundary_difference_map,
  }


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
  rng = np.random.default_rng(args.seed)
  rows: list[dict] = []
  kinds = (
    "local_persistent",
    "local_transient",
    "local_gradual",
    "global_gain",
    "global_offset",
    "translation_1px",
  )
  strengths = (0.1, 0.25, 0.5)
  sizes = (16, 32, 64)
  patch_count = 0
  for entry in load_manifest(args.manifest).get("patches", []):
    ordered = sorted(entry.get("s2", []), key=lambda item: item["date"])
    date_to_index = {item["date"]: index for index, item in enumerate(ordered)}
    if args.target_date not in date_to_index:
      continue
    boundary = date_to_index[args.target_date]
    if boundary < 1 or boundary >= len(ordered):
      continue
    cubes = [_load(args.multisenge_root / item["relpath"]) for item in ordered]
    mask = sequence_common_valid_mask(cubes, nodata_value=0.0)
    clean = bidirectional_temporal_context_boundary(
      cubes,
      mask,
      boundary_index=boundary,
      context_size=args.context_size,
      rank=args.rank,
      factorization=args.factorization,
      preprocessing=args.preprocessing,
    )
    clean_maps = _maps(clean)
    for repeat in range(args.repeats):
      for size in sizes:
        window = _find_window(mask, size, rng)
        if window is None:
          print(f"[SKIP] {entry['patch_id']}: no 90%-valid {size}x{size} window")
          continue
        label = np.zeros(mask.shape, dtype=np.uint8)
        label[window] = 1
        label_values = label[mask]
        for strength in strengths:
          for kind in kinds:
            if kind == "translation_1px" and strength != strengths[0]:
              continue
            modified = _intervene(
              cubes,
              boundary=boundary,
              context_size=args.context_size,
              kind=kind,
              strength=strength,
              window=window,
            )
            result = bidirectional_temporal_context_boundary(
              modified,
              mask,
              boundary_index=boundary,
              context_size=args.context_size,
              rank=args.rank,
              factorization=args.factorization,
              preprocessing=args.preprocessing,
            )
            for method, score in _maps(result).items():
              delta = np.maximum(score - clean_maps[method], 0.0)
              values = delta[mask]
              local = kind.startswith("local_")
              rows.append({
                "patch_id": entry["patch_id"],
                "boundary_date": args.target_date,
                "repeat": repeat,
                "window_size": size,
                "strength": strength,
                "injection": kind,
                "method": method,
                "delta_map_mean": float(np.mean(values)),
                "delta_map_p99": float(np.percentile(values, 99)),
                "delta_map_sum": float(np.sum(values)),
                "localization_auroc": float(roc_auc_score(label_values, values)) if local else float("nan"),
                "localization_ap": float(average_precision_score(label_values, values)) if local else float("nan"),
              })
    patch_count += 1
    print(f"[OK] {entry['patch_id']}: boundary={args.target_date}, valid={int(mask.sum())}")
    if args.max_patches and patch_count >= args.max_patches:
      break
  if not rows:
    raise RuntimeError("No eligible manifest entries were evaluated.")
  _write(args.output_dir / "intervention_scores.csv", rows)

  summary: list[dict] = []
  for injection in kinds:
    for method in METHODS:
      subset = [row for row in rows if row["injection"] == injection and row["method"] == method]
      summary.append({
        "injection": injection,
        "method": method,
        "n": len(subset),
        "mean_delta_map_sum": float(np.mean([row["delta_map_sum"] for row in subset])),
        "mean_localization_auroc": float(np.nanmean([row["localization_auroc"] for row in subset])) if injection.startswith("local_") else float("nan"),
        "mean_localization_ap": float(np.nanmean([row["localization_ap"] for row in subset])) if injection.startswith("local_") else float("nan"),
      })
  _write(args.output_dir / "intervention_summary.csv", summary)
  (args.output_dir / "run_metadata.json").write_text(json.dumps({
    "context_size": args.context_size,
    "rank": args.rank,
    "factorization": args.factorization,
    "preprocessing": args.preprocessing,
    "patch_count": patch_count,
    "interpretation": (
      "Controlled interventions on real backgrounds. Localization is measured against known "
      "injection masks; it is not performance on naturally occurring change labels."
    ),
  }, indent=2), encoding="utf-8")
  print(f"Done. Outputs in: {args.output_dir}")


if __name__ == "__main__":
  main()
