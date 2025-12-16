"""
Geodesic / temporal priors based on local subspace geometry.

This module produces *spatial maps* by computing PCA subspaces over local windows
and then measuring Grassmann geometry between those local subspaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from phase1.data.preprocessing import build_valid_mask, vectorize_cube
from phase1.ds.pca_utils import fit_pca_basis
from phase1.subspace.geodesic import grassmann_geodesic_distance, subspace_magnitude
from phase1.subspace.second_order_ds import second_order_difference_subspace

Array = np.ndarray


@dataclass
class GeodesicPriorConfig:
  rank_r: int = 6
  variance_threshold: Optional[float] = None
  use_randomized_pca: bool = True
  random_state: int = 1234
  seed: int = 1234
  pixel_sample: int = 0  # 0 means "use all valid pixels"
  nodata_value: float = 0.0
  min_valid_bands: int = 3
  score_normalization: Optional[str] = "percentile_99"  # "minmax" | "percentile_99" | None
  percentile: float = 99.0


def _window_positions(length: int, window: int, stride: int):
  positions = list(range(0, max(1, length - window + 1), stride))
  last_start = length - window
  if positions:
    if positions[-1] != last_start:
      positions.append(max(0, last_start))
  else:
    positions = [0]
  return positions


def _normalize_score(score: Array, method: Optional[str], percentile: float = 99.0) -> Array:
  s = score.astype(np.float32, copy=False)
  if method is None:
    return s
  if method == "minmax":
    s_min, s_max = float(np.min(s)), float(np.max(s))
    if s_max > s_min:
      return (s - s_min) / (s_max - s_min)
    return np.zeros_like(s)
  if method in ("percentile_99", "percentile"):
    high = float(np.percentile(s, float(percentile)))
    if high <= 0:
      return np.zeros_like(s)
    return (np.clip(s, 0, high) / high).astype(np.float32, copy=False)
  raise ValueError(f"Unknown normalization method: {method}")


def _fit_window_subspace(mat: Array, cfg: GeodesicPriorConfig, rng: np.random.Generator) -> Array:
  if mat.size == 0:
    return np.zeros((mat.shape[0], 0), dtype=np.float32)
  if cfg.pixel_sample and mat.shape[1] > int(cfg.pixel_sample):
    idx = rng.choice(mat.shape[1], size=int(cfg.pixel_sample), replace=False)
    mat = mat[:, idx]
  return fit_pca_basis(
    mat,
    rank=int(cfg.rank_r),
    variance_threshold=cfg.variance_threshold,
    random_state=int(cfg.random_state),
    use_randomized=bool(cfg.use_randomized_pca),
  ).basis.astype(np.float32, copy=False)


def sliding_window_geodesic_prior(
  x1: Array,
  x2: Array,
  *,
  window_size: int = 64,
  stride: int = 32,
  aggregator: str = "mean",
  metric: str = "geodesic_dist",  # "geodesic_dist" | "magnitude"
  cfg: Optional[GeodesicPriorConfig] = None,
) -> Dict[str, Array]:
  """
  Compute a per-pixel prior map based on local Grassmann geometry between
  PCA subspaces computed from local windows at t1 and t2.

  Returns:
    - score: (H,W) float32
    - valid_mask: (H,W) bool where at least one window contributed
  """
  cfg = cfg or GeodesicPriorConfig()
  if x1.shape != x2.shape:
    raise ValueError(f"Shape mismatch: {x1.shape} vs {x2.shape}")
  h, w = x1.shape[1:]

  if aggregator == "mean":
    acc = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
  elif aggregator == "max":
    acc = np.full((h, w), -np.inf, dtype=np.float32)
    counts = None
  else:
    raise ValueError(f"Unknown aggregator: {aggregator}")

  rng = np.random.default_rng(int(cfg.seed))
  ys = _window_positions(h, int(window_size), int(stride))
  xs = _window_positions(w, int(window_size), int(stride))

  for y0 in ys:
    for x0 in xs:
      sl_y = slice(y0, y0 + int(window_size))
      sl_x = slice(x0, x0 + int(window_size))
      sub1 = x1[:, sl_y, sl_x]
      sub2 = x2[:, sl_y, sl_x]
      vm1 = build_valid_mask(sub1, nodata_value=cfg.nodata_value, min_valid_bands=cfg.min_valid_bands)
      vm2 = build_valid_mask(sub2, nodata_value=cfg.nodata_value, min_valid_bands=cfg.min_valid_bands)
      vm = vm1 & vm2
      mat1, _ = vectorize_cube(sub1, vm)
      mat2, _ = vectorize_cube(sub2, vm)
      if mat1.size == 0 or mat2.size == 0:
        score = 0.0
      else:
        s1 = _fit_window_subspace(mat1, cfg, rng)
        s2 = _fit_window_subspace(mat2, cfg, rng)
        if metric == "geodesic_dist":
          score = grassmann_geodesic_distance(s1, s2)
        elif metric == "magnitude":
          score = subspace_magnitude(s1, s2)
        else:
          raise ValueError(f"Unknown metric: {metric}")

      if aggregator == "mean":
        acc[sl_y, sl_x] += float(score)
        counts[sl_y, sl_x] += 1.0
      else:
        acc[sl_y, sl_x] = np.maximum(acc[sl_y, sl_x], float(score))

  if aggregator == "mean":
    score_full = np.divide(acc, counts, out=np.zeros_like(acc), where=counts > 0)
    valid_mask = counts > 0
  else:
    score_full = np.where(np.isfinite(acc), acc, 0.0).astype(np.float32, copy=False)
    valid_mask = np.isfinite(acc)

  score_full = _normalize_score(score_full, cfg.score_normalization, percentile=cfg.percentile)
  return {"score": score_full, "valid_mask": valid_mask}


def sliding_window_second_order_prior(
  x1: Array,
  x2: Array,
  x3: Array,
  *,
  window_size: int = 64,
  stride: int = 32,
  aggregator: str = "mean",
  cfg: Optional[GeodesicPriorConfig] = None,
) -> Dict[str, Array]:
  """
  Compute a per-pixel second-order DS magnitude prior map on triples (t-1, t, t+1).
  """
  cfg = cfg or GeodesicPriorConfig()
  if x1.shape != x2.shape or x2.shape != x3.shape:
    raise ValueError(f"Shape mismatch: {x1.shape}, {x2.shape}, {x3.shape}")
  h, w = x1.shape[1:]

  if aggregator == "mean":
    acc_total = np.zeros((h, w), dtype=np.float32)
    acc_along = np.zeros((h, w), dtype=np.float32)
    acc_orth = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
  elif aggregator == "max":
    acc_total = np.full((h, w), -np.inf, dtype=np.float32)
    acc_along = np.full((h, w), -np.inf, dtype=np.float32)
    acc_orth = np.full((h, w), -np.inf, dtype=np.float32)
    counts = None
  else:
    raise ValueError(f"Unknown aggregator: {aggregator}")

  rng = np.random.default_rng(int(cfg.seed))
  ys = _window_positions(h, int(window_size), int(stride))
  xs = _window_positions(w, int(window_size), int(stride))

  for y0 in ys:
    for x0 in xs:
      sl_y = slice(y0, y0 + int(window_size))
      sl_x = slice(x0, x0 + int(window_size))
      sub1 = x1[:, sl_y, sl_x]
      sub2 = x2[:, sl_y, sl_x]
      sub3 = x3[:, sl_y, sl_x]
      vm1 = build_valid_mask(sub1, nodata_value=cfg.nodata_value, min_valid_bands=cfg.min_valid_bands)
      vm2 = build_valid_mask(sub2, nodata_value=cfg.nodata_value, min_valid_bands=cfg.min_valid_bands)
      vm3 = build_valid_mask(sub3, nodata_value=cfg.nodata_value, min_valid_bands=cfg.min_valid_bands)
      vm = vm1 & vm2 & vm3
      mat1, _ = vectorize_cube(sub1, vm)
      mat2, _ = vectorize_cube(sub2, vm)
      mat3, _ = vectorize_cube(sub3, vm)
      if mat1.size == 0 or mat2.size == 0 or mat3.size == 0:
        mag_total = 0.0
        mag_along = 0.0
        mag_orth = 0.0
      else:
        s1 = _fit_window_subspace(mat1, cfg, rng)
        s2 = _fit_window_subspace(mat2, cfg, rng)
        s3 = _fit_window_subspace(mat3, cfg, rng)
        res = second_order_difference_subspace(s1, s2, s3, decompose=True)
        mag_total = float(res.mag_total)
        mag_along = float(res.mag_along or 0.0)
        mag_orth = float(res.mag_orth or 0.0)

      if aggregator == "mean":
        acc_total[sl_y, sl_x] += mag_total
        acc_along[sl_y, sl_x] += mag_along
        acc_orth[sl_y, sl_x] += mag_orth
        counts[sl_y, sl_x] += 1.0
      else:
        acc_total[sl_y, sl_x] = np.maximum(acc_total[sl_y, sl_x], mag_total)
        acc_along[sl_y, sl_x] = np.maximum(acc_along[sl_y, sl_x], mag_along)
        acc_orth[sl_y, sl_x] = np.maximum(acc_orth[sl_y, sl_x], mag_orth)

  if aggregator == "mean":
    score_total = np.divide(acc_total, counts, out=np.zeros_like(acc_total), where=counts > 0)
    score_along = np.divide(acc_along, counts, out=np.zeros_like(acc_along), where=counts > 0)
    score_orth = np.divide(acc_orth, counts, out=np.zeros_like(acc_orth), where=counts > 0)
    valid_mask = counts > 0
  else:
    score_total = np.where(np.isfinite(acc_total), acc_total, 0.0).astype(np.float32, copy=False)
    score_along = np.where(np.isfinite(acc_along), acc_along, 0.0).astype(np.float32, copy=False)
    score_orth = np.where(np.isfinite(acc_orth), acc_orth, 0.0).astype(np.float32, copy=False)
    valid_mask = np.isfinite(acc_total)

  score_total = _normalize_score(score_total, cfg.score_normalization, percentile=cfg.percentile)
  score_along = _normalize_score(score_along, cfg.score_normalization, percentile=cfg.percentile)
  score_orth = _normalize_score(score_orth, cfg.score_normalization, percentile=cfg.percentile)
  return {"score_total": score_total, "score_along": score_along, "score_orth": score_orth, "valid_mask": valid_mask}

