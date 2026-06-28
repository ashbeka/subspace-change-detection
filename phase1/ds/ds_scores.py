"""
Difference Subspace scoring utilities for image-shaped Sentinel-2 tiles.

Source/provenance:
- Linear DS construction is delegated to `phase1.ds.pca_utils`; canonical DS is
  the Fukui/Maki TPAMI 2015 principal-vector formulation.
- Per-pixel projection energy uses the project-specific scoring adaptation
  `||D^T (x_post - x_pre)||^2`, where `D` is the DS basis and `x_*` is the
  selected sample vector.
- Sliding-window scoring is a spatial adaptation introduced for this project to
  test Sensei's concern that global pixel DS breaks spatial information. It is
  not claimed as the original TPAMI DS setting.

Implementation role:
- `compute_ds_scores` is the global pixel construction used by older prior runs.
- New spatial comparisons live in `phase1/scripts/compare_oscd_spatial_subspaces.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from phase1.data.preprocessing import build_valid_mask, devectorize_cube, vectorize_cube
from phase1.ds import pca_utils


Array = np.ndarray


@dataclass
class DSConfig:
    rank_r: int = 6
    variance_threshold: Optional[float] = None
    use_randomized_pca: bool = True
    random_state: int = 1234
    score_normalization: str = "percentile_99"  # or "minmax" / None
    percentile: float = 99.0
    nodata_value: Optional[float] = 0.0
    subspace_variant: str = "residual"  # legacy alias; prefer "canonical" for new DS runs


def _normalize_score(score: Array, method: Optional[str], percentile: float = 99.0) -> Array:
    s = score.astype(np.float32)
    if method is None:
        return s
    if method == "minmax":
        s_min, s_max = float(np.min(s)), float(np.max(s))
        if s_max > s_min:
            return (s - s_min) / (s_max - s_min)
        return np.zeros_like(s)
    if method == "percentile_99" or method == "percentile":
        high = np.percentile(s, percentile)
        if high <= 0:
            return np.zeros_like(s)
        s = np.clip(s, 0, high) / high
        return s
    raise ValueError(f"Unknown normalization method: {method}")


def _compute_ds_matrix_scores(
    x1_mat: Array,
    x2_mat: Array,
    cfg: DSConfig,
    return_components: bool = False,
) -> Dict[str, Array]:
    """Core DS computations on (C, N) matrices."""
    phi = pca_utils.fit_pca_basis(
        x1_mat,
        rank=cfg.rank_r,
        variance_threshold=cfg.variance_threshold,
        random_state=cfg.random_state,
        use_randomized=cfg.use_randomized_pca,
    ).basis
    psi = pca_utils.fit_pca_basis(
        x2_mat,
        rank=cfg.rank_r,
        variance_threshold=cfg.variance_threshold,
        random_state=cfg.random_state,
        use_randomized=cfg.use_randomized_pca,
    ).basis

    resolved_variant = pca_utils.resolve_subspace_variant(cfg.subspace_variant)
    d_basis = pca_utils.build_difference_subspace(phi, psi, variant=resolved_variant)
    diff = x2_mat - x1_mat
    proj_coeff = d_basis.T @ diff
    component_energy = proj_coeff * proj_coeff
    projection_energy = np.sum(component_energy, axis=0)

    r_phi = pca_utils.residual_projector(phi)
    r_psi = pca_utils.residual_projector(psi)
    cross_residual = pca_utils.cross_residual_energy(r_psi, x2_mat) + pca_utils.cross_residual_energy(r_phi, x1_mat)
    res = {
        "projection": projection_energy,
        "cross_residual": cross_residual,
        "subspace_variant": resolved_variant,
        "subspace_dim": int(d_basis.shape[1]),
        "ambient_dim": int(d_basis.shape[0]),
    }
    if return_components:
        res["components"] = component_energy
    return res


def compute_ds_scores(
    x1: Array,
    x2: Array,
    valid_mask: Optional[Array] = None,
    cfg: Optional[DSConfig] = None,
    normalize: bool = True,
    return_components: bool = False,
) -> Dict[str, Array]:
    """
    Compute DS projection and cross-residual maps for a single tile.
    Returns full-resolution arrays with invalid pixels set to 0.
    """
    cfg = cfg or DSConfig()
    if x1.shape != x2.shape:
        raise ValueError(f"Shape mismatch: {x1.shape} vs {x2.shape}")
    if valid_mask is None:
        vm1 = build_valid_mask(x1, nodata_value=cfg.nodata_value)
        vm2 = build_valid_mask(x2, nodata_value=cfg.nodata_value)
        valid_mask = vm1 & vm2

    mat1, idx = vectorize_cube(x1, valid_mask)
    mat2, _ = vectorize_cube(x2, valid_mask)

    if mat1.size == 0:
        raise RuntimeError("No valid pixels available for DS computation.")

    scores = _compute_ds_matrix_scores(mat1, mat2, cfg, return_components=return_components)
    h, w = x1.shape[1:]
    proj_full = devectorize_cube(scores["projection"][None, :], idx, (h, w), fill_value=0.0)[0]
    cross_full = devectorize_cube(scores["cross_residual"][None, :], idx, (h, w), fill_value=0.0)[0]

    if normalize:
        proj_full = _normalize_score(proj_full, cfg.score_normalization, percentile=cfg.percentile)
        cross_full = _normalize_score(cross_full, cfg.score_normalization, percentile=cfg.percentile)

    res = {
        "projection": proj_full,
        "cross_residual": cross_full,
        "valid_mask": valid_mask,
    }

    if return_components:
        r = scores["components"].shape[0]
        comp_full = np.zeros((r, h, w), dtype=np.float32)
        for i in range(r):
            comp_full[i] = devectorize_cube(scores["components"][i:i+1, :], idx, (h, w), fill_value=0.0)[0]
            if normalize:
                comp_full[i] = _normalize_score(comp_full[i], cfg.score_normalization, percentile=cfg.percentile)
        res["components"] = comp_full

    return res


def _window_positions(length: int, window: int, stride: int):
    positions = list(range(0, max(1, length - window + 1), stride))
    last_start = length - window
    if positions:
        if positions[-1] != last_start:
            positions.append(max(0, last_start))
    else:
        positions = [0]
    return positions


def sliding_window_ds(
    x1: Array,
    x2: Array,
    window_size: int = 64,
    stride: int = 32,
    aggregator: str = "mean",
    cfg: Optional[DSConfig] = None,
) -> Dict[str, Array]:
    """
    Compute DS scores over sliding windows and aggregate onto the full tile.
    """
    cfg = cfg or DSConfig()
    h, w = x1.shape[1:]
    if aggregator == "mean":
        acc_proj = np.zeros((h, w), dtype=np.float32)
        acc_cross = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)
    elif aggregator == "max":
        acc_proj = np.full((h, w), -np.inf, dtype=np.float32)
        acc_cross = np.full((h, w), -np.inf, dtype=np.float32)
        counts = None
    else:
        raise ValueError(f"Unknown aggregator: {aggregator}")

    y_positions = _window_positions(h, window_size, stride)
    x_positions = _window_positions(w, window_size, stride)

    for y in y_positions:
        for x in x_positions:
            sl_y = slice(y, y + window_size)
            sl_x = slice(x, x + window_size)
            sub1 = x1[:, sl_y, sl_x]
            sub2 = x2[:, sl_y, sl_x]
            vm1 = build_valid_mask(sub1, nodata_value=cfg.nodata_value)
            vm2 = build_valid_mask(sub2, nodata_value=cfg.nodata_value)
            valid_sub = vm1 & vm2
            sub_scores = compute_ds_scores(sub1, sub2, valid_mask=valid_sub, cfg=cfg, normalize=False)
            if aggregator == "mean":
                acc_proj[sl_y, sl_x] += sub_scores["projection"]
                acc_cross[sl_y, sl_x] += sub_scores["cross_residual"]
                counts[sl_y, sl_x] += 1.0
            else:  # max
                acc_proj[sl_y, sl_x] = np.maximum(acc_proj[sl_y, sl_x], sub_scores["projection"])
                acc_cross[sl_y, sl_x] = np.maximum(acc_cross[sl_y, sl_x], sub_scores["cross_residual"])

    if aggregator == "mean":
        proj_full = np.divide(acc_proj, counts, out=np.zeros_like(acc_proj), where=counts > 0)
        cross_full = np.divide(acc_cross, counts, out=np.zeros_like(acc_cross), where=counts > 0)
        valid_mask = counts > 0
    else:  # max
        proj_full = np.where(np.isfinite(acc_proj), acc_proj, 0.0)
        cross_full = np.where(np.isfinite(acc_cross), acc_cross, 0.0)
        valid_mask = np.isfinite(acc_proj)

    proj_full = _normalize_score(proj_full, cfg.score_normalization, percentile=cfg.percentile)
    cross_full = _normalize_score(cross_full, cfg.score_normalization, percentile=cfg.percentile)

    return {
        "projection": proj_full,
        "cross_residual": cross_full,
        "valid_mask": valid_mask,
    }
