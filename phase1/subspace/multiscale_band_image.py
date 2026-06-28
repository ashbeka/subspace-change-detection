"""Multiscale spatial Band-Image Difference Subspace geometry.

Source/provenance
-----------------
The first-order Difference Subspace (DS) calculation is the canonical
principal-vector construction from Fukui and Maki, TPAMI 2015, exposed by
``phase1.subspace.band_image_geometry``.  The multiscale spatial hierarchy is
a project adaptation motivated by senpai feedback: represent one complete
multispectral acquisition globally, then by corresponding ``2x2``, ``4x4``,
and finer spatial cells.

This module intentionally differs from the earlier fixed-grid pixel-spectrum
pyramid.  In each cell, one flattened band image is one sample, so the paired
matrices have shape ``N_cell_pixels x B_bands`` and the PCA/DS bases live in
spatial coordinates.  Independent cell bases are not added as Euclidean
vectors.  Pixel maps are fused after label-free upper-quantile scaling, while the
collection of cell subspaces is compared as a weighted product of Grassmann
factors using principal-angle distances.

Verification status
-------------------
Formula-level unit tests cover equal-image behavior, principal-angle metrics,
multiscale coverage, deterministic shifted grids, and synthetic localization.
OSCD evaluation is performed by the spatial comparison scripts and does not
turn this project adaptation into a paper-derived method.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import rankdata

from phase1.subspace.band_image_geometry import (
    band_image_ds_values,
    band_image_spatial_control_values,
)


Array = np.ndarray
DEFAULT_OFFSETS = ((0.0, 0.0),)
SHIFTED_OFFSETS = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))
SCORE_MODES = (
    "ds_magnitude",
    "ds_ratio",
    "projector_distance",
    "cross_reconstruction",
    "raw_l2",
    "pca_diff",
)


@dataclass(frozen=True)
class CellGeometry:
    level: int
    offset_y: float
    offset_x: float
    cell_y: int
    cell_x: int
    y0: int
    y1: int
    x0: int
    x1: int
    valid_pixels: int
    effective_rank: int
    geodesic_distance: float
    chordal_distance: float
    maximum_angle: float


@dataclass(frozen=True)
class MultiscaleBandImageResult:
    scale_maps: dict[int, Array]
    fused_map: Array
    product_weighted_map: Array
    product_geodesic_distance: float
    product_chordal_distance: float
    cell_records: tuple[CellGeometry, ...]
    scale_coverage: dict[int, Array]
    score_mode: str
    rank: int
    levels: tuple[int, ...]
    offsets: tuple[tuple[float, float], ...]


def _validate_inputs(first: Array, second: Array, valid_mask: Array) -> None:
    if first.ndim != 3 or second.shape != first.shape:
        raise ValueError(
            "Expected paired cubes shaped bands x height x width, got "
            f"{first.shape} and {second.shape}."
        )
    if valid_mask.shape != first.shape[1:]:
        raise ValueError(
            f"Mask shape {valid_mask.shape} does not match cube {first.shape}."
        )
    if first.shape[0] < 2:
        raise ValueError("At least two band-image samples are required.")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Multiscale Band-Image input contains non-finite values.")


def _shifted_edges(size: int, level: int, offset: float) -> np.ndarray:
    """Return clipped cell boundaries for one deterministic shifted tiling."""
    if size < 1 or level < 1:
        raise ValueError(f"Invalid size/level: size={size}, level={level}.")
    if not 0.0 <= float(offset) < 1.0:
        raise ValueError(f"Grid offset must be in [0,1), got {offset}.")
    cell = float(size) / float(level)
    raw = [int(round((k - float(offset)) * cell)) for k in range(-1, level + 2)]
    clipped = sorted({0, size, *[min(size, max(0, value)) for value in raw]})
    return np.asarray(clipped, dtype=np.int32)


def iter_grid_cells(
    height: int,
    width: int,
    level: int,
    offset: tuple[float, float] = (0.0, 0.0),
) -> Iterable[tuple[int, int, int, int, int, int]]:
    """Yield ``(iy, ix, y0, y1, x0, x1)`` for one complete tiling."""
    y_edges = _shifted_edges(height, level, float(offset[0]))
    x_edges = _shifted_edges(width, level, float(offset[1]))
    for iy, (y0, y1) in enumerate(zip(y_edges[:-1], y_edges[1:])):
        if y1 <= y0:
            continue
        for ix, (x0, x1) in enumerate(zip(x_edges[:-1], x_edges[1:])):
            if x1 <= x0:
                continue
            yield iy, ix, int(y0), int(y1), int(x0), int(x1)


def principal_angle_distances(first_basis: Array, second_basis: Array) -> tuple[Array, float, float]:
    """Return principal angles, geodesic distance, and chordal distance."""
    if first_basis.ndim != 2 or second_basis.ndim != 2:
        raise ValueError("Expected two basis matrices.")
    if first_basis.shape[0] != second_basis.shape[0]:
        raise ValueError("Grassmann factors must share their ambient dimension.")
    if first_basis.shape[1] == 0 or second_basis.shape[1] == 0:
        return np.zeros((0,), dtype=np.float32), 0.0, 0.0
    first_q = np.linalg.qr(first_basis.astype(np.float64, copy=False), mode="reduced")[0]
    second_q = np.linalg.qr(second_basis.astype(np.float64, copy=False), mode="reduced")[0]
    singular = np.linalg.svd(first_q.T @ second_q, compute_uv=False)
    singular = np.clip(singular, 0.0, 1.0)
    angles = np.arccos(singular)
    geodesic = float(np.linalg.norm(angles))
    chordal = float(np.linalg.norm(np.sin(angles)))
    return angles.astype(np.float32), geodesic, chordal


def rank_normalize_map(score: Array, valid_mask: Array) -> Array:
    """Map valid scores to empirical fractional ranks without labels."""
    out = np.zeros(score.shape, dtype=np.float32)
    values = np.asarray(score[valid_mask], dtype=np.float64)
    if values.size == 0:
        return out
    if not np.all(np.isfinite(values)):
        raise ValueError("Cannot rank-normalize non-finite scores.")
    if np.allclose(values, values[0]):
        return out
    ranks = rankdata(values, method="average")
    out[valid_mask] = ((ranks - 1.0) / max(1.0, float(values.size - 1))).astype(np.float32)
    return out


def quantile_scale_map(
    score: Array,
    valid_mask: Array,
    *,
    percentile: float = 99.5,
) -> Array:
    """Scale a nonnegative map by an unlabeled upper quantile and clip to one."""
    out = np.zeros(score.shape, dtype=np.float32)
    values = np.asarray(score[valid_mask], dtype=np.float64)
    if values.size == 0:
        return out
    if not np.all(np.isfinite(values)):
        raise ValueError("Cannot quantile-scale non-finite scores.")
    high = float(np.percentile(values, float(percentile)))
    if high <= 0.0:
        high = float(np.max(values))
    if high <= 0.0:
        return out
    out[valid_mask] = np.clip(score[valid_mask] / high, 0.0, 1.0).astype(np.float32)
    return out


def score_band_image_matrix(
    first: Array,
    second: Array,
    *,
    rank: int,
    seed: int,
    score_mode: str,
    basis_mode: str,
) -> tuple[Array, Array, Array, int, float, float, float]:
    if score_mode not in SCORE_MODES:
        raise ValueError(f"Unknown score_mode={score_mode!r}; expected {SCORE_MODES}.")
    if score_mode in {"raw_l2", "pca_diff"}:
        difference = second.astype(np.float32, copy=False) - first.astype(
            np.float32, copy=False
        )
        if score_mode == "raw_l2":
            score = np.linalg.norm(difference, axis=1).astype(np.float32)
        else:
            from phase1.ds import pca_utils

            effective_rank = max(1, min(int(rank), difference.shape[1]))
            basis = pca_utils.fit_pca_basis(
                difference.T,
                rank=effective_rank,
                variance_threshold=None,
                random_state=seed,
                use_randomized=True,
            ).basis
            coefficients = basis.T @ difference.T
            score = np.linalg.norm(coefficients, axis=0).astype(np.float32)
        return score, np.zeros((first.shape[0], 0), np.float32), np.zeros(
            (first.shape[0], 0), np.float32
        ), 0, 0.0, 0.0, 0.0

    ds = band_image_ds_values(
        first,
        second,
        rank=rank,
        seed=seed,
        basis_mode=basis_mode,
    )
    angles, geodesic, chordal = principal_angle_distances(ds.pre_basis, ds.post_basis)
    if score_mode == "ds_magnitude":
        score = ds.projected_magnitude
    elif score_mode == "ds_ratio":
        score = ds.projected_ratio
    elif score_mode == "projector_distance":
        score = band_image_spatial_control_values(
            first,
            second,
            rank=rank,
            seed=seed,
            mode="projector_distance",
            basis_mode=basis_mode,
            first_basis=ds.pre_basis,
            second_basis=ds.post_basis,
        )
    elif score_mode == "cross_reconstruction":
        score = band_image_spatial_control_values(
            first,
            second,
            rank=rank,
            seed=seed,
            mode="cross_reconstruction",
            basis_mode=basis_mode,
            first_basis=ds.pre_basis,
            second_basis=ds.post_basis,
        )
    else:  # pragma: no cover - guarded above
        raise AssertionError(score_mode)
    return (
        score.astype(np.float32, copy=False),
        ds.pre_basis,
        ds.post_basis,
        int(min(ds.pre_rank, ds.post_rank)),
        geodesic,
        chordal,
        float(np.max(angles)) if angles.size else 0.0,
    )


def multiscale_band_image_geometry(
    first: Array,
    second: Array,
    valid_mask: Array,
    *,
    rank: int,
    levels: Sequence[int] = (1, 2, 4),
    offsets: Sequence[tuple[float, float]] = DEFAULT_OFFSETS,
    seed: int = 1234,
    score_mode: str = "ds_magnitude",
    basis_mode: str = "centered_pca",
    min_valid_pixels: int = 32,
) -> MultiscaleBandImageResult:
    """Evaluate corresponding spatial Band-Image factors across a hierarchy."""
    _validate_inputs(first, second, valid_mask)
    normalized_levels = tuple(sorted({int(level) for level in levels}))
    normalized_offsets = tuple((float(y), float(x)) for y, x in offsets)
    if not normalized_levels or any(level < 1 for level in normalized_levels):
        raise ValueError(f"Invalid pyramid levels: {levels}.")
    if not normalized_offsets:
        raise ValueError("At least one grid offset is required.")
    height, width = valid_mask.shape
    scale_maps: dict[int, Array] = {}
    scale_coverage: dict[int, Array] = {}
    product_scale_maps: dict[int, Array] = {}
    records: list[CellGeometry] = []
    scale_geodesic_means: list[float] = []
    scale_chordal_means: list[float] = []

    for level in normalized_levels:
        score_sum = np.zeros((height, width), dtype=np.float64)
        product_sum = np.zeros((height, width), dtype=np.float64)
        coverage = np.zeros((height, width), dtype=np.float32)
        level_cell_payloads: list[tuple[tuple[slice, slice], Array, Array, float]] = []
        level_geodesics: list[float] = []
        level_chordals: list[float] = []

        active_offsets = normalized_offsets if level > 1 else ((0.0, 0.0),)
        for offset_y, offset_x in active_offsets:
            for iy, ix, y0, y1, x0, x1 in iter_grid_cells(
                height,
                width,
                level,
                (offset_y, offset_x),
            ):
                local_mask = valid_mask[y0:y1, x0:x1]
                count = int(np.sum(local_mask))
                if count < max(int(min_valid_pixels), 2):
                    continue
                first_matrix = first[:, y0:y1, x0:x1][:, local_mask].T
                second_matrix = second[:, y0:y1, x0:x1][:, local_mask].T
                (
                    local_values,
                    _first_basis,
                    _second_basis,
                    effective_rank,
                    geodesic,
                    chordal,
                    maximum_angle,
                ) = score_band_image_matrix(
                    first_matrix,
                    second_matrix,
                    rank=rank,
                    seed=seed,
                    score_mode=score_mode,
                    basis_mode=basis_mode,
                )
                local_score = np.zeros(local_mask.shape, dtype=np.float32)
                local_score[local_mask] = local_values
                slices = (slice(y0, y1), slice(x0, x1))
                level_cell_payloads.append((slices, local_mask, local_score, geodesic))
                level_geodesics.append(geodesic)
                level_chordals.append(chordal)
                records.append(
                    CellGeometry(
                        level=level,
                        offset_y=offset_y,
                        offset_x=offset_x,
                        cell_y=iy,
                        cell_x=ix,
                        y0=y0,
                        y1=y1,
                        x0=x0,
                        x1=x1,
                        valid_pixels=count,
                        effective_rank=effective_rank,
                        geodesic_distance=geodesic,
                        chordal_distance=chordal,
                        maximum_angle=maximum_angle,
                    )
                )

        positive_distances = np.asarray(
            [value for value in level_geodesics if value > 0.0], dtype=np.float64
        )
        distance_reference = (
            float(np.median(positive_distances)) if positive_distances.size else 1.0
        )
        for slices, local_mask, local_score, geodesic in level_cell_payloads:
            score_view = score_sum[slices]
            product_view = product_sum[slices]
            coverage_view = coverage[slices]
            score_view[local_mask] += local_score[local_mask]
            geometry_weight = np.clip(geodesic / max(distance_reference, 1e-12), 0.0, 4.0)
            product_view[local_mask] += geometry_weight * local_score[local_mask]
            coverage_view[local_mask] += 1.0

        scale_map = np.zeros((height, width), dtype=np.float32)
        product_map = np.zeros((height, width), dtype=np.float32)
        covered = coverage > 0.0
        scale_map[covered] = (score_sum[covered] / coverage[covered]).astype(np.float32)
        product_map[covered] = (product_sum[covered] / coverage[covered]).astype(
            np.float32
        )
        scale_map[~valid_mask] = 0.0
        product_map[~valid_mask] = 0.0
        scale_maps[level] = scale_map
        product_scale_maps[level] = product_map
        scale_coverage[level] = coverage
        if level_geodesics:
            scale_geodesic_means.append(float(np.mean(np.square(level_geodesics))))
            scale_chordal_means.append(float(np.mean(np.square(level_chordals))))

    fused_components = [quantile_scale_map(scale_maps[level], valid_mask) for level in normalized_levels]
    product_components = [
        quantile_scale_map(product_scale_maps[level], valid_mask) for level in normalized_levels
    ]
    fused = np.mean(np.stack(fused_components, axis=0), axis=0).astype(np.float32)
    product_weighted = np.mean(np.stack(product_components, axis=0), axis=0).astype(
        np.float32
    )
    fused[~valid_mask] = 0.0
    product_weighted[~valid_mask] = 0.0
    product_geodesic = float(np.sqrt(np.mean(scale_geodesic_means))) if scale_geodesic_means else 0.0
    product_chordal = float(np.sqrt(np.mean(scale_chordal_means))) if scale_chordal_means else 0.0
    return MultiscaleBandImageResult(
        scale_maps=scale_maps,
        fused_map=fused,
        product_weighted_map=product_weighted,
        product_geodesic_distance=product_geodesic,
        product_chordal_distance=product_chordal,
        cell_records=tuple(records),
        scale_coverage=scale_coverage,
        score_mode=score_mode,
        rank=int(rank),
        levels=normalized_levels,
        offsets=normalized_offsets,
    )
