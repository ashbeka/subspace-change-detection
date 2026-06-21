"""Band-image Difference Subspace and matched spatial controls.

Source/provenance
-----------------
Canonical Difference Subspace construction follows Fukui and Maki, TPAMI
2015, through :func:`phase1.ds.pca_utils.build_difference_subspace`.  The
satellite adaptation treats each complete aligned band image as one sample:
``X_t`` has shape ``N_spatial x B_band_samples``.  This sample definition came
from the project's senpai feedback and is not claimed as part of the TPAMI
paper.

The Gram, projector, and cross-reconstruction maps are project pressure
controls.  They share the exact input matrix, PCA centering axis, and rank with
Band-Image DS so that a DS-specific contribution is not inferred from a
preprocessing mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phase1.ds import pca_utils


Array = np.ndarray


@dataclass(frozen=True)
class BandImageDSResult:
    projected: Array
    projected_energy: Array
    projected_magnitude: Array
    projected_ratio: Array
    residual_energy: Array
    raw_energy: Array
    difference_basis: Array
    pre_basis: Array
    post_basis: Array
    pre_rank: int
    post_rank: int


def band_image_ds_values(
    first: Array,
    second: Array,
    *,
    rank: int,
    seed: int = 1234,
    eps: float = 1e-8,
) -> BandImageDSResult:
    """Return canonical DS evidence for paired ``N_spatial x B`` matrices."""
    if first.ndim != 2 or second.shape != first.shape:
        raise ValueError(
            f"Expected paired matrices shaped N_spatial x B, got {first.shape} and {second.shape}."
        )
    if first.shape[0] < 2 or first.shape[1] < 2:
        raise ValueError(f"Band-image DS requires at least 2x2 data, got {first.shape}.")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Band-image DS input contains non-finite values.")

    effective_rank = max(1, min(int(rank), first.shape[1] - 1))
    pre = pca_utils.fit_pca_basis(
        first,
        rank=effective_rank,
        variance_threshold=None,
        random_state=seed,
        use_randomized=True,
    )
    post = pca_utils.fit_pca_basis(
        second,
        rank=effective_rank,
        variance_threshold=None,
        random_state=seed,
        use_randomized=True,
    )
    difference_basis = pca_utils.build_difference_subspace(
        pre.basis,
        post.basis,
        variant="canonical",
    )

    difference = second.astype(np.float32, copy=False) - first.astype(
        np.float32, copy=False
    )
    if difference_basis.shape[1] == 0:
        projected = np.zeros_like(difference, dtype=np.float32)
    else:
        projected = difference_basis @ (difference_basis.T @ difference)
        projected = projected.astype(np.float32, copy=False)

    projected_energy = np.sum(projected * projected, axis=1).astype(np.float32)
    raw_energy = np.sum(difference * difference, axis=1).astype(np.float32)
    residual = difference - projected
    residual_energy = np.sum(residual * residual, axis=1).astype(np.float32)
    projected_ratio = np.divide(
        projected_energy,
        raw_energy + np.float32(eps),
        out=np.zeros_like(projected_energy),
        where=raw_energy > eps,
    ).astype(np.float32)
    return BandImageDSResult(
        projected=projected,
        projected_energy=projected_energy,
        projected_magnitude=np.sqrt(np.maximum(projected_energy, 0.0)).astype(
            np.float32
        ),
        projected_ratio=projected_ratio,
        residual_energy=residual_energy,
        raw_energy=raw_energy,
        difference_basis=difference_basis,
        pre_basis=pre.basis,
        post_basis=post.basis,
        pre_rank=pre.rank,
        post_rank=post.rank,
    )


def band_image_spatial_control_values(
    first: Array,
    second: Array,
    *,
    rank: int,
    seed: int,
    mode: str,
    eps: float = 1e-12,
    first_basis: Array | None = None,
    second_basis: Array | None = None,
) -> Array:
    """Return a per-position Gram, projector, or reconstruction control."""
    if first.ndim != 2 or second.shape != first.shape:
        raise ValueError(
            f"Expected paired matrices shaped N_spatial x B, got {first.shape} and {second.shape}."
        )
    if first.shape[0] < 2 or first.shape[1] < 2:
        raise ValueError(f"Band-image controls require at least 2x2 data, got {first.shape}.")

    first64 = first.astype(np.float64, copy=False)
    second64 = second.astype(np.float64, copy=False)
    # fit_pca_basis(X) fits sklearn PCA on X.T: the B columns are samples, so
    # centering subtracts the across-band mean at every spatial coordinate.
    first_centered = first64 - np.mean(first64, axis=1, keepdims=True)
    second_centered = second64 - np.mean(second64, axis=1, keepdims=True)

    if mode == "spatial_gram":
        first_scaled = first_centered / max(float(np.linalg.norm(first_centered)), eps)
        second_scaled = second_centered / max(float(np.linalg.norm(second_centered)), eps)
        first_gram_small = first_scaled.T @ first_scaled
        second_gram_small = second_scaled.T @ second_scaled
        cross_gram_small = first_scaled.T @ second_scaled
        first_norm_sq = np.einsum(
            "ni,ij,nj->n", first_scaled, first_gram_small, first_scaled, optimize=True
        )
        second_norm_sq = np.einsum(
            "ni,ij,nj->n", second_scaled, second_gram_small, second_scaled, optimize=True
        )
        cross = np.einsum(
            "ni,ij,nj->n", first_scaled, cross_gram_small, second_scaled, optimize=True
        )
        return np.sqrt(
            np.maximum(first_norm_sq + second_norm_sq - 2.0 * cross, 0.0)
        ).astype(np.float32)

    effective_rank = max(1, min(int(rank), first.shape[1] - 1))
    if first_basis is None:
        first_basis = pca_utils.fit_pca_basis(
            first,
            rank=effective_rank,
            variance_threshold=None,
            random_state=seed,
            use_randomized=True,
        ).basis
    if second_basis is None:
        second_basis = pca_utils.fit_pca_basis(
            second,
            rank=effective_rank,
            variance_threshold=None,
            random_state=seed,
            use_randomized=True,
        ).basis
    if first_basis.shape != (first.shape[0], effective_rank) or second_basis.shape != (
        second.shape[0],
        effective_rank,
    ):
        raise ValueError(
            "Provided Band-Image bases do not match the spatial dimension and effective rank."
        )
    first_basis = first_basis.astype(np.float64, copy=False)
    second_basis = second_basis.astype(np.float64, copy=False)
    first_basis = np.linalg.qr(first_basis, mode="reduced")[0]
    second_basis = np.linalg.qr(second_basis, mode="reduced")[0]

    if mode == "projector_distance":
        cross_basis = first_basis.T @ second_basis
        first_leverage = np.sum(first_basis * first_basis, axis=1)
        second_leverage = np.sum(second_basis * second_basis, axis=1)
        cross = np.einsum(
            "ni,ij,nj->n", first_basis, cross_basis, second_basis, optimize=True
        )
        return np.sqrt(
            np.maximum(first_leverage + second_leverage - 2.0 * cross, 0.0)
        ).astype(np.float32)

    if mode == "cross_reconstruction":
        second_cross = second_centered - first_basis @ (first_basis.T @ second_centered)
        first_cross = first_centered - second_basis @ (second_basis.T @ first_centered)
        second_self = second_centered - second_basis @ (second_basis.T @ second_centered)
        first_self = first_centered - first_basis @ (first_basis.T @ first_centered)
        excess = (
            np.sum(second_cross * second_cross, axis=1)
            - np.sum(second_self * second_self, axis=1)
            + np.sum(first_cross * first_cross, axis=1)
            - np.sum(first_self * first_self, axis=1)
        )
        return np.sqrt(np.maximum(excess, 0.0)).astype(np.float32)

    raise ValueError(f"Unknown band-image spatial control mode: {mode!r}")
