"""Pseudo-invariant-pixel conditioning for nuisance-aware change scores.

Source/provenance:
- Canty and Nielsen's iMAD relative-radiometric-normalization workflow uses
  high iMAD no-change probabilities to identify invariant pixels, then fits a
  symmetric (orthogonal) per-band regression.
- Fukui and Maki's canonical Difference Subspace construction is reused through
  ``phase1.ds.pca_utils``.

Research hypotheses under test (not established methods):
1. A per-band PIF regression can remove pervasive affine radiometric mismatch
   before ordinary change scoring.
2. PCA subspaces fitted only on PIF pixels describe the background relation;
   their Difference Subspace plus the PIF mean shift form a nuisance basis.
   Change energy orthogonal to that basis may reject pseudo-change.

These adaptations must be compared with iMAD, chronochrome, covariance
equalization, raw L2/CVA, and PCA-diff. They are not claimed as novel or valid
until those tests pass.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from phase1.baselines.ir_mad import IRMADResult, ir_mad_result
from phase1.ds import pca_utils

Array = np.ndarray


@dataclass(frozen=True)
class PIFContext:
    ir_mad: IRMADResult
    pif_mask: Array
    selected_pixels: int
    threshold_used: float


@dataclass(frozen=True)
class NuisanceDSResult:
    score: Array
    pif_mask: Array
    nuisance_basis: Array
    pif_pixels: int
    ds_rank: int


def multiscale_spatial_features(
    cube: Array,
    valid_mask: Array,
    sigmas: tuple[float, ...] = (0.0, 1.0, 2.0),
    eps: float = 1e-6,
) -> Array:
    """Stack raw and Gaussian-neighborhood band features without nodata bleed.

    Theiler and Perkins (2006, Section 5) motivate spatial smoothing as a
    spatio-spectral operator that can reduce small co-registration errors and
    suppress anomalies below the kernel's spatial support.  Here each scale
    remains a separate feature block, so downstream ablations can distinguish
    the benefit of spatial support from the benefit of a nuisance subspace.
    """
    if cube.ndim != 3 or valid_mask.shape != cube.shape[1:]:
        raise ValueError("Expected cube=(bands,height,width) and a matching valid mask.")
    if not sigmas:
        raise ValueError("At least one spatial scale is required.")
    mask = valid_mask.astype(np.float32, copy=False)
    blocks: list[Array] = []
    for sigma in sigmas:
        scale = float(sigma)
        if scale < 0.0:
            raise ValueError(f"Gaussian sigma must be non-negative, got {scale}.")
        if scale == 0.0:
            block = cube.astype(np.float32, copy=True)
        else:
            weight = gaussian_filter(mask, sigma=scale, mode="nearest")
            numerator = gaussian_filter(
                cube.astype(np.float32, copy=False) * mask[None, :, :],
                sigma=(0.0, scale, scale),
                mode="nearest",
            )
            block = numerator / np.maximum(weight[None, :, :], float(eps))
        block[:, ~valid_mask] = 0.0
        blocks.append(block.astype(np.float32, copy=False))
    return np.concatenate(blocks, axis=0)


def multiscale_pif_delta_residual_score(
    x1: Array,
    x2: Array,
    valid_mask: Array,
    sigmas: tuple[float, ...] = (0.0, 1.0, 2.0),
    nuisance_rank: int = 1,
    max_pif_fit_pixels: int = 100000,
    random_state: int = 1234,
    context: PIFContext | None = None,
    **context_kwargs,
) -> tuple[Array, PIFContext, Array]:
    """Reject PIF nuisance directions in a multiscale spatial-spectral space.

    This is a project hypothesis, not a named method from Theiler or
    Canty/Nielsen.  PIF selection is performed on the original spectral pair;
    the learned nuisance PCA and residual score operate on concatenated raw and
    Gaussian-neighborhood features.
    """
    if context is None:
        context = build_pif_context(
            x1, x2, valid_mask, random_state=random_state, **context_kwargs
        )
    features1 = multiscale_spatial_features(x1, valid_mask, sigmas=sigmas)
    features2 = multiscale_spatial_features(x2, valid_mask, sigmas=sigmas)
    score, context, nuisance = pif_delta_subspace_residual_score(
        features1,
        features2,
        valid_mask,
        nuisance_rank=nuisance_rank,
        max_pif_fit_pixels=max_pif_fit_pixels,
        random_state=random_state,
        context=context,
    )
    return score, context, nuisance


def build_pif_context(
    x1: Array,
    x2: Array,
    valid_mask: Array,
    pmin: float = 0.9,
    fallback_fraction: float = 0.1,
    minimum_pixels: int = 1000,
    ir_mad_iters: int = 10,
    max_fit_pixels: int = 200000,
    random_state: int = 1234,
) -> PIFContext:
    """Return a deterministic PIF mask from iMAD no-change probabilities."""
    result = ir_mad_result(
        x1,
        x2,
        valid_mask,
        iters=ir_mad_iters,
        downsample_max_pixels=max_fit_pixels,
        random_state=random_state,
    )
    probabilities = result.no_change_probability[valid_mask]
    pif = valid_mask & (result.no_change_probability >= float(pmin))
    threshold_used = float(pmin)
    required = min(max(int(minimum_pixels), 2 * x1.shape[0]), int(valid_mask.sum()))
    if int(pif.sum()) < required:
        keep = min(int(valid_mask.sum()), max(required, int(np.ceil(probabilities.size * fallback_fraction))))
        if keep <= 0:
            raise ValueError("No valid pixels are available for PIF selection.")
        # Select exactly ``keep`` pixels. Thresholding alone can select an
        # entire image when many underflowed p-values tie at zero.
        valid_indices = np.flatnonzero(valid_mask.reshape(-1))
        local_selected = np.argpartition(probabilities, probabilities.size - keep)[-keep:]
        selected_indices = valid_indices[local_selected]
        pif_flat = np.zeros(valid_mask.size, dtype=bool)
        pif_flat[selected_indices] = True
        pif = pif_flat.reshape(valid_mask.shape)
        threshold_used = float(np.min(probabilities[local_selected]))
    return PIFContext(
        ir_mad=result,
        pif_mask=pif,
        selected_pixels=int(pif.sum()),
        threshold_used=threshold_used,
    )


def _orthogonal_line(reference: Array, target: Array, eps: float = 1e-10) -> tuple[float, float]:
    """Fit target ~= slope * reference + intercept by orthogonal regression."""
    x = reference.astype(np.float64, copy=False)
    y = target.astype(np.float64, copy=False)
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    centered_x = x - mean_x
    centered_y = y - mean_y
    sxx = float(np.mean(centered_x * centered_x))
    syy = float(np.mean(centered_y * centered_y))
    sxy = float(np.mean(centered_x * centered_y))
    if abs(sxy) <= eps:
        slope = sxy / max(sxx, eps)
    else:
        slope = (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4.0 * sxy * sxy)) / (2.0 * sxy)
    if not np.isfinite(slope) or abs(slope) <= eps:
        slope = 1.0
    intercept = mean_y - slope * mean_x
    return float(slope), float(intercept)


def pif_radiometric_match(
    x1: Array,
    x2: Array,
    valid_mask: Array,
    context: PIFContext | None = None,
    **context_kwargs,
) -> tuple[Array, PIFContext, Array]:
    """Map the second date to the first date using PIF orthogonal regressions."""
    if context is None:
        context = build_pif_context(x1, x2, valid_mask, **context_kwargs)
    matched = x2.astype(np.float64, copy=True)
    coefficients = np.zeros((x1.shape[0], 2), dtype=np.float64)
    for band in range(x1.shape[0]):
        slope, intercept = _orthogonal_line(x1[band][context.pif_mask], x2[band][context.pif_mask])
        matched[band] = (matched[band] - intercept) / slope
        coefficients[band] = (slope, intercept)
    matched[:, ~valid_mask] = 0.0
    return matched.astype(np.float32), context, coefficients.astype(np.float32)


def pif_nuisance_ds_residual_score(
    x1: Array,
    x2: Array,
    valid_mask: Array,
    rank: int = 6,
    max_pif_fit_pixels: int = 100000,
    random_state: int = 1234,
    context: PIFContext | None = None,
    **context_kwargs,
) -> NuisanceDSResult:
    """Reject PIF-derived mean-shift and DS directions from pixel differences."""
    if context is None:
        context = build_pif_context(
            x1, x2, valid_mask, random_state=random_state, **context_kwargs
        )
    first = x1.reshape(x1.shape[0], -1)[:, context.pif_mask.reshape(-1)]
    second = x2.reshape(x2.shape[0], -1)[:, context.pif_mask.reshape(-1)]
    if first.shape[1] > int(max_pif_fit_pixels):
        rng = np.random.default_rng(int(random_state))
        index = rng.choice(first.shape[1], size=int(max_pif_fit_pixels), replace=False)
        first = first[:, index]
        second = second[:, index]
    effective_rank = max(1, min(int(rank), x1.shape[0] - 1, first.shape[1] - 1))
    pre = pca_utils.fit_pca_basis(
        first, rank=effective_rank, variance_threshold=None, random_state=random_state, use_randomized=False
    )
    post = pca_utils.fit_pca_basis(
        second, rank=effective_rank, variance_threshold=None, random_state=random_state, use_randomized=False
    )
    nuisance_ds = pca_utils.build_difference_subspace(pre.basis, post.basis, variant="canonical")
    mean_shift = np.mean(second - first, axis=1, keepdims=True)
    candidates = np.concatenate([mean_shift, nuisance_ds], axis=1)
    if float(np.linalg.norm(candidates)) > 1e-10:
        q, r = np.linalg.qr(candidates)
        diagonal = np.abs(np.diag(r))
        keep = diagonal > 1e-7
        nuisance_basis = q[:, keep]
    else:
        nuisance_basis = np.zeros((x1.shape[0], 0), dtype=np.float64)

    delta = x2.reshape(x2.shape[0], -1) - x1.reshape(x1.shape[0], -1)
    centered_delta = delta - mean_shift
    if nuisance_basis.shape[1]:
        centered_delta = centered_delta - nuisance_basis @ (nuisance_basis.T @ centered_delta)
    magnitude = np.sqrt(np.maximum(np.sum(centered_delta * centered_delta, axis=0), 0.0))
    score = np.zeros(valid_mask.size, dtype=np.float32)
    valid_flat = valid_mask.reshape(-1)
    score[valid_flat] = magnitude[valid_flat].astype(np.float32, copy=False)
    return NuisanceDSResult(
        score=score.reshape(valid_mask.shape),
        pif_mask=context.pif_mask,
        nuisance_basis=nuisance_basis.astype(np.float32),
        pif_pixels=context.selected_pixels,
        ds_rank=nuisance_ds.shape[1],
    )


def pif_delta_subspace_residual_score(
    x1: Array,
    x2: Array,
    valid_mask: Array,
    nuisance_rank: int = 2,
    max_pif_fit_pixels: int = 100000,
    random_state: int = 1234,
    context: PIFContext | None = None,
    **context_kwargs,
) -> tuple[Array, PIFContext, Array]:
    """Ablation: reject leading PCA directions of PIF difference vectors."""
    if context is None:
        context = build_pif_context(
            x1, x2, valid_mask, random_state=random_state, **context_kwargs
        )
    all_delta = x2.reshape(x2.shape[0], -1) - x1.reshape(x1.shape[0], -1)
    pif_delta = all_delta[:, context.pif_mask.reshape(-1)]
    if pif_delta.shape[1] > int(max_pif_fit_pixels):
        rng = np.random.default_rng(int(random_state))
        index = rng.choice(pif_delta.shape[1], size=int(max_pif_fit_pixels), replace=False)
        pif_delta = pif_delta[:, index]
    mean_shift = np.mean(pif_delta, axis=1, keepdims=True)
    centered_pif = pif_delta - mean_shift
    rank = max(1, min(int(nuisance_rank), x1.shape[0] - 1, centered_pif.shape[1] - 1))
    nuisance = pca_utils.fit_pca_basis(
        centered_pif, rank=rank, variance_threshold=None, random_state=random_state, use_randomized=False
    ).basis
    residual = all_delta - mean_shift
    residual = residual - nuisance @ (nuisance.T @ residual)
    magnitude = np.sqrt(np.maximum(np.sum(residual * residual, axis=0), 0.0))
    score = np.zeros(valid_mask.size, dtype=np.float32)
    valid_flat = valid_mask.reshape(-1)
    score[valid_flat] = magnitude[valid_flat].astype(np.float32, copy=False)
    return score.reshape(valid_mask.shape), context, nuisance.astype(np.float32)
