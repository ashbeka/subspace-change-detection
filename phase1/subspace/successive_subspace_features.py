"""Successive local Saab/PCA features for spatial subspace change analysis.

Source/provenance
-----------------
PixelHop (Kuo et al., JVCIR 2020) defines successive subspace learning through
near-to-far neighborhood expansion, unsupervised Saab subspace approximation,
and pooling between hops.  Saab separates a normalized constant DC direction
from PCA-derived AC directions.  PixelHop's later label-assisted regression
and classifier are not applicable to this unsupervised change-map experiment.

This module implements a deliberately scoped, source-labeled adaptation:

* one shared DC/AC transform is fit jointly to paired pre/post neighborhoods;
* the same transform is applied to both dates, preventing incomparable feature
  rotations;
* ``3x3`` neighborhood expansion and ``2x2`` max pooling are repeated;
* hop response maps become Band-Image samples and are compared by canonical
  DS or matched geometric controls.

The adjusted Saab bias is omitted because a common per-channel constant
cancels in paired differences and does not alter within-channel max-pooling
locations.  Consequently, this is reported as ``successive Saab-DS`` or a
PixelHop-inspired unsupervised representation, not a full PixelHop classifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import zoom
import torch
import torch.nn.functional as torch_f

from phase1.subspace.multiscale_band_image import (
    principal_angle_distances,
    quantile_scale_map,
    score_band_image_matrix,
)


Array = np.ndarray


@dataclass(frozen=True)
class SaabStage:
    kernels: Array  # output_channels x input_channels x patch x patch
    intercept: Array  # output_channels
    patch_size: int
    pool_size: int
    input_channels: int
    output_channels: int
    ac_explained_variance_ratio: Array


@dataclass(frozen=True)
class SuccessiveSaabResult:
    hop_maps: dict[int, Array]
    fused_map: Array
    product_weighted_map: Array
    hop_geodesic: dict[int, float]
    pre_features: tuple[Array, ...]
    post_features: tuple[Array, ...]
    feature_masks: tuple[Array, ...]
    stages: tuple[SaabStage, ...]
    score_mode: str
    rank: int


def _sample_patch_matrix(
    first: Array,
    second: Array,
    valid_mask: Array,
    *,
    patch_size: int,
    max_samples: int,
    rng: np.random.Generator,
) -> Array:
    margin = patch_size // 2
    if patch_size % 2 != 1 or patch_size < 1:
        raise ValueError("Successive Saab patch_size must be a positive odd integer.")
    valid = valid_mask.copy()
    if margin:
        valid[:margin] = False
        valid[-margin:] = False
        valid[:, :margin] = False
        valid[:, -margin:] = False
    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        raise ValueError("No valid neighborhood centers are available.")
    per_date = max(1, int(max_samples) // 2)
    if ys.size > per_date:
        chosen = rng.choice(ys.size, size=per_date, replace=False)
        ys = ys[chosen]
        xs = xs[chosen]

    def gather(cube: Array) -> Array:
        padded = np.pad(
            cube,
            ((0, 0), (margin, margin), (margin, margin)),
            mode="reflect",
        )
        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            (patch_size, patch_size),
            axis=(1, 2),
        )
        return windows[:, ys, xs].transpose(1, 0, 2, 3).reshape(ys.size, -1)

    return np.concatenate([gather(first), gather(second)], axis=0).astype(
        np.float32,
        copy=False,
    )


def fit_saab_stage(
    samples: Array,
    *,
    input_channels: int,
    patch_size: int,
    pool_size: int,
    energy_threshold: float,
    max_output_channels: int,
) -> SaabStage:
    """Fit one shared DC plus covariance-eigenvector AC Saab stage."""
    expected_dimension = int(input_channels) * int(patch_size) ** 2
    if samples.ndim != 2 or samples.shape[1] != expected_dimension:
        raise ValueError(
            f"Expected sample matrix (*,{expected_dimension}), got {samples.shape}."
        )
    if samples.shape[0] < 2:
        raise ValueError("At least two neighborhoods are required to fit Saab.")
    if not 0.0 < float(energy_threshold) <= 1.0:
        raise ValueError("energy_threshold must lie in (0,1].")
    if int(max_output_channels) < 1:
        raise ValueError("max_output_channels must be positive.")

    values = samples.astype(np.float64, copy=False)
    dimension = values.shape[1]
    dc_kernel = np.ones(dimension, dtype=np.float64) / np.sqrt(float(dimension))
    dc_coefficients = values @ dc_kernel
    ac = values - dc_coefficients[:, None] * dc_kernel[None, :]
    ac_mean = np.mean(ac, axis=0)
    ac_centered = ac - ac_mean[None, :]
    covariance = (ac_centered.T @ ac_centered) / float(max(1, ac_centered.shape[0] - 1))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    # Numerical covariance noise can reintroduce a tiny DC component.
    eigenvectors -= dc_kernel[:, None] * (dc_kernel @ eigenvectors)[None, :]
    eigenvectors, _ = np.linalg.qr(eigenvectors, mode="reduced")
    total = float(np.sum(eigenvalues))
    ratios = eigenvalues / total if total > 0.0 else np.zeros_like(eigenvalues)
    if total > 0.0:
        ac_count = int(np.searchsorted(np.cumsum(ratios), energy_threshold) + 1)
    else:
        ac_count = 0
    ac_count = max(0, min(ac_count, int(max_output_channels) - 1, dimension - 1))
    ac_kernels = eigenvectors[:, :ac_count]
    flat_kernels = np.concatenate([dc_kernel[:, None], ac_kernels], axis=1).T
    intercept = np.zeros(flat_kernels.shape[0], dtype=np.float64)
    if ac_count:
        intercept[1:] = -(ac_mean @ ac_kernels)
    kernels = flat_kernels.reshape(
        flat_kernels.shape[0],
        int(input_channels),
        int(patch_size),
        int(patch_size),
    )
    return SaabStage(
        kernels=kernels.astype(np.float32),
        intercept=intercept.astype(np.float32),
        patch_size=int(patch_size),
        pool_size=int(pool_size),
        input_channels=int(input_channels),
        output_channels=int(kernels.shape[0]),
        ac_explained_variance_ratio=ratios[:ac_count].astype(np.float32),
    )


def apply_saab_stage(
    cube: Array,
    stage: SaabStage,
    *,
    device: str = "cpu",
) -> Array:
    """Apply one shared stage with same-size reflect-padded convolution."""
    if cube.ndim != 3 or cube.shape[0] != stage.input_channels:
        raise ValueError(
            f"Expected {stage.input_channels}xHxW input, got {cube.shape}."
        )
    tensor = torch.from_numpy(cube.astype(np.float32, copy=False))[None].to(device)
    kernels = torch.from_numpy(stage.kernels).to(device)
    intercept = torch.from_numpy(stage.intercept).to(device)
    margin = stage.patch_size // 2
    if margin:
        tensor = torch_f.pad(tensor, (margin, margin, margin, margin), mode="reflect")
    with torch.no_grad():
        response = torch_f.conv2d(tensor, kernels, bias=intercept)
    return response[0].cpu().numpy().astype(np.float32, copy=False)


def _pool_features(features: Array, valid_mask: Array, pool_size: int) -> tuple[Array, Array]:
    if int(pool_size) <= 1:
        return features, valid_mask.copy()
    tensor = torch.from_numpy(features.astype(np.float32, copy=False))[None]
    pooled = torch_f.max_pool2d(tensor, kernel_size=pool_size, stride=pool_size)
    mask_tensor = torch.from_numpy(valid_mask.astype(np.float32))[None, None]
    valid_fraction = torch_f.avg_pool2d(
        mask_tensor,
        kernel_size=pool_size,
        stride=pool_size,
    )
    pooled_mask = valid_fraction[0, 0].numpy() >= 1.0 - 1e-6
    return pooled[0].numpy().astype(np.float32), pooled_mask


def _resize_score(score: Array, target_shape: tuple[int, int]) -> Array:
    factors = (target_shape[0] / score.shape[0], target_shape[1] / score.shape[1])
    resized = zoom(score, factors, order=1, mode="nearest", prefilter=False)
    return resized[: target_shape[0], : target_shape[1]].astype(np.float32, copy=False)


def successive_saab_band_image_geometry(
    first: Array,
    second: Array,
    valid_mask: Array,
    *,
    rank: int,
    hops: int = 2,
    patch_size: int = 3,
    pool_size: int = 2,
    energy_threshold: float = 0.95,
    max_output_channels: Sequence[int] = (16, 16),
    max_fit_samples: int = 30000,
    seed: int = 1234,
    score_mode: str = "ds_magnitude",
    basis_mode: str = "centered_pca",
    device: str = "cpu",
) -> SuccessiveSaabResult:
    """Fit a shared pairwise SSL hierarchy and compare each hop spatially."""
    if first.ndim != 3 or second.shape != first.shape:
        raise ValueError("Expected paired channels x height x width feature cubes.")
    if valid_mask.shape != first.shape[1:]:
        raise ValueError("Successive Saab mask does not match the input cube.")
    if int(hops) < 1:
        raise ValueError("At least one successive hop is required.")
    channel_limits = list(int(value) for value in max_output_channels)
    if len(channel_limits) < int(hops):
        channel_limits.extend([channel_limits[-1]] * (int(hops) - len(channel_limits)))
    rng = np.random.default_rng(int(seed))
    current_first = first.astype(np.float32, copy=False)
    current_second = second.astype(np.float32, copy=False)
    current_mask = valid_mask.astype(bool, copy=True)
    stages: list[SaabStage] = []
    pre_features: list[Array] = []
    post_features: list[Array] = []
    feature_masks: list[Array] = []

    for hop_index in range(int(hops)):
        samples = _sample_patch_matrix(
            current_first,
            current_second,
            current_mask,
            patch_size=patch_size,
            max_samples=max_fit_samples,
            rng=rng,
        )
        stage = fit_saab_stage(
            samples,
            input_channels=current_first.shape[0],
            patch_size=patch_size,
            pool_size=pool_size,
            energy_threshold=energy_threshold,
            max_output_channels=channel_limits[hop_index],
        )
        transformed_first = apply_saab_stage(current_first, stage, device=device)
        transformed_second = apply_saab_stage(current_second, stage, device=device)
        stages.append(stage)
        pre_features.append(transformed_first)
        post_features.append(transformed_second)
        feature_masks.append(current_mask.copy())
        if hop_index + 1 < int(hops):
            current_first, next_mask = _pool_features(
                transformed_first,
                current_mask,
                pool_size,
            )
            current_second, second_mask = _pool_features(
                transformed_second,
                current_mask,
                pool_size,
            )
            current_mask = next_mask & second_mask

    hop_maps: dict[int, Array] = {}
    hop_geodesic: dict[int, float] = {}
    positive_distances: list[float] = []
    original_shape = valid_mask.shape
    temporary: list[tuple[int, Array, float]] = []
    for hop_index, (pre, post, mask) in enumerate(
        zip(pre_features, post_features, feature_masks),
        start=1,
    ):
        first_matrix = pre[:, mask].T
        second_matrix = post[:, mask].T
        score, pre_basis, post_basis, _effective_rank, geodesic, _chordal, _maximum = (
            score_band_image_matrix(
                first_matrix,
                second_matrix,
                rank=min(int(rank), pre.shape[0] - 1),
                seed=seed,
                score_mode=score_mode,
                basis_mode=basis_mode,
            )
        )
        # Cross-check the helper's distance against the returned bases.
        _angles, direct_geodesic, _direct_chordal = principal_angle_distances(
            pre_basis,
            post_basis,
        )
        if not np.isclose(geodesic, direct_geodesic, atol=1e-5):
            raise AssertionError("Inconsistent hop principal-angle distance.")
        local = np.zeros(mask.shape, dtype=np.float32)
        local[mask] = score
        resized = _resize_score(local, original_shape)
        resized[~valid_mask] = 0.0
        hop_maps[hop_index] = resized
        hop_geodesic[hop_index] = geodesic
        temporary.append((hop_index, resized, geodesic))
        if geodesic > 0.0:
            positive_distances.append(geodesic)

    reference = float(np.median(positive_distances)) if positive_distances else 1.0
    product_components: list[Array] = []
    product_weights: list[float] = []
    for _hop_index, score, geodesic in temporary:
        product_components.append(quantile_scale_map(score, valid_mask))
        product_weights.append(
            float(np.clip(geodesic / max(reference, 1e-12), 0.0, 4.0))
        )
    fused = np.mean(
        np.stack([quantile_scale_map(score, valid_mask) for score in hop_maps.values()]),
        axis=0,
    ).astype(np.float32)
    weight_array = np.asarray(product_weights, dtype=np.float64)
    if float(np.sum(weight_array)) <= 0.0:
        weight_array = np.ones_like(weight_array)
    weight_array /= float(np.sum(weight_array))
    product = np.sum(
        np.stack(product_components, axis=0) * weight_array[:, None, None],
        axis=0,
    ).astype(np.float32)
    fused[~valid_mask] = 0.0
    product[~valid_mask] = 0.0
    return SuccessiveSaabResult(
        hop_maps=hop_maps,
        fused_map=fused,
        product_weighted_map=product,
        hop_geodesic=hop_geodesic,
        pre_features=tuple(pre_features),
        post_features=tuple(post_features),
        feature_masks=tuple(feature_masks),
        stages=tuple(stages),
        score_mode=score_mode,
        rank=int(rank),
    )
