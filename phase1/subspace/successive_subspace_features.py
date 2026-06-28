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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

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


@dataclass(frozen=True)
class SuccessiveSaabModel:
    """A label-free successive Saab hierarchy reusable across image pairs."""

    stages: tuple[SaabStage, ...]
    fit_pair_count: int
    fit_samples_per_stage: tuple[int, ...]
    energy_threshold: float
    max_output_channels: tuple[int, ...]
    seed: int
    source_description: str


PairFactory = Callable[[], Iterable[tuple[Array, Array, Array]]]


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


def _validate_pair(first: Array, second: Array, valid_mask: Array) -> None:
    if first.ndim != 3 or second.shape != first.shape:
        raise ValueError("Expected paired channels x height x width feature cubes.")
    if valid_mask.shape != first.shape[1:]:
        raise ValueError("Successive Saab mask does not match the input cube.")
    if not np.any(valid_mask):
        raise ValueError("Successive Saab requires at least one valid pixel.")


def _advance_through_stages(
    first: Array,
    second: Array,
    valid_mask: Array,
    stages: Sequence[SaabStage],
    *,
    device: str,
) -> tuple[Array, Array, Array]:
    """Return the input object expected by the stage after ``stages``."""
    current_first = first.astype(np.float32, copy=False)
    current_second = second.astype(np.float32, copy=False)
    current_mask = valid_mask.astype(bool, copy=True)
    for stage in stages:
        transformed_first = apply_saab_stage(current_first, stage, device=device)
        transformed_second = apply_saab_stage(current_second, stage, device=device)
        current_first, first_mask = _pool_features(
            transformed_first,
            current_mask,
            stage.pool_size,
        )
        current_second, second_mask = _pool_features(
            transformed_second,
            current_mask,
            stage.pool_size,
        )
        current_mask = first_mask & second_mask
    return current_first, current_second, current_mask


def fit_successive_saab_model(
    pair_factory: PairFactory,
    *,
    pair_count: int,
    input_channels: int,
    hops: int = 2,
    patch_size: int = 3,
    pool_size: int = 2,
    energy_threshold: float = 0.95,
    max_output_channels: Sequence[int] = (16, 16),
    max_fit_samples: int = 30000,
    seed: int = 1234,
    device: str = "cpu",
    source_description: str = "unspecified unlabeled training pairs",
) -> SuccessiveSaabModel:
    """Fit one reusable hierarchy from a deterministic factory of training pairs.

    The factory is replayed once per hop. This keeps large satellite scenes out
    of memory while ensuring that hop-2 samples are generated by the already
    frozen hop-1 transform. Labels are never accepted by this interface.
    """
    if int(pair_count) < 1:
        raise ValueError("pair_count must be positive.")
    if int(input_channels) < 1:
        raise ValueError("input_channels must be positive.")
    if int(hops) < 1:
        raise ValueError("At least one successive hop is required.")
    if int(max_fit_samples) < 2:
        raise ValueError("max_fit_samples must be at least two.")
    channel_limits = [int(value) for value in max_output_channels]
    if not channel_limits:
        raise ValueError("At least one output-channel limit is required.")
    if len(channel_limits) < int(hops):
        channel_limits.extend([channel_limits[-1]] * (int(hops) - len(channel_limits)))

    rng = np.random.default_rng(int(seed))
    stages: list[SaabStage] = []
    fitted_counts: list[int] = []
    per_pair_budget = max(2, int(np.ceil(float(max_fit_samples) / float(pair_count))))

    for hop_index in range(int(hops)):
        blocks: list[Array] = []
        observed_pairs = 0
        for raw_first, raw_second, raw_mask in pair_factory():
            _validate_pair(raw_first, raw_second, raw_mask)
            if raw_first.shape[0] != int(input_channels):
                raise ValueError(
                    f"Training pair has {raw_first.shape[0]} channels; expected {input_channels}."
                )
            current_first, current_second, current_mask = _advance_through_stages(
                raw_first,
                raw_second,
                raw_mask,
                stages,
                device=device,
            )
            blocks.append(
                _sample_patch_matrix(
                    current_first,
                    current_second,
                    current_mask,
                    patch_size=patch_size,
                    max_samples=per_pair_budget,
                    rng=rng,
                )
            )
            observed_pairs += 1
        if observed_pairs != int(pair_count):
            raise ValueError(
                f"pair_factory yielded {observed_pairs} pairs; expected {pair_count}."
            )
        samples = np.concatenate(blocks, axis=0)
        if samples.shape[0] > int(max_fit_samples):
            selected = rng.choice(
                samples.shape[0],
                size=int(max_fit_samples),
                replace=False,
            )
            samples = samples[selected]
        expected_channels = int(input_channels) if not stages else stages[-1].output_channels
        stage = fit_saab_stage(
            samples,
            input_channels=expected_channels,
            patch_size=patch_size,
            pool_size=pool_size,
            energy_threshold=energy_threshold,
            max_output_channels=channel_limits[hop_index],
        )
        stages.append(stage)
        fitted_counts.append(int(samples.shape[0]))

    return SuccessiveSaabModel(
        stages=tuple(stages),
        fit_pair_count=int(pair_count),
        fit_samples_per_stage=tuple(fitted_counts),
        energy_threshold=float(energy_threshold),
        max_output_channels=tuple(channel_limits[: int(hops)]),
        seed=int(seed),
        source_description=str(source_description),
    )


def save_successive_saab_model(path: Path, model: SuccessiveSaabModel) -> None:
    """Save a model without pickle so it can be inspected and reproduced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "fit_pair_count": model.fit_pair_count,
        "fit_samples_per_stage": list(model.fit_samples_per_stage),
        "energy_threshold": model.energy_threshold,
        "max_output_channels": list(model.max_output_channels),
        "seed": model.seed,
        "source_description": model.source_description,
        "stage_count": len(model.stages),
        "stages": [
            {
                "patch_size": stage.patch_size,
                "pool_size": stage.pool_size,
                "input_channels": stage.input_channels,
                "output_channels": stage.output_channels,
            }
            for stage in model.stages
        ],
    }
    arrays: dict[str, Array] = {
        "metadata_json": np.asarray(json.dumps(metadata)),
    }
    for index, stage in enumerate(model.stages):
        arrays[f"stage_{index}_kernels"] = stage.kernels
        arrays[f"stage_{index}_intercept"] = stage.intercept
        arrays[f"stage_{index}_ac_ratio"] = stage.ac_explained_variance_ratio
    np.savez_compressed(path, **arrays)


def load_successive_saab_model(path: Path) -> SuccessiveSaabModel:
    """Load a model written by :func:`save_successive_saab_model`."""
    with np.load(Path(path), allow_pickle=False) as package:
        metadata = json.loads(str(package["metadata_json"].item()))
        stages: list[SaabStage] = []
        for index, stage_metadata in enumerate(metadata["stages"]):
            stages.append(
                SaabStage(
                    kernels=np.asarray(package[f"stage_{index}_kernels"], dtype=np.float32),
                    intercept=np.asarray(package[f"stage_{index}_intercept"], dtype=np.float32),
                    patch_size=int(stage_metadata["patch_size"]),
                    pool_size=int(stage_metadata["pool_size"]),
                    input_channels=int(stage_metadata["input_channels"]),
                    output_channels=int(stage_metadata["output_channels"]),
                    ac_explained_variance_ratio=np.asarray(
                        package[f"stage_{index}_ac_ratio"],
                        dtype=np.float32,
                    ),
                )
            )
    return SuccessiveSaabModel(
        stages=tuple(stages),
        fit_pair_count=int(metadata["fit_pair_count"]),
        fit_samples_per_stage=tuple(int(value) for value in metadata["fit_samples_per_stage"]),
        energy_threshold=float(metadata["energy_threshold"]),
        max_output_channels=tuple(int(value) for value in metadata["max_output_channels"]),
        seed=int(metadata["seed"]),
        source_description=str(metadata["source_description"]),
    )


def apply_successive_saab_model(
    first: Array,
    second: Array,
    valid_mask: Array,
    model: SuccessiveSaabModel,
    *,
    rank: int,
    score_mode: str = "ds_magnitude",
    basis_mode: str = "centered_pca",
    device: str = "cpu",
) -> SuccessiveSaabResult:
    """Apply frozen Saab filters, then fit only the pre/post spatial DS object."""
    _validate_pair(first, second, valid_mask)
    if not model.stages:
        raise ValueError("The successive Saab model contains no stages.")
    if first.shape[0] != model.stages[0].input_channels:
        raise ValueError(
            f"Input has {first.shape[0]} channels; model expects "
            f"{model.stages[0].input_channels}."
        )
    current_first = first.astype(np.float32, copy=False)
    current_second = second.astype(np.float32, copy=False)
    current_mask = valid_mask.astype(bool, copy=True)
    pre_features: list[Array] = []
    post_features: list[Array] = []
    feature_masks: list[Array] = []
    for stage_index, stage in enumerate(model.stages):
        transformed_first = apply_saab_stage(current_first, stage, device=device)
        transformed_second = apply_saab_stage(current_second, stage, device=device)
        pre_features.append(transformed_first)
        post_features.append(transformed_second)
        feature_masks.append(current_mask.copy())
        if stage_index + 1 < len(model.stages):
            current_first, first_mask = _pool_features(
                transformed_first,
                current_mask,
                stage.pool_size,
            )
            current_second, second_mask = _pool_features(
                transformed_second,
                current_mask,
                stage.pool_size,
            )
            current_mask = first_mask & second_mask

    hop_maps: dict[int, Array] = {}
    hop_geodesic: dict[int, float] = {}
    original_shape = valid_mask.shape
    temporary: list[tuple[Array, float]] = []
    for hop_index, (pre, post, mask) in enumerate(
        zip(pre_features, post_features, feature_masks),
        start=1,
    ):
        score, pre_basis, post_basis, _effective_rank, geodesic, _chordal, _maximum = (
            score_band_image_matrix(
                pre[:, mask].T,
                post[:, mask].T,
                rank=min(int(rank), pre.shape[0] - 1),
                seed=model.seed,
                score_mode=score_mode,
                basis_mode=basis_mode,
            )
        )
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
        temporary.append((resized, geodesic))

    positive = [distance for _score, distance in temporary if distance > 0.0]
    reference = float(np.median(positive)) if positive else 1.0
    components = [quantile_scale_map(score, valid_mask) for score, _distance in temporary]
    weights = np.asarray(
        [float(np.clip(distance / max(reference, 1e-12), 0.0, 4.0)) for _score, distance in temporary],
        dtype=np.float64,
    )
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones_like(weights)
    weights /= float(np.sum(weights))
    fused = np.mean(np.stack(components, axis=0), axis=0).astype(np.float32)
    product = np.sum(
        np.stack(components, axis=0) * weights[:, None, None],
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
        stages=model.stages,
        score_mode=score_mode,
        rank=int(rank),
    )


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
    _validate_pair(first, second, valid_mask)

    def pair_factory() -> Iterable[tuple[Array, Array, Array]]:
        yield first, second, valid_mask

    model = fit_successive_saab_model(
        pair_factory,
        pair_count=1,
        input_channels=first.shape[0],
        hops=hops,
        patch_size=patch_size,
        pool_size=pool_size,
        energy_threshold=energy_threshold,
        max_output_channels=max_output_channels,
        max_fit_samples=max_fit_samples,
        seed=seed,
        device=device,
        source_description="one unlabeled pre/post pair (pair-adaptive)",
    )
    return apply_successive_saab_model(
        first,
        second,
        valid_mask,
        model,
        rank=rank,
        score_mode=score_mode,
        basis_mode=basis_mode,
        device=device,
    )
