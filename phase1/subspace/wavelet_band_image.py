"""Wavelet-domain spatial Band-Image Difference Subspace representations.

Source/provenance
-----------------
The two-dimensional discrete/stationary wavelet transforms follow standard
multiresolution analysis (Mallat, TPAMI 1989) and are executed by PyWavelets.
The transform is applied independently and identically to every aligned band
and date.  At one scale/orientation, the 13 flattened coefficient maps form
``X_t in R^(N_coefficients x 13)``; canonical DS then follows Fukui and Maki
through ``phase1.subspace.band_image_geometry``.

The critically sampled DWT path uses inverse wavelet reconstruction to map
projected signed coefficient evidence back to pixel space.  The stationary
wavelet transform (SWT) keeps full-resolution coefficient maps and is the main
localization variant because it avoids DWT decimation and is less sensitive to
grid phase.  Neither path is called Green Learning or PixelHop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pywt
from scipy.ndimage import distance_transform_edt

from phase1.subspace.band_image_geometry import band_image_ds_values
from phase1.subspace.multiscale_band_image import (
    quantile_scale_map,
    score_band_image_matrix,
)


Array = np.ndarray


@dataclass(frozen=True)
class WaveletBandImageResult:
    component_maps: dict[str, Array]
    fused_map: Array
    approximation_map: Array
    detail_map: Array
    component_geodesic: dict[str, float]
    transform: str
    wavelet: str
    levels: int
    score_mode: str
    rank: int


def _validate(first: Array, second: Array, valid_mask: Array, levels: int) -> None:
    if first.ndim != 3 or second.shape != first.shape:
        raise ValueError(
            f"Expected paired bands x height x width cubes, got {first.shape} and {second.shape}."
        )
    if valid_mask.shape != first.shape[1:]:
        raise ValueError("Wavelet valid mask does not match the spatial cube shape.")
    if int(levels) < 1:
        raise ValueError("Wavelet decomposition requires at least one level.")
    if first.shape[0] < 2:
        raise ValueError("At least two band coefficient maps are required.")


def fill_invalid_nearest(cube: Array, valid_mask: Array) -> Array:
    """Fill invalid pixels from the nearest valid coordinate before filtering."""
    if np.all(valid_mask):
        return cube.astype(np.float32, copy=True)
    if not np.any(valid_mask):
        raise ValueError("Cannot wavelet-transform an image without valid pixels.")
    indices = distance_transform_edt(
        ~valid_mask,
        return_distances=False,
        return_indices=True,
    )
    filled = cube[:, indices[0], indices[1]].astype(np.float32, copy=False)
    filled[:, valid_mask] = cube[:, valid_mask]
    return filled


def _pad_to_multiple(cube: Array, multiple: int) -> tuple[Array, tuple[int, int]]:
    height, width = cube.shape[1:]
    pad_h = (-height) % int(multiple)
    pad_w = (-width) % int(multiple)
    padded = np.pad(cube, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    return padded.astype(np.float32, copy=False), (height, width)


def _swt_components(cube: Array, *, wavelet: str, levels: int) -> dict[str, Array]:
    components: dict[str, list[Array]] = {}
    for band in cube:
        coeffs = pywt.swt2(
            band,
            wavelet,
            level=int(levels),
            start_level=0,
            trim_approx=False,
            norm=True,
        )
        components.setdefault(f"L{levels}_LL", []).append(
            np.asarray(coeffs[0][0], dtype=np.float32)
        )
        for index, (_approximation, details) in enumerate(coeffs):
            level = int(levels) - index
            for orientation, values in zip(("LH", "HL", "HH"), details):
                components.setdefault(f"L{level}_{orientation}", []).append(
                    np.asarray(values, dtype=np.float32)
                )
    return {
        name: np.stack(values, axis=0).astype(np.float32, copy=False)
        for name, values in components.items()
    }


def stationary_wavelet_band_image_geometry(
    first: Array,
    second: Array,
    valid_mask: Array,
    *,
    rank: int,
    levels: int = 2,
    wavelet: str = "haar",
    score_mode: str = "ds_magnitude",
    seed: int = 1234,
    basis_mode: str = "centered_pca",
) -> WaveletBandImageResult:
    """Return full-resolution SWT component and fused spatial geometry maps."""
    _validate(first, second, valid_mask, levels)
    first_filled = fill_invalid_nearest(first, valid_mask)
    second_filled = fill_invalid_nearest(second, valid_mask)
    multiple = 2 ** int(levels)
    first_padded, original_shape = _pad_to_multiple(first_filled, multiple)
    second_padded, _ = _pad_to_multiple(second_filled, multiple)
    padded_mask = np.zeros(first_padded.shape[1:], dtype=bool)
    padded_mask[: original_shape[0], : original_shape[1]] = valid_mask
    first_components = _swt_components(first_padded, wavelet=wavelet, levels=levels)
    second_components = _swt_components(second_padded, wavelet=wavelet, levels=levels)

    maps: dict[str, Array] = {}
    geodesics: dict[str, float] = {}
    for name in sorted(first_components):
        first_matrix = first_components[name][:, padded_mask].T
        second_matrix = second_components[name][:, padded_mask].T
        score, _pre, _post, _effective_rank, geodesic, _chordal, _maximum = (
            score_band_image_matrix(
                first_matrix,
                second_matrix,
                rank=rank,
                seed=seed,
                score_mode=score_mode,
                basis_mode=basis_mode,
            )
        )
        full = np.zeros(padded_mask.shape, dtype=np.float32)
        full[padded_mask] = score
        maps[name] = full[: original_shape[0], : original_shape[1]]
        maps[name][~valid_mask] = 0.0
        geodesics[name] = float(geodesic)

    approximation_names = [name for name in maps if name.endswith("_LL")]
    detail_names = [name for name in maps if not name.endswith("_LL")]
    approximation = np.mean(
        np.stack([quantile_scale_map(maps[name], valid_mask) for name in approximation_names]),
        axis=0,
    ).astype(np.float32)
    detail = np.mean(
        np.stack([quantile_scale_map(maps[name], valid_mask) for name in detail_names]),
        axis=0,
    ).astype(np.float32)
    fused = np.mean(
        np.stack([quantile_scale_map(score, valid_mask) for score in maps.values()]),
        axis=0,
    ).astype(np.float32)
    for output in (approximation, detail, fused):
        output[~valid_mask] = 0.0
    return WaveletBandImageResult(
        component_maps=maps,
        fused_map=fused,
        approximation_map=approximation,
        detail_map=detail,
        component_geodesic=geodesics,
        transform="swt",
        wavelet=wavelet,
        levels=int(levels),
        score_mode=score_mode,
        rank=int(rank),
    )


def _empty_dwt_like(coefficients: Sequence[object]) -> list[object]:
    empty: list[object] = [np.zeros_like(coefficients[0])]
    for details in coefficients[1:]:
        empty.append(tuple(np.zeros_like(value) for value in details))
    return empty


def decimated_wavelet_band_image_ds(
    first: Array,
    second: Array,
    valid_mask: Array,
    *,
    rank: int,
    levels: int = 2,
    wavelet: str = "haar",
    seed: int = 1234,
    basis_mode: str = "centered_pca",
) -> WaveletBandImageResult:
    """Return DWT DS maps reconstructed exactly from projected coefficients."""
    _validate(first, second, valid_mask, levels)
    first_filled = fill_invalid_nearest(first, valid_mask)
    second_filled = fill_invalid_nearest(second, valid_mask)
    multiple = 2 ** int(levels)
    first_padded, original_shape = _pad_to_multiple(first_filled, multiple)
    second_padded, _ = _pad_to_multiple(second_filled, multiple)
    first_coeffs = [
        pywt.wavedec2(band, wavelet, mode="periodization", level=int(levels))
        for band in first_padded
    ]
    second_coeffs = [
        pywt.wavedec2(band, wavelet, mode="periodization", level=int(levels))
        for band in second_padded
    ]
    descriptors: list[tuple[str, int, int | None]] = [(f"L{levels}_LL", 0, None)]
    for coefficient_index in range(1, int(levels) + 1):
        level = int(levels) - coefficient_index + 1
        for orientation_index, orientation in enumerate(("LH", "HL", "HH")):
            descriptors.append((f"L{level}_{orientation}", coefficient_index, orientation_index))

    maps: dict[str, Array] = {}
    geodesics: dict[str, float] = {}
    for name, coefficient_index, orientation_index in descriptors:
        if orientation_index is None:
            first_stack = np.stack([coeffs[0] for coeffs in first_coeffs], axis=0)
            second_stack = np.stack([coeffs[0] for coeffs in second_coeffs], axis=0)
        else:
            first_stack = np.stack(
                [coeffs[coefficient_index][orientation_index] for coeffs in first_coeffs],
                axis=0,
            )
            second_stack = np.stack(
                [coeffs[coefficient_index][orientation_index] for coeffs in second_coeffs],
                axis=0,
            )
        ds = band_image_ds_values(
            first_stack.reshape(first.shape[0], -1).T,
            second_stack.reshape(second.shape[0], -1).T,
            rank=rank,
            seed=seed,
            basis_mode=basis_mode,
        )
        reconstructed = []
        coefficient_shape = first_stack.shape[1:]
        for band_index in range(first.shape[0]):
            target = _empty_dwt_like(first_coeffs[band_index])
            projected = ds.projected[:, band_index].reshape(coefficient_shape)
            if orientation_index is None:
                target[0] = projected
            else:
                details = list(target[coefficient_index])
                details[orientation_index] = projected
                target[coefficient_index] = tuple(details)
            image = pywt.waverec2(target, wavelet, mode="periodization")
            reconstructed.append(image[: first_padded.shape[1], : first_padded.shape[2]])
        reconstructed_stack = np.stack(reconstructed, axis=0)
        score = np.linalg.norm(reconstructed_stack, axis=0).astype(np.float32)
        score = score[: original_shape[0], : original_shape[1]]
        score[~valid_mask] = 0.0
        maps[name] = score
        singular = np.linalg.svd(ds.pre_basis.T @ ds.post_basis, compute_uv=False)
        angles = np.arccos(np.clip(singular, 0.0, 1.0))
        geodesics[name] = float(np.linalg.norm(angles))

    approximation_names = [name for name in maps if name.endswith("_LL")]
    detail_names = [name for name in maps if not name.endswith("_LL")]
    approximation = np.mean(
        np.stack([quantile_scale_map(maps[name], valid_mask) for name in approximation_names]),
        axis=0,
    ).astype(np.float32)
    detail = np.mean(
        np.stack([quantile_scale_map(maps[name], valid_mask) for name in detail_names]),
        axis=0,
    ).astype(np.float32)
    fused = np.mean(
        np.stack([quantile_scale_map(score, valid_mask) for score in maps.values()]),
        axis=0,
    ).astype(np.float32)
    for output in (approximation, detail, fused):
        output[~valid_mask] = 0.0
    return WaveletBandImageResult(
        component_maps=maps,
        fused_map=fused,
        approximation_map=approximation,
        detail_map=detail,
        component_geodesic=geodesics,
        transform="dwt",
        wavelet=wavelet,
        levels=int(levels),
        score_mode="ds_magnitude",
        rank=int(rank),
    )
