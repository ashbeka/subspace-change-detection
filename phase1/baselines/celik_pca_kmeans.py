"""
Celik-style local PCA + k-means change-detection baseline.

Source/provenance:
- Based on the unsupervised PCA + k-means change-detection idea in Celik 2009:
  extract local difference-image patches, reduce them with PCA, and cluster
  projected patch features into changed/unchanged groups.
- For multispectral Sentinel-2 input, the default first converts the per-pixel
  band difference to a scalar CVA/L2 magnitude image, then extracts local
  patches from that image. This is closer to Celik's difference-image pipeline
  than concatenating every band inside every patch.
- ``feature_mode="multiband_patch"`` retains the older multiband-patch
  adaptation for controlled comparison, but it is not the default.
- PCA and k-means are fit on a seeded subset of valid patches and applied in
  chunks. This avoids materializing a potentially gigabyte-scale patch matrix.

Verification status:
- Useful spatial baseline pressure for patch-vector DS because both methods use
  local patch structure. It should be treated as an implementation of the Celik
  family, not a line-by-line reproduction of a specific public codebase.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


Array = np.ndarray


def _patch_view(diff: Array, patch_size: int) -> Array:
    """Return a strided ``(C, H, W, patch, patch)`` patch view."""
    pad = patch_size // 2
    padded = np.pad(diff, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    return np.lib.stride_tricks.sliding_window_view(
        padded,
        window_shape=(patch_size, patch_size),
        axis=(1, 2),
    )


def _features_at(windows: Array, flat_indices: Array) -> Array:
    """Materialize patch features only for selected flattened pixel indices."""
    _, _, width, _, _ = windows.shape
    rows = flat_indices // width
    cols = flat_indices % width
    selected = windows[:, rows, cols, :, :]
    return selected.transpose(1, 0, 2, 3).reshape(flat_indices.size, -1).astype(np.float32, copy=False)


def celik_score(
    x1: Array,
    x2: Array,
    patch_size: int = 9,
    pca_energy: float = 0.9,
    kmeans_init: str = "k-means++",
    max_iter: int = 100,
    valid_mask: Optional[Array] = None,
    random_state: int = 1234,
    downsample_max_side: Optional[int] = None,
    feature_mode: str = "spectral_norm",
    max_fit_samples: int = 20000,
    score_chunk_size: int = 20000,
) -> Array:
    """
    Apply Celik local PCA + k-means; returns a normalized change score map.
    """
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd.")
    diff_multiband = x2 - x1
    if feature_mode == "spectral_norm":
        diff = np.linalg.norm(diff_multiband, axis=0, keepdims=True).astype(np.float32)
    elif feature_mode == "multiband_patch":
        diff = diff_multiband.astype(np.float32, copy=False)
    else:
        raise ValueError("feature_mode must be 'spectral_norm' or 'multiband_patch'.")
    orig_h, orig_w = diff.shape[1:]
    step = 1
    # optional downsample for stability on large tiles
    if downsample_max_side is not None:
        max_side = max(orig_h, orig_w)
        if max_side > downsample_max_side:
            step = int(np.ceil(max_side / downsample_max_side))
            diff = diff[:, ::step, ::step]
            if valid_mask is not None:
                valid_mask = valid_mask[::step, ::step]
    windows = _patch_view(diff, patch_size)
    if valid_mask is not None:
        flat_mask = valid_mask.reshape(-1)
    else:
        flat_mask = np.ones(diff.shape[1] * diff.shape[2], dtype=bool)

    valid_indices = np.flatnonzero(flat_mask)
    if valid_indices.size == 0:
        raise RuntimeError("No valid pixels for Celik baseline.")

    rng = np.random.default_rng(random_state)
    if valid_indices.size > max_fit_samples:
        fit_indices = np.sort(rng.choice(valid_indices, size=max_fit_samples, replace=False))
    else:
        fit_indices = valid_indices
    feats_fit = _features_at(windows, fit_indices)

    pca = PCA(n_components=min(feats_fit.shape), svd_solver="full", random_state=random_state)
    pca.fit(feats_fit)
    cs = np.cumsum(pca.explained_variance_ratio_)
    keep = max(1, int(np.searchsorted(cs, pca_energy) + 1))
    proj_fit = pca.transform(feats_fit)[:, :keep]
    mags_fit = np.linalg.norm(proj_fit, axis=1)
    # Cluster on the seeded fit subset, then predict all valid patches.
    kmeans = KMeans(n_clusters=2, init=kmeans_init, max_iter=max_iter, random_state=random_state, n_init=10)
    labels_fit = kmeans.fit_predict(proj_fit)
    cluster_scores = [mags_fit[labels_fit == k].mean() for k in range(2)]
    change_cluster = int(np.argmax(cluster_scores))

    labels = np.empty(valid_indices.size, dtype=np.int8)
    mags = np.empty(valid_indices.size, dtype=np.float32)
    for start in range(0, valid_indices.size, score_chunk_size):
        stop = min(start + score_chunk_size, valid_indices.size)
        chunk = _features_at(windows, valid_indices[start:stop])
        proj = pca.transform(chunk)[:, :keep]
        labels[start:stop] = kmeans.predict(proj).astype(np.int8)
        mags[start:stop] = np.linalg.norm(proj, axis=1).astype(np.float32)

    score_flat = np.zeros_like(flat_mask, dtype=np.float32)
    score_flat[valid_indices] = (labels == change_cluster).astype(np.float32)

    # Smoothen by injecting magnitude as confidence and min-max normalize
    score_vals = np.zeros_like(flat_mask, dtype=np.float32)
    score_vals[valid_indices] = mags
    if mags.size > 0:
        vmin, vmax = mags.min(), mags.max()
        norm_mags = (mags - vmin) / (vmax - vmin + 1e-8) if vmax > vmin else np.zeros_like(mags)
        score_vals[valid_indices] = norm_mags
    score_flat = score_flat * score_vals

    score = score_flat.reshape(diff.shape[1], diff.shape[2])
    if step > 1:
        score = np.repeat(np.repeat(score, step, axis=0), step, axis=1)
        score = score[:orig_h, :orig_w]
    return score
