"""Load labeled bitemporal hyperspectral change-detection benchmarks.

Source/provenance
-----------------
- ``benton`` is the irrigated agricultural Hyperion dataset published by
  Sicong Liu and collaborators.  The repository documents acquisition dates,
  preprocessing, original Hyperion band indices, and binary/multiclass maps:
  https://github.com/SicongLiuRS/Hyperspectral-Change-Detection-Dataset-Irrigated-Agricultural-Area
- ``hermiston`` is the Hermiston Hyperion bundle linked by the PTCD/TDRD
  authors: https://github.com/zephyrhours/Hyperspectral-Change-Detection-PTCD

Project adaptation
------------------
The loaders return cubes as ``bands x rows x columns`` and keep original sensor
band indices for attribution.  Constant bands are removed jointly across the
two dates.  Preprocessing parameters are estimated from an unlabeled seeded
sample of both dates; labels never influence normalization or band removal.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


Array = np.ndarray


@dataclass(frozen=True)
class HSIChangePair:
    name: str
    first: Array  # bands x rows x columns
    second: Array
    labels: Array  # rows x columns, 0 unchanged / 1 changed
    valid_mask: Array
    original_band_indices: Array  # one-based sensor band numbers
    source_url: str
    preprocessing: str


def _benton_band_indices() -> Array:
    ranges = ((8, 57), (82, 119), (131, 164), (182, 184), (187, 220))
    return np.asarray(
        [band for first, last in ranges for band in range(first, last + 1)],
        dtype=np.int32,
    )


def _load_benton(root: Path) -> tuple[Array, Array, Array, Array, str]:
    folder = root / "benton"
    first = loadmat(folder / "PreImg_2004.mat")["img_2004"]
    second = loadmat(folder / "PostImg_2007.mat")["img_2007"]
    labels = loadmat(folder / "Reference_Map_Binary.mat")["Ref_map_binary"]
    return (
        np.moveaxis(first, -1, 0),
        np.moveaxis(second, -1, 0),
        labels,
        _benton_band_indices(),
        "https://github.com/SicongLiuRS/Hyperspectral-Change-Detection-Dataset-Irrigated-Agricultural-Area",
    )


def _load_hermiston(root: Path) -> tuple[Array, Array, Array, Array, str]:
    path = root / "hermiston" / "hermiston_download"
    with h5py.File(path, "r") as handle:
        # MATLAB v7.3 arrays are exposed in reversed axis order by h5py.
        first = np.asarray(handle["hsi_t1"], dtype=np.float32).transpose(0, 2, 1)
        second = np.asarray(handle["hsi_t2"], dtype=np.float32).transpose(0, 2, 1)
        labels = np.asarray(handle["hsi_gt"], dtype=np.uint8).T
    return (
        first,
        second,
        labels,
        np.arange(1, first.shape[0] + 1, dtype=np.int32),
        "https://github.com/zephyrhours/Hyperspectral-Change-Detection-PTCD",
    )


def _load_btcdnet_scene(
    root: Path, name: str
) -> tuple[Array, Array, Array, Array, str]:
    filenames = {"farmland": "Farmland.mat", "shenzhen": "Shenzhen.mat"}
    package = loadmat(root / "btcdnet" / filenames[name])
    first = np.moveaxis(package["T1"], -1, 0)
    second = np.moveaxis(package["T2"], -1, 0)
    raw_labels = package["Change"]
    if name == "farmland":
        # HyperSIGMA/BTCDNet convention: 1 unchanged, 2 changed.
        labels = (raw_labels == 2).astype(np.uint8)
    else:
        # Shenzhen convention in BTCDNet CVA.py: 0 ignored, 1 changed, 2 unchanged.
        labels = np.where(raw_labels == 0, 255, (raw_labels == 1).astype(np.uint8))
    return (
        first,
        second,
        labels,
        np.arange(1, first.shape[0] + 1, dtype=np.int32),
        "https://github.com/JeasunLok/BTCDNet",
    )


def _sample_joint_pixels(
    first: Array,
    second: Array,
    maximum: int,
    seed: int,
    valid_mask: Array | None = None,
) -> Array:
    bands = first.shape[0]
    joined = np.concatenate(
        [first.reshape(bands, -1), second.reshape(bands, -1)], axis=1
    )
    finite = np.all(np.isfinite(joined), axis=0)
    if valid_mask is not None:
        repeated_valid = np.concatenate([valid_mask.reshape(-1), valid_mask.reshape(-1)])
        finite &= repeated_valid
    index = np.flatnonzero(finite)
    if index.size > int(maximum):
        rng = np.random.default_rng(int(seed))
        index = rng.choice(index, size=int(maximum), replace=False)
    return joined[:, index].astype(np.float64, copy=False)


def _preprocess_pair(
    first: Array,
    second: Array,
    original_bands: Array,
    mode: str,
    band_policy: str,
    valid_mask: Array,
    seed: int,
) -> tuple[Array, Array, Array]:
    policy = str(band_policy).strip().lower().replace("-", "_")
    if policy == "common_hyperion_159":
        documented = set(_benton_band_indices().tolist())
        selected = np.asarray([int(value) in documented for value in original_bands])
        first = first[selected]
        second = second[selected]
        original_bands = original_bands[selected]
    elif policy != "nonconstant":
        raise ValueError(
            f"Unknown band_policy={band_policy!r}; expected nonconstant or common_hyperion_159."
        )
    sample = _sample_joint_pixels(
        first, second, maximum=100_000, seed=seed, valid_mask=valid_mask
    )
    joint_variance = np.var(sample, axis=1)
    joint_span = np.ptp(sample, axis=1)
    keep = np.isfinite(joint_variance) & (joint_variance > 1e-12) & (joint_span > 1e-10)
    if not np.any(keep):
        raise ValueError("No nonconstant finite hyperspectral bands remain.")
    first = first[keep].astype(np.float32, copy=False)
    second = second[keep].astype(np.float32, copy=False)
    original_bands = original_bands[keep]
    key = str(mode).strip().lower().replace("-", "_")

    if key == "none":
        return first, second, original_bands

    if key in {"joint_robust_zscore", "joint_zscore"}:
        sample = sample[keep]
        if key == "joint_robust_zscore":
            center = np.median(sample, axis=1)
            q25, q75 = np.percentile(sample, (25.0, 75.0), axis=1)
            scale = (q75 - q25) / 1.3489795003921634
            fallback = np.std(sample, axis=1)
            # Quantized/noisy bands can have an IQR of zero or one digital
            # count depending on the random normalization sample. Accepting a
            # tiny but nonzero IQR can amplify that band by orders of magnitude
            # and destabilize local eigenspaces. Use standard deviation when
            # robust scale is below 20% of the corresponding standard scale.
            scale = np.where(scale > np.maximum(1e-8, 0.2 * fallback), scale, fallback)
        else:
            center = np.mean(sample, axis=1)
            scale = np.std(sample, axis=1)
        scale = np.maximum(scale, 1e-8)
        first = (first - center[:, None, None]) / scale[:, None, None]
        second = (second - center[:, None, None]) / scale[:, None, None]
    elif key == "per_date_zscore":
        for cube in (first, second):
            matrix = cube.reshape(cube.shape[0], -1).astype(np.float64, copy=False)
            center = np.mean(matrix, axis=1)
            scale = np.maximum(np.std(matrix, axis=1), 1e-8)
            cube -= center[:, None, None]
            cube /= scale[:, None, None]
    else:
        raise ValueError(
            f"Unknown preprocessing={mode!r}; expected none, joint_robust_zscore, "
            "joint_zscore, or per_date_zscore."
        )
    return first.astype(np.float32), second.astype(np.float32), original_bands


def load_hsi_change_pair(
    name: str,
    root: str | Path = "data/HSI_change",
    preprocessing: str = "joint_robust_zscore",
    band_policy: str = "nonconstant",
    seed: int = 1234,
) -> HSIChangePair:
    """Load and normalize one supported HSI change pair."""
    key = str(name).strip().lower()
    root = Path(root)
    if key == "benton":
        first, second, labels, bands, source = _load_benton(root)
    elif key == "hermiston":
        first, second, labels, bands, source = _load_hermiston(root)
    elif key in {"farmland", "shenzhen"}:
        first, second, labels, bands, source = _load_btcdnet_scene(root, key)
    else:
        raise ValueError(
            f"Unknown HSI dataset {name!r}; expected benton, hermiston, farmland, or shenzhen."
        )

    if first.shape != second.shape:
        raise ValueError(f"Date cube mismatch: {first.shape} vs {second.shape}.")
    if first.shape[1:] != labels.shape:
        raise ValueError(f"Cube/label mismatch: {first.shape[1:]} vs {labels.shape}.")
    label_valid = np.isin(labels, (0, 1))
    first, second, bands = _preprocess_pair(
        first, second, bands, preprocessing, band_policy, label_valid, seed
    )
    # Ignored Shenzhen pixels contain -32768 fill values.  Keep them out of
    # normalization and replace them with the normalized center so neighboring
    # local windows are not dominated by the sentinel value.
    first[:, ~label_valid] = 0.0
    second[:, ~label_valid] = 0.0
    valid = (
        np.all(np.isfinite(first), axis=0)
        & np.all(np.isfinite(second), axis=0)
        & np.isin(labels, (0, 1))
    )
    return HSIChangePair(
        name=key,
        first=first,
        second=second,
        labels=(labels > 0).astype(np.uint8),
        valid_mask=valid,
        original_band_indices=bands,
        source_url=source,
        preprocessing=f"{preprocessing}; bands={band_policy}",
    )
