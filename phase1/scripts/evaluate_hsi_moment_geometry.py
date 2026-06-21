"""Evaluate factorized local hyperspectral geometry on real change benchmarks.

Question
--------
Does local leading-eigenspace orientation provide useful and interpretable HSI
change evidence after separating mean, covariance scale, eigenspectrum shape,
and full covariance change?

The script uses Benton as the development dataset and Hermiston as a frozen
transfer dataset by default.  It also reports the reverse transfer as a
robustness check.  Every local method is compared against raw L2/CVA, spectral
angle, PCA-diff, IR-MAD, chronochrome, covariance equalization, an unsupervised
band-selection control, approximate RBF-MMD, a shrinkage-SPD distance, and an
NMF abundance-change proxy.

Sources
-------
- Wu, Du, Zhang 2013 HSI subspace CD: DOI 10.1109/JSTARS.2013.2241396.
- Liu et al. 2019 HSI-CD review: DOI 10.1109/MGRS.2019.2898520.
- Nielsen 2007 IR-MAD and Schaum/Stocker covariance-equalization families are
  implemented in ``phase1.baselines``.
- Random Fourier features approximate a shift-invariant RBF kernel following
  Rahimi and Recht 2007; local mean-embedding distance approximates MMD.
- NMF is only an abundance-change proxy, not a claim of reproducing a specific
  VCA/FCLS hyperspectral unmixing paper.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter
from scipy.stats import spearmanr
from sklearn.decomposition import MiniBatchNMF, PCA
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from phase1.baselines.anomalous_change import chronochrome_score, covariance_equalization_score
from phase1.baselines.cva import cva_score
from phase1.baselines.ir_mad import ir_mad_score
from phase1.baselines.pca_diff import pca_diff_score
from phase1.data.hsi_change import HSIChangePair, load_hsi_change_pair
from phase1.eval.metrics import binary_metrics
from phase1.eval.thresholding import otsu_threshold
from phase1.subspace.hsi_moment_geometry import (
    factor_local_moments,
    shrinkage_log_euclidean_distance,
)


Array = np.ndarray
FACTOR_METHODS = (
    "local_mean",
    "log_covariance_scale",
    "eigenspectrum_shape",
    "eigenspace_orientation",
    "stability_gated_orientation",
    "first_order_ds_magnitude",
    "local_ds_projection_magnitude",
    "local_ds_projection_ratio",
    "local_ds_projection_fraction",
    "local_ds_residual_magnitude",
    "local_raw_magnitude",
    "local_inverse_raw_magnitude_diagnostic",
    "normalized_covariance_frobenius",
    "normalized_covariance_bures",
    "shrinkage_spd_logeuclidean",
    "factor_rank_fusion",
)


@dataclass(frozen=True)
class Configuration:
    preprocessing: str
    window: int
    stride: int
    rank: int

    @property
    def name(self) -> str:
        short = "robust" if self.preprocessing == "joint_robust_zscore" else "perdate"
        return f"{short}_w{self.window}_s{self.stride}_r{self.rank}"


@dataclass
class LocalResult:
    maps: dict[str, Array]
    rows: Array
    columns: Array
    orientation_attribution: Array
    covariance_attribution: Array
    changed_fraction: Array
    eigengap: Array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data/HSI_change"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--datasets", default="benton,hermiston,farmland,shenzhen")
    parser.add_argument(
        "--configs",
        default="joint_robust_zscore:5:3:3,joint_robust_zscore:5:3:5,"
        "joint_robust_zscore:9:5:3,joint_robust_zscore:9:5:5,"
        "joint_robust_zscore:9:5:8,joint_robust_zscore:13:7:5,"
        "joint_robust_zscore:13:7:8,per_date_zscore:9:5:5",
        help="Comma list preprocessing:window:stride:rank.",
    )
    parser.add_argument("--rff_features", type=int, default=32)
    parser.add_argument(
        "--band_policy",
        default="nonconstant",
        choices=("nonconstant", "common_hyperion_159"),
    )
    parser.add_argument("--global_pca_rank", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument(
        "--stability_seeds",
        default="",
        help="Optional comma-separated alternate seeds for selected-map stability checks.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def parse_configs(value: str) -> list[Configuration]:
    result: list[Configuration] = []
    for item in value.split(","):
        mode, window, stride, rank = item.strip().split(":")
        config = Configuration(mode, int(window), int(stride), int(rank))
        if config.window % 2 != 1:
            raise ValueError(f"Window must be odd: {config}.")
        result.append(config)
    return result


def normalize_score(score: Array, valid: Array) -> Array:
    output = np.zeros_like(score, dtype=np.float32)
    values = np.nan_to_num(score[valid].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        return output
    low, high = float(np.min(values)), float(np.max(values))
    output[valid] = ((values - low) / max(high - low, 1e-12)).astype(np.float32)
    return output


def percentile_ranks(score: Array, valid: Array) -> Array:
    values = score[valid]
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, values.size, dtype=np.float32)
    output = np.zeros_like(score, dtype=np.float32)
    output[valid] = ranks
    return output


def score_metrics(score: Array, labels: Array, valid: Array) -> dict[str, float]:
    y = labels[valid].astype(np.uint8)
    s = score[valid].astype(np.float64)
    precision, recall, thresholds = precision_recall_curve(y, s)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    iou = (precision * recall) / np.maximum(precision + recall - precision * recall, 1e-12)
    best = int(np.nanargmax(f1))
    threshold = float(thresholds[min(best, max(0, thresholds.size - 1))]) if thresholds.size else 0.0
    otsu = float(otsu_threshold(score, valid))
    otsu_metrics = binary_metrics(score >= otsu, labels, valid)
    auroc = float(roc_auc_score(y, s))
    inverse_auroc = float(roc_auc_score(y, -s))
    return {
        "auroc": auroc,
        "average_precision": float(average_precision_score(y, s)),
        # Diagnostics only. A method may not use labels to choose its score
        # polarity at inference time. These columns expose datasets where the
        # conventional "larger distance = more change" assumption is reversed.
        "inverse_auroc_diagnostic": inverse_auroc,
        "inverse_average_precision_diagnostic": float(average_precision_score(y, -s)),
        "direction_free_auroc_diagnostic": max(auroc, inverse_auroc),
        "best_f1": float(f1[best]),
        "best_iou": float(iou[best]),
        "best_threshold": threshold,
        "otsu_f1": float(otsu_metrics["f1"]),
        "otsu_iou": float(otsu_metrics["iou"]),
    }


def _interpolate_grid(values: Array, rows: Array, columns: Array, shape: tuple[int, int]) -> Array:
    interpolation = RegularGridInterpolator(
        (rows.astype(np.float64), columns.astype(np.float64)),
        values.astype(np.float64),
        bounds_error=False,
        fill_value=None,
    )
    rr, cc = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    return interpolation(np.column_stack([rr.ravel(), cc.ravel()])).reshape(shape).astype(np.float32)


def _global_pca_projection(pair: HSIChangePair, rank: int, seed: int) -> tuple[Array, Array]:
    first = np.moveaxis(pair.first, 0, -1).reshape(-1, pair.first.shape[0])
    second = np.moveaxis(pair.second, 0, -1).reshape(-1, pair.second.shape[0])
    joined = np.concatenate([first, second], axis=0)
    rng = np.random.default_rng(int(seed))
    if joined.shape[0] > 40_000:
        joined = joined[rng.choice(joined.shape[0], 40_000, replace=False)]
    model = PCA(
        n_components=min(int(rank), joined.shape[1]),
        svd_solver="randomized",
        random_state=int(seed),
    ).fit(joined)
    shape = pair.labels.shape
    return (
        model.transform(first).reshape(*shape, -1).astype(np.float32),
        model.transform(second).reshape(*shape, -1).astype(np.float32),
    )


def local_factor_maps(pair: HSIChangePair, config: Configuration, global_rank: int, seed: int) -> LocalResult:
    first = np.moveaxis(pair.first, 0, -1)
    second = np.moveaxis(pair.second, 0, -1)
    reduced_first, reduced_second = _global_pca_projection(pair, global_rank, seed)
    half = config.window // 2
    rows = np.arange(half, pair.labels.shape[0] - half, config.stride, dtype=np.int32)
    columns = np.arange(half, pair.labels.shape[1] - half, config.stride, dtype=np.int32)
    grids = {name: np.zeros((rows.size, columns.size), dtype=np.float32) for name in FACTOR_METHODS}
    orientation_attr = np.zeros((rows.size, columns.size, pair.first.shape[0]), dtype=np.float32)
    covariance_attr = np.zeros_like(orientation_attr)
    changed_fraction = np.zeros((rows.size, columns.size), dtype=np.float32)
    eigengap = np.zeros((rows.size, columns.size), dtype=np.float32)

    for row_index, row in enumerate(rows):
        row_slice = slice(row - half, row + half + 1)
        for column_index, column in enumerate(columns):
            column_slice = slice(column - half, column + half + 1)
            first_samples = first[row_slice, column_slice].reshape(-1, first.shape[-1])
            second_samples = second[row_slice, column_slice].reshape(-1, second.shape[-1])
            local_seed = int(seed + row * 1009 + column * 9176)
            geometry = factor_local_moments(
                first_samples, second_samples, rank=config.rank, seed=local_seed
            )
            stable = float(np.sqrt(max(geometry.first_eigengap * geometry.second_eigengap, 0.0)))
            center_difference = second[row, column] - first[row, column]
            if geometry.difference_basis.shape[1]:
                projected = geometry.difference_basis.T @ center_difference
                projected_magnitude = float(np.linalg.norm(projected))
            else:
                projected_magnitude = 0.0
            raw_magnitude = float(np.linalg.norm(center_difference))
            projected_fraction = (projected_magnitude / max(raw_magnitude, 1e-10)) ** 2
            residual_magnitude = float(
                np.sqrt(max(raw_magnitude**2 - projected_magnitude**2, 0.0))
            )
            values = {
                "local_mean": geometry.mean_distance,
                "log_covariance_scale": geometry.log_scale_change,
                "eigenspectrum_shape": geometry.eigenspectrum_hellinger,
                "eigenspace_orientation": geometry.orientation_chordal_normalized,
                "stability_gated_orientation": geometry.orientation_chordal_normalized * stable,
                "first_order_ds_magnitude": geometry.first_order_ds_magnitude,
                "local_ds_projection_magnitude": projected_magnitude,
                "local_ds_projection_ratio": projected_magnitude / max(raw_magnitude, 1e-10),
                "local_ds_projection_fraction": projected_fraction,
                "local_ds_residual_magnitude": residual_magnitude,
                "local_raw_magnitude": raw_magnitude,
                "local_inverse_raw_magnitude_diagnostic": 1.0 / max(raw_magnitude, 1e-10),
                "normalized_covariance_frobenius": geometry.normalized_covariance_frobenius,
                "normalized_covariance_bures": geometry.normalized_covariance_bures,
                "shrinkage_spd_logeuclidean": shrinkage_log_euclidean_distance(
                    reduced_first[row_slice, column_slice].reshape(-1, reduced_first.shape[-1]),
                    reduced_second[row_slice, column_slice].reshape(-1, reduced_second.shape[-1]),
                ),
            }
            for name, value in values.items():
                grids[name][row_index, column_index] = value
            orientation_attr[row_index, column_index] = geometry.orientation_attribution
            covariance_attr[row_index, column_index] = geometry.normalized_covariance_attribution
            changed_fraction[row_index, column_index] = float(
                np.mean(pair.labels[row_slice, column_slice])
            )
            eigengap[row_index, column_index] = stable

    maps = {
        name: normalize_score(_interpolate_grid(grid, rows, columns, pair.labels.shape), pair.valid_mask)
        for name, grid in grids.items()
        if name != "factor_rank_fusion"
    }
    maps["factor_rank_fusion"] = np.mean(
        [
            percentile_ranks(maps[name], pair.valid_mask)
            for name in (
                "local_mean",
                "log_covariance_scale",
                "eigenspectrum_shape",
                "stability_gated_orientation",
            )
        ],
        axis=0,
    ).astype(np.float32)
    return LocalResult(
        maps=maps,
        rows=rows,
        columns=columns,
        orientation_attribution=orientation_attr,
        covariance_attribution=covariance_attr,
        changed_fraction=changed_fraction,
        eigengap=eigengap,
    )


def rff_features(pair: HSIChangePair, count: int, seed: int) -> tuple[Array, Array, float]:
    first = np.moveaxis(pair.first, 0, -1).reshape(-1, pair.first.shape[0]).astype(np.float64)
    second = np.moveaxis(pair.second, 0, -1).reshape(-1, pair.second.shape[0]).astype(np.float64)
    joined = np.concatenate([first, second])
    rng = np.random.default_rng(int(seed))
    sample = joined[rng.choice(joined.shape[0], min(4096, joined.shape[0]), replace=False)]
    left = sample[rng.integers(0, sample.shape[0], size=2048)]
    right = sample[rng.integers(0, sample.shape[0], size=2048)]
    bandwidth = float(np.median(np.linalg.norm(left - right, axis=1)))
    bandwidth = max(bandwidth, 1e-6)
    weights = rng.normal(scale=1.0 / bandwidth, size=(joined.shape[1], int(count)))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=int(count))
    scale = np.sqrt(2.0 / max(1, int(count)))
    first_features = scale * np.cos(first @ weights + phase)
    second_features = scale * np.cos(second @ weights + phase)
    shape = pair.labels.shape
    return (
        first_features.reshape(*shape, -1).astype(np.float32),
        second_features.reshape(*shape, -1).astype(np.float32),
        bandwidth,
    )


def rff_mmd_map(first: Array, second: Array, window: int, valid: Array) -> Array:
    mean_first = np.stack(
        [uniform_filter(first[..., index], size=window, mode="reflect") for index in range(first.shape[-1])],
        axis=-1,
    )
    mean_second = np.stack(
        [uniform_filter(second[..., index], size=window, mode="reflect") for index in range(second.shape[-1])],
        axis=-1,
    )
    return normalize_score(np.linalg.norm(mean_second - mean_first, axis=-1), valid)


def nmf_abundance_score(pair: HSIChangePair, components: int, seed: int) -> Array:
    first = np.moveaxis(pair.first, 0, -1).reshape(-1, pair.first.shape[0]).astype(np.float64)
    second = np.moveaxis(pair.second, 0, -1).reshape(-1, pair.second.shape[0]).astype(np.float64)
    low = min(float(np.min(first)), float(np.min(second)))
    first = first - low + 1e-6
    second = second - low + 1e-6
    scale = max(float(np.percentile(np.concatenate([first.ravel(), second.ravel()]), 99.5)), 1e-6)
    first /= scale
    second /= scale
    joined = np.concatenate([first, second])
    rng = np.random.default_rng(int(seed))
    sample = joined[rng.choice(joined.shape[0], min(30_000, joined.shape[0]), replace=False)]
    model = MiniBatchNMF(
        n_components=min(int(components), pair.first.shape[0]),
        init="nndsvda",
        random_state=int(seed),
        batch_size=2048,
        max_iter=200,
    ).fit(sample)
    score = np.linalg.norm(model.transform(first) - model.transform(second), axis=1)
    return normalize_score(score.reshape(pair.labels.shape), pair.valid_mask)


def baseline_maps(pair: HSIChangePair, rff_count: int, seed: int) -> tuple[dict[str, Array], tuple[Array, Array], float]:
    raw = cva_score(pair.first, pair.second, pair.valid_mask)
    numerator = np.sum(pair.first * pair.second, axis=0)
    denominator = np.linalg.norm(pair.first, axis=0) * np.linalg.norm(pair.second, axis=0)
    sam = np.arccos(np.clip(numerator / np.maximum(denominator, 1e-12), -1.0, 1.0))
    difference = pair.second - pair.first
    band_variance = np.var(difference[:, pair.valid_mask], axis=1)
    selected = np.argsort(band_variance)[-min(12, difference.shape[0]) :]
    band_selected = np.linalg.norm(difference[selected], axis=0)
    pca_difference = pca_diff_score(
        pair.first,
        pair.second,
        pair.valid_mask,
        rank_S=min(12, pair.first.shape[0]),
        variance_threshold=None,
    )
    maps = {
        "raw_l2_cva": raw,
        "inverse_raw_l2_diagnostic": normalize_score(-raw, pair.valid_mask),
        "spectral_angle": normalize_score(sam, pair.valid_mask),
        "pca_difference": pca_difference,
        "inverse_pca_difference_diagnostic": normalize_score(
            -pca_difference, pair.valid_mask
        ),
        "ir_mad": ir_mad_score(pair.first, pair.second, pair.valid_mask, iters=8, random_state=seed),
        "chronochrome": chronochrome_score(pair.first, pair.second, pair.valid_mask, random_state=seed),
        "covariance_equalization": covariance_equalization_score(
            pair.first, pair.second, pair.valid_mask, random_state=seed
        ),
        "top_variance_band_selection": normalize_score(band_selected, pair.valid_mask),
        "nmf_abundance_change_proxy": nmf_abundance_score(pair, components=8, seed=seed),
    }
    first_rff, second_rff, bandwidth = rff_features(pair, rff_count, seed)
    return maps, (first_rff, second_rff), bandwidth


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_maps(pair: HSIChangePair, maps: dict[str, Array], path: Path, title: str) -> None:
    names = list(maps)
    columns = 4
    rows = int(np.ceil((len(names) + 1) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.6 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    axes[0].imshow(pair.labels, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("reference change map")
    for axis, name in zip(axes[1:], names):
        image = axis.imshow(maps[name], cmap="magma", vmin=0, vmax=1)
        axis.set_title(name.replace("_", " "), fontsize=9)
        figure.colorbar(image, ax=axis, fraction=0.046)
    for axis in axes[len(names) + 1 :]:
        axis.axis("off")
    for axis in axes[: len(names) + 1]:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def attribution_rows(pair: HSIChangePair, result: LocalResult) -> list[dict[str, object]]:
    changed = result.changed_fraction >= 0.5
    unchanged = result.changed_fraction <= 0.05
    rows: list[dict[str, object]] = []
    for index, sensor_band in enumerate(pair.original_band_indices):
        direct = np.abs(pair.second[index] - pair.first[index])
        direct_changed = float(np.mean(direct[pair.valid_mask & (pair.labels == 1)]))
        direct_unchanged = float(np.mean(direct[pair.valid_mask & (pair.labels == 0)]))
        direct_ap = float(
            average_precision_score(pair.labels[pair.valid_mask], direct[pair.valid_mask])
        )
        orientation_changed = float(np.mean(result.orientation_attribution[..., index][changed]))
        orientation_unchanged = float(np.mean(result.orientation_attribution[..., index][unchanged]))
        covariance_changed = float(np.mean(result.covariance_attribution[..., index][changed]))
        covariance_unchanged = float(np.mean(result.covariance_attribution[..., index][unchanged]))
        rows.append({
            "retained_band_index": index + 1,
            "original_sensor_band": int(sensor_band),
            "orientation_changed": orientation_changed,
            "orientation_unchanged": orientation_unchanged,
            "orientation_delta": orientation_changed - orientation_unchanged,
            "covariance_changed": covariance_changed,
            "covariance_unchanged": covariance_unchanged,
            "covariance_delta": covariance_changed - covariance_unchanged,
            "direct_absolute_difference_changed": direct_changed,
            "direct_absolute_difference_unchanged": direct_unchanged,
            "direct_absolute_difference_delta": direct_changed - direct_unchanged,
            "direct_absolute_difference_ap": direct_ap,
        })
    return rows


def top_contiguous_intervals(
    rows: list[dict[str, object]],
    field: str,
    width: int = 5,
    count: int = 6,
) -> list[dict[str, object]]:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    bands = np.asarray([int(row["original_sensor_band"]) for row in rows], dtype=np.int32)
    candidates: list[tuple[float, int, int]] = []
    for start in range(0, max(0, values.size - int(width) + 1)):
        end = start + int(width)
        # Do not call a window contiguous when original Hyperion indices cross
        # one of the documented removed/noisy-band gaps.
        if np.any(np.diff(bands[start:end]) != 1):
            continue
        candidates.append((float(np.mean(values[start:end])), start, end))
    selected: list[dict[str, object]] = []
    occupied: set[int] = set()
    for score, start, end in sorted(candidates, reverse=True):
        if any(index in occupied for index in range(start, end)):
            continue
        occupied.update(range(start, end))
        selected.append({
            "field": field,
            "first_original_band": int(bands[start]),
            "last_original_band": int(bands[end - 1]),
            "mean_score": score,
        })
        if len(selected) >= int(count):
            break
    return selected


def plot_attribution(rows: list[dict[str, object]], path: Path, title: str) -> None:
    bands = np.asarray([row["original_sensor_band"] for row in rows])
    orientation = np.asarray([row["orientation_delta"] for row in rows])
    covariance = np.asarray([row["covariance_delta"] for row in rows])
    figure, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True, constrained_layout=True)
    axes[0].plot(bands, orientation, color="#d55e00", linewidth=1.2)
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_ylabel("orientation delta")
    axes[1].plot(bands, covariance, color="#0072b2", linewidth=1.2)
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_ylabel("covariance delta")
    axes[1].set_xlabel("original Hyperion band index")
    figure.suptitle(title)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def block_bootstrap_delta(
    labels: Array,
    first: Array,
    second: Array,
    valid: Array,
    repeats: int,
    seed: int,
    block: int = 16,
) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    row_starts = list(range(0, labels.shape[0], block))
    col_starts = list(range(0, labels.shape[1], block))
    blocks = [(row, col) for row in row_starts for col in col_starts]
    deltas: list[float] = []
    for _ in range(int(repeats)):
        sampled_labels: list[Array] = []
        sampled_first: list[Array] = []
        sampled_second: list[Array] = []
        for index in rng.integers(0, len(blocks), size=len(blocks)):
            row, col = blocks[int(index)]
            selection = np.s_[row : row + block, col : col + block]
            mask = valid[selection]
            sampled_labels.append(labels[selection][mask])
            sampled_first.append(first[selection][mask])
            sampled_second.append(second[selection][mask])
        y = np.concatenate(sampled_labels)
        if np.unique(y).size < 2:
            continue
        deltas.append(
            float(average_precision_score(y, np.concatenate(sampled_first)))
            - float(average_precision_score(y, np.concatenate(sampled_second)))
        )
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "repeats": int(values.size),
    }


def map_stability(reference: Array, candidate: Array, valid: Array) -> dict[str, float]:
    """Measure deterministic-map agreement without using ground-truth labels."""
    first = reference[valid].astype(np.float64)
    second = candidate[valid].astype(np.float64)
    if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(first, second)[0, 1])
    return {
        "pearson": pearson,
        "spearman": float(spearmanr(first, second).statistic),
        "mean_absolute_error": float(np.mean(np.abs(first - second))),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    configs = parse_configs(args.configs)
    if args.smoke:
        configs = configs[:1]
    pair_cache: dict[tuple[str, str], HSIChangePair] = {}
    baseline_cache: dict[tuple[str, str], tuple[dict[str, Array], tuple[Array, Array], float]] = {}
    local_cache: dict[tuple[str, str], LocalResult] = {}
    metric_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []

    for dataset in datasets:
        for mode in sorted({config.preprocessing for config in configs}):
            pair = load_hsi_change_pair(
                dataset,
                args.data_root,
                mode,
                band_policy=args.band_policy,
                seed=args.seed,
            )
            pair_cache[(dataset, mode)] = pair
            start = time.perf_counter()
            baseline_cache[(dataset, mode)] = baseline_maps(pair, args.rff_features, args.seed)
            runtime_rows.append({
                "dataset": dataset,
                "configuration": mode,
                "stage": "shared_baselines",
                "seconds": time.perf_counter() - start,
            })
            baseline, _, bandwidth = baseline_cache[(dataset, mode)]
            for method, score in baseline.items():
                metric_rows.append({
                    "dataset": dataset,
                    "configuration": mode,
                    "method": method,
                    **score_metrics(score, pair.labels, pair.valid_mask),
                    "rff_bandwidth": bandwidth,
                })

        for config in configs:
            pair = pair_cache[(dataset, config.preprocessing)]
            start = time.perf_counter()
            local = local_factor_maps(pair, config, args.global_pca_rank, args.seed)
            local_cache[(dataset, config.name)] = local
            first_rff, second_rff = baseline_cache[(dataset, config.preprocessing)][1]
            local.maps["rff_mmd"] = rff_mmd_map(
                first_rff, second_rff, config.window, pair.valid_mask
            )
            runtime_rows.append({
                "dataset": dataset,
                "configuration": config.name,
                "stage": "local_geometry",
                "seconds": time.perf_counter() - start,
            })
            for method, score in local.maps.items():
                metric_rows.append({
                    "dataset": dataset,
                    "configuration": config.name,
                    "method": method,
                    **score_metrics(score, pair.labels, pair.valid_mask),
                    "rff_bandwidth": baseline_cache[(dataset, config.preprocessing)][2],
                })

    development = datasets[0]
    transfer = datasets[1] if len(datasets) > 1 else datasets[0]

    def select_configuration(method: str) -> Configuration:
        candidates = [
            row for row in metric_rows
            if row["dataset"] == development
            and row["method"] == method
            and str(row["configuration"]).startswith(("robust_", "perdate_"))
        ]
        selected = max(candidates, key=lambda row: float(row["average_precision"]))
        return next(config for config in configs if config.name == selected["configuration"])

    selected_orientation_config = select_configuration("stability_gated_orientation")
    selected_ds_config = select_configuration("local_ds_projection_magnitude")
    selected_ratio_config = select_configuration("local_ds_projection_ratio")
    selected_fusion_config = select_configuration("factor_rank_fusion")
    selected_maps: dict[str, dict[str, Array]] = {}
    attribution_summary: dict[str, list[dict[str, object]]] = {}
    for dataset in datasets:
        pair = pair_cache[(dataset, selected_orientation_config.preprocessing)]
        local = local_cache[(dataset, selected_orientation_config.name)]
        ds_local = local_cache[(dataset, selected_ds_config.name)]
        ratio_local = local_cache[(dataset, selected_ratio_config.name)]
        fusion_local = local_cache[(dataset, selected_fusion_config.name)]
        baseline = baseline_cache[(dataset, selected_orientation_config.preprocessing)][0]
        combined = {**baseline, **local.maps}
        for method in ("first_order_ds_magnitude", "local_ds_projection_magnitude"):
            combined[method] = ds_local.maps[method]
        for method in (
            "local_ds_projection_ratio",
            "local_ds_projection_fraction",
            "local_inverse_raw_magnitude_diagnostic",
        ):
            combined[method] = ratio_local.maps[method]
        combined["factor_rank_fusion"] = fusion_local.maps["factor_rank_fusion"]
        selected_maps[dataset] = combined
        plot_maps(
            pair,
            combined,
            args.output_dir / f"{dataset}_selected_score_maps.png",
            f"{dataset}: orientation={selected_orientation_config.name}; "
            f"DS={selected_ds_config.name}; ratio={selected_ratio_config.name}",
        )
        rows = attribution_rows(pair, local)
        attribution_summary[dataset] = rows
        _write_csv(args.output_dir / f"{dataset}_band_attribution.csv", rows)
        plot_attribution(
            rows,
            args.output_dir / f"{dataset}_band_attribution.png",
            f"{dataset}: changed-minus-unchanged band contribution",
        )
        np.savez_compressed(
            args.output_dir / f"{dataset}_selected_scores.npz",
            labels=pair.labels,
            valid_mask=pair.valid_mask,
            **combined,
        )

    baseline_names = {
        name
        for name in next(iter(baseline_cache.values()))[0]
        if not name.endswith("_diagnostic")
    }
    holdout_comparisons: dict[str, dict[str, object]] = {}
    polarity_stress_tests: dict[str, dict[str, object]] = {}
    for holdout_index, holdout in enumerate(datasets[1:] or datasets):
        pair = pair_cache[(holdout, selected_orientation_config.preprocessing)]
        best_control_row = max(
            (
                row for row in metric_rows
                if row["dataset"] == holdout and row["method"] in baseline_names
            ),
            key=lambda row: float(row["average_precision"]),
        )
        best_control_name = str(best_control_row["method"])
        best_control_mode = str(best_control_row["configuration"])
        control_score = baseline_cache[(holdout, best_control_mode)][0][best_control_name]
        orientation_score = local_cache[
            (holdout, selected_orientation_config.name)
        ].maps["stability_gated_orientation"]
        ds_score = local_cache[
            (holdout, selected_ds_config.name)
        ].maps["local_ds_projection_magnitude"]
        fusion_score = local_cache[
            (holdout, selected_fusion_config.name)
        ].maps["factor_rank_fusion"]
        seed_offset = 100 * holdout_index
        holdout_comparisons[holdout] = {
            "best_control": best_control_name,
            "best_control_preprocessing": best_control_mode,
            "best_control_average_precision": float(best_control_row["average_precision"]),
            "orientation_average_precision": score_metrics(
                orientation_score, pair.labels, pair.valid_mask
            )["average_precision"],
            "local_ds_projection_average_precision": score_metrics(
                ds_score, pair.labels, pair.valid_mask
            )["average_precision"],
            "factor_fusion_average_precision": score_metrics(
                fusion_score, pair.labels, pair.valid_mask
            )["average_precision"],
            "orientation_vs_best_control": block_bootstrap_delta(
                pair.labels, orientation_score, control_score, pair.valid_mask,
                args.bootstrap, args.seed + 41 + seed_offset,
            ),
            "local_ds_projection_vs_best_control": block_bootstrap_delta(
                pair.labels, ds_score, control_score, pair.valid_mask,
                args.bootstrap, args.seed + 42 + seed_offset,
            ),
            "factor_fusion_vs_best_control": block_bootstrap_delta(
                pair.labels, fusion_score, control_score, pair.valid_mask,
                args.bootstrap, args.seed + 43 + seed_offset,
            ),
        }

    # This is an intentionally adversarial diagnostic. It asks whether an
    # apparently strong projection-ratio result is merely a disguised inverse
    # raw-distance score. The dataset-specific ratio configuration is an
    # optimistic oracle and must never be reported as held-out model selection.
    for dataset_index, dataset in enumerate(datasets):
        ratio_candidates = [
            row
            for row in metric_rows
            if row["dataset"] == dataset and row["method"] == "local_ds_projection_ratio"
        ]
        best_ratio_row = max(ratio_candidates, key=lambda row: float(row["average_precision"]))
        best_ratio_config = str(best_ratio_row["configuration"])
        ratio_score = local_cache[(dataset, best_ratio_config)].maps[
            "local_ds_projection_ratio"
        ]
        ratio_pair = pair_cache[
            (dataset, next(config.preprocessing for config in configs if config.name == best_ratio_config))
        ]
        inverse_raw = baseline_cache[(dataset, next(
            config.preprocessing for config in configs if config.name == best_ratio_config
        ))][0]["inverse_raw_l2_diagnostic"]
        local_inverse = local_cache[(dataset, best_ratio_config)].maps[
            "local_inverse_raw_magnitude_diagnostic"
        ]
        polarity_stress_tests[dataset] = {
            "warning": "dataset-specific oracle configuration; diagnostic only",
            "best_ratio_configuration": best_ratio_config,
            "best_ratio_metrics": score_metrics(
                ratio_score, ratio_pair.labels, ratio_pair.valid_mask
            ),
            "inverse_raw_l2_metrics": score_metrics(
                inverse_raw, ratio_pair.labels, ratio_pair.valid_mask
            ),
            "local_inverse_raw_metrics": score_metrics(
                local_inverse, ratio_pair.labels, ratio_pair.valid_mask
            ),
            "ratio_vs_inverse_raw_l2": block_bootstrap_delta(
                ratio_pair.labels,
                ratio_score,
                inverse_raw,
                ratio_pair.valid_mask,
                args.bootstrap,
                args.seed + 700 + dataset_index,
            ),
            "ratio_vs_local_inverse_raw": block_bootstrap_delta(
                ratio_pair.labels,
                ratio_score,
                local_inverse,
                ratio_pair.valid_mask,
                args.bootstrap,
                args.seed + 800 + dataset_index,
            ),
        }

    alternate_seeds = [
        int(item.strip())
        for item in str(args.stability_seeds).split(",")
        if item.strip()
    ]
    seed_stability: dict[str, dict[str, object]] = {}
    if alternate_seeds:
        selected_hypotheses = {
            "stability_gated_orientation": selected_orientation_config,
            "local_ds_projection_magnitude": selected_ds_config,
            "local_ds_projection_ratio": selected_ratio_config,
            "factor_rank_fusion": selected_fusion_config,
        }
        for dataset in datasets:
            dataset_rows: dict[str, object] = {}
            for alternate_seed in alternate_seeds:
                seed_rows: dict[str, object] = {}
                alternate_cache: dict[str, LocalResult] = {}
                for method, config in selected_hypotheses.items():
                    if config.name not in alternate_cache:
                        alternate_pair = load_hsi_change_pair(
                            dataset,
                            args.data_root,
                            config.preprocessing,
                            band_policy=args.band_policy,
                            seed=alternate_seed,
                        )
                        alternate_cache[config.name] = local_factor_maps(
                            alternate_pair,
                            config,
                            args.global_pca_rank,
                            alternate_seed,
                        )
                    reference_pair = pair_cache[(dataset, config.preprocessing)]
                    reference_map = local_cache[(dataset, config.name)].maps[method]
                    candidate_map = alternate_cache[config.name].maps[method]
                    seed_rows[method] = map_stability(
                        reference_map, candidate_map, reference_pair.valid_mask
                    )
                attribution_config = selected_orientation_config
                reference_attribution = np.mean(
                    local_cache[(dataset, attribution_config.name)].orientation_attribution,
                    axis=(0, 1),
                )
                candidate_attribution = np.mean(
                    alternate_cache[attribution_config.name].orientation_attribution,
                    axis=(0, 1),
                )
                seed_rows["mean_band_attribution"] = {
                    "pearson": float(
                        np.corrcoef(reference_attribution, candidate_attribution)[0, 1]
                    ),
                    "spearman": float(
                        spearmanr(reference_attribution, candidate_attribution).statistic
                    ),
                    "mean_absolute_error": float(
                        np.mean(np.abs(reference_attribution - candidate_attribution))
                    ),
                }
                dataset_rows[str(alternate_seed)] = seed_rows
            seed_stability[dataset] = dataset_rows
    primary_comparison = holdout_comparisons[transfer]
    _write_csv(args.output_dir / "metrics.csv", metric_rows)
    _write_csv(args.output_dir / "runtime.csv", runtime_rows)
    summary = {
        "development_dataset": development,
        "transfer_dataset": transfer,
        "selected_orientation_configuration": selected_orientation_config.name,
        "selected_local_ds_configuration": selected_ds_config.name,
        "selected_local_ds_ratio_configuration": selected_ratio_config.name,
        "selected_factor_fusion_configuration": selected_fusion_config.name,
        "selection_metric": "development average_precision, separately per hypothesis",
        "band_policy": args.band_policy,
        "holdout_comparisons": holdout_comparisons,
        "polarity_stress_tests": polarity_stress_tests,
        "seed_stability": seed_stability,
        "datasets": {
            name: {
                "shape": list(pair_cache[(name, selected_orientation_config.preprocessing)].first.shape),
                "changed_pixels": int(np.sum(
                    pair_cache[(name, selected_orientation_config.preprocessing)].labels[
                        pair_cache[(name, selected_orientation_config.preprocessing)].valid_mask
                    ]
                )),
                "valid_pixels": int(np.sum(
                    pair_cache[(name, selected_orientation_config.preprocessing)].valid_mask
                )),
                "retained_original_bands": pair_cache[(name, selected_orientation_config.preprocessing)].original_band_indices.tolist(),
                "source": pair_cache[(name, selected_orientation_config.preprocessing)].source_url,
            }
            for name in datasets
        },
        "top_original_band_intervals": {
            dataset: {
                "orientation": top_contiguous_intervals(
                    rows, "orientation_delta"
                ),
                "normalized_covariance": top_contiguous_intervals(
                    rows, "covariance_delta"
                ),
                "direct_absolute_difference": top_contiguous_intervals(
                    rows, "direct_absolute_difference_delta"
                ),
            }
            for dataset, rows in attribution_summary.items()
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    top = sorted(
        [row for row in metric_rows if row["dataset"] == transfer],
        key=lambda row: float(row["average_precision"]),
        reverse=True,
    )[:15]
    lines = [
        "# HSI Moment-Factorization Experiment",
        "",
        f"- Development dataset: `{development}`",
        f"- Frozen transfer dataset: `{transfer}`",
        f"- Selected orientation configuration: `{selected_orientation_config.name}`",
        f"- Selected local DS configuration: `{selected_ds_config.name}`",
        f"- Selected local DS ratio configuration: `{selected_ratio_config.name}`",
        f"- Selected factor-fusion configuration: `{selected_fusion_config.name}`",
        f"- Best transfer baseline: `{primary_comparison['best_control']}`",
        "",
        "## Transfer Ranking",
        "",
        "| method | configuration | AP | AUROC | best F1 | Otsu F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['method']} | {row['configuration']} | {row['average_precision']:.4f} | "
        f"{row['auroc']:.4f} | {row['best_f1']:.4f} | {row['otsu_f1']:.4f} |"
        for row in top
    )
    lines.extend([
        "",
        "## Mandatory Decision Gate",
        "",
        f"- Orientation minus best-control AP: {primary_comparison['orientation_vs_best_control']}",
        f"- Local DS projection minus best-control AP: {primary_comparison['local_ds_projection_vs_best_control']}",
        f"- Factor-fusion minus best-control AP: {primary_comparison['factor_fusion_vs_best_control']}",
        "- A positive HSI claim is not allowed if the block-bootstrap interval overlaps or lies below zero.",
        "- Inverse-score columns and controls are diagnostics only: labels cannot choose score polarity in an unsupervised deployment.",
        "- A projection-ratio claim is rejected when inverse raw distance matches or exceeds it.",
        "- Band plots show original sensor-band indices, not calibrated wavelengths.",
    ])
    (args.output_dir / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
