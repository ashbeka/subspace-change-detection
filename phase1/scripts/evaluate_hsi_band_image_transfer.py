"""Pressure-test Band-Image subspace geometry on labeled bitemporal HSI.

Research question
-----------------
Does the spatial-axis Band-Image construction that produced useful candidate
localization on OSCD/xBD-S12 transfer, without label tuning, to bitemporal
hyperspectral change benchmarks with many more band-image samples?

Source/provenance
-----------------
- Dataset loading and label conventions come from ``phase1.data.hsi_change``.
- Canonical Difference Subspace follows Fukui and Maki, TPAMI 2015, through
  ``phase1.subspace.band_image_geometry``. Treating complete registered band
  images as samples is this project's satellite adaptation, not TPAMI theory.
- IR-MAD follows Nielsen 2007 through ``phase1.baselines.ir_mad``.
- PCA-diff is the project's classical PCA-on-difference control.

Protocol boundary
-----------------
The primary run fixes rank 11, centered PCA, joint robust normalization, and
all methods before scoring labels. These HSI scenes have appeared in other
project experiments, so this is a transfer pressure test rather than pristine
confirmatory evidence. No score polarity is reversed from labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from threadpoolctl import threadpool_limits

from phase1.baselines.ir_mad import ir_mad_score
from phase1.baselines.pca_diff import pca_diff_score
from phase1.data.hsi_change import HSIChangePair, load_hsi_change_pair
from phase1.eval.metrics import binary_metrics
from phase1.eval.thresholding import otsu_threshold
from phase1.subspace.band_image_geometry import (
    band_image_ds_values,
    band_image_spatial_control_values,
)


Array = np.ndarray
METHODS = (
    "raw_l2",
    "spectral_angle",
    "pca_diff",
    "smoothed_pca_sigma1",
    "ir_mad",
    "band_image_ds",
    "band_image_cross_reconstruction",
    "band_image_spatial_gram",
    "band_image_projector_distance",
)
PRIMARY_COMPARISONS = (
    ("band_image_projector_distance", "ir_mad"),
    ("band_image_projector_distance", "smoothed_pca_sigma1"),
    ("band_image_ds", "band_image_cross_reconstruction"),
    ("band_image_ds", "smoothed_pca_sigma1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen HSI Band-Image transfer pressure test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/HSI_change"))
    parser.add_argument(
        "--datasets", default="benton,hermiston,farmland,shenzhen"
    )
    parser.add_argument("--preprocessing", default="joint_robust_zscore")
    parser.add_argument(
        "--band-policy",
        choices=("nonconstant", "common_hyperion_159"),
        default="nonconstant",
    )
    parser.add_argument("--rank", type=int, default=11)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ir-mad-iters", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--bootstrap-block", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _place(values: Array, valid: Array) -> Array:
    output = np.zeros(valid.shape, dtype=np.float32)
    output[valid] = values.astype(np.float32, copy=False)
    return output


def spectral_angle_score(first: Array, second: Array, valid: Array) -> Array:
    dot = np.sum(first * second, axis=0)
    denominator = np.sqrt(
        np.sum(first * first, axis=0) * np.sum(second * second, axis=0)
    )
    cosine = np.divide(
        dot, denominator, out=np.ones_like(dot), where=denominator > 1e-10
    )
    output = np.arccos(np.clip(cosine, -1.0, 1.0)).astype(np.float32)
    output[~valid] = 0.0
    return output


def compute_score_maps(
    pair: HSIChangePair,
    *,
    rank: int,
    seed: int,
    ir_mad_iters: int,
) -> dict[str, Array]:
    """Compute frozen HSI maps using the shared Band-Image implementation."""
    first, second, valid = pair.first, pair.second, pair.valid_mask
    difference = second - first
    raw_l2 = np.sqrt(np.sum(difference * difference, axis=0)).astype(np.float32)
    raw_l2[~valid] = 0.0
    pca = pca_diff_score(
        first,
        second,
        valid,
        rank_S=min(int(rank), first.shape[0]),
        variance_threshold=None,
        random_state=int(seed),
    )
    smoothed = gaussian_filter(pca, sigma=1.0, mode="nearest").astype(np.float32)
    smoothed[~valid] = 0.0
    irmad = ir_mad_score(
        first,
        second,
        valid,
        iters=int(ir_mad_iters),
        downsample_max_pixels=100_000,
        random_state=int(seed),
    )

    first_matrix = first[:, valid].T.astype(np.float32, copy=False)
    second_matrix = second[:, valid].T.astype(np.float32, copy=False)
    ds = band_image_ds_values(
        first_matrix,
        second_matrix,
        rank=int(rank),
        seed=int(seed),
        basis_mode="centered_pca",
    )
    maps = {
        "raw_l2": raw_l2,
        "spectral_angle": spectral_angle_score(first, second, valid),
        "pca_diff": pca,
        "smoothed_pca_sigma1": smoothed,
        "ir_mad": irmad,
        "band_image_ds": _place(ds.projected_magnitude, valid),
        "band_image_cross_reconstruction": _place(
            band_image_spatial_control_values(
                first_matrix,
                second_matrix,
                rank=int(rank),
                seed=int(seed),
                mode="cross_reconstruction",
                first_basis=ds.pre_basis,
                second_basis=ds.post_basis,
            ),
            valid,
        ),
        "band_image_spatial_gram": _place(
            band_image_spatial_control_values(
                first_matrix,
                second_matrix,
                rank=int(rank),
                seed=int(seed),
                mode="spatial_gram",
            ),
            valid,
        ),
        "band_image_projector_distance": _place(
            band_image_spatial_control_values(
                first_matrix,
                second_matrix,
                rank=int(rank),
                seed=int(seed),
                mode="projector_distance",
                first_basis=ds.pre_basis,
                second_basis=ds.post_basis,
            ),
            valid,
        ),
    }
    return maps


def score_metrics(score: Array, labels: Array, valid: Array) -> dict[str, float]:
    y = labels[valid].astype(np.uint8)
    values = score[valid].astype(np.float64)
    precision, recall, thresholds = precision_recall_curve(y, values)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    iou = precision * recall / np.maximum(
        precision + recall - precision * recall, 1e-12
    )
    best = int(np.nanargmax(f1))
    threshold = (
        float(thresholds[min(best, thresholds.size - 1)]) if thresholds.size else 0.0
    )
    otsu = float(otsu_threshold(score, valid))
    otsu_result = binary_metrics(score >= otsu, labels, valid)
    return {
        "auroc": float(roc_auc_score(y, values)),
        "average_precision": float(average_precision_score(y, values)),
        "best_f1": float(f1[best]),
        "best_iou": float(iou[best]),
        "best_threshold": threshold,
        "otsu_f1": float(otsu_result["f1"]),
        "otsu_iou": float(otsu_result["iou"]),
    }


def spatial_block_bootstrap_deltas(
    labels: Array,
    maps: dict[str, Array],
    valid: Array,
    *,
    comparisons: tuple[tuple[str, str], ...] = PRIMARY_COMPARISONS,
    repeats: int,
    block: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Bootstrap spatial blocks once and evaluate all paired AP differences."""
    if repeats <= 0:
        return {}
    rng = np.random.default_rng(int(seed))
    blocks: list[Array] = []
    width = labels.shape[1]
    for row in range(0, labels.shape[0], int(block)):
        for column in range(0, labels.shape[1], int(block)):
            rr, cc = np.mgrid[
                row : min(row + block, labels.shape[0]),
                column : min(column + block, labels.shape[1]),
            ]
            flat = (rr * width + cc).reshape(-1)
            keep = valid.reshape(-1)[flat]
            if np.any(keep):
                blocks.append(flat[keep])
    if not blocks:
        return {}
    flat_labels = labels.reshape(-1)
    flat_maps = {name: score.reshape(-1) for name, score in maps.items()}
    samples = {f"{left}_minus_{right}": [] for left, right in comparisons}
    for _ in range(int(repeats)):
        selected = np.concatenate(
            [blocks[int(index)] for index in rng.integers(0, len(blocks), len(blocks))]
        )
        y = flat_labels[selected]
        if np.unique(y).size < 2:
            continue
        needed = {name for pair in comparisons for name in pair}
        aps = {
            name: float(average_precision_score(y, flat_maps[name][selected]))
            for name in needed
        }
        for left, right in comparisons:
            samples[f"{left}_minus_{right}"].append(aps[left] - aps[right])
    result: dict[str, dict[str, float]] = {}
    for name, values in samples.items():
        array = np.asarray(values, dtype=np.float64)
        result[name] = {
            "mean": float(np.mean(array)),
            "ci_low": float(np.percentile(array, 2.5)),
            "ci_high": float(np.percentile(array, 97.5)),
            "repeats": int(array.size),
        }
    return result


def _display_band(cube: Array, index: int, valid: Array) -> Array:
    values = cube[index].astype(np.float64)
    low, high = np.percentile(values[valid], (2.0, 98.0))
    return np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)


def pseudo_rgb(cube: Array, valid: Array) -> Array:
    bands = cube.shape[0]
    indices = [int(round(value * (bands - 1))) for value in (0.75, 0.5, 0.25)]
    return np.stack([_display_band(cube, index, valid) for index in indices], axis=-1)


def normalize_map(score: Array, valid: Array) -> Array:
    output = np.zeros(score.shape, dtype=np.float32)
    low, high = np.percentile(score[valid], (1.0, 99.0))
    output[valid] = np.clip(
        (score[valid] - low) / max(high - low, 1e-12), 0.0, 1.0
    )
    return output


def plot_maps(pair: HSIChangePair, maps: dict[str, Array], path: Path) -> None:
    shown = (
        "smoothed_pca_sigma1",
        "ir_mad",
        "band_image_ds",
        "band_image_cross_reconstruction",
        "band_image_projector_distance",
    )
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(pseudo_rgb(pair.first, pair.valid_mask))
    axes[0].set_title("pre pseudo-RGB")
    axes[1].imshow(pseudo_rgb(pair.second, pair.valid_mask))
    axes[1].set_title("post pseudo-RGB")
    axes[2].imshow(pair.labels, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("binary change label")
    for axis, method in zip(axes[3:], shown):
        image = axis.imshow(normalize_map(maps[method], pair.valid_mask), cmap="magma")
        axis.set_title(method.replace("_", " "), fontsize=9)
        figure.colorbar(image, ax=axis, fraction=0.046)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"{pair.name}: fixed rank-11 Band-Image transfer ({pair.first.shape[0]} bands)"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_ap_summary(metric_rows: list[dict[str, object]], path: Path) -> None:
    datasets = list(dict.fromkeys(str(row["dataset"]) for row in metric_rows))
    matrix = np.asarray(
        [
            [
                next(
                    float(row["average_precision"])
                    for row in metric_rows
                    if row["dataset"] == dataset and row["method"] == method
                )
                for method in METHODS
            ]
            for dataset in datasets
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    image = axis.imshow(matrix, cmap="viridis", aspect="auto")
    axis.set_yticks(range(len(datasets)), datasets)
    axis.set_xticks(
        range(len(METHODS)), [name.replace("_", "\n") for name in METHODS], rotation=35, ha="right"
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", fontsize=7,
                      color="white" if matrix[row, column] < np.median(matrix) else "black")
    axis.set_title("Average precision: fixed HSI Band-Image transfer")
    figure.colorbar(image, ax=axis, label="AP")
    figure.savefig(path, dpi=190)
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if args.smoke:
        datasets = datasets[:1]
        args.bootstrap = min(args.bootstrap, 20)

    metric_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    bootstrap: dict[str, object] = {}
    dataset_metadata: dict[str, object] = {}
    with threadpool_limits(limits=1):
        for dataset_index, dataset in enumerate(datasets):
            start = time.perf_counter()
            pair = load_hsi_change_pair(
                dataset,
                args.data_root,
                preprocessing=args.preprocessing,
                band_policy=args.band_policy,
                seed=args.seed,
            )
            load_seconds = time.perf_counter() - start
            start = time.perf_counter()
            maps = compute_score_maps(
                pair,
                rank=args.rank,
                seed=args.seed,
                ir_mad_iters=args.ir_mad_iters,
            )
            method_seconds = time.perf_counter() - start
            for method in METHODS:
                metric_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "bands": int(pair.first.shape[0]),
                        "rows": int(pair.labels.shape[0]),
                        "columns": int(pair.labels.shape[1]),
                        "valid_pixels": int(np.sum(pair.valid_mask)),
                        "changed_pixels": int(np.sum(pair.labels[pair.valid_mask])),
                        "prevalence": float(np.mean(pair.labels[pair.valid_mask])),
                        **score_metrics(maps[method], pair.labels, pair.valid_mask),
                    }
                )
            bootstrap[dataset] = spatial_block_bootstrap_deltas(
                pair.labels,
                maps,
                pair.valid_mask,
                repeats=args.bootstrap,
                block=args.bootstrap_block,
                seed=args.seed + 1000 * dataset_index,
            )
            runtime_rows.append(
                {
                    "dataset": dataset,
                    "load_seconds": load_seconds,
                    "all_methods_seconds": method_seconds,
                    "total_seconds": load_seconds + method_seconds,
                }
            )
            dataset_metadata[dataset] = {
                "source_url": pair.source_url,
                "shape_bands_rows_columns": list(pair.first.shape),
                "retained_bands": int(pair.first.shape[0]),
                "preprocessing": pair.preprocessing,
            }
            plot_maps(pair, maps, args.output_dir / f"{dataset}_score_maps.png")
            np.savez_compressed(
                args.output_dir / f"{dataset}_scores.npz",
                labels=pair.labels,
                valid_mask=pair.valid_mask,
                **maps,
            )
            print(
                f"[{dataset}] bands={pair.first.shape[0]} valid={np.sum(pair.valid_mask)} "
                f"methods={method_seconds:.1f}s",
                flush=True,
            )

    write_csv(args.output_dir / "metrics.csv", metric_rows)
    write_csv(args.output_dir / "runtime.csv", runtime_rows)
    plot_ap_summary(metric_rows, args.output_dir / "average_precision_summary.png")

    macro: dict[str, dict[str, float]] = {}
    for method in METHODS:
        rows = [row for row in metric_rows if row["method"] == method]
        macro[method] = {
            "mean_auroc": float(np.mean([float(row["auroc"]) for row in rows])),
            "mean_average_precision": float(
                np.mean([float(row["average_precision"]) for row in rows])
            ),
            "mean_best_f1": float(np.mean([float(row["best_f1"]) for row in rows])),
            "mean_otsu_f1": float(np.mean([float(row["otsu_f1"]) for row in rows])),
        }
    summary = {
        "evidence_status": "zero-tuning HSI transfer pressure test; not pristine confirmation",
        "protocol": {
            "datasets": datasets,
            "rank": int(args.rank),
            "basis_mode": "centered_pca",
            "preprocessing": args.preprocessing,
            "band_policy": args.band_policy,
            "score_polarity": "larger score means more change; never label-reversed",
            "primary_metric": "average_precision",
            "bootstrap_repeats": int(args.bootstrap),
            "bootstrap_block": int(args.bootstrap_block),
        },
        "datasets": dataset_metadata,
        "macro_metrics": macro,
        "spatial_block_bootstrap": bootstrap,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["macro_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
