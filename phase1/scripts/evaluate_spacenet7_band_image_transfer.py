"""Evaluate tiled Band-Image geometry on SpaceNet 7 building appearances.

Question
--------
Does the candidate-localization behavior observed on xBD-S12 survive a change
of sensor and task to monthly RGB imagery with persistent building IDs?

Source/provenance
-----------------
- SpaceNet 7 imagery and persistent building labels: Van Etten et al. 2021,
  loaded by ``phase1.data.spacenet7_dataset``.
- Canonical DS: Fukui and Maki, TPAMI 2015, with the project's Band-Image
  sample adaptation from ``phase1.subspace.band_image_geometry``.
- IR-MAD: Nielsen 2007 through ``phase1.baselines.ir_mad``.

Project adaptation
------------------
Each monthly pair is divided into non-overlapping 128x128 tiles to match the
xBD-S12 spatial support. RGB provides only three band-image samples, so
centered PCA is capped at rank two. First appearance of a persistent building
ID is treated as the transition target. This is not the SpaceNet SCOT metric.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from phase1.baselines.ir_mad import ir_mad_score
from phase1.baselines.pca_diff import pca_diff_score
from phase1.data.spacenet7_dataset import (
    first_appearance_features,
    load_spacenet7_image,
    rasterize_feature_mask,
    read_geojson_features,
    scan_spacenet7_aoi,
)
from phase1.scripts.evaluate_xbd_s12_external import (
    score_metrics,
    spectral_angle_score,
)
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
COMPARISONS = (
    ("band_image_projector_distance", "ir_mad"),
    ("band_image_projector_distance", "smoothed_pca_sigma1"),
    ("band_image_ds", "band_image_cross_reconstruction"),
    ("band_image_ds", "smoothed_pca_sigma1"),
)


@dataclass(frozen=True)
class AOIResult:
    aoi: str
    rows: list[dict[str, object]]
    dates: int
    evaluated_transitions: int
    skipped_transitions: int
    runtime_seconds: float
    representative_path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tiled SpaceNet7 Band-Image transfer evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-roots",
        default="data/SpaceNet7_sample,data/SpaceNet7_validation,data/SpaceNet7_confirmation",
    )
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--ir-mad-iters", type=int, default=10)
    parser.add_argument("--minimum-valid-pixels", type=int, default=256)
    parser.add_argument("--minimum-positive-pixels", type=int, default=2)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--maximum-aois", type=int, default=None)
    parser.add_argument("--maximum-transitions", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def discover_aois(data_roots: str, maximum: int | None = None) -> list[Path]:
    roots = [Path(item.strip()) for item in data_roots.split(",") if item.strip()]
    aois = sorted(
        [child for root in roots for child in root.iterdir() if child.is_dir()],
        key=lambda path: path.name,
    )
    return aois[:maximum] if maximum is not None else aois


def _place(values: Array, valid: Array) -> Array:
    output = np.zeros(valid.shape, dtype=np.float32)
    output[valid] = values.astype(np.float32, copy=False)
    return output


def tile_score_maps(
    first: Array,
    second: Array,
    valid: Array,
    *,
    rank: int,
    seed: int,
    ir_mad_iters: int,
) -> dict[str, Array]:
    """Compute the frozen method set on one RGB tile."""
    difference = second - first
    raw = np.sqrt(np.sum(difference * difference, axis=0)).astype(np.float32)
    raw[~valid] = 0.0
    pca = pca_diff_score(
        first,
        second,
        valid,
        rank_S=min(int(rank), first.shape[0]),
        variance_threshold=None,
        random_state=int(seed),
    )
    smooth = gaussian_filter(pca, sigma=1.0, mode="nearest").astype(np.float32)
    smooth[~valid] = 0.0
    irmad = ir_mad_score(
        first,
        second,
        valid,
        iters=int(ir_mad_iters),
        downsample_max_pixels=200_000,
        random_state=int(seed),
    )
    first_matrix = first[:, valid].T.astype(np.float32, copy=False)
    second_matrix = second[:, valid].T.astype(np.float32, copy=False)
    ds = band_image_ds_values(
        first_matrix,
        second_matrix,
        rank=int(rank),
        seed=int(seed),
    )
    return {
        "raw_l2": raw,
        "spectral_angle": spectral_angle_score(first, second, valid),
        "pca_diff": pca,
        "smoothed_pca_sigma1": smooth,
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


def tiled_score_maps(
    first: Array,
    second: Array,
    valid: Array,
    *,
    tile_size: int,
    minimum_valid_pixels: int,
    rank: int,
    seed: int,
    ir_mad_iters: int,
) -> tuple[dict[str, Array], Array]:
    """Assemble independently fitted 128x128 score tiles into one scene map."""
    output = {
        name: np.zeros(valid.shape, dtype=np.float32) for name in METHODS
    }
    scored = np.zeros(valid.shape, dtype=bool)
    for row in range(0, valid.shape[0], int(tile_size)):
        for column in range(0, valid.shape[1], int(tile_size)):
            selection = np.s_[
                row : min(row + tile_size, valid.shape[0]),
                column : min(column + tile_size, valid.shape[1]),
            ]
            tile_valid = valid[selection]
            if int(np.sum(tile_valid)) < int(minimum_valid_pixels):
                continue
            local_seed = int(seed + 1009 * row + 9176 * column)
            maps = tile_score_maps(
                first[:, selection[0], selection[1]],
                second[:, selection[0], selection[1]],
                tile_valid,
                rank=rank,
                seed=local_seed,
                ir_mad_iters=ir_mad_iters,
            )
            for name in METHODS:
                output[name][selection] = maps[name]
            scored[selection] = tile_valid
    return output, scored


def _normalize_display(score: Array, valid: Array) -> Array:
    output = np.zeros(score.shape, dtype=np.float32)
    low, high = np.percentile(score[valid], (1.0, 99.0))
    output[valid] = np.clip(
        (score[valid] - low) / max(high - low, 1e-12), 0.0, 1.0
    )
    return output


def save_representative(
    path: Path,
    *,
    first: Array,
    second: Array,
    target: Array,
    valid: Array,
    maps: dict[str, Array],
    date: str,
) -> None:
    np.savez_compressed(
        path,
        first=first,
        second=second,
        target=target,
        valid=valid,
        date=np.asarray(date),
        **maps,
    )


def process_aoi(aoi_root: Path, args: argparse.Namespace) -> AOIResult:
    started = time.perf_counter()
    observations = scan_spacenet7_aoi(aoi_root)
    loaded = [load_spacenet7_image(observation) for observation in observations]
    shape = loaded[0].valid_mask.shape
    if any(item.valid_mask.shape != shape for item in loaded):
        raise ValueError(f"SpaceNet7 shape mismatch under {aoi_root}.")
    features = [
        read_geojson_features(observation.matched_label_path)
        for observation in observations
    ]
    new_features = first_appearance_features(features)
    new_masks = [
        rasterize_feature_mask(
            current,
            out_shape=shape,
            transform=Affine.identity(),
            all_touched=True,
        )
        for current in new_features
    ]
    transition_indices = list(range(1, len(observations)))
    if args.maximum_transitions is not None:
        transition_indices = transition_indices[: int(args.maximum_transitions)]
    representative_index = max(
        transition_indices,
        key=lambda index: int(np.sum(new_masks[index])),
        default=None,
    )
    rows: list[dict[str, object]] = []
    skipped = 0
    representative_path: str | None = None
    for index in transition_indices:
        valid = loaded[index - 1].valid_mask & loaded[index].valid_mask
        target = new_masks[index] & valid
        if int(np.sum(target)) < int(args.minimum_positive_pixels):
            skipped += 1
            continue
        maps, scored = tiled_score_maps(
            loaded[index - 1].rgb,
            loaded[index].rgb,
            valid,
            tile_size=args.tile_size,
            minimum_valid_pixels=args.minimum_valid_pixels,
            rank=args.rank,
            seed=args.seed,
            ir_mad_iters=args.ir_mad_iters,
        )
        support = valid & scored
        if int(np.sum(target & support)) < int(args.minimum_positive_pixels):
            skipped += 1
            continue
        for method in METHODS:
            metrics = score_metrics(maps[method], target, support)
            rows.append(
                {
                    "aoi": aoi_root.name,
                    "date": observations[index].date,
                    "previous_date": observations[index - 1].date,
                    "method": method,
                    "valid_pixels": int(np.sum(support)),
                    "positive_pixels": int(np.sum(target & support)),
                    **metrics,
                }
            )
        if index == representative_index:
            representative = args.output_dir / f"{aoi_root.name}_representative.npz"
            save_representative(
                representative,
                first=loaded[index - 1].rgb,
                second=loaded[index].rgb,
                target=target,
                valid=support,
                maps=maps,
                date=observations[index].date,
            )
            representative_path = str(representative)
    return AOIResult(
        aoi=aoi_root.name,
        rows=rows,
        dates=len(observations),
        evaluated_transitions=len({row["date"] for row in rows}),
        skipped_transitions=skipped,
        runtime_seconds=time.perf_counter() - started,
        representative_path=representative_path,
    )


def paired_transition_deltas(
    rows: list[dict[str, object]],
) -> dict[str, dict[tuple[str, str], float]]:
    lookup = {
        (str(row["aoi"]), str(row["date"]), str(row["method"])): float(
            row["average_precision"]
        )
        for row in rows
        if math.isfinite(float(row["average_precision"]))
    }
    output: dict[str, dict[tuple[str, str], float]] = {}
    for left, right in COMPARISONS:
        key = f"{left}_minus_{right}"
        output[key] = {}
        for aoi, date, method in lookup:
            if method != left or (aoi, date, right) not in lookup:
                continue
            output[key][(aoi, date)] = lookup[(aoi, date, left)] - lookup[
                (aoi, date, right)
            ]
    return output


def hierarchical_bootstrap(
    deltas: dict[tuple[str, str], float],
    *,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    by_aoi: dict[str, list[float]] = defaultdict(list)
    for (aoi, _date), value in deltas.items():
        by_aoi[aoi].append(float(value))
    aois = sorted(by_aoi)
    observed = float(np.mean(list(deltas.values()))) if deltas else float("nan")
    if not aois or repeats <= 0:
        return {"observed": observed, "ci_low": float("nan"), "ci_high": float("nan"), "repeats": 0}
    rng = np.random.default_rng(int(seed))
    samples: list[float] = []
    for _ in range(int(repeats)):
        values: list[float] = []
        for index in rng.integers(0, len(aois), size=len(aois)):
            source = np.asarray(by_aoi[aois[int(index)]], dtype=np.float64)
            selected = source[rng.integers(0, source.size, size=source.size)]
            values.extend(selected.tolist())
        samples.append(float(np.mean(values)))
    array = np.asarray(samples)
    return {
        "observed": observed,
        "ci_low": float(np.percentile(array, 2.5)),
        "ci_high": float(np.percentile(array, 97.5)),
        "repeats": int(array.size),
        "aoi_count": int(len(aois)),
        "transition_count": int(len(deltas)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_representative(path: Path, output: Path, title: str) -> None:
    data = np.load(path)
    names = (
        "smoothed_pca_sigma1",
        "ir_mad",
        "band_image_ds",
        "band_image_cross_reconstruction",
        "band_image_projector_distance",
    )
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(np.moveaxis(data["first"], 0, -1))
    axes[0].set_title("previous RGB")
    axes[1].imshow(np.moveaxis(data["second"], 0, -1))
    axes[1].set_title("current RGB")
    axes[2].imshow(data["target"], cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("new-building target")
    valid = data["valid"].astype(bool)
    for axis, name in zip(axes[3:], names):
        image = axis.imshow(_normalize_display(data[name], valid), cmap="magma")
        axis.set_title(name.replace("_", " "), fontsize=9)
        figure.colorbar(image, ax=axis, fraction=0.046)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title)
    figure.savefig(output, dpi=175)
    plt.close(figure)


def plot_aoi_summary(rows: list[dict[str, object]], path: Path) -> None:
    aois = sorted({str(row["aoi"]) for row in rows})
    selected = (
        "smoothed_pca_sigma1",
        "ir_mad",
        "band_image_ds",
        "band_image_cross_reconstruction",
        "band_image_projector_distance",
    )
    matrix = np.asarray(
        [
            [
                np.mean(
                    [
                        float(row["average_precision"])
                        for row in rows
                        if row["aoi"] == aoi
                        and row["method"] == method
                        and math.isfinite(float(row["average_precision"]))
                    ]
                )
                for method in selected
            ]
            for aoi in aois
        ]
    )
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    image = axis.imshow(matrix, cmap="viridis", aspect="auto")
    axis.set_yticks(range(len(aois)), aois, fontsize=8)
    axis.set_xticks(
        range(len(selected)), [name.replace("_", "\n") for name in selected]
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.4f}", ha="center", va="center", fontsize=7)
    axis.set_title("SpaceNet7 mean monthly AP by AOI")
    figure.colorbar(image, ax=axis, label="AP")
    figure.savefig(path, dpi=190)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aois = discover_aois(args.data_roots, args.maximum_aois)
    results: list[AOIResult] = []
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = {executor.submit(process_aoi, aoi, args): aoi for aoi in aois}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"[{result.aoi}] transitions={result.evaluated_transitions} "
                    f"runtime={result.runtime_seconds:.1f}s",
                    flush=True,
                )
    results.sort(key=lambda item: item.aoi)
    rows = [row for result in results for row in result.rows]
    write_csv(args.output_dir / "transition_metrics.csv", rows)

    deltas = paired_transition_deltas(rows)
    comparison_summary = {
        name: hierarchical_bootstrap(
            values,
            repeats=args.bootstrap,
            seed=args.seed + 100 * index,
        )
        for index, (name, values) in enumerate(deltas.items())
    }
    macro: dict[str, dict[str, float]] = {}
    for method in METHODS:
        selected = [
            row
            for row in rows
            if row["method"] == method
            and math.isfinite(float(row["average_precision"]))
        ]
        macro[method] = {
            "mean_transition_ap": float(
                np.mean([float(row["average_precision"]) for row in selected])
            ),
            "mean_transition_auroc": float(
                np.mean([float(row["auroc"]) for row in selected])
            ),
            "mean_top_5pct_recall": float(
                np.mean([float(row["top_5pct_recall"]) for row in selected])
            ),
            "mean_top_5pct_enrichment": float(
                np.mean([float(row["top_5pct_enrichment"]) for row in selected])
            ),
        }
    summary = {
        "evidence_status": "RGB building-appearance transfer; not Sentinel-2 or SCOT evaluation",
        "protocol": {
            "aoi_count": len(results),
            "tile_size": args.tile_size,
            "rank": args.rank,
            "basis_mode": "centered_pca",
            "workers": args.workers,
            "primary_metric": "transition average precision",
            "target": "first appearance of persistent SpaceNet7 building IDs",
        },
        "aoi_runs": [
            {
                "aoi": result.aoi,
                "dates": result.dates,
                "evaluated_transitions": result.evaluated_transitions,
                "skipped_transitions": result.skipped_transitions,
                "runtime_seconds": result.runtime_seconds,
            }
            for result in results
        ],
        "macro_metrics": macro,
        "hierarchical_bootstrap": comparison_summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_aoi_summary(rows, args.output_dir / "aoi_average_precision.png")
    for result in results:
        if result.representative_path:
            plot_representative(
                Path(result.representative_path),
                args.output_dir / f"{result.aoi}_representative.png",
                f"{result.aoi}: representative building-appearance transition",
            )
    print(json.dumps({"macro_metrics": macro, "comparisons": comparison_summary}, indent=2))


if __name__ == "__main__":
    main()
