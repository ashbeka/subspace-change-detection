"""Frozen external validation of Band-Image DS on xBD-S12.

Protocol
--------
The predeclared protocol is tracked in
``docs/experiment_reports/xbd_s12_frozen_external_protocol_2026-06-22.md``.
This script uses only the official event split and published normalization.
It does not tune rank, scales, fusion weights, or thresholds on test labels.

Sources
-------
- xBD-S12: Dietrich et al., arXiv:2511.05461 and
  https://github.com/prs-eth/xbd-s12.
- Canonical DS: Fukui and Maki, TPAMI 2015, with the project Band-Image sample
  adaptation implemented in ``phase1.subspace.band_image_geometry``.
- IR-MAD: Nielsen 2007, implemented in ``phase1.baselines.ir_mad``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from phase1.baselines.ir_mad import ir_mad_score
from phase1.baselines.pca_diff import pca_diff_score
from phase1.data.xbd_s12 import (
    XBDS12Patch,
    build_label_index,
    damage_evaluation_mask,
    load_metadata,
    load_normalization,
    load_patch,
    select_records,
)
from phase1.eval.metrics import binary_metrics
from phase1.eval.thresholding import otsu_threshold
from phase1.subspace.band_image_geometry import (
    band_image_ds_values,
    band_image_spatial_control_values,
)


Array = np.ndarray
INDIVIDUAL_METHODS = (
    "raw_l2",
    "spectral_angle",
    "pca_diff",
    "smoothed_pca_sigma1",
    "multiscale_pca_sigma0_1_2",
    "ir_mad",
    "band_image_ds",
    "band_image_cross_reconstruction",
    "band_image_spatial_gram",
    "band_image_projector_distance",
)
FUSION_DEPENDENCIES = {
    "fusion_smoothed_pca_band": ("smoothed_pca_sigma1", "band_image_ds"),
    "fusion_smoothed_pca_irmad": ("smoothed_pca_sigma1", "ir_mad"),
    "fusion_smoothed_pca_band_irmad": (
        "smoothed_pca_sigma1",
        "band_image_ds",
        "ir_mad",
    ),
    "fusion_smoothed_pca_cross_irmad": (
        "smoothed_pca_sigma1",
        "band_image_cross_reconstruction",
        "ir_mad",
    ),
}
ALL_METHODS = INDIVIDUAL_METHODS + tuple(FUSION_DEPENDENCIES)
LABEL_VIEWS = (
    "full_scene_damage",
    "building_conditional_damage",
    "building_localization_diagnostic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen xBD-S12 external validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=Path("data/xbd_s12"))
    parser.add_argument(
        "--labels-root", type=Path, default=Path("data/xbd_s12_original_labels")
    )
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--rank", type=int, default=11)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ir-mad-iters", type=int, default=10)
    parser.add_argument("--maximum-patches", type=int, default=None)
    parser.add_argument("--patches-per-event", type=int, default=None)
    parser.add_argument("--include-metadata-nodata", action="store_true")
    parser.add_argument(
        "--boundary-buffer",
        type=int,
        default=0,
        help="Ignore pixels within this Euclidean distance of building boundaries.",
    )
    parser.add_argument("--maps-per-event", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument(
        "--event-only",
        action="store_true",
        help="Skip patch-level metrics while retaining exact event/global evaluation.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def percentile_rank_map(score: Array, valid: Array) -> Array:
    """Return deterministic within-patch ranks in [0,1]."""
    values = score[valid].astype(np.float64)
    output = np.zeros(score.shape, dtype=np.float32)
    if values.size == 0:
        return output
    if values.size == 1:
        output[valid] = 1.0
        return output
    output[valid] = ((rankdata(values, method="average") - 1.0) / (values.size - 1.0)).astype(
        np.float32
    )
    return output


def _place_values(values: Array, valid: Array) -> Array:
    output = np.zeros(valid.shape, dtype=np.float32)
    output[valid] = values.astype(np.float32, copy=False)
    return output


def spectral_angle_score(pre: Array, post: Array, valid: Array) -> Array:
    """Return the standard per-pixel spectral angle in radians."""
    dot = np.sum(pre * post, axis=0)
    denominator = np.sqrt(np.sum(pre * pre, axis=0) * np.sum(post * post, axis=0))
    cosine = np.divide(dot, denominator, out=np.ones_like(dot), where=denominator > 1e-8)
    score = np.arccos(np.clip(cosine, -1.0, 1.0)).astype(np.float32)
    score[~valid] = 0.0
    return score


def compute_score_maps(patch: XBDS12Patch, rank: int, seed: int, ir_mad_iters: int) -> dict[str, Array]:
    """Compute every frozen individual map and equal-rank fusion."""
    pre, post, valid = patch.pre, patch.post, patch.input_valid
    difference = post - pre
    raw_l2 = np.sqrt(np.sum(difference * difference, axis=0)).astype(np.float32)
    raw_l2[~valid] = 0.0
    spectral_angle = spectral_angle_score(pre, post, valid)
    pca = pca_diff_score(
        pre,
        post,
        valid,
        rank_S=rank,
        variance_threshold=None,
        random_state=seed,
    )
    smooth1 = gaussian_filter(pca, sigma=1.0, mode="nearest").astype(np.float32)
    smooth2 = gaussian_filter(pca, sigma=2.0, mode="nearest").astype(np.float32)
    multiscale = ((pca + smooth1 + smooth2) / 3.0).astype(np.float32)
    irmad = ir_mad_score(
        pre,
        post,
        valid,
        iters=ir_mad_iters,
        downsample_max_pixels=200000,
        random_state=seed,
    )

    first = pre[:, valid].T.astype(np.float32, copy=False)
    second = post[:, valid].T.astype(np.float32, copy=False)
    ds = band_image_ds_values(first, second, rank=rank, seed=seed)
    maps: dict[str, Array] = {
        "raw_l2": raw_l2,
        "spectral_angle": spectral_angle,
        "pca_diff": pca,
        "smoothed_pca_sigma1": smooth1,
        "multiscale_pca_sigma0_1_2": multiscale,
        "ir_mad": irmad,
        "band_image_ds": _place_values(ds.projected_magnitude, valid),
        "band_image_cross_reconstruction": _place_values(
            band_image_spatial_control_values(
                first,
                second,
                rank=rank,
                seed=seed,
                mode="cross_reconstruction",
                first_basis=ds.pre_basis,
                second_basis=ds.post_basis,
            ),
            valid,
        ),
        "band_image_spatial_gram": _place_values(
            band_image_spatial_control_values(
                first, second, rank=rank, seed=seed, mode="spatial_gram"
            ),
            valid,
        ),
        "band_image_projector_distance": _place_values(
            band_image_spatial_control_values(
                first,
                second,
                rank=rank,
                seed=seed,
                mode="projector_distance",
                first_basis=ds.pre_basis,
                second_basis=ds.post_basis,
            ),
            valid,
        ),
    }
    ranks = {name: percentile_rank_map(score, valid) for name, score in maps.items()}
    for name, dependencies in FUSION_DEPENDENCIES.items():
        maps[name] = np.mean([ranks[dependency] for dependency in dependencies], axis=0).astype(
            np.float32
        )
    return maps


def score_metrics(score: Array, target: Array, support: Array) -> dict[str, float | int]:
    values = score[support].astype(np.float64)
    labels = target[support].astype(np.uint8)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    result: dict[str, float | int] = {
        "pixels": int(labels.size),
        "positives": positives,
        "negatives": negatives,
        "positive_rate": float(positives / max(1, labels.size)),
        "score_mean": float(np.mean(values)) if values.size else float("nan"),
        "score_p99": float(np.percentile(values, 99.0)) if values.size else float("nan"),
    }
    if positives == 0 or negatives == 0:
        result.update(
            auroc=float("nan"),
            average_precision=float("nan"),
            best_f1=0.0,
            best_iou=0.0,
            otsu_f1=0.0,
            otsu_iou=0.0,
        )
        return result
    result["auroc"] = float(roc_auc_score(labels, values))
    result["average_precision"] = float(average_precision_score(labels, values))
    precision, recall, _ = precision_recall_curve(labels, values)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_f1 = float(np.max(f1))
    result["best_f1"] = best_f1
    result["best_iou"] = float(best_f1 / max(2.0 - best_f1, 1e-12))
    threshold = otsu_threshold(score, support)
    binary = binary_metrics(score >= threshold, target, support)
    result["otsu_f1"] = binary["f1"]
    result["otsu_iou"] = binary["iou"]
    return result


def rgb_preview(cube: Array) -> Array:
    rgb = np.moveaxis(cube[[3, 2, 1]], 0, -1).astype(np.float32)
    low, high = np.percentile(rgb, [2.0, 98.0])
    return np.clip((rgb - low) / max(float(high - low), 1e-8), 0.0, 1.0)


def save_comparison_grid(path: Path, patch: XBDS12Patch, maps: dict[str, Array]) -> None:
    names = (
        "raw_l2",
        "spectral_angle",
        "pca_diff",
        "smoothed_pca_sigma1",
        "multiscale_pca_sigma0_1_2",
        "ir_mad",
        "band_image_ds",
        "band_image_cross_reconstruction",
        "band_image_projector_distance",
        "band_image_spatial_gram",
        "fusion_smoothed_pca_band_irmad",
        "fusion_smoothed_pca_cross_irmad",
    )
    fig, axes = plt.subplots(4, 4, figsize=(14, 14), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(rgb_preview(patch.pre)); axes[0].set_title("S2 pre")
    axes[1].imshow(rgb_preview(patch.post)); axes[1].set_title("S2 post")
    axes[2].imshow(patch.mask, vmin=0, vmax=6, cmap="tab10"); axes[2].set_title("xBD damage mask")
    axes[3].imshow(((patch.mask >= 2) & (patch.mask <= 4)), cmap="gray"); axes[3].set_title("damage target")
    for axis, name in zip(axes[4:], names):
        axis.imshow(maps[name], cmap="magma")
        axis.set_title(name.replace("_", " "), fontsize=9)
    for axis in axes:
        axis.axis("off")
    fig.suptitle(f"{patch.record.disaster} / {patch.record.uid}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def event_pairwise(event_rows: list[dict[str, object]], bootstrap: int, seed: int) -> list[dict[str, object]]:
    lookup = {
        (str(row["label_view"]), str(row["disaster"]), str(row["method"])): float(
            row["average_precision"]
        )
        for row in event_rows
        if math.isfinite(float(row["average_precision"]))
    }
    comparisons = (
        ("band_image_ds", "band_image_cross_reconstruction"),
        ("band_image_ds", "pca_diff"),
        ("band_image_ds", "smoothed_pca_sigma1"),
        ("fusion_smoothed_pca_band_irmad", "fusion_smoothed_pca_cross_irmad"),
        ("fusion_smoothed_pca_band_irmad", "smoothed_pca_sigma1"),
        ("fusion_smoothed_pca_band_irmad", "pca_diff"),
    )
    rng = np.random.default_rng(seed)
    output: list[dict[str, object]] = []
    for view in LABEL_VIEWS:
        disasters = sorted({key[1] for key in lookup if key[0] == view})
        for first, second in comparisons:
            available = [
                disaster
                for disaster in disasters
                if (view, disaster, first) in lookup and (view, disaster, second) in lookup
            ]
            if not available:
                continue
            delta = np.asarray(
                [lookup[(view, disaster, first)] - lookup[(view, disaster, second)] for disaster in available],
                dtype=np.float64,
            )
            draws = rng.integers(0, delta.size, size=(max(1, bootstrap), delta.size))
            bootstrap_delta = np.mean(delta[draws], axis=1)
            nonzero = delta[np.abs(delta) > 1e-15]
            p_value = float(wilcoxon(nonzero).pvalue) if nonzero.size >= 2 else float("nan")
            output.append(
                {
                    "label_view": view,
                    "method_a": first,
                    "method_b": second,
                    "events": len(available),
                    "mean_delta_a_minus_b": float(np.mean(delta)),
                    "ci95_low": float(np.quantile(bootstrap_delta, 0.025)),
                    "ci95_high": float(np.quantile(bootstrap_delta, 0.975)),
                    "wins_a": int(np.sum(delta > 0.0)),
                    "ties": int(np.sum(delta == 0.0)),
                    "wins_b": int(np.sum(delta < 0.0)),
                    "wilcoxon_p": p_value,
                }
            )
    return output


def summarize_event_metrics(event_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in event_rows:
        grouped[(str(row["label_view"]), str(row["method"]))].append(row)
    output: list[dict[str, object]] = []
    for (view, method), rows in sorted(grouped.items()):
        def mean_finite(field: str) -> float:
            values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            return float(np.mean(values)) if values.size else float("nan")

        output.append(
            {
                "label_view": view,
                "method": method,
                "events": len(rows),
                "mean_event_auroc": mean_finite("auroc"),
                "mean_event_average_precision": mean_finite("average_precision"),
                "mean_event_best_f1": mean_finite("best_f1"),
                "mean_event_otsu_f1": mean_finite("otsu_f1"),
            }
        )
    return output


def _select_per_event(records: list, maximum: int | None) -> list:
    if maximum is None:
        return records
    counts: dict[str, int] = defaultdict(int)
    output = []
    for record in records:
        if counts[record.disaster] < maximum:
            output.append(record)
            counts[record.disaster] += 1
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.root)
    records = select_records(
        metadata,
        split=args.split,
        exclude_metadata_nodata=not args.include_metadata_nodata,
        maximum=None,
    )
    records = _select_per_event(records, args.patches_per_event)
    if args.maximum_patches is not None:
        records = records[: args.maximum_patches]
    labels = build_label_index(args.labels_root)
    low, high = load_normalization(args.root)
    missing = [record.uid for record in records if record.uid not in labels]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} labels under {args.labels_root}; examples: {missing[:5]}"
        )

    print(
        f"Frozen xBD-S12 evaluation: split={args.split}, patches={len(records)}, "
        f"events={len(set(record.disaster for record in records))}, rank={args.rank}"
    )
    patch_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    event_buffers: dict[tuple[str, str, str], list[Array]] = defaultdict(list)
    target_buffers: dict[tuple[str, str], list[Array]] = defaultdict(list)
    maps_saved: dict[str, int] = defaultdict(int)
    failures: list[dict[str, str]] = []
    start_all = time.perf_counter()

    for index, record in enumerate(records, start=1):
        started = time.perf_counter()
        try:
            patch = load_patch(
                args.root,
                record,
                labels=labels,
                low=low,
                high=high,
                mask_cache_root=args.root / "project_damage_masks_majority_v1",
            )
            maps = compute_score_maps(patch, args.rank, args.seed, args.ir_mad_iters)
            for view in LABEL_VIEWS:
                target, support = damage_evaluation_mask(
                    patch.mask,
                    patch.input_valid,
                    view,
                    boundary_buffer=args.boundary_buffer,
                )
                target_buffers[(view, record.disaster)].append(target[support].astype(np.uint8))
                for method in ALL_METHODS:
                    if not args.event_only:
                        metrics = score_metrics(maps[method], target, support)
                        patch_rows.append(
                            {
                                "split": record.event_split,
                                "disaster": record.disaster,
                                "disaster_type": record.properties.get("disaster_type", ""),
                                "peril": record.properties.get("peril", ""),
                                "uid": record.uid,
                                "s2_cloud_score_pre": record.properties.get("s2_cs_pre", ""),
                                "s2_cloud_score_post": record.properties.get("s2_cs_post", ""),
                                "s2_date_pre": record.properties.get("s2_date_pre", ""),
                                "s2_date_post": record.properties.get("s2_date_post", ""),
                                "building_count": record.properties.get("N_total", ""),
                                "label_view": view,
                                "method": method,
                                **metrics,
                            }
                        )
                    event_buffers[(view, record.disaster, method)].append(
                        percentile_rank_map(maps[method], support)[support].astype(np.float32)
                    )
            if (
                maps_saved[record.disaster] < args.maps_per_event
                and np.any((patch.mask >= 2) & (patch.mask <= 4))
            ):
                save_comparison_grid(
                    args.output_dir / "maps" / f"{record.disaster}__{record.uid}.png",
                    patch,
                    maps,
                )
                maps_saved[record.disaster] += 1
        except Exception as exc:  # keep an auditable failure ledger
            failures.append({"uid": record.uid, "disaster": record.disaster, "error": repr(exc)})
            print(f"[{index}/{len(records)}] FAILED {record.uid}: {exc}", flush=True)
            continue
        print(
            f"[{index}/{len(records)}] {record.disaster}/{record.uid} "
            f"{time.perf_counter()-started:.2f}s",
            flush=True,
        )

    for (view, disaster), target_parts in sorted(target_buffers.items()):
        target = np.concatenate(target_parts)
        support = np.ones(target.shape, dtype=bool)
        for method in ALL_METHODS:
            score = np.concatenate(event_buffers[(view, disaster, method)])
            event_rows.append(
                {
                    "split": args.split,
                    "disaster": disaster,
                    "label_view": view,
                    "method": method,
                    **score_metrics(score, target, support),
                }
            )

    global_rows: list[dict[str, object]] = []
    for view in LABEL_VIEWS:
        disasters = sorted(
            disaster for candidate_view, disaster in target_buffers if candidate_view == view
        )
        if not disasters:
            continue
        target = np.concatenate(
            [np.concatenate(target_buffers[(view, disaster)]) for disaster in disasters]
        )
        support = np.ones(target.shape, dtype=bool)
        for method in ALL_METHODS:
            score = np.concatenate(
                [
                    np.concatenate(event_buffers[(view, disaster, method)])
                    for disaster in disasters
                ]
            )
            global_rows.append(
                {
                    "split": args.split,
                    "label_view": view,
                    "method": method,
                    **score_metrics(score, target, support),
                }
            )

    pairwise = event_pairwise(event_rows, args.bootstrap, args.seed + 901)
    summary = summarize_event_metrics(event_rows)
    write_csv(args.output_dir / "patch_metrics.csv", patch_rows)
    write_csv(args.output_dir / "event_metrics.csv", event_rows)
    write_csv(args.output_dir / "event_pairwise_comparisons.csv", pairwise)
    write_csv(args.output_dir / "summary_by_view_method.csv", summary)
    write_csv(args.output_dir / "global_metrics.csv", global_rows)
    write_csv(args.output_dir / "failures.csv", failures)
    metadata_out = {
        "protocol": "docs/experiment_reports/xbd_s12_frozen_external_protocol_2026-06-22.md",
        "root": str(args.root),
        "labels_root": str(args.labels_root),
        "split": args.split,
        "rank": args.rank,
        "seed": args.seed,
        "ir_mad_iters": args.ir_mad_iters,
        "boundary_buffer": args.boundary_buffer,
        "event_only": args.event_only,
        "patches_requested": len(records),
        "patches_completed": len(records) - len(failures),
        "events": sorted(set(record.disaster for record in records)),
        "methods": list(ALL_METHODS),
        "elapsed_sec": time.perf_counter() - start_all,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata_out, indent=2), encoding="utf-8"
    )
    print(f"Wrote frozen external evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
