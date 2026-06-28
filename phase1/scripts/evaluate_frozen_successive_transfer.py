"""Evaluate train-fitted successive Saab features with canonical DS.

Source/provenance
-----------------
- Kuo et al. PixelHop/JVCIR 2020: successive local neighborhood expansion,
  Saab DC/AC subspace approximation, and inter-hop pooling.
- Fukui and Maki, TPAMI 2015: canonical Difference Subspace (DS).
- OSCD: official 14-city training and 10-city test split.
- xBD-S12: official event-disjoint Sentinel-2 train/test split and published
  normalization from the prs-eth/xbd-s12 release.

Project question
----------------
The earlier positive OSCD result fitted Saab filters independently to every
unlabeled pre/post pair. This script removes that transductive degree of
freedom: one hierarchy is fitted on training pairs, serialized, and applied
unchanged to all held-out pairs. DS is still constructed per pre/post pair in
the frozen feature space because the research object is their difference
subspace. Matched L2, PCA-diff, and cross-reconstruction controls use the same
frozen feature maps.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import zoom
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.data.oscd_dataset import OFFICIAL_TEST, OFFICIAL_TRAIN, OSCDEvaluatorDataset
from phase1.data.preprocessing import apply_normalization, load_band_stats
from phase1.data.xbd_s12 import (
    S2_BANDS,
    build_label_index,
    damage_evaluation_mask,
    deterministic_event_sample,
    load_metadata,
    load_normalization,
    load_patch,
    read_s2_pair,
    select_records,
)
from phase1.scripts.compare_oscd_spatial_subspaces import score_metrics as oscd_score_metrics
from phase1.scripts.evaluate_xbd_s12_external import (
    LABEL_VIEWS,
    percentile_rank_map,
    score_metrics as xbd_score_metrics,
)
from phase1.subspace.multiscale_band_image import quantile_scale_map, score_band_image_matrix
from phase1.subspace.successive_subspace_features import (
    SuccessiveSaabModel,
    apply_successive_saab_model,
    fit_successive_saab_model,
    save_successive_saab_model,
)


Array = np.ndarray
OSCD_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A")
OSCD_COMMON12_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 12, 8, 10, 11)
FROZEN_METHODS = (
    "frozen_successive_ds",
    "frozen_successive_ds_hop1",
    "frozen_successive_ds_hop2",
    "frozen_successive_l2",
    "frozen_successive_pca",
    "frozen_successive_cross",
)
XBD_REFERENCE_METHODS = (
    "raw_l2",
    "pca_diff",
    "smoothed_pca_sigma1",
    "ir_mad",
    "band_image_ds",
    "band_image_projector_distance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit successive Saab on training pairs and evaluate frozen held-out features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fit-source", choices=("oscd13", "oscd12", "xbd12"), required=True)
    parser.add_argument("--target", choices=("oscd", "xbd"), required=True)
    parser.add_argument("--input-normalization", choices=("dataset", "pair_band_zscore"), default="dataset")
    parser.add_argument("--oscd-root", type=Path, default=Path("data/OSCD"))
    parser.add_argument("--oscd-stats", type=Path, default=Path("phase1/data/oscd_band_stats.json"))
    parser.add_argument("--xbd-root", type=Path, default=Path("data/xbd_s12"))
    parser.add_argument("--xbd-labels-root", type=Path, default=Path("data/xbd_s12_original_labels"))
    parser.add_argument("--fit-patches-per-event", type=int, default=20)
    parser.add_argument("--evaluation-patches-per-event", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--energy-threshold", type=float, default=0.95)
    parser.add_argument("--max-channels", type=int, default=16)
    parser.add_argument("--max-fit-samples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--maps-per-unit", type=int, default=1)
    parser.add_argument(
        "--xbd-reference-event-csv",
        type=Path,
        default=Path("phase1/outputs/xbd_s12_frozen_test_unbuffered_complete_20260622_111613/event_metrics.csv"),
    )
    parser.add_argument(
        "--oscd-pair-adaptive-csv",
        type=Path,
        default=Path("phase1/outputs/multiresolution_frozen_test10_20260623/sweep_metrics_all.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return value


def pair_band_zscore(first: Array, second: Array, valid: Array) -> tuple[Array, Array]:
    """Apply one label-free paired bandwise standardization."""
    first_values = first[:, valid].astype(np.float64)
    second_values = second[:, valid].astype(np.float64)
    count = float(first_values.shape[1] + second_values.shape[1])
    mean = (np.sum(first_values, axis=1) + np.sum(second_values, axis=1)) / count
    second_moment = (
        np.sum(first_values * first_values, axis=1)
        + np.sum(second_values * second_values, axis=1)
    ) / count
    std = np.sqrt(np.maximum(second_moment - mean * mean, 1e-8))
    normalized_first = ((first - mean[:, None, None]) / std[:, None, None]).astype(np.float32)
    normalized_second = ((second - mean[:, None, None]) / std[:, None, None]).astype(np.float32)
    normalized_first[:, ~valid] = 0.0
    normalized_second[:, ~valid] = 0.0
    return normalized_first, normalized_second


def normalize_pair(first: Array, second: Array, valid: Array, mode: str) -> tuple[Array, Array]:
    if mode == "dataset":
        return first.astype(np.float32, copy=False), second.astype(np.float32, copy=False)
    return pair_band_zscore(first, second, valid)


def build_oscd_factory(args: argparse.Namespace, common12: bool):
    stats = load_band_stats(args.oscd_stats)
    dataset = OSCDEvaluatorDataset(
        root=args.oscd_root,
        split="train",
        band_order=list(OSCD_BANDS),
        min_valid_bands=3,
        val_from_train=0,
    )

    def factory() -> Iterable[tuple[Array, Array, Array]]:
        for city in OFFICIAL_TRAIN:
            sample = dataset.load_city(city)
            first, valid = apply_normalization(sample.x_pre, stats, valid_mask=sample.valid_mask)
            second, _ = apply_normalization(sample.x_post, stats, valid_mask=sample.valid_mask)
            if common12:
                first = first[np.asarray(OSCD_COMMON12_INDICES)]
                second = second[np.asarray(OSCD_COMMON12_INDICES)]
            yield (*normalize_pair(first, second, valid, args.input_normalization), valid)

    return factory, len(OFFICIAL_TRAIN), 12 if common12 else 13


def select_xbd_training_records(args: argparse.Namespace):
    records = select_records(load_metadata(args.xbd_root), split="train")
    return deterministic_event_sample(
        records,
        per_event=int(args.fit_patches_per_event),
        seed=int(args.seed),
    )


def build_xbd_factory(args: argparse.Namespace):
    records = select_xbd_training_records(args)
    low, high = load_normalization(args.xbd_root)

    def factory() -> Iterable[tuple[Array, Array, Array]]:
        for record in records:
            first, second, valid = read_s2_pair(args.xbd_root, record.uid, low, high)
            yield (*normalize_pair(first, second, valid, args.input_normalization), valid)

    return factory, len(records), len(S2_BANDS)


def fit_model(args: argparse.Namespace, device: str) -> SuccessiveSaabModel:
    if args.fit_source == "oscd13":
        factory, pair_count, channels = build_oscd_factory(args, common12=False)
    elif args.fit_source == "oscd12":
        factory, pair_count, channels = build_oscd_factory(args, common12=True)
    else:
        factory, pair_count, channels = build_xbd_factory(args)
    return fit_successive_saab_model(
        factory,
        pair_count=pair_count,
        input_channels=channels,
        hops=args.hops,
        energy_threshold=args.energy_threshold,
        max_output_channels=(args.max_channels,) * args.hops,
        max_fit_samples=args.max_fit_samples,
        seed=args.seed,
        device=device,
        source_description=(
            f"fit_source={args.fit_source}; split=train; normalization={args.input_normalization}; "
            f"pairs={pair_count}"
        ),
    )


def resize_score(score: Array, target_shape: tuple[int, int]) -> Array:
    factors = (target_shape[0] / score.shape[0], target_shape[1] / score.shape[1])
    output = zoom(score, factors, order=1, mode="nearest", prefilter=False)
    return output[: target_shape[0], : target_shape[1]].astype(np.float32, copy=False)


def matched_feature_maps(
    first: Array,
    second: Array,
    valid: Array,
    model: SuccessiveSaabModel,
    *,
    rank: int,
    device: str,
) -> dict[str, Array]:
    """Compute DS and matched controls without repeating feature extraction."""
    ds_result = apply_successive_saab_model(
        first,
        second,
        valid,
        model,
        rank=rank,
        score_mode="ds_magnitude",
        device=device,
    )
    maps: dict[str, Array] = {
        "frozen_successive_ds": ds_result.fused_map,
        "frozen_successive_ds_hop1": ds_result.hop_maps[1],
        "frozen_successive_ds_hop2": ds_result.hop_maps.get(2, ds_result.hop_maps[1]),
    }
    mode_names = {
        "raw_l2": "frozen_successive_l2",
        "pca_diff": "frozen_successive_pca",
        "cross_reconstruction": "frozen_successive_cross",
    }
    for score_mode, method_name in mode_names.items():
        components: list[Array] = []
        for pre_features, post_features, feature_mask in zip(
            ds_result.pre_features,
            ds_result.post_features,
            ds_result.feature_masks,
        ):
            score, *_ = score_band_image_matrix(
                pre_features[:, feature_mask].T,
                post_features[:, feature_mask].T,
                rank=min(int(rank), pre_features.shape[0] - 1),
                seed=model.seed,
                score_mode=score_mode,
                basis_mode="centered_pca",
            )
            local = np.zeros(feature_mask.shape, dtype=np.float32)
            local[feature_mask] = score
            resized = resize_score(local, valid.shape)
            resized[~valid] = 0.0
            components.append(quantile_scale_map(resized, valid))
        maps[method_name] = np.mean(np.stack(components), axis=0).astype(np.float32)
        maps[method_name][~valid] = 0.0
    return maps


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired_rows(
    rows: list[dict[str, object]],
    *,
    unit_field: str,
    view_field: str | None,
    metric: str,
    bootstrap: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    views = sorted({str(row.get(view_field, "all")) for row in rows}) if view_field else ["all"]
    methods = sorted({str(row["method"]) for row in rows})
    output: list[dict[str, object]] = []
    for view in views:
        selected = [row for row in rows if not view_field or str(row.get(view_field)) == view]
        lookup = {
            (str(row[unit_field]), str(row["method"])): float(row[metric])
            for row in selected
            if math.isfinite(float(row[metric]))
        }
        units = sorted({unit for unit, _method in lookup})
        primary = "frozen_successive_ds"
        for second in methods:
            if second == primary:
                continue
            available = [unit for unit in units if (unit, primary) in lookup and (unit, second) in lookup]
            if not available:
                continue
            delta = np.asarray([lookup[(unit, primary)] - lookup[(unit, second)] for unit in available])
            draws = rng.integers(0, delta.size, size=(max(1, bootstrap), delta.size))
            sampled = np.mean(delta[draws], axis=1)
            nonzero = delta[np.abs(delta) > 1e-15]
            p_value = float(wilcoxon(nonzero).pvalue) if nonzero.size >= 2 else float("nan")
            output.append(
                {
                    "view": view,
                    "method_a": primary,
                    "method_b": second,
                    "units": len(available),
                    "mean_delta_a_minus_b": float(np.mean(delta)),
                    "ci95_low": float(np.quantile(sampled, 0.025)),
                    "ci95_high": float(np.quantile(sampled, 0.975)),
                    "wins_a": int(np.sum(delta > 0.0)),
                    "wins_b": int(np.sum(delta < 0.0)),
                    "wilcoxon_p": p_value,
                }
            )
    return output


def summarize_rows(rows: list[dict[str, object]], view_field: str | None) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get(view_field, "all")) if view_field else "all", str(row["method"]))].append(row)
    output: list[dict[str, object]] = []
    for (view, method), values in sorted(grouped.items()):
        def mean(field: str) -> float:
            finite = [float(row[field]) for row in values if field in row and math.isfinite(float(row[field]))]
            return float(np.mean(finite)) if finite else float("nan")
        output.append(
            {
                "view": view,
                "method": method,
                "units": len(values),
                "mean_auroc": mean("auroc"),
                "mean_average_precision": mean("average_precision"),
                "mean_best_f1": mean("best_f1"),
                "mean_otsu_f1": mean("otsu_f1"),
            }
        )
    return output


def save_map_grid(path: Path, first: Array, second: Array, target: Array, maps: dict[str, Array]) -> None:
    methods = [name for name in FROZEN_METHODS if name in maps]
    panels = 3 + len(methods)
    columns = 3
    rows = int(np.ceil(panels / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 4.2 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    rgb_indices = (3, 2, 1)
    for axis, image, title in zip(
        axes[:3],
        (first[list(rgb_indices)].transpose(1, 2, 0), second[list(rgb_indices)].transpose(1, 2, 0), target),
        ("Pre", "Post", "Target"),
    ):
        if image.ndim == 3:
            low, high = np.percentile(image, (2, 98))
            axis.imshow(np.clip((image - low) / max(high - low, 1e-8), 0.0, 1.0))
        else:
            axis.imshow(image, cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    for axis, method in zip(axes[3:], methods):
        axis.imshow(maps[method], cmap="magma", vmin=0.0, vmax=1.0)
        axis.set_title(method.replace("frozen_successive_", ""))
        axis.axis("off")
    for axis in axes[panels:]:
        axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def evaluate_oscd(args: argparse.Namespace, model: SuccessiveSaabModel, device: str, rank: int) -> list[dict[str, object]]:
    if model.stages[0].input_channels != 13:
        raise ValueError("OSCD target requires an oscd13 model.")
    stats = load_band_stats(args.oscd_stats)
    dataset = OSCDEvaluatorDataset(
        root=args.oscd_root,
        split="test",
        band_order=list(OSCD_BANDS),
        min_valid_bands=3,
        val_from_train=0,
    )
    rows: list[dict[str, object]] = []
    map_count = 0
    for city in OFFICIAL_TEST:
        sample = dataset.load_city(city)
        first, valid = apply_normalization(sample.x_pre, stats, valid_mask=sample.valid_mask)
        second, _ = apply_normalization(sample.x_post, stats, valid_mask=sample.valid_mask)
        first, second = normalize_pair(first, second, valid, args.input_normalization)
        started = time.perf_counter()
        maps = matched_feature_maps(first, second, valid, model, rank=rank, device=device)
        runtime = time.perf_counter() - started
        raw = np.linalg.norm(second - first, axis=0).astype(np.float32)
        raw[~valid] = 0.0
        for method, score in maps.items():
            metrics = oscd_score_metrics(score, sample.y[0], valid, raw)
            rows.append({"city": city, "method": method, "runtime_sec_all_methods": runtime, **metrics})
        if map_count < int(args.maps_per_unit):
            save_map_grid(
                args.output_dir / "maps" / f"{city}.png",
                first,
                second,
                sample.y[0],
                maps,
            )
            map_count += 1
        print(f"OSCD {city}: {runtime:.2f}s AP={rows[-6]['average_precision']:.4f}", flush=True)

    for reference in read_csv(args.oscd_pair_adaptive_csv):
        method = str(reference.get("method", ""))
        city = str(reference.get("city", ""))
        if city not in OFFICIAL_TEST or str(reference.get("config", "")) != "frozen_rank12":
            continue
        if method == "successive_saab_h2_ds_fused":
            copied = dict(reference)
            copied["method"] = "pair_adaptive_successive_ds"
            rows.append(copied)
        elif method in {"smoothed_pca_sigma1", "pca_diff", "raw_l2", "ir_mad"}:
            rows.append(dict(reference))
    return rows


def evaluate_xbd(args: argparse.Namespace, model: SuccessiveSaabModel, device: str, rank: int) -> list[dict[str, object]]:
    if model.stages[0].input_channels != 12:
        raise ValueError("xBD-S12 target requires a 12-channel model.")
    records = select_records(load_metadata(args.xbd_root), split="test")
    if args.evaluation_patches_per_event is not None:
        records = deterministic_event_sample(
            records,
            per_event=int(args.evaluation_patches_per_event),
            seed=int(args.seed) + 101,
        )
    labels = build_label_index(args.xbd_labels_root)
    low, high = load_normalization(args.xbd_root)
    buffers: dict[tuple[str, str, str], list[Array]] = defaultdict(list)
    targets: dict[tuple[str, str], list[Array]] = defaultdict(list)
    map_counts: dict[str, int] = defaultdict(int)
    for index, record in enumerate(records, start=1):
        patch = load_patch(
            args.xbd_root,
            record,
            labels=labels,
            low=low,
            high=high,
            mask_cache_root=args.xbd_root / "project_damage_masks_majority_v1",
        )
        first, second = normalize_pair(patch.pre, patch.post, patch.input_valid, args.input_normalization)
        maps = matched_feature_maps(first, second, patch.input_valid, model, rank=rank, device=device)
        for view in LABEL_VIEWS:
            target, support = damage_evaluation_mask(patch.mask, patch.input_valid, view)
            if not np.any(support):
                continue
            targets[(view, record.disaster)].append(target[support].astype(np.uint8))
            for method, score in maps.items():
                buffers[(view, record.disaster, method)].append(
                    percentile_rank_map(score, support)[support].astype(np.float32)
                )
        if map_counts[record.disaster] < int(args.maps_per_unit):
            full_target, _support = damage_evaluation_mask(
                patch.mask,
                patch.input_valid,
                "full_scene_damage",
            )
            save_map_grid(
                args.output_dir / "maps" / f"{record.disaster}_{record.uid}.png",
                first,
                second,
                full_target,
                maps,
            )
            map_counts[record.disaster] += 1
        if index % 100 == 0 or index == len(records):
            print(f"xBD-S12 {index}/{len(records)} patches", flush=True)

    event_rows: list[dict[str, object]] = []
    for (view, disaster), target_parts in sorted(targets.items()):
        target = np.concatenate(target_parts)
        for method in FROZEN_METHODS:
            score = np.concatenate(buffers[(view, disaster, method)])
            event_rows.append(
                {
                    "label_view": view,
                    "disaster": disaster,
                    "method": method,
                    **xbd_score_metrics(score, target, np.ones(target.shape, dtype=bool)),
                }
            )

    available_units = {(str(row["label_view"]), str(row["disaster"])) for row in event_rows}
    for reference in read_csv(args.xbd_reference_event_csv):
        key = (str(reference.get("label_view", "")), str(reference.get("disaster", "")))
        if key not in available_units or str(reference.get("method", "")) not in XBD_REFERENCE_METHODS:
            continue
        event_rows.append(dict(reference))
    return event_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.target == "oscd" and args.fit_source != "oscd13":
        raise ValueError("OSCD evaluation requires --fit-source oscd13.")
    if args.target == "xbd" and args.fit_source not in {"xbd12", "oscd12"}:
        raise ValueError("xBD evaluation requires --fit-source xbd12 or oscd12.")
    device = resolve_device(args.device)
    rank = int(args.rank if args.rank is not None else (12 if args.target == "oscd" else 11))
    started = time.perf_counter()
    model = fit_model(args, device)
    save_successive_saab_model(args.output_dir / "frozen_successive_saab_model.npz", model)
    fit_seconds = time.perf_counter() - started
    print(
        f"Fitted {len(model.stages)} hops from {model.fit_pair_count} pairs on {device} "
        f"in {fit_seconds:.2f}s; channels={[stage.output_channels for stage in model.stages]}",
        flush=True,
    )
    if args.target == "oscd":
        rows = evaluate_oscd(args, model, device, rank)
        unit_field, view_field = "city", None
    else:
        rows = evaluate_xbd(args, model, device, rank)
        unit_field, view_field = "disaster", "label_view"
    summary = summarize_rows(rows, view_field)
    pairwise = paired_rows(
        rows,
        unit_field=unit_field,
        view_field=view_field,
        metric="average_precision",
        bootstrap=args.bootstrap,
        seed=args.seed + 7001,
    )
    write_csv(args.output_dir / "unit_metrics.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "pairwise_ap.csv", pairwise)
    metadata = {
        "fit_source": args.fit_source,
        "target": args.target,
        "input_normalization": args.input_normalization,
        "rank": rank,
        "device": device,
        "fit_seconds": fit_seconds,
        "model_source": model.source_description,
        "fit_pair_count": model.fit_pair_count,
        "fit_samples_per_stage": list(model.fit_samples_per_stage),
        "stage_output_channels": [stage.output_channels for stage in model.stages],
        "test_labels_used_for_model_or_configuration": False,
        "methods": list(FROZEN_METHODS),
        "arguments": vars(args) | {"output_dir": str(args.output_dir)},
    }
    metadata["arguments"] = {key: str(value) if isinstance(value, Path) else value for key, value in metadata["arguments"].items()}
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote frozen transfer evidence to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
