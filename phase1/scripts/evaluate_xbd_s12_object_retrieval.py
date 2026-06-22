"""Evaluate xBD-S12 building-object retrieval from fixed change score maps.

Source/provenance
-----------------
- Building polygons and damage classes come from original xBD labels.
- Sentinel-2 pairs, official normalization, and event splits come from
  Dietrich et al. xBD-S12 and the official ``prs-eth/xbd-s12`` code.
- Score maps reuse the already verified xBD-S12 evaluator implementations:
  Band-Image projector geometry, IR-MAD, PCA-diff, and raw L2/CVA.

Project adaptation
------------------
This script tests the candidate-triage interpretation at object level. It does
not train on labels. An object is retained only if its polygon intersects the
official downsampled categorical mask at the same class. Pixel-percentile
scores are summarized per object by maximum, 90th percentile, and mean.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from threadpoolctl import threadpool_limits

from phase1.data.xbd_s12 import (
    build_label_index,
    deterministic_event_sample,
    load_metadata,
    load_normalization,
    load_patch,
    rasterize_xbd_instances,
    select_records,
)
from phase1.scripts.evaluate_xbd_s12_external import (
    compute_score_maps,
    percentile_rank_map,
    score_metrics,
    write_csv,
)


METHODS = (
    "band_image_projector_distance",
    "ir_mad",
    "pca_diff",
    "raw_l2",
)
STATISTICS = ("maximum", "p90", "mean")
BUDGETS = (0.01, 0.05, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate xBD-S12 object-level candidate retrieval."
    )
    parser.add_argument("--root", type=Path, default=Path("data/xbd_s12"))
    parser.add_argument(
        "--labels-root", type=Path, default=Path("data/xbd_s12_original_labels")
    )
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--patches-per-event", type=int, default=None)
    parser.add_argument("--rank", type=int, default=11)
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--ir-mad-iters", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def process_patch(
    record,
    *,
    root: Path,
    labels: dict[str, Path],
    low: np.ndarray,
    high: np.ndarray,
    rank: int,
    seed: int,
    ir_mad_iters: int,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    try:
        patch = load_patch(
            root,
            record,
            labels=labels,
            low=low,
            high=high,
            mask_cache_root=root / "project_damage_masks_majority_v1",
        )
        objects = rasterize_xbd_instances(labels[record.uid], patch.mask)
        maps = compute_score_maps(patch, rank, seed, ir_mad_iters)
        ranks = {
            method: percentile_rank_map(maps[method], patch.input_valid)
            for method in METHODS
        }
        rows: list[dict[str, object]] = []
        for obj in objects:
            support = obj.mask & patch.input_valid
            if not np.any(support):
                continue
            for method in METHODS:
                values = ranks[method][support].astype(np.float64)
                rows.append(
                    {
                        "split": record.event_split,
                        "disaster": record.disaster,
                        "uid": record.uid,
                        "object_id": obj.object_id,
                        "damage_class": obj.damage_class,
                        "damaged": int(2 <= obj.damage_class <= 4),
                        "pixels": int(np.sum(support)),
                        "method": method,
                        "maximum": float(np.max(values)),
                        "p90": float(np.percentile(values, 90.0)),
                        "mean": float(np.mean(values)),
                    }
                )
        return rows, None
    except Exception as exc:
        return [], {"uid": record.uid, "disaster": record.disaster, "error": repr(exc)}


def classification_rows(object_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    # Object rows are repeated by method, not statistic. Regroup directly by
    # event/method and evaluate each aggregate statistic.
    event_method: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in object_rows:
        event_method[(str(row["disaster"]), str(row["method"]))].append(row)
    output: list[dict[str, object]] = []
    for (event, method), rows in sorted(event_method.items()):
        labels = np.asarray([int(row["damaged"]) for row in rows], dtype=np.uint8)
        support = np.ones(labels.shape, dtype=bool)
        for statistic in STATISTICS:
            scores = np.asarray([float(row[statistic]) for row in rows], dtype=np.float64)
            metrics = score_metrics(scores, labels, support)
            output.append(
                {
                    "disaster": event,
                    "method": method,
                    "statistic": statistic,
                    "objects": len(rows),
                    **metrics,
                }
            )
    return output


def hit_rate_rows(object_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in object_rows:
        grouped[(str(row["disaster"]), str(row["method"]))].append(row)
    output: list[dict[str, object]] = []
    for (event, method), rows in sorted(grouped.items()):
        damaged = [row for row in rows if int(row["damaged"]) == 1]
        intact = [row for row in rows if int(row["damaged"]) == 0]
        for budget in BUDGETS:
            threshold = 1.0 - budget
            damaged_hits = sum(float(row["maximum"]) >= threshold for row in damaged)
            intact_hits = sum(float(row["maximum"]) >= threshold for row in intact)
            all_hits = damaged_hits + intact_hits
            output.append(
                {
                    "disaster": event,
                    "method": method,
                    "pixel_percentile_budget": budget,
                    "damaged_objects": len(damaged),
                    "intact_objects": len(intact),
                    "damaged_hits": damaged_hits,
                    "intact_hits": intact_hits,
                    "damaged_object_recall": damaged_hits / max(1, len(damaged)),
                    "intact_object_hit_rate": intact_hits / max(1, len(intact)),
                    "precision_among_hit_buildings": damaged_hits / max(1, all_hits),
                }
            )
    return output


def summarize(rows: list[dict[str, object]], keys: tuple[str, ...], fields: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    output = []
    for values, group in sorted(grouped.items()):
        result: dict[str, object] = {key: value for key, value in zip(keys, values)}
        result["events"] = len(group)
        for field in fields:
            numeric = np.asarray([float(row[field]) for row in group])
            numeric = numeric[np.isfinite(numeric)]
            result[f"mean_event_{field}"] = float(np.mean(numeric)) if numeric.size else float("nan")
        output.append(result)
    return output


def paired_ap_rows(
    classification: list[dict[str, object]], bootstrap: int, seed: int
) -> list[dict[str, object]]:
    lookup = {
        (str(row["disaster"]), str(row["method"]), str(row["statistic"])): float(
            row["average_precision"]
        )
        for row in classification
        if math.isfinite(float(row["average_precision"]))
    }
    comparisons = (
        ("band_image_projector_distance", "ir_mad"),
        ("band_image_projector_distance", "pca_diff"),
        ("band_image_projector_distance", "raw_l2"),
    )
    rng = np.random.default_rng(seed)
    output = []
    for statistic in STATISTICS:
        events = sorted({key[0] for key in lookup if key[2] == statistic})
        for first, second in comparisons:
            available = [
                event
                for event in events
                if (event, first, statistic) in lookup and (event, second, statistic) in lookup
            ]
            if not available:
                continue
            delta = np.asarray(
                [lookup[(event, first, statistic)] - lookup[(event, second, statistic)] for event in available]
            )
            draws = rng.integers(0, len(delta), size=(bootstrap, len(delta)))
            boot = np.mean(delta[draws], axis=1)
            nonzero = delta[np.abs(delta) > 1e-15]
            output.append(
                {
                    "statistic": statistic,
                    "method_a": first,
                    "method_b": second,
                    "events": len(available),
                    "mean_delta_a_minus_b": float(np.mean(delta)),
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "wins_a": int(np.sum(delta > 0)),
                    "ties": int(np.sum(delta == 0)),
                    "wins_b": int(np.sum(delta < 0)),
                    "wilcoxon_p": (
                        float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 2 else float("nan")
                    ),
                }
            )
    return output


def main() -> None:
    args = parse_args()
    records = select_records(load_metadata(args.root), split=args.split)
    if args.patches_per_event is not None:
        records = deterministic_event_sample(
            records, per_event=args.patches_per_event, seed=args.seed
        )
    labels = build_label_index(args.labels_root)
    low, high = load_normalization(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def task(record):
        return process_patch(
            record,
            root=args.root,
            labels=labels,
            low=low,
            high=high,
            rank=args.rank,
            seed=args.seed,
            ir_mad_iters=args.ir_mad_iters,
        )

    object_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            for index, (rows, failure) in enumerate(executor.map(task, records), start=1):
                object_rows.extend(rows)
                if failure is not None:
                    failures.append(failure)
                if index % 25 == 0 or index == len(records):
                    print(
                        f"[{index}/{len(records)}] objects={len(object_rows)//len(METHODS)} "
                        f"failures={len(failures)}",
                        flush=True,
                    )

    classification = classification_rows(object_rows)
    hit_rates = hit_rate_rows(object_rows)
    classification_summary = summarize(
        classification,
        ("method", "statistic"),
        ("auroc", "average_precision", "best_f1"),
    )
    hit_summary = summarize(
        hit_rates,
        ("method", "pixel_percentile_budget"),
        (
            "damaged_object_recall",
            "intact_object_hit_rate",
            "precision_among_hit_buildings",
        ),
    )
    paired = paired_ap_rows(classification, args.bootstrap, args.seed + 991)
    write_csv(args.output_dir / "object_scores.csv", object_rows)
    write_csv(args.output_dir / "event_object_classification.csv", classification)
    write_csv(args.output_dir / "event_object_hit_rates.csv", hit_rates)
    write_csv(args.output_dir / "summary_object_classification.csv", classification_summary)
    write_csv(args.output_dir / "summary_object_hit_rates.csv", hit_summary)
    write_csv(args.output_dir / "paired_object_ap.csv", paired)
    write_csv(args.output_dir / "failures.csv", failures)
    metadata = {
        "purpose": "building-object candidate retrieval and damage-vs-intact diagnosis",
        "split": args.split,
        "rank": args.rank,
        "seed": args.seed,
        "patches_per_event": args.patches_per_event,
        "workers": args.workers,
        "patches_requested": len(records),
        "patches_completed": len(records) - len(failures),
        "visible_objects": len(object_rows) // len(METHODS),
        "events": sorted(set(record.disaster for record in records)),
        "elapsed_sec": time.perf_counter() - started,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote object-level evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
