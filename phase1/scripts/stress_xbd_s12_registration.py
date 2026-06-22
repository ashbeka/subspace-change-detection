"""Stress-test xBD-S12 score maps under controlled post-image misregistration.

Source/provenance
-----------------
The fixed methods and xBD-S12 loader are reused from the external validation.
Subpixel perturbations use bilinear image shifts and nearest-neighbor validity
shifts. This is a project sensitivity experiment, not part of DS theory.

Protocol
--------
Develop on official training disasters. Shift only the post-event Sentinel-2
cube while labels and pre-event data remain fixed. Evaluate full-scene AP and
visible damaged-building hit recall. A method is robust only if its retained
performance and degradation compare favorably with established baselines.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift as image_shift
from threadpoolctl import threadpool_limits

from phase1.data.xbd_s12 import (
    XBDS12Patch,
    build_label_index,
    damage_evaluation_mask,
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
DISPLAY = {
    "band_image_projector_distance": "Band-image projector",
    "ir_mad": "IR-MAD",
    "pca_diff": "PCA-diff",
    "raw_l2": "Raw L2 / CVA",
}


@dataclass(frozen=True)
class ShiftCondition:
    name: str
    magnitude: float
    dy: float
    dx: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress-test xBD-S12 maps under controlled registration shifts."
    )
    parser.add_argument("--root", type=Path, default=Path("data/xbd_s12"))
    parser.add_argument(
        "--labels-root", type=Path, default=Path("data/xbd_s12_original_labels")
    )
    parser.add_argument("--patches-per-event", type=int, default=20)
    parser.add_argument("--magnitudes", default="0.25,0.5,1.0")
    parser.add_argument("--rank", type=int, default=11)
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--ir-mad-iters", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--summarize-existing", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def registration_degradation_rows(
    event_rows: list[dict[str, object]],
    object_rows: list[dict[str, object]],
    *,
    bootstrap: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    output: list[dict[str, object]] = []

    pixel_baseline = {
        (str(row["disaster"]), str(row["method"])): float(row["average_precision"])
        for row in event_rows
        if float(row["magnitude"]) == 0.0
    }
    pixel_shifted: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    for row in event_rows:
        magnitude = float(row["magnitude"])
        if magnitude > 0:
            pixel_shifted[(str(row["disaster"]), str(row["method"]), magnitude)].append(
                float(row["average_precision"])
            )

    object_baseline = {
        (str(row["disaster"]), str(row["method"])): float(row["damaged_object_recall"])
        for row in object_rows
        if float(row["magnitude"]) == 0.0 and row["statistic"] == "p90"
    }
    object_shifted: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    for row in object_rows:
        magnitude = float(row["magnitude"])
        if magnitude > 0 and row["statistic"] == "p90":
            object_shifted[(str(row["disaster"]), str(row["method"]), magnitude)].append(
                float(row["damaged_object_recall"])
            )

    for task, baseline, shifted in (
        ("full_scene_average_precision", pixel_baseline, pixel_shifted),
        ("damaged_object_p90_recall_at_5pct", object_baseline, object_shifted),
    ):
        magnitudes = sorted({key[2] for key in shifted})
        events = sorted({key[0] for key in shifted})
        for magnitude in magnitudes:
            for method in METHODS:
                delta = np.asarray(
                    [
                        float(np.mean(shifted[(event, method, magnitude)]))
                        - baseline[(event, method)]
                        for event in events
                    ]
                )
                draws = rng.integers(0, len(delta), size=(bootstrap, len(delta)))
                boot = np.mean(delta[draws], axis=1)
                output.append(
                    {
                        "task": task,
                        "magnitude": magnitude,
                        "method": method,
                        "events": len(events),
                        "mean_shift_minus_baseline": float(np.mean(delta)),
                        "ci95_low": float(np.quantile(boot, 0.025)),
                        "ci95_high": float(np.quantile(boot, 0.975)),
                        "improved_events": int(np.sum(delta > 0)),
                        "ties": int(np.sum(delta == 0)),
                        "degraded_events": int(np.sum(delta < 0)),
                    }
                )
    return output


def shift_conditions(magnitudes: tuple[float, ...]) -> tuple[ShiftCondition, ...]:
    output = [ShiftCondition("shift0", 0.0, 0.0, 0.0)]
    directions = (
        ("x", 0.0, 1.0),
        ("y", 1.0, 0.0),
        ("diag_pos", 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)),
        ("diag_neg", -1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)),
    )
    for magnitude in magnitudes:
        for name, unit_y, unit_x in directions:
            output.append(
                ShiftCondition(
                    name=f"shift{magnitude:g}_{name}",
                    magnitude=magnitude,
                    dy=magnitude * unit_y,
                    dx=magnitude * unit_x,
                )
            )
    return tuple(output)


def shifted_patch(patch: XBDS12Patch, condition: ShiftCondition) -> XBDS12Patch:
    if condition.magnitude == 0.0:
        return patch
    post = image_shift(
        patch.post,
        shift=(0.0, condition.dy, condition.dx),
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32)
    shifted_valid = image_shift(
        patch.input_valid.astype(np.uint8),
        shift=(condition.dy, condition.dx),
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(bool)
    return XBDS12Patch(
        record=patch.record,
        pre=patch.pre,
        post=post,
        mask=patch.mask,
        input_valid=patch.input_valid & shifted_valid,
    )


def process_patch(
    record,
    *,
    root: Path,
    labels: dict[str, Path],
    low: np.ndarray,
    high: np.ndarray,
    conditions: tuple[ShiftCondition, ...],
    rank: int,
    seed: int,
    ir_mad_iters: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object] | None]:
    try:
        base = load_patch(
            root,
            record,
            labels=labels,
            low=low,
            high=high,
            mask_cache_root=root / "project_damage_masks_majority_v1",
        )
        objects = rasterize_xbd_instances(labels[record.uid], base.mask)
        pixel_parts: list[dict[str, object]] = []
        object_parts: list[dict[str, object]] = []
        for condition in conditions:
            patch = shifted_patch(base, condition)
            maps = compute_score_maps(patch, rank, seed, ir_mad_iters)
            target, support = damage_evaluation_mask(
                patch.mask, patch.input_valid, "full_scene_damage"
            )
            ranks = {
                method: percentile_rank_map(maps[method], patch.input_valid)
                for method in METHODS
            }
            pixel_parts.append(
                {
                    "event": record.disaster,
                    "condition": condition.name,
                    "magnitude": condition.magnitude,
                    "target": target[support].astype(np.uint8),
                    "scores": {
                        method: ranks[method][support].astype(np.float32)
                        for method in METHODS
                    },
                }
            )
            for method in METHODS:
                for statistic in ("maximum", "p90"):
                    damaged_total = damaged_hits = intact_total = intact_hits = 0
                    for obj in objects:
                        object_support = obj.mask & patch.input_valid
                        if not np.any(object_support):
                            continue
                        values = ranks[method][object_support].astype(np.float64)
                        score = (
                            float(np.max(values))
                            if statistic == "maximum"
                            else float(np.percentile(values, 90.0))
                        )
                        if 2 <= obj.damage_class <= 4:
                            damaged_total += 1
                            damaged_hits += int(score >= 0.95)
                        elif obj.damage_class == 1:
                            intact_total += 1
                            intact_hits += int(score >= 0.95)
                    object_parts.append(
                        {
                            "event": record.disaster,
                            "condition": condition.name,
                            "magnitude": condition.magnitude,
                            "method": method,
                            "statistic": statistic,
                            "damaged_total": damaged_total,
                            "damaged_hits": damaged_hits,
                            "intact_total": intact_total,
                            "intact_hits": intact_hits,
                        }
                    )
        return pixel_parts, object_parts, None
    except Exception as exc:
        return [], [], {"uid": record.uid, "disaster": record.disaster, "error": repr(exc)}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_existing:
        event_rows = read_csv(args.output_dir / "event_pixel_metrics.csv")
        object_rows = read_csv(args.output_dir / "event_object_metrics.csv")
        write_csv(
            args.output_dir / "registration_degradation.csv",
            registration_degradation_rows(
                event_rows,
                object_rows,
                bootstrap=args.bootstrap,
                seed=args.seed + 771,
            ),
        )
        print(f"Updated registration degradation summary in {args.output_dir}")
        return
    magnitudes = tuple(
        sorted({float(value) for value in args.magnitudes.split(",") if value.strip()})
    )
    if any(value <= 0 for value in magnitudes):
        raise ValueError("All nonzero shift magnitudes must be positive.")
    conditions = shift_conditions(magnitudes)
    records = deterministic_event_sample(
        select_records(load_metadata(args.root), split="train"),
        per_event=args.patches_per_event,
        seed=args.seed,
    )
    labels = build_label_index(args.labels_root)
    low, high = load_normalization(args.root)
    started = time.perf_counter()

    target_buffers: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    score_buffers: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    object_counters: dict[tuple[str, str, str, str], np.ndarray] = defaultdict(
        lambda: np.zeros(4, dtype=np.int64)
    )
    failures: list[dict[str, object]] = []

    def task(record):
        return process_patch(
            record,
            root=args.root,
            labels=labels,
            low=low,
            high=high,
            conditions=conditions,
            rank=args.rank,
            seed=args.seed,
            ir_mad_iters=args.ir_mad_iters,
        )

    # Outer patch parallelism is deliberate. Limit nested OpenBLAS pools to
    # one thread each; unrestricted nested SVD/eigensolver pools are unstable
    # and wasteful on Windows under repeated concurrent calls.
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            for index, (pixel_parts, object_parts, failure) in enumerate(
                executor.map(task, records), start=1
            ):
                if failure is not None:
                    failures.append(failure)
                for part in pixel_parts:
                    event = str(part["event"])
                    condition = str(part["condition"])
                    target_buffers[(event, condition)].append(part["target"])
                    for method, score in part["scores"].items():
                        score_buffers[(event, condition, method)].append(score)
                for part in object_parts:
                    key = (
                        str(part["event"]),
                        str(part["condition"]),
                        str(part["method"]),
                        str(part["statistic"]),
                    )
                    object_counters[key] += np.asarray(
                        [
                            part["damaged_total"],
                            part["damaged_hits"],
                            part["intact_total"],
                            part["intact_hits"],
                        ],
                        dtype=np.int64,
                    )
                if index % 10 == 0 or index == len(records):
                    print(f"[{index}/{len(records)}] failures={len(failures)}", flush=True)

    condition_lookup = {condition.name: condition for condition in conditions}
    event_rows: list[dict[str, object]] = []
    for (event, condition), target_parts in sorted(target_buffers.items()):
        target = np.concatenate(target_parts)
        support = np.ones(target.shape, dtype=bool)
        for method in METHODS:
            score = np.concatenate(score_buffers[(event, condition, method)])
            metrics = score_metrics(score, target, support)
            event_rows.append(
                {
                    "disaster": event,
                    "condition": condition,
                    "magnitude": condition_lookup[condition].magnitude,
                    "dy": condition_lookup[condition].dy,
                    "dx": condition_lookup[condition].dx,
                    "method": method,
                    **metrics,
                }
            )

    object_rows: list[dict[str, object]] = []
    for (event, condition, method, statistic), counts in sorted(object_counters.items()):
        damaged_total, damaged_hits, intact_total, intact_hits = counts.tolist()
        object_rows.append(
            {
                "disaster": event,
                "condition": condition,
                "magnitude": condition_lookup[condition].magnitude,
                "method": method,
                "statistic": statistic,
                "damaged_total": damaged_total,
                "intact_total": intact_total,
                "damaged_object_recall": damaged_hits / max(1, damaged_total),
                "intact_object_hit_rate": intact_hits / max(1, intact_total),
            }
        )

    pixel_grouped: dict[tuple[float, str], list[float]] = defaultdict(list)
    for row in event_rows:
        pixel_grouped[(float(row["magnitude"]), str(row["method"]))].append(
            float(row["average_precision"])
        )
    pixel_summary = [
        {
            "magnitude": magnitude,
            "method": method,
            "event_direction_observations": len(values),
            "mean_event_direction_average_precision": float(np.mean(values)),
        }
        for (magnitude, method), values in sorted(pixel_grouped.items())
    ]

    object_grouped: dict[tuple[float, str, str], list[float]] = defaultdict(list)
    for row in object_rows:
        object_grouped[
            (float(row["magnitude"]), str(row["method"]), str(row["statistic"]))
        ].append(float(row["damaged_object_recall"]))
    object_summary = [
        {
            "magnitude": magnitude,
            "method": method,
            "statistic": statistic,
            "event_direction_observations": len(values),
            "mean_event_direction_damaged_object_recall": float(np.mean(values)),
        }
        for (magnitude, method, statistic), values in sorted(object_grouped.items())
    ]

    write_csv(args.output_dir / "event_pixel_metrics.csv", event_rows)
    write_csv(args.output_dir / "event_object_metrics.csv", object_rows)
    write_csv(args.output_dir / "summary_pixel_metrics.csv", pixel_summary)
    write_csv(args.output_dir / "summary_object_metrics.csv", object_summary)
    write_csv(
        args.output_dir / "registration_degradation.csv",
        registration_degradation_rows(
            event_rows,
            object_rows,
            bootstrap=args.bootstrap,
            seed=args.seed + 771,
        ),
    )
    write_csv(args.output_dir / "failures.csv", failures)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    all_magnitudes = (0.0,) + magnitudes
    for method in METHODS:
        axes[0].plot(
            all_magnitudes,
            [
                next(
                    row["mean_event_direction_average_precision"]
                    for row in pixel_summary
                    if float(row["magnitude"]) == magnitude and row["method"] == method
                )
                for magnitude in all_magnitudes
            ],
            marker="o",
            label=DISPLAY[method],
        )
        axes[1].plot(
            all_magnitudes,
            [
                next(
                    row["mean_event_direction_damaged_object_recall"]
                    for row in object_summary
                    if float(row["magnitude"]) == magnitude
                    and row["method"] == method
                    and row["statistic"] == "p90"
                )
                for magnitude in all_magnitudes
            ],
            marker="o",
            label=DISPLAY[method],
        )
    axes[0].set_title("Full-scene damaged-pixel AP")
    axes[1].set_title("Damaged-building p90 hit recall at 5%")
    for axis in axes:
        axis.set_xlabel("Injected post-image shift (Sentinel pixels)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean event/direction metric")
    axes[1].set_ylabel("Mean event/direction metric")
    axes[0].legend()
    fig.suptitle("xBD-S12 training-event registration sensitivity")
    fig.savefig(args.output_dir / "registration_sensitivity.png", dpi=180)
    plt.close(fig)

    metadata = {
        "purpose": "controlled post-image registration sensitivity on training events",
        "patches_per_event": args.patches_per_event,
        "patches_requested": len(records),
        "patches_completed": len(records) - len(failures),
        "events": sorted(set(record.disaster for record in records)),
        "conditions": [condition.__dict__ for condition in conditions],
        "workers": args.workers,
        "elapsed_sec": time.perf_counter() - started,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote registration stress test to {args.output_dir}")


if __name__ == "__main__":
    main()
