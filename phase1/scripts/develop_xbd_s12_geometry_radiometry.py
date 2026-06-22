"""Develop spatial-geometry plus radiometry hypotheses on xBD-S12 train events.

This script is deliberately separated from the frozen test evaluator. It uses
only official training disasters to pressure-test centered PCA versus
uncentered autocorrelation subspaces, rank sensitivity, and fixed label-free
projector/radiometry combinations. Test-event results must not select these
choices.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from phase1.data.xbd_s12 import (
    build_label_index,
    damage_evaluation_mask,
    deterministic_event_sample,
    load_metadata,
    load_normalization,
    load_patch,
    select_records,
)
from phase1.scripts.evaluate_xbd_s12_external import (
    _place_values,
    percentile_rank_map,
    score_metrics,
    write_csv,
)
from phase1.subspace.band_image_geometry import (
    BASIS_MODES,
    band_image_ds_values,
    band_image_spatial_control_values,
)


Array = np.ndarray
VIEWS = (
    "full_scene_damage",
    "building_conditional_damage",
    "building_localization_diagnostic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Develop xBD-S12 geometry/radiometry hypotheses on training events."
    )
    parser.add_argument("--root", type=Path, default=Path("data/xbd_s12"))
    parser.add_argument(
        "--labels-root", type=Path, default=Path("data/xbd_s12_original_labels")
    )
    parser.add_argument("--ranks", default="2,4,6,8,10,11")
    parser.add_argument(
        "--basis-modes", default="centered_pca,uncentered_autocorrelation"
    )
    parser.add_argument("--patches-per-event", type=int, default=20)
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def geometry_maps(
    first: Array,
    second: Array,
    valid: Array,
    raw_l2: Array,
    *,
    rank: int,
    seed: int,
    basis_mode: str,
) -> dict[str, Array]:
    prefix = f"{basis_mode}__rank{rank}"
    ds = band_image_ds_values(
        first,
        second,
        rank=rank,
        seed=seed,
        basis_mode=basis_mode,
    )
    projector = _place_values(
        band_image_spatial_control_values(
            first,
            second,
            rank=rank,
            seed=seed,
            mode="projector_distance",
            basis_mode=basis_mode,
            first_basis=ds.pre_basis,
            second_basis=ds.post_basis,
        ),
        valid,
    )
    cross = _place_values(
        band_image_spatial_control_values(
            first,
            second,
            rank=rank,
            seed=seed,
            mode="cross_reconstruction",
            basis_mode=basis_mode,
            first_basis=ds.pre_basis,
            second_basis=ds.post_basis,
        ),
        valid,
    )
    ds_map = _place_values(ds.projected_magnitude, valid)
    projector_rank = percentile_rank_map(projector, valid)
    raw_rank = percentile_rank_map(raw_l2, valid)
    return {
        f"{prefix}__projector": projector,
        f"{prefix}__ds": ds_map,
        f"{prefix}__cross": cross,
        f"{prefix}__projector_raw_mean": (
            0.5 * (projector_rank + raw_rank)
        ).astype(np.float32),
        f"{prefix}__projector_raw_product": (
            projector_rank * raw_rank
        ).astype(np.float32),
    }


def summarize_events(event_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in event_rows:
        grouped[(str(row["label_view"]), str(row["method"]))].append(row)
    output = []
    for (view, method), rows in sorted(grouped.items()):
        output.append(
            {
                "label_view": view,
                "method": method,
                "events": len(rows),
                "mean_event_auroc": float(np.nanmean([float(r["auroc"]) for r in rows])),
                "mean_event_average_precision": float(
                    np.nanmean([float(r["average_precision"]) for r in rows])
                ),
                "mean_event_best_f1": float(
                    np.nanmean([float(r["best_f1"]) for r in rows])
                ),
            }
        )
    return output


def paired_rows(
    event_rows: list[dict[str, object]], bootstrap: int, seed: int
) -> list[dict[str, object]]:
    lookup = {
        (str(row["label_view"]), str(row["disaster"]), str(row["method"])): float(
            row["average_precision"]
        )
        for row in event_rows
        if math.isfinite(float(row["average_precision"]))
    }
    methods = sorted({key[2] for key in lookup})
    prefixes = sorted({method.rsplit("__", 1)[0] for method in methods if "__rank" in method})
    comparisons = []
    for prefix in prefixes:
        comparisons.extend(
            [
                (f"{prefix}__ds", f"{prefix}__cross", "ds_minus_cross"),
                (
                    f"{prefix}__projector_raw_mean",
                    f"{prefix}__projector",
                    "mean_fusion_minus_projector",
                ),
                (
                    f"{prefix}__projector_raw_product",
                    f"{prefix}__projector",
                    "product_fusion_minus_projector",
                ),
                (
                    f"{prefix}__projector_raw_mean",
                    "raw_l2",
                    "mean_fusion_minus_raw",
                ),
            ]
        )
    rng = np.random.default_rng(seed)
    output = []
    for view in VIEWS:
        events = sorted({key[1] for key in lookup if key[0] == view})
        for first, second, comparison in comparisons:
            available = [
                event
                for event in events
                if (view, event, first) in lookup and (view, event, second) in lookup
            ]
            if not available:
                continue
            delta = np.asarray(
                [lookup[(view, event, first)] - lookup[(view, event, second)] for event in available]
            )
            draws = rng.integers(0, len(delta), size=(bootstrap, len(delta)))
            boot = np.mean(delta[draws], axis=1)
            nonzero = delta[np.abs(delta) > 1e-15]
            p_value = float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 2 else float("nan")
            output.append(
                {
                    "label_view": view,
                    "comparison": comparison,
                    "method_a": first,
                    "method_b": second,
                    "events": len(available),
                    "mean_delta_a_minus_b": float(np.mean(delta)),
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "wins_a": int(np.sum(delta > 0)),
                    "ties": int(np.sum(delta == 0)),
                    "wins_b": int(np.sum(delta < 0)),
                    "wilcoxon_p": p_value,
                }
            )
    return output


def main() -> None:
    args = parse_args()
    ranks = tuple(sorted({int(value) for value in args.ranks.split(",") if value.strip()}))
    modes = tuple(value.strip() for value in args.basis_modes.split(",") if value.strip())
    invalid = [mode for mode in modes if mode not in BASIS_MODES]
    if invalid:
        raise ValueError(f"Unknown basis modes: {invalid}; expected {BASIS_MODES}.")

    records = select_records(load_metadata(args.root), split="train")
    records = deterministic_event_sample(
        records, per_event=args.patches_per_event, seed=args.seed
    )
    labels = build_label_index(args.labels_root)
    low, high = load_normalization(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_buffers: dict[tuple[str, str], list[Array]] = defaultdict(list)
    score_buffers: dict[tuple[str, str, str], list[Array]] = defaultdict(list)
    failures = []
    started_all = time.perf_counter()

    print(
        f"Training-event geometry development: patches={len(records)}, "
        f"events={len(set(r.disaster for r in records))}, ranks={ranks}, modes={modes}"
    )
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
            difference = patch.post - patch.pre
            raw_l2 = np.sqrt(np.sum(difference * difference, axis=0)).astype(np.float32)
            raw_l2[~patch.input_valid] = 0.0
            first = patch.pre[:, patch.input_valid].T.astype(np.float32, copy=False)
            second = patch.post[:, patch.input_valid].T.astype(np.float32, copy=False)
            maps = {"raw_l2": raw_l2}
            for mode in modes:
                for rank in ranks:
                    maps.update(
                        geometry_maps(
                            first,
                            second,
                            patch.input_valid,
                            raw_l2,
                            rank=rank,
                            seed=args.seed,
                            basis_mode=mode,
                        )
                    )
            for view in VIEWS:
                target, support = damage_evaluation_mask(
                    patch.mask, patch.input_valid, view
                )
                target_buffers[(view, record.disaster)].append(
                    target[support].astype(np.uint8)
                )
                for method, score in maps.items():
                    score_buffers[(view, record.disaster, method)].append(
                        percentile_rank_map(score, support)[support].astype(np.float32)
                    )
        except Exception as exc:
            failures.append(
                {"uid": record.uid, "disaster": record.disaster, "error": repr(exc)}
            )
            print(f"[{index}/{len(records)}] FAILED {record.uid}: {exc}", flush=True)
            continue
        print(
            f"[{index}/{len(records)}] {record.disaster}/{record.uid} "
            f"{time.perf_counter()-started:.2f}s",
            flush=True,
        )

    methods = sorted({key[2] for key in score_buffers})
    event_rows = []
    for (view, event), target_parts in sorted(target_buffers.items()):
        target = np.concatenate(target_parts)
        support = np.ones(target.shape, dtype=bool)
        for method in methods:
            score = np.concatenate(score_buffers[(view, event, method)])
            event_rows.append(
                {
                    "label_view": view,
                    "disaster": event,
                    "method": method,
                    **score_metrics(score, target, support),
                }
            )

    summary = summarize_events(event_rows)
    paired = paired_rows(event_rows, args.bootstrap, args.seed + 7001)
    write_csv(args.output_dir / "event_metrics.csv", event_rows)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "paired_comparisons.csv", paired)
    write_csv(args.output_dir / "failures.csv", failures)
    metadata = {
        "purpose": "train-event development; not independent test confirmation",
        "ranks": ranks,
        "basis_modes": modes,
        "patches_per_event": args.patches_per_event,
        "seed": args.seed,
        "patches_requested": len(records),
        "patches_completed": len(records) - len(failures),
        "events": sorted(set(record.disaster for record in records)),
        "elapsed_sec": time.perf_counter() - started_all,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote training-event geometry development to {args.output_dir}")


if __name__ == "__main__":
    main()
