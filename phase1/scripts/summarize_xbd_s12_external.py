"""Summarize the frozen xBD-S12 transfer and boundary-stress runs.

This reporting script does not recompute methods or select parameters. It
compares already completed frozen outputs, visualizes event-level evidence,
and separates damage retrieval, damage discrimination, and building
localization so that a localization effect is not mislabeled as damage
specificity.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    "band_image_projector_distance",
    "ir_mad",
    "fusion_smoothed_pca_band_irmad",
    "band_image_ds",
    "band_image_cross_reconstruction",
    "pca_diff",
    "raw_l2",
)
DISPLAY = {
    "band_image_projector_distance": "Band-image projector",
    "ir_mad": "IR-MAD",
    "fusion_smoothed_pca_band_irmad": "PCA + DS + IR-MAD",
    "band_image_ds": "Band-image DS",
    "band_image_cross_reconstruction": "Cross-reconstruction",
    "pca_diff": "PCA-diff",
    "raw_l2": "Raw L2 / CVA",
}
VIEWS = (
    "full_scene_damage",
    "building_conditional_damage",
    "building_localization_diagnostic",
)
VIEW_DISPLAY = {
    "full_scene_damage": "Full-scene damaged-pixel retrieval",
    "building_conditional_damage": "Damage vs intact buildings",
    "building_localization_diagnostic": "Building localization diagnostic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create xBD-S12 external-validation evidence figures."
    )
    parser.add_argument("--unbuffered", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def event_lookup(folder: Path) -> dict[tuple[str, str, str], float]:
    return {
        (row["label_view"], row["disaster"], row["method"]): float(
            row["average_precision"]
        )
        for row in read_csv(folder / "event_metrics.csv")
    }


def global_lookup(folder: Path) -> dict[tuple[str, str], dict[str, float]]:
    output: dict[tuple[str, str], dict[str, float]] = {}
    for row in read_csv(folder / "global_metrics.csv"):
        output[(row["label_view"], row["method"])] = {
            key: float(row[key])
            for key in ("average_precision", "auroc", "positive_rate", "best_f1")
        }
    return output


def event_heatmap(folder: Path, output: Path) -> None:
    lookup = event_lookup(folder)
    events = sorted({key[1] for key in lookup if key[0] == "full_scene_damage"})
    values = np.asarray(
        [[lookup[("full_scene_damage", event, method)] for event in events] for method in METHODS]
    )
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    image = ax.imshow(values * 100.0, cmap="viridis", aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{values[row, column]*100:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if values[row, column] > np.nanmedian(values) else "black")
    ax.set_xticks(range(len(events)), [event.replace("-", "\n") for event in events], fontsize=8)
    ax.set_yticks(range(len(METHODS)), [DISPLAY[method] for method in METHODS])
    ax.set_title("xBD-S12 unseen-event average precision (%)\nfull-scene damaged-pixel retrieval")
    fig.colorbar(image, ax=ax, label="Average precision (%)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def task_profile(folder: Path, output: Path) -> None:
    lookup = global_lookup(folder)
    values = np.asarray(
        [
            [
                lookup[(view, method)]["average_precision"]
                / max(lookup[(view, method)]["positive_rate"], 1e-12)
                for view in VIEWS
            ]
            for method in METHODS
        ]
    )
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    image = ax.imshow(values, cmap="magma", aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{values[row, column]:.2f}x", ha="center", va="center", fontsize=8,
                    color="white" if values[row, column] > np.nanmedian(values) else "black")
    ax.set_xticks(range(len(VIEWS)), [VIEW_DISPLAY[view] for view in VIEWS], rotation=12, ha="right")
    ax.set_yticks(range(len(METHODS)), [DISPLAY[method] for method in METHODS])
    ax.set_title("Global AP lift over class prevalence\n(task separation, not cross-task accuracy)")
    fig.colorbar(image, ax=ax, label="AP / prevalence")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def boundary_profile(unbuffered: Path, boundary: Path, output: Path) -> None:
    lookups = (global_lookup(unbuffered), global_lookup(boundary))
    run_names = ("Unbuffered primary", "3-pixel boundary stress")
    views = ("full_scene_damage", "building_localization_diagnostic")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    x = np.arange(len(METHODS), dtype=np.float64)
    width = 0.38
    for axis, view in zip(axes, views):
        for index, (lookup, name) in enumerate(zip(lookups, run_names)):
            values = [
                lookup[(view, method)]["average_precision"]
                / max(lookup[(view, method)]["positive_rate"], 1e-12)
                for method in METHODS
            ]
            axis.bar(x + (index - 0.5) * width, values, width=width, label=name)
        axis.set_title(VIEW_DISPLAY[view])
        axis.set_xticks(x, [DISPLAY[method] for method in METHODS], rotation=55, ha="right", fontsize=8)
        axis.set_ylabel("Global AP / prevalence")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.suptitle("Boundary sensitivity: ranking lift after removing building edges")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def paired_deltas(folder: Path, output: Path) -> None:
    lookup = event_lookup(folder)
    events = sorted({key[1] for key in lookup if key[0] == "full_scene_damage"})
    comparisons = (
        ("band_image_ds", "band_image_cross_reconstruction", "DS - cross-reconstruction"),
        ("band_image_projector_distance", "pca_diff", "Projector - PCA-diff"),
        ("band_image_projector_distance", "ir_mad", "Projector - IR-MAD"),
    )
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    x = np.arange(len(events), dtype=np.float64)
    width = 0.24
    for index, (first, second, label) in enumerate(comparisons):
        delta = [
            lookup[("full_scene_damage", event, first)]
            - lookup[("full_scene_damage", event, second)]
            for event in events
        ]
        ax.bar(x + (index - 1) * width, delta, width=width, label=label)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x, [event.replace("-", "\n") for event in events], fontsize=8)
    ax.set_ylabel("Event AP difference")
    ax.set_title("Paired event evidence for the full-scene task")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_summary(unbuffered: Path, boundary: Path, output: Path) -> None:
    primary = global_lookup(unbuffered)
    stress = global_lookup(boundary)
    payload = {
        "unbuffered": str(unbuffered),
        "boundary_stress": str(boundary),
        "global": {
            run: {
                view: {method: lookup[(view, method)] for method in METHODS}
                for view in VIEWS
            }
            for run, lookup in (("unbuffered", primary), ("boundary3", stress))
        },
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_heatmap(args.unbuffered, args.output_dir / "event_ap_heatmap.png")
    task_profile(args.unbuffered, args.output_dir / "task_profile_ap_lift.png")
    boundary_profile(
        args.unbuffered, args.boundary, args.output_dir / "boundary_sensitivity_ap_lift.png"
    )
    paired_deltas(args.unbuffered, args.output_dir / "paired_event_ap_deltas.png")
    write_summary(args.unbuffered, args.boundary, args.output_dir / "analysis_summary.json")
    print(f"Wrote xBD-S12 evidence summary to {args.output_dir}")


if __name__ == "__main__":
    main()
