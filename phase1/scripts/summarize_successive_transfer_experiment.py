"""Summarize frozen successive Saab-DS transfer experiments.

This script only reads completed CSV outputs. It does not refit a model, choose
hyperparameters, or inspect labels beyond the metrics already written by
``evaluate_frozen_successive_transfer.py``.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create compact figures for frozen successive transfer evidence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--oscd-primary", type=Path, required=True)
    parser.add_argument("--oscd-seed7", type=Path, required=True)
    parser.add_argument("--oscd-seed2026", type=Path, required=True)
    parser.add_argument("--xbd-trainfit", type=Path, required=True)
    parser.add_argument("--xbd-oscd-native", type=Path, required=True)
    parser.add_argument("--xbd-oscd-pairz", type=Path, required=True)
    parser.add_argument("--xbd-trainfit-pairz", type=Path, required=True)
    parser.add_argument("--xbd-trainfit-pairz100", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summary_value(root: Path, method: str, *, view: str = "all", metric: str = "mean_average_precision") -> float:
    for row in read_rows(root / "summary.csv"):
        if row.get("method") == method and row.get("view", "all") == view:
            return float(row[metric])
    return float("nan")


def pairwise_value(root: Path, method_b: str, *, view: str = "all") -> tuple[float, float, float]:
    for row in read_rows(root / "pairwise_ap.csv"):
        if row.get("method_b") == method_b and row.get("view", "all") == view:
            return (
                float(row["mean_delta_a_minus_b"]),
                float(row["ci95_low"]),
                float(row["ci95_high"]),
            )
    return (float("nan"), float("nan"), float("nan"))


def save_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(max(8.0, 0.7 * len(labels)), 4.8), constrained_layout=True)
    ax.bar(labels, values, color="#4472C4")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(axis="y", alpha=0.25)
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_delta(path: Path, rows: list[tuple[str, float, float, float]], title: str) -> None:
    labels = [row[0] for row in rows]
    means = np.asarray([row[1] for row in rows], dtype=float)
    lows = np.asarray([row[2] for row in rows], dtype=float)
    highs = np.asarray([row[3] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.8 * len(labels)), 4.8), constrained_layout=True)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.errorbar(
        labels,
        means,
        yerr=np.vstack([means - lows, highs - means]),
        fmt="o",
        capsize=4,
        color="#C00000",
    )
    ax.set_title(title)
    ax.set_ylabel("AP delta: frozen successive DS minus baseline")
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oscd_methods = [
        "pair_adaptive_successive_ds",
        "frozen_successive_ds",
        "frozen_successive_cross",
        "smoothed_pca_sigma1",
        "pca_diff",
        "frozen_successive_l2",
        "raw_l2",
    ]
    save_bar(
        args.output_dir / "oscd_trainfit_ap.png",
        [name.replace("frozen_successive_", "frozen ").replace("pair_adaptive_", "pair adaptive ") for name in oscd_methods],
        [summary_value(args.oscd_primary, method) for method in oscd_methods],
        "OSCD official test: train-fitted successive Saab-DS",
        "Mean city AP",
    )
    save_delta(
        args.output_dir / "oscd_trainfit_ap_deltas.png",
        [
            ("cross", *pairwise_value(args.oscd_primary, "frozen_successive_cross")),
            ("L2", *pairwise_value(args.oscd_primary, "frozen_successive_l2")),
            ("PCA", *pairwise_value(args.oscd_primary, "frozen_successive_pca")),
            ("smoothed PCA", *pairwise_value(args.oscd_primary, "smoothed_pca_sigma1")),
            ("pair-adaptive", *pairwise_value(args.oscd_primary, "pair_adaptive_successive_ds")),
        ],
        "OSCD paired city deltas",
    )
    seed_labels = ["seed 7", "seed 1234", "seed 2026"]
    seed_values = [
        summary_value(args.oscd_seed7, "frozen_successive_ds"),
        summary_value(args.oscd_primary, "frozen_successive_ds"),
        summary_value(args.oscd_seed2026, "frozen_successive_ds"),
    ]
    save_bar(
        args.output_dir / "oscd_seed_stability.png",
        seed_labels,
        seed_values,
        "OSCD train-fitted seed stability",
        "Mean city AP",
    )
    variants = [
        ("xBD train", args.xbd_trainfit),
        ("xBD train + z", args.xbd_trainfit_pairz),
        ("xBD train100 + z", args.xbd_trainfit_pairz100),
        ("OSCD12 -> xBD", args.xbd_oscd_native),
        ("OSCD12 -> xBD + z", args.xbd_oscd_pairz),
    ]
    xbd_labels = [label for label, _root in variants] + ["Band-Image projector", "IR-MAD", "PCA-diff"]
    xbd_values = [summary_value(root, "frozen_successive_ds", view="full_scene_damage") for _label, root in variants]
    xbd_values += [
        summary_value(args.xbd_trainfit, "band_image_projector_distance", view="full_scene_damage"),
        summary_value(args.xbd_trainfit, "ir_mad", view="full_scene_damage"),
        summary_value(args.xbd_trainfit, "pca_diff", view="full_scene_damage"),
    ]
    save_bar(
        args.output_dir / "xbd_successive_transfer_ap.png",
        xbd_labels,
        xbd_values,
        "xBD-S12 full-scene damage retrieval",
        "Mean event AP",
    )
    save_delta(
        args.output_dir / "xbd_best_variant_ap_deltas.png",
        [
            ("cross", *pairwise_value(args.xbd_trainfit_pairz100, "frozen_successive_cross", view="full_scene_damage")),
            ("L2", *pairwise_value(args.xbd_trainfit_pairz100, "frozen_successive_l2", view="full_scene_damage")),
            ("PCA", *pairwise_value(args.xbd_trainfit_pairz100, "frozen_successive_pca", view="full_scene_damage")),
            ("Band-Image DS", *pairwise_value(args.xbd_trainfit_pairz100, "band_image_ds", view="full_scene_damage")),
            ("Projector", *pairwise_value(args.xbd_trainfit_pairz100, "band_image_projector_distance", view="full_scene_damage")),
            ("IR-MAD", *pairwise_value(args.xbd_trainfit_pairz100, "ir_mad", view="full_scene_damage")),
        ],
        "xBD-S12 best successive variant paired event deltas",
    )
    copy_if_exists(args.oscd_primary / "maps" / "chongqing.png", args.output_dir / "oscd_chongqing_trainfit_map.png")
    xbd_map_dir = args.xbd_trainfit / "maps"
    for source in sorted(xbd_map_dir.glob("*.png"))[:2]:
        copy_if_exists(source, args.output_dir / f"xbd_{source.name}")
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
