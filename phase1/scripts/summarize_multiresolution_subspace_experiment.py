"""Create compact figures for the OSCD multiresolution subspace experiment.

This script only summarizes already-computed CSV evidence. It never refits a
method or uses test labels to select a configuration. Training inputs contain
the rank, representation, pyramid, and wavelet ablations; the test input is the
single frozen official-test evaluation.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-sweep", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--pyramid-sweep", type=Path, required=True)
    parser.add_argument("--wavelet-sweep", type=Path, required=True)
    parser.add_argument("--test-sweep", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def value(row: dict[str, str], field: str) -> float:
    return float(row[field])


def find_row(rows: list[dict[str, str]], config: str, method: str) -> dict[str, str]:
    return next(row for row in rows if row["config"] == config and row["method"] == method)


def plot_test_summary(test_dir: Path, output_dir: Path) -> None:
    rows = read_rows(test_dir / "sweep_summary_by_config_method.csv")
    methods = [
        ("successive_saab_h2_ds_fused", "Successive Saab + DS"),
        ("successive_saab_h2_cross_fused", "Successive Saab + cross"),
        ("smoothed_pca_sigma1", "Smoothed PCA-diff"),
        ("successive_saab_h2_l2_fused", "Successive Saab + L2"),
        ("pca_diff", "PCA-diff"),
        ("band_image_norm", "Global Band-Image DS"),
        ("wavelet_swt_db2_l2_ds_ll", "Wavelet LL + DS"),
        ("ir_mad", "IR-MAD"),
        ("raw_l2", "Raw L2 / CVA"),
    ]
    selected = [find_row(rows, "frozen_rank12", method) for method, _ in methods]
    labels = [label for _, label in methods]
    y_positions = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.6), sharey=True)
    specifications = [
        ("mean_average_precision", "Mean AP", "#d95f02"),
        ("mean_auroc", "Mean AUROC", "#1b9e77"),
        ("mean_otsu_f1", "Mean Otsu F1", "#7570b3"),
    ]
    for axis, (field, title, color) in zip(axes, specifications):
        values = [value(row, field) for row in selected]
        axis.barh(y_positions, values, color=color, alpha=0.9)
        axis.set_title(title)
        axis.set_xlim(0.0, 1.0)
        axis.grid(axis="x", alpha=0.25)
        for index, metric_value in enumerate(values):
            axis.text(metric_value + 0.012, index, f"{metric_value:.3f}", va="center", fontsize=8)
    axes[0].set_yticks(y_positions, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Frozen OSCD test cities: unsupervised changed-area evidence")
    fig.tight_layout()
    fig.savefig(output_dir / "frozen_test_method_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_city_deltas(test_dir: Path, output_dir: Path) -> None:
    rows = [
        row
        for row in read_rows(test_dir / "sweep_metrics_all.csv")
        if row["config"] == "frozen_rank12"
    ]
    cities = sorted({row["city"] for row in rows})
    by_city = {
        city: {
            row["method"]: value(row, "average_precision")
            for row in rows
            if row["city"] == city
        }
        for city in cities
    }
    pca_delta = np.asarray(
        [
            by_city[city]["successive_saab_h2_ds_fused"]
            - by_city[city]["smoothed_pca_sigma1"]
            for city in cities
        ]
    )
    cross_delta = np.asarray(
        [
            by_city[city]["successive_saab_h2_ds_fused"]
            - by_city[city]["successive_saab_h2_cross_fused"]
            for city in cities
        ]
    )
    order = np.argsort(pca_delta)
    cities = [cities[index] for index in order]
    pca_delta = pca_delta[order]
    cross_delta = cross_delta[order]
    x_positions = np.arange(len(cities))
    width = 0.38
    fig, axis = plt.subplots(figsize=(12.0, 5.2))
    axis.axhline(0.0, color="#333333", linewidth=1.0)
    axis.bar(
        x_positions - width / 2,
        pca_delta,
        width,
        label="DS - smoothed PCA",
        color="#d95f02",
    )
    axis.bar(
        x_positions + width / 2,
        cross_delta,
        width,
        label="DS - cross reconstruction",
        color="#1b9e77",
    )
    axis.set_xticks(x_positions, cities, rotation=35, ha="right")
    axis.set_ylabel("Paired AP difference")
    axis.set_title("Frozen test behavior by city")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "frozen_test_city_ap_deltas.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_training_ablations(args: argparse.Namespace, output_dir: Path) -> None:
    rank_rows = read_rows(args.rank_sweep / "sweep_summary_by_config_method.csv")
    ranks = [3, 6, 9, 12]
    rank_methods = [
        ("successive_saab_h2_ds_fused", "Successive Saab + DS"),
        ("successive_saab_h2_cross_fused", "Successive Saab + cross"),
        ("smoothed_pca_sigma1", "Smoothed PCA-diff"),
    ]

    feature = np.zeros((3, 3), dtype=np.float64)
    energies = ["0p90", "0p95", "0p99"]
    channels = [8, 16, 24]
    for row_index, energy in enumerate(energies):
        for column_index, channel in enumerate(channels):
            path = (
                args.feature_root
                / f"e{energy}_c{channel}"
                / "sweep_summary_by_config_method.csv"
            )
            rows = read_rows(path)
            row = next(
                item
                for item in rows
                if item["method"] == "successive_saab_h2_ds_fused"
            )
            feature[row_index, column_index] = value(
                row, "mean_average_precision"
            )

    pyramid_rows = read_rows(args.pyramid_sweep / "sweep_summary_by_config_method.csv")
    wavelet_rows = read_rows(args.wavelet_sweep / "sweep_summary_by_config_method.csv")
    family_entries = [
        ("Smoothed PCA", find_row(pyramid_rows, "rank12", "smoothed_pca_sigma1")),
        (
            "Pyramid shifted",
            find_row(
                pyramid_rows,
                "rank6",
                "multiscale_band_image_1_2_4_shifted",
            ),
        ),
        (
            "Product pyramid",
            find_row(
                pyramid_rows,
                "rank12",
                "multiscale_band_image_product_1_2_4_shifted",
            ),
        ),
        (
            "Wavelet db2 LL",
            find_row(wavelet_rows, "rank12", "wavelet_swt_db2_l2_ds_ll"),
        ),
        (
            "Successive Saab + DS",
            find_row(rank_rows, "rank12", "successive_saab_h2_ds_fused"),
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))
    for method, label in rank_methods:
        values = [
            value(find_row(rank_rows, f"rank{rank}", method), "mean_average_precision")
            for rank in ranks
        ]
        axes[0].plot(ranks, values, marker="o", linewidth=2, label=label)
    axes[0].set_title("A. Rank selection on training cities")
    axes[0].set_xlabel("Subspace rank")
    axes[0].set_ylabel("Mean AP")
    axes[0].set_xticks(ranks)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    image = axes[1].imshow(
        feature,
        cmap="viridis",
        vmin=float(feature.min()) - 0.005,
        vmax=float(feature.max()) + 0.005,
    )
    axes[1].set_title("B. Successive representation ablation")
    axes[1].set_xticks(range(3), channels)
    axes[1].set_yticks(range(3), ["90%", "95%", "99%"])
    axes[1].set_xlabel("Maximum channels per hop")
    axes[1].set_ylabel("Retained AC energy")
    for row_index in range(3):
        for column_index in range(3):
            axes[1].text(
                column_index,
                row_index,
                f"{feature[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                color="white",
            )
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="Mean AP")

    names = [name for name, _ in family_entries]
    values = [value(row, "mean_average_precision") for _, row in family_entries]
    colors = ["#999999", "#7570b3", "#7570b3", "#1b9e77", "#d95f02"]
    axes[2].barh(np.arange(len(names)), values, color=colors)
    axes[2].set_yticks(np.arange(len(names)), names)
    axes[2].invert_yaxis()
    axes[2].set_xlim(0.0, 0.30)
    axes[2].set_xlabel("Mean AP")
    axes[2].set_title("C. Training family comparison")
    axes[2].grid(axis="x", alpha=0.25)
    for index, metric_value in enumerate(values):
        axes[2].text(
            metric_value + 0.004,
            index,
            f"{metric_value:.3f}",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "training_ablation_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_test_summary(args.test_sweep, args.output_dir)
    plot_city_deltas(args.test_sweep, args.output_dir)
    plot_training_ablations(args, args.output_dir)
    print(f"Wrote figures: {args.output_dir}")


if __name__ == "__main__":
    main()
