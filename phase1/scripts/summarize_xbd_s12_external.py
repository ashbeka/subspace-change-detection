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
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon


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
    parser.add_argument("--train-sweep", type=Path, default=None)
    parser.add_argument("--train-confirmation", type=Path, default=None)
    parser.add_argument("--train-classical", type=Path, default=None)
    parser.add_argument("--test-budget", type=Path, default=None)
    parser.add_argument("--object-train", type=Path, default=None)
    parser.add_argument("--object-test", type=Path, default=None)
    parser.add_argument("--registration-near", type=Path, default=None)
    parser.add_argument("--registration-large", type=Path, default=None)
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


def train_rank_sensitivity(folder: Path, output: Path) -> None:
    rows = read_csv(folder / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, view in zip(
        axes, ("full_scene_damage", "building_localization_diagnostic")
    ):
        for mode, label in (
            ("centered_pca", "Centered PCA"),
            ("uncentered_autocorrelation", "Uncentered autocorrelation"),
        ):
            points = []
            for row in rows:
                prefix = f"{mode}__rank"
                suffix = "__projector"
                if (
                    row["label_view"] == view
                    and row["method"].startswith(prefix)
                    and row["method"].endswith(suffix)
                ):
                    rank = int(row["method"][len(prefix) : -len(suffix)])
                    points.append((rank, float(row["mean_event_average_precision"])))
            points.sort()
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                label=label,
            )
        axis.set_title(VIEW_DISPLAY[view])
        axis.set_xlabel("Subspace rank")
        axis.set_ylabel("Mean training-event AP")
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Training-event projector rank and centering sensitivity")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def train_confirmation_profile(folder: Path, output: Path) -> None:
    rows = read_csv(folder / "summary.csv")
    lookup = {
        (row["label_view"], row["method"]): float(row["mean_event_average_precision"])
        for row in rows
    }
    methods = (
        "uncentered_autocorrelation__rank11__projector",
        "centered_pca__rank11__projector",
        "uncentered_autocorrelation__rank11__projector_raw_mean",
        "uncentered_autocorrelation__rank11__ds",
        "uncentered_autocorrelation__rank11__cross",
        "raw_l2",
    )
    labels = (
        "Uncentered projector",
        "Centered projector",
        "Projector + raw mean",
        "Uncentered DS",
        "Cross-reconstruction",
        "Raw L2 / CVA",
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, view in zip(axes, VIEWS):
        values = [lookup[(view, method)] for method in methods]
        axis.bar(np.arange(len(methods)), values)
        axis.set_title(VIEW_DISPLAY[view])
        axis.set_xticks(np.arange(len(methods)), labels, rotation=60, ha="right", fontsize=8)
        axis.set_ylabel("Mean training-event AP")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("1,100-patch training-event confirmation: task decomposition")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def operational_budget_profile(
    train_folder: Path,
    test_folder: Path,
    output: Path,
) -> None:
    methods = (
        "band_image_projector_distance",
        "ir_mad",
        "pca_diff",
        "raw_l2",
    )
    budgets = (1, 5, 10)

    def lookup(folder: Path) -> dict[str, dict[str, float]]:
        return {
            row["method"]: {key: float(value) for key, value in row.items() if key.startswith("mean_event_top_")}
            for row in read_csv(folder / "summary_by_view_method.csv")
            if row["label_view"] == "full_scene_damage" and row["method"] in methods
        }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for row_index, (folder, split_name) in enumerate(
        ((train_folder, "Training events"), (test_folder, "Unseen test events"))
    ):
        values = lookup(folder)
        for column_index, (metric, ylabel) in enumerate(
            (("recall", "Damaged-pixel recall"), ("enrichment", "Lift over prevalence"))
        ):
            axis = axes[row_index, column_index]
            for method in methods:
                axis.plot(
                    budgets,
                    [values[method][f"mean_event_top_{budget}pct_{metric}"] for budget in budgets],
                    marker="o",
                    label=DISPLAY[method],
                )
            axis.set_title(f"{split_name}: {ylabel}")
            axis.set_xlabel("Pixel review budget (%)")
            axis.set_ylabel(ylabel)
            axis.set_xticks(budgets)
            axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle(
        "Full-scene xBD-S12 candidate ranking at fixed analyst budgets\n"
        "test metrics are secondary post-hoc operational analysis"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def nuisance_stratification(
    folder: Path,
    output_figure: Path,
    output_csv: Path,
    *,
    bootstrap: int = 5000,
    seed: int = 8417,
) -> None:
    """Relate patch AP deltas to event-relative acquisition covariates.

    This is exploratory, post-hoc failure analysis. Covariates are ranked
    within each disaster before low/middle/high grouping, preventing a large
    event from defining all strata. Confidence intervals resample disasters,
    not individual patches.
    """
    methods = (
        "band_image_projector_distance",
        "ir_mad",
        "pca_diff",
        "raw_l2",
    )
    rows = [
        row
        for row in read_csv(folder / "patch_metrics.csv")
        if row["label_view"] == "full_scene_damage" and row["method"] in methods
    ]
    patches: dict[str, dict[str, object]] = {}
    for row in rows:
        ap = float(row["average_precision"])
        if not np.isfinite(ap):
            continue
        item = patches.setdefault(
            row["uid"],
            {
                "uid": row["uid"],
                "event": row["disaster"],
                "min_pair_cloudscore_plus": min(
                    float(row["s2_cloud_score_pre"]),
                    float(row["s2_cloud_score_post"]),
                ),
                "cloudscore_change": abs(
                    float(row["s2_cloud_score_post"])
                    - float(row["s2_cloud_score_pre"])
                ),
                "date_gap_days": abs(
                    (_iso_datetime(row["s2_date_post"]) - _iso_datetime(row["s2_date_pre"])).days
                ),
                "scores": {},
            },
        )
        item["scores"][row["method"]] = ap  # type: ignore[index]
    complete = [
        item for item in patches.values() if all(method in item["scores"] for method in methods)
    ]
    factors = (
        ("min_pair_cloudscore_plus", "Minimum pair CloudScore+"),
        ("cloudscore_change", "Absolute CloudScore+ change"),
        ("date_gap_days", "Pre/post date gap (days)"),
    )
    baselines = ("ir_mad", "pca_diff", "raw_l2")
    rng = np.random.default_rng(seed)
    output_rows: list[dict[str, object]] = []
    plot_values: dict[tuple[str, str], tuple[list[float], list[float], list[float]]] = {}

    for factor, _ in factors:
        event_items: dict[str, list[dict[str, object]]] = defaultdict(list)
        for item in complete:
            event_items[str(item["event"])].append(item)
        for event_rows in event_items.values():
            values = np.asarray([float(item[factor]) for item in event_rows])
            percentiles = (rankdata(values, method="average") - 0.5) / len(values)
            for item, percentile in zip(event_rows, percentiles):
                item[f"{factor}__stratum"] = min(2, int(percentile * 3.0))

        for baseline in baselines:
            event_bin_means: dict[str, list[float]] = {}
            centered_factor: list[float] = []
            centered_delta: list[float] = []
            for event, event_rows in event_items.items():
                factor_values = np.asarray([float(item[factor]) for item in event_rows])
                deltas = np.asarray(
                    [
                        float(item["scores"]["band_image_projector_distance"])  # type: ignore[index]
                        - float(item["scores"][baseline])  # type: ignore[index]
                        for item in event_rows
                    ]
                )
                centered_factor.extend((factor_values - np.mean(factor_values)).tolist())
                centered_delta.extend((deltas - np.mean(deltas)).tolist())
                means = []
                for stratum in range(3):
                    selected = [
                        delta
                        for item, delta in zip(event_rows, deltas)
                        if int(item[f"{factor}__stratum"]) == stratum
                    ]
                    means.append(float(np.mean(selected)) if selected else float("nan"))
                event_bin_means[event] = means

            rho, rho_p = spearmanr(centered_factor, centered_delta)
            events = sorted(event_bin_means)
            matrix = np.asarray([event_bin_means[event] for event in events])
            means = np.nanmean(matrix, axis=0)
            low = np.full(3, np.nan, dtype=np.float64)
            high = np.full(3, np.nan, dtype=np.float64)
            for column in range(3):
                finite = matrix[np.isfinite(matrix[:, column]), column]
                if finite.size:
                    draws = rng.integers(0, finite.size, size=(bootstrap, finite.size))
                    boot = np.mean(finite[draws], axis=1)
                    low[column], high[column] = np.quantile(boot, (0.025, 0.975))
            paired = matrix[np.isfinite(matrix[:, 0]) & np.isfinite(matrix[:, 2])]
            if paired.size:
                draws = rng.integers(0, len(paired), size=(bootstrap, len(paired)))
                high_minus_low = np.mean(paired[draws, 2] - paired[draws, 0], axis=1)
                delta_ci = np.quantile(high_minus_low, (0.025, 0.975))
            else:
                delta_ci = (float("nan"), float("nan"))
            output_rows.append(
                {
                    "factor": factor,
                    "baseline": baseline,
                    "patches": len(complete),
                    "events": len(events),
                    "high_low_paired_events": len(paired),
                    "within_event_spearman_rho": float(rho),
                    "within_event_spearman_p": float(rho_p),
                    "low_mean_ap_delta": float(means[0]),
                    "middle_mean_ap_delta": float(means[1]),
                    "high_mean_ap_delta": float(means[2]),
                    "high_minus_low": float(means[2] - means[0]),
                    "high_minus_low_ci95_low": float(delta_ci[0]),
                    "high_minus_low_ci95_high": float(delta_ci[1]),
                }
            )
            plot_values[(factor, baseline)] = (means.tolist(), low.tolist(), high.tolist())

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    x = np.arange(3)
    for axis, (factor, display) in zip(axes, factors):
        for baseline in baselines:
            means, low, high = plot_values[(factor, baseline)]
            axis.errorbar(
                x,
                means,
                yerr=[np.asarray(means) - np.asarray(low), np.asarray(high) - np.asarray(means)],
                marker="o",
                capsize=3,
                label=f"Projector - {DISPLAY[baseline]}",
            )
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_xticks(x, ("Low", "Middle", "High"))
        axis.set_title(display)
        axis.set_ylabel("Mean within-event patch AP delta")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Exploratory xBD-S12 nuisance strata (event-relative tertiles)\n"
        "error bars: event-cluster bootstrap 95% intervals"
    )
    fig.savefig(output_figure, dpi=180)
    plt.close(fig)


def object_candidate_profile(
    train_folder: Path,
    test_folder: Path,
    output_figure: Path,
    output_pairwise: Path,
    *,
    bootstrap: int = 5000,
    seed: int = 1927,
) -> None:
    methods = (
        "band_image_projector_distance",
        "ir_mad",
        "pca_diff",
        "raw_l2",
    )
    budgets = (0.01, 0.05, 0.10)

    def summary_lookup(folder: Path) -> dict[tuple[str, float], dict[str, float]]:
        return {
            (row["method"], float(row["pixel_percentile_budget"])): {
                key: float(value)
                for key, value in row.items()
                if key.startswith("mean_event_")
            }
            for row in read_csv(folder / "summary_object_hit_rates.csv")
        }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for row_index, (folder, split_name) in enumerate(
        ((train_folder, "Training events"), (test_folder, "Unseen test events"))
    ):
        lookup = summary_lookup(folder)
        for column_index, (field, title) in enumerate(
            (
                ("mean_event_damaged_object_recall", "Damaged-building hit recall"),
                ("mean_event_intact_object_hit_rate", "Intact-building hit rate"),
            )
        ):
            axis = axes[row_index, column_index]
            for method in methods:
                axis.plot(
                    [budget * 100 for budget in budgets],
                    [lookup[(method, budget)][field] for budget in budgets],
                    marker="o",
                    label=DISPLAY[method],
                )
            axis.set_title(f"{split_name}: {title}")
            axis.set_xlabel("Scene pixel-score percentile budget (%)")
            axis.set_ylabel(title)
            axis.set_xticks([1, 5, 10])
            axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle(
        "xBD-S12 object-level coverage/specificity tradeoff\n"
        "object hit = any visible polygon pixel above the scene percentile threshold"
    )
    fig.savefig(output_figure, dpi=180)
    plt.close(fig)

    rng = np.random.default_rng(seed)
    pairwise_rows: list[dict[str, object]] = []
    for split_name, folder in (("train", train_folder), ("test", test_folder)):
        rows = read_csv(folder / "event_object_hit_rates.csv")
        lookup = {
            (row["disaster"], row["method"], float(row["pixel_percentile_budget"])): float(
                row["damaged_object_recall"]
            )
            for row in rows
        }
        events = sorted({row["disaster"] for row in rows})
        for budget in budgets:
            for baseline in methods[1:]:
                delta = np.asarray(
                    [
                        lookup[(event, methods[0], budget)]
                        - lookup[(event, baseline, budget)]
                        for event in events
                    ]
                )
                draws = rng.integers(0, len(delta), size=(bootstrap, len(delta)))
                boot = np.mean(delta[draws], axis=1)
                nonzero = delta[np.abs(delta) > 1e-15]
                pairwise_rows.append(
                    {
                        "split": split_name,
                        "pixel_percentile_budget": budget,
                        "method_a": methods[0],
                        "method_b": baseline,
                        "events": len(events),
                        "mean_recall_delta_a_minus_b": float(np.mean(delta)),
                        "ci95_low": float(np.quantile(boot, 0.025)),
                        "ci95_high": float(np.quantile(boot, 0.975)),
                        "wins_a": int(np.sum(delta > 0)),
                        "ties": int(np.sum(delta == 0)),
                        "wins_b": int(np.sum(delta < 0)),
                        "wilcoxon_p": (
                            float(wilcoxon(nonzero).pvalue)
                            if len(nonzero) >= 2
                            else float("nan")
                        ),
                    }
                )
    with output_pairwise.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairwise_rows[0]))
        writer.writeheader()
        writer.writerows(pairwise_rows)


def object_size_sensitivity(
    train_folder: Path,
    test_folder: Path,
    output_figure: Path,
    output_csv: Path,
) -> None:
    methods = (
        "band_image_projector_distance",
        "ir_mad",
        "pca_diff",
        "raw_l2",
    )
    statistics = ("maximum", "p90", "mean")
    output_rows: list[dict[str, object]] = []
    plot_lookup: dict[tuple[str, str, str, int], float] = {}

    for split_name, folder in (("train", train_folder), ("test", test_folder)):
        rows = read_csv(folder / "object_scores.csv")
        object_sizes = {
            (row["disaster"], row["uid"], row["object_id"]): int(row["pixels"])
            for row in rows
            if row["method"] == methods[0]
        }
        by_event: dict[str, list[tuple[tuple[str, str, str], int]]] = defaultdict(list)
        for key, pixels in object_sizes.items():
            by_event[key[0]].append((key, pixels))
        strata: dict[tuple[str, str, str], int] = {}
        for event_rows in by_event.values():
            ranks = (rankdata([pixels for _, pixels in event_rows], method="average") - 0.5) / len(
                event_rows
            )
            for (key, _), percentile in zip(event_rows, ranks):
                strata[key] = min(2, int(percentile * 3.0))

        counters: dict[tuple[str, str, int, str], list[int]] = defaultdict(lambda: [0, 0])
        for row in rows:
            if int(row["damaged"]) != 1:
                continue
            key = (row["disaster"], row["uid"], row["object_id"])
            stratum = strata[key]
            for statistic in statistics:
                counter = counters[(row["disaster"], row["method"], stratum, statistic)]
                counter[0] += 1
                counter[1] += int(float(row[statistic]) >= 0.95)

        grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
        for (event, method, stratum, statistic), (total, hits) in counters.items():
            grouped[(method, stratum, statistic)].append(hits / max(1, total))
        for (method, stratum, statistic), recalls in sorted(grouped.items()):
            mean_recall = float(np.mean(recalls))
            output_rows.append(
                {
                    "split": split_name,
                    "method": method,
                    "statistic": statistic,
                    "size_stratum": ("small", "medium", "large")[stratum],
                    "events": len(recalls),
                    "mean_event_damaged_object_recall_at_5pct": mean_recall,
                }
            )
            plot_lookup[(split_name, method, statistic, stratum)] = mean_recall

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    x = np.arange(3)
    for row_index, split_name in enumerate(("train", "test")):
        for column_index, statistic in enumerate(statistics):
            axis = axes[row_index, column_index]
            for method in methods:
                axis.plot(
                    x,
                    [plot_lookup[(split_name, method, statistic, stratum)] for stratum in range(3)],
                    marker="o",
                    label=DISPLAY[method],
                )
            axis.set_xticks(x, ("Small", "Medium", "Large"))
            axis.set_title(f"{split_name.title()}: {statistic} object score")
            axis.set_ylabel("Damaged-building hit recall at 5%")
            axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle(
        "Object-size sensitivity of xBD-S12 candidate hits\n"
        "size tertiles are defined independently inside each disaster"
    )
    fig.savefig(output_figure, dpi=180)
    plt.close(fig)


def registration_profile(
    near_folder: Path,
    large_folder: Path,
    output_figure: Path,
    output_csv: Path,
) -> None:
    methods = (
        "band_image_projector_distance",
        "ir_mad",
        "pca_diff",
        "raw_l2",
    )
    pixel_rows = read_csv(near_folder / "summary_pixel_metrics.csv") + [
        row
        for row in read_csv(large_folder / "summary_pixel_metrics.csv")
        if float(row["magnitude"]) > 0
    ]
    object_rows = read_csv(near_folder / "summary_object_metrics.csv") + [
        row
        for row in read_csv(large_folder / "summary_object_metrics.csv")
        if float(row["magnitude"]) > 0
    ]
    pixel_lookup = {
        (float(row["magnitude"]), row["method"]): float(
            row["mean_event_direction_average_precision"]
        )
        for row in pixel_rows
    }
    object_lookup = {
        (float(row["magnitude"]), row["method"]): float(
            row["mean_event_direction_damaged_object_recall"]
        )
        for row in object_rows
        if row["statistic"] == "p90"
    }
    magnitudes = sorted({key[0] for key in pixel_lookup})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for method in methods:
        axes[0].plot(
            magnitudes,
            [pixel_lookup[(magnitude, method)] for magnitude in magnitudes],
            marker="o",
            label=DISPLAY[method],
        )
        axes[1].plot(
            magnitudes,
            [object_lookup[(magnitude, method)] for magnitude in magnitudes],
            marker="o",
            label=DISPLAY[method],
        )
    axes[0].set_title("Full-scene damaged-pixel AP")
    axes[1].set_title("Damaged-building p90 hit recall at 5%")
    for axis in axes:
        axis.set_xlabel("Injected post-image shift (Sentinel pixels)")
        axis.set_xticks(magnitudes)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean training event/direction metric")
    axes[1].set_ylabel("Mean training event/direction metric")
    axes[0].legend()
    fig.suptitle("xBD-S12 controlled registration sensitivity")
    fig.savefig(output_figure, dpi=180)
    plt.close(fig)

    degradation = read_csv(near_folder / "registration_degradation.csv") + [
        row
        for row in read_csv(large_folder / "registration_degradation.csv")
        if float(row["magnitude"]) > 0
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(degradation[0]))
        writer.writeheader()
        writer.writerows(degradation)


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
    if args.train_sweep is not None:
        train_rank_sensitivity(
            args.train_sweep, args.output_dir / "train_projector_rank_sensitivity.png"
        )
    if args.train_confirmation is not None:
        train_confirmation_profile(
            args.train_confirmation,
            args.output_dir / "train_task_confirmation.png",
        )
    if args.train_classical is not None and args.test_budget is not None:
        operational_budget_profile(
            args.train_classical,
            args.test_budget,
            args.output_dir / "operational_budget_metrics.png",
        )
    nuisance_stratification(
        args.unbuffered,
        args.output_dir / "nuisance_stratification.png",
        args.output_dir / "nuisance_stratification.csv",
    )
    if args.object_train is not None and args.object_test is not None:
        object_candidate_profile(
            args.object_train,
            args.object_test,
            args.output_dir / "object_candidate_tradeoff.png",
            args.output_dir / "object_candidate_pairwise.csv",
        )
        object_size_sensitivity(
            args.object_train,
            args.object_test,
            args.output_dir / "object_size_sensitivity.png",
            args.output_dir / "object_size_sensitivity.csv",
        )
    if args.registration_near is not None and args.registration_large is not None:
        registration_profile(
            args.registration_near,
            args.registration_large,
            args.output_dir / "registration_sensitivity.png",
            args.output_dir / "registration_degradation.csv",
        )
    write_summary(args.unbuffered, args.boundary, args.output_dir / "analysis_summary.json")
    print(f"Wrote xBD-S12 evidence summary to {args.output_dir}")


if __name__ == "__main__":
    main()
