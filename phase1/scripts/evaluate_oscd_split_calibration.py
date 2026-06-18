"""Evaluate split-safe threshold calibration for saved OSCD score maps.

Source/provenance:
- OSCD supplies official train/test city groups. Calibration choices are fitted
  on training cities and then frozen for test cities.
- F1 and IoU use the standard binary segmentation definitions in
  ``phase1.eval.metrics``.

Project evaluation choice:
- Different unsupervised methods produce scores on incompatible scales. Each
  city map is therefore converted to an exact top-fraction decision rule. The
  selected changed-area fraction is fitted on training cities only. This is a
  rank-based calibration diagnostic, not part of DS, PCA-diff, or IR-MAD theory.
- The method answers whether useful ranking evidence can be converted into a
  held-out binary mask without tuning on the evaluated city.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.data.oscd_dataset import OSCDEvaluatorDataset
from phase1.eval.metrics import binary_metrics
from phase1.eval.utils import suppress_rasterio_warnings
from phase1.scripts.compare_oscd_spatial_subspaces import BAND_ORDER, normalize_for_display
from phase1.scripts.sweep_oscd_spatial_subspaces import method_display_name


DEFAULT_METHODS = "raw_l2,pca_diff,band_image_norm,ir_mad,rank_fusion_pca_band_irmad"


@dataclass(frozen=True)
class RankedProfile:
    city: str
    split: str
    method: str
    valid_count: int
    positive_count: int
    cumulative_positive_desc: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True, help="Sweep containing runs/*/score_maps/*.npy.")
    parser.add_argument("--oscd-root", type=Path, default=Path("data/OSCD"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--config", default="", help="Optional sweep config name when the root contains multiple configs.")
    parser.add_argument("--grid-size", type=int, default=300)
    parser.add_argument("--max-fraction", type=float, default=0.5)
    parser.add_argument("--visual-cities", default="brasilia,dubai,norcia")
    return parser.parse_args()


def fraction_grid(size: int, maximum: float) -> np.ndarray:
    if size < 3:
        raise ValueError("grid size must be at least 3")
    if not 0.01 <= maximum <= 1.0:
        raise ValueError("maximum fraction must be in [0.01, 1]")
    low_count = max(2, size // 2)
    high_count = max(2, size - low_count + 1)
    low = np.geomspace(1e-4, min(0.05, maximum), low_count)
    high = np.linspace(min(0.05, maximum), maximum, high_count)
    return np.unique(np.concatenate(([0.0], low, high)))


def build_ranked_profile(city: str, split: str, method: str, score: np.ndarray, target: np.ndarray, mask: np.ndarray) -> RankedProfile:
    valid_indices = np.flatnonzero(mask.reshape(-1))
    if valid_indices.size == 0:
        raise ValueError(f"{city}/{method} has no valid pixels")
    values = score.reshape(-1)[valid_indices]
    labels = target.reshape(-1)[valid_indices].astype(np.uint8, copy=False)
    # Stable sorting makes exact top-k decisions reproducible when scores tie.
    order = np.argsort(-values, kind="mergesort")
    cumulative = np.cumsum(labels[order], dtype=np.int64)
    return RankedProfile(
        city=city,
        split=split,
        method=method,
        valid_count=int(valid_indices.size),
        positive_count=int(labels.sum()),
        cumulative_positive_desc=cumulative,
    )


def counts_at_fraction(profile: RankedProfile, fraction: float) -> tuple[int, int, int, int]:
    selected = int(np.floor(float(fraction) * profile.valid_count + 0.5))
    selected = min(max(selected, 0), profile.valid_count)
    tp = int(profile.cumulative_positive_desc[selected - 1]) if selected else 0
    fp = selected - tp
    fn = profile.positive_count - tp
    tn = profile.valid_count - tp - fp - fn
    return tp, fp, fn, tn


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    false_discovery_fraction = fp / (tp + fp) if tp + fp else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "false_positive_rate": false_positive_rate,
        "false_discovery_fraction": false_discovery_fraction,
    }


def macro_metrics(profiles: list[RankedProfile], fraction: float) -> dict[str, float]:
    rows = [metrics_from_counts(*counts_at_fraction(profile, fraction)) for profile in profiles]
    if not rows:
        return {key: float("nan") for key in metrics_from_counts(0, 0, 0, 0)}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def fit_fraction(profiles: list[RankedProfile], grid: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    if not profiles:
        raise ValueError("training profiles are required")
    curve: list[dict[str, float]] = []
    best_fraction = 0.0
    best_f1 = -1.0
    for fraction in grid:
        metrics = macro_metrics(profiles, float(fraction))
        curve.append({"fraction": float(fraction), **metrics})
        if metrics["f1"] > best_f1 + 1e-12:
            best_f1 = metrics["f1"]
            best_fraction = float(fraction)
    return best_fraction, curve


def exact_top_fraction_prediction(score: np.ndarray, mask: np.ndarray, fraction: float) -> np.ndarray:
    valid_indices = np.flatnonzero(mask.reshape(-1))
    selected = int(np.floor(float(fraction) * valid_indices.size + 0.5))
    selected = min(max(selected, 0), valid_indices.size)
    prediction = np.zeros(mask.size, dtype=np.uint8)
    if selected:
        values = score.reshape(-1)[valid_indices]
        order = np.argsort(-values, kind="mergesort")
        prediction[valid_indices[order[:selected]]] = 1
    return prediction.reshape(mask.shape)


def read_sweep_metrics(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    output: dict[tuple[str, str], dict[str, float]] = {}
    if not path.exists():
        return output
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output[(row["city"], row["method"])] = {
                "otsu_f1": float(row.get("otsu_f1", "nan")),
                "otsu_iou": float(row.get("otsu_iou", "nan")),
                "best_f1": float(row.get("best_f1", "nan")),
                "best_iou": float(row.get("best_iou", "nan")),
                "auroc": float(row.get("auroc", "nan")),
                "average_precision": float(row.get("average_precision", "nan")),
            }
    return output


def discover_runs(sweep_root: Path, config: str) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for metadata_path in sorted((sweep_root / "runs").glob("*/run_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        city = str(metadata["city"])
        run_config = metadata_path.parent.name.rsplit("__", 1)[0]
        if config and run_config != config:
            continue
        if city in runs:
            raise ValueError(f"Multiple runs found for {city}; specify --config")
        runs[city] = metadata_path.parent
    if not runs:
        raise FileNotFoundError(f"No completed runs found under {sweep_root / 'runs'}")
    return runs


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_test_comparisons(rows: list[dict[str, object]], baseline: str = "pca_diff", seed: int = 1234) -> list[dict[str, object]]:
    """Compare held-out city metrics against a fixed baseline."""
    test_rows = [row for row in rows if row["split"] == "test"]
    methods = sorted({str(row["method"]) for row in test_rows})
    by_key = {(str(row["city"]), str(row["method"])): row for row in test_rows}
    cities = sorted({str(row["city"]) for row in test_rows if str(row["method"]) == baseline})
    rng = np.random.default_rng(seed)
    output: list[dict[str, object]] = []
    for method in methods:
        if method == baseline:
            continue
        for metric in ("f1", "iou", "false_discovery_fraction"):
            deltas = np.asarray(
                [float(by_key[(city, method)][metric]) - float(by_key[(city, baseline)][metric]) for city in cities],
                dtype=np.float64,
            )
            nonzero = deltas[np.abs(deltas) > 1e-12]
            p_value = float(wilcoxon(nonzero).pvalue) if nonzero.size else 1.0
            samples = rng.choice(deltas, size=(10000, deltas.size), replace=True).mean(axis=1)
            ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
            output.append(
                {
                    "metric": metric,
                    "method": method,
                    "baseline": baseline,
                    "n_test_cities": int(deltas.size),
                    "mean_delta_method_minus_baseline": float(np.mean(deltas)),
                    "bootstrap_95ci_low": float(ci_low),
                    "bootstrap_95ci_high": float(ci_high),
                    "wins": int(np.sum(deltas > 1e-12)),
                    "ties": int(np.sum(np.abs(deltas) <= 1e-12)),
                    "losses": int(np.sum(deltas < -1e-12)),
                    "wilcoxon_p_two_sided": p_value,
                }
            )
    return output


def plot_curves(path: Path, curves: dict[str, list[dict[str, float]]], selected: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method, rows in curves.items():
        x = [row["fraction"] for row in rows]
        y = [row["f1"] for row in rows]
        ax.plot(x, y, label=method_display_name(method), linewidth=1.8)
        fraction = selected[method]
        chosen = min(rows, key=lambda row: abs(row["fraction"] - fraction))
        ax.scatter([fraction], [chosen["f1"]], s=30)
    ax.set_xlabel("predicted changed-area fraction (fitted on train cities)")
    ax.set_ylabel("mean train-city F1")
    ax.set_title("OSCD split-safe rank calibration")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_test_heatmap(path: Path, rows: list[dict[str, object]], methods: list[str], cities: list[str]) -> None:
    matrix = np.full((len(methods), len(cities)), np.nan, dtype=np.float64)
    method_index = {method: index for index, method in enumerate(methods)}
    city_index = {city: index for index, city in enumerate(cities)}
    for row in rows:
        if row["split"] == "test":
            matrix[method_index[str(row["method"])], city_index[str(row["city"])]] = float(row["f1"])
    fig, ax = plt.subplots(figsize=(max(10, len(cities) * 0.8), max(4, len(methods) * 0.65)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(0.05, float(np.nanmax(matrix))))
    ax.set_xticks(np.arange(len(cities)), cities, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(methods)), [method_display_name(method) for method in methods])
    ax.set_title("Held-out test-city F1 using train-fitted changed-area fractions")
    fig.colorbar(image, ax=ax, label="F1", fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def error_image(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = np.zeros((*target.shape, 3), dtype=np.float32)
    target_bool = target.astype(bool)
    pred_bool = prediction.astype(bool)
    image[np.logical_and(pred_bool, target_bool) & mask] = (0.1, 0.8, 0.2)  # TP
    image[np.logical_and(pred_bool, ~target_bool) & mask] = (0.95, 0.15, 0.1)  # FP
    image[np.logical_and(~pred_bool, target_bool) & mask] = (0.1, 0.4, 1.0)  # FN
    return image


def plot_city_diagnostic(path: Path, city: str, target: np.ndarray, mask: np.ndarray, methods: list[str], score_paths: dict[str, Path], fractions: dict[str, float]) -> None:
    fig, axes = plt.subplots(len(methods), 3, figsize=(12, max(3.2, len(methods) * 2.8)), squeeze=False)
    for row_index, method in enumerate(methods):
        score = np.load(score_paths[method])
        prediction = exact_top_fraction_prediction(score, mask, fractions[method])
        axes[row_index, 0].imshow(normalize_for_display(score, mask, 99.5), cmap="magma", vmin=0, vmax=1)
        axes[row_index, 0].set_title(f"{method_display_name(method)} score")
        axes[row_index, 1].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].set_title(f"prediction, top {fractions[method] * 100:.1f}%")
        axes[row_index, 2].imshow(error_image(prediction, target, mask))
        axes[row_index, 2].set_title("TP green / FP red / FN blue")
        for axis in axes[row_index]:
            axis.axis("off")
    fig.suptitle(f"{city}: split-safe calibration diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    suppress_rasterio_warnings()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    runs = discover_runs(args.sweep_root, args.config)
    source_metrics = read_sweep_metrics(args.sweep_root / "sweep_metrics_all.csv")

    datasets = {
        split: OSCDEvaluatorDataset(args.oscd_root, split=split, band_order=BAND_ORDER)
        for split in ("train", "test")
    }
    city_lookup = {city: split for split, dataset in datasets.items() for city in dataset.cities}
    sample_cache: dict[str, object] = {}
    profiles: dict[str, list[RankedProfile]] = {method: [] for method in methods}
    score_paths_by_city: dict[str, dict[str, Path]] = {}

    for city, run_dir in sorted(runs.items()):
        split = city_lookup.get(city)
        if split not in datasets:
            continue
        sample = datasets[split].load_city(city)
        sample_cache[city] = sample
        target = sample.y[0].astype(np.uint8)
        mask = sample.valid_mask.astype(bool)
        score_paths_by_city[city] = {}
        for method in methods:
            score_path = run_dir / "score_maps" / f"{method}.npy"
            if not score_path.exists():
                raise FileNotFoundError(f"Missing {score_path}; rerun the sweep with --save-npy")
            score = np.load(score_path, mmap_mode="r")
            if score.shape != target.shape:
                raise ValueError(f"Shape mismatch for {city}/{method}: {score.shape} vs {target.shape}")
            profiles[method].append(build_ranked_profile(city, split, method, score, target, mask))
            score_paths_by_city[city][method] = score_path

    grid = fraction_grid(args.grid_size, args.max_fraction)
    selected: dict[str, float] = {}
    curves: dict[str, list[dict[str, float]]] = {}
    per_city_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for method in methods:
        train_profiles = [profile for profile in profiles[method] if profile.split == "train"]
        test_profiles = [profile for profile in profiles[method] if profile.split == "test"]
        fraction, curve = fit_fraction(train_profiles, grid)
        selected[method] = fraction
        curves[method] = curve
        train_metrics = macro_metrics(train_profiles, fraction)
        test_metrics = macro_metrics(test_profiles, fraction)

        for profile in profiles[method]:
            counts = counts_at_fraction(profile, fraction)
            row_metrics = metrics_from_counts(*counts)
            source = source_metrics.get((profile.city, method), {})
            per_city_rows.append(
                {
                    "city": profile.city,
                    "split": profile.split,
                    "method": method,
                    "fitted_fraction": fraction,
                    "label_positive_rate": profile.positive_count / profile.valid_count,
                    **row_metrics,
                    **source,
                }
            )

        test_source = [source_metrics[(profile.city, method)] for profile in test_profiles if (profile.city, method) in source_metrics]
        summary_rows.append(
            {
                "method": method,
                "display_name": method_display_name(method),
                "train_fitted_fraction": fraction,
                **{f"train_macro_{key}": value for key, value in train_metrics.items()},
                **{f"test_macro_{key}": value for key, value in test_metrics.items()},
                "test_mean_otsu_f1": float(np.mean([row["otsu_f1"] for row in test_source])) if test_source else float("nan"),
                "test_mean_oracle_f1": float(np.mean([row["best_f1"] for row in test_source])) if test_source else float("nan"),
                "test_mean_auroc": float(np.mean([row["auroc"] for row in test_source])) if test_source else float("nan"),
                "test_mean_average_precision": float(np.mean([row["average_precision"] for row in test_source])) if test_source else float("nan"),
            }
        )

    write_rows(args.output_dir / "split_calibration_summary.csv", summary_rows)
    write_rows(args.output_dir / "split_calibration_per_city.csv", per_city_rows)
    pairwise_rows = paired_test_comparisons(per_city_rows)
    write_rows(args.output_dir / "split_calibration_paired_test.csv", pairwise_rows)
    plot_curves(args.output_dir / "train_fraction_calibration_curves.png", curves, selected)
    test_cities = sorted({row["city"] for row in per_city_rows if row["split"] == "test"})
    plot_test_heatmap(args.output_dir / "test_city_calibrated_f1_heatmap.png", per_city_rows, methods, test_cities)

    visual_cities = [city.strip() for city in args.visual_cities.split(",") if city.strip()]
    for city in visual_cities:
        if city not in sample_cache or city not in score_paths_by_city:
            continue
        sample = sample_cache[city]
        plot_city_diagnostic(
            args.output_dir / f"calibration_diagnostic_{city}.png",
            city,
            sample.y[0].astype(np.uint8),
            sample.valid_mask.astype(bool),
            methods,
            score_paths_by_city[city],
            selected,
        )

    metadata = {
        "sweep_root": str(args.sweep_root),
        "oscd_root": str(args.oscd_root),
        "methods": methods,
        "train_cities": sorted(city for city in runs if city_lookup.get(city) == "train"),
        "test_cities": sorted(city for city in runs if city_lookup.get(city) == "test"),
        "fraction_grid": {"size": int(grid.size), "minimum_nonzero": float(grid[1]), "maximum": float(grid[-1])},
        "selected_fractions": selected,
        "evaluation_rule": "fit predicted changed-area fraction by macro F1 on official OSCD train cities; freeze and evaluate on official test cities",
        "limitations": [
            "The calibrated fraction uses training labels and is not an unsupervised threshold.",
            "OSCD test labels are used only for final evaluation and diagnostics.",
            "False positives are not automatically semantic pseudo-change categories; maps require visual/source-data inspection.",
        ],
        "summary": summary_rows,
        "paired_test_comparisons": pairwise_rows,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote split-safe calibration results: {args.output_dir}")


if __name__ == "__main__":
    main()
