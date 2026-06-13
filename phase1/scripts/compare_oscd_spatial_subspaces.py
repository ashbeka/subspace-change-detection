"""
Compare spatial DS constructions on one OSCD city.

This script treats OSCD change detection as binary pixel segmentation with
continuous prior scores. Each method produces a score map. The script compares
those maps against OSCD labels using threshold-free ranking metrics, thresholded
segmentation metrics, simple baselines, and construction metadata.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics as skm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.baselines.pca_diff import pca_diff_score
from phase1.data.oscd_dataset import OSCDEvaluatorDataset
from phase1.data.preprocessing import apply_normalization, load_band_stats
from phase1.ds import pca_utils
from phase1.ds.ds_scores import DSConfig, compute_ds_scores
from phase1.eval.metrics import binary_metrics
from phase1.eval.thresholding import apply_threshold, otsu_threshold
from phase1.eval.utils import suppress_rasterio_warnings


BAND_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
RGB_INDICES = (3, 2, 1)  # B04, B03, B02
DEFAULT_METHODS = "global_pixel,window128,patch3,patch5"


@dataclass
class MethodSpec:
    name: str
    family: str
    window_size: int | None = None
    stride: int | None = None
    patch_size: int | None = None
    aggregator: str = "mean"


@dataclass
class ConstructionCard:
    method: str
    source_reference: str
    sample_unit: str
    input_object: str
    subspace_count: str
    basis_shape: str
    fitting_method: str
    score_definition: str
    spatial_information: str
    code_path: str
    verification: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare global, window-local, and patch-vector DS score maps on one OSCD city.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--oscd_root", "--oscd-root", type=Path, default=Path("data/OSCD"))
    ap.add_argument("--stats_path", "--stats-path", type=Path, default=Path("phase1/data/oscd_band_stats.json"))
    ap.add_argument("--city", default="beirut")
    ap.add_argument("--split", default="auto", choices=["auto", "train", "val", "test"])
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--methods", default=DEFAULT_METHODS, help="Comma list: global_pixel, window128, window128s64, window128s64max, patch3, patch5.")
    ap.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_fit_samples", "--max-fit-samples", type=int, default=20000, help="Maximum pixel/patch samples used to fit patch PCA subspaces.")
    ap.add_argument("--score_chunk_size", "--score-chunk-size", type=int, default=25000, help="Patch scoring chunk size.")
    ap.add_argument("--score_percentile", "--score-percentile", type=float, default=99.0, help="Percentile used for display normalization.")
    ap.add_argument("--save_npy", "--save-npy", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def _iter_splits(split: str) -> Iterable[str]:
    if split == "auto":
        return ("train", "val", "test")
    return (split,)


def _load_city(args: argparse.Namespace):
    for split in _iter_splits(args.split):
        ds = OSCDEvaluatorDataset(
            root=args.oscd_root,
            split=split,
            band_order=BAND_ORDER,
            nodata_value=0.0,
            min_valid_bands=3,
            val_from_train=3,
        )
        if args.city in ds.cities:
            return split, ds.load_city(args.city)
    raise ValueError(f"City {args.city!r} was not found in split={args.split!r}.")


def parse_method_spec(text: str) -> MethodSpec:
    key = text.strip().lower().replace("-", "_")
    if key == "global" or key == "global_pixel":
        return MethodSpec(name="global_pixel", family="global_pixel")

    m = re.fullmatch(r"window(?P<size>\d+)(?:s(?P<stride>\d+))?(?P<agg>max|mean)?", key)
    if m:
        size = int(m.group("size"))
        stride = int(m.group("stride") or max(1, size // 2))
        agg = m.group("agg") or "mean"
        return MethodSpec(name=f"window{size}s{stride}{agg}", family="local_window", window_size=size, stride=stride, aggregator=agg)

    m = re.fullmatch(r"patch(?P<size>\d+)", key)
    if m:
        size = int(m.group("size"))
        if size % 2 != 1:
            raise ValueError(f"Patch size must be odd, got {size}.")
        return MethodSpec(name=f"patch{size}", family="patch_vector", patch_size=size)

    raise ValueError(f"Unknown method spec {text!r}.")


def normalize_for_display(score: np.ndarray, valid_mask: np.ndarray, percentile: float) -> np.ndarray:
    out = np.zeros_like(score, dtype=np.float32)
    vals = score[valid_mask].astype(np.float64)
    if vals.size == 0:
        return out
    low = float(np.nanmin(vals))
    high = float(np.nanpercentile(vals, percentile))
    if not np.isfinite(high) or high <= low:
        high = float(np.nanmax(vals))
    if high <= low:
        return out
    out[valid_mask] = np.clip((score[valid_mask] - low) / (high - low), 0.0, 1.0)
    return out


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def score_metrics(score: np.ndarray, target: np.ndarray, valid_mask: np.ndarray, raw_l2: np.ndarray) -> dict[str, float | int]:
    y = target[valid_mask].astype(np.uint8).reshape(-1)
    s = score[valid_mask].astype(np.float64).reshape(-1)
    raw = raw_l2[valid_mask].astype(np.float64).reshape(-1)

    result: dict[str, float | int] = {
        "valid_pixels": int(y.size),
        "positive_pixels": int(np.sum(y == 1)),
        "positive_rate": float(np.mean(y == 1)) if y.size else float("nan"),
        "raw_l2_corr": safe_corr(s, raw),
    }
    if y.size == 0 or np.unique(y).size < 2 or float(np.std(s)) == 0.0:
        result.update(
            {
                "auroc": float("nan"),
                "average_precision": float("nan"),
                "best_f1": float("nan"),
                "best_iou": float("nan"),
                "best_threshold": float("nan"),
            }
        )
    else:
        result["auroc"] = float(skm.roc_auc_score(y, s))
        result["average_precision"] = float(skm.average_precision_score(y, s))
        precision, recall, thresholds = skm.precision_recall_curve(y, s)
        f1 = (2.0 * precision * recall) / (precision + recall + 1e-12)
        best_idx = int(np.nanargmax(f1))
        result["best_f1"] = float(f1[best_idx])
        result["best_iou"] = float(f1[best_idx] / (2.0 - f1[best_idx] + 1e-12))
        if thresholds.size == 0:
            result["best_threshold"] = float("nan")
        else:
            result["best_threshold"] = float(thresholds[min(best_idx, thresholds.size - 1)])

    otsu = float(otsu_threshold(score, valid_mask))
    otsu_pred = apply_threshold(score, otsu)
    otsu_metrics = binary_metrics(otsu_pred, target, valid_mask=valid_mask)
    result["otsu_threshold"] = otsu
    for key, value in otsu_metrics.items():
        result[f"otsu_{key}"] = value
    return result


def raw_l2_score(x1: np.ndarray, x2: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    diff = x2 - x1
    score = np.sqrt(np.sum(diff * diff, axis=0, dtype=np.float64)).astype(np.float32)
    score[~valid_mask] = 0.0
    return score


def global_pixel_ds_score(x1: np.ndarray, x2: np.ndarray, valid_mask: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, ConstructionCard]:
    cfg = DSConfig(rank_r=args.rank, variance_threshold=None, random_state=args.seed, subspace_variant="canonical", score_normalization=None)
    out = compute_ds_scores(x1, x2, valid_mask=valid_mask, cfg=cfg, normalize=False)
    card = ConstructionCard(
        method="global_pixel",
        source_reference="Fukui/Maki first-order DS principal-vector construction, adapted by using Sentinel-2 pixel vectors as samples.",
        sample_unit="one valid pixel location",
        input_object="X_pre, X_post in R^(13 x N), columns are 13-band pixel vectors",
        subspace_count="one PCA subspace for pre date and one PCA subspace for post date",
        basis_shape=f"Phi, Psi in R^(13 x {args.rank}); D in R^(13 x <= {args.rank})",
        fitting_method="rank-r PCA on all valid pixel vectors per date image",
        score_definition="per pixel: ||D^T (x_post - x_pre)||^2",
        spatial_information="pixel position is not used while fitting; position returns only when scores are reshaped to the image grid",
        code_path="phase1.ds.ds_scores.compute_ds_scores + phase1.ds.pca_utils.difference_subspace_canonical",
        verification="compared against raw L2, PCA-diff, OSCD labels, and spatial variants in this script",
    )
    return out["projection"].astype(np.float32), card


def _window_positions(length: int, window: int, stride: int) -> list[int]:
    starts = list(range(0, max(1, length - window + 1), stride))
    last = max(0, length - window)
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def local_window_ds_score(x1: np.ndarray, x2: np.ndarray, valid_mask: np.ndarray, args: argparse.Namespace, spec: MethodSpec) -> tuple[np.ndarray, ConstructionCard]:
    assert spec.window_size is not None and spec.stride is not None
    cfg = DSConfig(rank_r=args.rank, variance_threshold=None, random_state=args.seed, subspace_variant="canonical", score_normalization=None)
    h, w = valid_mask.shape
    if spec.aggregator == "mean":
        acc = np.zeros((h, w), dtype=np.float64)
        counts = np.zeros((h, w), dtype=np.float32)
    elif spec.aggregator == "max":
        acc = np.full((h, w), -np.inf, dtype=np.float64)
        counts = np.zeros((h, w), dtype=np.float32)
    else:
        raise ValueError(f"Unknown aggregator: {spec.aggregator}")

    y_starts = _window_positions(h, spec.window_size, spec.stride)
    x_starts = _window_positions(w, spec.window_size, spec.stride)
    windows_used = 0
    windows_skipped = 0
    min_pixels = max(args.rank + 1, 16)

    for y0 in y_starts:
        for x0 in x_starts:
            sl_y = slice(y0, y0 + spec.window_size)
            sl_x = slice(x0, x0 + spec.window_size)
            sub_valid = valid_mask[sl_y, sl_x]
            if int(np.sum(sub_valid)) < min_pixels:
                windows_skipped += 1
                continue
            sub_scores = compute_ds_scores(x1[:, sl_y, sl_x], x2[:, sl_y, sl_x], valid_mask=sub_valid, cfg=cfg, normalize=False)
            score = sub_scores["projection"].astype(np.float64, copy=False)
            if spec.aggregator == "mean":
                acc[sl_y, sl_x] += score
                counts[sl_y, sl_x] += sub_valid.astype(np.float32)
            else:
                current = acc[sl_y, sl_x]
                current[sub_valid] = np.maximum(current[sub_valid], score[sub_valid])
                acc[sl_y, sl_x] = current
                counts[sl_y, sl_x] += sub_valid.astype(np.float32)
            windows_used += 1

    if spec.aggregator == "mean":
        out = np.divide(acc, counts, out=np.zeros_like(acc), where=counts > 0)
    else:
        out = np.where(np.isfinite(acc), acc, 0.0)

    card = ConstructionCard(
        method=spec.name,
        source_reference="Local spatial modeling adaptation of first-order DS; uses the same canonical DS formula but fits subspaces inside spatial windows.",
        sample_unit=f"one valid pixel inside a {spec.window_size}x{spec.window_size} window",
        input_object="For each window: X_pre_w, X_post_w in R^(13 x N_w)",
        subspace_count=f"one pre/post PCA subspace pair per sliding window; windows used={windows_used}, skipped={windows_skipped}",
        basis_shape=f"per window Phi_w, Psi_w in R^(13 x {args.rank}); D_w in R^(13 x <= {args.rank})",
        fitting_method=f"rank-r PCA inside each window, stride={spec.stride}, overlap aggregation={spec.aggregator}",
        score_definition="per covered pixel/window: ||D_w^T (x_post - x_pre)||^2, then aggregate overlapping scores",
        spatial_information="preserves coarse regional context because nearby pixels share a local subspace model; exact pixel coordinates are still not features",
        code_path="phase1/scripts/compare_oscd_spatial_subspaces.py::local_window_ds_score",
        verification="evaluated against global DS, raw L2, PCA-diff, OSCD labels, raw-L2 correlation, runtime, and visual maps",
    )
    return out.astype(np.float32), card


def patch_valid_mask(valid_mask: np.ndarray, patch_size: int) -> np.ndarray:
    margin = patch_size // 2
    h, w = valid_mask.shape
    out = np.zeros_like(valid_mask, dtype=bool)
    if h <= 2 * margin or w <= 2 * margin:
        return out
    inner = valid_mask[margin : h - margin, margin : w - margin].copy()
    for dy in range(-margin, margin + 1):
        for dx in range(-margin, margin + 1):
            inner &= valid_mask[margin + dy : h - margin + dy, margin + dx : w - margin + dx]
    out[margin : h - margin, margin : w - margin] = inner
    return out


def patch_matrix(cube: np.ndarray, rows: np.ndarray, cols: np.ndarray, patch_size: int) -> np.ndarray:
    margin = patch_size // 2
    c = cube.shape[0]
    mat = np.empty((c * patch_size * patch_size, rows.size), dtype=np.float32)
    pos = 0
    for dy in range(-margin, margin + 1):
        for dx in range(-margin, margin + 1):
            mat[pos : pos + c] = cube[:, rows + dy, cols + dx]
            pos += c
    return mat


def patch_vector_ds_score(x1: np.ndarray, x2: np.ndarray, valid_mask: np.ndarray, args: argparse.Namespace, spec: MethodSpec) -> tuple[np.ndarray, ConstructionCard, np.ndarray]:
    assert spec.patch_size is not None
    rng = np.random.default_rng(args.seed)
    pv = patch_valid_mask(valid_mask, spec.patch_size)
    rows, cols = np.where(pv)
    if rows.size == 0:
        raise RuntimeError(f"No valid patch centers for {spec.name}.")

    fit_n = min(int(args.max_fit_samples), rows.size)
    fit_idx = rng.choice(rows.size, size=fit_n, replace=False)
    fit_rows = rows[fit_idx]
    fit_cols = cols[fit_idx]
    x1_fit = patch_matrix(x1, fit_rows, fit_cols, spec.patch_size)
    x2_fit = patch_matrix(x2, fit_rows, fit_cols, spec.patch_size)

    pre = pca_utils.fit_pca_basis(x1_fit, rank=args.rank, variance_threshold=None, random_state=args.seed, use_randomized=True)
    post = pca_utils.fit_pca_basis(x2_fit, rank=args.rank, variance_threshold=None, random_state=args.seed, use_randomized=True)
    d_basis = pca_utils.build_difference_subspace(pre.basis, post.basis, variant="canonical")

    out = np.zeros(valid_mask.shape, dtype=np.float32)
    for start in range(0, rows.size, int(args.score_chunk_size)):
        end = min(start + int(args.score_chunk_size), rows.size)
        rr = rows[start:end]
        cc = cols[start:end]
        diff = patch_matrix(x2, rr, cc, spec.patch_size) - patch_matrix(x1, rr, cc, spec.patch_size)
        if d_basis.shape[1] == 0:
            energy = np.zeros(rr.size, dtype=np.float32)
        else:
            coeff = d_basis.T @ diff
            energy = np.sum(coeff * coeff, axis=0).astype(np.float32)
        out[rr, cc] = energy

    d = 13 * spec.patch_size * spec.patch_size
    card = ConstructionCard(
        method=spec.name,
        source_reference="Spatial-support adaptation of first-order DS; uses the same canonical DS formula but changes the sample vector from a pixel to a local patch.",
        sample_unit=f"one centered {spec.patch_size}x{spec.patch_size} patch",
        input_object=f"X_pre_patch, X_post_patch in R^({d} x N_patch), columns are flattened local 13-band patches",
        subspace_count="one PCA subspace for all sampled pre-date patches and one PCA subspace for all sampled post-date patches",
        basis_shape=f"Phi, Psi in R^({d} x {args.rank}); D in R^({d} x <= {args.rank})",
        fitting_method=f"rank-r PCA on up to {fit_n} valid patch vectors per date image; all valid patch centers are scored in chunks",
        score_definition="per patch center: ||D^T (patch_post - patch_pre)||^2",
        spatial_information="preserves local neighborhood layout inside each flattened patch; border/invalid centers are excluded from patch evaluation",
        code_path="phase1/scripts/compare_oscd_spatial_subspaces.py::patch_vector_ds_score",
        verification="evaluated against global DS, local-window DS, raw L2, PCA-diff, OSCD labels, raw-L2 correlation, runtime, and visual maps",
    )
    return out, card, pv


def rgb_preview(cube: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    rgb = np.stack([cube[i] for i in RGB_INDICES], axis=-1).astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.float32)
    for ch in range(3):
        vals = rgb[..., ch][valid_mask]
        if vals.size == 0:
            continue
        lo, hi = np.percentile(vals, [2, 98])
        if hi <= lo:
            continue
        out[..., ch] = np.clip((rgb[..., ch] - lo) / (hi - lo), 0.0, 1.0)
    return out


def save_score_png(path: Path, score: np.ndarray, valid_mask: np.ndarray, percentile: float, cmap: str = "magma") -> None:
    disp = normalize_for_display(score, valid_mask, percentile)
    plt.figure(figsize=(8, 7))
    plt.imshow(disp, cmap=cmap, vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=160)
    plt.close()


def save_comparison_grid(path: Path, x_pre: np.ndarray, x_post: np.ndarray, target: np.ndarray, valid_mask: np.ndarray, maps: list[tuple[str, np.ndarray]], percentile: float) -> None:
    panels: list[tuple[str, np.ndarray, str | None]] = [
        ("pre RGB preview", rgb_preview(x_pre, valid_mask), None),
        ("post RGB preview", rgb_preview(x_post, valid_mask), None),
        ("OSCD label", target.astype(np.float32), "gray"),
    ]
    for name, score in maps:
        panels.append((name, normalize_for_display(score, valid_mask, percentile), "magma"))

    cols = 4
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, image, cmap) in zip(axes.ravel(), panels):
        ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    suppress_rasterio_warnings()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = args.output_dir / "score_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    split, sample = _load_city(args)
    if sample.y is None:
        raise RuntimeError(f"OSCD labels are required for comparison, but city {sample.city!r} has no loaded label.")
    target = sample.y[0].astype(np.uint8)
    eval_mask = sample.valid_mask.astype(bool)

    stats = load_band_stats(args.stats_path)
    x1_norm, valid_mask = apply_normalization(sample.x_pre, stats, valid_mask=sample.valid_mask, nodata_value=0.0)
    x2_norm, _ = apply_normalization(sample.x_post, stats, valid_mask=sample.valid_mask, nodata_value=0.0)
    eval_mask &= valid_mask

    print(f"Loaded {sample.city}/{split}: pre={sample.x_pre.shape}, post={sample.x_post.shape}, valid={int(eval_mask.sum())}")
    print("Task: binary pixel change-segmentation scoring on OSCD labels.")
    print("Meaningful here means: label ranking, thresholded segmentation, baseline pressure, spatial construction clarity, and map inspection.")

    methods = [parse_method_spec(x) for x in args.methods.split(",") if x.strip()]
    rows: list[dict[str, object]] = []
    cards: list[ConstructionCard] = []
    map_outputs: list[tuple[str, np.ndarray]] = []

    raw_l2 = raw_l2_score(x1_norm, x2_norm, eval_mask)
    pca_diff = pca_diff_score(x1_norm, x2_norm, eval_mask, rank_S=args.rank, variance_threshold=None, random_state=args.seed)
    baselines = [("raw_l2", "baseline", raw_l2), ("pca_diff", "baseline", pca_diff)]

    for name, family, score in baselines:
        metrics = score_metrics(score, target, eval_mask, raw_l2)
        rows.append({"method": name, "family": family, "runtime_sec": 0.0, "eval_valid_fraction": float(np.mean(eval_mask)), **metrics})
        map_outputs.append((name, score))
        save_score_png(maps_dir / f"{name}.png", score, eval_mask, args.score_percentile)
        if args.save_npy:
            np.save(maps_dir / f"{name}.npy", score.astype(np.float32))

    for spec in methods:
        print(f"Running {spec.name}...")
        start = time.perf_counter()
        method_mask = eval_mask
        if spec.family == "global_pixel":
            score, card = global_pixel_ds_score(x1_norm, x2_norm, eval_mask, args)
        elif spec.family == "local_window":
            score, card = local_window_ds_score(x1_norm, x2_norm, eval_mask, args, spec)
        elif spec.family == "patch_vector":
            score, card, method_mask = patch_vector_ds_score(x1_norm, x2_norm, eval_mask, args, spec)
            method_mask = method_mask & eval_mask
        else:
            raise AssertionError(f"Unhandled method family: {spec.family}")
        runtime = time.perf_counter() - start
        metrics = score_metrics(score, target, method_mask, raw_l2)
        rows.append({"method": spec.name, "family": spec.family, "runtime_sec": float(runtime), "eval_valid_fraction": float(np.mean(method_mask)), **metrics})
        cards.append(card)
        map_outputs.append((spec.name, score))
        save_score_png(maps_dir / f"{spec.name}.png", score, method_mask, args.score_percentile)
        if args.save_npy:
            np.save(maps_dir / f"{spec.name}.npy", score.astype(np.float32))
        print(f"  done in {runtime:.2f}s, AUROC={metrics['auroc']:.4f}, AP={metrics['average_precision']:.4f}, Otsu F1={metrics['otsu_f1']:.4f}")

    csv_path = args.output_dir / "spatial_subspace_metrics.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "method",
        "family",
        "auroc",
        "average_precision",
        "best_f1",
        "best_iou",
        "otsu_f1",
        "otsu_iou",
        "otsu_precision",
        "otsu_recall",
        "raw_l2_corr",
        "runtime_sec",
        "valid_pixels",
        "positive_pixels",
        "positive_rate",
        "eval_valid_fraction",
    ]
    ordered = [x for x in preferred if x in fieldnames] + [x for x in fieldnames if x not in preferred]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)

    save_comparison_grid(args.output_dir / "comparison_grid.png", sample.x_pre, sample.x_post, target, eval_mask, map_outputs, args.score_percentile)

    metadata = {
        "city": sample.city,
        "split": split,
        "task": "binary pixel-level change segmentation scoring",
        "input_shape": {"pre": list(sample.x_pre.shape), "post": list(sample.x_post.shape), "label": list(target.shape)},
        "band_order": BAND_ORDER,
        "rank": args.rank,
        "methods": [asdict(x) for x in methods],
        "output_files": {
            "metrics_csv": str(csv_path),
            "comparison_grid": str(args.output_dir / "comparison_grid.png"),
            "score_maps_dir": str(maps_dir),
        },
        "meaningful_criteria": {
            "label_alignment": "AUROC and average precision test whether changed pixels receive higher scores than unchanged pixels across thresholds.",
            "thresholdability": "Otsu F1/IoU tests whether an unsupervised threshold can turn the score map into a binary change mask.",
            "oracle_upper_bound": "best F1/IoU is an optimistic diagnostic threshold chosen with labels, not a deployable score.",
            "baseline_pressure": "raw_l2 and pca_diff are included so DS is not only compared against itself.",
            "redundancy_check": "raw_l2_corr checks whether a DS score is nearly the same as naive spectral difference.",
            "spatial_support": "construction cards describe whether pixel position/neighborhood information is used during fitting or scoring.",
        },
        "metric_sources": {
            "OSCD/Daudt": "OSCD is a pixel-level binary change-detection benchmark; Daudt et al. report precision/recall/F1 for the change class.",
            "scikit-learn": "AUROC and average precision follow sklearn.metrics.roc_auc_score and average_precision_score definitions.",
            "Otsu": "Otsu thresholding maximizes between-class gray-level variance for unsupervised binary threshold selection.",
        },
        "construction_cards": [asdict(x) for x in cards],
    }
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print()
    print(f"Wrote metrics: {csv_path}")
    print(f"Wrote metadata: {args.output_dir / 'run_metadata.json'}")
    print(f"Wrote comparison grid: {args.output_dir / 'comparison_grid.png'}")


if __name__ == "__main__":
    main()
