"""
Diagnose OSCD Pseudo-Change vs. Semantic Change using Component-Wise Difference Subspace (CW-DS).

Source/provenance:
- Project hypothesis: Monolithic DS underperforms because the highest-variance
  difference components capture pseudo-changes (seasonality, illumination).
- This script decouples the $r$ subspace components, correlates them with OSCD
  ground truth (urban change) and NDVI difference (pseudo-change), and
  tests whether filtering the components improves AP.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.baselines.pca_diff import pca_diff_score
from phase1.data.oscd_dataset import OSCDEvaluatorDataset
from phase1.data.preprocessing import apply_normalization, load_band_stats, build_valid_mask
from phase1.ds.ds_scores import DSConfig, compute_ds_scores
from phase1.eval.metrics import binary_metrics
from phase1.eval.utils import suppress_rasterio_warnings

BAND_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
RGB_INDICES = (3, 2, 1)  # B04, B03, B02
B08_IDX = BAND_ORDER.index("B08")
B04_IDX = BAND_ORDER.index("B04")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oscd_root", type=Path, default=Path("data/OSCD"))
    ap.add_argument("--stats_path", type=Path, default=Path("phase1/data/oscd_band_stats.json"))
    ap.add_argument("--city", default="beirut")
    ap.add_argument("--split", default="auto", choices=["auto", "train", "val", "test"])
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--output_dir", type=Path, required=True)
    return ap.parse_args()

def _load_city(args: argparse.Namespace):
    for split in ["train", "val", "test"] if args.split == "auto" else [args.split]:
        ds = OSCDEvaluatorDataset(
            root=args.oscd_root,
            split=split,
            band_order=BAND_ORDER,
            nodata_value=0.0,
            min_valid_bands=3,
        )
        if args.city in ds.cities:
            return split, ds.load_city(args.city)
    raise ValueError(f"City {args.city!r} not found.")

def compute_raw_l2(x1, x2):
    return np.sqrt(np.sum((x2 - x1)**2, axis=0))

def compute_ndvi(x):
    b08 = x[B08_IDX]
    b04 = x[B04_IDX]
    return (b08 - b04) / (b08 + b04 + 1e-8)

def normalize_map(score: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    s = score.astype(np.float32)
    high = np.percentile(s[s > 0] if np.any(s > 0) else s, percentile)
    if high <= 0:
        return np.zeros_like(s)
    return np.clip(s, 0, high) / high

def make_rgb(x: np.ndarray) -> np.ndarray:
    rgb = x[list(RGB_INDICES), :, :].transpose(1, 2, 0)
    rgb = np.clip(rgb * 3.0, 0, 1)  # simple brighten
    return rgb

def main() -> None:
    suppress_rasterio_warnings()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.city}...")
    split, sample = _load_city(args)
    stats = load_band_stats(args.stats_path)

    x1, _ = apply_normalization(sample.x_pre, stats, valid_mask=sample.valid_mask)
    x2, _ = apply_normalization(sample.x_post, stats, valid_mask=sample.valid_mask)
    gt = sample.y
    valid_mask = sample.valid_mask

    # Compute NDVI diff
    ndvi1 = compute_ndvi(sample.x_pre)
    ndvi2 = compute_ndvi(sample.x_post)
    delta_ndvi = np.abs(ndvi2 - ndvi1)

    print(f"Computing Component-Wise DS (rank {args.rank})...")
    cfg = DSConfig(rank_r=args.rank, score_normalization=None)
    ds_res = compute_ds_scores(x1, x2, valid_mask=valid_mask, cfg=cfg, normalize=False, return_components=True)

    components = ds_res["components"]  # (r, H, W)
    full_ds = ds_res["projection"]

    r = components.shape[0]

    # Correlate components
    v_mask = valid_mask.flatten()
    gt_flat = gt.flatten()[v_mask]
    ndvi_flat = delta_ndvi.flatten()[v_mask]

    comp_corrs = []
    print("\nComponent Correlations (Valid pixels only):")
    for i in range(r):
        c_flat = components[i].flatten()[v_mask]

        # Pearson correlation
        if np.std(c_flat) < 1e-8 or np.std(gt_flat) < 1e-8:
            corr_gt = 0.0
        else:
            corr_gt, _ = pearsonr(c_flat, gt_flat)

        if np.std(c_flat) < 1e-8 or np.std(ndvi_flat) < 1e-8:
            corr_ndvi = 0.0
        else:
            corr_ndvi, _ = pearsonr(c_flat, ndvi_flat)

        comp_corrs.append((corr_gt, corr_ndvi))
        print(f"  Component {i}: corr_GT = {corr_gt:.4f}, corr_NDVI = {corr_ndvi:.4f}")

    # Filtering Logic: Retain components more correlated with GT than NDVI, and positively correlated with GT
    urban_indices = [i for i, (c_gt, c_ndvi) in enumerate(comp_corrs) if c_gt > c_ndvi and c_gt > 0.01]
    pseudo_indices = [i for i in range(r) if i not in urban_indices]

    print(f"\nUrban Components: {urban_indices}")
    print(f"Pseudo-Change Components: {pseudo_indices}")

    if len(urban_indices) > 0:
        filtered_ds = np.sum(components[urban_indices], axis=0)
    else:
        print("Warning: No urban components found. Falling back to Full DS.")
        filtered_ds = full_ds.copy()

    # Compute other baselines
    pca_diff = pca_diff_score(x1, x2, valid_mask)
    raw_l2 = compute_raw_l2(x1, x2)

    # Normalize for display and metrics
    maps_to_eval = {
        "Full_DS": normalize_map(full_ds),
        "Filtered_DS": normalize_map(filtered_ds),
        "PCA-diff": normalize_map(pca_diff),
        "Raw_L2": normalize_map(raw_l2),
    }

    from sklearn.metrics import average_precision_score
    from phase1.eval.metrics import auroc_score

    print("\nEvaluating Methods...")
    metrics_list = []

    for name, score_map in maps_to_eval.items():
        s_flat = score_map.flatten()[v_mask]
        auroc = auroc_score(score_map, gt, valid_mask)
        if len(np.unique(gt_flat)) > 1:
            ap = average_precision_score(gt_flat, s_flat)
        else:
            ap = float("nan")

        m = {
            "method": name,
            "auroc": auroc,
            "average_precision": ap
        }
        metrics_list.append(m)
        print(f"  {name:12s}: AUROC = {auroc:.4f}, AP = {ap:.4f}")

    # Plotting
    print("Generating visualization...")
    fig, axes = plt.subplots(4, max(4, r), figsize=(4 * max(4, r), 16))
    for ax in axes.flat:
        ax.axis("off")

    # Row 0: Context
    axes[0, 0].imshow(make_rgb(sample.x_pre))
    axes[0, 0].set_title("Pre RGB")
    axes[0, 1].imshow(make_rgb(sample.x_post))
    axes[0, 1].set_title("Post RGB")
    axes[0, 2].imshow(gt[0], cmap="gray")
    axes[0, 2].set_title("OSCD GT Label")
    axes[0, 3].imshow(normalize_map(delta_ndvi), cmap="hot")
    axes[0, 3].set_title("|Delta NDVI|")

    # Row 1: Individual Components
    for i in range(r):
        c_gt, c_ndvi = comp_corrs[i]
        tag = "URBAN" if i in urban_indices else "PSEUDO"
        axes[1, i].imshow(normalize_map(components[i]), cmap="viridis")
        axes[1, i].set_title(f"Comp {i} ({tag})\nrGT:{c_gt:.2f} rNDVI:{c_ndvi:.2f}")

    # Row 2: Aggregated Maps
    axes[2, 0].imshow(maps_to_eval["Full_DS"], cmap="viridis")
    axes[2, 0].set_title("Full DS")
    axes[2, 1].imshow(maps_to_eval["Filtered_DS"], cmap="viridis")
    axes[2, 1].set_title("Filtered DS")
    axes[2, 2].imshow(maps_to_eval["PCA-diff"], cmap="viridis")
    axes[2, 2].set_title("PCA-diff")
    axes[2, 3].imshow(maps_to_eval["Raw_L2"], cmap="viridis")
    axes[2, 3].set_title("Raw L2")

    plt.tight_layout()
    fig_path = args.output_dir / f"comparison_components_grid_{args.city}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization: {fig_path}")

    # Save metrics
    metrics_path = args.output_dir / f"metrics_{args.city}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

if __name__ == "__main__":
    main()
