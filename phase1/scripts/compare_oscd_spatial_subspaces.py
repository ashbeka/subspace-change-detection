"""
Compare spatial Difference Subspace constructions on one OSCD city.

Source/provenance:
- Canonical linear DS comes from the Fukui/Maki first-order DS formulation
  implemented in `phase1.ds.pca_utils`.
- OSCD binary change detection follows the Sentinel-2 before/after benchmark
  introduced by Daudt et al. for fully convolutional/Siamese change detection.
- Raw L2/CVA and PCA-diff are included as classical baseline pressure; PCA-diff
  is related to PCA-based unsupervised change detection such as Celik 2009.
- AUROC and average precision use scikit-learn definitions; Otsu thresholding
  is used only as an unsupervised threshold diagnostic.

Project adaptation under test:
- global_pixel: one 13-band pixel is one sample, so spatial position is not used
  while fitting PCA.
- local_window: fit a pre/post DS pair inside sliding spatial windows.
- spatial_pyramid: fit canonical DS in fixed grids at multiple spatial scales.
- patch_vector: flatten a local 13-band patch into one sample vector so local
  neighborhood layout enters the subspace.
- band_image_ds: flatten each Sentinel-2 band image into one high-dimensional
  spatial sample vector, then score projected band-difference images.

Allowed claim:
- This script can test whether spatial support improves DS score maps on OSCD.
  It cannot by itself prove a general satellite-specific subspace method.
"""
from __future__ import annotations

import argparse
import csv
import gc
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

from phase1.baselines.celik_pca_kmeans import celik_score
from phase1.baselines.anomalous_change import chronochrome_score, covariance_equalization_score
from phase1.baselines.ir_mad import ir_mad_score
from phase1.baselines.pca_diff import pca_diff_score
from phase1.data.oscd_dataset import OSCDEvaluatorDataset
from phase1.data.preprocessing import apply_normalization, load_band_stats
from phase1.ds import pca_utils
from phase1.ds.ds_scores import DSConfig, compute_ds_scores
from phase1.eval.metrics import binary_metrics
from phase1.eval.thresholding import apply_threshold, otsu_threshold
from phase1.eval.utils import suppress_rasterio_warnings
from phase1.subspace.pif_nuisance import (
    PIFContext,
    build_pif_context,
    pif_delta_subspace_residual_score,
    pif_nuisance_ds_residual_score,
    pif_radiometric_match,
    multiscale_pif_delta_residual_score,
    multiscale_spatial_features,
)


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
    score_mode: str = "energy_sq"
    pyramid_levels: tuple[int, ...] | None = None
    spatial_sigmas: tuple[float, ...] | None = None


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
        description="Compare global, window-local, patch-vector, and band-image DS score maps on one OSCD city.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--oscd_root", "--oscd-root", type=Path, default=Path("data/OSCD"))
    ap.add_argument("--stats_path", "--stats-path", type=Path, default=Path("phase1/data/oscd_band_stats.json"))
    ap.add_argument("--city", default="beirut")
    ap.add_argument("--split", default="auto", choices=["auto", "train", "val", "test"])
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument(
        "--methods",
        default=DEFAULT_METHODS,
        help=(
            "Comma list: global_pixel, window128, window128s64, window128s64max, patch3, patch5, "
            "band_image_ds, band_image_norm, band_image_ratio, band_image_residual, "
            "band_image_spatial_gram, band_image_projector_distance, band_image_cross_reconstruction, "
            "celik_pca_kmeans, ir_mad, rank_fusion_pca_band, rank_fusion_band_irmad, "
            "chronochrome, covariance_equalization, pif_radiometric_l2, pif_radiometric_pca, "
            "pif_nuisance_ds_r1, pif_nuisance_ds_r2, pif_nuisance_ds_r3, "
            "pif_delta_residual_r1, pif_delta_residual_r2, pif_delta_residual_r3, "
            "smoothed_l2_sigma1, smoothed_pca_sigma1, smoothed_chronochrome_sigma1, "
            "smoothed_global_ds_sigma1, multiscale_global_ds_sigma0_1_2, "
            "smoothed_magnitude_weighted_ds_sigma1, smoothed_magnitude_weighted_ds_sigma2, "
            "local_autocorrelation_ds_magnitude_w32s16, local_autocorrelation_ds_magnitude_w64s32, "
            "local_centered_ds_magnitude_w32s16, "
            "post_smoothed_pca_sigma1, post_smoothed_raw_l2_sigma1, "
            "smoothed_pif_delta_residual_sigma1_r1, multiscale_l2_sigma0_1_2, "
            "multiscale_pca_sigma0_1_2, multiscale_pif_delta_residual_sigma0_1_2_r1, "
            "rank_fusion_pca_irmad, rank_fusion_pca_band_irmad, "
            "spatial_pyramid_1_2_4_energy, spatial_pyramid_1_2_4_norm, spatial_pyramid_1_2_4_8_norm. "
            "Legacy aliases such as flatbands are still accepted."
        ),
    )
    ap.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_fit_samples", "--max-fit-samples", type=int, default=20000, help="Maximum pixel/patch samples used to fit patch PCA subspaces.")
    ap.add_argument("--score_chunk_size", "--score-chunk-size", type=int, default=25000, help="Patch scoring chunk size.")
    ap.add_argument("--score_percentile", "--score-percentile", type=float, default=99.0, help="Percentile used for display normalization.")
    ap.add_argument("--save_band_attribution", "--save-band-attribution", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--celik_patch_size", "--celik-patch-size", type=int, default=9)
    ap.add_argument("--celik_pca_energy", "--celik-pca-energy", type=float, default=0.9)
    ap.add_argument("--celik_downsample_max_side", "--celik-downsample-max-side", type=int, default=384)
    ap.add_argument("--celik_feature_mode", "--celik-feature-mode", choices=("spectral_norm", "multiband_patch"), default="spectral_norm")
    ap.add_argument("--celik_max_fit_samples", "--celik-max-fit-samples", type=int, default=20000)
    ap.add_argument("--ir_mad_iters", "--ir-mad-iters", type=int, default=10)
    ap.add_argument("--ir_mad_downsample_max_pixels", "--ir-mad-downsample-max-pixels", type=int, default=200000)
    ap.add_argument("--pif_probability", "--pif-probability", type=float, default=0.9)
    ap.add_argument("--pif_fallback_fraction", "--pif-fallback-fraction", type=float, default=0.1)
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

    if key in {"celik", "celik_pca_kmeans", "pca_kmeans", "pcakmeans"}:
        return MethodSpec(name="celik_pca_kmeans", family="celik_pca_kmeans")
    if key in {"ir_mad", "irmad", "imad"}:
        return MethodSpec(name="ir_mad", family="ir_mad")
    if key in {"chronochrome", "chronochrome_symmetric", "cc"}:
        return MethodSpec(name="chronochrome", family="chronochrome")
    if key in {"covariance_equalization", "covariance_equalisation", "cov_equalization", "ce"}:
        return MethodSpec(name="covariance_equalization", family="covariance_equalization")
    if key in {"pif_radiometric_l2", "pif_normalized_l2", "pif_matched_l2"}:
        return MethodSpec(name="pif_radiometric_l2", family="pif_radiometric", score_mode="raw_l2")
    if key in {"pif_radiometric_pca", "pif_normalized_pca", "pif_matched_pca"}:
        return MethodSpec(name="pif_radiometric_pca", family="pif_radiometric", score_mode="pca_diff")
    if key in {"pif_nuisance_ds", "pif_nuisance_ds_residual"}:
        return MethodSpec(name="pif_nuisance_ds_r2", family="pif_nuisance_ds", window_size=2)
    pif_ds_match = re.fullmatch(r"pif_nuisance_ds_r(?P<rank>\d+)", key)
    if pif_ds_match:
        nuisance_rank = int(pif_ds_match.group("rank"))
        return MethodSpec(
            name=f"pif_nuisance_ds_r{nuisance_rank}",
            family="pif_nuisance_ds",
            window_size=nuisance_rank,
        )
    pif_delta_match = re.fullmatch(r"pif_delta_residual_r(?P<rank>\d+)", key)
    if pif_delta_match:
        nuisance_rank = int(pif_delta_match.group("rank"))
        return MethodSpec(
            name=f"pif_delta_residual_r{nuisance_rank}",
            family="pif_delta_residual",
            window_size=nuisance_rank,
        )
    spatial_specs = {
        "smoothed_l2_sigma1": ("spatial_feature_l2", (1.0,), 0),
        "smoothed_l2_sigma2": ("spatial_feature_l2", (2.0,), 0),
        "smoothed_pca_sigma1": ("spatial_feature_pca", (1.0,), 0),
        "smoothed_pca_sigma2": ("spatial_feature_pca", (2.0,), 0),
        "smoothed_chronochrome_sigma1": ("spatial_feature_chronochrome", (1.0,), 0),
        "smoothed_chronochrome_sigma2": ("spatial_feature_chronochrome", (2.0,), 0),
        "smoothed_global_ds_sigma1": ("spatial_feature_ds", (1.0,), 0),
        "smoothed_global_ds_sigma2": ("spatial_feature_ds", (2.0,), 0),
        "global_pixel_magnitude_weighted_ds": ("spatial_feature_weighted_ds", (0.0,), 0),
        "smoothed_magnitude_weighted_ds_sigma1": ("spatial_feature_weighted_ds", (1.0,), 0),
        "smoothed_magnitude_weighted_ds_sigma2": ("spatial_feature_weighted_ds", (2.0,), 0),
        "smoothed_pif_delta_residual_sigma1_r1": ("spatial_pif_delta", (1.0,), 1),
        "smoothed_pif_delta_residual_sigma2_r1": ("spatial_pif_delta", (2.0,), 1),
        "multiscale_l2_sigma0_1_2": ("spatial_feature_l2", (0.0, 1.0, 2.0), 0),
        "multiscale_pca_sigma0_1_2": ("spatial_feature_pca", (0.0, 1.0, 2.0), 0),
        "multiscale_global_ds_sigma0_1_2": ("spatial_feature_ds", (0.0, 1.0, 2.0), 0),
        "multiscale_magnitude_weighted_ds_sigma0_1_2": ("spatial_feature_weighted_ds", (0.0, 1.0, 2.0), 0),
        "multiscale_pif_delta_residual_sigma0_1_2_r1": ("spatial_pif_delta", (0.0, 1.0, 2.0), 1),
    }
    if key in spatial_specs:
        family, sigmas, nuisance_rank = spatial_specs[key]
        return MethodSpec(
            name=key,
            family=family,
            window_size=nuisance_rank or None,
            spatial_sigmas=sigmas,
        )
    post_smoothing_specs = {
        "post_smoothed_pca_sigma1": ("pca_diff", (1.0,)),
        "post_smoothed_pca_sigma2": ("pca_diff", (2.0,)),
        "post_smoothed_raw_l2_sigma1": ("raw_l2", (1.0,)),
        "post_smoothed_raw_l2_sigma2": ("raw_l2", (2.0,)),
    }
    if key in post_smoothing_specs:
        dependency, sigmas = post_smoothing_specs[key]
        return MethodSpec(
            name=key,
            family="score_post_smoothing",
            score_mode=dependency,
            spatial_sigmas=sigmas,
        )
    local_magnitude_match = re.fullmatch(
        r"local_(?P<fit>autocorrelation|centered)_ds_magnitude_w(?P<size>\d+)s(?P<stride>\d+)",
        key,
    )
    if local_magnitude_match:
        size = int(local_magnitude_match.group("size"))
        stride = int(local_magnitude_match.group("stride"))
        return MethodSpec(
            name=key,
            family="local_ds_magnitude",
            window_size=size,
            stride=stride,
            score_mode=local_magnitude_match.group("fit"),
        )
    fusion_dependencies = {
        "rank_fusion_pca_band": "pca_diff+band_image_norm",
        "rank_fusion_band_irmad": "band_image_norm+ir_mad",
        "rank_fusion_pca_irmad": "pca_diff+ir_mad",
        "rank_fusion_pca_band_irmad": "pca_diff+band_image_norm+ir_mad",
        "rank_fusion_smoothed_pca_band": "smoothed_pca_sigma1+band_image_norm",
        "rank_fusion_smoothed_pca_irmad": "smoothed_pca_sigma1+ir_mad",
        "rank_fusion_smoothed_pca_band_irmad": "smoothed_pca_sigma1+band_image_norm+ir_mad",
        "rank_fusion_smoothed_pca_cross_irmad": "smoothed_pca_sigma1+band_image_cross_reconstruction+ir_mad",
    }
    if key in fusion_dependencies:
        return MethodSpec(name=key, family="rank_fusion", score_mode=fusion_dependencies[key])
    pyramid_specs = {
        "spatial_pyramid_1_2_4_energy": ((1, 2, 4), "energy_sq"),
        "spatial_pyramid_1_2_4_norm": ((1, 2, 4), "energy_norm"),
        "spatial_pyramid_1_2_4_8_norm": ((1, 2, 4, 8), "energy_norm"),
    }
    if key in pyramid_specs:
        levels, score_mode = pyramid_specs[key]
        return MethodSpec(name=key, family="spatial_pyramid", score_mode=score_mode, pyramid_levels=levels)

    if key in {"band_image_ds", "band_image", "band_images", "band_image_energy", "flatbands", "flat_bands", "flattened_band", "flattened_bands"}:
        return MethodSpec(name="band_image_ds", family="band_image", score_mode="energy_sq")
    if key in {"band_image_norm", "band_image_magnitude"}:
        return MethodSpec(name="band_image_norm", family="band_image", score_mode="energy_norm")
    if key in {"band_image_ratio", "band_image_projection_ratio"}:
        return MethodSpec(name="band_image_ratio", family="band_image", score_mode="energy_ratio")
    if key in {"band_image_residual", "band_image_residual_energy"}:
        return MethodSpec(name="band_image_residual", family="band_image", score_mode="residual_energy")
    band_image_controls = {
        "band_image_spatial_gram": "spatial_gram",
        "band_image_gram": "spatial_gram",
        "band_image_projector_distance": "projector_distance",
        "band_image_projector": "projector_distance",
        "band_image_cross_reconstruction": "cross_reconstruction",
        "band_image_cross_residual": "cross_reconstruction",
    }
    if key in band_image_controls:
        return MethodSpec(
            name={
                "spatial_gram": "band_image_spatial_gram",
                "projector_distance": "band_image_projector_distance",
                "cross_reconstruction": "band_image_cross_reconstruction",
            }[band_image_controls[key]],
            family="band_image_control",
            score_mode=band_image_controls[key],
        )

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


def spatial_pyramid_ds_score(
    x1: np.ndarray,
    x2: np.ndarray,
    valid_mask: np.ndarray,
    args: argparse.Namespace,
    spec: MethodSpec,
) -> tuple[np.ndarray, ConstructionCard]:
    """Fit canonical pixel-spectral DS independently in fixed multiscale grids."""
    if not spec.pyramid_levels:
        raise ValueError(f"No pyramid levels configured for {spec.name}.")
    cfg = DSConfig(
        rank_r=args.rank,
        variance_threshold=None,
        random_state=args.seed,
        subspace_variant="canonical",
        score_normalization=None,
    )
    height, width = valid_mask.shape
    scale_maps: list[np.ndarray] = []
    cells_used = 0
    cells_skipped = 0
    min_pixels = max(args.rank + 1, 16)

    for level in spec.pyramid_levels:
        y_edges = np.linspace(0, height, level + 1, dtype=int)
        x_edges = np.linspace(0, width, level + 1, dtype=int)
        scale_score = np.zeros((height, width), dtype=np.float32)
        for iy in range(level):
            for ix in range(level):
                sl_y = slice(int(y_edges[iy]), int(y_edges[iy + 1]))
                sl_x = slice(int(x_edges[ix]), int(x_edges[ix + 1]))
                cell_valid = valid_mask[sl_y, sl_x]
                if int(np.sum(cell_valid)) < min_pixels:
                    cells_skipped += 1
                    continue
                result = compute_ds_scores(
                    x1[:, sl_y, sl_x],
                    x2[:, sl_y, sl_x],
                    valid_mask=cell_valid,
                    cfg=cfg,
                    normalize=False,
                )["projection"].astype(np.float32)
                if spec.score_mode == "energy_norm":
                    result = np.sqrt(np.maximum(result, 0.0)).astype(np.float32)
                elif spec.score_mode != "energy_sq":
                    raise ValueError(f"Unsupported pyramid score mode: {spec.score_mode}")
                scale_score[sl_y, sl_x] = result
                cells_used += 1
        scale_maps.append(scale_score)

    output = np.mean(np.stack(scale_maps, axis=0), axis=0, dtype=np.float32)
    output[~valid_mask] = 0.0
    level_text = ", ".join(f"{level}x{level}" for level in spec.pyramid_levels)
    card = ConstructionCard(
        method=spec.name,
        source_reference=(
            "Hierarchical spatial-support experiment motivated by the project Green Learning/wavelet note; "
            "uses canonical first-order DS at each grid cell but does not claim to implement PixelHop or a wavelet transform."
        ),
        sample_unit="one valid 13-band pixel inside one fixed pyramid grid cell",
        input_object=f"for each cell and date: X_cell in R^(13 x N_cell), at grid scales {level_text}",
        subspace_count=f"one pre/post PCA+DS pair per grid cell; cells used={cells_used}, skipped={cells_skipped}",
        basis_shape=f"per cell Phi, Psi in R^(13 x {args.rank}); canonical D in R^(13 x <= {args.rank})",
        fitting_method=f"rank-{args.rank} PCA and canonical DS independently at levels {level_text}; equal-weight mean across scale maps",
        score_definition=(
            "per cell pixel: ||D_cell^T delta||, then equal-weight mean across scales"
            if spec.score_mode == "energy_norm"
            else "per cell pixel: ||D_cell^T delta||^2, then equal-weight mean across scales"
        ),
        spatial_information="preserves coarse-to-fine regional support through fixed grid membership; exact coordinates are not PCA features and grid boundaries remain a risk",
        code_path="phase1.scripts.compare_oscd_spatial_subspaces.spatial_pyramid_ds_score",
        verification="must beat or explain global/window/patch DS and be pressure-tested against PCA-diff, IR-MAD, labels, maps, and grid-boundary artifacts",
    )
    return output, card


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


def band_image_ds_score(x1: np.ndarray, x2: np.ndarray, valid_mask: np.ndarray, args: argparse.Namespace, spec: MethodSpec) -> tuple[np.ndarray, ConstructionCard, dict[str, np.ndarray]]:
    """Score change by treating each band image as one spatial vector.

    Matrix convention:
      X_pre_flat, X_post_flat in R^(N_valid_pixels x 13)

    PCA therefore learns a basis in pixel-position space from 13 band-image
    samples. To turn the result back into a change map, each band-difference
    image is projected into the spatial DS and pixel-wise projected evidence is
    reduced across bands.
    """
    c, h, w = x1.shape
    if c != len(BAND_ORDER):
        raise ValueError(f"Expected {len(BAND_ORDER)} bands, got {c}.")

    rows, cols = np.where(valid_mask)
    if rows.size == 0:
        raise RuntimeError("No valid pixels available for band-image DS.")

    # d = valid spatial positions, n = band-image samples.
    x1_mat = x1[:, rows, cols].T.astype(np.float32, copy=False)
    x2_mat = x2[:, rows, cols].T.astype(np.float32, copy=False)
    max_centered_rank = max(1, min(int(args.rank), x1_mat.shape[1] - 1))

    pre = pca_utils.fit_pca_basis(
        x1_mat,
        rank=max_centered_rank,
        variance_threshold=None,
        random_state=args.seed,
        use_randomized=True,
    )
    post = pca_utils.fit_pca_basis(
        x2_mat,
        rank=max_centered_rank,
        variance_threshold=None,
        random_state=args.seed,
        use_randomized=True,
    )
    d_basis = pca_utils.build_difference_subspace(pre.basis, post.basis, variant="canonical")

    diff = x2_mat - x1_mat
    if d_basis.shape[1] == 0:
        projected = np.zeros_like(diff, dtype=np.float32)
    else:
        coeff = d_basis.T @ diff
        projected = d_basis @ coeff

    projected_energy = np.sum(projected * projected, axis=1).astype(np.float32)
    raw_energy = np.sum(diff * diff, axis=1).astype(np.float32)
    residual = diff - projected
    residual_energy = np.sum(residual * residual, axis=1).astype(np.float32)

    if spec.score_mode == "energy_sq":
        score_values = projected_energy
        score_definition = "per pixel: sum_bands (P_D delta_band)^2"
    elif spec.score_mode == "energy_norm":
        score_values = np.sqrt(np.maximum(projected_energy, 0.0)).astype(np.float32)
        score_definition = "per pixel: sqrt(sum_bands (P_D delta_band)^2)"
    elif spec.score_mode == "energy_ratio":
        eps = np.float32(1e-8)
        score_values = np.divide(projected_energy, raw_energy + eps, out=np.zeros_like(projected_energy), where=raw_energy > eps).astype(np.float32)
        score_definition = "per pixel: sum_bands (P_D delta_band)^2 / (sum_bands delta_band^2 + 1e-8)"
    elif spec.score_mode == "residual_energy":
        score_values = residual_energy
        score_definition = "diagnostic per pixel: sum_bands (delta_band - P_D delta_band)^2"
    else:
        raise ValueError(f"Unknown band-image score mode: {spec.score_mode!r}")

    out = np.zeros((h, w), dtype=np.float32)
    out[rows, cols] = score_values

    band_maps: dict[str, np.ndarray] = {}
    for idx, band in enumerate(BAND_ORDER):
        one = np.zeros((h, w), dtype=np.float32)
        one[rows, cols] = (projected[:, idx] * projected[:, idx]).astype(np.float32)
        band_maps[band] = one

    card = ConstructionCard(
        method=spec.name,
        source_reference=(
            "Senpai/Jang-style spatial-band sample-definition test; uses the same canonical first-order DS formula "
            "but changes the sample from one pixel vector to one full flattened band image."
        ),
        sample_unit="one full Sentinel-2 band image flattened over valid pixel positions",
        input_object=f"X_pre_flat, X_post_flat in R^({rows.size} x {c}); columns are flattened band images",
        subspace_count="one PCA subspace from the 13 pre-date band images and one PCA subspace from the 13 post-date band images",
        basis_shape=f"Phi, Psi in R^({rows.size} x {pre.rank}); D in R^({rows.size} x {d_basis.shape[1]})",
        fitting_method=(
            f"rank-r PCA in valid-pixel spatial ambient space; requested rank={args.rank}, "
            f"actual rank={pre.rank}, centered-sample cap={max_centered_rank}"
        ),
        score_definition=score_definition,
        spatial_information="preserves spatial layout inside each band-image vector, but has only 13 band samples on Sentinel-2",
        code_path="phase1/scripts/compare_oscd_spatial_subspaces.py::band_image_ds_score",
        verification="compared against global/patch/window DS, raw L2, PCA-diff, OSCD labels, raw-L2 correlation, runtime, and visual maps",
    )
    return out, card, band_maps


def band_image_spatial_control_values(
    first: np.ndarray,
    second: np.ndarray,
    *,
    rank: int,
    seed: int,
    mode: str,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return a matched per-position null for a band-image DS construction.

    ``first`` and ``second`` are shaped ``N_spatial x B_bands``. All controls
    use the same centered matrices and rank as Band-Image DS:

    - ``spatial_gram`` is the row norm of the difference between trace-
      normalized full spatial Gram operators. It retains every singular mode.
    - ``projector_distance`` is the row norm of ``P_pre-P_post`` for the
      rank-matched projectors. It isolates subspace orientation without using
      the observed spectral-difference vector.
    - ``cross_reconstruction`` is the symmetric residual from reconstructing
      each date's centered band images with the other date's subspace.

    The formulas avoid materializing an ``N x N`` spatial operator.
    """
    if first.ndim != 2 or second.shape != first.shape:
        raise ValueError(
            f"Expected paired matrices shaped N x B, got {first.shape} and {second.shape}."
        )
    if first.shape[0] < 2 or first.shape[1] < 2:
        raise ValueError(f"Band-image controls require at least 2x2 data, got {first.shape}.")

    first64 = first.astype(np.float64, copy=False)
    second64 = second.astype(np.float64, copy=False)
    # ``fit_pca_basis`` receives (d, n) = (spatial positions, band images)
    # and transposes it to 13 samples x N spatial features for sklearn.  PCA
    # therefore subtracts the across-band mean at every spatial position.  The
    # matched controls must use that identical centering convention; centering
    # each band over space would define a different statistical object.
    first_centered = first64 - np.mean(first64, axis=1, keepdims=True)
    second_centered = second64 - np.mean(second64, axis=1, keepdims=True)

    if mode == "spatial_gram":
        first_scaled = first_centered / max(float(np.linalg.norm(first_centered)), eps)
        second_scaled = second_centered / max(float(np.linalg.norm(second_centered)), eps)
        first_gram_small = first_scaled.T @ first_scaled
        second_gram_small = second_scaled.T @ second_scaled
        cross_gram_small = first_scaled.T @ second_scaled
        first_norm_sq = np.einsum(
            "ni,ij,nj->n", first_scaled, first_gram_small, first_scaled, optimize=True
        )
        second_norm_sq = np.einsum(
            "ni,ij,nj->n", second_scaled, second_gram_small, second_scaled, optimize=True
        )
        cross = np.einsum(
            "ni,ij,nj->n", first_scaled, cross_gram_small, second_scaled, optimize=True
        )
        return np.sqrt(np.maximum(first_norm_sq + second_norm_sq - 2.0 * cross, 0.0)).astype(
            np.float32
        )

    effective_rank = max(1, min(int(rank), first.shape[1] - 1))
    first_basis = pca_utils.fit_pca_basis(
        first,
        rank=effective_rank,
        variance_threshold=None,
        random_state=seed,
        use_randomized=True,
    ).basis.astype(np.float64, copy=False)
    second_basis = pca_utils.fit_pca_basis(
        second,
        rank=effective_rank,
        variance_threshold=None,
        random_state=seed,
        use_randomized=True,
    ).basis.astype(np.float64, copy=False)
    # The PCA helper stores float32 bases. Re-orthonormalize in float64 before
    # subtracting nearly equal projectors to prevent cancellation from
    # appearing as a small false change on identical inputs.
    first_basis = np.linalg.qr(first_basis, mode="reduced")[0]
    second_basis = np.linalg.qr(second_basis, mode="reduced")[0]

    if mode == "projector_distance":
        cross_basis = first_basis.T @ second_basis
        first_leverage = np.sum(first_basis * first_basis, axis=1)
        second_leverage = np.sum(second_basis * second_basis, axis=1)
        cross = np.einsum(
            "ni,ij,nj->n", first_basis, cross_basis, second_basis, optimize=True
        )
        return np.sqrt(
            np.maximum(first_leverage + second_leverage - 2.0 * cross, 0.0)
        ).astype(np.float32)

    if mode == "cross_reconstruction":
        second_residual = second_centered - first_basis @ (first_basis.T @ second_centered)
        first_residual = first_centered - second_basis @ (second_basis.T @ first_centered)
        second_self_residual = second_centered - second_basis @ (second_basis.T @ second_centered)
        first_self_residual = first_centered - first_basis @ (first_basis.T @ first_centered)
        excess = (
            np.sum(second_residual * second_residual, axis=1)
            - np.sum(second_self_residual * second_self_residual, axis=1)
            + np.sum(first_residual * first_residual, axis=1)
            - np.sum(first_self_residual * first_self_residual, axis=1)
        )
        return np.sqrt(
            np.maximum(excess, 0.0)
        ).astype(np.float32)

    raise ValueError(f"Unknown band-image spatial control mode: {mode!r}")


def band_image_spatial_control_score(
    x1: np.ndarray,
    x2: np.ndarray,
    valid_mask: np.ndarray,
    args: argparse.Namespace,
    spec: MethodSpec,
) -> tuple[np.ndarray, ConstructionCard]:
    """Map a matched band-image null back to the original spatial grid."""
    rows, cols = np.where(valid_mask)
    if rows.size == 0:
        raise RuntimeError("No valid pixels available for band-image spatial controls.")
    first = x1[:, rows, cols].T.astype(np.float32, copy=False)
    second = x2[:, rows, cols].T.astype(np.float32, copy=False)
    values = band_image_spatial_control_values(
        first,
        second,
        rank=args.rank,
        seed=args.seed,
        mode=str(spec.score_mode),
    )
    output = np.zeros(valid_mask.shape, dtype=np.float32)
    output[rows, cols] = values

    definitions = {
        "spatial_gram": (
            "row norm of trace-normalized full spatial Gram difference",
            "full centered band-image matrices; no rank truncation",
            "retains all spatial second-moment modes while removing total matrix scale",
        ),
        "projector_distance": (
            "row norm of P_pre-P_post",
            f"rank-{min(args.rank, first.shape[1] - 1)} pre/post spatial projectors",
            "isolates spatial eigenspace orientation without projecting the spectral-difference vector",
        ),
        "cross_reconstruction": (
            "symmetric excess cross-subspace reconstruction residual magnitude over self-reconstruction",
            f"rank-{min(args.rank, first.shape[1] - 1)} pre/post spatial bases",
            "tests cross-prediction from the same spatial subspaces without constructing DS directions",
        ),
    }
    definition, basis, purpose = definitions[str(spec.score_mode)]
    return output, ConstructionCard(
        method=spec.name,
        source_reference=(
            "Matched linear-algebra null for the project Band-Image DS adaptation; "
            "projector and Gram identities are standard subspace/covariance geometry."
        ),
        sample_unit="one full Sentinel-2 band image flattened over the common valid positions",
        input_object=f"X_pre, X_post in R^({rows.size} x {first.shape[1]})",
        subspace_count="zero for full Gram; one rank-matched pre/post pair for projector or cross-reconstruction",
        basis_shape=basis,
        fitting_method="center the 13 band-image samples at each spatial position, matching sklearn PCA on X.T; use the same requested rank/seed as Band-Image DS",
        score_definition=definition,
        spatial_information=f"preserves the same valid spatial-coordinate axis as Band-Image DS; {purpose}",
        code_path="phase1.scripts.compare_oscd_spatial_subspaces.band_image_spatial_control_values",
        verification=(
            "equal-matrix, scale-invariance, explicit NxN formula-equivalence, and changed-support tests; "
            "then paired OSCD city-wise pressure against Band-Image DS"
        ),
    )


def celik_pca_kmeans_score(x1: np.ndarray, x2: np.ndarray, valid_mask: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, ConstructionCard]:
    score = celik_score(
        x1,
        x2,
        patch_size=int(args.celik_patch_size),
        pca_energy=float(args.celik_pca_energy),
        valid_mask=valid_mask,
        random_state=int(args.seed),
        downsample_max_side=int(args.celik_downsample_max_side) if args.celik_downsample_max_side else None,
        feature_mode=str(args.celik_feature_mode),
        max_fit_samples=int(args.celik_max_fit_samples),
        score_chunk_size=int(args.score_chunk_size),
    )
    card = ConstructionCard(
        method="celik_pca_kmeans",
        source_reference=(
            "Celik 2009 PCA+k-means unsupervised change detection: build a difference image, "
            "represent local neighborhoods in a PCA feature space, then cluster features into changed/unchanged groups."
        ),
        sample_unit=f"one local scalar difference-magnitude patch, patch_size={args.celik_patch_size}",
        input_object=(
            "CVA/L2 magnitude image ||X_post-X_pre||_2; local scalar patches are flattened for PCA/k-means"
            if args.celik_feature_mode == "spectral_norm"
            else "Difference tensor Delta=X_post-X_pre; local 13-band patches are flattened (explicit project adaptation)"
        ),
        subspace_count="one PCA feature space for local difference patches, followed by k=2 clustering",
        basis_shape=f"PCA keeps components explaining {args.celik_pca_energy:.2f} cumulative variance",
        fitting_method=(
            f"PCA and k-means on a seeded subset of at most {args.celik_max_fit_samples} valid local patches; "
            "all valid patches are predicted in chunks; larger projected-magnitude cluster is treated as changed"
        ),
        score_definition="binary changed-cluster assignment multiplied by normalized projected-feature magnitude",
        spatial_information="uses local patch neighborhoods, so it is the closest traditional spatial baseline for patch-vector DS",
        code_path="phase1.baselines.celik_pca_kmeans.celik_score",
        verification="paper-family implementation; compared in this script against OSCD labels and DS/PCA-diff/raw L2",
    )
    return score.astype(np.float32), card


def ir_mad_baseline_score(x1: np.ndarray, x2: np.ndarray, valid_mask: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, ConstructionCard]:
    score = ir_mad_score(
        x1,
        x2,
        valid_mask,
        iters=int(args.ir_mad_iters),
        downsample_max_pixels=int(args.ir_mad_downsample_max_pixels),
        random_state=int(args.seed),
    )
    card = ConstructionCard(
        method="ir_mad",
        source_reference=(
            "Nielsen 2007 IR-MAD / GEE iMAD: canonical variates from paired multiband dates, "
            "MAD variate differences, and iterative chi-square survival weighting of likely unchanged pixels."
        ),
        sample_unit="one paired valid pixel across pre/post Sentinel-2 dates",
        input_object="X_pre, X_post in R^(13 x N), paired by pixel location",
        subspace_count="CCA transform pair, not a DS subspace; included as established multivariate classical baseline",
        basis_shape="A, B in R^(13 x 13), canonical correlations rho in R^13",
        fitting_method="iteratively reweighted CCA on a seeded valid-pixel subsample, then final transform applied to all valid pixels",
        score_definition="per pixel: chi-square statistic sum_i MAD_i^2 / (2 * (1 - rho_i))",
        spatial_information="standard/global IR-MAD is spectral-paired, not spatial-local; pixel coordinates are used only for pairing and map reshaping",
        code_path="phase1.baselines.ir_mad.ir_mad_score",
        verification="formula repaired against Nielsen/GEE iMAD weighting; still treated as compact baseline, not a full geospatial package",
    )
    return score.astype(np.float32), card


def local_ds_magnitude_score(
    x1: np.ndarray,
    x2: np.ndarray,
    valid_mask: np.ndarray,
    args: argparse.Namespace,
    spec: MethodSpec,
) -> tuple[np.ndarray, ConstructionCard]:
    """Map intrinsic DS magnitude from overlapping local spectral subspaces."""
    if spec.window_size is None or spec.stride is None:
        raise ValueError(f"{spec.name} requires window and stride.")
    height, width = valid_mask.shape
    accumulator = np.zeros((height, width), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.float64)
    windows_used = 0
    windows_skipped = 0
    fit_mode = str(spec.score_mode)
    for row in _window_positions(height, spec.window_size, spec.stride):
        row_slice = slice(row, row + spec.window_size)
        for column in _window_positions(width, spec.window_size, spec.stride):
            column_slice = slice(column, column + spec.window_size)
            local_mask = valid_mask[row_slice, column_slice]
            minimum = max(int(args.rank) + 1, 2 * int(args.rank))
            if int(local_mask.sum()) < minimum:
                windows_skipped += 1
                continue
            first = x1[:, row_slice, column_slice][:, local_mask]
            second = x2[:, row_slice, column_slice][:, local_mask]
            effective_rank = min(int(args.rank), first.shape[0] - 1, first.shape[1] - 1)
            if fit_mode == "autocorrelation":
                pre = pca_utils.fit_autocorrelation_basis(first, rank=effective_rank)
                post = pca_utils.fit_autocorrelation_basis(second, rank=effective_rank)
            elif fit_mode == "centered":
                pre = pca_utils.fit_covariance_basis(first, rank=effective_rank)
                post = pca_utils.fit_covariance_basis(second, rank=effective_rank)
            else:
                raise ValueError(f"Unknown local subspace fit mode: {fit_mode!r}")
            components = pca_utils.difference_subspace_canonical_components(pre.basis, post.basis)
            magnitude = float(np.sum(components.squared_pair_magnitudes))
            local_accumulator = accumulator[row_slice, column_slice]
            local_counts = counts[row_slice, column_slice]
            local_accumulator[local_mask] += magnitude
            local_counts[local_mask] += 1.0
            windows_used += 1
    score = np.divide(accumulator, counts, out=np.zeros_like(accumulator), where=counts > 0)
    score[~valid_mask] = 0.0
    fit_description = (
        "uncentered R=XX^T/N eigenspaces stated in Fukui--Maki Section 3.2"
        if fit_mode == "autocorrelation"
        else "centered local covariance PCA ablation"
    )
    return score.astype(np.float32), ConstructionCard(
        method=spec.name,
        source_reference=(
            "Fukui--Maki TPAMI 2015 canonical subspaces and MagTool calcMagnitude: "
            "m(P,Q)=2*sum_i(1-cos(theta_i)); local-window mapping is the project adaptation."
        ),
        sample_unit=f"one valid 13-band pixel inside each {spec.window_size}x{spec.window_size} neighborhood",
        input_object="For each neighborhood: X_pre_w, X_post_w in R^(13 x N_w)",
        subspace_count=f"one pre/post subspace pair per neighborhood; used={windows_used}, skipped={windows_skipped}",
        basis_shape=f"Phi_w, Psi_w in R^(13 x {args.rank})",
        fitting_method=f"{fit_description}; stride={spec.stride}",
        score_definition="window evidence is intrinsic first-order DS magnitude 2*sum(1-cos(theta_i)); overlapping evidence is averaged per pixel",
        spatial_information="the image coordinates determine local sample membership and each output pixel aggregates only covering neighborhoods",
        code_path="phase1.scripts.compare_oscd_spatial_subspaces.local_ds_magnitude_score",
        verification="centered-vs-autocorrelation construction ablation, rank/window sensitivity, OSCD labels, PCA-diff and spatial-PCA pressure",
    )


def anomalous_change_baseline_score(
    method: str,
    x1: np.ndarray,
    x2: np.ndarray,
    valid_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, ConstructionCard]:
    if method == "chronochrome":
        score = chronochrome_score(
            x1,
            x2,
            valid_mask,
            direction="symmetric",
            max_fit_pixels=int(args.ir_mad_downsample_max_pixels),
            random_state=int(args.seed),
        )
        reference = "Theiler and Perkins 2006 equations 14-18; Schaum/Stocker chronochrome family."
        fitting = "fit forward and reverse cross-date linear predictors on a seeded paired-pixel subset"
        definition = "mean percentile rank of forward and reverse prediction-residual Mahalanobis scores"
        basis = "L_forward and L_reverse in R^(13 x 13), with residual covariance matrices"
    elif method == "covariance_equalization":
        score = covariance_equalization_score(
            x1,
            x2,
            valid_mask,
            max_fit_pixels=int(args.ir_mad_downsample_max_pixels),
            random_state=int(args.seed),
        )
        reference = "Theiler and Perkins 2006 equation 19; Schaum/Stocker covariance equalization."
        fitting = "fit separate symmetric whitening transforms and a covariance for their paired residual"
        definition = "Mahalanobis magnitude of X_pre^(-1/2)x_pre - X_post^(-1/2)x_post"
        basis = "two 13 x 13 whitening transforms plus a 13 x 13 residual covariance"
    else:
        raise ValueError(f"Unknown anomalous-change method: {method}")
    return score.astype(np.float32), ConstructionCard(
        method=method,
        source_reference=reference,
        sample_unit="one co-registered valid pre/post Sentinel-2 pixel pair",
        input_object="X_pre, X_post in R^(13 x N), paired by pixel location",
        subspace_count="no DS basis; the normal cross-date relationship is modeled directly",
        basis_shape=basis,
        fitting_method=fitting,
        score_definition=definition,
        spatial_information="pixel location supplies pairing and map reconstruction; neighborhood context is not modeled",
        code_path=f"phase1.baselines.anomalous_change.{method}_score",
        verification="equation-level synthetic tests plus OSCD city-wise pressure comparison",
    )


def pif_method_score(
    spec: MethodSpec,
    x1: np.ndarray,
    x2: np.ndarray,
    valid_mask: np.ndarray,
    args: argparse.Namespace,
    context: PIFContext,
) -> tuple[np.ndarray, ConstructionCard]:
    common = {
        "pmin": float(args.pif_probability),
        "fallback_fraction": float(args.pif_fallback_fraction),
        "ir_mad_iters": int(args.ir_mad_iters),
        "max_fit_pixels": int(args.ir_mad_downsample_max_pixels),
    }
    if spec.family == "pif_radiometric":
        matched, context, coefficients = pif_radiometric_match(
            x1, x2, valid_mask, context=context, **common
        )
        if spec.score_mode == "raw_l2":
            score = raw_l2_score(x1, matched, valid_mask)
            definition = "raw spectral L2 after PIF orthogonal per-band radiometric matching"
        else:
            score = pca_diff_score(
                x1, matched, valid_mask, rank_S=args.rank, variance_threshold=None, random_state=args.seed
            )
            definition = "PCA-diff after PIF orthogonal per-band radiometric matching"
        card = ConstructionCard(
            method=spec.name,
            source_reference="Canty/Nielsen 2008 and GEE iMAD Part 3 relative radiometric normalization.",
            sample_unit="one iMAD-selected pseudo-invariant paired pixel per spectral band",
            input_object=f"{context.selected_pixels} PIF pixel pairs selected at effective p >= {context.threshold_used:.6g}",
            subspace_count="no new DS; iMAD identifies invariant pixels and 13 symmetric regression lines align the dates",
            basis_shape="13 slope/intercept pairs; " + str(coefficients.shape),
            fitting_method="orthogonal regression target ~= slope * reference + intercept, then invert target onto reference scale",
            score_definition=definition,
            spatial_information="global radiometric alignment; coordinates are retained for paired fitting and map reconstruction",
            code_path="phase1.subspace.pif_nuisance.pif_radiometric_match",
            verification="synthetic affine-shift test and OSCD pressure against unnormalized L2/PCA-diff and anomalous-change baselines",
        )
        return score.astype(np.float32), card

    if spec.family == "pif_nuisance_ds":
        pif_subspace_rank = int(spec.window_size or 2)
        result = pif_nuisance_ds_residual_score(
            x1,
            x2,
            valid_mask,
            rank=pif_subspace_rank,
            random_state=int(args.seed),
            context=context,
            **common,
        )
        return result.score.astype(np.float32), ConstructionCard(
            method=spec.name,
            source_reference=(
                "Project hypothesis combining iMAD PIF selection with Fukui/Maki canonical DS; "
                "not an established paper method or a novelty claim."
            ),
            sample_unit="one 13-band PIF pixel vector; only likely unchanged pixels fit the nuisance geometry",
            input_object=f"X_pre_PIF, X_post_PIF in R^(13 x {result.pif_pixels})",
            subspace_count="two rank-r PIF PCA subspaces; their canonical DS plus PIF mean shift forms one nuisance basis",
            basis_shape=f"nuisance basis in R^(13 x {result.nuisance_basis.shape[1]}), canonical DS rank={result.ds_rank}",
            fitting_method="iMAD PIF selection -> pre/post PIF PCA -> canonical DS -> append mean-shift direction -> QR",
            score_definition="per pixel: norm of centered spectral difference after projection outside the PIF nuisance basis",
            spatial_information="global spectral nuisance model; pixel pairing and output coordinates retained, neighborhoods not modeled",
            code_path="phase1.subspace.pif_nuisance.pif_nuisance_ds_residual_score",
            verification="synthetic nuisance/change separation and comparison with chronochrome, covariance equalization, iMAD, PCA-diff",
        )

    if spec.family == "pif_delta_residual":
        nuisance_rank = int(spec.window_size or 1)
        score, context, nuisance = pif_delta_subspace_residual_score(
            x1,
            x2,
            valid_mask,
            nuisance_rank=nuisance_rank,
            random_state=int(args.seed),
            context=context,
            **common,
        )
        return score.astype(np.float32), ConstructionCard(
            method=spec.name,
            source_reference="PIF-conditioned nuisance-PCA ablation; related to subspace-projection anomaly detection.",
            sample_unit="one 13-band difference vector from an iMAD-selected PIF pixel",
            input_object=f"Delta_PIF in R^(13 x {context.selected_pixels})",
            subspace_count="one PCA nuisance subspace fitted to stable-pixel difference vectors",
            basis_shape=f"U_nuisance in R^(13 x {nuisance.shape[1]})",
            fitting_method="subtract PIF mean shift, fit PCA to PIF differences, reject leading nuisance directions",
            score_definition="per pixel: norm of spectral difference outside the PIF-difference nuisance subspace",
            spatial_information="global spectral nuisance model; pixel coordinates are preserved but neighborhoods are not modeled",
            code_path="phase1.subspace.pif_nuisance.pif_delta_subspace_residual_score",
            verification="ablation against the canonical nuisance-DS hypothesis and established ACD baselines",
        )
    raise ValueError(f"Unhandled PIF method family: {spec.family}")


def spatial_feature_score(
    spec: MethodSpec,
    x1: np.ndarray,
    x2: np.ndarray,
    valid_mask: np.ndarray,
    args: argparse.Namespace,
    context: PIFContext | None,
) -> tuple[np.ndarray, ConstructionCard]:
    """Score Gaussian-neighborhood features with matched non-subspace controls."""
    if spec.spatial_sigmas is None:
        raise ValueError(f"{spec.name} does not define spatial scales.")
    sigmas = tuple(float(value) for value in spec.spatial_sigmas)
    scale_text = ", ".join(f"sigma={value:g}" for value in sigmas)
    source = (
        "Theiler and Perkins 2006 Section 5 motivates smoothing/spatio-spectral operators for "
        "misregistration and characteristic anomaly size; the score variants are controlled project ablations."
    )

    if spec.family == "spatial_pif_delta":
        if context is None:
            raise ValueError("PIF context is required for spatial PIF nuisance scoring.")
        nuisance_rank = int(spec.window_size or 1)
        score, _, nuisance = multiscale_pif_delta_residual_score(
            x1,
            x2,
            valid_mask,
            sigmas=sigmas,
            nuisance_rank=nuisance_rank,
            random_state=int(args.seed),
            context=context,
            pmin=float(args.pif_probability),
            fallback_fraction=float(args.pif_fallback_fraction),
            ir_mad_iters=int(args.ir_mad_iters),
            max_fit_pixels=int(args.ir_mad_downsample_max_pixels),
        )
        definition = "norm of multiscale spatial-spectral difference after rejecting the leading PIF nuisance direction"
        fitting = "iMAD PIF selection on raw bands -> masked Gaussian features -> PIF delta PCA -> residual norm"
        subspaces = "one rank-1 nuisance PCA subspace fitted to multiscale PIF difference vectors"
        basis = f"U_nuisance in R^({13 * len(sigmas)} x {nuisance.shape[1]})"
        code_path = "phase1.subspace.pif_nuisance.multiscale_pif_delta_residual_score"
    else:
        features1 = multiscale_spatial_features(x1, valid_mask, sigmas=sigmas)
        features2 = multiscale_spatial_features(x2, valid_mask, sigmas=sigmas)
        if spec.family == "spatial_feature_l2":
            score = raw_l2_score(features1, features2, valid_mask)
            definition = "Euclidean magnitude of the matched multiscale spatial-spectral difference vector"
            fitting = "no fitted subspace; masked Gaussian features followed by L2"
            subspaces = "none; this isolates the contribution of spatial filtering"
            basis = "not applicable"
            code_path = "phase1.subspace.pif_nuisance.multiscale_spatial_features + raw_l2_score"
        elif spec.family == "spatial_feature_pca":
            score = pca_diff_score(
                features1,
                features2,
                valid_mask,
                rank_S=min(int(args.rank), features1.shape[0] - 1),
                variance_threshold=None,
                random_state=int(args.seed),
            )
            definition = "PCA-diff magnitude in the multiscale spatial-spectral feature space"
            fitting = "masked Gaussian features -> PCA-diff using the experiment rank"
            subspaces = "one PCA difference representation; non-DS spatial control"
            basis = f"PCA feature space from R^{features1.shape[0]} with rank <= {args.rank}"
            code_path = "phase1.subspace.pif_nuisance.multiscale_spatial_features + phase1.baselines.pca_diff"
        elif spec.family == "spatial_feature_chronochrome":
            score = chronochrome_score(
                features1,
                features2,
                valid_mask,
                direction="symmetric",
                max_fit_pixels=int(args.ir_mad_downsample_max_pixels),
                random_state=int(args.seed),
            )
            definition = "symmetric chronochrome prediction-residual anomaly on spatially filtered bands"
            fitting = "masked Gaussian features -> forward/reverse linear cross-date prediction"
            subspaces = "none; established relation-modeling pressure after the same spatial operator"
            basis = "cross-date linear predictor and residual covariance"
            code_path = "phase1.subspace.pif_nuisance.multiscale_spatial_features + phase1.baselines.anomalous_change"
        elif spec.family == "spatial_feature_ds":
            cfg = DSConfig(
                rank_r=min(int(args.rank), features1.shape[0] - 1),
                variance_threshold=None,
                random_state=int(args.seed),
                subspace_variant="canonical",
                score_normalization=None,
            )
            output = compute_ds_scores(
                features1,
                features2,
                valid_mask=valid_mask,
                cfg=cfg,
                normalize=False,
            )
            score = output["projection"]
            definition = "canonical DS projected energy of the paired multiscale spatial-spectral difference"
            fitting = "masked Gaussian features -> separate rank-r pre/post PCA -> canonical Difference Subspace"
            subspaces = "one pre-date and one post-date PCA subspace, followed by one canonical DS"
            basis = f"Phi/Psi in R^({features1.shape[0]} x {cfg.rank_r}); D in R^({features1.shape[0]} x <= {cfg.rank_r})"
            code_path = "phase1.subspace.pif_nuisance.multiscale_spatial_features + phase1.ds.ds_scores.compute_ds_scores"
        elif spec.family == "spatial_feature_weighted_ds":
            matrix1 = features1[:, valid_mask]
            matrix2 = features2[:, valid_mask]
            effective_rank = min(int(args.rank), features1.shape[0] - 1, matrix1.shape[1] - 1)
            pre = pca_utils.fit_pca_basis(
                matrix1,
                rank=effective_rank,
                variance_threshold=None,
                random_state=int(args.seed),
                use_randomized=True,
            )
            post = pca_utils.fit_pca_basis(
                matrix2,
                rank=effective_rank,
                variance_threshold=None,
                random_state=int(args.seed),
                use_randomized=True,
            )
            components = pca_utils.difference_subspace_canonical_components(pre.basis, post.basis)
            delta = matrix2 - matrix1
            coefficients = components.basis.T @ delta
            weighted_energy = np.sum(
                components.squared_pair_magnitudes[:, None] * coefficients * coefficients,
                axis=0,
            )
            score = np.zeros(valid_mask.shape, dtype=np.float32)
            score[valid_mask] = weighted_energy.astype(np.float32, copy=False)
            definition = (
                "canonical DS projection energy weighted by each principal-pair magnitude "
                "2(1-cos(theta_i))"
            )
            fitting = (
                "masked Gaussian features -> pre/post rank-r PCA -> canonical principal pairs -> "
                "magnitude-weighted projection"
            )
            subspaces = "one pre/post PCA pair and one canonical DS with retained principal-pair magnitudes"
            basis = (
                f"D in R^({features1.shape[0]} x {components.basis.shape[1]}); "
                f"{components.squared_pair_magnitudes.size} geometric weights"
            )
            code_path = (
                "phase1.ds.pca_utils.difference_subspace_canonical_components + "
                "phase1.scripts.compare_oscd_spatial_subspaces.spatial_feature_score"
            )
        else:
            raise ValueError(f"Unhandled spatial feature family: {spec.family}")

    return score.astype(np.float32), ConstructionCard(
        method=spec.name,
        source_reference=source,
        sample_unit=f"one co-registered pixel represented at {scale_text}",
        input_object=f"Z_pre, Z_post in R^({13 * len(sigmas)} x N); feature blocks retain band identity by scale",
        subspace_count=subspaces,
        basis_shape=basis,
        fitting_method=fitting,
        score_definition=definition,
        spatial_information=(
            "Gaussian neighborhoods encode local spatial support while masked normalization prevents invalid pixels "
            "from bleeding into valid neighborhoods; exact coordinates remain unchanged."
        ),
        code_path=code_path,
        verification=(
            "same-city comparison against unsmoothed L2/PCA/chronochrome and PIF spectral residuals; "
            "city-wise AP/AUROC/Otsu metrics plus map inspection"
        ),
    )


def percentile_rank_score(score: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Convert valid scores to deterministic empirical percentile ranks."""
    values = np.asarray(score[valid_mask], dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float32)
    if values.size <= 1:
        ranks.fill(0.0)
    else:
        ranks[order] = np.linspace(0.0, 1.0, values.size, dtype=np.float32)
    output = np.zeros_like(score, dtype=np.float32)
    output[valid_mask] = ranks
    return output


def post_smoothed_score(
    spec: MethodSpec,
    scores: dict[str, np.ndarray],
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, ConstructionCard]:
    """Apply the same masked Gaussian operator after scalar change scoring."""
    dependency = str(spec.score_mode)
    if dependency not in scores:
        raise ValueError(f"{spec.name} requires the precomputed score {dependency!r}.")
    sigmas = spec.spatial_sigmas or (1.0,)
    if len(sigmas) != 1:
        raise ValueError("Post-smoothed score control expects exactly one sigma.")
    sigma = float(sigmas[0])
    smoothed = multiscale_spatial_features(
        scores[dependency][None, :, :], valid_mask, sigmas=(sigma,)
    )[0]
    return smoothed.astype(np.float32), ConstructionCard(
        method=spec.name,
        source_reference="Generic masked Gaussian score-map regularization; control for the Theiler-inspired pre-feature smoothing experiment.",
        sample_unit="one scalar change score and its Gaussian spatial neighborhood",
        input_object=f"the already computed {dependency} score map",
        subspace_count="none beyond the dependency method",
        basis_shape="not applicable",
        fitting_method=f"compute {dependency}, then masked Gaussian smoothing with sigma={sigma:g}",
        score_definition="Gaussian-weighted local average of the scalar change score",
        spatial_information="spatial support enters only after detection, isolating post-processing from spatial feature construction",
        code_path="phase1.scripts.compare_oscd_spatial_subspaces.post_smoothed_score",
        verification="direct city-wise comparison with pre-smoothed PCA/L2 under the same sigma",
    )


def rank_fusion_score(
    name: str,
    dependency_text: str,
    scores: dict[str, np.ndarray],
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, ConstructionCard]:
    dependencies = dependency_text.split("+")
    missing = [item for item in dependencies if item not in scores]
    if missing:
        raise ValueError(
            f"{name} requires {missing}. List computed methods such as band_image_norm and ir_mad before the fusion."
        )
    ranked = [percentile_rank_score(scores[item], valid_mask) for item in dependencies]
    fused = np.mean(np.stack(ranked, axis=0), axis=0, dtype=np.float32)
    fused[~valid_mask] = 0.0
    card = ConstructionCard(
        method=name,
        source_reference="Label-free score-level rank aggregation; diagnostic ensemble rather than a new subspace method.",
        sample_unit="one valid pixel score from each component method",
        input_object=f"component score maps: {', '.join(dependencies)}",
        subspace_count="none at fusion stage; component methods retain their own constructions",
        basis_shape="not applicable",
        fitting_method="convert each component map to within-image empirical percentile ranks, then take an unweighted mean",
        score_definition="mean component percentile rank per valid pixel",
        spatial_information="inherits component map layout; no neighborhood operation or label fitting is introduced",
        code_path="phase1.scripts.compare_oscd_spatial_subspaces.rank_fusion_score",
        verification="fixed equal-weight, label-free diagnostic; must be compared city-wise against every component",
    )
    return fused, card


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


def save_band_attribution_grid(path: Path, band_maps: dict[str, np.ndarray], valid_mask: np.ndarray, percentile: float) -> None:
    panels = [(band, band_maps[band]) for band in BAND_ORDER if band in band_maps]
    cols = 4
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (band, score) in zip(axes.ravel(), panels):
        ax.imshow(normalize_for_display(score, valid_mask, percentile), cmap="viridis", vmin=0, vmax=1)
        ax.set_title(band, fontsize=10)
    fig.suptitle("Band-Image DS projected-energy attribution by Sentinel-2 band", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_band_attribution_csv(path: Path, band_maps: dict[str, np.ndarray], valid_mask: np.ndarray) -> None:
    totals = []
    for band in BAND_ORDER:
        score = band_maps.get(band)
        total = float(np.sum(score[valid_mask], dtype=np.float64)) if score is not None else 0.0
        totals.append((band, total))
    denom = sum(total for _, total in totals)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["band", "projected_energy_sum", "projected_energy_fraction"])
        writer.writeheader()
        for band, total in totals:
            writer.writerow(
                {
                    "band": band,
                    "projected_energy_sum": total,
                    "projected_energy_fraction": total / denom if denom > 0.0 else 0.0,
                }
            )


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

    print(f"Loaded {sample.city}/{split}: pre={sample.x_pre.shape}, post={sample.x_post.shape}, valid={int(eval_mask.sum())}", flush=True)
    print("Task: binary pixel change-segmentation scoring on OSCD labels.", flush=True)
    print("Meaningful here means: label ranking, thresholded segmentation, baseline pressure, spatial construction clarity, and map inspection.", flush=True)

    methods = [parse_method_spec(x) for x in args.methods.split(",") if x.strip()]
    pif_families = {"pif_radiometric", "pif_nuisance_ds", "pif_delta_residual", "spatial_pif_delta"}
    pif_context: PIFContext | None = None
    if any(spec.family in pif_families for spec in methods):
        print("Estimating shared iMAD pseudo-invariant pixels for PIF-conditioned methods...", flush=True)
        pif_context = build_pif_context(
            x1_norm,
            x2_norm,
            eval_mask,
            pmin=float(args.pif_probability),
            fallback_fraction=float(args.pif_fallback_fraction),
            ir_mad_iters=int(args.ir_mad_iters),
            max_fit_pixels=int(args.ir_mad_downsample_max_pixels),
            random_state=int(args.seed),
        )
        print(
            f"  selected {pif_context.selected_pixels} PIF pixels "
            f"(effective probability threshold={pif_context.threshold_used:.6g})",
            flush=True,
        )
        if args.save_npy:
            np.save(maps_dir / "pif_mask.npy", pif_context.pif_mask.astype(np.uint8))
    pif_diagnostics = (
        {
            "pif_pixels": pif_context.selected_pixels,
            "pif_fraction": pif_context.selected_pixels / max(1, int(eval_mask.sum())),
            "pif_effective_threshold": pif_context.threshold_used,
            "pif_requested_threshold": float(args.pif_probability),
        }
        if pif_context is not None
        else {}
    )
    rows: list[dict[str, object]] = []
    cards: list[ConstructionCard] = []
    map_outputs: list[tuple[str, np.ndarray]] = []
    score_by_name: dict[str, np.ndarray] = {}
    band_attribution_outputs: list[dict[str, str]] = []

    raw_l2 = raw_l2_score(x1_norm, x2_norm, eval_mask)
    pca_diff = pca_diff_score(x1_norm, x2_norm, eval_mask, rank_S=args.rank, variance_threshold=None, random_state=args.seed)
    baselines = [("raw_l2", "baseline", raw_l2), ("pca_diff", "baseline", pca_diff)]

    for name, family, score in baselines:
        metrics = score_metrics(score, target, eval_mask, raw_l2)
        rows.append({"method": name, "family": family, "runtime_sec": 0.0, "eval_valid_fraction": float(np.mean(eval_mask)), **pif_diagnostics, **metrics})
        map_outputs.append((name, score))
        score_by_name[name] = score
        save_score_png(maps_dir / f"{name}.png", score, eval_mask, args.score_percentile)
        if args.save_npy:
            np.save(maps_dir / f"{name}.npy", score.astype(np.float32))

    for spec in methods:
        print(f"Running {spec.name}...", flush=True)
        start = time.perf_counter()
        method_mask = eval_mask
        if spec.family == "global_pixel":
            score, card = global_pixel_ds_score(x1_norm, x2_norm, eval_mask, args)
        elif spec.family == "band_image":
            score, card, band_maps = band_image_ds_score(x1_norm, x2_norm, eval_mask, args, spec)
            if args.save_band_attribution:
                grid_path = args.output_dir / f"band_attribution_{spec.name}.png"
                csv_attr_path = args.output_dir / f"band_attribution_{spec.name}.csv"
                save_band_attribution_grid(grid_path, band_maps, eval_mask, args.score_percentile)
                write_band_attribution_csv(csv_attr_path, band_maps, eval_mask)
                band_attribution_outputs.append({"method": spec.name, "grid": str(grid_path), "csv": str(csv_attr_path)})
        elif spec.family == "band_image_control":
            score, card = band_image_spatial_control_score(x1_norm, x2_norm, eval_mask, args, spec)
        elif spec.family == "local_window":
            score, card = local_window_ds_score(x1_norm, x2_norm, eval_mask, args, spec)
        elif spec.family == "local_ds_magnitude":
            score, card = local_ds_magnitude_score(x1_norm, x2_norm, eval_mask, args, spec)
        elif spec.family == "spatial_pyramid":
            score, card = spatial_pyramid_ds_score(x1_norm, x2_norm, eval_mask, args, spec)
        elif spec.family == "patch_vector":
            score, card, method_mask = patch_vector_ds_score(x1_norm, x2_norm, eval_mask, args, spec)
            method_mask = method_mask & eval_mask
        elif spec.family == "celik_pca_kmeans":
            score, card = celik_pca_kmeans_score(x1_norm, x2_norm, eval_mask, args)
        elif spec.family == "ir_mad":
            score, card = ir_mad_baseline_score(x1_norm, x2_norm, eval_mask, args)
        elif spec.family in {"chronochrome", "covariance_equalization"}:
            score, card = anomalous_change_baseline_score(spec.family, x1_norm, x2_norm, eval_mask, args)
        elif spec.family in pif_families:
            if pif_context is None:
                raise AssertionError("PIF context was not initialized.")
            if spec.family == "spatial_pif_delta":
                score, card = spatial_feature_score(spec, x1_norm, x2_norm, eval_mask, args, pif_context)
            else:
                score, card = pif_method_score(spec, x1_norm, x2_norm, eval_mask, args, pif_context)
        elif spec.family in {"spatial_feature_l2", "spatial_feature_pca", "spatial_feature_chronochrome", "spatial_feature_ds", "spatial_feature_weighted_ds"}:
            score, card = spatial_feature_score(spec, x1_norm, x2_norm, eval_mask, args, pif_context)
        elif spec.family == "rank_fusion":
            score, card = rank_fusion_score(spec.name, spec.score_mode, score_by_name, eval_mask)
        elif spec.family == "score_post_smoothing":
            score, card = post_smoothed_score(spec, score_by_name, eval_mask)
        else:
            raise AssertionError(f"Unhandled method family: {spec.family}")
        runtime = time.perf_counter() - start
        metrics = score_metrics(score, target, method_mask, raw_l2)
        rows.append({"method": spec.name, "family": spec.family, "runtime_sec": float(runtime), "eval_valid_fraction": float(np.mean(method_mask)), **pif_diagnostics, **metrics})
        cards.append(card)
        map_outputs.append((spec.name, score))
        score_by_name[spec.name] = score
        save_score_png(maps_dir / f"{spec.name}.png", score, method_mask, args.score_percentile)
        if args.save_npy:
            np.save(maps_dir / f"{spec.name}.npy", score.astype(np.float32))
        print(f"  done in {runtime:.2f}s, AUROC={metrics['auroc']:.4f}, AP={metrics['average_precision']:.4f}, Otsu F1={metrics['otsu_f1']:.4f}", flush=True)
        gc.collect()

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
            "band_attribution": band_attribution_outputs,
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
