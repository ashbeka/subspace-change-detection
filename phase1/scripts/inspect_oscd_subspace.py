"""
Inspect how OSCD subspaces are built and how DS variants behave.

Source/provenance:
- Uses the same OSCD Sentinel-2 sample definition as the current global-pixel
  DS path: each valid pixel becomes one 13-dimensional sample vector.
- Compares the old residual-stack project variant with projector-eig and
  canonical DS variants implemented in `phase1.ds.pca_utils`.
- Raw spectral L2 correlation is included to detect whether a DS score behaves
  like naive pixel differencing.

Purpose:
- Read-only explanation and sanity check. Use this before claiming that a
  subspace construction is paper-faithful or meaningfully different from raw L2.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.data.oscd_dataset import OSCDEvaluatorDataset
from phase1.data.preprocessing import apply_normalization, load_band_stats, vectorize_cube
from phase1.ds import pca_utils
from phase1.eval.utils import suppress_rasterio_warnings


BAND_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
VARIANTS = ["legacy_residual_stack", "eig", "canonical"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Inspect OSCD PCA/DS construction for one city.")
    ap.add_argument("--oscd_root", type=Path, default=Path("data/OSCD"))
    ap.add_argument("--stats_path", type=Path, default=Path("phase1/data/oscd_band_stats.json"))
    ap.add_argument("--city", default="beirut")
    ap.add_argument("--split", default="auto", choices=["auto", "train", "val", "test"])
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--sample_pixels", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=1234)
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


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    suppress_rasterio_warnings()
    args = parse_args()
    split, sample = _load_city(args)
    stats = load_band_stats(args.stats_path)

    x1_norm, valid_mask = apply_normalization(sample.x_pre, stats, valid_mask=sample.valid_mask, nodata_value=0.0)
    x2_norm, _ = apply_normalization(sample.x_post, stats, valid_mask=sample.valid_mask, nodata_value=0.0)
    x1_mat, _ = vectorize_cube(x1_norm, valid_mask)
    x2_mat, _ = vectorize_cube(x2_norm, valid_mask)

    pre_pca = pca_utils.fit_pca_basis(
        x1_mat,
        rank=args.rank,
        variance_threshold=None,
        random_state=args.seed,
        use_randomized=True,
    )
    post_pca = pca_utils.fit_pca_basis(
        x2_mat,
        rank=args.rank,
        variance_threshold=None,
        random_state=args.seed,
        use_randomized=True,
    )
    phi = pre_pca.basis
    psi = post_pca.basis

    rng = np.random.default_rng(args.seed)
    n_pixels = x1_mat.shape[1]
    n_sample = min(int(args.sample_pixels), n_pixels)
    sel = rng.choice(n_pixels, size=n_sample, replace=False)
    diff = (x2_mat[:, sel] - x1_mat[:, sel]).astype(np.float32, copy=False)
    raw_l2 = np.sum(diff * diff, axis=0)

    print("OSCD subspace inspection")
    print(f"  city/split: {sample.city}/{split}")
    print(f"  cube shape: pre={sample.x_pre.shape}, post={sample.x_post.shape}")
    print(f"  matrix shape: X_pre={x1_mat.shape}, X_post={x2_mat.shape}")
    print("  interpretation: each valid pixel is one 13-D spectral sample")
    print(f"  valid pixels: {n_pixels}")
    print(f"  PCA rank requested: {args.rank}")
    print(f"  Phi/Psi basis shapes: {phi.shape}, {psi.shape}")
    print(f"  pre explained variance first {pre_pca.rank}: {np.round(pre_pca.explained_variance_ratio, 6).tolist()}")
    print(f"  post explained variance first {post_pca.rank}: {np.round(post_pca.explained_variance_ratio, 6).tolist()}")
    print(f"  sampled pixels for score comparison: {n_sample}")
    print()

    for variant in VARIANTS:
        d_basis = pca_utils.build_difference_subspace(phi, psi, variant=variant)
        energy = np.sum((d_basis.T @ diff) ** 2, axis=0) if d_basis.shape[1] else np.zeros_like(raw_l2)
        mean_ratio = float(np.mean(energy / (raw_l2 + 1e-9)))
        print(f"{variant}")
        print(f"  D shape: {d_basis.shape}")
        print(f"  rank(D): {np.linalg.matrix_rank(d_basis, tol=1e-6)}")
        print(f"  corr(score, raw_l2): {_safe_corr(raw_l2, energy):.6f}")
        print(f"  mean score/raw_l2 energy ratio: {mean_ratio:.6f}")
        print(f"  score mean/std: {float(np.mean(energy)):.6f} / {float(np.std(energy)):.6f}")
        print()


if __name__ == "__main__":
    main()
