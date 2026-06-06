"""
Venus DS/KDS/KGDS audit demo for Fukui and Maki TPAMI 2015.

This script uses the representation Sensei's Venus files imply:

  one downsampled whole image view = one sample vector
  one sculpture set = matrix X in R^(3024 x 300) for 63x48 views

Implemented here:
- linear PCA subspaces and linear canonical DS visualization in image space;
- paper-formula KDS/KGDS construction in RKHS:
  1. fit kernel subspace bases e_i = sum_l a_li phi(x_l);
  2. build E^T E from kernel inner products between nonlinear bases;
  3. take smallest positive eigen-directions for KDS/KGDS;
  4. project images with Eq. 16/17 using kernel evaluations.

Not implemented here:
- preimage reconstruction for visualizing RKHS basis/projection images.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1.ds import pca_utils
from phase1.subspace import kernel_difference_subspace as kds_math


VENUS_FILES = {
    "nothing": ("venus_tpami2015_no_accessories.mat", "venus_nothing"),
    "earrings": ("venus_tpami2015_earrings.mat", "venus_er2"),
    "earrings_necklace": ("venus_tpami2015_earrings_necklace.mat", "venus_er_neck"),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Venus linear DS plus paper-formula KDS/KGDS diagnostics.")
    ap.add_argument("--venus_root", type=Path, default=Path("data/venus_tpami2015"))
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--width", type=int, default=63)
    ap.add_argument("--height", type=int, default=48)
    ap.add_argument("--linear_rank", type=int, default=10)
    ap.add_argument("--kds_subspace_rank", type=int, default=100)
    ap.add_argument("--kds_rank", type=int, default=100)
    ap.add_argument("--kgds_subspace_rank", type=int, default=150)
    ap.add_argument("--kgds_rank", type=int, default=300)
    ap.add_argument("--sigma2", type=float, default=5.0)
    ap.add_argument("--eig_eps", type=float, default=1e-9)
    ap.add_argument("--no_l2_normalize", action="store_true")
    return ap.parse_args()


def _default_output_dir() -> Path:
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("phase1/outputs") / f"venus_kds_audit_{tag}"


def _load_venus_matrix(
    path: Path,
    key: str,
    width: int,
    height: int,
    l2_normalize: bool,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    data = sio.loadmat(path)
    if key not in data:
        keys = [k for k in data if not k.startswith("__")]
        raise KeyError(f"{path} does not contain key {key!r}; available keys: {keys}")
    arr = data[key]
    if arr.ndim != 4:
        raise ValueError(f"Expected (H,W,1,N) array in {path}, got {arr.shape}")

    frames = []
    columns = []
    for i in range(arr.shape[3]):
        im = Image.fromarray(arr[:, :, 0, i])
        im = im.resize((width, height), Image.Resampling.BILINEAR)
        frame = np.asarray(im, dtype=np.float32) / 255.0
        vec = frame.reshape(-1).astype(np.float64)
        if l2_normalize:
            vec = vec / (np.linalg.norm(vec) + 1e-12)
        frames.append(frame)
        columns.append(vec)
    return np.stack(columns, axis=1), np.stack(frames, axis=0), tuple(int(v) for v in arr.shape)


def _pca_basis(x: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    x_centered = x - np.mean(x, axis=1, keepdims=True)
    u, s, _ = np.linalg.svd(x_centered, full_matrices=False)
    eig = (s * s) / max(1, x.shape[1] - 1)
    ratio = eig / max(float(np.sum(eig)), 1e-12)
    r = min(int(rank), u.shape[1])
    return u[:, :r].astype(np.float32), ratio[:r].astype(np.float32)


def _signed_image(vec: np.ndarray, height: int, width: int) -> np.ndarray:
    img = vec.reshape(height, width)
    limit = float(np.max(np.abs(img)))
    if limit <= 0:
        return np.zeros_like(img)
    return img / limit


def _save_montage(frames_by_label: Dict[str, np.ndarray], out_path: Path) -> None:
    cols = 10
    fig, axes = plt.subplots(len(frames_by_label), cols, figsize=(15, 5), squeeze=False)
    for row, (label, frames) in enumerate(frames_by_label.items()):
        idx = np.linspace(0, frames.shape[0] - 1, cols, dtype=int)
        for col, frame_idx in enumerate(idx):
            ax = axes[row, col]
            ax.imshow(frames[frame_idx], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label)
            ax.set_title(str(int(frame_idx)), fontsize=8)
    fig.suptitle("Venus downsampled view montage")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_component_grid(
    title: str,
    components: Dict[str, np.ndarray],
    out_path: Path,
    height: int,
    width: int,
    max_cols: int = 8,
) -> None:
    rows = len(components)
    cols = min(max_cols, max(max(1, v.shape[1]) for v in components.values()))
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.2 * rows), squeeze=False)
    for row, (label, basis) in enumerate(components.items()):
        for col in range(cols):
            ax = axes[row, col]
            if col < basis.shape[1]:
                ax.imshow(_signed_image(basis[:, col], height, width), cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label)
            ax.set_title(f"c{col + 1}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _basis_norm_error(subspace: kds_math.KernelSubspace) -> float:
    gram = kds_math.rbf_kernel(subspace.samples, sigma2=subspace.sigma2)
    ident = subspace.coeffs.T @ gram @ subspace.coeffs
    return float(np.max(np.abs(ident - np.eye(subspace.rank))))


def _kds_norm_error(kds: kds_math.KernelDifferenceSubspace) -> float:
    if kds.rank == 0:
        return 0.0
    ident = kds_math.rkhs_basis_gram(kds)
    return float(np.max(np.abs(ident - np.eye(kds.rank))))


def _safe_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[1] == 0 or b.shape[1] == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.linalg.svd(a.T @ b, compute_uv=False).astype(np.float32)


def _normalize_for_plot(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    vmax = float(np.max(values)) if values.size else 0.0
    if vmax <= 0:
        return np.zeros_like(values)
    return values / vmax


def _save_projection_energy_plot(
    title: str,
    kds: kds_math.KernelDifferenceSubspace,
    matrices: Dict[str, np.ndarray],
    labels: Iterable[str],
    out_path: Path,
) -> Dict[str, Dict[str, float]]:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    stats: Dict[str, Dict[str, float]] = {}
    for label in labels:
        energy = kds_math.projection_energy(kds, matrices[label])
        ax.plot(_normalize_for_plot(energy), label=label, linewidth=1.4)
        stats[label] = {
            "min": float(np.min(energy)),
            "max": float(np.max(energy)),
            "mean": float(np.mean(energy)),
            "std": float(np.std(energy)),
        }
    ax.set_title(title)
    ax.set_xlabel("view index")
    ax.set_ylabel("normalized squared projection")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return stats


def _save_projection_heatmap(
    title: str,
    kds: kds_math.KernelDifferenceSubspace,
    matrix: np.ndarray,
    out_path: Path,
    max_components: int = 40,
) -> None:
    coords = kds_math.project_kernel_difference(kds, matrix)
    shown = coords[: min(max_components, coords.shape[0]), :]
    limit = float(np.percentile(np.abs(shown), 99.0)) if shown.size else 1.0
    limit = max(limit, 1e-12)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(shown, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_title(title)
    ax.set_xlabel("view index")
    ax.set_ylabel("KDS/KGDS component")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_spectrum_plot(
    title: str,
    values: Dict[str, np.ndarray],
    out_path: Path,
    max_points: int = 80,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, eigvals in values.items():
        y = np.asarray(eigvals, dtype=np.float64)[:max_points]
        ax.plot(y, marker="o", markersize=2.5, linewidth=1.2, label=label)
    ax.set_title(title)
    ax.set_xlabel("component")
    ax.set_ylabel("eigenvalue")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _fit_kernel_subspaces(
    matrices: Dict[str, np.ndarray],
    labels: Iterable[str],
    rank: int,
    sigma2: float,
    eig_eps: float,
) -> Dict[str, kds_math.KernelSubspace]:
    return {
        label: kds_math.fit_kernel_subspace(
            matrices[label],
            rank=rank,
            sigma2=sigma2,
            label=label,
            eig_eps=eig_eps,
        )
        for label in labels
    }


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or _default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    venus_root = args.venus_root if args.venus_root.is_absolute() else ROOT / args.venus_root

    l2_normalize = not args.no_l2_normalize
    matrices: Dict[str, np.ndarray] = {}
    frames: Dict[str, np.ndarray] = {}
    raw_shapes: Dict[str, Tuple[int, int, int, int]] = {}
    for label, (file_name, key) in VENUS_FILES.items():
        x, f, raw_shape = _load_venus_matrix(venus_root / file_name, key, args.width, args.height, l2_normalize)
        matrices[label] = x
        frames[label] = f
        raw_shapes[label] = raw_shape

    _save_montage(frames, out_dir / "venus_montage.png")

    pca_bases = {}
    pca_ratios = {}
    for label, x in matrices.items():
        basis, ratio = _pca_basis(x, args.linear_rank)
        pca_bases[label] = basis
        pca_ratios[label] = ratio
    _save_component_grid(
        "Linear PCA basis images",
        pca_bases,
        out_dir / "venus_linear_pca_basis.png",
        args.height,
        args.width,
        max_cols=8,
    )

    pair_labels = ("earrings", "nothing")
    d_linear = pca_utils.difference_subspace_canonical(pca_bases[pair_labels[0]], pca_bases[pair_labels[1]])
    _save_component_grid(
        f"Linear canonical DS basis: {pair_labels[0]} vs {pair_labels[1]}",
        {f"{pair_labels[0]}_vs_{pair_labels[1]}": d_linear},
        out_dir / "venus_linear_ds_basis.png",
        args.height,
        args.width,
        max_cols=8,
    )

    pair_subspaces = _fit_kernel_subspaces(matrices, pair_labels, args.kds_subspace_rank, args.sigma2, args.eig_eps)
    kds_pair = kds_math.kernel_difference_subspace(
        [pair_subspaces[pair_labels[0]], pair_subspaces[pair_labels[1]]],
        rank=args.kds_rank,
        eig_eps=args.eig_eps,
    )
    kds_energy_stats = _save_projection_energy_plot(
        f"Paper-formula KDS projection: {pair_labels[0]} vs {pair_labels[1]}",
        kds_pair,
        matrices,
        pair_labels,
        out_dir / "venus_kernel_kds_pair_energy.png",
    )
    _save_projection_heatmap(
        f"KDS coordinates for {pair_labels[0]} views",
        kds_pair,
        matrices[pair_labels[0]],
        out_dir / "venus_kernel_kds_pair_coordinates.png",
    )

    kgds_labels = ("earrings_necklace", "earrings", "nothing")
    kgds_subspaces = _fit_kernel_subspaces(matrices, kgds_labels, args.kgds_subspace_rank, args.sigma2, args.eig_eps)
    kgds = kds_math.kernel_difference_subspace(
        [kgds_subspaces[label] for label in kgds_labels],
        rank=args.kgds_rank,
        eig_eps=args.eig_eps,
    )
    kgds_energy_stats = _save_projection_energy_plot(
        "Paper-formula KGDS projection: three Venus classes",
        kgds,
        matrices,
        kgds_labels,
        out_dir / "venus_kernel_kgds_three_class_energy.png",
    )
    _save_projection_heatmap(
        "KGDS coordinates for earrings_necklace views",
        kgds,
        matrices["earrings_necklace"],
        out_dir / "venus_kernel_kgds_three_class_coordinates.png",
    )

    _save_spectrum_plot(
        "Kernel subspace and KDS/KGDS eigen spectra",
        {
            "KDS selected small eigvals": kds_pair.eigvals,
            "KGDS selected small eigvals": kgds.eigvals,
            "earrings KPCA eigvals": pair_subspaces["earrings"].eigvals,
            "nothing KPCA eigvals": pair_subspaces["nothing"].eigvals,
        },
        out_dir / "venus_kernel_spectra.png",
    )

    pair_basis_errors = {label: _basis_norm_error(subspace) for label, subspace in pair_subspaces.items()}
    kgds_basis_errors = {label: _basis_norm_error(subspace) for label, subspace in kgds_subspaces.items()}

    summary = {
        "status": "paper_formula_kds_kgds_projection_no_preimage_reconstruction",
        "paper_mapping": {
            "sample_vector": "one downsampled whole grayscale Venus view",
            "matrix_per_sculpture": "R^(height*width x 300)",
            "kernel": "exp(-||x-y||^2 / sigma2)",
            "kds_step": "basis vectors are kernel combinations; KDS/KGDS from smallest positive eigenvectors of E^T E; projection uses Eq. 16/17",
            "not_implemented": "preimage reconstruction/search used for TPAMI visual emphasis figures",
        },
        "venus_root": str(venus_root),
        "venus_files": {label: {"file": file_name, "mat_key": key} for label, (file_name, key) in VENUS_FILES.items()},
        "raw_shapes": raw_shapes,
        "downsampled_image_shape_hw": [int(args.height), int(args.width)],
        "whole_image_matrix_shape": {k: list(v.shape) for k, v in matrices.items()},
        "l2_normalize_columns": bool(l2_normalize),
        "sigma2": float(args.sigma2),
        "linear_rank": int(args.linear_rank),
        "linear_pca_explained_ratio": {k: v.tolist() for k, v in pca_ratios.items()},
        "linear_pair": list(pair_labels),
        "linear_ds_shape": list(d_linear.shape),
        "linear_principal_cosines": _safe_cosines(pca_bases[pair_labels[0]], pca_bases[pair_labels[1]]).tolist(),
        "kds_pair": list(pair_labels),
        "kds_kernel_subspace_rank_requested": int(args.kds_subspace_rank),
        "kds_kernel_subspace_rank_actual": {label: int(s.rank) for label, s in pair_subspaces.items()},
        "kds_rank_requested": int(args.kds_rank),
        "kds_rank_actual": int(kds_pair.rank),
        "kds_selected_eigvals_first10": kds_pair.eigvals[:10].tolist(),
        "kds_selected_eigvals_last10": kds_pair.eigvals[-10:].tolist(),
        "kds_subspace_basis_norm_error_max": pair_basis_errors,
        "kds_basis_norm_error_max": _kds_norm_error(kds_pair),
        "kds_projection_energy_stats": kds_energy_stats,
        "kgds_labels": list(kgds_labels),
        "kgds_kernel_subspace_rank_requested": int(args.kgds_subspace_rank),
        "kgds_kernel_subspace_rank_actual": {label: int(s.rank) for label, s in kgds_subspaces.items()},
        "kgds_rank_requested": int(args.kgds_rank),
        "kgds_rank_actual": int(kgds.rank),
        "kgds_selected_eigvals_first10": kgds.eigvals[:10].tolist(),
        "kgds_selected_eigvals_last10": kgds.eigvals[-10:].tolist(),
        "kgds_subspace_basis_norm_error_max": kgds_basis_errors,
        "kgds_basis_norm_error_max": _kds_norm_error(kgds),
        "kgds_projection_energy_stats": kgds_energy_stats,
        "outputs": [
            "venus_montage.png",
            "venus_linear_pca_basis.png",
            "venus_linear_ds_basis.png",
            "venus_kernel_kds_pair_energy.png",
            "venus_kernel_kds_pair_coordinates.png",
            "venus_kernel_kgds_three_class_energy.png",
            "venus_kernel_kgds_three_class_coordinates.png",
            "venus_kernel_spectra.png",
            "run_summary.json",
        ],
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved Venus DS/KDS/KGDS audit outputs to {out_dir}")
    print(f"KDS rank: {kds_pair.rank}, KDS norm error: {_kds_norm_error(kds_pair):.3e}")
    print(f"KGDS rank: {kgds.rank}, KGDS norm error: {_kds_norm_error(kgds):.3e}")
    print("Preimage reconstruction is not implemented; outputs are RKHS projection diagnostics.")


if __name__ == "__main__":
    main()
