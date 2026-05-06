"""
Venus subspace/KPCA audit demo for the TPAMI 2015-style data.

This is a small, explicit prototype:
- linear DS uses whole downsampled images as vectors;
- the KPCA/KDS section uses a shared RBF-KPCA coordinate embedding and then
  applies canonical DS in that nonlinear coordinate space.

It is not a paper-faithful preimage reconstruction implementation of KDS/KGDS.
Outputs are labeled accordingly.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

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


VENUS_FILES = {
    "nothing": ("venus_nothing.mat", "venus_nothing"),
    "earrings": ("venus_er2.mat", "venus_er2"),
    "earrings_necklace": ("venus_er_ne.mat", "venus_er_neck"),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a small Venus linear-DS and KPCA/KDS prototype.")
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--width", type=int, default=63)
    ap.add_argument("--height", type=int, default=48)
    ap.add_argument("--linear_rank", type=int, default=10)
    ap.add_argument("--kpca_components", type=int, default=150)
    ap.add_argument("--kernel_subspace_rank", type=int, default=100)
    ap.add_argument("--sigma2", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1234)
    return ap.parse_args()


def _default_output_dir() -> Path:
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("phase1/outputs") / f"venus_kds_audit_{tag}"


def _load_venus_matrix(path: Path, key: str, width: int, height: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
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
        vec = frame.reshape(-1)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        frames.append(frame)
        columns.append(vec)
    return np.stack(columns, axis=1), np.stack(frames, axis=0), arr.shape


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
    fig, axes = plt.subplots(len(frames_by_label), cols, figsize=(15, 5))
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


def _save_component_grid(title: str, components: Dict[str, np.ndarray], out_path: Path, height: int, width: int, max_cols: int = 8) -> None:
    rows = len(components)
    cols = min(max_cols, max(v.shape[1] for v in components.values()))
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


def _rbf_kernel(x: np.ndarray, sigma2: float) -> np.ndarray:
    sq_norm = np.sum(x * x, axis=0, keepdims=True)
    d2 = sq_norm.T + sq_norm - 2.0 * (x.T @ x)
    return np.exp(-np.maximum(d2, 0.0) / float(sigma2)).astype(np.float64)


def _center_kernel(k: np.ndarray) -> np.ndarray:
    n = k.shape[0]
    one = np.ones((n, n), dtype=k.dtype) / float(n)
    return k - one @ k - k @ one + one @ k @ one


def _kpca_coordinates(x: np.ndarray, components: int, sigma2: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = _center_kernel(_rbf_kernel(x, sigma2))
    eigvals, eigvecs = np.linalg.eigh(k)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    keep = eigvals > 1e-9
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]
    m = min(int(components), eigvecs.shape[1])
    coords = eigvecs[:, :m] * np.sqrt(eigvals[:m])[None, :]
    ratio = eigvals[:m] / max(float(np.sum(eigvals)), 1e-12)
    return coords.astype(np.float32), eigvals[:m].astype(np.float32), ratio.astype(np.float32)


def _safe_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[1] == 0 or b.shape[1] == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.linalg.svd(a.T @ b, compute_uv=False).astype(np.float32)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or _default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    matrices: Dict[str, np.ndarray] = {}
    frames: Dict[str, np.ndarray] = {}
    raw_shapes: Dict[str, Tuple[int, ...]] = {}
    for label, (file_name, key) in VENUS_FILES.items():
        x, f, raw_shape = _load_venus_matrix(Path(file_name), key, args.width, args.height)
        matrices[label] = x
        frames[label] = f
        raw_shapes[label] = tuple(int(v) for v in raw_shape)

    _save_montage(frames, out_dir / "venus_montage.png")

    pca_bases = {}
    pca_ratios = {}
    for label, x in matrices.items():
        basis, ratio = _pca_basis(x, args.linear_rank)
        pca_bases[label] = basis
        pca_ratios[label] = ratio
    _save_component_grid("Linear PCA basis images", pca_bases, out_dir / "venus_linear_pca_basis.png", args.height, args.width, max_cols=8)

    pair_a = "earrings"
    pair_b = "nothing"
    d_linear = pca_utils.difference_subspace_canonical(pca_bases[pair_a], pca_bases[pair_b])
    _save_component_grid(
        f"Linear canonical DS basis: {pair_a} vs {pair_b}",
        {f"{pair_a}_vs_{pair_b}": d_linear},
        out_dir / "venus_linear_ds_basis.png",
        args.height,
        args.width,
        max_cols=8,
    )

    combined = np.concatenate([matrices[pair_a], matrices[pair_b]], axis=1)
    coords, kpca_eigvals, kpca_ratio = _kpca_coordinates(combined, args.kpca_components, args.sigma2)
    n_a = matrices[pair_a].shape[1]
    coords_a = coords[:n_a].T
    coords_b = coords[n_a:].T
    kpca_rank = min(args.kernel_subspace_rank, coords_a.shape[0], coords_a.shape[1], coords_b.shape[1])
    basis_a = pca_utils.fit_pca_basis(coords_a, rank=kpca_rank, variance_threshold=None, use_randomized=False).basis
    basis_b = pca_utils.fit_pca_basis(coords_b, rank=kpca_rank, variance_threshold=None, use_randomized=False).basis
    d_kpca = pca_utils.difference_subspace_canonical(basis_a, basis_b)
    kpca_cosines = _safe_cosines(basis_a, basis_b)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(kpca_ratio[: min(50, kpca_ratio.shape[0])], marker="o", markersize=2)
    axes[0].set_title("Shared RBF-KPCA variance ratio")
    axes[0].set_xlabel("component")
    axes[0].set_ylabel("ratio")
    axes[1].imshow(d_kpca, aspect="auto", cmap="coolwarm", vmin=-float(np.max(np.abs(d_kpca)) or 1.0), vmax=float(np.max(np.abs(d_kpca)) or 1.0))
    axes[1].set_title("Prototype KPCA-coordinate DS basis")
    axes[1].set_xlabel("DS component")
    axes[1].set_ylabel("KPCA coordinate")
    fig.suptitle("KPCA/KDS prototype diagnostics")
    fig.tight_layout()
    fig.savefig(out_dir / "venus_kpca_kds_prototype.png", dpi=160)
    plt.close(fig)

    summary = {
        "status": "prototype_not_paper_faithful_preimage_kds",
        "raw_shapes": raw_shapes,
        "downsampled_image_shape_hw": [args.height, args.width],
        "whole_image_matrix_shape": {k: list(v.shape) for k, v in matrices.items()},
        "linear_rank": int(args.linear_rank),
        "linear_pca_explained_ratio": {k: v.tolist() for k, v in pca_ratios.items()},
        "linear_pair": [pair_a, pair_b],
        "linear_ds_shape": list(d_linear.shape),
        "linear_principal_cosines": _safe_cosines(pca_bases[pair_a], pca_bases[pair_b]).tolist(),
        "kpca_pair": [pair_a, pair_b],
        "kpca_sigma2": float(args.sigma2),
        "kpca_components": int(coords.shape[1]),
        "kpca_eigenvalues_first10": kpca_eigvals[:10].tolist(),
        "kernel_subspace_rank": int(kpca_rank),
        "kpca_coordinate_basis_shapes": [list(basis_a.shape), list(basis_b.shape)],
        "kpca_coordinate_ds_shape": list(d_kpca.shape),
        "kpca_principal_cosines_first20": kpca_cosines[:20].tolist(),
        "outputs": [
            "venus_montage.png",
            "venus_linear_pca_basis.png",
            "venus_linear_ds_basis.png",
            "venus_kpca_kds_prototype.png",
            "run_summary.json",
        ],
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved Venus audit outputs to {out_dir}")
    print(f"Linear DS basis shape: {d_linear.shape}")
    print(f"Prototype KPCA-coordinate DS basis shape: {d_kpca.shape}")
    print("KPCA/KDS note: this is a diagnostic prototype, not a paper-faithful preimage reconstruction.")


if __name__ == "__main__":
    main()
