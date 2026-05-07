"""
Kernel difference-subspace utilities for the TPAMI 2015 KDS/KGDS equations.

The representation follows Fukui and Maki's kernel formulation:

- each nonlinear subspace basis vector is stored as a coefficient vector over
  training samples, e_i = sum_l a_li phi(x_l);
- KDS/KGDS basis vectors are linear combinations of those nonlinear subspace
  bases, d_j = sum_i b_ij e_i;
- projection of a new sample is computed only through kernel evaluations.

This module intentionally does not implement preimage reconstruction. It gives
the RKHS projection coordinates needed before a separate preimage search step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class KernelSubspace:
    """Kernel-PCA subspace represented by training samples and coefficients."""

    samples: Array  # shape (d, n_samples)
    coeffs: Array  # shape (n_samples, rank)
    eigvals: Array  # shape (rank,)
    sigma2: float
    label: str = ""

    @property
    def rank(self) -> int:
        return int(self.coeffs.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[1])


@dataclass(frozen=True)
class KernelDifferenceSubspace:
    """KDS/KGDS basis represented by coefficients over kernel subspaces."""

    subspaces: tuple[KernelSubspace, ...]
    basis_coeffs_over_subspace_bases: Array  # shape (sum_ranks, kds_rank)
    eigvals: Array  # selected positive eigenvalues of E^T E
    sample_coeffs_by_subspace: tuple[Array, ...]  # each shape (n_samples_i, kds_rank)
    basis_gram: Array  # E^T E, shape (sum_ranks, sum_ranks)

    @property
    def rank(self) -> int:
        return int(self.basis_coeffs_over_subspace_bases.shape[1])

    @property
    def labels(self) -> list[str]:
        return [s.label for s in self.subspaces]

    @property
    def sigma2(self) -> float:
        return float(self.subspaces[0].sigma2)


def rbf_kernel(x: Array, y: Array | None = None, sigma2: float = 5.0) -> Array:
    """
    Compute exp(-||x-y||^2 / sigma2) for column-oriented matrices.

    Parameters
    ----------
    x:
        Matrix with samples in columns, shape `(d, n)`.
    y:
        Optional second matrix with samples in columns, shape `(d, m)`.
        If omitted, `y=x`.
    sigma2:
        Squared bandwidth parameter used by the TPAMI Venus experiment.
    """
    x = np.asarray(x, dtype=np.float64)
    y = x if y is None else np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected 2D column matrices, got {x.shape} and {y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Feature dimensions differ: {x.shape[0]} != {y.shape[0]}")
    if float(sigma2) <= 0:
        raise ValueError("sigma2 must be positive.")

    x_norm = np.sum(x * x, axis=0, keepdims=True)
    y_norm = np.sum(y * y, axis=0, keepdims=True)
    d2 = x_norm.T + y_norm - 2.0 * (x.T @ y)
    return np.exp(-np.maximum(d2, 0.0) / float(sigma2))


def fit_kernel_subspace(
    samples: Array,
    rank: int,
    sigma2: float = 5.0,
    label: str = "",
    eig_eps: float = 1e-10,
) -> KernelSubspace:
    """
    Fit a nonlinear subspace from samples using the paper's kernel coefficient form.

    If `K v_i = lambda_i v_i` and `v_i` has unit Euclidean norm, the stored
    coefficient is `a_i = v_i / sqrt(lambda_i)`, so
    `a_i.T @ K @ a_i = 1`.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected samples shaped (d, n), got {x.shape}")
    if x.shape[1] == 0:
        raise ValueError("Cannot fit a kernel subspace with zero samples.")
    if rank <= 0:
        raise ValueError("rank must be positive.")

    k = rbf_kernel(x, sigma2=sigma2)
    eigvals, eigvecs = np.linalg.eigh(k)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    keep = eigvals > float(eig_eps)
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]
    r = min(int(rank), eigvecs.shape[1])
    if r == 0:
        raise ValueError("No positive kernel eigenvalues survived eig_eps.")

    eigvals = eigvals[:r]
    coeffs = eigvecs[:, :r] / np.sqrt(eigvals[None, :])
    return KernelSubspace(
        samples=x,
        coeffs=coeffs.astype(np.float64, copy=False),
        eigvals=eigvals.astype(np.float64, copy=False),
        sigma2=float(sigma2),
        label=label,
    )


def basis_inner_product_matrix(subspaces: Sequence[KernelSubspace]) -> Array:
    """Build the matrix E^T E between all nonlinear subspace basis vectors."""
    if len(subspaces) < 2:
        raise ValueError("KDS/KGDS needs at least two kernel subspaces.")
    sigma2 = float(subspaces[0].sigma2)
    for s in subspaces:
        if abs(float(s.sigma2) - sigma2) > 1e-12:
            raise ValueError("All kernel subspaces must use the same sigma2.")

    sizes = [s.rank for s in subspaces]
    offsets = np.cumsum([0] + sizes)
    gram = np.zeros((offsets[-1], offsets[-1]), dtype=np.float64)
    for i, si in enumerate(subspaces):
        for j, sj in enumerate(subspaces[i:], start=i):
            kij = rbf_kernel(si.samples, sj.samples, sigma2=sigma2)
            block = si.coeffs.T @ kij @ sj.coeffs
            ri = slice(offsets[i], offsets[i + 1])
            rj = slice(offsets[j], offsets[j + 1])
            gram[ri, rj] = block
            if i != j:
                gram[rj, ri] = block.T
    return 0.5 * (gram + gram.T)


def kernel_difference_subspace(
    subspaces: Sequence[KernelSubspace],
    rank: int | None = None,
    eig_eps: float = 1e-9,
) -> KernelDifferenceSubspace:
    """
    Construct KDS/KGDS from nonlinear subspaces.

    The paper obtains difference directions from eigenvectors of `E^T E`
    corresponding to the smallest eigenvalues. Exact zero eigenvalues are skipped
    because they produce the zero RKHS vector; the selected positive directions
    are normalized by `sqrt(lambda)` so their RKHS norms are one.
    """
    subs = tuple(subspaces)
    gram = basis_inner_product_matrix(subs)
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if rank is None:
        candidates = np.arange(eigvals.shape[0])
    else:
        candidates = np.arange(min(max(0, int(rank)), eigvals.shape[0]))
    keep = candidates[eigvals[candidates] > float(eig_eps)]
    if keep.size == 0:
        empty_blocks = tuple(np.zeros((s.n_samples, 0), dtype=np.float64) for s in subs)
        return KernelDifferenceSubspace(
            subspaces=subs,
            basis_coeffs_over_subspace_bases=np.zeros((gram.shape[0], 0), dtype=np.float64),
            eigvals=np.zeros((0,), dtype=np.float64),
            sample_coeffs_by_subspace=empty_blocks,
            basis_gram=gram,
        )

    selected_vals = eigvals[keep]
    b = eigvecs[:, keep] / np.sqrt(selected_vals[None, :])

    sizes = [s.rank for s in subs]
    offsets = np.cumsum([0] + sizes)
    sample_coeffs: list[Array] = []
    for i, subspace in enumerate(subs):
        block = b[offsets[i] : offsets[i + 1], :]
        sample_coeffs.append(subspace.coeffs @ block)

    return KernelDifferenceSubspace(
        subspaces=subs,
        basis_coeffs_over_subspace_bases=b.astype(np.float64, copy=False),
        eigvals=selected_vals.astype(np.float64, copy=False),
        sample_coeffs_by_subspace=tuple(sample_coeffs),
        basis_gram=gram,
    )


def project_kernel_difference(kds: KernelDifferenceSubspace, samples: Array) -> Array:
    """
    Project input samples onto a KDS/KGDS basis using only kernel evaluations.

    This implements the paper's Eq. 16/17:
    z_i = sum_j sum_l b_ij a_h(j)l^z(j) k(x_l^z(j), x).
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected samples shaped (d, n), got {x.shape}")
    if kds.rank == 0:
        return np.zeros((0, x.shape[1]), dtype=np.float64)

    coords = np.zeros((kds.rank, x.shape[1]), dtype=np.float64)
    for subspace, coeffs in zip(kds.subspaces, kds.sample_coeffs_by_subspace):
        k_cross = rbf_kernel(subspace.samples, x, sigma2=subspace.sigma2)
        coords += coeffs.T @ k_cross
    return coords


def projection_energy(kds: KernelDifferenceSubspace, samples: Array) -> Array:
    """Return squared KDS/KGDS projection norm per input sample."""
    coords = project_kernel_difference(kds, samples)
    return np.sum(coords * coords, axis=0)


def rkhs_basis_gram(kds: KernelDifferenceSubspace) -> Array:
    """Return inner products between the constructed KDS/KGDS basis vectors."""
    b = kds.basis_coeffs_over_subspace_bases
    return b.T @ kds.basis_gram @ b
