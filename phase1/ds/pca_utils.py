"""
PCA helpers for DS (spec Section 4.1).

Functions here handle rank selection by variance retention, orthonormalization,
and projector utilities used by DS scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA


Array = np.ndarray


@dataclass
class PCABasis:
    basis: Array  # shape (d, r)
    explained_variance_ratio: Array  # shape (r,)
    rank: int


def fit_pca_basis(
    x: Array,
    rank: Optional[int] = None,
    variance_threshold: Optional[float] = 0.95,
    random_state: int = 1234,
    use_randomized: bool = True,
) -> PCABasis:
    """
    Fit PCA on X (d, n) and return a basis with either fixed rank or energy threshold.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected (d, n) matrix, got {x.shape}")
    # sklearn expects samples x features, so transpose
    samples = x.T
    solver = "randomized" if use_randomized else "full"
    if variance_threshold is None and rank is not None:
        n_components = max(1, min(int(rank), min(samples.shape)))
    else:
        # Fit full PCA once, then truncate to desired energy.
        n_components = min(samples.shape)
    pca_full = PCA(n_components=n_components, svd_solver=solver, random_state=random_state)
    pca_full.fit(samples)

    if variance_threshold is not None:
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        r = int(np.searchsorted(cumsum, variance_threshold) + 1)
    elif rank is not None:
        r = int(rank)
    else:
        r = pca_full.components_.shape[0]

    r = max(1, min(r, pca_full.components_.shape[0]))
    basis = pca_full.components_[:r].T  # (d, r)
    explained = pca_full.explained_variance_ratio_[:r]
    return PCABasis(basis=basis.astype(np.float32), explained_variance_ratio=explained.astype(np.float32), rank=r)


def orthonormalize(mat: Array) -> Array:
    """Return an orthonormal basis spanning columns of `mat` via QR."""
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {mat.shape}")
    q, _ = np.linalg.qr(mat)
    return q


def residual_projector(basis: Array) -> Array:
    """
    Compute residual projector R = I - P where P = basis basis^T.
    basis is assumed orthonormal (d, r).
    """
    d = basis.shape[0]
    return np.eye(d, dtype=basis.dtype) - basis @ basis.T


def project(basis: Array, x: Array) -> Array:
    """
    Project data matrix (d, n) onto basis (d, r) -> (r, n).
    """
    return basis.T @ x


def reconstruct(basis: Array, coeffs: Array) -> Array:
    """Reconstruct from projection coefficients."""
    return basis @ coeffs


def cross_residual_energy(residual_proj: Array, x: Array) -> Array:
    """Compute squared residual norms for each column of x."""
    rx = residual_proj @ x
    return np.sum(rx * rx, axis=0)


def _empty_basis(dim: int, dtype=np.float32) -> Array:
    return np.zeros((int(dim), 0), dtype=dtype)


def resolve_subspace_variant(variant: str) -> str:
    """
    Normalize configured DS variant names.

    `residual` is retained as a backwards-compatible alias for the original
    residual-stack construction. New experiments should prefer `canonical`.
    """
    key = str(variant or "legacy_residual_stack").strip().lower().replace("-", "_")
    aliases = {
        "residual": "legacy_residual_stack",
        "residual_stack": "legacy_residual_stack",
        "legacy": "legacy_residual_stack",
        "legacy_residual": "legacy_residual_stack",
        "legacy_residual_stack": "legacy_residual_stack",
        "eig": "eig",
        "projector_eig": "eig",
        "projector_eigen": "eig",
        "projector_eigen_ds": "eig",
        "canonical": "canonical",
        "principal": "canonical",
        "principal_vectors": "canonical",
        "principal_vector": "canonical",
    }
    if key not in aliases:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unknown subspace_variant={variant!r}. Valid normalized variants: {valid}")
    return aliases[key]


def legacy_residual_stack_difference_subspace(phi: Array, psi: Array) -> Array:
    """
    Original project construction: D = orth([R_psi * phi, R_phi * psi]).

    This is retained for reproducibility. It is not the paper-faithful default
    for new DS experiments because with equal-rank subspaces it can span up to
    2r directions and behave almost like a raw spectral-difference projection.
    """
    if phi.shape[0] != psi.shape[0]:
        raise ValueError("Bases must have same dimensionality.")
    r_phi = residual_projector(phi)
    r_psi = residual_projector(psi)
    stacked = np.concatenate([r_psi @ phi, r_phi @ psi], axis=1)
    return orthonormalize(stacked)


def difference_subspace(phi: Array, psi: Array) -> Array:
    """
    Backwards-compatible alias for the original residual-stack DS variant.

    Prefer `difference_subspace_canonical` or `build_difference_subspace(...,
    variant="canonical")` for paper-faithful first-order DS.
    """
    return legacy_residual_stack_difference_subspace(phi, psi)


def difference_subspace_eig(phi: Array, psi: Array, eps: float = 1e-6) -> Array:
    """
    Construct difference subspace using eigen-decomposition of the sum of projectors.

    Let G = P_phi + P_psi. Eigenvalues near 2 correspond to the common subspace,
    eigenvalues in (0,1) correspond to directions present in one subspace but not the other,
    and eigenvalues near 0 lie outside both. We select eigenvectors with eigenvalues in (eps, 1-eps).
    """
    if phi.shape[0] != psi.shape[0]:
        raise ValueError("Bases must have same dimensionality.")
    p_phi = phi @ phi.T
    p_psi = psi @ psi.T
    g = p_phi + p_psi
    eigvals, eigvecs = np.linalg.eigh(g)
    idx = np.where((eigvals > eps) & (eigvals < 1.0 - eps))[0]
    if idx.size == 0:
        return _empty_basis(phi.shape[0], dtype=phi.dtype)
    d_basis = eigvecs[:, idx]
    return orthonormalize(d_basis)


def difference_subspace_canonical(phi: Array, psi: Array, eps: float = 1e-6) -> Array:
    """
    Paper-faithful first-order DS from principal vectors.

    If Phi^T Psi = U Sigma V^T, the DS basis is formed as:
      D = (Phi U - Psi V) {2(I - Sigma)}^{-1/2}

    Near-shared directions with cos(theta) close to one are excluded. If the
    two subspaces are equal, an empty basis `(d, 0)` is returned.
    """
    if phi.ndim != 2 or psi.ndim != 2:
        raise ValueError("Expected 2D basis matrices.")
    if phi.shape[0] != psi.shape[0]:
        raise ValueError("Bases must have same dimensionality.")
    if phi.shape[1] == 0 or psi.shape[1] == 0:
        return _empty_basis(phi.shape[0], dtype=phi.dtype)

    a = phi
    b = psi
    if a.shape[1] > b.shape[1]:
        a, b = b, a

    u, s, vt = np.linalg.svd(a.T @ b, full_matrices=False)
    s = np.clip(s.astype(np.float64, copy=False), 0.0, 1.0)
    idx = np.where((1.0 - s) > float(eps))[0]
    if idx.size == 0:
        return _empty_basis(a.shape[0], dtype=a.dtype)

    au = a @ u[:, idx]
    bv = b @ vt.T[:, idx]
    scale = 1.0 / np.sqrt(2.0 * (1.0 - s[idx]))
    d = (au - bv) * scale[None, :]
    q, r = np.linalg.qr(d)
    diag = np.abs(np.diag(r)) if r.size else np.array([], dtype=np.float32)
    if diag.size:
        keep = diag > max(float(eps), 1e-8)
        q = q[:, keep] if np.any(keep) else q[:, :0]
    return q.astype(np.float32, copy=False)


def build_difference_subspace(phi: Array, psi: Array, variant: str = "canonical", eps: float = 1e-6) -> Array:
    """Dispatch DS construction by normalized variant name."""
    resolved = resolve_subspace_variant(variant)
    if resolved == "legacy_residual_stack":
        return legacy_residual_stack_difference_subspace(phi, psi)
    if resolved == "eig":
        return difference_subspace_eig(phi, psi, eps=eps)
    if resolved == "canonical":
        return difference_subspace_canonical(phi, psi, eps=eps)
    raise AssertionError(f"Unhandled resolved subspace variant: {resolved}")
