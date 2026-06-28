"""Core subspace / Difference-Subspace algebra (numpy, faithful to lab MagTool).

Conventions (match references/reference_code/MagTool-main/MagTool-main/magnitude.py and the lab
toolbox): a subspace is represented by an ORTHONORMAL BASIS with vectors as COLUMNS, shape
(ambient_dim, k). Data matrices are (ambient_dim, n_samples) with samples as columns.

All formulas re-derived from MagTool and cross-checked by tests/test_temporal_ds.py:
  magnitude(B1,B2) = 2 * sum_i (1 - cos theta_i)          (magnitude.py:223)
  identical subspaces -> 0 ; fully orthogonal k-dim -> 2k (max).
"""
from __future__ import annotations

import numpy as np

ETOL = 1e-6


# --------------------------------------------------------------------------------------
# Basis construction
# --------------------------------------------------------------------------------------
def pca_subspace(data: np.ndarray, dim: int, center: bool = True,
                 energy: float | None = None) -> np.ndarray:
    """Top-k left singular vectors of `data` (ambient_dim, n_samples).

    `dim` is the maximum rank. If `energy` in (0,1) is given, k is the smallest number of
    components whose cumulative squared-singular-value (variance) ratio reaches `energy`, capped at
    `dim` (lab-faithful: cvlPCA selects dim by cumsum(eigVal)/sum(eigVal) >= cRate). This avoids
    padding the subspace with arbitrary noise directions when the data is genuinely low-rank --
    the failure mode that makes DS noise-dominated.

    `center` subtracts the mean sample vector (mean over columns). Default True; set False for the
    lab autocorrelation-PCA convention (preserves a homogeneous region's dominant material direction).

    Returns an orthonormal basis (ambient_dim, k), k >= 1.
    """
    X = np.asarray(data, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"data must be 2D (ambient, samples), got {X.shape}")
    if center:
        X = X - X.mean(axis=1, keepdims=True)
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    kmax = min(dim, U.shape[1])
    if energy is not None and s.sum() > 0:
        ratio = np.cumsum(s ** 2) / np.sum(s ** 2)
        k = int(np.searchsorted(ratio, energy) + 1)
        k = max(1, min(k, kmax))
    else:
        k = kmax
    return np.ascontiguousarray(U[:, :k])


def orthonormalize(B: np.ndarray) -> np.ndarray:
    """QR-orthonormalize columns of B."""
    Q, _ = np.linalg.qr(np.asarray(B, dtype=np.float64))
    return Q


# --------------------------------------------------------------------------------------
# Canonical angles & magnitude
# --------------------------------------------------------------------------------------
def canonical_cosines(B1: np.ndarray, B2: np.ndarray) -> np.ndarray:
    """Cosines of canonical angles between subspaces span(B1), span(B2).

    s_i = singular values of B1^T B2, clipped to [0, 1] for numerical safety.
    Length = min(k1, k2).
    """
    G = np.asarray(B1).T @ np.asarray(B2)
    s = np.linalg.svd(G, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def canonical_angles(B1: np.ndarray, B2: np.ndarray) -> np.ndarray:
    return np.arccos(canonical_cosines(B1, B2))


def magnitude(B1: np.ndarray, B2: np.ndarray) -> float:
    """DS magnitude = 2 * sum_i (1 - cos theta_i), generalized to unequal-dim subspaces.

    MagTool calcMagnitude (magnitude.py:223) sums over min(k1,k2) canonical angles. We sum over
    max(k1,k2): the |k1-k2| extra dimensions of the larger subspace are counted as fully orthogonal
    (cos=0), so a genuinely NEW direction (e.g. a post-event material) contributes its full weight
    instead of being dropped. For equal dims this is identical to MagTool.

    Identical subspaces -> 0. Fully orthogonal -> 2*max(k1,k2).
    """
    s = canonical_cosines(B1, B2)              # min(k1,k2) cosines in [0,1]
    k = max(np.asarray(B1).shape[1], np.asarray(B2).shape[1])
    return float(2.0 * (k - np.sum(s)))


def similarity(B1: np.ndarray, B2: np.ndarray) -> float:
    """Sum of canonical cosines (MagTool calcSimilarity)."""
    return float(np.sum(canonical_cosines(B1, B2)))


# --------------------------------------------------------------------------------------
# Projector-sum subspaces (difference / karcher / sum / overlap), faithful to MagTool
# --------------------------------------------------------------------------------------
def _eig_projector_sum(B1: np.ndarray, B2: np.ndarray):
    """Eigendecomposition of P1+P2 = B1 B1^T + B2 B2^T, eigenvalues descending, snapped.

    Eigenvalues are 1 +/- cos(theta_i), plus exact 2 (shared dims) and 1 (single-cover dims).
    Mirrors MagTool calc*Subspace + adjustEig (magnitude.py:86-207).
    """
    B1 = np.asarray(B1, dtype=np.float64)
    B2 = np.asarray(B2, dtype=np.float64)
    G = B1 @ B1.T + B2 @ B2.T
    vals, vecs = np.linalg.eigh(G)
    vals = vals[::-1]
    vecs = vecs[:, ::-1]
    # adjustEig: snap ~2 -> 2 (overlap), <=etol -> 0 (inexpressible)
    snapped = vals.copy()
    snapped[(vals >= 2.0) | (vals > 2.0 - ETOL)] = 2.0
    snapped[(vals <= 0.0) | (vals < ETOL)] = 0.0
    return snapped, vecs


def difference_subspace(B1: np.ndarray, B2: np.ndarray) -> np.ndarray:
    """Difference Subspace basis: eigenvectors with eigenvalue in (0, 1).  (magnitude.py:158-186)

    Empty (shape (ambient, 0)) when the subspaces are identical.
    """
    vals, vecs = _eig_projector_sum(B1, B2)
    start = int(np.argmax(vals < 1.0)) if np.any(vals < 1.0) else len(vals)
    # index_end: first eigenvalue < 0 (after snapping, that's the first 0.0)
    below0 = np.where(vals < ETOL)[0]
    end = int(below0[0]) if below0.size else len(vals)
    return np.ascontiguousarray(vecs[:, start:end])


def karcher_subspace(B1: np.ndarray, B2: np.ndarray) -> np.ndarray:
    """Karcher/principal mean subspace: eigenvectors with eigenvalue >= 1.  (magnitude.py:134-156)"""
    vals, vecs = _eig_projector_sum(B1, B2)
    idx = int(np.argmax(vals < 1.0)) if np.any(vals < 1.0) else len(vals)
    return np.ascontiguousarray(vecs[:, :idx])


def sum_subspace(B1: np.ndarray, B2: np.ndarray) -> np.ndarray:
    """Span of the union: eigenvectors with eigenvalue > 0.  (magnitude.py:86-108)"""
    vals, vecs = _eig_projector_sum(B1, B2)
    idx = int(np.argmax(vals < ETOL)) if np.any(vals < ETOL) else len(vals)
    return np.ascontiguousarray(vecs[:, :idx])


# --------------------------------------------------------------------------------------
# First / second order decomposition (velocity / acceleration), faithful to MagTool
# --------------------------------------------------------------------------------------
def _sum_basis(B1: np.ndarray, B3: np.ndarray) -> np.ndarray:
    """W spanning span(B1) + span(B3) via eigenvectors of X X^T (X=[B1 B3]) with eigval>=etol."""
    X = np.concatenate((np.asarray(B1), np.asarray(B3)), axis=1)
    vals, W = np.linalg.eigh(X @ X.T)
    vals = vals[::-1]
    W = W[:, ::-1]
    idx = int(np.argmax(vals < ETOL)) if np.any(vals < ETOL) else len(vals)
    return W[:, :idx]


def first_order_decomposed(B1: np.ndarray, B2: np.ndarray, B3: np.ndarray):
    """[mag_along, mag_orth] for the FIRST-order geodesic decomposition. (MagTool calc1stMagDecomposed)

    mag_orth  = how much B2 leaves the plane span(B1,B3).
    mag_along = magnitude of B2's in-plane projection vs B1 (along-geodesic, vs the start).
    """
    W = _sum_basis(B1, B3)
    s = np.linalg.svd(W.T @ np.asarray(B2), compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    Q, _ = np.linalg.qr(W.T @ np.asarray(B2))
    B2_proj = W @ Q
    mag_orth = float(2.0 * (len(s) - np.sum(s)))
    mag_along = magnitude(B2_proj, B1)
    return [mag_along, mag_orth]


def second_order_decomposed(B1: np.ndarray, B2: np.ndarray, B3: np.ndarray):
    """[mag_along, mag_orth] for the SECOND-order decomposition. (MagTool calc2ndMagDecomposed)

    mag_along is measured against the Karcher mean M of (B1, B3): the deviation of the actual
    middle subspace B2 from the constant-velocity midpoint = acceleration along the geodesic.
    mag_orth is the off-geodesic (out-of-plane) component.
    """
    W = _sum_basis(B1, B3)
    M = karcher_subspace(B1, B3)
    s = np.linalg.svd(W.T @ np.asarray(B2), compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    Q, _ = np.linalg.qr(W.T @ np.asarray(B2))
    B2_proj = W @ Q
    mag_orth = float(2.0 * (len(s) - np.sum(s)))
    mag_along = magnitude(B2_proj, M)
    return [mag_along, mag_orth]


def second_order_magnitude(B1: np.ndarray, B2: np.ndarray, B3: np.ndarray) -> float:
    """Direct second-order DS magnitude per Fukui 2024: ||D(B2, Karcher(B1,B3))||.

    = how far the actual middle subspace deviates from the midpoint of its neighbors (acceleration).
    Constant-velocity geodesic motion -> B2 == M -> 0.
    """
    M = karcher_subspace(B1, B3)
    return magnitude(B2, M)
