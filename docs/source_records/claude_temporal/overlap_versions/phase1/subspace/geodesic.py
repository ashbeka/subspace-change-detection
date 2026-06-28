"""
Grassmann / geodesic helpers for subspace time-series analysis.

We treat each (orthonormal) basis matrix S ∈ R^{n×k} as a point on Gr(k, n).
Principal angles are computed via SVD of S1ᵀ S2.

References:
- Fukui & Maki (2015) Difference Subspace and Its Generalization
- Fukui et al. (2024) Second-order difference subspace (geodesic projection ω)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


def orthonormalize(basis: Array) -> Array:
  if basis.ndim != 2:
    raise ValueError(f"Expected 2D basis matrix, got shape={basis.shape}")
  if basis.size == 0:
    return basis.astype(np.float32, copy=False)
  q, r = np.linalg.qr(basis)
  diag = np.abs(np.diag(r)) if r.size else np.array([], dtype=np.float32)
  if diag.size:
    keep = diag > 1e-8
    if np.any(keep):
      q = q[:, keep]
    else:
      q = q[:, :0]
  return q.astype(np.float32, copy=False)


def principal_cosines(basis1: Array, basis2: Array) -> Array:
  """
  Return cosines of principal angles between two subspaces as singular values of basis1ᵀ basis2.

  Inputs:
    basis1: (n, k1) orthonormal columns
    basis2: (n, k2) orthonormal columns

  Output:
    s: (min(k1,k2),) in [0,1]
  """
  if basis1.ndim != 2 or basis2.ndim != 2:
    raise ValueError("Expected 2D basis matrices.")
  if basis1.shape[0] != basis2.shape[0]:
    raise ValueError(f"Ambient dimension mismatch: {basis1.shape} vs {basis2.shape}")
  if basis1.shape[1] == 0 or basis2.shape[1] == 0:
    return np.zeros((0,), dtype=np.float32)
  m = basis1.T @ basis2
  _, s, _ = np.linalg.svd(m, full_matrices=False)
  s = np.clip(s.astype(np.float32), 0.0, 1.0)
  return s


def principal_angles(basis1: Array, basis2: Array) -> Array:
  """
  Principal angles θ ∈ [0, π/2] between two subspaces.
  """
  s = principal_cosines(basis1, basis2)
  if s.size == 0:
    return s
  return np.arccos(s).astype(np.float32)


def grassmann_geodesic_distance(basis1: Array, basis2: Array) -> float:
  """
  Geodesic distance on Gr(k, n) using principal angles: ρ = ||θ||₂.
  """
  theta = principal_angles(basis1, basis2)
  if theta.size == 0:
    return 0.0
  return float(np.sqrt(np.sum(theta * theta)))


def subspace_magnitude(basis1: Array, basis2: Array) -> float:
  """
  Fukui-style magnitude from singular values s = cos(θ):
    Mag = 2 * (k - Σ s_i), where k = min(dim(S1), dim(S2)).
  """
  s = principal_cosines(basis1, basis2)
  k = int(s.shape[0])
  return float(2.0 * (k - float(np.sum(s))))


def sum_subspace(basis1: Array, basis2: Array, eps: float = 1e-8) -> Array:
  """
  Orthonormal basis W for the sum subspace span([basis1, basis2]).
  """
  if basis1.shape[0] != basis2.shape[0]:
    raise ValueError(f"Ambient dimension mismatch: {basis1.shape} vs {basis2.shape}")
  if basis1.shape[1] == 0 and basis2.shape[1] == 0:
    return np.zeros((basis1.shape[0], 0), dtype=np.float32)
  x = np.concatenate([basis1, basis2], axis=1).astype(np.float32, copy=False)
  u, s, _ = np.linalg.svd(x, full_matrices=False)
  keep = s > float(eps)
  if not np.any(keep):
    return np.zeros((x.shape[0], 0), dtype=np.float32)
  return u[:, keep].astype(np.float32, copy=False)


def geodesic_projection(W: Array, S: Array) -> Array:
  """
  Geodesic projection ω(S) of a subspace S onto a (higher-dimensional) subspace W.

  Lemma (Fukui et al. 2024): if Wᵀ S = U Σ Vᵀ, then ω(S) = W U.
  """
  if W.ndim != 2 or S.ndim != 2:
    raise ValueError("Expected 2D basis matrices.")
  if W.shape[0] != S.shape[0]:
    raise ValueError(f"Ambient dimension mismatch: {W.shape} vs {S.shape}")
  if W.shape[1] == 0 or S.shape[1] == 0:
    return np.zeros((W.shape[0], 0), dtype=np.float32)
  m = W.T @ S
  U, _, _ = np.linalg.svd(m, full_matrices=False)
  k = min(W.shape[1], S.shape[1])
  U = U[:, :k]
  return (W @ U).astype(np.float32, copy=False)


@dataclass(frozen=True)
class SecondOrderDecomposition:
  mag_total: float
  mag_along: float
  mag_orth: float

