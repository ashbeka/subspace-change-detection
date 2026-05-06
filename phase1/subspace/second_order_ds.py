"""
Second-order Difference Subspace (Fukui et al. 2024) utilities.

Key definition:
  D2(S1, S2, S3) = D(S2, M(S1, S3))

where M(.,.) is a PCS/Karcher-mean-style "mean subspace" computed from the
sum of projection matrices (Fukui & Maki 2015; Fukui et al. 2024).

This module operates on subspace bases (orthonormal columns), not on image pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from phase1.ds.pca_utils import difference_subspace_canonical as _difference_subspace_canonical
from phase1.subspace.geodesic import geodesic_projection, subspace_magnitude, sum_subspace

Array = np.ndarray


def principal_component_subspace(basis1: Array, basis2: Array, eig_tol: float = 1e-6) -> Array:
  """
  PCS / Karcher-subspace-style mean M between two subspaces.

  We form G = P1 + P2 where Pi = basis_i basis_iᵀ and take eigenvectors whose
  eigenvalues are > 1 (Fukui & Maki 2015).
  """
  if basis1.ndim != 2 or basis2.ndim != 2:
    raise ValueError("Expected 2D basis matrices.")
  if basis1.shape[0] != basis2.shape[0]:
    raise ValueError(f"Ambient dimension mismatch: {basis1.shape} vs {basis2.shape}")
  if basis1.shape[1] == 0 or basis2.shape[1] == 0:
    return np.zeros((basis1.shape[0], 0), dtype=np.float32)

  p1 = basis1 @ basis1.T
  p2 = basis2 @ basis2.T
  g = (p1 + p2).astype(np.float32, copy=False)
  eigvals, eigvecs = np.linalg.eigh(g)
  order = np.argsort(eigvals)[::-1]
  eigvals = eigvals[order].astype(np.float32)
  eigvecs = eigvecs[:, order]

  # Numerical cleanup similar in spirit to the senpai reference implementation.
  eigvals = np.where(eigvals >= 2.0 - float(eig_tol), 2.0, eigvals)
  eigvals = np.where(eigvals <= float(eig_tol), 0.0, eigvals)

  keep = eigvals > 1.0
  if not np.any(keep):
    return np.zeros((basis1.shape[0], 0), dtype=np.float32)
  return eigvecs[:, keep].astype(np.float32, copy=False)


def difference_subspace_canonical(basis1: Array, basis2: Array, eps: float = 1e-6) -> Array:
  """
  Canonical-angle DS basis via Fukui (Eq. D = (ΦU − ΨV){2(I−Σ)}^{-1/2}).

  Works for potentially different subspace dimensions by using k=min(d1,d2).
  Components with cos(θ)≈1 (intersection/shared directions) are excluded.
  """
  return _difference_subspace_canonical(basis1, basis2, eps=eps)


@dataclass(frozen=True)
class SecondOrderDSResult:
  mean_subspace: Array
  d2_basis: Array
  mag_total: float
  mag_along: Optional[float] = None
  mag_orth: Optional[float] = None


def second_order_difference_subspace(
  S1: Array,
  S2: Array,
  S3: Array,
  *,
  eps: float = 1e-6,
  eig_tol: float = 1e-6,
  decompose: bool = True,
) -> SecondOrderDSResult:
  """
  Compute D2(S1,S2,S3) and its magnitude.

  If decompose=True, also compute:
    - W = sum_subspace(S1,S3)
    - ω = geodesic_projection(W, S2)
    - mag_orth = Mag(S2, ω)
    - mag_along = Mag(ω, M(S1,S3))
  """
  m = principal_component_subspace(S1, S3, eig_tol=eig_tol)
  d2 = difference_subspace_canonical(S2, m, eps=eps)
  mag_total = subspace_magnitude(S2, m)

  if not decompose:
    return SecondOrderDSResult(mean_subspace=m, d2_basis=d2, mag_total=mag_total)

  w = sum_subspace(S1, S3, eps=1e-8)
  omega = geodesic_projection(w, S2)
  mag_orth = subspace_magnitude(S2, omega)
  mag_along = subspace_magnitude(omega, m)
  return SecondOrderDSResult(
    mean_subspace=m,
    d2_basis=d2,
    mag_total=mag_total,
    mag_along=mag_along,
    mag_orth=mag_orth,
  )
