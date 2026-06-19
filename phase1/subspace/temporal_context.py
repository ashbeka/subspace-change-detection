"""Bidirectional temporal-context subspaces for registered image sequences.

Source/provenance:
- Fukui and Maki (TPAMI 2015) supplies the canonical Difference Subspace
  construction used to compare two fitted subspaces.
- Dagobert et al. (IPOL 2022), Section 2 and the accompanying implementation,
  motivates comparing a backward set of registered images with a forward set
  at each temporal boundary.  Their detector uses non-negative least squares
  and an a-contrario NFA test; this module does not reproduce that detector.

Satellite adaptation:
For boundary ``t``, the ``V`` dates before the boundary and the ``V`` dates
after it form two temporal contexts.  In the per-band variant, each spectral
band produces matrices ``X^-_(t,c), X^+_(t,c) in R^(N x V)``, where rows are
fixed spatial coordinates and columns are dates.  PCA/SVD fits one backward
and one forward temporal subspace per band.  Canonical DS then measures their
orientation change.  A joint variant instead fits ``R^(N x (B V))`` matrices.

The implementation also exposes a symmetric orthogonal-projection novelty
map.  That control is a linear-subspace counterpart to temporal novelty
filtering, not the IPOL paper's NNLS cone or NFA decision rule.

Verification status:
Synthetic tests cover indexing, identical contexts, localized persistent
change, and radiometric invariance under centered L2 preprocessing.  Real-data
usefulness remains an experimental question and must be judged against labels
or an external detector without claiming equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from phase1.ds.pca_utils import difference_subspace_canonical_components
from phase1.subspace.geodesic import grassmann_geodesic_distance, subspace_magnitude

Array = np.ndarray


@dataclass(frozen=True)
class TemporalContextBasis:
  basis: Array
  singular_values: Array
  numerical_rank: int
  retained_rank: int


@dataclass(frozen=True)
class TemporalContextBoundary:
  boundary_index: int
  backward_indices: tuple[int, ...]
  forward_indices: tuple[int, ...]
  ds_map: Array
  projection_novelty_map: Array
  raw_boundary_difference_map: Array
  ds_magnitude: float
  geodesic_distance: float
  retained_factors: int
  factor_magnitudes: tuple[float, ...]
  factor_geodesic_distances: tuple[float, ...]


def temporal_context_indices(
  n_dates: int,
  boundary_index: int,
  context_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
  """Return endpoint-replicated backward/forward indices for one boundary.

  ``boundary_index`` is the first date in the forward context, so valid
  boundaries are ``1 .. n_dates-1``.  Replication keeps a fixed context size
  near sequence endpoints and matches the boundary convention used by the
  IPOL reference implementation.
  """
  count = int(n_dates)
  boundary = int(boundary_index)
  size = int(context_size)
  if count < 2:
    raise ValueError("At least two dates are required.")
  if not 1 <= boundary < count:
    raise ValueError(f"boundary_index must be in [1, {count - 1}], got {boundary}")
  if size < 1:
    raise ValueError("context_size must be positive.")
  backward = tuple(max(0, index) for index in range(boundary - size, boundary))
  forward = tuple(min(count - 1, index) for index in range(boundary, boundary + size))
  return backward, forward


def _prepare_columns(matrix: Array, preprocessing: str, eps: float) -> Array:
  key = str(preprocessing).strip().lower().replace("-", "_")
  valid = {"uncentered", "centered", "column_l2", "centered_column_l2"}
  if key not in valid:
    raise ValueError(f"Unknown preprocessing={preprocessing!r}; expected {sorted(valid)}")
  output = np.asarray(matrix, dtype=np.float64).copy()
  if key in {"centered", "centered_column_l2"}:
    output -= np.mean(output, axis=0, keepdims=True)
  if key in {"column_l2", "centered_column_l2"}:
    norms = np.linalg.norm(output, axis=0, keepdims=True)
    output /= np.maximum(norms, float(eps))
  return output


def fit_temporal_context_basis(
  matrix: Array,
  *,
  rank: int,
  preprocessing: str = "centered_column_l2",
  eps: float = 1e-10,
  rank_rtol: float = 1e-6,
) -> TemporalContextBasis:
  """Fit a numerically non-degenerate SVD basis to ``matrix in R^(N x K)``.

  Registered images make ``N`` much larger than ``K``.  We therefore solve
  the equivalent snapshot eigenproblem ``X.T X v = sigma^2 v`` and recover
  left singular vectors as ``u = X v / sigma``.  This changes computation,
  not the fitted subspace.
  """
  if matrix.ndim != 2 or min(matrix.shape) < 1:
    raise ValueError(f"Expected a non-empty 2D matrix, got {matrix.shape}")
  prepared = _prepare_columns(matrix, preprocessing, eps)
  if prepared.shape[0] >= prepared.shape[1]:
    gram = prepared.T @ prepared
    eigenvalues, right = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    right = right[:, order]
    singular_values = np.sqrt(eigenvalues)
  else:
    _, singular_values, right_t = np.linalg.svd(prepared, full_matrices=False)
    right = right_t.T
  if singular_values.size == 0 or float(singular_values[0]) <= float(eps):
    numerical_rank = 0
  else:
    tolerance = max(
      max(prepared.shape) * np.finfo(np.float64).eps * float(singular_values[0]),
      float(rank_rtol) * float(singular_values[0]),
    )
    numerical_rank = int(np.sum(singular_values > max(float(eps), tolerance)))
  requested = max(1, int(rank))
  retained = min(requested, numerical_rank)
  if retained:
    basis64 = prepared @ right[:, :retained]
    basis64 /= singular_values[:retained][None, :]
    # The snapshot recovery is already orthonormal analytically.  QR removes
    # only accumulated floating-point error without changing its span.
    basis, _ = np.linalg.qr(basis64)
    basis = basis[:, :retained].astype(np.float32, copy=False)
  else:
    basis = np.zeros((prepared.shape[0], 0), dtype=np.float32)
  return TemporalContextBasis(
    basis=basis,
    singular_values=singular_values[:retained].astype(np.float32, copy=False),
    numerical_rank=numerical_rank,
    retained_rank=retained,
  )


def _context_matrix(
  cubes: Sequence[Array],
  indices: Sequence[int],
  valid_mask: Array,
  *,
  band_index: int | None,
) -> Array:
  if band_index is None:
    columns = [cube[:, valid_mask].T for index in indices for cube in (cubes[index],)]
    # Each date contributes B complete band-image columns.
    return np.concatenate(columns, axis=1).astype(np.float64, copy=False)
  return np.stack(
    [cubes[index][band_index][valid_mask] for index in indices], axis=1
  ).astype(np.float64, copy=False)


def _matched_bases(
  backward_matrix: Array,
  forward_matrix: Array,
  *,
  rank: int,
  preprocessing: str,
) -> tuple[Array, Array]:
  backward = fit_temporal_context_basis(
    backward_matrix, rank=rank, preprocessing=preprocessing
  )
  forward = fit_temporal_context_basis(
    forward_matrix, rank=rank, preprocessing=preprocessing
  )
  shared_rank = min(backward.retained_rank, forward.retained_rank)
  return backward.basis[:, :shared_rank], forward.basis[:, :shared_rank]


def _spatial_ds_contribution(backward_basis: Array, forward_basis: Array) -> tuple[Array, float]:
  if backward_basis.shape[1] == 0 or forward_basis.shape[1] == 0:
    return np.zeros(backward_basis.shape[0], dtype=np.float64), 0.0
  components = difference_subspace_canonical_components(backward_basis, forward_basis)
  if components.basis.shape[1] == 0:
    return np.zeros(backward_basis.shape[0], dtype=np.float64), 0.0
  contribution = np.sum(
    components.basis.astype(np.float64) ** 2
    * components.squared_pair_magnitudes.astype(np.float64)[None, :],
    axis=1,
  )
  return contribution, float(np.sum(components.squared_pair_magnitudes))


def _projection_residual(basis: Array, vector: Array) -> Array:
  vector64 = np.asarray(vector, dtype=np.float64)
  if basis.shape[1] == 0:
    return vector64
  basis64 = basis.astype(np.float64, copy=False)
  return vector64 - basis64 @ (basis64.T @ vector64)


def bidirectional_temporal_context_boundary(
  cubes: Sequence[Array],
  valid_mask: Array,
  *,
  boundary_index: int,
  context_size: int,
  rank: int,
  factorization: str = "per_band",
  preprocessing: str = "centered_column_l2",
) -> TemporalContextBoundary:
  """Compare backward and forward temporal contexts at one date boundary."""
  if not cubes:
    raise ValueError("At least one cube is required.")
  shape = cubes[0].shape
  if len(shape) != 3 or valid_mask.shape != shape[1:]:
    raise ValueError("Cubes must be (bands, height, width) and match valid_mask.")
  if any(cube.shape != shape for cube in cubes):
    raise ValueError("All cubes must have the same band and spatial dimensions.")
  if not np.any(valid_mask):
    raise ValueError("valid_mask contains no valid spatial locations.")

  factor_key = str(factorization).strip().lower().replace("-", "_")
  if factor_key not in {"per_band", "joint"}:
    raise ValueError("factorization must be 'per_band' or 'joint'.")
  backward_indices, forward_indices = temporal_context_indices(
    len(cubes), boundary_index, context_size
  )
  n_valid = int(np.sum(valid_mask))
  factor_ds: list[Array] = []
  factor_novelty: list[Array] = []
  magnitudes: list[float] = []
  distances: list[float] = []

  bands: Sequence[int | None] = range(shape[0]) if factor_key == "per_band" else (None,)
  for band_index in bands:
    backward_matrix = _context_matrix(
      cubes, backward_indices, valid_mask, band_index=band_index
    )
    forward_matrix = _context_matrix(
      cubes, forward_indices, valid_mask, band_index=band_index
    )
    backward_basis, forward_basis = _matched_bases(
      backward_matrix,
      forward_matrix,
      rank=rank,
      preprocessing=preprocessing,
    )
    contribution, magnitude = _spatial_ds_contribution(backward_basis, forward_basis)
    factor_ds.append(contribution)
    magnitudes.append(magnitude)
    distances.append(
      grassmann_geodesic_distance(backward_basis, forward_basis)
      if backward_basis.shape[1] and forward_basis.shape[1]
      else 0.0
    )

    # Cross-boundary novelty uses the first post-date against the backward
    # basis and the last pre-date against the forward basis.  Prepare these
    # columns exactly as the fitted context columns before projection.
    if band_index is None:
      pre_vectors = cubes[boundary_index - 1][:, valid_mask].T
      post_vectors = cubes[boundary_index][:, valid_mask].T
      prepared_pre = _prepare_columns(pre_vectors, preprocessing, 1e-10)
      prepared_post = _prepare_columns(post_vectors, preprocessing, 1e-10)
      residuals = []
      for column in range(prepared_pre.shape[1]):
        residuals.append(_projection_residual(forward_basis, prepared_pre[:, column]) ** 2)
        residuals.append(_projection_residual(backward_basis, prepared_post[:, column]) ** 2)
      novelty = np.mean(np.stack(residuals, axis=0), axis=0)
    else:
      pre_vector = cubes[boundary_index - 1][band_index][valid_mask][:, None]
      post_vector = cubes[boundary_index][band_index][valid_mask][:, None]
      prepared_pre = _prepare_columns(pre_vector, preprocessing, 1e-10)[:, 0]
      prepared_post = _prepare_columns(post_vector, preprocessing, 1e-10)[:, 0]
      novelty = 0.5 * (
        _projection_residual(forward_basis, prepared_pre) ** 2
        + _projection_residual(backward_basis, prepared_post) ** 2
      )
    factor_novelty.append(novelty)

  ds_values = np.mean(np.stack(factor_ds, axis=0), axis=0)
  novelty_values = np.mean(np.stack(factor_novelty, axis=0), axis=0)
  raw_values = np.sqrt(np.mean(
    (cubes[boundary_index][:, valid_mask] - cubes[boundary_index - 1][:, valid_mask]) ** 2,
    axis=0,
  ))
  ds_map = np.zeros(valid_mask.shape, dtype=np.float32)
  novelty_map = np.zeros(valid_mask.shape, dtype=np.float32)
  raw_map = np.zeros(valid_mask.shape, dtype=np.float32)
  ds_map[valid_mask] = ds_values.astype(np.float32, copy=False)
  novelty_map[valid_mask] = novelty_values.astype(np.float32, copy=False)
  raw_map[valid_mask] = raw_values.astype(np.float32, copy=False)
  return TemporalContextBoundary(
    boundary_index=int(boundary_index),
    backward_indices=backward_indices,
    forward_indices=forward_indices,
    ds_map=ds_map,
    projection_novelty_map=novelty_map,
    raw_boundary_difference_map=raw_map,
    ds_magnitude=float(np.mean(magnitudes)) if magnitudes else 0.0,
    geodesic_distance=float(np.sqrt(np.sum(np.square(distances)))) if distances else 0.0,
    retained_factors=len(magnitudes),
    factor_magnitudes=tuple(float(value) for value in magnitudes),
    factor_geodesic_distances=tuple(float(value) for value in distances),
  )


def bidirectional_temporal_context_sequence(
  cubes: Sequence[Array],
  valid_mask: Array,
  *,
  context_size: int,
  rank: int,
  factorization: str = "per_band",
  preprocessing: str = "centered_column_l2",
) -> list[TemporalContextBoundary]:
  """Evaluate every temporal boundary in a registered image sequence."""
  return [
    bidirectional_temporal_context_boundary(
      cubes,
      valid_mask,
      boundary_index=boundary,
      context_size=context_size,
      rank=rank,
      factorization=factorization,
      preprocessing=preprocessing,
    )
    for boundary in range(1, len(cubes))
  ]
