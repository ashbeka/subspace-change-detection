"""Order-aware temporal subspaces for multispectral image sequences.

Source/provenance:
- Kanai et al. (2023), *Time-series Anomaly Detection based on
  Difference Subspace between Signal Subspaces*, constructs an SSA signal
  subspace from the leading eigenvectors of a sliding-window trajectory
  matrix.  Their Eq. (1) is for a scalar time series.
- The block trajectory matrix below is the standard multivariate extension:
  each scalar lag is replaced by one flattened spatial-spectral observation.
  This satellite adaptation is project-designed and must not be described as
  an equation copied directly from Kanai et al.

For observations ``x_t in R^D`` and lag ``L``, the trajectory matrix is

``H_L = [h_1, ..., h_(T-L+1)] in R^((D*L) x (T-L+1))``

where ``h_j = [x_j; x_(j+1); ...; x_(j+L-1)]``.  Unlike an unordered matrix
``[x_1, ..., x_T]``, this representation normally changes when dates are
permuted because temporal neighborhoods are part of each column.

The module also exposes a first-difference representation.  It is a simpler
order-aware control, not an SSA method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class TemporalRepresentationSubspace:
  basis: Array
  singular_values: Array
  explained_energy_ratio: Array
  representation_matrix: Array
  representation: str
  preprocessing: str
  lag: int
  n_dates: int
  n_columns: int
  n_features: int
  rank: int


def flattened_observation_matrix(cubes: Sequence[Array], valid_mask: Array) -> Array:
  """Return ``[x_1, ..., x_T] in R^(D x T)`` with fixed feature ordering."""
  if len(cubes) < 2:
    raise ValueError("At least two temporal observations are required.")
  shape = cubes[0].shape
  if len(shape) != 3:
    raise ValueError(f"Expected cubes shaped (bands, height, width), got {shape}")
  if valid_mask.shape != shape[1:]:
    raise ValueError(f"Mask shape {valid_mask.shape} does not match cube shape {shape}")
  if not np.any(valid_mask):
    raise ValueError("valid_mask contains no spatial locations.")
  if any(cube.shape != shape for cube in cubes):
    raise ValueError("All temporal observations must have the same cube shape.")
  matrix = np.stack(
    [cube[:, valid_mask].reshape(-1).astype(np.float64, copy=False) for cube in cubes],
    axis=1,
  )
  if not np.all(np.isfinite(matrix)):
    raise ValueError("Temporal observations contain non-finite values inside valid_mask.")
  return matrix


def temporal_representation_matrix(
  cubes: Sequence[Array],
  valid_mask: Array,
  *,
  representation: str,
  lag: int = 3,
  preprocessing: str = "feature_centered",
  eps: float = 1e-12,
) -> Array:
  """Build an unordered, first-difference, or block-Hankel matrix.

  Preprocessing is applied to the completed representation matrix:

  - ``uncentered``: no centering or normalization;
  - ``feature_centered``: subtract each row mean across columns;
  - ``column_l2``: normalize each representation column;
  - ``feature_centered_column_l2``: center then normalize.
  """
  observations = flattened_observation_matrix(cubes, valid_mask)
  key = str(representation).strip().lower().replace("-", "_")
  if key == "unordered":
    matrix = observations
    effective_lag = 1
  elif key in {"difference", "first_difference"}:
    matrix = np.diff(observations, axis=1)
    effective_lag = 1
  elif key in {"trajectory", "hankel", "ssa"}:
    effective_lag = int(lag)
    if effective_lag < 2:
      raise ValueError("A trajectory representation requires lag >= 2.")
    if effective_lag >= observations.shape[1]:
      raise ValueError(
        f"lag={effective_lag} must be smaller than n_dates={observations.shape[1]}."
      )
    columns = []
    for start in range(observations.shape[1] - effective_lag + 1):
      columns.append(observations[:, start : start + effective_lag].reshape(-1, order="F"))
    matrix = np.stack(columns, axis=1)
  else:
    raise ValueError(
      f"Unknown representation={representation!r}; expected unordered, difference, or trajectory."
    )

  prep = str(preprocessing).strip().lower().replace("-", "_")
  prep = {
    "observation_l2": "column_l2",
    "feature_centered_observation_l2": "feature_centered_column_l2",
  }.get(prep, prep)
  valid_preprocessing = {
    "uncentered",
    "feature_centered",
    "column_l2",
    "feature_centered_column_l2",
  }
  if prep not in valid_preprocessing:
    raise ValueError(f"Unknown preprocessing={preprocessing!r}; expected {sorted(valid_preprocessing)}")
  if prep in {"feature_centered", "feature_centered_column_l2"}:
    matrix = matrix - np.mean(matrix, axis=1, keepdims=True)
  if prep in {"column_l2", "feature_centered_column_l2"}:
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    matrix = matrix / np.maximum(norms, float(eps))
  return matrix


def build_temporal_representation_subspace(
  cubes: Sequence[Array],
  valid_mask: Array,
  *,
  rank: int,
  representation: str,
  lag: int = 3,
  preprocessing: str = "feature_centered",
  singular_tol: float = 1e-6,
) -> TemporalRepresentationSubspace:
  """Fit a PCA basis to one temporal representation matrix."""
  matrix = temporal_representation_matrix(
    cubes,
    valid_mask,
    representation=representation,
    lag=lag,
    preprocessing=preprocessing,
  )
  left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
  if singular_values.size:
    threshold = float(singular_tol) * max(float(singular_values[0]), 1.0)
    numerical_rank = int(np.sum(singular_values > threshold))
  else:
    numerical_rank = 0
  effective_rank = min(max(int(rank), 0), numerical_rank)
  basis = (
    left[:, :effective_rank].astype(np.float32, copy=False)
    if effective_rank
    else np.zeros((matrix.shape[0], 0), dtype=np.float32)
  )
  energy = singular_values * singular_values
  total = float(np.sum(energy))
  explained = (
    energy[:effective_rank] / total
    if total > 0.0
    else np.zeros(effective_rank, dtype=np.float64)
  )
  key = str(representation).strip().lower().replace("-", "_")
  return TemporalRepresentationSubspace(
    basis=basis,
    singular_values=singular_values.astype(np.float32, copy=False),
    explained_energy_ratio=explained.astype(np.float32, copy=False),
    representation_matrix=matrix.astype(np.float32, copy=False),
    representation=key,
    preprocessing=str(preprocessing).strip().lower().replace("-", "_"),
    lag=int(lag) if key in {"trajectory", "hankel", "ssa"} else 1,
    n_dates=len(cubes),
    n_columns=int(matrix.shape[1]),
    n_features=int(matrix.shape[0]),
    rank=effective_rank,
  )


def representation_energy_change(
  first: TemporalRepresentationSubspace,
  second: TemporalRepresentationSubspace,
) -> float:
  """Absolute log change of representation-matrix Frobenius energy."""
  first_energy = float(np.linalg.norm(first.representation_matrix))
  second_energy = float(np.linalg.norm(second.representation_matrix))
  return abs(float(np.log(max(second_energy, 1e-12) / max(first_energy, 1e-12))))


def representation_spectrum_change(
  first: TemporalRepresentationSubspace,
  second: TemporalRepresentationSubspace,
) -> float:
  """L2 change between unit-normalized singular-value spectra."""
  size = min(first.singular_values.size, second.singular_values.size)
  if size == 0:
    return 0.0
  left = first.singular_values[:size].astype(np.float64)
  right = second.singular_values[:size].astype(np.float64)
  left /= max(float(np.linalg.norm(left)), 1e-12)
  right /= max(float(np.linalg.norm(right)), 1e-12)
  return float(np.linalg.norm(left - right))


def representation_covariance_change(
  first: TemporalRepresentationSubspace,
  second: TemporalRepresentationSubspace,
) -> float:
  """Compare trace-normalized second-moment operators in Frobenius norm.

  For representation matrix ``X``, the compared operator is
  ``C_X = X X^T / trace(X X^T)``. Unlike a Grassmann/DS distance, it retains
  both principal-direction orientation and relative singular-value energy.
  Small Gram matrices are used so the ambient covariance is never formed.

  This is a covariance-operator diagnostic, not part of Fukui's DS theory.
  """
  left = first.representation_matrix.astype(np.float64, copy=False)
  right = second.representation_matrix.astype(np.float64, copy=False)
  if left.shape[0] != right.shape[0]:
    raise ValueError(f"Ambient dimension mismatch: {left.shape} vs {right.shape}")
  left_energy = float(np.sum(left * left))
  right_energy = float(np.sum(right * right))
  if left_energy <= 0.0 and right_energy <= 0.0:
    return 0.0
  if left_energy <= 0.0 or right_energy <= 0.0:
    return 1.0
  left_gram = left.T @ left
  right_gram = right.T @ right
  cross_gram = left.T @ right
  distance_squared = (
    float(np.sum(left_gram * left_gram)) / (left_energy * left_energy)
    + float(np.sum(right_gram * right_gram)) / (right_energy * right_energy)
    - 2.0 * float(np.sum(cross_gram * cross_gram)) / (left_energy * right_energy)
  )
  return float(np.sqrt(max(distance_squared, 0.0)))
