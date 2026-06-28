"""Seasonal subspaces built from repeated multispectral observations.

Source/provenance:
- Kanai et al. (2023), *Time-series Anomaly Detection based on Difference
  Subspace between Signal Subspaces*, motivates comparing subspaces fitted to
  temporal signal sets.
- Fukui et al. (2024), *Second-order Difference Subspace*, supplies the
  first/second DS magnitude and geodesic-decomposition objects used after the
  seasonal subspaces are built.
- The exact satellite construction below is a project adaptation, not a formula
  copied from either paper.

For one fixed patch/field and one season/year, each aligned multispectral date
cube is one related observation.  With ``B`` bands, ``N`` common valid spatial
locations, and ``M`` date composites, the matrix is
``X_y in R^((B*N) x M)``.  Its leading left singular vectors form one seasonal
observation subspace.  Unlike the per-date band-image construction, the sample
set here is the temporal sequence itself.

The construction intentionally discards order within a season.  It should be
compared with Hankel/SSA or another order-aware embedding if this simpler
hypothesis survives initial tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class SeasonalObservationSubspace:
  basis: Array
  singular_values: Array
  explained_energy_ratio: Array
  observation_matrix: Array
  n_observations: int
  n_features: int
  rank: int
  preprocessing: str


def seasonal_observation_matrix(
  cubes: Sequence[Array],
  valid_mask: Array,
  *,
  preprocessing: str = "feature_centered",
  eps: float = 1e-12,
) -> Array:
  """Stack repeated date cubes as columns of ``R^((B*N) x M)``.

  Supported preprocessing:

  - ``uncentered`` keeps raw spatial-spectral values;
  - ``feature_centered`` removes each spatial-band feature's seasonal mean;
  - ``observation_l2`` normalizes each date column;
  - ``feature_centered_observation_l2`` centers then normalizes.

  All dates must use the same band order, spatial grid, and validity mask.
  """
  if len(cubes) < 2:
    raise ValueError("At least two repeated observations are required.")
  first_shape = cubes[0].shape
  if len(first_shape) != 3:
    raise ValueError(f"Expected cubes shaped (bands, height, width), got {first_shape}")
  if valid_mask.shape != first_shape[1:]:
    raise ValueError(f"Mask shape {valid_mask.shape} does not match cube shape {first_shape}")
  if not np.any(valid_mask):
    raise ValueError("valid_mask contains no spatial locations.")
  if any(cube.shape != first_shape for cube in cubes):
    raise ValueError("All repeated observations must have the same cube shape.")

  key = str(preprocessing).strip().lower().replace("-", "_")
  valid = {
    "uncentered",
    "feature_centered",
    "observation_l2",
    "feature_centered_observation_l2",
  }
  if key not in valid:
    raise ValueError(f"Unknown preprocessing={preprocessing!r}; expected {sorted(valid)}")

  columns = [cube[:, valid_mask].reshape(-1).astype(np.float64, copy=False) for cube in cubes]
  matrix = np.stack(columns, axis=1)
  if not np.all(np.isfinite(matrix)):
    raise ValueError("Seasonal observations contain non-finite values inside valid_mask.")
  if key in {"feature_centered", "feature_centered_observation_l2"}:
    matrix = matrix - np.mean(matrix, axis=1, keepdims=True)
  if key in {"observation_l2", "feature_centered_observation_l2"}:
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    matrix = matrix / np.maximum(norms, float(eps))
  return matrix


def build_seasonal_observation_subspace(
  cubes: Sequence[Array],
  valid_mask: Array,
  *,
  rank: int,
  preprocessing: str = "feature_centered",
  singular_tol: float = 1e-6,
) -> SeasonalObservationSubspace:
  """Fit one seasonal observation subspace from repeated date cubes."""
  matrix = seasonal_observation_matrix(
    cubes,
    valid_mask,
    preprocessing=preprocessing,
  )
  left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
  if singular_values.size == 0:
    numerical_rank = 0
  else:
    threshold = float(singular_tol) * max(float(singular_values[0]), 1.0)
    numerical_rank = int(np.sum(singular_values > threshold))
  effective_rank = min(max(int(rank), 0), numerical_rank)
  if effective_rank == 0:
    basis = np.zeros((matrix.shape[0], 0), dtype=np.float32)
  else:
    basis = left[:, :effective_rank].astype(np.float32, copy=False)
  energy = singular_values * singular_values
  total = float(np.sum(energy))
  explained = (
    energy[:effective_rank] / total
    if total > 0.0
    else np.zeros(effective_rank, dtype=np.float64)
  )
  return SeasonalObservationSubspace(
    basis=basis,
    singular_values=singular_values.astype(np.float32, copy=False),
    explained_energy_ratio=explained.astype(np.float32, copy=False),
    observation_matrix=matrix.astype(np.float32, copy=False),
    n_observations=int(matrix.shape[1]),
    n_features=int(matrix.shape[0]),
    rank=effective_rank,
    preprocessing=str(preprocessing).strip().lower().replace("-", "_"),
  )


def singular_energy_change(first: SeasonalObservationSubspace, second: SeasonalObservationSubspace) -> float:
  """Absolute log change in centered seasonal matrix energy.

  This is a magnitude control, not a DS score.  Subspace spans are intentionally
  insensitive to some amplitude changes; the energy term makes that lost
  information explicit rather than smuggling it into a geometric claim.
  """
  first_energy = float(np.linalg.norm(first.observation_matrix))
  second_energy = float(np.linalg.norm(second.observation_matrix))
  return abs(float(np.log(max(second_energy, 1e-12) / max(first_energy, 1e-12))))


def normalized_singular_spectrum_change(
  first: SeasonalObservationSubspace,
  second: SeasonalObservationSubspace,
) -> float:
  """L2 change between normalized singular-value spectra."""
  size = min(first.singular_values.size, second.singular_values.size)
  if size == 0:
    return 0.0
  left = first.singular_values[:size].astype(np.float64)
  right = second.singular_values[:size].astype(np.float64)
  left /= max(float(np.linalg.norm(left)), 1e-12)
  right /= max(float(np.linalg.norm(right)), 1e-12)
  return float(np.linalg.norm(left - right))
