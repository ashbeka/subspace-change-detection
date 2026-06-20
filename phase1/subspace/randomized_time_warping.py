"""Randomized Time Warping (RTW) hypo-subspace construction.

Source/provenance
-----------------
Suryanto, Xue, and Fukui, *Randomized Time Warping for Motion
Recognition*, Image and Vision Computing 54 (2016),
https://doi.org/10.1016/j.imavis.2016.07.003.

Hiraoka et al., *Attention Mechanism in Randomized Time Warping*
(2025), arXiv:2508.16366, Sections 2.1--2.3.

The bundled MATLAB reference ``references/reference_code/Subspace
Toolbox/RTW/TEfeatures.m`` independently confirms the core sampling rule:
select ``R`` sequence elements without replacement, sort their indices to
retain temporal order, concatenate them, and repeat ``L`` times.

Satellite adaptation
--------------------
One sequence element is a multispectral feature vector for one date. The
paper's Time Elastic (TE) features and uncentered PCA/eigen-subspace are kept.
Any radiometric standardization is performed by the experiment runner and is
reported as an adaptation, not RTW theory.

Verification status
-------------------
Formula/shape tests cover ordered sampling, exact feature construction,
identical-subspace similarity, basis-rotation invariance, and invalid ranks.
The method name does not prove timing invariance; that is the experimental
hypothesis tested by the MultiSenGE RTW gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class RTWHypoSubspace:
  """One RTW hypo-subspace fitted from sampled Time Elastic features."""

  basis: Array
  singular_values: Array
  explained_energy_ratio: Array
  feature_matrix: Array
  sample_indices: Array
  subsequence_length: int
  n_samples: int
  rank: int
  n_steps: int
  n_features: int


def ordered_subsequence_indices(
  n_steps: int,
  subsequence_length: int,
  n_samples: int,
  rng: np.random.Generator,
) -> Array:
  """Return ``(n_samples, subsequence_length)`` sorted sample indices."""
  n_steps = int(n_steps)
  subsequence_length = int(subsequence_length)
  n_samples = int(n_samples)
  if n_steps < 2:
    raise ValueError("RTW requires at least two sequence elements.")
  if subsequence_length < 2:
    raise ValueError("subsequence_length must be at least 2.")
  if subsequence_length > n_steps:
    raise ValueError(
      f"subsequence_length={subsequence_length} exceeds n_steps={n_steps}."
    )
  if n_samples < 1:
    raise ValueError("n_samples must be positive.")
  output = np.empty((n_samples, subsequence_length), dtype=np.int32)
  for sample in range(n_samples):
    output[sample] = np.sort(
      rng.choice(n_steps, size=subsequence_length, replace=False)
    )
  return output


def time_elastic_feature_matrix(
  sequence: Array,
  sample_indices: Array,
) -> Array:
  """Construct ``F in R^(dR x L)`` from an ordered feature sequence.

  ``sequence`` is ``(N, d)`` and every row of ``sample_indices`` contains one
  increasing ``R``-tuple. Each output column concatenates the corresponding
  sequence rows in temporal order.
  """
  values = np.asarray(sequence, dtype=np.float64)
  indices = np.asarray(sample_indices, dtype=np.int64)
  if values.ndim != 2:
    raise ValueError(f"Expected sequence shaped (steps, features), got {values.shape}.")
  if indices.ndim != 2:
    raise ValueError(f"Expected sample_indices shaped (samples, R), got {indices.shape}.")
  if values.shape[0] < 2 or values.shape[1] < 1:
    raise ValueError("Sequence must contain at least two steps and one feature.")
  if not np.all(np.isfinite(values)):
    raise ValueError("Sequence contains non-finite values.")
  if indices.size == 0:
    raise ValueError("sample_indices cannot be empty.")
  if np.min(indices) < 0 or np.max(indices) >= values.shape[0]:
    raise ValueError("sample_indices contain an out-of-range index.")
  if np.any(np.diff(indices, axis=1) <= 0):
    raise ValueError("Every sampled tuple must be strictly increasing.")
  selected = values[indices]
  return selected.reshape(indices.shape[0], -1).T


def fit_rtw_hypo_subspace(
  sequence: Array,
  *,
  subsequence_length: int,
  n_samples: int,
  rank: int,
  rng: np.random.Generator,
  sample_indices: Array | None = None,
  singular_tol: float = 1e-9,
) -> RTWHypoSubspace:
  """Fit the uncentered RTW PCA hypo-subspace described by Hiraoka et al."""
  values = np.asarray(sequence, dtype=np.float64)
  if values.ndim != 2:
    raise ValueError(f"Expected sequence shaped (steps, features), got {values.shape}.")
  if rank < 1:
    raise ValueError("rank must be positive.")
  if sample_indices is None:
    indices = ordered_subsequence_indices(
      values.shape[0], subsequence_length, n_samples, rng
    )
  else:
    indices = np.asarray(sample_indices, dtype=np.int32)
    if indices.shape != (int(n_samples), int(subsequence_length)):
      raise ValueError(
        "sample_indices shape does not match n_samples and subsequence_length."
      )
  features = time_elastic_feature_matrix(values, indices)
  left, singular_values, _ = np.linalg.svd(features, full_matrices=False)
  threshold = float(singular_tol) * max(float(singular_values[0]), 1.0)
  numerical_rank = int(np.sum(singular_values > threshold))
  effective_rank = min(int(rank), numerical_rank)
  if effective_rank < 1:
    raise ValueError("Time Elastic feature matrix has zero numerical rank.")
  energy = singular_values * singular_values
  explained = energy[:effective_rank] / max(float(np.sum(energy)), 1e-15)
  return RTWHypoSubspace(
    basis=left[:, :effective_rank],
    singular_values=singular_values,
    explained_energy_ratio=explained,
    feature_matrix=features,
    sample_indices=indices,
    subsequence_length=int(subsequence_length),
    n_samples=int(n_samples),
    rank=effective_rank,
    n_steps=int(values.shape[0]),
    n_features=int(values.shape[1]),
  )


def canonical_correlations(first_basis: Array, second_basis: Array) -> Array:
  """Return canonical correlations between two orthonormal basis matrices."""
  first = np.asarray(first_basis, dtype=np.float64)
  second = np.asarray(second_basis, dtype=np.float64)
  if first.ndim != 2 or second.ndim != 2:
    raise ValueError("Basis matrices must be two-dimensional.")
  if first.shape[0] != second.shape[0]:
    raise ValueError(f"Ambient dimensions differ: {first.shape} versus {second.shape}.")
  if first.shape[1] < 1 or second.shape[1] < 1:
    return np.zeros(0, dtype=np.float64)
  values = np.linalg.svd(first.T @ second, compute_uv=False)
  return np.clip(values, 0.0, 1.0)


def rtw_similarity(
  first_basis: Array,
  second_basis: Array,
  *,
  n_components: int | None = None,
) -> float:
  """Return the MSM/RTW mean squared canonical correlation."""
  correlations = canonical_correlations(first_basis, second_basis)
  if correlations.size == 0:
    return 0.0
  count = correlations.size if n_components is None else min(int(n_components), correlations.size)
  if count < 1:
    raise ValueError("n_components must be positive when provided.")
  return float(np.mean(correlations[:count] ** 2))


def rtw_dissimilarity(
  first_basis: Array,
  second_basis: Array,
  *,
  n_components: int | None = None,
) -> float:
  """Return ``1 - RTW similarity`` so larger values indicate more change."""
  return float(1.0 - rtw_similarity(
    first_basis,
    second_basis,
    n_components=n_components,
  ))


def compare_rtw_sequences(
  first_sequence: Array,
  second_sequence: Array,
  *,
  subsequence_length: int,
  n_samples: int,
  rank: int,
  seed: int,
  replicates: int = 1,
) -> dict[str, float | list[float]]:
  """Average independently sampled RTW comparisons across deterministic seeds."""
  if replicates < 1:
    raise ValueError("replicates must be positive.")
  scores: list[float] = []
  similarities: list[float] = []
  energies: list[float] = []
  for replicate in range(int(replicates)):
    seed_sequence = np.random.SeedSequence([int(seed), int(replicate)])
    first_seed, second_seed = seed_sequence.spawn(2)
    first = fit_rtw_hypo_subspace(
      first_sequence,
      subsequence_length=subsequence_length,
      n_samples=n_samples,
      rank=rank,
      rng=np.random.default_rng(first_seed),
    )
    second = fit_rtw_hypo_subspace(
      second_sequence,
      subsequence_length=subsequence_length,
      n_samples=n_samples,
      rank=rank,
      rng=np.random.default_rng(second_seed),
    )
    similarity = rtw_similarity(first.basis, second.basis)
    similarities.append(similarity)
    scores.append(1.0 - similarity)
    energies.append(
      abs(float(np.log(
        max(np.linalg.norm(second.feature_matrix), 1e-15)
        / max(np.linalg.norm(first.feature_matrix), 1e-15)
      )))
    )
  return {
    "dissimilarity": float(np.mean(scores)),
    "dissimilarity_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
    "similarity": float(np.mean(similarities)),
    "te_energy_change": float(np.mean(energies)),
    "replicate_scores": scores,
  }
