"""Strong temporal-sequence controls for the RTW invariance experiment.

The module intentionally mixes non-geometric and geometric nulls. They test
whether RTW contributes more than ordinary phase-invariant Fourier/harmonic
descriptors, dynamic time warping, or a non-random temporal subspace.

Time-weighted DTW follows Maus et al. (2016),
https://doi.org/10.1109/JSTARS.2016.2517118. The local cost is the Euclidean
feature distance plus ``1/(1+exp(-alpha*(elapsed-beta)))``. The implementation
was cross-checked against the public ``vwmaus/twdtw`` package source. This is a
pairwise distance adaptation, not the package's full subsequence classifier.
"""

from __future__ import annotations

import numpy as np

from phase1.subspace.randomized_time_warping import rtw_dissimilarity

Array = np.ndarray


def _validate(sequence: Array) -> Array:
  values = np.asarray(sequence, dtype=np.float64)
  if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
    raise ValueError(f"Expected (steps, features) with at least two steps, got {values.shape}.")
  if not np.all(np.isfinite(values)):
    raise ValueError("Sequence contains non-finite values.")
  return values


def preprocess_pair(first: Array, second: Array, mode: str) -> tuple[Array, Array]:
  """Apply a reported radiometric preprocessing mode to a sequence pair."""
  left = _validate(first).copy()
  right = _validate(second).copy()
  if left.shape[1] != right.shape[1]:
    raise ValueError("Sequences must have the same number of features.")
  key = str(mode).strip().lower().replace("-", "_")
  if key == "raw":
    return left, right
  if key == "reference_zscore":
    mean = np.mean(left, axis=0, keepdims=True)
    scale = np.std(left, axis=0, keepdims=True)
    scale = np.maximum(scale, 1e-6)
    return (left - mean) / scale, (right - mean) / scale
  if key == "per_sequence_zscore":
    def standardize(values: Array) -> Array:
      mean = np.mean(values, axis=0, keepdims=True)
      scale = np.maximum(np.std(values, axis=0, keepdims=True), 1e-6)
      return (values - mean) / scale
    return standardize(left), standardize(right)
  raise ValueError(
    f"Unknown preprocessing={mode!r}; expected raw, reference_zscore, or per_sequence_zscore."
  )


def resample_sequence(sequence: Array, times: Array, grid: Array) -> Array:
  """Linearly interpolate every feature onto a common one-dimensional grid."""
  values = _validate(sequence)
  source_times = np.asarray(times, dtype=np.float64)
  target_grid = np.asarray(grid, dtype=np.float64)
  if source_times.ndim != 1 or source_times.size != values.shape[0]:
    raise ValueError("times must be one-dimensional and match sequence length.")
  if np.any(np.diff(source_times) <= 0):
    raise ValueError("times must be strictly increasing.")
  return np.stack(
    [np.interp(target_grid, source_times, values[:, feature]) for feature in range(values.shape[1])],
    axis=1,
  )


def pair_on_common_grid(
  first: Array,
  first_times: Array,
  second: Array,
  second_times: Array,
  *,
  length: int = 48,
) -> tuple[Array, Array]:
  """Resample two sequences to their shared normalized temporal support."""
  if length < 4:
    raise ValueError("length must be at least 4.")
  first_times = np.asarray(first_times, dtype=np.float64)
  second_times = np.asarray(second_times, dtype=np.float64)
  start = max(float(first_times[0]), float(second_times[0]))
  stop = min(float(first_times[-1]), float(second_times[-1]))
  if stop <= start:
    raise ValueError("Sequences have no overlapping temporal support.")
  grid = np.linspace(start, stop, int(length))
  return (
    resample_sequence(first, first_times, grid),
    resample_sequence(second, second_times, grid),
  )


def aligned_rms_distance(
  first: Array,
  first_times: Array,
  second: Array,
  second_times: Array,
  *,
  length: int = 48,
) -> float:
  left, right = pair_on_common_grid(
    first, first_times, second, second_times, length=length
  )
  return float(np.sqrt(np.mean((left - right) ** 2)))


def mean_spectrum_distance(first: Array, second: Array) -> float:
  left, right = _validate(first), _validate(second)
  return float(np.linalg.norm(np.mean(left, axis=0) - np.mean(right, axis=0)) / np.sqrt(left.shape[1]))


def seasonal_amplitude_distance(first: Array, second: Array) -> float:
  left, right = _validate(first), _validate(second)
  return float(np.linalg.norm(np.ptp(left, axis=0) - np.ptp(right, axis=0)) / np.sqrt(left.shape[1]))


def mean_spectral_angle_distance(
  first: Array,
  first_times: Array,
  second: Array,
  second_times: Array,
  *,
  length: int = 48,
) -> float:
  left, right = pair_on_common_grid(
    first, first_times, second, second_times, length=length
  )
  numerator = np.sum(left * right, axis=1)
  denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
  cosine = np.clip(numerator / np.maximum(denominator, 1e-12), -1.0, 1.0)
  return float(np.mean(np.arccos(cosine)))


def fourier_magnitude_distance(
  first: Array,
  first_times: Array,
  second: Array,
  second_times: Array,
  *,
  length: int = 48,
  harmonics: int = 6,
) -> float:
  """Phase-invariant distance between normalized non-DC Fourier magnitudes."""
  left, right = pair_on_common_grid(
    first, first_times, second, second_times, length=length
  )
  left = left - np.mean(left, axis=0, keepdims=True)
  right = right - np.mean(right, axis=0, keepdims=True)
  left_mag = np.abs(np.fft.rfft(left, axis=0))[1 : harmonics + 1]
  right_mag = np.abs(np.fft.rfft(right, axis=0))[1 : harmonics + 1]
  left_mag /= max(float(np.linalg.norm(left_mag)), 1e-12)
  right_mag /= max(float(np.linalg.norm(right_mag)), 1e-12)
  return float(np.linalg.norm(left_mag - right_mag))


def _harmonic_fit(sequence: Array, times: Array, order: int, grid: Array) -> Array:
  values = _validate(sequence)
  t = np.asarray(times, dtype=np.float64)
  period = max(float(t[-1] - t[0]), 1.0)
  phase = 2.0 * np.pi * (t - t[0]) / period
  design = [np.ones_like(phase)]
  for harmonic in range(1, int(order) + 1):
    design.extend([np.sin(harmonic * phase), np.cos(harmonic * phase)])
  coefficients, *_ = np.linalg.lstsq(np.stack(design, axis=1), values, rcond=None)
  grid_phase = 2.0 * np.pi * np.asarray(grid, dtype=np.float64)
  grid_design = [np.ones_like(grid_phase)]
  for harmonic in range(1, int(order) + 1):
    grid_design.extend([
      np.sin(harmonic * grid_phase),
      np.cos(harmonic * grid_phase),
    ])
  return np.stack(grid_design, axis=1) @ coefficients


def harmonic_phase_aligned_distance(
  first: Array,
  first_times: Array,
  second: Array,
  second_times: Array,
  *,
  order: int = 3,
  length: int = 72,
) -> float:
  """Fit smooth seasonal curves and minimize their RMS over circular phase."""
  grid = np.linspace(0.0, 1.0, int(length), endpoint=False)
  left = _harmonic_fit(first, first_times, order, grid)
  right = _harmonic_fit(second, second_times, order, grid)
  distances = [np.sqrt(np.mean((left - np.roll(right, shift, axis=0)) ** 2)) for shift in range(length)]
  return float(np.min(distances))


def _local_costs(first: Array, second: Array) -> Array:
  left, right = _validate(first), _validate(second)
  if left.shape[1] != right.shape[1]:
    raise ValueError("Sequences must have matching feature dimensions.")
  return np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)


def dtw_distance(first: Array, second: Array, *, local_cost: Array | None = None) -> float:
  """Normalized open-unconstrained DTW distance with Euclidean local costs."""
  costs = _local_costs(first, second) if local_cost is None else np.asarray(local_cost, dtype=np.float64)
  rows, cols = costs.shape
  accumulated = np.full((rows + 1, cols + 1), np.inf)
  path_length = np.zeros((rows + 1, cols + 1), dtype=np.int32)
  accumulated[0, 0] = 0.0
  for row in range(1, rows + 1):
    for col in range(1, cols + 1):
      options = (
        (accumulated[row - 1, col - 1], path_length[row - 1, col - 1]),
        (accumulated[row - 1, col], path_length[row - 1, col]),
        (accumulated[row, col - 1], path_length[row, col - 1]),
      )
      previous_cost, previous_length = min(options, key=lambda item: item[0])
      accumulated[row, col] = costs[row - 1, col - 1] + previous_cost
      path_length[row, col] = previous_length + 1
  return float(accumulated[rows, cols] / max(int(path_length[rows, cols]), 1))


def time_weighted_dtw_distance(
  first: Array,
  first_times: Array,
  second: Array,
  second_times: Array,
  *,
  alpha: float = 0.1,
  beta: float = 50.0,
  cycle_length: float = 366.0,
) -> float:
  """TWDTW with the published/package logistic elapsed-time penalty."""
  costs = _local_costs(first, second)
  left_times = np.asarray(first_times, dtype=np.float64)
  right_times = np.asarray(second_times, dtype=np.float64)
  if left_times.size != costs.shape[0] or right_times.size != costs.shape[1]:
    raise ValueError("Time arrays must match sequence lengths.")
  elapsed = np.abs(left_times[:, None] - right_times[None, :])
  elapsed = np.minimum(elapsed, float(cycle_length) - np.minimum(elapsed, float(cycle_length)))
  penalty = 1.0 / (1.0 + np.exp(-float(alpha) * (elapsed - float(beta))))
  return dtw_distance(first, second, local_cost=costs + penalty)


def _soft_dtw_cost(first: Array, second: Array, gamma: float) -> float:
  costs = _local_costs(first, second) ** 2
  rows, cols = costs.shape
  accumulated = np.full((rows + 1, cols + 1), np.inf)
  accumulated[0, 0] = 0.0
  for row in range(1, rows + 1):
    for col in range(1, cols + 1):
      previous = np.asarray([
        accumulated[row - 1, col - 1],
        accumulated[row - 1, col],
        accumulated[row, col - 1],
      ])
      minimum = float(np.min(previous))
      soft_min = minimum - float(gamma) * np.log(
        np.sum(np.exp(-(previous - minimum) / float(gamma)))
      )
      accumulated[row, col] = costs[row - 1, col - 1] + soft_min
  return float(accumulated[rows, cols])


def soft_dtw_divergence(first: Array, second: Array, *, gamma: float = 0.1) -> float:
  """Nonnegative soft-DTW divergence, zero for identical sequences."""
  if gamma <= 0.0:
    raise ValueError("gamma must be positive.")
  cross = _soft_dtw_cost(first, second, gamma)
  self_first = _soft_dtw_cost(first, first, gamma)
  self_second = _soft_dtw_cost(second, second, gamma)
  return float(max(cross - 0.5 * (self_first + self_second), 0.0))


def snapshot_subspace_distance(first: Array, second: Array, *, rank: int = 3) -> float:
  """Order-invariant PCA/MSM control using dates as spectral samples."""
  left, right = _validate(first), _validate(second)
  if left.shape[1] != right.shape[1]:
    raise ValueError("Feature dimensions differ.")
  left_basis = np.linalg.svd(left.T, full_matrices=False)[0][:, :rank]
  right_basis = np.linalg.svd(right.T, full_matrices=False)[0][:, :rank]
  return rtw_dissimilarity(left_basis, right_basis)


def _hankel_basis(sequence: Array, lag: int, rank: int) -> Array:
  values = _validate(sequence)
  if lag < 2 or lag >= values.shape[0]:
    raise ValueError("lag must be in [2, n_steps-1].")
  columns = [values[start : start + lag].reshape(-1) for start in range(values.shape[0] - lag + 1)]
  matrix = np.stack(columns, axis=1)
  matrix -= np.mean(matrix, axis=1, keepdims=True)
  left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
  numerical_rank = int(np.sum(singular > 1e-9 * max(float(singular[0]), 1.0)))
  return left[:, : min(int(rank), numerical_rank)]


def mssa_subspace_distance(first: Array, second: Array, *, lag: int = 4, rank: int = 3) -> float:
  """Canonical-angle control on deterministic multichannel Hankel subspaces."""
  return rtw_dissimilarity(
    _hankel_basis(first, lag, rank),
    _hankel_basis(second, lag, rank),
  )
