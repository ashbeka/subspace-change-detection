"""Formula and failure-boundary checks for seasonal observation subspaces."""

from __future__ import annotations

import unittest

import numpy as np

from phase1.subspace.geodesic import subspace_magnitude
from phase1.subspace.seasonal_observations import (
  build_seasonal_observation_subspace,
  seasonal_observation_matrix,
)


def _season(amplitude: float, *, phase: float = 0.0) -> list[np.ndarray]:
  rows, cols = 8, 9
  spatial = np.linspace(0.7, 1.3, rows * cols, dtype=np.float32).reshape(rows, cols)
  cubes = []
  for month in range(12):
    angle = 2.0 * np.pi * month / 12.0 + phase
    profile = np.sin(angle)
    cube = np.stack([
      0.20 + 0.03 * amplitude * profile * spatial,
      0.25 - 0.08 * amplitude * profile * spatial,
      0.30 + 0.14 * amplitude * profile * spatial,
    ]).astype(np.float32)
    cubes.append(cube)
  return cubes


class SeasonalObservationSubspaceTests(unittest.TestCase):
  def test_matrix_uses_dates_as_columns(self):
    cubes = _season(1.0)
    mask = np.ones(cubes[0].shape[1:], dtype=bool)
    matrix = seasonal_observation_matrix(cubes, mask, preprocessing="feature_centered")
    self.assertEqual(matrix.shape, (3 * 8 * 9, 12))
    np.testing.assert_allclose(np.mean(matrix, axis=1), 0.0, atol=1e-7)

  def test_equal_seasons_have_zero_difference(self):
    cubes = _season(1.0)
    mask = np.ones(cubes[0].shape[1:], dtype=bool)
    first = build_seasonal_observation_subspace(cubes, mask, rank=2)
    second = build_seasonal_observation_subspace([cube.copy() for cube in cubes], mask, rank=2)
    self.assertLess(subspace_magnitude(first.basis, second.basis), 1e-6)

  def test_feature_centering_removes_global_offset(self):
    cubes = _season(1.0)
    shifted = [cube + 3.0 for cube in cubes]
    mask = np.ones(cubes[0].shape[1:], dtype=bool)
    first = build_seasonal_observation_subspace(cubes, mask, rank=2)
    second = build_seasonal_observation_subspace(shifted, mask, rank=2)
    self.assertLess(subspace_magnitude(first.basis, second.basis), 1e-6)

  def test_unordered_subspace_is_invariant_to_date_permutation(self):
    cubes = _season(1.0)
    permuted = [cubes[index] for index in (5, 1, 9, 0, 11, 3, 7, 2, 8, 4, 10, 6)]
    mask = np.ones(cubes[0].shape[1:], dtype=bool)
    first = build_seasonal_observation_subspace(cubes, mask, rank=2)
    second = build_seasonal_observation_subspace(permuted, mask, rank=2)
    self.assertLess(subspace_magnitude(first.basis, second.basis), 1e-6)

  def test_rank_is_limited_by_centered_temporal_samples(self):
    cubes = _season(1.0)
    mask = np.ones(cubes[0].shape[1:], dtype=bool)
    fitted = build_seasonal_observation_subspace(cubes, mask, rank=20)
    self.assertLessEqual(fitted.rank, 11)
    self.assertEqual(fitted.basis.shape[0], 3 * 8 * 9)


if __name__ == "__main__":
  unittest.main()
