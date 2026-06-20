"""Formula and invariance checks for order-aware temporal representations."""

from __future__ import annotations

import unittest

import numpy as np

from phase1.subspace.geodesic import subspace_magnitude
from phase1.subspace.temporal_trajectory import (
  build_temporal_representation_subspace,
  flattened_observation_matrix,
  representation_covariance_change,
  temporal_representation_matrix,
)


def _sequence() -> list[np.ndarray]:
  rows, cols = 5, 6
  yy, xx = np.indices((rows, cols), dtype=np.float32)
  first = 0.2 + xx / 20.0
  second = 0.3 + yy / 18.0
  cubes = []
  for index in range(8):
    angle = 2.0 * np.pi * index / 8.0
    cube = np.stack([
      first + 0.07 * np.sin(angle) * second,
      second + 0.11 * np.cos(angle) * first,
      0.25 + 0.05 * np.sin(2.0 * angle + 0.4) * (first + second),
    ]).astype(np.float32)
    cubes.append(cube)
  return cubes


class TemporalTrajectorySubspaceTests(unittest.TestCase):
  def setUp(self):
    self.cubes = _sequence()
    self.mask = np.ones(self.cubes[0].shape[1:], dtype=bool)

  def test_block_trajectory_matrix_has_expected_shape(self):
    matrix = temporal_representation_matrix(
      self.cubes,
      self.mask,
      representation="trajectory",
      lag=3,
      preprocessing="uncentered",
    )
    self.assertEqual(matrix.shape, (3 * 5 * 6 * 3, 6))

  def test_equal_sequences_have_zero_difference(self):
    for representation, lag in (("unordered", 1), ("difference", 1), ("trajectory", 3)):
      first = build_temporal_representation_subspace(
        self.cubes,
        self.mask,
        rank=2,
        representation=representation,
        lag=lag,
      )
      second = build_temporal_representation_subspace(
        [cube.copy() for cube in self.cubes],
        self.mask,
        rank=2,
        representation=representation,
        lag=lag,
      )
      self.assertLess(subspace_magnitude(first.basis, second.basis), 1e-6)

  def test_unordered_full_span_is_date_permutation_invariant(self):
    order = (4, 0, 6, 2, 7, 1, 5, 3)
    first = build_temporal_representation_subspace(
      self.cubes,
      self.mask,
      rank=8,
      representation="unordered",
      preprocessing="uncentered",
    )
    second = build_temporal_representation_subspace(
      [self.cubes[index] for index in order],
      self.mask,
      rank=8,
      representation="unordered",
      preprocessing="uncentered",
    )
    self.assertLess(subspace_magnitude(first.basis, second.basis), 1e-5)

  def test_unordered_full_span_is_invertible_date_mixing_invariant(self):
    observations = flattened_observation_matrix(self.cubes, self.mask)
    rng = np.random.default_rng(42)
    mixing = rng.normal(size=(len(self.cubes), len(self.cubes)))
    mixing += 3.0 * np.eye(len(self.cubes))
    mixed = observations @ mixing
    mixed_cubes = [
      mixed[:, index].reshape(self.cubes[0].shape).astype(np.float32)
      for index in range(mixed.shape[1])
    ]
    first = build_temporal_representation_subspace(
      self.cubes,
      self.mask,
      rank=8,
      representation="unordered",
      preprocessing="uncentered",
    )
    second = build_temporal_representation_subspace(
      mixed_cubes,
      self.mask,
      rank=8,
      representation="unordered",
      preprocessing="uncentered",
    )
    self.assertLess(subspace_magnitude(first.basis, second.basis), 2e-5)

  def test_trajectory_subspace_responds_to_nontrivial_date_permutation(self):
    order = (4, 0, 6, 2, 7, 1, 5, 3)
    first = build_temporal_representation_subspace(
      self.cubes,
      self.mask,
      rank=2,
      representation="trajectory",
      lag=3,
    )
    second = build_temporal_representation_subspace(
      [self.cubes[index] for index in order],
      self.mask,
      rank=2,
      representation="trajectory",
      lag=3,
    )
    self.assertGreater(subspace_magnitude(first.basis, second.basis), 0.05)

  def test_difference_subspace_responds_to_nontrivial_date_permutation(self):
    order = (4, 0, 6, 2, 7, 1, 5, 3)
    first = build_temporal_representation_subspace(
      self.cubes,
      self.mask,
      rank=2,
      representation="difference",
    )
    second = build_temporal_representation_subspace(
      [self.cubes[index] for index in order],
      self.mask,
      rank=2,
      representation="difference",
    )
    self.assertGreater(subspace_magnitude(first.basis, second.basis), 0.01)

  def test_covariance_change_is_zero_for_identical_representations(self):
    fitted = build_temporal_representation_subspace(
      self.cubes,
      self.mask,
      rank=2,
      representation="trajectory",
      lag=2,
      preprocessing="uncentered",
    )
    self.assertAlmostEqual(representation_covariance_change(fitted, fitted), 0.0, places=7)

  def test_covariance_change_retains_within_span_energy(self):
    scaled = [cube.copy() for cube in self.cubes]
    scaled[1] *= 2.0
    first = build_temporal_representation_subspace(
      self.cubes,
      self.mask,
      rank=2,
      representation="unordered",
      preprocessing="uncentered",
    )
    second = build_temporal_representation_subspace(
      scaled,
      self.mask,
      rank=2,
      representation="unordered",
      preprocessing="uncentered",
    )
    self.assertGreater(representation_covariance_change(first, second), 1e-4)

  def test_covariance_change_matches_explicit_operator(self):
    first = build_temporal_representation_subspace(
      self.cubes,
      self.mask,
      rank=2,
      representation="unordered",
      preprocessing="uncentered",
    )
    second_cubes = [cube.copy() for cube in self.cubes]
    second_cubes[2][0] *= 1.4
    second = build_temporal_representation_subspace(
      second_cubes,
      self.mask,
      rank=2,
      representation="unordered",
      preprocessing="uncentered",
    )
    left = first.representation_matrix.astype(np.float64)
    right = second.representation_matrix.astype(np.float64)
    left_operator = left @ left.T / np.sum(left * left)
    right_operator = right @ right.T / np.sum(right * right)
    expected = float(np.linalg.norm(left_operator - right_operator))
    self.assertAlmostEqual(representation_covariance_change(first, second), expected, places=7)


if __name__ == "__main__":
  unittest.main()
