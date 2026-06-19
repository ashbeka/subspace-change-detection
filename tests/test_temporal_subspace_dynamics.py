"""Formula and construction checks for sequential band-image subspaces."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import ndimage

from phase1.subspace.second_order_ds import second_order_difference_subspace
from phase1.subspace.second_order_ds import principal_component_subspace
from phase1.subspace.temporal_band_images import (
  build_band_image_subspace,
  build_tiled_band_image_subspaces,
  sequence_common_valid_mask,
  spatial_grid_bounds,
  spatial_difference_contribution,
  temporal_difference_measurements,
  tiled_temporal_decomposition,
)
from phase1.subspace.temporal_context import (
  bidirectional_temporal_context_boundary,
  fit_temporal_context_basis,
  temporal_context_indices,
)
from phase1.subspace.geodesic import (
  grassmann_geodesic_interpolate,
  subspace_magnitude,
)
from phase1.scripts.evaluate_temporal_context_registration_curve import estimate_phase_translation


def _grassmann_line(position: float, angles: np.ndarray) -> np.ndarray:
  """Create a point on a known Grassmann geodesic in R^(2r)."""
  rank = int(angles.size)
  basis = np.zeros((2 * rank, rank), dtype=np.float64)
  for index, angle in enumerate(angles):
    basis[index, index] = np.cos(position * angle)
    basis[rank + index, index] = np.sin(position * angle)
  return basis.astype(np.float32)


class TemporalSubspaceDynamicsTests(unittest.TestCase):
  def test_phase_correlation_recovers_global_translation(self):
    rng = np.random.default_rng(37)
    reference = ndimage.gaussian_filter(rng.normal(size=(96, 96)), sigma=1.0)
    moving = ndimage.shift(reference, shift=(0.0, 1.5), order=1, mode="nearest")
    dy, dx = estimate_phase_translation(reference, moving)
    self.assertAlmostEqual(dy, 0.0, delta=0.05)
    self.assertAlmostEqual(dx, -1.5, delta=0.08)

  def test_snapshot_temporal_basis_matches_direct_svd_span(self):
    rng = np.random.default_rng(41)
    matrix = rng.normal(size=(80, 7))
    fitted = fit_temporal_context_basis(
      matrix, rank=4, preprocessing="uncentered"
    )
    direct, _, _ = np.linalg.svd(matrix, full_matrices=False)
    cosines = np.linalg.svd(fitted.basis.T @ direct[:, :4], compute_uv=False)
    np.testing.assert_allclose(cosines, np.ones(4), atol=1e-5)

  def test_temporal_context_indices_replicate_sequence_endpoints(self):
    backward, forward = temporal_context_indices(5, 1, 3)
    self.assertEqual(backward, (0, 0, 0))
    self.assertEqual(forward, (1, 2, 3))
    backward, forward = temporal_context_indices(5, 4, 3)
    self.assertEqual(backward, (1, 2, 3))
    self.assertEqual(forward, (4, 4, 4))

  def test_identical_temporal_contexts_have_zero_ds_and_novelty(self):
    rng = np.random.default_rng(43)
    cube = rng.normal(size=(3, 10, 12)).astype(np.float32)
    cubes = [cube.copy() for _ in range(8)]
    mask = np.ones(cube.shape[1:], dtype=bool)
    result = bidirectional_temporal_context_boundary(
      cubes,
      mask,
      boundary_index=4,
      context_size=3,
      rank=2,
      factorization="per_band",
      preprocessing="centered_column_l2",
    )
    self.assertLess(result.ds_magnitude, 1e-6)
    self.assertLess(float(np.max(result.ds_map)), 1e-6)
    self.assertLess(float(np.max(result.projection_novelty_map)), 1e-6)

  def test_temporal_context_ds_localizes_persistent_change(self):
    rng = np.random.default_rng(47)
    base = rng.normal(size=(3, 16, 16)).astype(np.float32)
    cubes = [base.copy() for _ in range(8)]
    cubes[4][0, 3:8, 5:11] += 5.0
    for index in range(5, len(cubes)):
      cubes[index] = cubes[4].copy()
    mask = np.ones(base.shape[1:], dtype=bool)
    result = bidirectional_temporal_context_boundary(
      cubes,
      mask,
      boundary_index=4,
      context_size=3,
      rank=2,
      factorization="per_band",
      preprocessing="centered_column_l2",
    )
    changed = result.ds_map[3:8, 5:11]
    unchanged_mask = np.ones(mask.shape, dtype=bool)
    unchanged_mask[3:8, 5:11] = False
    self.assertGreater(result.ds_magnitude, 1e-4)
    self.assertGreater(float(np.mean(changed)), 4.0 * float(np.mean(result.ds_map[unchanged_mask])))
    self.assertGreater(
      float(np.mean(result.projection_novelty_map[3:8, 5:11])),
      float(np.mean(result.projection_novelty_map[unchanged_mask])),
    )

  def test_centered_l2_temporal_context_is_global_gain_offset_invariant(self):
    rng = np.random.default_rng(53)
    base = rng.normal(size=(3, 12, 14)).astype(np.float32)
    cubes = [
      (scale * base + offset).astype(np.float32)
      for scale, offset in zip(
        (0.7, 0.9, 1.1, 1.4, 1.8, 2.1),
        (-3.0, -1.0, 0.0, 2.0, 4.0, 7.0),
      )
    ]
    mask = np.ones(base.shape[1:], dtype=bool)
    result = bidirectional_temporal_context_boundary(
      cubes,
      mask,
      boundary_index=3,
      context_size=3,
      rank=2,
      factorization="per_band",
      preprocessing="centered_column_l2",
    )
    self.assertLess(result.ds_magnitude, 1e-5)
    self.assertLess(float(np.max(result.ds_map)), 1e-5)

  def test_principal_vector_mean_matches_projector_eigenspace(self):
    rng = np.random.default_rng(19)
    first, _ = np.linalg.qr(rng.normal(size=(12, 4)))
    second, _ = np.linalg.qr(rng.normal(size=(12, 4)))
    mean = principal_component_subspace(first, second)
    projector_sum = first @ first.T + second @ second.T
    eigenvalues, eigenvectors = np.linalg.eigh(projector_sum)
    expected = eigenvectors[:, eigenvalues > 1.0 + 1e-6]
    singular_values = np.linalg.svd(mean.T @ expected, compute_uv=False)
    np.testing.assert_allclose(singular_values, np.ones(4), atol=1e-5)

  def test_equal_speed_geodesic_has_zero_second_order_magnitude(self):
    angles = np.array([0.2, 0.35], dtype=np.float64)
    s1 = _grassmann_line(0.0, angles)
    s2 = _grassmann_line(0.5, angles)
    s3 = _grassmann_line(1.0, angles)
    result = second_order_difference_subspace(s1, s2, s3, decompose=True)
    self.assertLess(result.mag_total, 1e-5)
    self.assertLess(float(result.mag_along or 0.0), 1e-5)
    self.assertLess(float(result.mag_orth or 0.0), 1e-5)

  def test_nonuniform_motion_on_geodesic_is_along_component(self):
    angles = np.array([0.3, 0.5], dtype=np.float64)
    s1 = _grassmann_line(0.0, angles)
    s2 = _grassmann_line(0.2, angles)
    s3 = _grassmann_line(1.0, angles)
    result = second_order_difference_subspace(s1, s2, s3, decompose=True)
    self.assertGreater(result.mag_total, 1e-4)
    self.assertLess(float(result.mag_orth or 0.0), 1e-5)
    self.assertGreater(float(result.mag_along or 0.0), 1e-4)

  def test_departure_from_endpoint_sum_has_orthogonal_component(self):
    angles = np.array([0.25, 0.4], dtype=np.float64)
    s1 = _grassmann_line(0.0, angles)
    s2 = _grassmann_line(0.5, angles).astype(np.float64)
    s3 = _grassmann_line(1.0, angles)
    # Rotate one middle direction into a coordinate outside span(S1,S3).
    s2[:, 0] *= np.cos(0.3)
    s2 = np.vstack([s2, np.zeros((1, s2.shape[1]))])
    s2[-1, 0] = np.sin(0.3)
    s1 = np.vstack([s1, np.zeros((1, s1.shape[1]))])
    s3 = np.vstack([s3, np.zeros((1, s3.shape[1]))])
    result = second_order_difference_subspace(s1, s2.astype(np.float32), s3, decompose=True)
    self.assertGreater(float(result.mag_orth or 0.0), 1e-4)

  def test_band_image_subspace_uses_spatial_locations_as_ambient_dimension(self):
    rows, cols = 4, 5
    grid = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
    cube = np.stack([grid, grid**2, np.flipud(grid)], axis=0)
    mask = np.ones((rows, cols), dtype=bool)
    result = build_band_image_subspace(cube, mask, rank=2, preprocessing="centered_band_l2")
    self.assertEqual(result.basis.shape, (rows * cols, 2))
    self.assertEqual(result.n_bands, 3)
    np.testing.assert_allclose(result.basis.T @ result.basis, np.eye(2), atol=1e-5)

  def test_sequence_mask_keeps_only_locations_valid_at_every_date(self):
    first = np.ones((2, 3, 3), dtype=np.float32)
    second = np.ones((2, 3, 3), dtype=np.float32)
    first[:, 0, 0] = 0.0
    second[:, 1, 1] = 0.0
    mask = sequence_common_valid_mask([first, second], nodata_value=0.0)
    self.assertEqual(int(mask.sum()), 7)
    self.assertFalse(bool(mask[0, 0]))
    self.assertFalse(bool(mask[1, 1]))

  def test_temporal_measurements_expose_irregular_gap_ratio(self):
    angles = np.array([0.2, 0.3], dtype=np.float64)
    bases = [_grassmann_line(x, angles) for x in (0.0, 0.5, 1.0)]
    metrics = temporal_difference_measurements(bases, ["20200101", "20200106", "20200121"])
    self.assertEqual(metrics["second_gap_ratio"], [3.0])
    self.assertEqual(metrics["adjacent_gap_days"], [5.0, 15.0])

  def test_time_aware_deviation_is_zero_for_irregular_constant_speed_motion(self):
    angles = np.array([0.2, 0.3], dtype=np.float64)
    # The middle sample occurs one quarter of the way through the endpoint
    # interval (5 days followed by 15 days), not at its midpoint.
    bases = [_grassmann_line(x, angles) for x in (0.0, 0.25, 1.0)]
    metrics = temporal_difference_measurements(bases, ["20200101", "20200106", "20200121"])
    self.assertGreater(metrics["second_magnitude"][0], 1e-4)
    self.assertLess(metrics["time_aware_geodesic_deviation_magnitude"][0], 1e-5)
    self.assertLess(metrics["time_aware_geodesic_deviation"][0], 1e-5)

  def test_geodesic_interpolation_matches_known_line(self):
    angles = np.array([0.25, 0.45], dtype=np.float64)
    start = _grassmann_line(0.0, angles)
    end = _grassmann_line(1.0, angles)
    expected = _grassmann_line(0.3, angles)
    actual = grassmann_geodesic_interpolate(start, end, 0.3)
    singular_values = np.linalg.svd(actual.T @ expected, compute_uv=False)
    np.testing.assert_allclose(singular_values, np.ones(2), atol=1e-5)

  def test_spatial_contribution_sums_to_first_order_magnitude(self):
    angles = np.array([0.2, 0.4], dtype=np.float64)
    first = _grassmann_line(0.0, angles)
    second = _grassmann_line(1.0, angles)
    mask = np.ones((2, 2), dtype=bool)
    contribution = spatial_difference_contribution(first, second, mask)
    self.assertAlmostEqual(float(np.sum(contribution)), subspace_magnitude(first, second), places=5)

  def test_spatial_grid_cells_cover_each_location_once(self):
    coverage = np.zeros((7, 10), dtype=int)
    bounds = spatial_grid_bounds(coverage.shape, 3)
    self.assertEqual(len(bounds), 9)
    for _, _, row_start, row_end, col_start, col_end in bounds:
      coverage[row_start:row_end, col_start:col_end] += 1
    np.testing.assert_array_equal(coverage, np.ones_like(coverage))

  def test_local_temporal_decomposition_is_spatially_selective(self):
    rng = np.random.default_rng(31)
    endpoint = rng.normal(size=(3, 12, 12)).astype(np.float32)
    middle = endpoint.copy()
    pattern = np.zeros((12, 12), dtype=np.float32)
    pattern[:6, :6] = np.indices((6, 6)).sum(axis=0) % 2
    middle[0] += 4.0 * pattern
    mask = np.ones((12, 12), dtype=bool)
    fitted = [
      build_tiled_band_image_subspaces(
        cube,
        mask,
        grid_size=2,
        rank=3,
        preprocessing="centered",
        min_valid_locations=8,
      )
      for cube in (endpoint, middle, endpoint)
    ]
    result = tiled_temporal_decomposition(
      fitted[0], fitted[1], fitted[2], output_shape=mask.shape, time_fraction=0.5
    )
    self.assertEqual(len(result.tile_rows), 4)
    changed = next(row for row in result.tile_rows if row["tile_row"] == 0 and row["tile_col"] == 0)
    unchanged = [row for row in result.tile_rows if row is not changed]
    self.assertGreater(float(changed["time_aware"]), 1e-4)
    self.assertLess(max(float(row["time_aware"]) for row in unchanged), 1e-5)
    self.assertAlmostEqual(
      float(np.sum(result.time_aware)),
      sum(float(row["time_aware"]) for row in result.tile_rows),
      places=5,
    )


if __name__ == "__main__":
  unittest.main()
