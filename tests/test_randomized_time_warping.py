import unittest

import numpy as np

from phase1.baselines.temporal_sequence import (
  dtw_distance,
  fourier_magnitude_distance,
  harmonic_phase_aligned_distance,
  mssa_subspace_distance,
  preprocess_pair,
  soft_dtw_divergence,
  time_weighted_dtw_distance,
)
from phase1.subspace.randomized_time_warping import (
  canonical_correlations,
  fit_rtw_hypo_subspace,
  ordered_subsequence_indices,
  rtw_dissimilarity,
  rtw_similarity,
  time_elastic_feature_matrix,
)


class RandomizedTimeWarpingTests(unittest.TestCase):
  def setUp(self) -> None:
    time = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    self.sequence = np.stack([
      np.sin(time),
      np.cos(time),
      0.5 * np.sin(2.0 * time),
    ], axis=1)
    self.times = np.linspace(0.0, 365.0, 16)

  def test_ordered_indices_are_strictly_increasing(self) -> None:
    indices = ordered_subsequence_indices(16, 6, 40, np.random.default_rng(4))
    self.assertEqual(indices.shape, (40, 6))
    self.assertTrue(np.all(np.diff(indices, axis=1) > 0))
    self.assertGreaterEqual(int(np.min(indices)), 0)
    self.assertLess(int(np.max(indices)), 16)

  def test_time_elastic_feature_concatenation(self) -> None:
    sequence = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    indices = np.asarray([[0, 2], [0, 1]], dtype=np.int32)
    features = time_elastic_feature_matrix(sequence, indices)
    expected = np.asarray([
      [1.0, 1.0],
      [2.0, 2.0],
      [5.0, 3.0],
      [6.0, 4.0],
    ])
    np.testing.assert_allclose(features, expected)

  def test_paper_style_subspace_shape(self) -> None:
    model = fit_rtw_hypo_subspace(
      self.sequence,
      subsequence_length=6,
      n_samples=100,
      rank=4,
      rng=np.random.default_rng(7),
    )
    self.assertEqual(model.feature_matrix.shape, (18, 100))
    self.assertEqual(model.basis.shape, (18, 4))
    self.assertAlmostEqual(float(np.linalg.norm(model.basis.T @ model.basis - np.eye(4))), 0.0, places=10)

  def test_identical_features_give_unit_similarity(self) -> None:
    indices = ordered_subsequence_indices(16, 6, 120, np.random.default_rng(10))
    first = fit_rtw_hypo_subspace(
      self.sequence,
      subsequence_length=6,
      n_samples=120,
      rank=4,
      rng=np.random.default_rng(1),
      sample_indices=indices,
    )
    second = fit_rtw_hypo_subspace(
      self.sequence,
      subsequence_length=6,
      n_samples=120,
      rank=4,
      rng=np.random.default_rng(2),
      sample_indices=indices,
    )
    self.assertAlmostEqual(rtw_similarity(first.basis, second.basis), 1.0, places=10)
    self.assertAlmostEqual(rtw_dissimilarity(first.basis, second.basis), 0.0, places=10)

  def test_similarity_is_basis_rotation_invariant(self) -> None:
    rng = np.random.default_rng(12)
    basis, _ = np.linalg.qr(rng.normal(size=(20, 5)))
    rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    np.testing.assert_allclose(
      canonical_correlations(basis, basis @ rotation),
      np.ones(5),
      atol=1e-10,
    )

  def test_invalid_subsequence_length_is_rejected(self) -> None:
    with self.assertRaises(ValueError):
      ordered_subsequence_indices(8, 9, 10, np.random.default_rng(1))

  def test_pair_preprocessing_modes(self) -> None:
    shifted = 2.0 * self.sequence + 3.0
    left, right = preprocess_pair(self.sequence, shifted, "per_sequence_zscore")
    np.testing.assert_allclose(left, right, atol=1e-10)
    self.assertAlmostEqual(float(np.mean(left)), 0.0, places=10)

  def test_fourier_and_harmonic_controls_ignore_circular_phase(self) -> None:
    shifted = np.roll(self.sequence, 4, axis=0)
    fourier = fourier_magnitude_distance(
      self.sequence, self.times, shifted, self.times, length=64, harmonics=5
    )
    harmonic = harmonic_phase_aligned_distance(
      self.sequence, self.times, shifted, self.times, order=3, length=64
    )
    self.assertLess(fourier, 0.15)
    self.assertLess(harmonic, 0.15)

  def test_dtw_family_identity(self) -> None:
    self.assertAlmostEqual(dtw_distance(self.sequence, self.sequence), 0.0, places=12)
    self.assertGreaterEqual(
      time_weighted_dtw_distance(
        self.sequence, self.times, self.sequence, self.times
      ),
      0.0,
    )
    self.assertAlmostEqual(
      soft_dtw_divergence(self.sequence, self.sequence),
      0.0,
      places=10,
    )

  def test_mssa_detects_order_destruction(self) -> None:
    rng = np.random.default_rng(32)
    permuted = self.sequence[rng.permutation(len(self.sequence))]
    identity = mssa_subspace_distance(self.sequence, self.sequence, lag=4, rank=3)
    destroyed = mssa_subspace_distance(self.sequence, permuted, lag=4, rank=3)
    self.assertAlmostEqual(identity, 0.0, places=10)
    self.assertGreater(destroyed, 0.02)


if __name__ == "__main__":
  unittest.main()
