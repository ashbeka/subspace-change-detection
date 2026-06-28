import unittest

import numpy as np

from phase1.subspace.hsi_moment_geometry import (
    factor_local_moments,
    projector_row_energy,
)


def exact_samples(covariance: np.ndarray, repeats: int = 8) -> np.ndarray:
    """Create zero-mean samples whose covariance is proportional to covariance."""
    values, vectors = np.linalg.eigh(covariance)
    root = vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))
    signed = np.concatenate([np.eye(covariance.shape[0]), -np.eye(covariance.shape[0])])
    return np.tile(signed @ root.T, (repeats, 1))


class HSIMomentGeometryTests(unittest.TestCase):
    def test_pure_mean_shift_only_changes_mean_factor(self) -> None:
        rng = np.random.default_rng(1)
        first = rng.normal(size=(200, 10))
        second = first + np.linspace(0.1, 1.0, 10)
        result = factor_local_moments(first, second, rank=4, seed=1)
        self.assertGreater(result.mean_distance, 1.0)
        self.assertLess(result.log_scale_change, 1e-10)
        self.assertLess(result.eigenspectrum_hellinger, 1e-6)
        self.assertLess(result.orientation_chordal, 1e-5)

    def test_isotropic_scale_does_not_create_orientation(self) -> None:
        rng = np.random.default_rng(2)
        first = rng.normal(size=(250, 12))
        second = 2.0 * first
        result = factor_local_moments(first, second, rank=5, seed=2)
        self.assertAlmostEqual(result.log_scale_change, np.log(4.0), places=5)
        self.assertLess(result.orientation_chordal, 1e-5)
        self.assertLess(result.eigenspectrum_hellinger, 1e-5)

    def test_orientation_change_with_matched_eigenvalues_is_detected(self) -> None:
        dimension = 10
        values = np.asarray([9.0, 7.0, 5.0, 3.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1])
        first_covariance = np.diag(values)
        angle = np.deg2rad(35.0)
        rotation = np.eye(dimension)
        rotation[[0, 5], [0, 5]] = np.cos(angle)
        rotation[0, 5] = -np.sin(angle)
        rotation[5, 0] = np.sin(angle)
        second_covariance = rotation @ first_covariance @ rotation.T
        first = exact_samples(first_covariance)
        second = exact_samples(second_covariance)
        result = factor_local_moments(first, second, rank=4, seed=3)
        self.assertLess(result.mean_distance, 1e-10)
        self.assertLess(result.log_scale_change, 1e-10)
        self.assertLess(result.eigenspectrum_hellinger, 1e-5)
        self.assertGreater(result.orientation_chordal, 0.4)

    def test_eigenvalue_change_with_fixed_basis_is_not_orientation(self) -> None:
        first_values = np.asarray([9.0, 7.0, 5.0, 3.0, 1.0, 0.8, 0.6, 0.4])
        second_values = np.asarray([12.0, 5.5, 4.0, 2.5, 1.0, 0.8, 0.6, 0.4])
        # Equalize trace to isolate normalized eigenspectrum shape.
        second_values *= np.sum(first_values) / np.sum(second_values)
        first = exact_samples(np.diag(first_values))
        second = exact_samples(np.diag(second_values))
        result = factor_local_moments(first, second, rank=4, seed=4)
        self.assertLess(result.log_scale_change, 1e-10)
        self.assertGreater(result.eigenspectrum_hellinger, 0.02)
        self.assertLess(result.orientation_chordal, 1e-5)

    def test_projector_attribution_is_basis_invariant_and_conservative(self) -> None:
        rng = np.random.default_rng(5)
        first, _ = np.linalg.qr(rng.normal(size=(14, 5)))
        second, _ = np.linalg.qr(rng.normal(size=(14, 5)))
        rotation_first, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        rotation_second, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        expected = projector_row_energy(first, second)
        actual = projector_row_energy(first @ rotation_first, second @ rotation_second)
        cosines = np.linalg.svd(first.T @ second, compute_uv=False)
        chordal_squared = float(np.sum(1.0 - cosines**2))
        self.assertTrue(np.allclose(expected, actual, atol=1e-10))
        self.assertAlmostEqual(float(np.sum(expected)), 2.0 * chordal_squared, places=8)


if __name__ == "__main__":
    unittest.main()
