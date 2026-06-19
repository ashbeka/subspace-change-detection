import unittest

import numpy as np

from phase1.ds import pca_utils


def _random_basis(dim: int, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dim, rank)))
    return q.astype(np.float32)


class DifferenceSubspaceTests(unittest.TestCase):
    def test_autocorrelation_basis_retains_mean_direction(self):
        rng = np.random.default_rng(14)
        mean_direction = np.array([1.0, 2.0, -1.0, 0.5], dtype=np.float64)
        mean_direction /= np.linalg.norm(mean_direction)
        samples = 8.0 * mean_direction[:, None] + rng.normal(scale=0.2, size=(4, 500))

        uncentered = pca_utils.fit_autocorrelation_basis(samples, rank=1).basis[:, 0]
        centered = pca_utils.fit_pca_basis(
            samples, rank=1, variance_threshold=None, use_randomized=False
        ).basis[:, 0]

        self.assertGreater(abs(float(uncentered @ mean_direction)), 0.99)
        self.assertLess(abs(float(centered @ mean_direction)), 0.5)

    def test_covariance_basis_matches_centered_pca_span(self):
        rng = np.random.default_rng(15)
        samples = rng.normal(size=(7, 400)) + rng.normal(size=(7, 1))
        expected = pca_utils.fit_pca_basis(
            samples, rank=4, variance_threshold=None, use_randomized=False
        ).basis
        actual = pca_utils.fit_covariance_basis(samples, rank=4).basis
        cosines = np.linalg.svd(expected.T @ actual, compute_uv=False)
        self.assertTrue(np.all(cosines > 1.0 - 1e-5), cosines)

    def test_equal_subspaces_have_empty_paper_faithful_ds(self):
        phi = _random_basis(13, 6, seed=1)

        eig = pca_utils.difference_subspace_eig(phi, phi)
        canonical = pca_utils.difference_subspace_canonical(phi, phi)

        self.assertEqual(eig.shape, (13, 0))
        self.assertEqual(canonical.shape, (13, 0))

    def test_random_subspaces_have_expected_paper_faithful_dim(self):
        phi = _random_basis(13, 6, seed=1)
        psi = _random_basis(13, 6, seed=2)

        eig = pca_utils.difference_subspace_eig(phi, psi)
        canonical = pca_utils.difference_subspace_canonical(phi, psi)

        self.assertEqual(eig.shape, (13, 6))
        self.assertEqual(canonical.shape, (13, 6))

    def test_eig_and_canonical_span_same_random_ds(self):
        phi = _random_basis(13, 6, seed=3)
        psi = _random_basis(13, 6, seed=4)

        eig = pca_utils.difference_subspace_eig(phi, psi)
        canonical = pca_utils.difference_subspace_canonical(phi, psi)
        cosines = np.linalg.svd(eig.T @ canonical, compute_uv=False)

        self.assertTrue(np.all(cosines > 1.0 - 1e-4), cosines)

    def test_residual_alias_is_legacy_residual_stack(self):
        phi = _random_basis(13, 6, seed=5)
        psi = _random_basis(13, 6, seed=6)

        self.assertEqual(pca_utils.resolve_subspace_variant("residual"), "legacy_residual_stack")
        legacy = pca_utils.build_difference_subspace(phi, psi, variant="residual")

        self.assertEqual(legacy.shape, (13, 12))

    def test_canonical_components_retain_first_order_magnitude(self):
        phi = _random_basis(13, 6, seed=7)
        psi = _random_basis(13, 6, seed=8)

        components = pca_utils.difference_subspace_canonical_components(phi, psi)
        canonical = pca_utils.difference_subspace_canonical(phi, psi)
        correlations = np.linalg.svd(phi.T @ psi, compute_uv=False)

        span_cosines = np.linalg.svd(components.basis.T @ canonical, compute_uv=False)
        expected_magnitude = 2.0 * (len(correlations) - float(np.sum(correlations)))
        self.assertTrue(np.all(span_cosines > 1.0 - 1e-4), span_cosines)
        self.assertAlmostEqual(float(np.sum(components.squared_pair_magnitudes)), expected_magnitude, places=5)
        self.assertTrue(np.allclose(components.basis.T @ components.basis, np.eye(6), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
