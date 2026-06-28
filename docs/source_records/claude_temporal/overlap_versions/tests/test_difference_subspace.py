import unittest

import numpy as np

from phase1.ds import pca_utils


def _random_basis(dim: int, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dim, rank)))
    return q.astype(np.float32)


class DifferenceSubspaceTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
