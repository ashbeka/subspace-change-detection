import unittest

import numpy as np

from phase1.subspace import kernel_difference_subspace as kds


def _samples(dim: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(dim, n))
    x = x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-12)
    return x


class KernelDifferenceSubspaceTests(unittest.TestCase):
    def test_kernel_subspace_basis_is_orthonormal_in_rkhs(self):
        x = _samples(5, 18, seed=1)
        sub = kds.fit_kernel_subspace(x, rank=4, sigma2=5.0, label="a")
        gram = kds.rbf_kernel(x, sigma2=5.0)
        basis_gram = sub.coeffs.T @ gram @ sub.coeffs

        self.assertEqual(sub.coeffs.shape, (18, 4))
        self.assertTrue(np.allclose(basis_gram, np.eye(4), atol=1e-7), basis_gram)

    def test_kernel_difference_subspace_has_expected_shapes(self):
        a = kds.fit_kernel_subspace(_samples(6, 20, seed=2), rank=5, sigma2=5.0, label="a")
        b = kds.fit_kernel_subspace(_samples(6, 20, seed=3), rank=5, sigma2=5.0, label="b")

        kd = kds.kernel_difference_subspace([a, b], rank=5)
        coords = kds.project_kernel_difference(kd, _samples(6, 7, seed=4))

        self.assertEqual(kd.basis_gram.shape, (10, 10))
        self.assertEqual(kd.basis_coeffs_over_subspace_bases.shape, (10, 5))
        self.assertEqual(kd.sample_coeffs_by_subspace[0].shape, (20, 5))
        self.assertEqual(kd.sample_coeffs_by_subspace[1].shape, (20, 5))
        self.assertEqual(coords.shape, (5, 7))

    def test_identical_kernel_subspaces_do_not_backfill_with_common_directions(self):
        x = _samples(5, 18, seed=10)
        a = kds.fit_kernel_subspace(x, rank=4, sigma2=5.0, label="a")
        b = kds.fit_kernel_subspace(x, rank=4, sigma2=5.0, label="a_copy")

        kd = kds.kernel_difference_subspace([a, b], rank=4)

        self.assertEqual(kd.rank, 0)
        self.assertEqual(kd.sample_coeffs_by_subspace[0].shape, (18, 0))
        self.assertEqual(kd.sample_coeffs_by_subspace[1].shape, (18, 0))

    def test_kernel_difference_basis_is_normalized_in_rkhs(self):
        a = kds.fit_kernel_subspace(_samples(6, 20, seed=5), rank=4, sigma2=5.0, label="a")
        b = kds.fit_kernel_subspace(_samples(6, 20, seed=6), rank=4, sigma2=5.0, label="b")

        kd = kds.kernel_difference_subspace([a, b], rank=4)
        basis_gram = kds.rkhs_basis_gram(kd)

        self.assertTrue(np.allclose(basis_gram, np.eye(4), atol=1e-6), basis_gram)

    def test_projection_energy_is_one_value_per_sample(self):
        a = kds.fit_kernel_subspace(_samples(4, 12, seed=7), rank=3, sigma2=5.0, label="a")
        b = kds.fit_kernel_subspace(_samples(4, 12, seed=8), rank=3, sigma2=5.0, label="b")
        kd = kds.kernel_difference_subspace([a, b], rank=3)

        energy = kds.projection_energy(kd, _samples(4, 9, seed=9))

        self.assertEqual(energy.shape, (9,))
        self.assertTrue(np.all(energy >= 0.0), energy)


if __name__ == "__main__":
    unittest.main()
