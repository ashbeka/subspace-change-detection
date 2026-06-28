import unittest
from argparse import Namespace

import numpy as np

from phase1.ds import pca_utils
from phase1.scripts.compare_oscd_spatial_subspaces import (
    band_image_ds_score,
    band_image_spatial_control_values,
    parse_method_spec,
)
from phase1.subspace.band_image_geometry import (
    band_image_ds_values,
    band_image_spatial_control_values as shared_control_values,
)


class BandImageSpatialControlTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20260622)
        self.first = rng.normal(size=(18, 5)).astype(np.float32)
        self.second = self.first.copy()
        self.second[:5, 0] += np.linspace(0.5, 2.0, 5)
        self.second[8:13, 3] -= 1.2

    def test_equal_matrices_produce_zero_change(self) -> None:
        for mode in ("spatial_gram", "projector_distance", "cross_reconstruction"):
            score = band_image_spatial_control_values(
                self.first,
                self.first,
                rank=3,
                seed=7,
                mode=mode,
            )
            self.assertTrue(np.allclose(score, 0.0, atol=2e-5), (mode, score.max()))

    def test_public_method_names_resolve_to_matched_controls(self) -> None:
        expected = {
            "band_image_spatial_gram": "spatial_gram",
            "band_image_projector_distance": "projector_distance",
            "band_image_cross_reconstruction": "cross_reconstruction",
        }
        for name, mode in expected.items():
            spec = parse_method_spec(name)
            self.assertEqual(spec.name, name)
            self.assertEqual(spec.family, "band_image_control")
            self.assertEqual(spec.score_mode, mode)

    def test_normalized_spatial_gram_matches_explicit_operator(self) -> None:
        score = band_image_spatial_control_values(
            self.first,
            self.second,
            rank=3,
            seed=7,
            mode="spatial_gram",
        )
        # Columns are the band-image samples.  The PCA helper transposes this
        # matrix, so sklearn centers across bands at each spatial coordinate.
        first = self.first.astype(np.float64) - np.mean(self.first, axis=1, keepdims=True)
        second = self.second.astype(np.float64) - np.mean(self.second, axis=1, keepdims=True)
        first /= np.linalg.norm(first)
        second /= np.linalg.norm(second)
        explicit = np.linalg.norm(first @ first.T - second @ second.T, axis=1)
        self.assertTrue(np.allclose(score, explicit, atol=1e-7))

    def test_spatial_gram_uses_the_same_sample_axis_as_band_image_pca(self) -> None:
        score = band_image_spatial_control_values(
            self.first,
            self.second,
            rank=3,
            seed=7,
            mode="spatial_gram",
        )
        first_wrong = self.first.astype(np.float64) - np.mean(
            self.first, axis=0, keepdims=True
        )
        second_wrong = self.second.astype(np.float64) - np.mean(
            self.second, axis=0, keepdims=True
        )
        first_wrong /= np.linalg.norm(first_wrong)
        second_wrong /= np.linalg.norm(second_wrong)
        wrong_axis_score = np.linalg.norm(
            first_wrong @ first_wrong.T - second_wrong @ second_wrong.T,
            axis=1,
        )
        self.assertFalse(np.allclose(score, wrong_axis_score, atol=1e-5))

    def test_projector_distance_matches_explicit_projectors(self) -> None:
        rank = 3
        score = band_image_spatial_control_values(
            self.first,
            self.second,
            rank=rank,
            seed=7,
            mode="projector_distance",
        )
        first_basis = pca_utils.fit_pca_basis(
            self.first,
            rank=rank,
            variance_threshold=None,
            random_state=7,
            use_randomized=True,
        ).basis
        second_basis = pca_utils.fit_pca_basis(
            self.second,
            rank=rank,
            variance_threshold=None,
            random_state=7,
            use_randomized=True,
        ).basis
        explicit = np.linalg.norm(
            first_basis @ first_basis.T - second_basis @ second_basis.T,
            axis=1,
        )
        self.assertTrue(np.allclose(score, explicit, atol=1e-6))

    def test_uncentered_projector_matches_autocorrelation_bases(self) -> None:
        rank = 3
        score = shared_control_values(
            self.first,
            self.second,
            rank=rank,
            seed=7,
            mode="projector_distance",
            basis_mode="uncentered_autocorrelation",
        )
        first_basis = pca_utils.fit_autocorrelation_basis(self.first, rank).basis
        second_basis = pca_utils.fit_autocorrelation_basis(self.second, rank).basis
        explicit = np.linalg.norm(
            first_basis @ first_basis.T - second_basis @ second_basis.T,
            axis=1,
        )
        self.assertTrue(np.allclose(score, explicit, atol=1e-6))

    def test_uncentered_equal_matrices_produce_zero_geometry(self) -> None:
        ds = band_image_ds_values(
            self.first,
            self.first,
            rank=3,
            seed=7,
            basis_mode="uncentered_autocorrelation",
        )
        projector = shared_control_values(
            self.first,
            self.first,
            rank=3,
            seed=7,
            mode="projector_distance",
            basis_mode="uncentered_autocorrelation",
        )
        self.assertTrue(np.allclose(ds.projected_magnitude, 0.0, atol=1e-6))
        self.assertTrue(np.allclose(projector, 0.0, atol=1e-6))

    def test_cross_reconstruction_matches_pca_centering_convention(self) -> None:
        rank = 3
        score = band_image_spatial_control_values(
            self.first,
            self.second,
            rank=rank,
            seed=7,
            mode="cross_reconstruction",
        )
        first = self.first.astype(np.float64) - np.mean(
            self.first, axis=1, keepdims=True
        )
        second = self.second.astype(np.float64) - np.mean(
            self.second, axis=1, keepdims=True
        )
        first_basis = pca_utils.fit_pca_basis(
            self.first,
            rank=rank,
            variance_threshold=None,
            random_state=7,
            use_randomized=True,
        ).basis.astype(np.float64)
        second_basis = pca_utils.fit_pca_basis(
            self.second,
            rank=rank,
            variance_threshold=None,
            random_state=7,
            use_randomized=True,
        ).basis.astype(np.float64)
        first_basis = np.linalg.qr(first_basis, mode="reduced")[0]
        second_basis = np.linalg.qr(second_basis, mode="reduced")[0]
        second_cross = second - first_basis @ (first_basis.T @ second)
        first_cross = first - second_basis @ (second_basis.T @ first)
        second_self = second - second_basis @ (second_basis.T @ second)
        first_self = first - first_basis @ (first_basis.T @ first)
        explicit = np.sqrt(
            np.maximum(
                np.sum(second_cross**2, axis=1)
                - np.sum(second_self**2, axis=1)
                + np.sum(first_cross**2, axis=1)
                - np.sum(first_self**2, axis=1),
                0.0,
            )
        )
        self.assertTrue(np.allclose(score, explicit, atol=1e-6))

    def test_controls_are_equivariant_to_common_spatial_permutation(self) -> None:
        permutation = np.random.default_rng(99).permutation(self.first.shape[0])
        for mode in ("spatial_gram", "projector_distance", "cross_reconstruction"):
            original = band_image_spatial_control_values(
                self.first,
                self.second,
                rank=3,
                seed=7,
                mode=mode,
            )
            permuted = band_image_spatial_control_values(
                self.first[permutation],
                self.second[permutation],
                rank=3,
                seed=7,
                mode=mode,
            )
            self.assertTrue(np.allclose(permuted, original[permutation], atol=2e-5), mode)

    def test_shared_controls_match_oscd_entrypoint(self) -> None:
        for mode in ("spatial_gram", "projector_distance", "cross_reconstruction"):
            expected = band_image_spatial_control_values(
                self.first, self.second, rank=3, seed=7, mode=mode
            )
            actual = shared_control_values(
                self.first, self.second, rank=3, seed=7, mode=mode
            )
            self.assertTrue(np.allclose(actual, expected, atol=1e-7), mode)

    def test_shared_controls_can_reuse_ds_bases_without_changing_scores(self) -> None:
        ds = band_image_ds_values(self.first, self.second, rank=3, seed=7)
        for mode in ("projector_distance", "cross_reconstruction"):
            refit = shared_control_values(
                self.first, self.second, rank=3, seed=7, mode=mode
            )
            reused = shared_control_values(
                self.first,
                self.second,
                rank=3,
                seed=7,
                mode=mode,
                first_basis=ds.pre_basis,
                second_basis=ds.post_basis,
            )
            self.assertTrue(np.allclose(reused, refit, atol=1e-7), mode)

    def test_shared_ds_matches_oscd_band_image_score(self) -> None:
        rng = np.random.default_rng(71)
        first_cube = rng.normal(size=(13, 4, 5)).astype(np.float32)
        second_cube = first_cube.copy()
        second_cube[:, 1:3, 2:4] += rng.normal(size=(13, 2, 2)).astype(np.float32)
        valid = np.ones((4, 5), dtype=bool)
        spec = parse_method_spec("band_image_norm")
        score, _, _ = band_image_ds_score(
            first_cube,
            second_cube,
            valid,
            Namespace(rank=6, seed=7),
            spec,
        )
        first = first_cube[:, valid].T
        second = second_cube[:, valid].T
        shared = band_image_ds_values(first, second, rank=6, seed=7)
        self.assertTrue(
            np.allclose(score[valid], shared.projected_magnitude, atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
