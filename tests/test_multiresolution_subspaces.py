import unittest

import numpy as np

from phase1.subspace.multiscale_band_image import (
    SHIFTED_OFFSETS,
    multiscale_band_image_geometry,
    principal_angle_distances,
)
from phase1.subspace.successive_subspace_features import (
    fit_saab_stage,
    successive_saab_band_image_geometry,
)
from phase1.subspace.wavelet_band_image import (
    decimated_wavelet_band_image_ds,
    stationary_wavelet_band_image_geometry,
)


class MultiscaleBandImageTests(unittest.TestCase):
    def test_known_principal_angles(self) -> None:
        first = np.eye(4, dtype=np.float32)[:, :2]
        second = np.stack([np.eye(4)[:, 0], np.eye(4)[:, 2]], axis=1).astype(
            np.float32
        )
        angles, geodesic, chordal = principal_angle_distances(first, second)
        self.assertTrue(np.allclose(angles, [0.0, np.pi / 2.0], atol=1e-6))
        self.assertAlmostEqual(geodesic, np.pi / 2.0, places=6)
        self.assertAlmostEqual(chordal, 1.0, places=6)

    def test_equal_images_produce_zero_hierarchy(self) -> None:
        rng = np.random.default_rng(11)
        cube = rng.normal(size=(6, 24, 28)).astype(np.float32)
        valid = np.ones((24, 28), dtype=bool)
        result = multiscale_band_image_geometry(
            cube,
            cube.copy(),
            valid,
            rank=3,
            levels=(1, 2, 4),
            offsets=SHIFTED_OFFSETS,
            min_valid_pixels=8,
        )
        self.assertTrue(np.allclose(result.fused_map, 0.0, atol=1e-6))
        self.assertTrue(np.allclose(result.product_weighted_map, 0.0, atol=1e-6))
        self.assertAlmostEqual(result.product_geodesic_distance, 0.0, places=6)
        for coverage in result.scale_coverage.values():
            self.assertTrue(np.all(coverage[valid] > 0.0))

    def test_local_spectral_structure_change_ranks_changed_block_higher(self) -> None:
        rng = np.random.default_rng(12)
        first = rng.normal(scale=0.3, size=(6, 32, 32)).astype(np.float32)
        second = first.copy()
        pattern = np.linspace(-1.0, 1.0, 8, dtype=np.float32)[None, :]
        second[0, 12:20, 12:20] += 2.0 + pattern
        second[1, 12:20, 12:20] -= 1.2 - pattern
        second[4, 12:20, 12:20] += 0.8
        valid = np.ones((32, 32), dtype=bool)
        changed = np.zeros((32, 32), dtype=bool)
        changed[12:20, 12:20] = True
        result = multiscale_band_image_geometry(
            first,
            second,
            valid,
            rank=3,
            levels=(1, 2, 4),
            offsets=SHIFTED_OFFSETS,
            min_valid_pixels=8,
        )
        self.assertGreater(
            float(np.mean(result.fused_map[changed])),
            float(np.mean(result.fused_map[~changed])),
        )


class WaveletBandImageTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(21)
        self.first = rng.normal(scale=0.2, size=(6, 24, 28)).astype(np.float32)
        self.valid = np.ones((24, 28), dtype=bool)

    def test_equal_images_produce_zero_swt_and_dwt(self) -> None:
        swt = stationary_wavelet_band_image_geometry(
            self.first,
            self.first.copy(),
            self.valid,
            rank=3,
            levels=2,
        )
        dwt = decimated_wavelet_band_image_ds(
            self.first,
            self.first.copy(),
            self.valid,
            rank=3,
            levels=2,
        )
        self.assertTrue(np.allclose(swt.fused_map, 0.0, atol=1e-6))
        self.assertTrue(np.allclose(dwt.fused_map, 0.0, atol=1e-6))
        self.assertEqual(swt.fused_map.shape, self.valid.shape)
        self.assertEqual(dwt.fused_map.shape, self.valid.shape)

    def test_wavelet_components_localize_structured_change(self) -> None:
        second = self.first.copy()
        second[0, 8:16, 10:18] += 1.5
        second[2, 8:16, 10:18] -= 0.9
        changed = np.zeros(self.valid.shape, dtype=bool)
        changed[8:16, 10:18] = True
        swt = stationary_wavelet_band_image_geometry(
            self.first,
            second,
            self.valid,
            rank=3,
            levels=2,
        )
        self.assertIn("L2_LL", swt.component_maps)
        self.assertGreater(
            float(np.mean(swt.fused_map[changed])),
            float(np.mean(swt.fused_map[~changed])),
        )


class SuccessiveSaabTests(unittest.TestCase):
    def test_dc_ac_kernels_are_orthonormal(self) -> None:
        rng = np.random.default_rng(31)
        samples = rng.normal(size=(300, 18)).astype(np.float32)
        stage = fit_saab_stage(
            samples,
            input_channels=2,
            patch_size=3,
            pool_size=2,
            energy_threshold=0.9,
            max_output_channels=8,
        )
        flat = stage.kernels.reshape(stage.output_channels, -1)
        self.assertTrue(np.allclose(flat @ flat.T, np.eye(stage.output_channels), atol=1e-5))

    def test_equal_pair_produces_zero_successive_maps(self) -> None:
        rng = np.random.default_rng(32)
        cube = rng.normal(scale=0.2, size=(4, 20, 24)).astype(np.float32)
        valid = np.ones((20, 24), dtype=bool)
        result = successive_saab_band_image_geometry(
            cube,
            cube.copy(),
            valid,
            rank=3,
            hops=2,
            max_output_channels=(8, 8),
            max_fit_samples=400,
            seed=9,
        )
        self.assertTrue(np.allclose(result.fused_map, 0.0, atol=2e-5))
        self.assertEqual(result.fused_map.shape, valid.shape)
        self.assertEqual(len(result.stages), 2)


if __name__ == "__main__":
    unittest.main()
