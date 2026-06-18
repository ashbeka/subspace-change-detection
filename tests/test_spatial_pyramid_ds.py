import unittest
from types import SimpleNamespace

import numpy as np

from phase1.scripts.compare_oscd_spatial_subspaces import parse_method_spec, spatial_pyramid_ds_score


class SpatialPyramidDSTests(unittest.TestCase):
    def test_descriptive_method_name_parses_levels(self) -> None:
        spec = parse_method_spec("spatial_pyramid_1_2_4_norm")
        self.assertEqual(spec.family, "spatial_pyramid")
        self.assertEqual(spec.pyramid_levels, (1, 2, 4))
        self.assertEqual(spec.score_mode, "energy_norm")

    def test_equal_images_produce_zero_pyramid_score(self) -> None:
        rng = np.random.default_rng(12)
        cube = rng.normal(size=(13, 16, 20)).astype(np.float32)
        valid = np.ones((16, 20), dtype=bool)
        spec = parse_method_spec("spatial_pyramid_1_2_4_norm")
        score, card = spatial_pyramid_ds_score(
            cube,
            cube.copy(),
            valid,
            SimpleNamespace(rank=3, seed=1234),
            spec,
        )
        self.assertEqual(score.shape, valid.shape)
        self.assertLess(float(np.max(np.abs(score))), 1e-5)
        self.assertIn("1x1", card.input_object)
        self.assertIn("4x4", card.input_object)


if __name__ == "__main__":
    unittest.main()
