import unittest

import numpy as np

from phase1.scripts.evaluate_spacenet7_band_image_transfer import (
    METHODS,
    hierarchical_bootstrap,
    tile_score_maps,
    tiled_score_maps,
)


class SpaceNet7BandImageTransferTests(unittest.TestCase):
    def test_tile_scores_are_finite_and_spatial(self) -> None:
        rng = np.random.default_rng(44)
        first = rng.uniform(0, 1, size=(3, 16, 16)).astype(np.float32)
        second = first + rng.normal(scale=0.01, size=first.shape).astype(np.float32)
        second[:, 5:10, 6:11] += 0.15
        valid = np.ones((16, 16), dtype=bool)
        maps = tile_score_maps(
            first, second, valid, rank=2, seed=7, ir_mad_iters=2
        )
        self.assertEqual(set(maps), set(METHODS))
        for name, score in maps.items():
            self.assertEqual(score.shape, valid.shape, name)
            self.assertTrue(np.all(np.isfinite(score)), name)

    def test_tiled_scores_respect_invalid_tiles(self) -> None:
        rng = np.random.default_rng(45)
        first = rng.uniform(0, 1, size=(3, 16, 16)).astype(np.float32)
        second = first + 0.02
        valid = np.ones((16, 16), dtype=bool)
        valid[:8, :8] = False
        maps, scored = tiled_score_maps(
            first,
            second,
            valid,
            tile_size=8,
            minimum_valid_pixels=8,
            rank=2,
            seed=7,
            ir_mad_iters=2,
        )
        self.assertFalse(np.any(scored[:8, :8]))
        self.assertTrue(np.all(scored[8:, 8:]))
        for score in maps.values():
            self.assertTrue(np.all(score[:8, :8] == 0.0))

    def test_hierarchical_bootstrap_is_deterministic(self) -> None:
        deltas = {
            ("a", "1"): 0.1,
            ("a", "2"): 0.2,
            ("b", "1"): -0.1,
            ("b", "2"): 0.0,
        }
        first = hierarchical_bootstrap(deltas, repeats=20, seed=9)
        second = hierarchical_bootstrap(deltas, repeats=20, seed=9)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["observed"], 0.05)
        self.assertEqual(first["aoi_count"], 2)


if __name__ == "__main__":
    unittest.main()
