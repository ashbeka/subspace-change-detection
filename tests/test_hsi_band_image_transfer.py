import unittest

import numpy as np

from phase1.data.hsi_change import HSIChangePair
from phase1.scripts.evaluate_hsi_band_image_transfer import (
    METHODS,
    compute_score_maps,
    spatial_block_bootstrap_deltas,
)


class HSIBandImageTransferTests(unittest.TestCase):
    def make_pair(self) -> HSIChangePair:
        rng = np.random.default_rng(20260622)
        first = rng.normal(size=(6, 12, 10)).astype(np.float32)
        second = first + rng.normal(scale=0.03, size=first.shape).astype(np.float32)
        labels = np.zeros((12, 10), dtype=np.uint8)
        labels[3:8, 4:8] = 1
        second[:, 3:8, 4:8] += np.linspace(0.2, 1.0, 6)[:, None, None]
        valid = np.ones(labels.shape, dtype=bool)
        return HSIChangePair(
            name="synthetic",
            first=first,
            second=second,
            labels=labels,
            valid_mask=valid,
            original_band_indices=np.arange(1, 7, dtype=np.int32),
            source_url="synthetic",
            preprocessing="none",
        )

    def test_frozen_map_set_is_finite_and_spatial(self) -> None:
        pair = self.make_pair()
        maps = compute_score_maps(pair, rank=3, seed=7, ir_mad_iters=2)
        self.assertEqual(set(maps), set(METHODS))
        for name, score in maps.items():
            self.assertEqual(score.shape, pair.labels.shape, name)
            self.assertTrue(np.all(np.isfinite(score)), name)

    def test_block_bootstrap_is_deterministic(self) -> None:
        pair = self.make_pair()
        maps = compute_score_maps(pair, rank=3, seed=7, ir_mad_iters=2)
        kwargs = dict(repeats=8, block=4, seed=19)
        first = spatial_block_bootstrap_deltas(
            pair.labels, maps, pair.valid_mask, **kwargs
        )
        second = spatial_block_bootstrap_deltas(
            pair.labels, maps, pair.valid_mask, **kwargs
        )
        self.assertEqual(first, second)
        valid_repeats = first[
            "band_image_ds_minus_band_image_cross_reconstruction"
        ]["repeats"]
        self.assertGreater(valid_repeats, 0)
        self.assertLessEqual(valid_repeats, 8)


if __name__ == "__main__":
    unittest.main()
