import unittest

import numpy as np

from phase1.baselines.ir_mad import ir_mad_score


class IRMADBaselineTests(unittest.TestCase):
    def test_identical_images_are_near_zero(self):
        rng = np.random.default_rng(123)
        x1 = rng.normal(size=(4, 20, 20)).astype(np.float32)
        x2 = x1.copy()
        valid = np.ones((20, 20), dtype=bool)

        score = ir_mad_score(x1, x2, valid, iters=5, downsample_max_pixels=10000, random_state=123)

        self.assertEqual(score.shape, valid.shape)
        self.assertTrue(np.all(np.isfinite(score)))
        self.assertLess(float(np.max(score)), 1e-4)

    def test_synthetic_changed_block_scores_higher(self):
        rng = np.random.default_rng(456)
        x1 = rng.normal(size=(4, 24, 24)).astype(np.float32)
        x2 = x1 + rng.normal(scale=0.03, size=x1.shape).astype(np.float32)
        changed = np.zeros((24, 24), dtype=bool)
        changed[8:16, 9:17] = True
        x2[0, changed] += 3.0
        x2[1, changed] -= 2.0
        valid = np.ones((24, 24), dtype=bool)

        score = ir_mad_score(x1, x2, valid, iters=8, downsample_max_pixels=10000, random_state=456)

        self.assertTrue(np.all(np.isfinite(score)))
        self.assertGreater(float(score[changed].mean()), float(score[~changed].mean()) + 0.2)


if __name__ == "__main__":
    unittest.main()
