import unittest

import numpy as np

from phase1.scripts.compare_oscd_spatial_subspaces import percentile_rank_score, rank_fusion_score


class ScoreRankFusionTests(unittest.TestCase):
    def test_percentile_ranks_preserve_order_and_mask(self) -> None:
        score = np.asarray([[3.0, 1.0], [9.0, 5.0]], dtype=np.float32)
        valid = np.asarray([[True, True], [False, True]])
        ranked = percentile_rank_score(score, valid)
        self.assertEqual(float(ranked[1, 0]), 0.0)
        self.assertLess(float(ranked[0, 1]), float(ranked[0, 0]))
        self.assertLess(float(ranked[0, 0]), float(ranked[1, 1]))

    def test_equal_weight_fusion_uses_component_ranks(self) -> None:
        valid = np.ones((2, 2), dtype=bool)
        scores = {
            "a": np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
            "b": np.asarray([[3.0, 2.0], [1.0, 0.0]], dtype=np.float32),
        }
        fused, card = rank_fusion_score("test_fusion", "a+b", scores, valid)
        np.testing.assert_allclose(fused, np.full((2, 2), 0.5, dtype=np.float32), atol=1e-6)
        self.assertEqual(card.method, "test_fusion")


if __name__ == "__main__":
    unittest.main()
