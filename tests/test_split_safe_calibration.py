import unittest

import numpy as np

from phase1.scripts.evaluate_oscd_split_calibration import (
    build_ranked_profile,
    counts_at_fraction,
    exact_top_fraction_prediction,
    fit_fraction,
    fraction_grid,
    paired_test_comparisons,
)


class SplitSafeCalibrationTests(unittest.TestCase):
    def test_top_fraction_selects_highest_scores(self) -> None:
        score = np.asarray([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32)
        mask = np.ones((2, 2), dtype=bool)
        prediction = exact_top_fraction_prediction(score, mask, 0.5)
        np.testing.assert_array_equal(prediction, np.asarray([[0, 1], [0, 1]], dtype=np.uint8))

    def test_profile_counts_match_expected_confusion(self) -> None:
        score = np.asarray([[0.9, 0.8, 0.2, 0.1]], dtype=np.float32)
        target = np.asarray([[1, 0, 1, 0]], dtype=np.uint8)
        mask = np.ones_like(target, dtype=bool)
        profile = build_ranked_profile("toy", "train", "method", score, target, mask)
        self.assertEqual(counts_at_fraction(profile, 0.5), (1, 1, 1, 1))

    def test_fit_fraction_uses_profiles_supplied_as_training_data(self) -> None:
        score = np.asarray([[0.9, 0.8, 0.2, 0.1]], dtype=np.float32)
        target = np.asarray([[1, 1, 0, 0]], dtype=np.uint8)
        mask = np.ones_like(target, dtype=bool)
        profile = build_ranked_profile("toy", "train", "method", score, target, mask)
        fraction, _ = fit_fraction([profile], np.asarray([0.25, 0.5, 0.75]))
        self.assertEqual(fraction, 0.5)

    def test_fraction_grid_is_sorted_and_bounded(self) -> None:
        grid = fraction_grid(20, 0.5)
        self.assertEqual(grid[0], 0.0)
        self.assertAlmostEqual(grid[-1], 0.5)
        self.assertTrue(np.all(np.diff(grid) > 0.0))

    def test_paired_comparison_uses_test_rows_only(self) -> None:
        rows = [
            {"city": "a", "split": "test", "method": "pca_diff", "f1": 0.2, "iou": 0.1, "false_discovery_fraction": 0.6},
            {"city": "a", "split": "test", "method": "candidate", "f1": 0.3, "iou": 0.2, "false_discovery_fraction": 0.5},
            {"city": "b", "split": "test", "method": "pca_diff", "f1": 0.4, "iou": 0.3, "false_discovery_fraction": 0.4},
            {"city": "b", "split": "test", "method": "candidate", "f1": 0.5, "iou": 0.4, "false_discovery_fraction": 0.3},
            {"city": "ignored", "split": "train", "method": "pca_diff", "f1": 1.0, "iou": 1.0, "false_discovery_fraction": 0.0},
            {"city": "ignored", "split": "train", "method": "candidate", "f1": 0.0, "iou": 0.0, "false_discovery_fraction": 1.0},
        ]
        comparison = paired_test_comparisons(rows, seed=1)
        f1 = next(row for row in comparison if row["metric"] == "f1")
        self.assertAlmostEqual(f1["mean_delta_method_minus_baseline"], 0.1)
        self.assertEqual(f1["wins"], 2)


if __name__ == "__main__":
    unittest.main()
