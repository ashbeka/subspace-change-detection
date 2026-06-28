import unittest
from unittest.mock import patch

import numpy as np

from phase1.baselines.anomalous_change import chronochrome_score, covariance_equalization_score
from phase1.baselines.ir_mad import IRMADResult
from phase1.subspace.pif_nuisance import (
    PIFContext,
    build_pif_context,
    pif_nuisance_ds_residual_score,
    pif_radiometric_match,
    multiscale_pif_delta_residual_score,
    multiscale_spatial_features,
)


class AnomalousChangeTests(unittest.TestCase):
    def _paired_scene(self):
        rng = np.random.default_rng(52)
        height, width, bands = 40, 40, 5
        first = rng.normal(size=(bands, height, width))
        transform = np.array(
            [
                [1.10, 0.08, 0.00, 0.00, 0.00],
                [0.00, 0.90, 0.06, 0.00, 0.00],
                [0.00, 0.00, 1.05, 0.04, 0.00],
                [0.03, 0.00, 0.00, 0.95, 0.02],
                [0.00, 0.02, 0.00, 0.00, 1.08],
            ]
        )
        second = (transform @ first.reshape(bands, -1)).reshape(first.shape)
        second += rng.normal(scale=0.03, size=second.shape)
        changed = np.zeros((height, width), dtype=bool)
        changed[13:24, 16:27] = True
        second[0, changed] += 2.5
        second[3, changed] -= 1.8
        valid = np.ones((height, width), dtype=bool)
        return first.astype(np.float32), second.astype(np.float32), changed, valid

    def test_chronochrome_detects_deviation_from_normal_relationship(self):
        first, second, changed, valid = self._paired_scene()
        score = chronochrome_score(first, second, valid, max_fit_pixels=10000, random_state=52)
        self.assertGreater(float(score[changed].mean()), float(score[~changed].mean()) + 0.4)

    def test_covariance_equalization_detects_changed_block(self):
        first, second, changed, valid = self._paired_scene()
        score = covariance_equalization_score(first, second, valid, max_fit_pixels=10000, random_state=52)
        self.assertGreater(float(score[changed].mean()), float(score[~changed].mean()) + 0.2)


class PIFConditioningTests(unittest.TestCase):
    @staticmethod
    def _context(pif_mask):
        dummy = np.zeros_like(pif_mask, dtype=np.float32)
        return PIFContext(
            ir_mad=IRMADResult(dummy, np.ones_like(dummy), np.ones(4, dtype=np.float32), 1),
            pif_mask=pif_mask,
            selected_pixels=int(pif_mask.sum()),
            threshold_used=0.9,
        )

    def test_pif_radiometric_match_removes_affine_background_shift(self):
        rng = np.random.default_rng(71)
        first = rng.normal(size=(4, 30, 30)).astype(np.float32)
        slopes = np.array([1.2, 0.8, 1.1, 0.9], dtype=np.float32)[:, None, None]
        offsets = np.array([0.4, -0.2, 0.1, 0.3], dtype=np.float32)[:, None, None]
        second = slopes * first + offsets
        changed = np.zeros((30, 30), dtype=bool)
        changed[9:17, 12:20] = True
        second[1, changed] += 2.0
        valid = np.ones((30, 30), dtype=bool)
        context = self._context(~changed)

        matched, _, _ = pif_radiometric_match(first, second, valid, context=context)
        before = np.linalg.norm(second - first, axis=0)
        after = np.linalg.norm(matched - first, axis=0)

        self.assertLess(float(after[~changed].mean()), 0.02 * float(before[~changed].mean()))
        self.assertGreater(float(after[changed].mean()), float(after[~changed].mean()) + 1.0)

    def test_pif_fallback_keeps_exact_count_when_probabilities_tie(self):
        first = np.zeros((4, 20, 20), dtype=np.float32)
        valid = np.ones((20, 20), dtype=bool)
        dummy = IRMADResult(
            score=np.zeros((20, 20), dtype=np.float32),
            no_change_probability=np.zeros((20, 20), dtype=np.float32),
            canonical_correlations=np.zeros(4, dtype=np.float32),
            iterations=1,
        )
        with patch("phase1.subspace.pif_nuisance.ir_mad_result", return_value=dummy):
            context = build_pif_context(
                first,
                first,
                valid,
                pmin=0.9,
                fallback_fraction=0.1,
                minimum_pixels=10,
            )
        self.assertEqual(context.selected_pixels, 40)

    def test_pif_nuisance_ds_preserves_rare_change_signal(self):
        rng = np.random.default_rng(88)
        first = rng.normal(size=(4, 36, 36)).astype(np.float32)
        second = first.copy()
        # Pervasive low-dimensional radiometric nuisance.
        second[0] += 0.55
        second[1] += 0.25 * first[0]
        changed = np.zeros((36, 36), dtype=bool)
        changed[11:22, 14:25] = True
        second[2, changed] += 2.5
        valid = np.ones((36, 36), dtype=bool)
        context = self._context(~changed)

        result = pif_nuisance_ds_residual_score(
            first, second, valid, rank=2, max_pif_fit_pixels=10000, random_state=88, context=context
        )
        self.assertTrue(np.all(np.isfinite(result.score)))
        self.assertGreater(float(result.score[changed].mean()), 2.0 * float(result.score[~changed].mean()))

    def test_masked_spatial_features_do_not_bleed_nodata(self):
        cube = np.ones((3, 25, 25), dtype=np.float32)
        valid = np.ones((25, 25), dtype=bool)
        valid[9:16, 9:16] = False
        cube[:, ~valid] = 0.0

        features = multiscale_spatial_features(cube, valid, sigmas=(0.0, 2.0))

        self.assertEqual(features.shape, (6, 25, 25))
        self.assertTrue(np.allclose(features[3:, valid], 1.0, atol=1e-5))
        self.assertTrue(np.all(features[:, ~valid] == 0.0))

    def test_multiscale_pif_residual_preserves_coherent_change(self):
        rng = np.random.default_rng(131)
        first = rng.normal(scale=0.4, size=(4, 48, 48)).astype(np.float32)
        second = first + rng.normal(scale=0.04, size=first.shape).astype(np.float32)
        # Pervasive nuisance plus isolated high-frequency artifacts.
        second[0] += 0.35
        artifact_rows = rng.integers(0, 48, size=80)
        artifact_cols = rng.integers(0, 48, size=80)
        second[1, artifact_rows, artifact_cols] += 1.2
        changed = np.zeros((48, 48), dtype=bool)
        changed[17:31, 19:33] = True
        second[2, changed] += 1.5
        valid = np.ones((48, 48), dtype=bool)
        context = self._context(~changed)

        score, _, nuisance = multiscale_pif_delta_residual_score(
            first,
            second,
            valid,
            sigmas=(0.0, 1.0, 2.0),
            nuisance_rank=1,
            max_pif_fit_pixels=10000,
            random_state=131,
            context=context,
        )

        self.assertEqual(nuisance.shape, (12, 1))
        self.assertTrue(np.all(np.isfinite(score)))
        self.assertGreater(float(score[changed].mean()), 2.0 * float(score[~changed].mean()))


if __name__ == "__main__":
    unittest.main()
