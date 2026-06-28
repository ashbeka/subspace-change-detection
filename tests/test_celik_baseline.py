import unittest

import numpy as np

from phase1.baselines.celik_pca_kmeans import celik_score


class CelikBaselineTests(unittest.TestCase):
    def test_spectral_norm_mode_localizes_synthetic_change(self) -> None:
        rng = np.random.default_rng(7)
        x1 = rng.normal(0.0, 0.05, size=(3, 40, 40)).astype(np.float32)
        x2 = x1.copy()
        x2[:, 14:28, 12:30] += 2.0
        valid = np.ones((40, 40), dtype=bool)

        score = celik_score(
            x1,
            x2,
            patch_size=5,
            valid_mask=valid,
            random_state=11,
            feature_mode="spectral_norm",
            max_fit_samples=800,
            score_chunk_size=200,
        )

        changed = score[16:26, 14:28]
        unchanged = np.concatenate((score[:8].ravel(), score[-8:].ravel()))
        self.assertEqual(score.shape, valid.shape)
        self.assertGreater(float(changed.mean()), float(unchanged.mean()) + 0.25)

    def test_seeded_fit_is_reproducible(self) -> None:
        rng = np.random.default_rng(9)
        x1 = rng.normal(size=(4, 24, 24)).astype(np.float32)
        x2 = x1 + rng.normal(scale=0.1, size=x1.shape).astype(np.float32)
        valid = np.ones((24, 24), dtype=bool)
        kwargs = dict(
            patch_size=3,
            valid_mask=valid,
            random_state=123,
            feature_mode="spectral_norm",
            max_fit_samples=300,
            score_chunk_size=128,
        )

        first = celik_score(x1, x2, **kwargs)
        second = celik_score(x1, x2, **kwargs)
        np.testing.assert_allclose(first, second, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
