import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from phase1.data.xbd_s12 import (
    damage_evaluation_mask,
    load_metadata,
    normalize_s2,
    rasterize_xbd_label,
    select_records,
)
from phase1.scripts.evaluate_xbd_s12_external import spectral_angle_score


class XBDS12DataTests(unittest.TestCase):
    def test_spectral_angle_is_zero_for_scale_and_positive_for_rotation(self) -> None:
        pre = np.asarray([[[1.0, 1.0]], [[0.0, 0.0]]], dtype=np.float32)
        post = np.asarray([[[2.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
        score = spectral_angle_score(pre, post, np.ones((1, 2), dtype=bool))
        self.assertAlmostEqual(float(score[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(score[0, 1]), np.pi / 2.0, places=6)

    def test_official_normalization_clips_to_unit_interval(self) -> None:
        cube = np.stack(
            [np.asarray([[-1.0, 0.5, 2.0]], dtype=np.float32)] * 12,
            axis=0,
        )
        normalized = normalize_s2(
            cube,
            np.zeros(12, dtype=np.float32),
            np.ones(12, dtype=np.float32),
        )
        self.assertTrue(np.allclose(normalized[:, 0, 0], 0.0))
        self.assertTrue(np.allclose(normalized[:, 0, 1], 0.5))
        self.assertTrue(np.allclose(normalized[:, 0, 2], 1.0))

    def test_metadata_event_split_and_nodata_filter(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "xbd_uid": "a",
                            "disaster": "palu-tsunami",
                            "event_split": "train",
                            "xbd_tier": "train",
                            "N_px_no_data": 0,
                        },
                        "geometry": None,
                    },
                    {
                        "type": "Feature",
                        "properties": {
                            "xbd_uid": "b",
                            "disaster": "sunda-tsunami",
                            "event_split": "test",
                            "xbd_tier": "test",
                            "N_px_no_data": 12,
                        },
                        "geometry": None,
                    },
                ],
            }
            (root / "xbd_s12_metadata.geojson").write_text(
                json.dumps(package), encoding="utf-8"
            )
            records = load_metadata(root)
            self.assertEqual([record.uid for record in select_records(records, split="train")], ["a"])
            self.assertEqual(select_records(records, split="test"), [])
            self.assertEqual(
                [
                    record.uid
                    for record in select_records(
                        records, split="test", exclude_metadata_nodata=False
                    )
                ],
                ["b"],
            )

    def test_rasterized_damage_polygon_survives_majority_downsampling(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample_post_disaster.json"
            package = {
                "features": {
                    "xy": [
                        {
                            "wkt": "POLYGON ((0 0, 31 0, 31 31, 0 31, 0 0))",
                            "properties": {"subtype": "major-damage"},
                        },
                        {
                            "wkt": "POLYGON ((64 64, 95 64, 95 95, 64 95, 64 64))",
                            "properties": {"subtype": "un-classified"},
                        },
                    ]
                }
            }
            path.write_text(json.dumps(package), encoding="utf-8")
            mask = rasterize_xbd_label(path, output_size=128)
            self.assertEqual(mask.shape, (128, 128))
            self.assertGreater(int(np.sum(mask == 3)), 0)
            self.assertGreater(int(np.sum(mask == 5)), 0)

    def test_damage_views_keep_semantics_separate(self) -> None:
        mask = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
        valid = np.ones(mask.shape, dtype=bool)
        target, support = damage_evaluation_mask(mask, valid, "full_scene_damage")
        self.assertTrue(np.array_equal(target, [[0, 0, 1], [1, 1, 0]]))
        self.assertTrue(np.array_equal(support, [[1, 1, 1], [1, 1, 0]]))
        _, building_support = damage_evaluation_mask(
            mask, valid, "building_conditional_damage"
        )
        self.assertTrue(np.array_equal(building_support, [[0, 1, 1], [1, 1, 0]]))


if __name__ == "__main__":
    unittest.main()
