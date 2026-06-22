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
from phase1.scripts.prepare_xbd_s12_external import (
    ensure_normalization,
    summarize_prepared_release,
)


class XBDS12DataTests(unittest.TestCase):
    def test_model_config_supplies_missing_official_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "config.json"
            source.write_text(
                json.dumps(
                    {
                        "normalization_stats": {
                            "s2": {"1st": list(range(12)), "99th": list(range(12, 24))}
                        }
                    }
                ),
                encoding="utf-8",
            )
            result = ensure_normalization(root, source)
            package = json.loads((root / "normalization.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "official_model_config_fallback")
            self.assertEqual(len(package["s2"]["1st"]), 12)
            repeated = ensure_normalization(root, source)
            self.assertEqual(
                repeated["status"], "official_model_config_fallback_existing"
            )

    def test_existing_release_summary_counts_only_release_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "s2").mkdir()
            (root / "s2" / "a.tif").write_bytes(b"123")
            (root / "xbd_s12_metadata.geojson").write_bytes(b"12345")
            (root / "normalization.json").write_bytes(b"not-in-archive")
            summary = summarize_prepared_release(root)
            self.assertEqual(summary["files"], 2)
            self.assertEqual(summary["bytes"], 8)

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
        building_target, localization_support = damage_evaluation_mask(
            mask, valid, "building_localization_diagnostic"
        )
        self.assertTrue(np.array_equal(building_target, [[0, 1, 1], [1, 1, 0]]))
        self.assertTrue(np.array_equal(localization_support, [[1, 1, 1], [1, 1, 0]]))

    def test_boundary_buffer_removes_both_sides_of_building_edge(self) -> None:
        mask = np.zeros((11, 11), dtype=np.uint8)
        mask[3:8, 3:8] = 3
        valid = np.ones(mask.shape, dtype=bool)
        target, support = damage_evaluation_mask(
            mask,
            valid,
            "full_scene_damage",
            boundary_buffer=1,
        )
        self.assertTrue(target[5, 5])
        self.assertTrue(support[5, 5])
        self.assertFalse(support[3, 5])
        self.assertFalse(support[2, 5])
        self.assertTrue(support[0, 0])


if __name__ == "__main__":
    unittest.main()
