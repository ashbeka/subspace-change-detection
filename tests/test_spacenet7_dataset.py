"""Shape, identity, and validity checks for the SpaceNet 7 loader."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

from phase1.data.spacenet7_dataset import (
  first_appearance_features,
  load_spacenet7_image,
  rasterize_feature_mask,
  read_geojson_features,
  scan_spacenet7_aoi,
)


def _feature(identifier: int, left: float = 0.0) -> dict:
  return {
    "type": "Feature",
    "properties": {"Id": identifier},
    "geometry": {
      "type": "Polygon",
      "coordinates": [[[left, 4.0], [left + 1.0, 4.0], [left + 1.0, 3.0], [left, 3.0], [left, 4.0]]],
    },
  }


def _write_geojson(path: Path, features: list[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")


class SpaceNet7DatasetTests(unittest.TestCase):
  def test_first_appearance_uses_persistent_ids(self):
    sequences = [[_feature(1)], [_feature(1), _feature(2)], [_feature(2), _feature(3)]]
    selected = first_appearance_features(sequences)
    self.assertEqual([[feature["properties"]["Id"] for feature in date] for date in selected], [[1], [2], [3]])

  def test_rasterize_feature_mask(self):
    mask = rasterize_feature_mask(
      [_feature(1)],
      out_shape=(4, 4),
      transform=from_origin(0.0, 4.0, 1.0, 1.0),
    )
    self.assertEqual(mask.dtype, np.bool_)
    self.assertTrue(mask[0, 0])
    self.assertEqual(int(np.sum(mask)), 1)

  def test_matched_pixel_coordinates_use_identity_transform(self):
    pixel_feature = {
      "type": "Feature",
      "properties": {"Id": 1},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[2.0, 1.0], [3.0, 1.0], [3.0, 2.0], [2.0, 2.0], [2.0, 1.0]]],
      },
    }
    mask = rasterize_feature_mask(
      [pixel_feature],
      out_shape=(4, 4),
      transform=rasterio.Affine.identity(),
    )
    self.assertTrue(mask[1, 2])
    self.assertEqual(int(np.sum(mask)), 1)

  def test_scan_and_load_combine_alpha_and_udm(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      image_dir = root / "images"
      matched_dir = root / "labels_match_pix"
      udm_dir = root / "labels"
      image_dir.mkdir()
      transform = from_origin(0.0, 4.0, 1.0, 1.0)
      for date in ("2018_02", "2018_01"):
        stem = f"global_monthly_{date}_mosaic_TEST"
        values = np.full((4, 4, 4), 128, dtype=np.uint8)
        values[3] = 255
        values[3, 3, 3] = 0
        with rasterio.open(
          image_dir / f"{stem}.tif",
          "w",
          driver="GTiff",
          width=4,
          height=4,
          count=4,
          dtype="uint8",
          crs="EPSG:3857",
          transform=transform,
        ) as dataset:
          dataset.write(values)
        _write_geojson(matched_dir / f"{stem}_Buildings.geojson", [_feature(1)])
        _write_geojson(udm_dir / f"{stem}_UDM.geojson", [_feature(99)] if date == "2018_01" else [])

      observations = scan_spacenet7_aoi(root)
      self.assertEqual([observation.date for observation in observations], ["2018_01", "2018_02"])
      image = load_spacenet7_image(observations[0])
      self.assertEqual(image.rgb.shape, (3, 4, 4))
      self.assertFalse(image.valid_mask[0, 0])
      self.assertFalse(image.valid_mask[3, 3])
      self.assertTrue(image.valid_mask[1, 1])
      self.assertEqual(len(read_geojson_features(observations[0].matched_label_path)), 1)

  def test_scan_rejects_unmatched_dates(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / "images").mkdir()
      (root / "labels_match_pix").mkdir()
      (root / "images" / "global_monthly_2018_01_mosaic_TEST.tif").touch()
      _write_geojson(
        root / "labels_match_pix" / "global_monthly_2018_02_mosaic_TEST_Buildings.geojson",
        [_feature(1)],
      )
      with self.assertRaises(FileNotFoundError):
        scan_spacenet7_aoi(root)


if __name__ == "__main__":
  unittest.main()
