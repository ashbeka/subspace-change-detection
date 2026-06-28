from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import h5py
import numpy as np

from phase1.baselines.temporal_sequence import (
  bandwise_correlation_distance,
  best_circular_shift_correlation_distance,
  best_circular_shift_rms_distance,
  centered_snapshot_subspace_distance,
  seasonal_summary_distance,
  shift_orbit_subspace_distance,
  symmetric_pca_reconstruction_error,
)
from phase1.data.breizhcrops_sequences import sample_breizhcrops_region


class TemporalControlTests(unittest.TestCase):
  def setUp(self) -> None:
    self.times = np.arange(48, dtype=np.float64)
    phase = 2.0 * np.pi * self.times / len(self.times)
    self.sequence = np.stack([
      np.sin(phase),
      np.cos(phase),
      np.sin(2.0 * phase + 0.2),
    ], axis=1)

  def test_global_shift_controls_remove_circular_phase(self) -> None:
    shifted = np.roll(self.sequence, 9, axis=0)
    self.assertLess(best_circular_shift_rms_distance(
      self.sequence, self.times, shifted, self.times, length=48
    ), 1e-12)
    self.assertLess(best_circular_shift_correlation_distance(
      self.sequence, self.times, shifted, self.times, length=48
    ), 1e-12)

  def test_shift_orbit_subspace_is_circular_shift_invariant(self) -> None:
    shifted = np.roll(self.sequence, 7, axis=0)
    self.assertLess(shift_orbit_subspace_distance(
      self.sequence, self.times, shifted, self.times, length=48
    ), 1e-10)

  def test_identity_controls_are_zero(self) -> None:
    self.assertLess(bandwise_correlation_distance(
      self.sequence, self.times, self.sequence, self.times
    ), 1e-12)
    self.assertLess(seasonal_summary_distance(self.sequence, self.sequence), 1e-12)
    self.assertLess(symmetric_pca_reconstruction_error(
      self.sequence, self.sequence, rank=3
    ), 1e-12)
    self.assertLess(centered_snapshot_subspace_distance(
      self.sequence, self.sequence, rank=3
    ), 1e-12)


class BreizhCropsLoaderTests(unittest.TestCase):
  def test_official_column_layout_and_quality_filter(self) -> None:
    with TemporaryDirectory() as temporary:
      root = Path(temporary)
      folder = root / "2017" / "L2A"
      folder.mkdir(parents=True)
      (root / "classmapping.csv").write_text(
        ",id,classname,code\n1,0,barley,ORH\n", encoding="utf-8"
      )
      key = "data/BreizhCrops/L2A_img/data/2017/L2A/frh01/csv/42.csv"
      (folder / "frh01.csv").write_text(
        "meanQA60,id,CODE_CULTU,path,idx,sequencelength\n"
        f"0.0,42,ORH,/{key},1,13\n",
        encoding="utf-8",
      )
      timestamps = (
        np.datetime64("2017-01-01", "ns").astype(np.int64)
        + np.arange(13, dtype=np.int64) * 10 * 24 * 60 * 60 * 10**9
      )
      raw = np.zeros((13, 14), dtype=np.float64)
      raw[:, 0] = timestamps
      raw[:, 1:11] = 1000.0
      raw[-1, 11] = 1.0
      with h5py.File(folder / "frh01.h5", "w") as database:
        database.create_dataset(key, data=raw)
      records, metadata = sample_breizhcrops_region(
        root, "frh01", max_per_class=1, seed=3, min_steps=12
      )
      self.assertEqual(len(records), 1)
      self.assertEqual(records[0].values.shape, (12, 10))
      self.assertTrue(np.allclose(records[0].values, 0.1))
      self.assertEqual(metadata["class_counts"], {0: 1})


if __name__ == "__main__":
  unittest.main()
