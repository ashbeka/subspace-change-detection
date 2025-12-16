"""
OSCD segmentation dataset for Phase 2 (spec Phase 2 Section 3).

Wraps the Phase 1 OSCD loader and adds:
- stacking of pre/post S2 bands,
- optional priors (DS / PCA-diff / pixel-diff) loaded from saved change maps,
- patch extraction and basic augmentations.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]

from phase1.data.oscd_dataset import OSCDEvaluatorDataset
from phase1.data.preprocessing import apply_normalization, load_band_stats

from .transforms import ComposeTransforms, RandomFlipRotate, RandomGaussianNoise


Array = np.ndarray
def _load_priors(
  change_maps_root: Path,
  split: str,
  city: str,
  priors_cfg: Dict[str, bool],
) -> List[Array]:
  priors: List[Array] = []
  for method_key, enabled in priors_cfg.items():
    if not enabled:
      continue
    method_name = {
      "ds_projection": "ds_projection",
      "ds_cross_residual": "ds_cross_residual",
      "pca_diff": "pca_diff",
      "pixel_diff": "pixel_diff",
      "celik": "celik",
      "ir_mad": "ir_mad",
      "geodesic_dist": "geodesic_dist",
    }.get(method_key)
    if method_name is None:
      continue
    score_path = change_maps_root / split / method_name / f"{city}_score.npy"
    if not score_path.exists():
      raise FileNotFoundError(f"Missing prior score map at {score_path}")
    scores = np.load(score_path).astype(np.float32)
    s_min, s_max = float(scores.min()), float(scores.max())
    if s_max > s_min:
      scores = (scores - s_min) / (s_max - s_min)
    else:
      scores = np.zeros_like(scores, dtype=np.float32)
    priors.append(scores[None, ...])
  return priors


class OSCDSegmentationDataset(Dataset):
  """
  Patch-wise segmentation dataset built on top of the Phase 1 OSCD loader.
  """

  @staticmethod
  def _infer_raw_channels(band_order: List[str], feats_cfg: Dict) -> int:
    n_bands = int(len(band_order))
    if not feats_cfg.get("use_raw_s2", True):
      return 0
    if feats_cfg.get("use_pre_post_stack", True):
      return 2 * n_bands
    return n_bands

  def __init__(
    self,
    oscd_root: Path,
    split: str,
    cfg: Dict,
    phase1_change_maps_root: Optional[Path] = None,
    stats_path: Optional[Path] = None,
  ):
    self.oscd_root = Path(oscd_root)
    self.split = split
    self.cfg = cfg
    self.phase1_change_maps_root = Path(phase1_change_maps_root) if phase1_change_maps_root else None

    band_order: List[str] = cfg["dataset"].get("band_order", [])
    if not band_order:
      # reuse Phase 1 default band order if not provided
      band_order = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B09",
        "B10",
        "B11",
        "B12",
        "B8A",
      ]
    self.band_order = band_order

    self.patch_size: int = int(cfg["dataset"].get("patch_size", 256))
    self.patch_overlap: int = int(cfg["dataset"].get("patch_overlap", 64))
    self.is_train = split == "train"

    cache_cfg = cfg.get("dataset", {}).get("cache", {}) or {}
    self.cache_cities: bool = bool(cache_cfg.get("cities", cfg["dataset"].get("cache_cities", True)))
    self.cache_max_cities: int = int(cache_cfg.get("max_cities", cfg["dataset"].get("cache_max_cities", 0)) or 0)
    # City cache stores precomputed arrays to avoid re-reading and re-normalizing the full tile for each patch.
    self._city_cache: "OrderedDict[str, tuple[Array, Array, Array]]" = OrderedDict()

    # Phase 1 dataset to get tiles
    self.oscd_ds = OSCDEvaluatorDataset(
      self.oscd_root,
      split,
      band_order,
      nodata_value=0.0,
      min_valid_bands=cfg["dataset"].get("min_valid_bands", 3),
      stats_path=None,
      val_cities=cfg["dataset"]["split"].get("val", []),
      val_from_train=0,
    )

    # band stats for normalization (reuse Phase 1 stats)
    if stats_path is None:
      stats_path = ROOT / "phase1" / "data" / "oscd_band_stats.json"
    self.stats = load_band_stats(Path(stats_path))

    # precompute patch index list (city, y, x)
    self.patches: List[Tuple[str, int, int]] = []
    cities_for_split = cfg["dataset"]["split"][split]
    for city in cities_for_split:
      h, w = self._get_city_shape(city)
      stride = max(1, self.patch_size - self.patch_overlap)

      def _positions(length: int, window: int, stride_val: int) -> List[int]:
        positions = list(range(0, max(1, length - window + 1), stride_val))
        last = length - window
        if positions:
          if positions[-1] != last:
            positions.append(max(0, last))
        else:
          positions = [0]
        return positions

      ys = _positions(h, self.patch_size, stride)
      xs = _positions(w, self.patch_size, stride)
      for y in ys:
        for x in xs:
          self.patches.append((city, y, x))

    # build transforms
    aug_cfg = cfg["dataset"].get("augmentations", {})
    tfs = []
    feats_cfg = self.cfg.get("features", {})
    self.n_raw_channels = self._infer_raw_channels(self.band_order, feats_cfg)
    if self.is_train:
      enable_flip = bool(aug_cfg.get("flip", True))
      enable_rotate90 = bool(aug_cfg.get("rotate90", True))
      if enable_flip or enable_rotate90:
        tfs.append(RandomFlipRotate(enable_flip=enable_flip, enable_rotate90=enable_rotate90))
      if bool(aug_cfg.get("noise", False)):
        sigma = float(aug_cfg.get("noise_sigma", 0.01))
        tfs.append(RandomGaussianNoise(sigma=sigma, n_raw_channels=self.n_raw_channels))
    self.transforms = ComposeTransforms(tfs) if tfs else None

  def _get_city_shape(self, city: str) -> tuple[int, int]:
    # Avoid loading all bands just to get H/W when possible.
    try:
      import rasterio  # type: ignore
    except Exception:
      sample = self.oscd_ds.load_city(city)
      h, w = sample.x_pre.shape[1:]
      return int(h), int(w)

    images_dir = self.oscd_root / "onera_satellite_change_detection dataset__images" / city / "imgs_1_rect"
    fp = images_dir / f"{self.band_order[0]}.tif"
    with rasterio.open(fp) as src:
      return int(src.height), int(src.width)

  def _load_city_arrays(self, city: str) -> tuple[Array, Array, Array]:
    sample = self.oscd_ds.load_city(city)

    # normalize pre/post once per city (tile-level); patches are just views/slices.
    x1_norm, vm = apply_normalization(sample.x_pre, self.stats, valid_mask=sample.valid_mask, nodata_value=0.0)
    x2_norm, _ = apply_normalization(sample.x_post, self.stats, valid_mask=sample.valid_mask, nodata_value=0.0)

    feats = []
    feats_cfg = self.cfg["features"]
    if feats_cfg.get("use_raw_s2", True):
      if feats_cfg.get("use_pre_post_stack", True):
        feats.append(x1_norm)
        feats.append(x2_norm)
      else:
        feats.append(x2_norm - x1_norm)

    if self.phase1_change_maps_root is not None:
      priors_cfg = feats_cfg.get("priors", {})
      if priors_cfg:
        priors = _load_priors(self.phase1_change_maps_root, self.split, city, priors_cfg)
        feats.extend(priors)

    x_full = np.concatenate(feats, axis=0).astype(np.float32)
    if sample.y is not None:
      y_full = sample.y.astype(np.float32)
    else:
      y_full = np.zeros((1, x_full.shape[1], x_full.shape[2]), dtype=np.float32)
    valid_full = vm.astype(np.float32)
    return x_full, y_full, valid_full

  def _get_city_arrays(self, city: str) -> tuple[Array, Array, Array]:
    if not self.cache_cities:
      return self._load_city_arrays(city)
    cached = self._city_cache.get(city)
    if cached is not None:
      self._city_cache.move_to_end(city)
      return cached
    arrays = self._load_city_arrays(city)
    self._city_cache[city] = arrays
    if self.cache_max_cities > 0:
      while len(self._city_cache) > self.cache_max_cities:
        self._city_cache.popitem(last=False)
    return arrays

  def __len__(self) -> int:
    return len(self.patches)

  def __getitem__(self, idx: int):
    city, y0, x0 = self.patches[idx]
    x_full, y_full, valid_full = self._get_city_arrays(city)

    sl_y = slice(y0, y0 + self.patch_size)
    sl_x = slice(x0, x0 + self.patch_size)
    x_patch = x_full[:, sl_y, sl_x]
    y_patch = y_full[:, sl_y, sl_x]
    v_patch = valid_full[sl_y, sl_x]

    x_tensor = torch.from_numpy(x_patch)
    y_tensor = torch.from_numpy(y_patch)
    v_tensor = torch.from_numpy(v_patch[None, ...])

    if self.transforms is not None:
      x_tensor, y_tensor, v_tensor = self.transforms(x_tensor, y_tensor, v_tensor)

    return {"city": city, "split": self.split, "x": x_tensor, "y": y_tensor, "valid": v_tensor}
