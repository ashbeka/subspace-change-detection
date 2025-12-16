"""
Placeholder damage dataset adapter (spec Section 10).

To be implemented for xBD/xBD-S12 or similar datasets in future phases.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import csv

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from PIL import Image


Array = np.ndarray


def _positions(length: int, window: int, stride: int) -> List[int]:
  positions = list(range(0, max(1, length - window + 1), stride))
  last = length - window
  if positions:
    if positions[-1] != last:
      positions.append(max(0, last))
  else:
    positions = [0]
  return positions


def _read_raster(path: Path) -> Array:
  with rasterio.open(path) as src:
    arr = src.read()  # (C,H,W)
  return arr.astype(np.float32)


def _read_image(path: Path) -> Array:
  path = Path(path)
  if path.suffix.lower() in {".tif", ".tiff"}:
    return _read_raster(path)
  img = Image.open(path)
  arr = np.array(img)
  if arr.ndim == 2:
    arr = arr[None, ...]
  else:
    arr = arr.transpose(2, 0, 1)
  return arr.astype(np.float32)


def _read_mask(path: Path) -> Array:
  m = _read_image(path)
  if m.ndim == 3 and m.shape[0] > 1:
    # If a color image is given as a mask, take the first channel.
    m = m[:1]
  if m.ndim == 2:
    m = m[None, ...]
  return m.astype(np.int64)


class DamageDatasetAdapter(Dataset):
  def __init__(self, root: Path, split: str, config: Dict[str, Any]):
    self.root = Path(root)
    self.split = split
    self.config = config
    ds_cfg = config.get("dataset", {})
    csv_cfg = ds_cfg.get("index_csv", {})
    csv_path = csv_cfg.get(split) if isinstance(csv_cfg, dict) else None
    if csv_path is None:
      csv_path = f"{split}.csv"
    self.csv_path = (self.root / csv_path).resolve()
    if not self.csv_path.exists():
      raise FileNotFoundError(
        f"DamageDatasetAdapter expects an index CSV at {self.csv_path}. "
        "Provide dataset.index_csv.{split} in the config or create {split}.csv under the dataset root."
      )

    self.patch_size = int(ds_cfg.get("patch_size", 256))
    self.patch_overlap = int(ds_cfg.get("patch_overlap", 64))
    stride = max(1, self.patch_size - self.patch_overlap)

    feats_cfg = config.get("features", {})
    self.use_pre_post_stack = bool(feats_cfg.get("use_pre_post_stack", True))

    # Load rows.
    self._rows: List[Dict[str, str]] = []
    with self.csv_path.open("r", encoding="utf-8") as f:
      reader = csv.DictReader(f)
      for row in reader:
        self._rows.append(row)
    if not self._rows:
      raise ValueError(f"No rows found in {self.csv_path}")

    # Precompute patch index list: (row_idx, y0, x0).
    self._patches: List[Tuple[int, int, int]] = []
    for i, row in enumerate(self._rows):
      pre_rel = row.get("pre") or row.get("pre_path") or row.get("pre_image")
      if not pre_rel:
        raise ValueError("CSV must include a 'pre' (or 'pre_path'/'pre_image') column.")
      pre_path = (self.root / pre_rel).resolve()
      with rasterio.open(pre_path) as src:
        h, w = src.height, src.width
      ys = _positions(h, self.patch_size, stride)
      xs = _positions(w, self.patch_size, stride)
      for y0 in ys:
        for x0 in xs:
          self._patches.append((i, y0, x0))

  def __len__(self) -> int:
    return len(self._patches)

  def __getitem__(self, idx: int):
    row_idx, y0, x0 = self._patches[idx]
    row = self._rows[row_idx]
    sample_id = row.get("id") or row.get("name") or str(row_idx)

    pre_rel = row.get("pre") or row.get("pre_path") or row.get("pre_image")
    post_rel = row.get("post") or row.get("post_path") or row.get("post_image")
    mask_rel = row.get("mask") or row.get("mask_path") or row.get("label") or row.get("label_path")
    if not pre_rel or not post_rel or not mask_rel:
      raise ValueError("CSV must include pre/post/mask columns (e.g., 'pre','post','mask').")

    pre = _read_image(self.root / pre_rel)
    post = _read_image(self.root / post_rel)
    mask = _read_mask(self.root / mask_rel)

    # Patch crop.
    ys = slice(y0, y0 + self.patch_size)
    xs = slice(x0, x0 + self.patch_size)
    pre_p = pre[:, ys, xs]
    post_p = post[:, ys, xs]
    y_p = mask[:, ys, xs]

    if self.use_pre_post_stack:
      x_p = np.concatenate([pre_p, post_p], axis=0)
    else:
      x_p = post_p - pre_p

    # Generic valid mask: all pixels valid (dataset-specific masking can be added later).
    valid = np.ones((1, x_p.shape[1], x_p.shape[2]), dtype=np.float32)

    return {
      "id": sample_id,
      "split": self.split,
      "x": torch.from_numpy(x_p.astype(np.float32)),
      "y": torch.from_numpy(y_p.astype(np.int64)),
      "valid": torch.from_numpy(valid),
    }

