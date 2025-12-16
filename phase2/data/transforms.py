"""
Data augmentation utilities for Phase 2 (spec Section 3.2).
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import torch


class RandomFlipRotate:
  """Random flips and 90-degree rotations applied consistently to (x, y, valid)."""

  def __init__(self, enable_flip: bool = True, enable_rotate90: bool = True):
    self.enable_flip = enable_flip
    self.enable_rotate90 = enable_rotate90

  def __call__(self, x: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # x: (C,H,W), y: (1,H,W), valid: (1,H,W)
    if self.enable_flip:
      if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=[2])
        y = torch.flip(y, dims=[2])
        valid = torch.flip(valid, dims=[2])
      if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=[1])
        y = torch.flip(y, dims=[1])
        valid = torch.flip(valid, dims=[1])
    if self.enable_rotate90:
      k = int(torch.randint(0, 4, (1,)).item())
      if k:
        x = torch.rot90(x, k, dims=[1, 2])
        y = torch.rot90(y, k, dims=[1, 2])
        valid = torch.rot90(valid, k, dims=[1, 2])
    return x, y, valid


class RandomGaussianNoise:
  """Add low-level Gaussian noise to spectral channels only."""

  def __init__(self, sigma: float = 0.01, n_raw_channels: int | None = None):
    self.sigma = sigma
    self.n_raw_channels = n_raw_channels

  def __call__(self, x: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_raw = self.n_raw_channels if self.n_raw_channels is not None else int(x.shape[0])
    n_raw = max(0, min(n_raw, int(x.shape[0])))
    if n_raw == 0 or self.sigma <= 0:
      return x, y, valid
    x = x.clone()
    noise = torch.randn_like(x[:n_raw]) * float(self.sigma)
    if valid is not None:
      noise = noise * (valid > 0).to(noise.dtype)
    x[:n_raw] = x[:n_raw] + noise
    return x, y, valid


class ComposeTransforms:
  def __init__(self, transforms: List[Callable]):
    self.transforms = transforms

  def __call__(self, x: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    for t in self.transforms:
      x, y, valid = t(x, y, valid)
    return x, y, valid

