"""
Siamese U-Net baseline for change detection.

Source/provenance:
- Inspired by the FC-Siamese family used for OSCD change detection by Daudt et
  al.: pre and post images are encoded with shared weights and feature
  differences are decoded into a binary change map.
- This implementation uses absolute differences between matching encoder
  feature maps, then a U-Net-like decoder.

Project adaptation:
- The model expects exactly a pre/post Sentinel-2 stack with shape
  (B, 2*n_bands, H, W). Prior channels are intentionally rejected here because
  this baseline is meant to isolate a Siamese raw-image comparison.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .unet2d import DoubleConv, Down, Up


class SharedEncoder(nn.Module):
  def __init__(self, in_channels: int, base_channels: int = 64, depth: int = 4):
    super().__init__()
    if depth < 2:
      raise ValueError("depth must be >= 2")
    chs = [base_channels * (2 ** i) for i in range(depth)]
    self.chs = chs
    self.inc = DoubleConv(in_channels, chs[0])
    downs = []
    for i in range(1, depth):
      downs.append(Down(chs[i - 1], chs[i]))
    self.down_blocks = nn.ModuleList(downs)

  def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
    x1 = self.inc(x)
    xs = [x1]
    x_cur = x1
    for down in self.down_blocks:
      x_cur = down(x_cur)
      xs.append(x_cur)
    return xs


class SiameseUNet2D(nn.Module):
  def __init__(self, n_bands: int, base_channels: int = 64, depth: int = 4, num_classes: int = 1):
    super().__init__()
    if n_bands <= 0:
      raise ValueError("n_bands must be positive")
    self.n_bands = int(n_bands)
    self.depth = int(depth)
    self.encoder = SharedEncoder(self.n_bands, base_channels=base_channels, depth=depth)

    chs = self.encoder.chs
    ups = []
    for i in range(depth - 1, 0, -1):
      ups.append(Up(chs[i], chs[i - 1]))
    self.up_blocks = nn.ModuleList(ups)

    self.out_conv = nn.Conv2d(chs[0], num_classes, kernel_size=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
      raise ValueError(f"Expected (B,C,H,W), got {tuple(x.shape)}")
    if x.size(1) < 2 * self.n_bands:
      raise ValueError(f"Expected at least {2 * self.n_bands} channels, got {x.size(1)}")
    if x.size(1) != 2 * self.n_bands:
      raise ValueError(
        f"SiameseUNet2D expects exactly 2*n_bands channels ({2 * self.n_bands}); got {x.size(1)}. "
        "Disable priors and use_pre_post_stack=true for this model."
      )

    x1 = x[:, : self.n_bands]
    x2 = x[:, self.n_bands :]

    f1 = self.encoder(x1)
    f2 = self.encoder(x2)
    diffs = [torch.abs(b - a) for a, b in zip(f1, f2)]

    x_cur = diffs[-1]
    for i, up in enumerate(self.up_blocks):
      skip = diffs[-(i + 2)]
      x_cur = up(x_cur, skip)
    return self.out_conv(x_cur)
