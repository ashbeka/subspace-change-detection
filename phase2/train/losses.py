"""
Segmentation losses for Phase 2 (spec Section 5.2).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
  def __init__(
    self,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    smooth: float = 1.0,
    pos_weight: float | None = None,
  ):
    super().__init__()
    self.bce_weight = bce_weight
    self.dice_weight = dice_weight
    self.smooth = smooth
    self.pos_weight = pos_weight

  def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # logits: (B,1,H,W), targets: (B,1,H,W), valid_mask: (B,1,H,W)
    probs = torch.sigmoid(logits)
    valid = valid_mask.bool()
    if valid.any():
      pw = None
      if self.pos_weight is not None:
        pw = torch.as_tensor(self.pos_weight, dtype=logits.dtype, device=logits.device)
      bce = F.binary_cross_entropy_with_logits(logits[valid], targets[valid], reduction="mean", pos_weight=pw)
    else:
      bce = logits.new_tensor(0.0)

    # dice over valid pixels
    probs_flat = probs[valid]
    targets_flat = targets[valid]
    intersection = (probs_flat * targets_flat).sum()
    denom = probs_flat.sum() + targets_flat.sum()
    dice = 1.0 - (2.0 * intersection + self.smooth) / (denom + self.smooth)

    loss = self.bce_weight * bce + self.dice_weight * dice
    return loss, bce.detach(), dice.detach()

