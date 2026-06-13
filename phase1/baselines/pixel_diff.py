"""
Raw spectral pixel-difference baseline.

Source/provenance:
- This is the simplest form of multiband change scoring: subtract pre/post
  spectral vectors and take the L2 magnitude per pixel.
- It is intentionally close to CVA and is used to check whether DS-like scores
  are doing anything beyond naive spectral difference.

Verification status:
- Generic baseline; no paper-specific novelty claim.
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def pixel_l2_difference(x1: Array, x2: Array, valid_mask: Array) -> Array:
    """Compute L2 norm of spectral difference per pixel."""
    diff = x2 - x1
    score = np.linalg.norm(diff, axis=0)
    score[~valid_mask] = 0.0
    # min-max normalize within tile
    if np.any(valid_mask):
        v = score[valid_mask]
        vmin, vmax = v.min(), v.max()
        if vmax > vmin:
            score = (score - vmin) / (vmax - vmin)
        else:
            score[:] = 0.0
    return score.astype(np.float32)
