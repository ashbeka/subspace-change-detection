"""
Change Vector Analysis baseline.

Source/provenance:
- CVA is a classical remote-sensing change-detection baseline: form the
  multiband difference vector at each pixel and use its magnitude as a change
  score.
- In this project, CVA is implemented as the L2 norm of the 13-band Sentinel-2
  difference vector. Thresholding is handled upstream by Otsu/global/evaluation
  code.

Verification status:
- This is a simple baseline pressure method, not a novel project contribution.
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def cva_score(x1: Array, x2: Array, valid_mask: Array) -> Array:
    diff = x2 - x1
    score = np.linalg.norm(diff, axis=0)
    score[~valid_mask] = 0.0
    if np.any(valid_mask):
        v = score[valid_mask]
        vmin, vmax = v.min(), v.max()
        if vmax > vmin:
            score = (score - vmin) / (vmax - vmin)
        else:
            score[:] = 0.0
    return score.astype(np.float32)
