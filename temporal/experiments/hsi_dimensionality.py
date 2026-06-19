"""#1 EXPERIMENT: does the spectral-signal Difference Subspace beat spectral-angle on real hyperspectral CD,
and does the DS-SAM advantage GROW with band count? (the dimensionality-threshold result)

Pre-registered: construction frozen (Hankel window w=20, energy-rank, centered); falsifier = DS never beats
SAM at full bands -> the wall is fundamental (#10). Always report SAM (amplitude-invariant) as the null.

Usage: cd <worktree> && <venv>/python.exe -m temporal.experiments.hsi_dimensionality <path.mat>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

from temporal import hsi_spectral_ds as H

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "hsi_dim")
os.makedirs(OUT, exist_ok=True)
W_HANKEL, RANK, ENERGY = 20, 8, 0.99
BAND_COUNTS = [13, 30, 50, 100]   # + full appended at runtime
METHODS = ["ds", "sam", "cva", "rawmean"]


def load_mat_cd(path):
    """Flexible loader: returns (img1 HxWxB, img2 HxWxB, labels HxW binary). Finds the two largest 3-D
    arrays as the date cubes and a 2-D array (with few unique values) as the change label."""
    m = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    cubes = sorted([(k, v) for k, v in m.items() if isinstance(v, np.ndarray) and v.ndim == 3],
                   key=lambda kv: -kv[1].size)
    twod = [(k, v) for k, v in m.items() if isinstance(v, np.ndarray) and v.ndim == 2 and v.size > 100]
    if len(cubes) < 2:
        raise ValueError(f"need 2 image cubes; found keys { {k: v.shape for k, v in m.items() if isinstance(v, np.ndarray)} }")
    (k1, c1), (k2, c2) = cubes[0], cubes[1]
    # label = the 2-D array with the fewest unique values (binary/categorical change map), shape ~ cube HxW
    HW = c1.shape[:2]
    lbl_cands = [(k, v) for k, v in twod if v.shape == HW or v.shape == HW[::-1]]
    if not lbl_cands:
        lbl_cands = twod
    lbl_cands.sort(key=lambda kv: len(np.unique(kv[1])))
    lk, lab = lbl_cands[0]
    if lab.shape != HW:
        lab = lab.T
    print(f"  cubes: {k1}{c1.shape} {k2}{c2.shape} | label: {lk}{lab.shape} uniques={np.unique(lab)[:6]}")
    return c1.astype(float), c2.astype(float), lab


def main():
    path = sys.argv[1]
    name = os.path.splitext(os.path.basename(path))[0]
    img1, img2, lab = load_mat_cd(path)
    Hh, Ww, B = img1.shape
    X1 = img1.reshape(-1, B); X2 = img2.reshape(-1, B); y = lab.reshape(-1)
    # binarize label: treat max class / nonzero-vs-zero. Many sets: 0=no-change, >0 (or 255/2)=change.
    uniq = np.unique(y)
    ybin = (y == uniq.max()).astype(int) if len(uniq) <= 3 else (y > 0).astype(int)
    # drop 'unknown' pixels if a 3-value label (0 unchanged, 1 changed, 2 unknown) — keep {0, changed}
    valid = np.ones(len(y), bool)
    if len(uniq) == 3:  # likely unchanged/changed/unknown
        # changed = the class with intermediate count? safest: changed=middle value, unknown=other. Use heuristic:
        ybin = (y == uniq[1]).astype(int); valid = (y != uniq[2]) if uniq.max() == uniq[2] else valid
    print(f"  {name}: {Hh}x{Ww}x{B}, changed frac={ybin[valid].mean():.3f}, n_valid={valid.sum()}")

    # balanced sample for the sweep (speed): up to 4000 changed + 4000 unchanged
    rng = np.random.default_rng(0)
    ch = np.where(valid & (ybin == 1))[0]; un = np.where(valid & (ybin == 0))[0]
    n = min(4000, len(ch), len(un))
    idx = np.concatenate([rng.choice(ch, n, replace=False), rng.choice(un, n, replace=False)])
    ys = ybin[idx]

    # per-band standardize using the sampled pixels' stats (date-wise)
    def zscore(X):
        mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True) + 1e-9
        return (X - mu) / sd
    full = list(range(B))

    # L0: intrinsic rank of spectral-SSA subspaces (must be >1)
    ranks = [H.intrinsic_rank(zscore(X1[idx])[i], w=W_HANKEL) for i in range(0, len(idx), 50)]
    print(f"  L0 intrinsic spectral-SSA rank (energy95): median={np.median(ranks):.1f} max={np.max(ranks)} "
          f"(>1 needed; rank1 => DS degenerates to SAM)")

    results = {}
    for bc in BAND_COUNTS + [B]:
        bands = full if bc >= B else sorted(rng.choice(full, bc, replace=False))
        a1 = zscore(X1[idx][:, bands]); a2 = zscore(X2[idx][:, bands])
        ww = min(W_HANKEL, max(3, len(bands) // 2))
        aucs = {}
        for meth in METHODS:
            sc = H.score_image(a1, a2, method=meth, w=ww, rank=RANK)
            try:
                aucs[meth] = float(roc_auc_score(ys, sc))
            except Exception:
                aucs[meth] = float("nan")
        aucs["ds_minus_sam"] = round(aucs["ds"] - aucs["sam"], 4)
        results[bc] = aucs
        print(f"  bands={bc:>4} (w={ww}): DS={aucs['ds']:.3f} SAM={aucs['sam']:.3f} CVA={aucs['cva']:.3f} "
              f"raw={aucs['rawmean']:.3f}  | DS-SAM={aucs['ds_minus_sam']:+.3f}")

    json.dump({"name": name, "shape": [Hh, Ww, B], "L0_rank_median": float(np.median(ranks)),
               "results": results}, open(os.path.join(OUT, f"{name}.json"), "w"), indent=2)
    print("\n=== DIMENSIONALITY-THRESHOLD VERDICT ===")
    gaps = [(bc, results[bc]["ds_minus_sam"]) for bc in sorted(results)]
    print("  DS-SAM gap vs band count:", "  ".join(f"{bc}:{g:+.3f}" for bc, g in gaps))
    grows = gaps[-1][1] > gaps[0][1] and gaps[-1][1] > 0.01
    print(f"  hypothesis (DS-SAM grows with bands AND DS>SAM at full): {'SUPPORTED' if grows else 'NOT supported'}")
    print(f"saved -> {OUT}/{name}.json")


if __name__ == "__main__":
    main()
