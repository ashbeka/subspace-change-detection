"""
Closing Sensei's CCA / S3CCA ask: the Canonical-Correlation family for change detection.

Sensei named two papers (S3CCA: Smoothly Structured Sparse CCA, ICPR2014; and
temporally-regularized CCA) and said "look into CCA". This runs the CCA family
on a real benchmark so we can SHOW we tried it (per the senpai rule), and place
it on the naive->complex ladder beside DS and CVA.

Methods (per-pixel change score, AUC vs change GT):
  cva        : ||x_post - x_pre||  (naive amplitude)
  sam        : spectral angle      (naive direction)
  cca        : single-pass CCA/MAD chi-square (Nielsen MAD = CCA between the two dates)
  ir_mad     : iteratively-reweighted CCA (the field-standard CCA change detector)
  sparse_cca : soft-thresholded (sparse) canonical vectors  -> the S3CCA "sparse" flavor
  ds         : canonical Difference-Subspace projection magnitude

Data: Benton (real bitemporal 159-band HSI); change = multiclass classes 1-6, no-change = 7.

Run: .\.venv\Scripts\python.exe -m phase1.experiments.cca_change
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from phase1.ds import pca_utils

BENTON = ROOT / "data" / "HSI_change" / "benton"
OUT = ROOT / "phase1" / "outputs" / "cca_change"


def auc(y, s):
    a = roc_auc_score(y, s)
    return float(max(a, 1 - a))


def cca_directions(X, Y, reg=1e-3, k=None):
    """Standard CCA. X,Y: (N,B). Returns rho, A, B (canonical dirs as columns)."""
    N, B = X.shape
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    Sxx = Xc.T @ Xc / N + reg * np.eye(B)
    Syy = Yc.T @ Yc / N + reg * np.eye(B)
    Sxy = Xc.T @ Yc / N
    # whiten
    ix = np.linalg.inv(np.linalg.cholesky(Sxx))
    iy = np.linalg.inv(np.linalg.cholesky(Syy))
    M = ix @ Sxy @ iy.T
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    k = k or B
    A = (ix.T @ U[:, :k])
    Bv = (iy.T @ Vt.T[:, :k])
    return s[:k], A, Bv


def mad_chi2(Xc, Yc, A, Bv, rho):
    """Per-pixel MAD chi-square change score from canonical variates."""
    U = Xc @ A
    V = Yc @ Bv
    MAD = U - V                      # (N,k)
    var = 2.0 * (1.0 - np.clip(rho, 0, 0.999) ** 2)[None, :]
    return np.sum(MAD * MAD / (var + 1e-9), axis=1)


def soft_threshold(A, frac=0.5):
    """Zero out the smallest |loadings| per column (sparse canonical vectors)."""
    out = A.copy()
    for j in range(A.shape[1]):
        col = np.abs(A[:, j])
        thr = np.quantile(col, frac)
        out[col < thr, j] = 0.0
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pre = sio.loadmat(BENTON / "PreImg_2004.mat")["img_2004"].astype(np.float64)
    post = sio.loadmat(BENTON / "PostImg_2007.mat")["img_2007"].astype(np.float64)
    gt = sio.loadmat(BENTON / "Reference_Map_Multiclass.mat")["Ref_map_multiclass"].astype(int)
    H, W, B = pre.shape
    Xp = pre.reshape(-1, B); Xq = post.reshape(-1, B)
    both = np.concatenate([Xp, Xq], 0); mu, sd = both.mean(0), both.std(0) + 1e-6
    Xp = (Xp - mu) / sd; Xq = (Xq - mu) / sd
    y = (gt.reshape(-1) >= 1) & (gt.reshape(-1) <= 6)
    y = y.astype(np.uint8)

    scores = {}
    diff = Xq - Xp
    scores["cva"] = np.linalg.norm(diff, axis=1)
    dot = np.sum(Xp * Xq, axis=1)
    scores["sam"] = np.arccos(np.clip(dot / (np.linalg.norm(Xp, axis=1) * np.linalg.norm(Xq, axis=1) + 1e-9), -1, 1))

    rho, A, Bv = cca_directions(Xp, Xq, reg=1e-2)
    Xc = Xp - Xp.mean(0, keepdims=True); Yc = Xq - Xq.mean(0, keepdims=True)
    scores["cca"] = mad_chi2(Xc, Yc, A, Bv, rho)
    As, Bs = soft_threshold(A, 0.5), soft_threshold(Bv, 0.5)
    scores["sparse_cca"] = mad_chi2(Xc, Yc, As, Bs, rho)

    # IR-MAD via the project baseline (expects (C,H,W))
    from phase1.baselines.ir_mad import ir_mad_score
    valid = np.ones((H, W), bool)
    scores["ir_mad"] = ir_mad_score(((pre - mu) / sd).transpose(2, 0, 1).astype(np.float32),
                                    ((post - mu) / sd).transpose(2, 0, 1).astype(np.float32),
                                    valid, iters=10).reshape(-1)

    # canonical DS projection magnitude (global-pixel construction)
    pre_b = pca_utils.fit_pca_basis(Xp.T, rank=12, variance_threshold=None).basis
    post_b = pca_utils.fit_pca_basis(Xq.T, rank=12, variance_threshold=None).basis
    D = pca_utils.build_difference_subspace(pre_b, post_b, variant="canonical")
    scores["ds"] = np.sum((D.T @ diff.T) ** 2, axis=0) if D.shape[1] else np.zeros(len(y))

    res = {m: {"auc": auc(y, s), "ap": float(average_precision_score(y, s))} for m, s in scores.items()}
    order = sorted(res, key=lambda m: res[m]["auc"], reverse=True)
    print(f"{'method':>12}{'AUC':>9}{'AP':>9}")
    for m in order:
        print(f"{m:>12}{res[m]['auc']:>9.4f}{res[m]['ap']:>9.4f}")
    json.dump(res, open(OUT / "results.json", "w"), indent=2)
    print(f"\nCCA family ranked: {order}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
