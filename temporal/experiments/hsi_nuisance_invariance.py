"""H-A decisive test: does an INVARIANT-residual method (SFA-CD) beat SAM/CVA/DS as date-to-date radiometric
NUISANCE grows? The clean Salinas proxy said DS<SAM<CVA (no nuisance). H-A claims subspaces win specifically
when there IS nuisance, because the winners model the no-change transform and flag the residual.

Synthetic bitemporal CD on real Salinas spectra: date2 = affine+illumination nuisance applied to ALL pixels
(the confound) + a genuine class-swap change on a labeled subset. Sweep nuisance strength; AUC per method.
Prediction: CVA collapses (sees nuisance as change), SAM/DS degrade (per-band gain distorts shape), SFA-CD
stays robust (learns+removes the affine nuisance, residual=change). Pre-registered; trivial nulls reported.

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.hsi_nuisance_invariance
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.io import loadmat
from scipy.linalg import eigh
from sklearn.metrics import roc_auc_score

from temporal import hsi_spectral_ds as Hs

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "hsi_nuisance")
os.makedirs(OUT, exist_ok=True)


def sfa_cd(X1, X2, eps=1e-3):
    """Bitemporal SFA change detection (Wu/Du/Zhang). X1,X2: (N,B) corresponding standardized pixels.
    Returns per-pixel chi-square change score concentrated in the slow/invariant components."""
    N, B = X1.shape
    D = X1 - X2
    A = D.T @ D / N + eps * np.eye(B)                     # difference covariance (change/derivative)
    Bm = (X1.T @ X1 + X2.T @ X2) / (2 * N) + eps * np.eye(B)   # signal covariance
    lam, W = eigh(A, Bm)                                   # ascending lam: slow/most-invariant first
    Y1, Y2 = X1 @ W, X2 @ W
    Dj = Y1 - Y2
    sig2 = Dj.var(0) + 1e-9                                # variance of each SFA difference component
    return np.sum(Dj ** 2 / sig2, axis=1)                 # chi-square; small-var (slow) comps up-weighted


def apply_nuisance(X, alpha, rng):
    """Global affine + illumination radiometric nuisance of strength alpha (the date-to-date confound)."""
    B = X.shape[1]
    gain = 1.0 + alpha * rng.standard_normal(B)           # per-band gain
    offset = alpha * rng.standard_normal(B)               # per-band offset
    illum = 1.0 + alpha * rng.standard_normal()           # global illumination scale
    return X * gain * illum + offset


def main():
    m = loadmat("data_hsi/salinas.mat")
    cube = [v for k, v in m.items() if not k.startswith("__") and v.ndim == 3][0].astype(float)
    gt = loadmat("data_hsi/salinas_gt.mat")
    lab = [v for k, v in gt.items() if not k.startswith("__") and v.ndim == 2][0]
    B = cube.shape[2]; X = cube.reshape(-1, B); y = lab.reshape(-1)
    classes = [c for c in np.unique(y) if c > 0]
    rng = np.random.default_rng(0)
    # build a labeled bitemporal scene: N pixels, 12% genuine change (class swap)
    pix = rng.choice(np.where(y > 0)[0], 8000, replace=False)
    X1 = X[pix].copy()
    changed = np.zeros(len(pix), bool); changed[rng.choice(len(pix), int(0.12 * len(pix)), replace=False)] = True
    X2_clean = X1.copy()
    for i in np.where(changed)[0]:                        # swap changed pixels to a different class' spectrum
        cj = rng.choice([c for c in classes if c != y[pix[i]]])
        X2_clean[i] = X[rng.choice(np.where(y == cj)[0])]

    def znorm(a, b):                                      # joint per-band standardize (date-independent stats)
        mu = np.concatenate([a, b]).mean(0, keepdims=True); sd = np.concatenate([a, b]).std(0, keepdims=True) + 1e-9
        return (a - mu) / sd, (b - mu) / sd

    print("H-A nuisance-invariance (Salinas synthetic bitemporal CD): AUC(change) vs nuisance strength")
    print(f"{'alpha':>6} {'SFA-CD':>7} {'SAM':>6} {'CVA':>6} {'DS':>6} {'rawL1':>6}")
    results = {}
    for alpha in [0.0, 0.05, 0.1, 0.2, 0.4]:
        X2 = apply_nuisance(X2_clean, alpha, np.random.default_rng(42))
        a1, a2 = znorm(X1, X2)
        sc = {}
        sc["sfa_cd"] = sfa_cd(a1, a2)
        sc["sam"] = np.array([Hs.sam(a1[i], a2[i]) for i in range(len(a1))])
        sc["cva"] = np.array([Hs.cva(a1[i], a2[i]) for i in range(len(a1))])
        # DS only at a few alphas (slow): subsample for the spectral-SSA per-pixel
        sub = rng.choice(len(a1), 2500, replace=False)
        ds = np.full(len(a1), np.nan); ds[sub] = [Hs.ds_score(a1[i], a2[i], w=20, rank=8) for i in sub]
        sc["ds"] = ds
        sc["rawl1"] = np.mean(np.abs(a1 - a2), axis=1)
        au = {}
        for k, v in sc.items():
            mask = ~np.isnan(v)
            au[k] = float(roc_auc_score(changed[mask], v[mask]))
        results[alpha] = au
        print(f"{alpha:>6} {au['sfa_cd']:.3f} {au['sam']:.3f} {au['cva']:.3f} {au['ds']:.3f} {au['rawl1']:.3f}")

    json.dump(results, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print("\n=== H-A VERDICT ===")
    a0, aT = min(results), max(results)
    print(f"  clean(a={a0}): SFA-CD={results[a0]['sfa_cd']:.3f} SAM={results[a0]['sam']:.3f} CVA={results[a0]['cva']:.3f}")
    print(f"  nuisance(a={aT}): SFA-CD={results[aT]['sfa_cd']:.3f} SAM={results[aT]['sam']:.3f} CVA={results[aT]['cva']:.3f}")
    win = results[aT]['sfa_cd'] > max(results[aT]['sam'], results[aT]['cva']) + 0.02
    print(f"  H-A (invariant method wins UNDER nuisance): {'SUPPORTED' if win else 'NOT supported'}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
