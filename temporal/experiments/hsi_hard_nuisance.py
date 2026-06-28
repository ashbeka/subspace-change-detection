"""1a GATING TEST: do LINEAR invariant-residual methods (SFA-CD, IR-MAD) degrade under HARDER nuisance
(nonlinear, spatially-varying), opening room for a kernel/local method — or do they hold (=> our novelty
must be attribution, not a new invariance)? Pre-registered; nulls (SAM/CVA) always reported.

Synthetic bitemporal CD on real Salinas (raw reflectance): date2 = nuisance(date1) + 12% class-swap change.
Nuisance types: affine (1 linear map), nonlinear (per-band gamma+gain+offset), spatial (location-dependent
gain field). Sweep strength alpha; AUC per method per type.

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.hsi_hard_nuisance
"""
from __future__ import annotations

import json
import os

import numpy as np
import scipy.linalg as sla
from scipy.io import loadmat
from scipy.stats import chi2 as chi2dist
from sklearn.metrics import roc_auc_score

from temporal import hsi_spectral_ds as Hs
from temporal.experiments.hsi_nuisance_invariance import sfa_cd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "hsi_hard_nuisance")
os.makedirs(OUT, exist_ok=True)


def irmad(X1, X2, iters=6, eps=1e-4):
    """Iteratively-reweighted MAD change statistic (the established affine-invariant CD bar)."""
    N, B = X1.shape
    w = np.ones(N); chi2 = np.zeros(N)
    for _ in range(iters):
        ws = w.sum()
        m1 = (w[:, None] * X1).sum(0) / ws; m2 = (w[:, None] * X2).sum(0) / ws
        A1 = X1 - m1; A2 = X2 - m2
        S11 = (w[:, None] * A1).T @ A1 / ws + eps * np.eye(B)
        S22 = (w[:, None] * A2).T @ A2 / ws + eps * np.eye(B)
        S12 = (w[:, None] * A1).T @ A2 / ws
        M = sla.solve(S11, S12) @ sla.solve(S22, S12.T)
        ev, V = sla.eig(M)
        a = V.real[:, np.argsort(ev.real)]                 # ascending canonical corr^2 = most-change first
        b = sla.solve(S22, S12.T) @ a
        Mv = A1 @ a - A2 @ b
        var = Mv.var(0) + 1e-9
        chi2 = np.sum(Mv ** 2 / var, axis=1)
        w = 1 - chi2dist.cdf(chi2, df=B)
    return chi2


def nuisance(X, kind, alpha, pos, rng):
    """Apply nuisance to RAW reflectance X (N,B). pos=(row,col) normalized in [0,1] for spatial."""
    B = X.shape[1]
    if kind == "affine":
        g = 1 + alpha * rng.standard_normal(B); o = alpha * rng.standard_normal(B) * X.std()
        return X * g * (1 + alpha * rng.standard_normal()) + o
    if kind == "nonlinear":                                # per-band gamma (nonlinear in reflectance) + gain/offset
        gamma = np.clip(1 + alpha * rng.standard_normal(B), 0.4, 2.5)
        g = 1 + 0.5 * alpha * rng.standard_normal(B); o = 0.3 * alpha * rng.standard_normal(B) * X.std()
        Xp = np.clip(X, 1e-6, None)
        return g * (Xp ** gamma) + o
    if kind == "spatial":                                  # location-dependent multiplicative field (smooth gradient)
        r, c = pos
        field = 1 + alpha * (2 * (np.sin(np.pi * r) * np.cos(np.pi * c)))   # per-pixel gain in [1-2a,1+2a]
        g = 1 + 0.3 * alpha * rng.standard_normal(B)
        return (X * g) * field[:, None]
    raise ValueError(kind)


def main():
    m = loadmat("data_hsi/salinas.mat")
    cube = [v for k, v in m.items() if not k.startswith("__") and v.ndim == 3][0].astype(float)
    gt = loadmat("data_hsi/salinas_gt.mat")
    lab = [v for k, v in gt.items() if not k.startswith("__") and v.ndim == 2][0]
    Hh, Ww, B = cube.shape; X = cube.reshape(-1, B); y = lab.reshape(-1)
    classes = [c for c in np.unique(y) if c > 0]
    rng = np.random.default_rng(0)
    flat = rng.choice(np.where(y > 0)[0], 6000, replace=False)
    rows, cols = np.unravel_index(flat, (Hh, Ww))
    pos = (rows / Hh, cols / Ww)
    X1 = X[flat].copy()
    changed = np.zeros(len(flat), bool); changed[rng.choice(len(flat), int(0.12 * len(flat)), replace=False)] = True
    X2c = X1.copy()
    for i in np.where(changed)[0]:
        cj = rng.choice([c for c in classes if c != y[flat[i]]])
        X2c[i] = X[rng.choice(np.where(y == cj)[0])]

    def znorm(a, b):
        s = np.concatenate([a, b]); mu = s.mean(0, keepdims=True); sd = s.std(0, keepdims=True) + 1e-9
        return (a - mu) / sd, (b - mu) / sd

    methods = ["sfa_cd", "irmad", "sam", "cva"]
    results = {}
    for kind in ["affine", "nonlinear", "spatial"]:
        print(f"\n=== nuisance: {kind} ===")
        print(f"{'alpha':>6} {'SFA-CD':>7} {'IR-MAD':>7} {'SAM':>6} {'CVA':>6}")
        results[kind] = {}
        for alpha in [0.0, 0.1, 0.2, 0.4]:
            X2 = nuisance(X2c, kind, alpha, pos, np.random.default_rng(42))
            a1, a2 = znorm(X1, X2)
            sc = {"sfa_cd": sfa_cd(a1, a2), "irmad": irmad(a1, a2),
                  "sam": np.array([Hs.sam(a1[i], a2[i]) for i in range(len(a1))]),
                  "cva": np.array([Hs.cva(a1[i], a2[i]) for i in range(len(a1))])}
            au = {k: float(roc_auc_score(changed, v)) for k, v in sc.items()}
            results[kind][alpha] = au
            print(f"{alpha:>6} {au['sfa_cd']:.3f} {au['irmad']:.3f} {au['sam']:.3f} {au['cva']:.3f}")

    json.dump(results, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print("\n=== 1a GATING VERDICT (does LINEAR invariant-residual degrade under harder nuisance?) ===")
    for kind in ["nonlinear", "spatial"]:
        aT = max(results[kind])
        sfa = results[kind][aT]["sfa_cd"]; irm = results[kind][aT]["irmad"]; best_lin = max(sfa, irm)
        a0 = results[kind][min(results[kind])]
        drop = max(a0["sfa_cd"], a0["irmad"]) - best_lin
        print(f"  {kind}: best linear-invariant AUC at a={aT} = {best_lin:.3f} (drop from clean {drop:+.3f}). "
              f"{'ROOM for kernel/local (linear degrades)' if best_lin < 0.9 else 'linear HOLDS -> pivot to attribution'}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
