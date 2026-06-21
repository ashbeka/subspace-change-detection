"""BET 1 / E2 — the REGISTRATION pillar. On the real spatial Salinas image, with actual pixel-shift
misregistration: do per-pixel methods false-alarm under misregistration while patch/set methods stay robust,
and does the SET-SUBSPACE beat the simpler patch methods (mean / shrinkage-correlation)? Pre-reg: bet1_design §4.

Both 'change' and 'no-change' date2 patches carry the SAME misregistration shift; only 'change' also has an
injected distributional change. AUC(change vs no-change) measures: can the method detect the change ON TOP of
the misregistration? Registration-robust methods keep AUC under shift; per-pixel collapses (shift swamps signal).

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.bet1_e2_registration
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.io import loadmat
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

from temporal import subspace as ss

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "bet1_e2_registration")
os.makedirs(OUT, exist_ok=True)
W, NB, M = 7, 100, 250
KEYS = ["per_pixel_CVA", "per_pixel_SAM", "patch_mean_CVA", "patch_mean_SAM", "shrink_corr", "SET_SUBSPACE"]


def corr_from_cov(C):
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None)); return C / np.outer(d, d)


def scores(P1, P2, k):
    c = (W * W) // 2
    m1, m2 = P1.mean(0), P2.mean(0)
    sam_m = float(np.arccos(np.clip(m1 @ m2 / (np.linalg.norm(m1) * np.linalg.norm(m2) + 1e-12), -1, 1)))
    pp_sam = float(np.arccos(np.clip(P1[c] @ P2[c] / (np.linalg.norm(P1[c]) * np.linalg.norm(P2[c]) + 1e-12), -1, 1)))
    Rl1 = corr_from_cov(LedoitWolf().fit(P1).covariance_); Rl2 = corr_from_cov(LedoitWolf().fit(P2).covariance_)
    S1 = ss.pca_subspace(P1.T, dim=k, center=True); S2 = ss.pca_subspace(P2.T, dim=k, center=True)
    return {"per_pixel_CVA": float(np.linalg.norm(P1[c] - P2[c])), "per_pixel_SAM": pp_sam,
            "patch_mean_CVA": float(np.linalg.norm(m1 - m2)), "patch_mean_SAM": sam_m,
            "shrink_corr": float(np.linalg.norm(Rl1 - Rl2)), "SET_SUBSPACE": float(ss.magnitude(S1, S2))}


def main():
    m = loadmat("data_hsi/salinas.mat")
    cube = [v for k, v in m.items() if not k.startswith("__") and v.ndim == 3][0].astype(float)
    gt = loadmat("data_hsi/salinas_gt.mat")
    lab = [v for k, v in gt.items() if not k.startswith("__") and v.ndim == 2][0]
    Hh, Ww, B = cube.shape
    bands = np.linspace(0, B - 1, NB).astype(int); cube = cube[:, :, bands]
    y = lab.reshape(-1)
    cnt = {c: int((y == c).sum()) for c in np.unique(y) if c > 0}
    top = sorted(cnt, key=cnt.get, reverse=True)
    poolA = cube.reshape(-1, NB)[lab.reshape(-1) == top[0]]
    poolB = cube.reshape(-1, NB)[lab.reshape(-1) == top[1]]
    d = poolB.mean(0) - poolA.mean(0); d /= np.linalg.norm(d) + 1e-12
    nzvec = 0.10 * poolA.std(0); k = min(8, W * W - 1)
    rng = np.random.default_rng(0)
    smax = 2
    locs = [(r, c) for r in range(2, Hh - W - smax - 2) for c in range(2, Ww - W - 2)]
    rng.shuffle(locs); locs = locs[:M * 3]
    print(f"Salinas spatial E2: {W}x{W}={W*W}px patches, B={NB}, misregistration test; {len(locs)} candidate locs")

    results = {}
    for sh in [0, 2]:
        for alpha in [0.0, 0.1]:
            sc = {kk: {"chg": [], "noc": []} for kk in KEYS}
            for (r, c) in locs[:M]:
                P1 = cube[r:r + W, c:c + W].reshape(-1, NB)
                base = cube[r + sh:r + sh + W, c:c + W].reshape(-1, NB).astype(float)   # misregistered date2
                # no-change
                P2n = (base + nzvec * rng.standard_normal((W * W, NB))) * (1 + alpha * rng.standard_normal())
                # change: inject sub-pixel mix (clear-ish distributional change)
                bc = base.copy(); nrep = int(0.4 * W * W)
                bc[rng.choice(W * W, nrep, replace=False)] = poolB[rng.choice(len(poolB), nrep, replace=False)]
                P2c = (bc + nzvec * rng.standard_normal((W * W, NB))) * (1 + alpha * rng.standard_normal())
                for tag, P2 in [("chg", P2c), ("noc", P2n)]:
                    dd = scores(P1, P2, k)
                    for kk in KEYS:
                        sc[kk][tag].append(dd[kk])
            yv = np.r_[np.ones(M), np.zeros(M)]
            row = {kk: float(max(roc_auc_score(yv, np.r_[sc[kk]["chg"], sc[kk]["noc"]]),
                                 1 - roc_auc_score(yv, np.r_[sc[kk]["chg"], sc[kk]["noc"]]))) for kk in KEYS}
            results[f"sh{sh}_a{alpha}"] = row

    print(f"\n  {'condition':>12}" + "".join(f"{k:>16}" for k in KEYS))
    for cond, row in results.items():
        print(f"  {cond:>12}" + "".join(f"{row[k]:>16.3f}" for k in KEYS))

    print("\n=== VERDICT ===")
    clean = results["sh0_a0.0"]; misreg = results["sh2_a0.1"]
    pp_drop = clean["per_pixel_CVA"] - misreg["per_pixel_CVA"]
    best_patch = max(misreg["patch_mean_CVA"], misreg["patch_mean_SAM"], misreg["shrink_corr"])
    print(f"  per-pixel CVA drop under misreg+illum: {pp_drop:+.3f} (clean {clean['per_pixel_CVA']:.3f} -> {misreg['per_pixel_CVA']:.3f})")
    print(f"  under misreg+illum: best patch null={best_patch:.3f}  SET_SUBSPACE={misreg['SET_SUBSPACE']:.3f}")
    print(f"  SET beats best patch null under misreg: {misreg['SET_SUBSPACE'] > best_patch + 0.03}")
    print(f"  (registration robustness is SHARED by patch methods; subspace must beat patch-mean/shrink-corr to matter)")
    json.dump(results, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
