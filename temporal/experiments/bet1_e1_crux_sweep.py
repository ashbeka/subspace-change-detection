"""BET 1 / E1 — THE CRUX SWEEP. Does a low-rank SET-SUBSPACE comparison beat the regularized N<<B competitors
(shrinkage-covariance / shrinkage-correlation, log-Euclidean SPD) for SUBTLE distributional change, in the
small-sample high-dim regime where the sample covariance is rank-deficient? (Per-pixel/patch-mean are weaker
nulls handled in E2.) Pre-registration: docs/research/bet1_design.md. Falsifier: if SET-SUBSPACE never beats
shrinkage at small N (its predicted home), Bet 1 dies.

Semi-synthetic on REAL Salinas spectra (a real material's covariance). Sweep N (pixels/patch) at fixed B.
Change types: (newdir) mean-PRESERVING new-direction loading [g_i zero-mean]; (mix) sub-pixel mixing (replace
fraction f of pixels with material B). Subtle magnitude. Nuisance: global illumination. AUC(changed vs unchanged).

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.bet1_e1_crux_sweep
"""
from __future__ import annotations

import json
import os

import numpy as np
import scipy.linalg as sla
from scipy.io import loadmat
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

from temporal import subspace as ss

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "bet1_e1_crux_sweep")
os.makedirs(OUT, exist_ok=True)
NB, M, NS = 100, 60, [9, 16, 25, 49, 100, 200]
KEYS = ["patch_mean_CVA", "patch_mean_SAM", "sample_corr", "shrink_corr", "shrink_cov_SPD",
        "SET_SUBSPACE_DS", "SET_SUBSPACE_ang"]


def corr_from_cov(C):
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None)); return C / np.outer(d, d)


def scores(P1, P2, k):
    m1, m2 = P1.mean(0), P2.mean(0)
    sam = float(np.arccos(np.clip(m1 @ m2 / (np.linalg.norm(m1) * np.linalg.norm(m2) + 1e-12), -1, 1)))
    R1s = np.nan_to_num(np.corrcoef(P1.T)); R2s = np.nan_to_num(np.corrcoef(P2.T))
    C1 = LedoitWolf().fit(P1).covariance_; C2 = LedoitWolf().fit(P2).covariance_
    Rl1, Rl2 = corr_from_cov(C1), corr_from_cov(C2)
    L1 = sla.logm(C1).real; L2 = sla.logm(C2).real
    S1 = ss.pca_subspace(P1.T, dim=k, center=True); S2 = ss.pca_subspace(P2.T, dim=k, center=True)
    cos = ss.canonical_cosines(S1, S2)
    return {"patch_mean_CVA": float(np.linalg.norm(m1 - m2)), "patch_mean_SAM": sam,
            "sample_corr": float(np.linalg.norm(R1s - R2s)),
            "shrink_corr": float(np.linalg.norm(Rl1 - Rl2)),
            "shrink_cov_SPD": float(np.linalg.norm(L1 - L2)),
            "SET_SUBSPACE_DS": float(ss.magnitude(S1, S2)),
            "SET_SUBSPACE_ang": float(np.mean(1 - cos))}


def main():
    m = loadmat("data_hsi/salinas.mat")
    cube = [v for k, v in m.items() if not k.startswith("__") and v.ndim == 3][0].astype(float)
    gt = loadmat("data_hsi/salinas_gt.mat")
    lab = [v for k, v in gt.items() if not k.startswith("__") and v.ndim == 2][0]
    B = cube.shape[2]; X = cube.reshape(-1, B); y = lab.reshape(-1)
    cnt = {c: int((y == c).sum()) for c in np.unique(y) if c > 0}
    top = sorted(cnt, key=cnt.get, reverse=True)
    cA, cB = top[0], top[1]
    bands = np.linspace(0, B - 1, NB).astype(int)
    poolA = X[y == cA][:, bands]; poolB = X[y == cB][:, bands]
    d = poolB.mean(0) - poolA.mean(0); d /= np.linalg.norm(d) + 1e-12
    nzvec = 0.10 * poolA.std(0)                                        # per-band additive noise
    proj_std = float((poolA @ d).std())                               # natural spread along d
    s_new = 2.0 * proj_std                                             # rank-1 addition = 2x natural spread
    f_mix = 0.5                                                        # mixing fraction (discriminable)
    rng = np.random.default_rng(0)
    print(f"Salinas N<<B crux sweep: B={NB} bands, classes A={cA}(|{cnt[cA]}|) B={cB}; "
          f"newdir s={s_new:.1f} (2x along-d spread {proj_std:.1f}), mix f={f_mix}")

    results = {}
    for ct in ["newdir", "mix"]:
        for alpha in [0.0, 0.1]:
            print(f"\n=== change={ct}  illum_alpha={alpha} ===")
            print(f"  {'N':>4}" + "".join(f"{k:>17}" for k in KEYS))
            results[f"{ct}_a{alpha}"] = {}
            for Nv in NS:
                k = min(8, Nv - 1)
                sc = {kk: {"chg": [], "unc": []} for kk in KEYS}
                for tag in ("chg", "unc"):
                    for _ in range(M):
                        r = np.random.default_rng(rng.integers(1 << 30))
                        P1 = poolA[r.choice(len(poolA), Nv, replace=False)] + nzvec * r.standard_normal((Nv, NB))
                        base = poolA[r.choice(len(poolA), Nv, replace=False)].astype(float)
                        if tag == "chg":
                            if ct == "newdir":
                                base = base + s_new * r.standard_normal((Nv, 1)) * d[None, :]   # mean-preserving
                            else:
                                nrep = max(1, int(f_mix * Nv))
                                base[r.choice(Nv, nrep, replace=False)] = poolB[r.choice(len(poolB), nrep, replace=False)]
                        P2 = (base + nzvec * r.standard_normal((Nv, NB))) * (1.0 + alpha * r.standard_normal())
                        dd = scores(P1, P2, k)
                        for kk in KEYS:
                            sc[kk][tag].append(dd[kk])
                yv = np.r_[np.ones(M), np.zeros(M)]
                row = {}
                for kk in KEYS:
                    v = np.r_[sc[kk]["chg"], sc[kk]["unc"]]
                    row[kk] = float(max(roc_auc_score(yv, v), 1 - roc_auc_score(yv, v)))
                results[f"{ct}_a{alpha}"][Nv] = row
                print(f"  {Nv:>4}" + "".join(f"{row[kk]:>17.3f}" for kk in KEYS))

    print("\n=== VERDICT (does SET-SUBSPACE beat shrinkage at small N, its predicted home?) ===")
    for cond, byN in results.items():
        Nsmall = NS[0]
        r = byN[Nsmall]
        sub = max(r["SET_SUBSPACE_DS"], r["SET_SUBSPACE_ang"])
        shr = max(r["shrink_corr"], r["shrink_cov_SPD"])
        print(f"  {cond} @N={Nsmall}: SET={sub:.3f} vs shrink={shr:.3f} -> {'SUBSPACE WINS' if sub > shr + 0.03 else 'shrinkage matches/wins'}")
    json.dump({k: {str(n): r for n, r in v.items()} for k, v in results.items()},
              open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
