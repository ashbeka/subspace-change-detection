"""BITEMP-HSI (C1 + C5) on the REAL Hermiston bitemporal hyperspectral benchmark (242 bands, 5-class
change-TYPE GT) — the first REAL bitemporal HSI test (all prior were synthetic / Salinas class-proxy).

PROBLEM: does genuine high spectral dimensionality give a difference-subspace / canonical-angle representation
a structural edge over SAM/CVA/IR-MAD — (C1) for DETECTION (does DS−SAM AUC turn positive & grow with band
count?), and (C5) for ATTRIBUTION (does the DS basis name the change-driving bands better than per-band diff)?
Pre-registered nulls reported in every row. Prior (both research-minings + our results): DS detection does NOT
improve with dimensionality → expect the decisive closure row; the positive HOPE is attribution.

CONSTRUCTIONS (ledger):
- detection DS = per-pixel wavelength-Hankel spectral-SSA subspace (multi-dim at high band count; DS magnitude
  between date1/date2 subspaces). SAM = rank-1 limit. CVA = L2 diff. IR-MAD = established affine-invariant bar.
- attribution DS = BAND-space difference subspace between PCA(date1 changed pixels) and PCA(date2 changed
  pixels); per-band leverage = row energy of the DS basis. Null = per-band |mean(date1)-mean(date2)|.

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.hermiston_bitemporal
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.io import loadmat
import scipy.linalg as sla
from scipy.stats import chi2 as chi2dist
from sklearn.metrics import roc_auc_score

from temporal import hsi_spectral_ds as Hs
from temporal import subspace as ss

DIR = "data_hsi/ChangeDetectionDataset/Hermiston"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "hermiston_bitemporal")
os.makedirs(OUT, exist_ok=True)
W, RANK = 15, 8


def auc(y, s):
    a = roc_auc_score(y, s); return float(max(a, 1 - a))


def irmad(X1, X2, iters=6, eps=1e-3):
    N, B = X1.shape
    w = np.ones(N); chi2 = np.zeros(N)
    for _ in range(iters):
        ws = w.sum()
        m1 = (w[:, None] * X1).sum(0) / ws; m2 = (w[:, None] * X2).sum(0) / ws
        A1, A2 = X1 - m1, X2 - m2
        S11 = (w[:, None] * A1).T @ A1 / ws + eps * np.eye(B)
        S22 = (w[:, None] * A2).T @ A2 / ws + eps * np.eye(B)
        S12 = (w[:, None] * A1).T @ A2 / ws
        Mm = sla.solve(S11, S12) @ sla.solve(S22, S12.T)
        ev, V = sla.eig(Mm)
        a = V.real[:, np.argsort(ev.real)]
        b = sla.solve(S22, S12.T) @ a
        Mv = A1 @ a - A2 @ b
        chi2 = np.sum(Mv ** 2 / (Mv.var(0) + 1e-9), axis=1)
        w = 1 - chi2dist.cdf(chi2, df=B)
    return chi2


def main():
    g = lambda f: loadmat(os.path.join(DIR, f))
    X1 = [v for k, v in g("hermiston2004.mat").items() if not k.startswith("__")][0].astype(float)
    X2 = [v for k, v in g("hermiston2007.mat").items() if not k.startswith("__")][0].astype(float)
    GT = [v for k, v in g("rdChangesHermiston_5classes.mat").items() if not k.startswith("__")][0]
    H, Wd, B = X1.shape
    X1 = X1.reshape(-1, B); X2 = X2.reshape(-1, B); y = GT.reshape(-1).astype(int)
    changed = y > 0
    print(f"Hermiston: {H}x{Wd} px, {B} bands; change classes {dict(zip(*np.unique(y, return_counts=True)))}")

    # joint per-band z-score (radiometric normalization, date-independent)
    both = np.concatenate([X1, X2]); mu, sd = both.mean(0), both.std(0) + 1e-9
    X1n, X2n = (X1 - mu) / sd, (X2 - mu) / sd

    rng = np.random.default_rng(0)
    nch = np.where(changed)[0]; nno = np.where(~changed)[0]
    k = min(3000, len(nch), len(nno))
    sel = np.concatenate([rng.choice(nch, k, replace=False), rng.choice(nno, k, replace=False)])
    ych = changed[sel]
    A1, A2 = X1n[sel], X2n[sel]

    # ---- DETECTION ----
    sam = np.array([Hs.sam(A1[i], A2[i]) for i in range(len(sel))])
    cva = np.linalg.norm(A1 - A2, axis=1)
    ds = np.array([Hs.ds_score(A1[i], A2[i], W, RANK) for i in range(len(sel))])
    irm = irmad(A1, A2)
    print("\n=== DETECTION (AUC, balanced subset; direction-agnostic) ===")
    res = {"detection": {"DS_wavhankel": auc(ych, ds), "SAM": auc(ych, sam), "CVA": auc(ych, cva), "IR_MAD": auc(ych, irm)}}
    for kk, vv in res["detection"].items():
        print(f"  {kk:>12}: {vv:.3f}")

    # ---- C1: DS - SAM vs band count ----
    print("\n=== C1: DS−SAM AUC gap vs band count (does multi-dimensionality help DS?) ===")
    res["dim_sweep"] = {}
    print(f"  {'bands':>6} {'DS':>6} {'SAM':>6} {'gap':>7}")
    for Bc in [20, 40, 80, 160, B]:
        a1b, a2b = A1[:, :Bc], A2[:, :Bc]
        dsb = np.array([Hs.ds_score(a1b[i], a2b[i], W, RANK) for i in range(len(sel))])
        samb = np.array([Hs.sam(a1b[i], a2b[i]) for i in range(len(sel))])
        ad, asv = auc(ych, dsb), auc(ych, samb)
        res["dim_sweep"][Bc] = {"DS": ad, "SAM": asv, "gap": ad - asv}
        print(f"  {Bc:>6} {ad:.3f} {asv:.3f} {ad-asv:>+7.3f}")

    # ---- C5: band attribution (BAND-space DS basis leverage vs per-band diff) ----
    def band_leverage(idx):
        try:
            s1 = ss.pca_subspace(np.ascontiguousarray(X1n[idx].T), dim=10, center=False)
            s2 = ss.pca_subspace(np.ascontiguousarray(X2n[idx].T), dim=10, center=False)
            D = ss.difference_subspace(s1, s2)
            lev = (D ** 2).sum(1) if D.shape[1] else np.zeros(B)     # per-band DS-basis energy
        except np.linalg.LinAlgError:
            lev = np.zeros(B)
        return lev / (lev.sum() + 1e-12)
    all_ch = np.where(changed)[0]
    ds_lev = band_leverage(rng.choice(all_ch, min(4000, len(all_ch)), replace=False))
    pb_diff = np.abs(X1n[all_ch].mean(0) - X2n[all_ch].mean(0)); pb_diff /= pb_diff.sum() + 1e-12
    corr = float(np.corrcoef(ds_lev, pb_diff)[0, 1])
    print("\n=== C5: attribution (DS-basis band leverage vs per-band diff over changed pixels) ===")
    print(f"  DS top-5 bands:        {sorted(np.argsort(ds_lev)[-5:][::-1].tolist())}")
    print(f"  per-band-diff top-5:   {sorted(np.argsort(pb_diff)[-5:][::-1].tolist())}")
    print(f"  corr(DS-leverage, per-band-diff) = {corr:.3f}  ({'redundant' if corr > 0.8 else 'DS adds a different ranking'})")
    # per change-type band signature (do types localize to different bands?)
    print("  per change-type DS top-3 bands:")
    type_bands = {}
    for c in [cc for cc in np.unique(y) if cc > 0]:
        ci = np.where(y == c)[0]
        if len(ci) > 200:
            lev = band_leverage(rng.choice(ci, min(2000, len(ci)), replace=False))
            type_bands[int(c)] = sorted(np.argsort(lev)[-3:][::-1].tolist())
            print(f"    class {c}: {type_bands[int(c)]}")
    res["attribution"] = {"corr_ds_vs_perband": corr, "ds_top5": np.argsort(ds_lev)[-5:][::-1].tolist(),
                          "type_bands": type_bands}

    print("\n=== VERDICT ===")
    d = res["detection"]; gap_full = res["dim_sweep"][B]["gap"]
    gaps = [res["dim_sweep"][b]["gap"] for b in res["dim_sweep"]]
    grows = gaps[-1] > gaps[0]
    print(f"  C1 detection: DS {d['DS_wavhankel']:.3f} vs SAM {d['SAM']:.3f} (gap {gap_full:+.3f}); "
          f"DS−SAM grows with bands: {grows} -> {'DS HELPS at high dim' if gap_full > 0.03 and grows else 'CLOSED: DS does not beat SAM even at 242 bands (decisive real-data row)'}")
    print(f"  C5 attribution: DS gives a {'DIFFERENT' if corr < 0.8 else 'redundant'} band ranking vs per-band diff; "
          f"change-types {'localize to different bands' if len(set(map(tuple, type_bands.values()))) > 1 else 'share bands'}")
    json.dump(res, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
