"""Material-signature subspace change detection on hyperspectral (Salinas 204-band) — Paper B, the live
positive bet. A MATERIAL is a SUBSPACE (the span of its spectral signatures, capturing intra-material
variation), not a point. Detect MATERIAL change (class transition) robustly to illumination/per-band nuisance,
where per-pixel SAM/CVA are fooled. Tests the lab's MSM/GDS essence smartly adapted to HSI CD.

CONSTRUCTION (ledger): per-class material SUBSPACE = top-K uncentered PCA of that class's training spectra
(D=204-dim, so genuinely multi-dim → DS≠SAM). Common (across-material) subspace C = top-RC PCA of the class
MEAN spectra; the GDS = the orthogonal complement of C (removes the shared spectral shape ≈ illumination/common
nuisance), the Fukui-Maki discriminative-subspace essence. CD score = membership-vector distance
||a(date2)-a(date1)||, a = per-class angle profile (nuisance-robust methods keep unchanged pixels' profile put).

Methods: CVA (raw |x1-x2|, naive null) · SAM_centroid (angle to class MEANS — the key null) · MSM (angle to
class SUBSPACES) · GDS_centroid (SAM after removing C) · GDS_MSM (MSM in GDS space).
HYPOTHESIS: as nuisance α grows, GDS/MSM stay robust while SAM_centroid degrades and CVA collapses. If
MSM≈GDS≈SAM_centroid → the subspace adds nothing (diagnostic). Pre-registered; nulls always reported.

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.material_subspace_cd
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

from temporal.subspace import pca_subspace

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "material_subspace_cd")
os.makedirs(OUT, exist_ok=True)
K = 6        # class material-subspace dim
RC = 3       # common-subspace dim removed by the GDS


def sam_vec(X, m):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    mn = m / (np.linalg.norm(m) + 1e-12)
    return np.arccos(np.clip(Xn @ mn, -1.0, 1.0))


def angle_to_subspace(X, B):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cos = np.linalg.norm(Xn @ B, axis=1)
    return np.arccos(np.clip(cos, 0.0, 1.0))


def apply_nuisance(X, alpha, rng):
    B = X.shape[1]
    gain = 1.0 + alpha * rng.standard_normal(B)
    offset = alpha * rng.standard_normal(B) * X.std()
    illum = 1.0 + alpha * rng.standard_normal()
    return X * gain * illum + offset


def auc(y, s):
    a = roc_auc_score(y, s)
    return float(max(a, 1 - a))


def main():
    m = loadmat("data_hsi/salinas.mat")
    cube = [v for k, v in m.items() if not k.startswith("__") and v.ndim == 3][0].astype(float)
    gt = loadmat("data_hsi/salinas_gt.mat")
    lab = [v for k, v in gt.items() if not k.startswith("__") and v.ndim == 2][0]
    D = cube.shape[2]; X = cube.reshape(-1, D); y = lab.reshape(-1)
    classes = [c for c in np.unique(y) if c > 0]
    rng = np.random.default_rng(0)

    train, test = {}, {}
    for c in classes:
        idx = np.where(y == c)[0].copy(); rng.shuffle(idx); h = len(idx) // 2
        train[c], test[c] = idx[:h], idx[h:]
    means = {c: X[train[c]].mean(0) for c in classes}
    subs = {c: pca_subspace(X[train[c]].T, dim=K, center=False) for c in classes}
    Mmat = np.stack([means[c] / (np.linalg.norm(means[c]) + 1e-12) for c in classes], axis=1)   # (D, nclass)
    C = pca_subspace(Mmat, dim=RC, center=False)                                                 # common subspace
    gds = lambda Z: Z - (Z @ C) @ C.T
    subs_g = {}
    for c in classes:
        Bg = subs[c] - C @ (C.T @ subs[c]); Bg, _ = np.linalg.qr(Bg); subs_g[c] = Bg
    means_g = {c: gds(means[c][None, :])[0] for c in classes}

    # bitemporal test scene: 15% genuine material (class) change
    sel = rng.choice(np.concatenate([test[c] for c in classes]), 4000, replace=False)
    X1 = X[sel].copy(); y1 = y[sel]
    changed = np.zeros(len(sel), bool); changed[rng.choice(len(sel), int(0.15 * len(sel)), replace=False)] = True
    X2c = X1.copy()
    for i in np.where(changed)[0]:
        cj = rng.choice([c for c in classes if c != y1[i]])
        X2c[i] = X[rng.choice(test[cj])]

    def membership(Z, mode):
        if mode == "sam":
            return np.stack([sam_vec(Z, means[c]) for c in classes], axis=1)
        if mode == "msm":
            return np.stack([angle_to_subspace(Z, subs[c]) for c in classes], axis=1)
        if mode == "gds_sam":
            Zg = gds(Z); return np.stack([sam_vec(Zg, means_g[c]) for c in classes], axis=1)
        if mode == "gds_msm":
            Zg = gds(Z); return np.stack([angle_to_subspace(Zg, subs_g[c]) for c in classes], axis=1)

    print(f"Salinas material-subspace CD: AUC(detect material change) vs nuisance; K={K} RC={RC}")
    print(f"{'alpha':>6} {'CVA':>6} {'SAM_cen':>8} {'MSM':>6} {'GDS_cen':>8} {'GDS_MSM':>8}")
    results = {}
    for alpha in [0.0, 0.1, 0.2, 0.4]:
        X2 = apply_nuisance(X2c, alpha, np.random.default_rng(42))
        row = {"CVA": auc(changed, np.linalg.norm(X1 - X2, axis=1))}
        for mode in ["sam", "msm", "gds_sam", "gds_msm"]:
            a1, a2 = membership(X1, mode), membership(X2, mode)
            row[{"sam": "SAM_cen", "msm": "MSM", "gds_sam": "GDS_cen", "gds_msm": "GDS_MSM"}[mode]] = \
                auc(changed, np.linalg.norm(a2 - a1, axis=1))
        results[alpha] = row
        print(f"{alpha:>6} {row['CVA']:>6.3f} {row['SAM_cen']:>8.3f} {row['MSM']:>6.3f} "
              f"{row['GDS_cen']:>8.3f} {row['GDS_MSM']:>8.3f}")

    aT = max(results)
    print("\n=== VERDICT (does the SUBSPACE beat the MEAN under nuisance?) ===")
    r0, rT = results[min(results)], results[aT]
    print(f"  clean: SAM_cen={r0['SAM_cen']:.3f} MSM={r0['MSM']:.3f} GDS_cen={r0['GDS_cen']:.3f} GDS_MSM={r0['GDS_MSM']:.3f}")
    print(f"  nuisance(a={aT}): SAM_cen={rT['SAM_cen']:.3f} MSM={rT['MSM']:.3f} GDS_cen={rT['GDS_cen']:.3f} GDS_MSM={rT['GDS_MSM']:.3f}")
    best_sub = max(rT['MSM'], rT['GDS_cen'], rT['GDS_MSM'])
    win = best_sub > rT['SAM_cen'] + 0.03
    print(f"  best subspace method beats SAM-centroid under nuisance by {best_sub - rT['SAM_cen']:+.3f} -> "
          f"{'SUBSPACE HELPS' if win else 'no real subspace edge (diagnostic)'}")
    json.dump({str(k): v for k, v in results.items()}, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
