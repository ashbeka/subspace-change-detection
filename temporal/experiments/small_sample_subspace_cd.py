"""THE TWIST — small-sample (N<<B) set-subspace change detection on REAL Hermiston, under misregistration +
illumination. Every prior loss was at N>>B, where a full covariance/correlation beats a low-rank subspace.
The REAL hyperspectral regime is N<<B (a 5x5 patch = 25 pixels, 242 bands): there the sample covariance is
rank-deficient/noisy, SPD breaks, per-pixel methods false-alarm under misregistration — and a low-rank
SET-SUBSPACE is the well-posed, denoised, permutation-invariant (registration-robust), scale-invariant
(illumination-robust) tool. This is the regime the lab's set-as-subspace methods are built for, and the one
I never tested.

CLAIM: under realistic nuisance (1-px misregistration + global illumination), set-subspace CD separates real
change from no-change BETTER than per-pixel CVA (registration-fooled), patch-mean (misses distribution / illum),
and the sample correlation-matrix distance (the killer null — but NOISY at N<<B). Real change + real no-change
from Hermiston GT; nuisance applied to both. AUC(change vs no-change) across (misreg, illum) conditions.

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.small_sample_subspace_cd
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

from temporal import subspace as ss

DIR = "data_hsi/ChangeDetectionDataset/Hermiston"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "small_sample_subspace_cd")
os.makedirs(OUT, exist_ok=True)
WIN, KDIM, NLOC = 5, 8, 400          # 5x5=25 px patch, subspace dim, locations/class


def scores(W1, W2):
    c = WIN * WIN // 2
    m1, m2 = W1.mean(0), W2.mean(0)
    sam_m = float(np.arccos(np.clip(m1 @ m2 / (np.linalg.norm(m1) * np.linalg.norm(m2) + 1e-12), -1, 1)))
    S1 = ss.pca_subspace(W1.T, dim=KDIM, center=False); S2 = ss.pca_subspace(W2.T, dim=KDIM, center=False)
    sub = ss.magnitude(S1, S2)                                       # set-subspace canonical-angle distance
    R1 = np.corrcoef(W1.T); R2 = np.corrcoef(W2.T)                   # sample correlation (noisy at N<<B)
    corr = float(np.linalg.norm(np.nan_to_num(R1) - np.nan_to_num(R2)))
    return {"per_pixel_CVA": float(np.linalg.norm(W1[c] - W2[c])),
            "patch_mean_CVA": float(np.linalg.norm(m1 - m2)), "patch_mean_SAM": sam_m,
            "SET_SUBSPACE": sub, "corr_Frob": corr}


def main():
    X1 = [v for k, v in loadmat(os.path.join(DIR, "hermiston2004.mat")).items() if not k.startswith("__")][0].astype(float)
    X2 = [v for k, v in loadmat(os.path.join(DIR, "hermiston2007.mat")).items() if not k.startswith("__")][0].astype(float)
    GT = [v for k, v in loadmat(os.path.join(DIR, "rdChangesHermiston_5classes.mat")).items() if not k.startswith("__")][0]
    Hh, Ww, B = X1.shape
    h = WIN // 2
    rng = np.random.default_rng(0)

    def sample_locs(mask, n):
        rs, cs = np.where(mask[h + 1:Hh - h - 1, h + 1:Ww - h - 1])
        idx = rng.choice(len(rs), min(n, len(rs)), replace=False)
        return list(zip(rs[idx] + h + 1, cs[idx] + h + 1))
    chg = sample_locs(GT > 0, NLOC); noc = sample_locs(GT == 0, NLOC)
    print(f"N<<B regime: patch {WIN}x{WIN}={WIN*WIN} px, B={B} bands; {len(chg)} change + {len(noc)} no-change locs")

    keys = ["per_pixel_CVA", "patch_mean_CVA", "patch_mean_SAM", "SET_SUBSPACE", "corr_Frob"]
    results = {}
    for sh in [0, 1]:                                                # misregistration shift (rows)
        for alpha in [0.0, 0.1]:                                     # global illumination nuisance
            data = {"chg": [], "noc": []}
            for tag, locs in [("chg", chg), ("noc", noc)]:
                for (r, c) in locs:
                    W1 = X1[r - h:r + h + 1, c - h:c + h + 1].reshape(-1, B)
                    W2 = X2[r - h + sh:r + h + 1 + sh, c - h:c + h + 1].reshape(-1, B)
                    W2 = W2 * (1.0 + alpha * rng.standard_normal())  # illumination on date2
                    data[tag].append(scores(W1, W2))
            y = np.r_[np.ones(len(chg)), np.zeros(len(noc))]
            row = {}
            for kk in keys:
                s = np.r_[[d[kk] for d in data["chg"]], [d[kk] for d in data["noc"]]]
                row[kk] = float(max(roc_auc_score(y, s), 1 - roc_auc_score(y, s)))
            results[f"sh{sh}_a{alpha}"] = row

    print(f"\n  {'condition':>16}" + "".join(f"{k:>15}" for k in keys))
    for cond, row in results.items():
        print(f"  {cond:>16}" + "".join(f"{row[k]:>15.3f}" for k in keys))

    print("\n=== VERDICT (does SET-SUBSPACE win in the realistic N<<B + nuisance regime?) ===")
    clean = results["sh0_a0.0"]; nuis = results["sh1_a0.1"]
    sub_drop = clean["SET_SUBSPACE"] - nuis["SET_SUBSPACE"]
    pp_drop = clean["per_pixel_CVA"] - nuis["per_pixel_CVA"]
    best_null_nuis = max(nuis["per_pixel_CVA"], nuis["patch_mean_CVA"], nuis["patch_mean_SAM"], nuis["corr_Frob"])
    win = nuis["SET_SUBSPACE"] > best_null_nuis + 0.03
    print(f"  clean(sh0,a0):   SET={clean['SET_SUBSPACE']:.3f}  per-pix-CVA={clean['per_pixel_CVA']:.3f}  corr={clean['corr_Frob']:.3f}")
    print(f"  nuisance(sh1,a.1): SET={nuis['SET_SUBSPACE']:.3f}  per-pix-CVA={nuis['per_pixel_CVA']:.3f}  corr={nuis['corr_Frob']:.3f}")
    print(f"  SET robust (drop {sub_drop:+.3f}) vs per-pixel (drop {pp_drop:+.3f}); SET beats best null under nuisance: {win}")
    print(f"  => {'POSITIVE — set-subspace is the robust well-posed CD in the N<<B + nuisance regime' if win else 'no unique win (see which null matches under nuisance)'}")
    json.dump(results, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
