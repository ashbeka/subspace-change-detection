"""DECISIVE validation of the orientation lever on REAL Hermiston spectra (semi-synthetic, Codex Ch1/Ch4).
Claim to validate: the CENTERED-covariance ORIENTATION (Grassmann eigenspace) detects a MEAN-PRESERVING
distributional change ROBUSTLY to global illumination, where (i) mean-based CD (CVA/SAM) MISSES it (mean
preserved), and (ii) full-covariance methods (SPD log-Euclidean, ||ΔΣ||_F) are FOOLED by illumination scale.
PRE-REGISTERED KILLER NULL: the CORRELATION-matrix distance is ALSO scale-invariant — if it matches orientation,
orientation is redundant. Orientation must beat it (it should be more sensitive to a LOW-RANK orientation change).

Semi-synthetic on real spectra: material A = real pixels of a Hermiston GT class. date1 = A patch. changed date2
= A + s*g_i*d (d = unit(mean_B - mean_A), g_i zero-mean → MEAN PRESERVED, covariance gains direction d). stable
date2 = resampled A (no real change). Both date2 × global illumination (1+alpha*N(0,1)). AUC(changed vs stable)
per descriptor vs illumination alpha. >=40 patches/condition; all nulls reported.

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.hermiston_orientation_validate
"""
from __future__ import annotations

import json
import os

import numpy as np
import scipy.linalg as sla
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

from temporal import subspace as ss

DIR = "data_hsi/ChangeDetectionDataset/Hermiston"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "hermiston_orientation_validate")
os.makedirs(OUT, exist_ok=True)
NB, N, KDIM, M = 50, 400, 5, 40            # bands used, pixels/patch, orientation dim, patches/condition


def descriptors(P1, P2):
    m1, m2 = P1.mean(0), P2.mean(0)
    C1 = np.cov(P1.T) + 1e-6 * np.eye(P1.shape[1]); C2 = np.cov(P2.T) + 1e-6 * np.eye(P2.shape[1])
    e1 = np.linalg.eigh(C1)[1][:, -KDIM:]; e2 = np.linalg.eigh(C2)[1][:, -KDIM:]
    d1 = np.sqrt(np.diag(C1)); d2 = np.sqrt(np.diag(C2))
    R1 = C1 / np.outer(d1, d1); R2 = C2 / np.outer(d2, d2)               # correlation (per-band scale removed)
    sam = float(np.arccos(np.clip(m1 @ m2 / (np.linalg.norm(m1) * np.linalg.norm(m2) + 1e-12), -1, 1)))
    return {"CVA": float(np.linalg.norm(m2 - m1)), "SAM_mean": sam,
            "ORIENTATION": float(ss.magnitude(e1, e2)),
            "SPD": float(np.linalg.norm(sla.logm(C1) - sla.logm(C2))),
            "cov_Frob": float(np.linalg.norm(C1 - C2)),
            "corr_dist": float(np.linalg.norm(R1 - R2))}                 # the KILLER scale-invariant null


def main():
    X = [v for k, v in loadmat(os.path.join(DIR, "hermiston2004.mat")).items() if not k.startswith("__")][0].astype(float)
    GT = [v for k, v in loadmat(os.path.join(DIR, "rdChangesHermiston_5classes.mat")).items() if not k.startswith("__")][0]
    B = X.shape[2]; X = X.reshape(-1, B); y = GT.reshape(-1)
    bands = np.linspace(0, B - 1, NB).astype(int); X = X[:, bands]        # even band subset (N>NB for stable cov)
    classes = [c for c in np.unique(y) if (y == c).sum() > N + 50]
    pools = {c: X[y == c] for c in classes}
    rng = np.random.default_rng(0)
    keys = ["CVA", "SAM_mean", "ORIENTATION", "SPD", "cov_Frob", "corr_dist"]

    def make(kind, alpha, r):
        cA = r.choice(classes); A = pools[cA][r.choice(len(pools[cA]), N, replace=False)]
        if kind == "changed":
            cB = r.choice([c for c in classes if c != cA])
            d = pools[cB].mean(0) - A.mean(0); d /= np.linalg.norm(d) + 1e-12
            s = 0.6 * np.linalg.norm(A.std(0))                            # heterogeneity strength
            P2 = A + s * r.standard_normal((N, 1)) * d[None, :]          # MEAN-PRESERVING (g zero-mean)
        else:
            P2 = pools[cA][r.choice(len(pools[cA]), N, replace=False)]    # resampled A, no real change
        illum = 1.0 + alpha * r.standard_normal()                        # global illumination nuisance
        return A, P2 * illum

    print(f"Hermiston orientation validation (semi-synthetic, NB={NB} bands, N={N}/patch, k={KDIM}); classes {classes}")
    print(f"{'alpha':>6}" + "".join(f"{k:>11}" for k in keys))
    results = {}
    for alpha in [0.0, 0.05, 0.1, 0.2]:
        sc = {k: {"changed": [], "stable": []} for k in keys}
        for kind in ("changed", "stable"):
            for _ in range(M):
                P1, P2 = make(kind, alpha, np.random.default_rng(rng.integers(1 << 30)))
                dd = descriptors(P1, P2)
                for k in keys:
                    sc[k][kind].append(dd[k])
        yv = np.r_[np.ones(M), np.zeros(M)]
        row = {}
        for k in keys:
            s = np.r_[sc[k]["changed"], sc[k]["stable"]]
            row[k] = float(max(roc_auc_score(yv, s), 1 - roc_auc_score(yv, s)))
        results[alpha] = row
        print(f"{alpha:>6}" + "".join(f"{row[k]:>11.3f}" for k in keys))

    aT = max(results)
    print("\n=== VERDICT ===")
    r0, rT = results[0.0], results[aT]
    print(f"  no nuisance (a=0): ORIENTATION={r0['ORIENTATION']:.3f}, CVA={r0['CVA']:.3f}, SAM={r0['SAM_mean']:.3f}, "
          f"SPD={r0['SPD']:.3f}, corr={r0['corr_dist']:.3f}")
    print(f"  nuisance (a={aT}):  ORIENTATION={rT['ORIENTATION']:.3f}, CVA={rT['CVA']:.3f}, SPD={rT['SPD']:.3f}, "
          f"cov_Frob={rT['cov_Frob']:.3f}, corr={rT['corr_dist']:.3f}")
    catches = r0["ORIENTATION"] > 0.85 and r0["CVA"] < 0.65               # catches mean-preserving change CVA misses
    robust = rT["ORIENTATION"] > 0.85 and rT["SPD"] < 0.7                 # survives illumination where SPD collapses
    beats_corr = rT["ORIENTATION"] > rT["corr_dist"] + 0.05               # beats the scale-invariant killer null
    print(f"  catches mean-preserving change CVA misses: {catches}")
    print(f"  robust to illumination where SPD/cov-Frob collapse: {robust}")
    print(f"  beats the correlation-matrix (scale-invariant) null: {beats_corr}")
    print(f"  => {'STRONG POSITIVE: orientation uniquely detects mean-preserving change, illumination-robust, beyond all nulls' if (catches and robust and beats_corr) else 'PARTIAL/redundant — see which null matches (likely corr_dist)'}")
    json.dump({str(k): v for k, v in results.items()}, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
