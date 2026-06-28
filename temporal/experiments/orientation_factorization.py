"""Codex Challenge 1 — FACTORIZED local spectral-distribution change: mean | dispersion(scale) | covariance
ORIENTATION. The structural lever for geometry is the ORIENTATION component: the top-k eigenspace of the
CENTERED covariance is a Grassmann point, invariant to mean (centering) and to isotropic eigenvalue scale —
so it isolates an orientation-only change that a full covariance / SPD distance CONFLATES with scale.

This is a CHARACTERIZATION/interpretability contribution, NOT a detection-AUC claim: the win is a producible,
scale-invariant factorized output (which component moved), not beating a detector. Decision gate (Codex):
the orientation descriptor must isolate an orientation-ONLY change (matched mean AND eigenvalues) where the
SPD/Frobenius covariance distance cannot disentangle it from a scale change.

Regimes (local patch of N spectra, date1->date2): mean-only | scale-only (eigenvalues x s, eigenvectors fixed) |
orient-only (eigenVECTORS rotated, eigenVALUES fixed, mean fixed) | stable. Descriptors: mean-shift; dispersion
|Δtr Σ|; ORIENTATION = DS magnitude between top-k centered-cov eigenspaces (Grassmann); SPD = log-Euclidean
covariance distance (null); Frobenius ||ΔΣ|| (null). >=8 seeds; nulls always reported.

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.orientation_factorization
"""
from __future__ import annotations

import json
import os

import numpy as np
import scipy.linalg as sla
from sklearn.metrics import roc_auc_score

from temporal import subspace as ss

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "orientation_factorization")
os.makedirs(OUT, exist_ok=True)
B, N, KDIM, SEEDS = 10, 300, 3, 40
REGIMES = ["mean", "scale", "orient", "stable"]


def rand_cov(rng):
    V, _ = np.linalg.qr(rng.standard_normal((B, B)))
    lam = np.sort(rng.uniform(0.3, 3.0, B))[::-1]
    return V, lam


def patch(mu, V, lam, rng):
    X = rng.standard_normal((N, B)) * np.sqrt(lam)            # diagonal in eigenbasis
    return X @ V.T + mu                                        # rotate to ambient + shift


def descriptors(P1, P2):
    m1, m2 = P1.mean(0), P2.mean(0)
    C1 = np.cov(P1.T) + 1e-6 * np.eye(B); C2 = np.cov(P2.T) + 1e-6 * np.eye(B)
    e1 = np.linalg.eigh(C1)[1][:, -KDIM:]; e2 = np.linalg.eigh(C2)[1][:, -KDIM:]   # top-k centered eigenspace
    orient = ss.magnitude(e1, e2)                             # Grassmann angle (scale + mean invariant)
    spd = float(np.linalg.norm(sla.logm(C1) - sla.logm(C2)))  # log-Euclidean SPD distance (null)
    return {"mean_shift": float(np.linalg.norm(m2 - m1)),
            "dispersion": float(abs(np.trace(C2) - np.trace(C1))),
            "ORIENTATION": float(orient), "SPD": spd,
            "Frobenius": float(np.linalg.norm(C1 - C2))}


def main():
    rng = np.random.default_rng(0)
    data = {r: [] for r in REGIMES}
    for r in REGIMES:
        for _ in range(SEEDS):
            rr = np.random.default_rng(rng.integers(1 << 30))
            mu = rr.standard_normal(B) * 2; V, lam = rand_cov(rr)
            P1 = patch(mu, V, lam, rr)
            if r == "mean":
                P2 = patch(mu + rr.standard_normal(B) * 1.2, V, lam, rr)
            elif r == "scale":
                P2 = patch(mu, V, lam * 2.2, rr)                       # eigenvalues scaled, eigenvectors FIXED
            elif r == "orient":
                Vp, _ = np.linalg.qr(V + 0.6 * rr.standard_normal((B, B)))  # eigenVECTORS rotated
                P2 = patch(mu, Vp, lam, rr)                            # eigenVALUES fixed, mean fixed
            else:
                P2 = patch(mu, V, lam, rr)
            data[r].append(descriptors(P1, P2))
    keys = list(data["mean"][0].keys())
    arr = {r: {k: np.array([d[k] for d in data[r]]) for k in keys} for r in REGIMES}

    def auc(pos, neg, k):
        y = np.r_[np.ones(SEEDS), np.zeros(SEEDS)]; s = np.r_[arr[pos][k], arr[neg][k]]
        return float(max(roc_auc_score(y, s), 1 - roc_auc_score(y, s)))

    print("Each descriptor's AUC for detecting each change-type vs stable:")
    print(f"  {'descriptor':>12}" + "".join(f"{r:>9}" for r in ["mean", "scale", "orient"]))
    res = {}
    for k in keys:
        res[k] = {r: auc(r, "stable", k) for r in ["mean", "scale", "orient"]}
        print(f"  {k:>12}" + "".join(f"{res[k][r]:>9.3f}" for r in ["mean", "scale", "orient"]))

    print("\n=== THE FACTORIZATION TEST: disentangle ORIENTATION-only from SCALE-only ===")
    print(f"  {'descriptor':>12} {'AUC(orient vs scale)':>22}")
    diseng = {}
    for k in keys:
        diseng[k] = auc("orient", "scale", k)
        print(f"  {k:>12} {diseng[k]:>22.3f}")

    print("\n=== VERDICT ===")
    o, spd, fro = diseng["ORIENTATION"], diseng["SPD"], diseng["Frobenius"]
    # orientation must FIRE on orient-change, IGNORE scale-change, and DISENTANGLE where SPD conflates
    scale_inv = res["ORIENTATION"]["scale"] < 0.65          # orientation ignores scale (scale-invariant)
    fires = res["ORIENTATION"]["orient"] > 0.85             # orientation catches orientation change
    uniq = o > 0.8 and spd < 0.65                           # disentangles orient-vs-scale where SPD cannot
    print(f"  ORIENTATION: fires on orient={res['ORIENTATION']['orient']:.3f}, ignores scale={res['ORIENTATION']['scale']:.3f} (scale-invariant={scale_inv})")
    print(f"  disentangle orient-vs-scale: ORIENTATION={o:.3f} vs SPD={spd:.3f} vs Frobenius={fro:.3f}")
    print(f"  => {'FACTORIZATION WORKS — geometry uniquely isolates scale-invariant orientation; SPD conflates it with scale' if (scale_inv and fires and uniq) else 'NO unique factorization (SPD/Frobenius disentangle too, or orientation not scale-invariant)'}")
    json.dump({"detect": res, "disentangle": diseng}, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
