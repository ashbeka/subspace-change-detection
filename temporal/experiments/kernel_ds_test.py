"""KERNEL / NONLINEAR Difference Subspace (the untested variant). Implemented via Random Fourier Features (RFF)
approximating an RBF kernel: lift each pixel x -> phi(x) in R^D, then the LINEAR subspace of the lifted patch =
the kernel subspace; kernel-DS = DS magnitude between two patches' RFF-subspaces. Does a NONLINEAR subspace
escape the redundancy that the linear subspace hit everywhere? Nulls: linear-DS, correlation matrix, CVA,
patch-mean-SAM. Honest prior LOW (~15%): the H-C autoencoder (also nonlinear) already showed geometry redundant
in nonlinear feature space. Two tests: (A) real Hermiston bitemporal patches; (B) controlled distributional change.

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.kernel_ds_test
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

from temporal import subspace as ss

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "kernel_ds_test")
os.makedirs(OUT, exist_ok=True)
DRFF, KSUB = 256, 8


def median_gamma(X, rng, n=500):
    s = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    d2 = np.sum((s[:, None, :] - s[None, :, :]) ** 2, axis=2)
    med = np.median(d2[d2 > 0])
    return 1.0 / (med + 1e-9)


def make_rff(B, gamma, rng):
    Wm = np.sqrt(2 * gamma) * rng.standard_normal((B, DRFF)); bm = rng.uniform(0, 2 * np.pi, DRFF)
    return lambda X: np.sqrt(2.0 / DRFF) * np.cos(X @ Wm + bm)


def auc(y, s):
    return float(max(roc_auc_score(y, s), 1 - roc_auc_score(y, s)))


def corr_dist(P1, P2):
    return float(np.linalg.norm(np.nan_to_num(np.corrcoef(P1.T)) - np.nan_to_num(np.corrcoef(P2.T))))


def lin_ds(P1, P2):
    return float(ss.magnitude(ss.pca_subspace(P1.T, KSUB, center=True), ss.pca_subspace(P2.T, KSUB, center=True)))


def ker_ds(P1, P2, phi):
    return float(ss.magnitude(ss.pca_subspace(phi(P1).T, KSUB, center=True), ss.pca_subspace(phi(P2).T, KSUB, center=True)))


def sam_mean(P1, P2):
    m1, m2 = P1.mean(0), P2.mean(0)
    return float(np.arccos(np.clip(m1 @ m2 / (np.linalg.norm(m1) * np.linalg.norm(m2) + 1e-12), -1, 1)))


def main():
    rng = np.random.default_rng(0)
    # ---- Part A: real Hermiston bitemporal patches ----
    D = "data_hsi/ChangeDetectionDataset/Hermiston"
    X1 = [v for k, v in loadmat(os.path.join(D, "hermiston2004.mat")).items() if not k.startswith("__")][0].astype(float)
    X2 = [v for k, v in loadmat(os.path.join(D, "hermiston2007.mat")).items() if not k.startswith("__")][0].astype(float)
    GT = [v for k, v in loadmat(os.path.join(D, "rdChangesHermiston_5classes.mat")).items() if not k.startswith("__")][0]
    Hh, Ww, B = X1.shape; nb = np.linspace(0, B - 1, 100).astype(int)
    X1 = X1[:, :, nb]; X2 = X2[:, :, nb]; B = 100
    both = np.concatenate([X1.reshape(-1, B), X2.reshape(-1, B)]); mu, sd = both.mean(0), both.std(0) + 1e-6
    X1 = (X1 - mu) / sd; X2 = (X2 - mu) / sd
    phiA = make_rff(B, median_gamma(both / sd, rng), rng)
    W = 7; h = W // 2
    chg = np.argwhere(GT[h:Hh - h, h:Ww - h] > 0) + h; noc = np.argwhere(GT[h:Hh - h, h:Ww - h] == 0) + h
    k = min(400, len(chg), len(noc))
    sel = list(map(tuple, chg[rng.choice(len(chg), k, replace=False)])) + list(map(tuple, noc[rng.choice(len(noc), k, replace=False)]))
    yv = np.r_[np.ones(k), np.zeros(k)]
    res = {"A_hermiston": {m: [] for m in ["kernel_DS", "linear_DS", "corr", "CVA_center", "patchmean_SAM"]}}
    for (r, c) in sel:
        P1 = X1[r - h:r + h + 1, c - h:c + h + 1].reshape(-1, B); P2 = X2[r - h:r + h + 1, c - h:c + h + 1].reshape(-1, B)
        res["A_hermiston"]["kernel_DS"].append(ker_ds(P1, P2, phiA))
        res["A_hermiston"]["linear_DS"].append(lin_ds(P1, P2))
        res["A_hermiston"]["corr"].append(corr_dist(P1, P2))
        res["A_hermiston"]["CVA_center"].append(float(np.linalg.norm(P1[W * W // 2] - P2[W * W // 2])))
        res["A_hermiston"]["patchmean_SAM"].append(sam_mean(P1, P2))
    print("=== Part A: real Hermiston bitemporal (AUC change vs no-change) ===")
    A = {m: auc(yv, np.array(v)) for m, v in res["A_hermiston"].items()}
    for m, v in A.items():
        print(f"  {m:>15}: {v:.3f}")

    # ---- Part B: controlled distributional (mean-preserving newdir) on Salinas ----
    m = loadmat("data_hsi/salinas.mat")
    cube = [v for k, v in m.items() if not k.startswith("__") and v.ndim == 3][0].astype(float)
    gt = loadmat("data_hsi/salinas_gt.mat"); lab = [v for k, v in gt.items() if not k.startswith("__") and v.ndim == 2][0]
    Bs = cube.shape[2]; Xs = cube.reshape(-1, Bs)[:, np.linspace(0, Bs - 1, 100).astype(int)]; ys = lab.reshape(-1)
    cnt = {c: int((ys == c).sum()) for c in np.unique(ys) if c > 0}; top = sorted(cnt, key=cnt.get, reverse=True)
    poolA = Xs[ys == top[0]]; poolB = Xs[ys == top[1]]
    d = poolB.mean(0) - poolA.mean(0); d /= np.linalg.norm(d) + 1e-12
    nz = 0.1 * poolA.std(0); s_new = 2.0 * float((poolA @ d).std()); N = 49
    phiB = make_rff(100, median_gamma((poolA - poolA.mean(0)) / (poolA.std(0) + 1e-9), rng), rng)
    resB = {m: {"chg": [], "unc": []} for m in ["kernel_DS", "linear_DS", "corr", "patchmean_CVA"]}
    for tag in ("chg", "unc"):
        for _ in range(80):
            r = np.random.default_rng(rng.integers(1 << 30))
            P1 = poolA[r.choice(len(poolA), N, replace=False)] + nz * r.standard_normal((N, 100))
            base = poolA[r.choice(len(poolA), N, replace=False)].astype(float)
            if tag == "chg":
                base = base + s_new * r.standard_normal((N, 1)) * d[None, :]
            P2 = base + nz * r.standard_normal((N, 100))
            resB["kernel_DS"][tag].append(ker_ds(P1, P2, phiB))
            resB["linear_DS"][tag].append(lin_ds(P1, P2))
            resB["corr"][tag].append(corr_dist(P1, P2))
            resB["patchmean_CVA"][tag].append(float(np.linalg.norm(P1.mean(0) - P2.mean(0))))
    yb = np.r_[np.ones(80), np.zeros(80)]
    print("\n=== Part B: controlled mean-preserving distributional change (AUC) ===")
    Bres = {m: auc(yb, np.r_[resB[m]["chg"], resB[m]["unc"]]) for m in resB}
    for m, v in Bres.items():
        print(f"  {m:>15}: {v:.3f}")

    print("\n=== VERDICT ===")
    a_win = A["kernel_DS"] > max(A["linear_DS"], A["corr"], A["CVA_center"], A["patchmean_SAM"]) + 0.03
    b_win = Bres["kernel_DS"] > max(Bres["linear_DS"], Bres["corr"], Bres["patchmean_CVA"]) + 0.03
    print(f"  kernel-DS beats all nulls on Hermiston: {a_win}; on controlled distributional: {b_win}")
    print(f"  => {'KERNEL-DS POSITIVE' if (a_win or b_win) else 'kernel-DS redundant too (nonlinear lift does not escape the pattern)'}")
    json.dump({"A": A, "B": Bres}, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
