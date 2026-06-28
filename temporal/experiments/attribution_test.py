"""Codex Ch2 — contiguous-band ATTRIBUTION of change (the last qualitatively-different stone). The 'win' is a
producible, accurate, contiguous band-naming OUTPUT, not a detection AUC. Validatable via band-INJECTION:
inject a change confined to a KNOWN contiguous spectral interval into real Hermiston spectra, add noise + a
broadband illumination distractor, and measure which method recovers the true interval.

Methods (per-band attribution score; GT = bands in the injected interval):
  perband_meandiff   |Δ mean| per band                 (null; confounded by a global-gain distractor)
  perband_stddiff    |Δ std|  per band                 (variance attributor; mean-removed null)
  perband_standardized |Δ mean|/pooled_std             (stronger null)
  DS_basis           per-band energy of the difference subspace between centered date1/date2 subspaces
  DS_smooth          DS_basis with a contiguity (moving-average) prior  (the S3CCA-essence: smooth interval)
HYPOTHESIS: under noise + a broadband (global-gain) distractor, DS_basis/DS_smooth localize the injected
contiguous interval MORE accurately/contiguously (higher Average Precision) than per-band differencing —
because the low-rank difference subspace captures the dominant LOCALIZED change direction and is mean-removed
(robust to the global-gain distractor). KILLER NULL = perband_stddiff (also mean-removed). >=40 trials.

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.attribution_test
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import average_precision_score

from temporal import subspace as ss

DIR = "data_hsi/ChangeDetectionDataset/Hermiston"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "attribution_test")
os.makedirs(OUT, exist_ok=True)
N, KDIM, L, M = 200, 8, 24, 40            # pixels/patch, subspace dim, injected interval width, trials


def attributions(d1, d2):
    m1, m2 = d1.mean(0), d2.mean(0); s1, s2 = d1.std(0), d2.std(0)
    pooled = 0.5 * (s1 + s2) + 1e-9
    S1 = ss.pca_subspace(d1.T, dim=KDIM, center=True); S2 = ss.pca_subspace(d2.T, dim=KDIM, center=True)
    D = ss.difference_subspace(S1, S2)
    dsb = (D ** 2).sum(1) if D.shape[1] else np.zeros(d1.shape[1])
    win = 7; ker = np.ones(win) / win
    dssm = np.convolve(dsb, ker, mode="same")
    return {"perband_meandiff": np.abs(m2 - m1), "perband_stddiff": np.abs(s2 - s1),
            "perband_standardized": np.abs(m2 - m1) / pooled, "DS_basis": dsb, "DS_smooth": dssm}


def main():
    X = [v for k, v in loadmat(os.path.join(DIR, "hermiston2004.mat")).items() if not k.startswith("__")][0].astype(float)
    GTm = [v for k, v in loadmat(os.path.join(DIR, "rdChangesHermiston_5classes.mat")).items() if not k.startswith("__")][0]
    Hh, Ww, B = X.shape; X = X.reshape(-1, B); y = GTm.reshape(-1)
    pool = X[y == 0]                                          # abundant 'no-change' material as the spectra pool
    sd = X.std(0) + 1e-9                                      # per-band scale for noise/injection sizing
    rng = np.random.default_rng(0)
    keys = ["perband_meandiff", "perband_stddiff", "perband_standardized", "DS_basis", "DS_smooth"]

    conds = {"clean": (0.0, 0.0), "noise": (0.4, 0.0), "noise+distractor": (0.4, 0.10)}
    results = {}
    for cond, (nz, dist) in conds.items():
        ap = {k: [] for k in keys}
        for _ in range(M):
            r = np.random.default_rng(rng.integers(1 << 30))
            b0 = r.integers(10, B - L - 10); gt = np.zeros(B); gt[b0:b0 + L] = 1
            A = pool[r.choice(len(pool), N, replace=False)].astype(float)
            d1 = A + nz * sd * r.standard_normal(A.shape)
            d2 = A.copy()
            bump = sd[b0:b0 + L]                              # inject mean+variance change in [b0,b0+L]
            d2[:, b0:b0 + L] += (0.8 + 0.5 * r.standard_normal((N, 1))) * bump[None, :]
            d2 = d2 + nz * sd * r.standard_normal(A.shape)
            if dist > 0:
                d2 = d2 * (1.0 + dist * r.standard_normal())  # broadband global-gain distractor
            att = attributions(d1, d2)
            for k in keys:
                ap[k].append(float(average_precision_score(gt, att[k])))
        results[cond] = {k: float(np.mean(ap[k])) for k in keys}

    base = float(L) / B                                       # random-guess AP
    print(f"Attribution AP (recover injected {L}-band interval; random baseline AP={base:.3f}); {M} trials")
    print(f"  {'condition':>18}" + "".join(f"{k:>21}" for k in keys))
    for cond, row in results.items():
        print(f"  {cond:>18}" + "".join(f"{row[k]:>21.3f}" for k in keys))

    print("\n=== VERDICT (does DS attribution beat per-band differencing under noise+distractor?) ===")
    nd = results["noise+distractor"]
    best_null = max(nd["perband_meandiff"], nd["perband_stddiff"], nd["perband_standardized"])
    best_ds = max(nd["DS_basis"], nd["DS_smooth"])
    print(f"  noise+distractor: best DS={best_ds:.3f}  best per-band null={best_null:.3f}  gap={best_ds-best_null:+.3f}")
    print(f"  => {'POSITIVE — subspace attribution localizes the interval better under nuisance' if best_ds > best_null + 0.05 else 'redundant — per-band differencing matches/beats DS attribution'}")
    json.dump(results, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
