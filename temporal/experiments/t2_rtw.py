"""T2 go/no-go — RTW (Randomized Time Warping) warp-invariant subspace for phenological-phase-invariant CD.
The last positive bet (research-mining: highest novelty, RTW never used in RS/CD, Sensei-endorsed).

DECISIVE QUESTION: can a warp-invariant RTW subspace tell a NUISANCE (a nonlinear time-warp of the seasonal
cycle — same shape, distorted timing) from a REAL change (a same-phase cycle-SHAPE change), where the standard
phase-aware null CANNOT? A constant phase shift is handled by phase-aligned matching; the decisive nuisance is a
NONLINEAR monotonic warp that a single phase term cannot remove. If RTW separates warp-vs-shape >> phase-aligned
matching, warp-invariance is a genuine lever the scalar/harmonic null lacks (GO). If phase-aligned matches it,
the lever collapses to deseasonalization (NO-GO). DTW = the warp-robust competitor RTW must at least match.

CONSTRUCTION (ledger): RTW subspace = top-r left singular vectors of a matrix whose columns are K random
MONOTONIC time-warps of the series (the warp-orbit of the shape); two series compared by DS magnitude between
their RTW subspaces (warp-invariant by construction). Pre-registered; nulls always reported; >=8 seeds.

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.t2_rtw
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from temporal import subspace as ss

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "t2_rtw")
os.makedirs(OUT, exist_ok=True)
T, K, R, SEEDS = 60, 80, 3, 40


def base_cycle(rng, h2=0.4, h3=0.0):
    t = np.linspace(0, 1, T)
    return (np.sin(2 * np.pi * t) + h2 * np.sin(4 * np.pi * t + rng.uniform(0, 0.3))
            + h3 * np.sin(6 * np.pi * t) + 0.05 * rng.standard_normal(T))


def random_warp(x, rng, strength=1.0):
    """Resample x along a random NONLINEAR monotonic time-warp (same shape, distorted timing)."""
    t = np.linspace(0, 1, T)
    inc = np.exp(strength * rng.standard_normal(T))            # positive increments -> monotonic
    w = np.cumsum(inc); w = (w - w[0]) / (w[-1] - w[0])
    return np.interp(t, w, x)


def rtw_subspace(x, rng):
    cols = [random_warp(x, rng, strength=0.8) for _ in range(K)]
    M = np.stack(cols, axis=1)                                  # (T, K) warp-orbit of the shape
    U, _, _ = np.linalg.svd(M - M.mean(1, keepdims=True), full_matrices=False)
    return U[:, :R]


def dtw(a, b):
    n = len(a); D = np.full((n + 1, n + 1), np.inf); D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            c = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, n]))


def phase_aligned_l2(a, b):
    """Best-integer-shift L2 — invariant to a CONSTANT phase offset but NOT to a nonlinear warp (the null)."""
    return float(min(np.linalg.norm(a - np.roll(b, s)) for s in range(-T // 4, T // 4 + 1)))


def main():
    rng = np.random.default_rng(0)
    rows = {m: {"warp": [], "shape": []} for m in ["RTW", "DTW", "phase_L2", "raw_L2"]}
    for _ in range(SEEDS):
        r = np.random.default_rng(rng.integers(1 << 30))
        x_ref = base_cycle(r, h2=0.4)
        x_warp = random_warp(x_ref, r, strength=1.4)                       # NUISANCE: STRONG nonlinear warp
        x_shape_raw = base_cycle(r, h2=0.9, h3=0.4)                        # REAL: same phase, changed shape
        # MAGNITUDE-MATCH the shape change to the warp (so raw-L2 cannot separate by magnitude -> ~0.5 control)
        wmag = np.linalg.norm(x_warp - x_ref)
        d = x_shape_raw - x_ref; x_shape = x_ref + d * (wmag / (np.linalg.norm(d) + 1e-12))
        Sref = rtw_subspace(x_ref, r)
        for tag, xv in [("warp", x_warp), ("shape", x_shape)]:
            rows["RTW"][tag].append(ss.magnitude(Sref, rtw_subspace(xv, r)))
            rows["DTW"][tag].append(dtw(x_ref, xv))
            rows["phase_L2"][tag].append(phase_aligned_l2(x_ref, xv))
            rows["raw_L2"][tag].append(float(np.linalg.norm(x_ref - xv)))

    y = np.r_[np.ones(SEEDS), np.zeros(SEEDS)]                              # shape=1 (real), warp=0 (nuisance)
    print("T2 RTW go/no-go — separate SHAPE-change (real) from nonlinear WARP (nuisance):")
    print(f"  {'method':>10} {'AUC(shape>warp)':>16} {'mean warp':>11} {'mean shape':>11}")
    res = {}
    for m in rows:
        s = np.r_[rows[m]["shape"], rows[m]["warp"]]
        a = roc_auc_score(y, s); a = float(max(a, 1 - a))
        res[m] = {"auc": a, "mean_warp": float(np.mean(rows[m]["warp"])), "mean_shape": float(np.mean(rows[m]["shape"]))}
        print(f"  {m:>10} {a:>16.3f} {res[m]['mean_warp']:>11.3f} {res[m]['mean_shape']:>11.3f}")

    print("\n=== VERDICT ===")
    rtw, ph, dt = res["RTW"]["auc"], res["phase_L2"]["auc"], res["DTW"]["auc"]
    go = rtw > 0.8 and rtw > ph + 0.05
    print(f"  RTW separates warp/shape: {rtw:.3f} | phase-aligned null: {ph:.3f} | DTW: {dt:.3f}")
    print(f"  => {'GO — warp-invariance is a real lever beyond phase-aware matching' if go else 'NO-GO — collapses to phase-aware / does not beat the null'}"
          f"  (RTW vs DTW: {'competitive' if rtw >= dt - 0.05 else 'DTW better'})")
    json.dump(res, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
