"""H-B-1b — the ONLY places a subspace trajectory can be non-redundant with the patch-MEAN vector:
(C2) MEAN-PRESERVING structural change, and (C1) DIRECTION-change under amplitude nuisance.
H-B-1 showed the subspace is redundant with ||Δ²(mean spectrum)|| for clean direction-change. Here we
test the cases the mean-vector CANNOT see — but with HONEST stronger nulls so we don't fool ourselves:
  - normalized-mean curvature  ||Δ²( m/||m|| )||   (amplitude-invariant scalar; the SAM-style null)
  - per-band VARIANCE-vector trajectory  ||Δ v||, ||Δ² v||  where v(t)=Var_pixels(band)  (a 2nd-order
    but NON-subspace null — captures spread without any Grassmann geometry)

C2 construction (mean & total-variance preserved): patch_i = A + s * g_i * dvec(t), g_i~N(0,1).
  stable : dvec fixed = d0.                          (no structural change)
  rotate : dvec(t) rotates d0 -> d1 (orthonormal).   (spread DIRECTION changes; mean=A, total var=s^2 const)
Detection question: rotate vs stable. Mean nulls MUST be ~0.5 (mean constant). Does the subspace detect it
AND beat the per-band variance-vector null? If var-vector also catches it, the subspace is STILL redundant.

C1 construction: straight vs turn (from H-B-1) with per-date GLOBAL illumination * PER-BAND gain nuisance.
  Compare kappa vs raw ||Δ²m|| and normalized ||Δ²(m/||m||)|| as nuisance grows.

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.hb1b_structure
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from temporal import dynamics as dyn
from temporal import subspace as ss

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "hb1b_structure")
os.makedirs(OUT, exist_ok=True)

T, N, K, M = 9, 80, 2, 50
B = 30


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def frame(rng):
    Q, _ = np.linalg.qr(rng.standard_normal((B, B)))
    A = np.abs(Q[:, 0]) * 3 + 2.0                      # positive base spectrum
    d0, d1 = unit(Q[:, 1]), unit(Q[:, 2])             # orthonormal spread directions
    return A, d0, d1


def c2_patches(regime, A, d0, d1, s, rng):
    patches = []
    for t in range(T):
        if regime == "stable":
            dvec = d0
        else:                                          # rotate d0 -> d1 over the series
            th = (np.pi / 2) * t / (T - 1)
            dvec = np.cos(th) * d0 + np.sin(th) * d1
        g = rng.standard_normal((N, 1))
        P = A[None, :] + s * g * dvec[None, :] + 0.01 * rng.standard_normal((N, B))
        patches.append(P.T)
    return patches


def curves_c2(patches):
    bases = [ss.pca_subspace(P, dim=K, center=False) for P in patches]
    mean_series = np.array([P.mean(1) for P in patches])
    var_series = np.array([P.var(1) for P in patches])              # per-band variance vector (non-subspace 2nd-order)
    vel = dyn.velocity_curve(bases)
    sub_vel = float(np.max(vel))
    sub_d2 = float(np.max(dyn.acceleration_curve(bases)["d2"]))
    m = mean_series; mn = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    sc_d2 = float(np.max(np.linalg.norm(m[2:] - 2 * m[1:-1] + m[:-2], axis=1)))
    scn_d2 = float(np.max(np.linalg.norm(mn[2:] - 2 * mn[1:-1] + mn[:-2], axis=1)))
    var_vel = float(np.max(np.linalg.norm(np.diff(var_series, axis=0), axis=1)))
    return {"sub_vel": sub_vel, "sub_d2": sub_d2, "sc_d2": sc_d2, "scn_d2": scn_d2, "var_vel": var_vel}


def run_c2():
    print("\n################  C2: MEAN-PRESERVING structural change (rotate vs stable)  ################")
    rng = np.random.default_rng(0)
    keys = ["sub_vel", "sub_d2", "sc_d2", "scn_d2", "var_vel"]
    data = {"stable": [], "rotate": []}
    for r in ("stable", "rotate"):
        for _ in range(M):
            A, d0, d1 = frame(np.random.default_rng(rng.integers(1 << 30)))
            data[r].append(curves_c2(c2_patches(r, A, d0, d1, 0.6, np.random.default_rng(rng.integers(1 << 30)))))
    arr = {r: {k: np.array([d[k] for d in data[r]]) for k in keys} for r in data}
    y = np.r_[np.ones(M), np.zeros(M)]
    print(f"  {'stat':>10}  AUC(rotate vs stable)   [SUB=subspace, others=non-subspace nulls]")
    res = {}
    for k in keys:
        s = np.r_[arr["rotate"][k], arr["stable"][k]]
        a = float(max(roc_auc_score(y, s), 1 - roc_auc_score(y, s)))
        res[k] = a
        tag = "SUBSPACE" if k.startswith("sub") else ("mean-null" if k in ("sc_d2", "scn_d2") else "var-vector-null")
        print(f"  {k:>10}  {a:6.3f}   ({tag})")
    sub = max(res["sub_vel"], res["sub_d2"]); mean_null = max(res["sc_d2"], res["scn_d2"]); var_null = res["var_vel"]
    print(f"\n  mean-nulls ~0.5? {mean_null:.3f} (should be, mean constant). "
          f"subspace={sub:.3f}  var-vector-null={var_null:.3f}")
    verdict = ("SUBSPACE UNIQUELY WINS (var-vector null also fails)" if sub > 0.8 and var_null < 0.65
               else "subspace detects it BUT per-band variance-vector also does -> still redundant"
               if sub > 0.8 and var_null > 0.8 else "subspace weak")
    print(f"  => C2 verdict: {verdict}")
    return res


def c1_patches(regime, A, u1, u2, alpha, rng):
    T0 = 4; step = 0.5
    patches = []
    m = A.copy()
    for t in range(T):
        if t > 0:
            d = u1 if (regime == "straight" or t <= T0) else u2
            m = m + step * d
        g = rng.standard_normal((N, 1)) * 0.1
        P = A[None, :] * 0 + m[None, :] + g * u2[None, :] + 0.01 * rng.standard_normal((N, B))
        illum = 1.0 + alpha * rng.standard_normal()                # global illumination
        gain = 1.0 + alpha * rng.standard_normal(B)                # per-band gain
        P = P * illum * gain[None, :]
        patches.append(P.T)
    return patches


def run_c1():
    print("\n################  C1: DIRECTION-change under amplitude nuisance (turn vs straight)  ################")
    rng = np.random.default_rng(1)
    print(f"  {'alpha':>6} {'kappa':>7} {'sc_d2(raw)':>11} {'scn_d2(norm)':>13}")
    res = {}
    for alpha in [0.0, 0.1, 0.3]:
        sc = {"kappa": {"turn": [], "straight": []}, "sc_d2": {"turn": [], "straight": []},
              "scn_d2": {"turn": [], "straight": []}}
        for r in ("turn", "straight"):
            for _ in range(M):
                A, d0, d1 = frame(np.random.default_rng(rng.integers(1 << 30)))
                P = c1_patches(r, A, d0, d1, alpha, np.random.default_rng(rng.integers(1 << 30)))
                bases = [ss.pca_subspace(p, dim=K, center=False) for p in P]
                vel = dyn.velocity_curve(bases)
                vmax = float(np.max(vel)) + 1e-12
                kap = [ss.magnitude(ss.difference_subspace(bases[t - 1], bases[t]),
                                    ss.difference_subspace(bases[t], bases[t + 1]))
                       if (vel[t - 1] > .25 * vmax and vel[t] > .25 * vmax) else 0.0
                       for t in range(1, len(bases) - 1)]
                ms = np.array([p.mean(1) for p in P]); mn = ms / (np.linalg.norm(ms, axis=1, keepdims=True) + 1e-12)
                sc["kappa"][r].append(max(kap) if kap else 0.0)
                sc["sc_d2"][r].append(float(np.max(np.linalg.norm(ms[2:] - 2 * ms[1:-1] + ms[:-2], axis=1))))
                sc["scn_d2"][r].append(float(np.max(np.linalg.norm(mn[2:] - 2 * mn[1:-1] + mn[:-2], axis=1))))
        y = np.r_[np.ones(M), np.zeros(M)]
        row = {}
        for k in sc:
            s = np.r_[sc[k]["turn"], sc[k]["straight"]]
            row[k] = float(max(roc_auc_score(y, s), 1 - roc_auc_score(y, s)))
        res[alpha] = row
        print(f"  {alpha:>6} {row['kappa']:>7.3f} {row['sc_d2']:>11.3f} {row['scn_d2']:>13.3f}")
    print("  => C1: does kappa beat BOTH mean-nulls as nuisance grows? (else subspace redundant)")
    return res


def main():
    out = {"C2_structure": run_c2(), "C1_nuisance": run_c1()}
    json.dump(out, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
