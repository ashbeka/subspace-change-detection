"""H-B-1b VERIFY — is the C2 positive (subspace detects mean-preserving structural change) real, or does a
trivial 2nd-order null match it? And is it SCALE-INVARIANT (the actual novelty)?

Three mean-preserving regimes (patch_i = A + s(t)*g_i*dvec(t) + noise; mean = A always):
  stable : dvec=d0 fixed,         s const          (nothing changes)
  rotate : dvec d0->d1,           s const          (STRUCTURAL/direction change; variance const)
  scale  : dvec=d0 fixed,         s grows s0->s1    (VARIANCE growth; NO structural change)
Statistics (peak over t):
  SUBSPACE     : sub_vel  (k=2 PCA subspace velocity), eig_angle (leading-eigvec turning = k=1 subspace)
  2nd-order nulls (non-Grassmann): cov_frob (||ΔCov||_F), var_vel (||Δ per-band var||)
  mean nulls   : sc_d2 (||Δ²m||), scn_d2 (||Δ²(m/||m||)||)
Money comparisons:
  rotate/stable  : does the subspace detect structural change? (mean nulls must fail)
  scale/stable   : SCALE test — subspace should NOT fire (scale-invariant); cov_frob/var WILL fire
  rotate/scale   : THE novelty — subspace separates structural change from variance growth;
                   a scale-SENSITIVE null (cov_frob/var) CANNOT (both 'changed the covariance')

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.hb1b_verify
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from temporal import dynamics as dyn
from temporal import subspace as ss

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "hb1b_verify")
os.makedirs(OUT, exist_ok=True)

T, N, K, M, B = 9, 80, 2, 60, 30
REGIMES = ["stable", "rotate", "scale"]
KEYS = ["sub_vel", "eig_angle", "cov_frob", "var_vel", "sc_d2", "scn_d2"]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def frame(rng):
    Q, _ = np.linalg.qr(rng.standard_normal((B, B)))
    A = np.abs(Q[:, 0]) * 3 + 2.0
    return A, unit(Q[:, 1]), unit(Q[:, 2])


def patches(regime, A, d0, d1, rng, s0=0.6, s1=1.6):
    out = []
    for t in range(T):
        frac = t / (T - 1)
        if regime == "stable":
            dvec, s = d0, s0
        elif regime == "rotate":
            th = (np.pi / 2) * frac; dvec, s = np.cos(th) * d0 + np.sin(th) * d1, s0
        else:                                              # scale: fixed direction, growing variance
            dvec, s = d0, s0 + (s1 - s0) * frac
        g = rng.standard_normal((N, 1))
        out.append((A[None, :] + s * g * dvec[None, :] + 0.01 * rng.standard_normal((N, B))).T)
    return out


def stats(ps):
    bases = [ss.pca_subspace(p, dim=K, center=False) for p in ps]
    sub_vel = float(np.max(dyn.velocity_curve(bases)))
    covs = [np.cov(p) for p in ps]                          # (B,B) per date
    eigvecs = [np.linalg.eigh(c)[1][:, -1] for c in covs]   # leading eigenvector
    eig_angle = float(np.max([np.arccos(min(1.0, abs(eigvecs[t] @ eigvecs[t + 1]))) for t in range(T - 1)]))
    cov_frob = float(np.max([np.linalg.norm(covs[t + 1] - covs[t]) for t in range(T - 1)]))
    var_series = np.array([p.var(1) for p in ps])
    var_vel = float(np.max(np.linalg.norm(np.diff(var_series, axis=0), axis=1)))
    m = np.array([p.mean(1) for p in ps]); mn = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    sc_d2 = float(np.max(np.linalg.norm(m[2:] - 2 * m[1:-1] + m[:-2], axis=1)))
    scn_d2 = float(np.max(np.linalg.norm(mn[2:] - 2 * mn[1:-1] + mn[:-2], axis=1)))
    return {"sub_vel": sub_vel, "eig_angle": eig_angle, "cov_frob": cov_frob, "var_vel": var_vel,
            "sc_d2": sc_d2, "scn_d2": scn_d2}


def main():
    rng = np.random.default_rng(0)
    data = {r: [] for r in REGIMES}
    for r in REGIMES:
        for _ in range(M):
            A, d0, d1 = frame(np.random.default_rng(rng.integers(1 << 30)))
            data[r].append(stats(patches(r, A, d0, d1, np.random.default_rng(rng.integers(1 << 30)))))
    arr = {r: {k: np.array([d[k] for d in data[r]]) for k in KEYS} for r in REGIMES}

    def auc(pos, neg, k):
        y = np.r_[np.ones(M), np.zeros(M)]; s = np.r_[arr[pos][k], arr[neg][k]]
        return float(max(roc_auc_score(y, s), 1 - roc_auc_score(y, s)))

    comps = [("rotate", "stable", "structural change detected?"),
             ("scale", "stable", "scale test (subspace should IGNORE)"),
             ("rotate", "scale", "NOVELTY: structural vs variance growth")]
    out = {}
    print(f"  {'comparison':<48}" + "".join(f"{k:>10}" for k in KEYS))
    for pos, neg, label in comps:
        row = {k: auc(pos, neg, k) for k in KEYS}
        out[f"{pos}_vs_{neg}"] = row
        print(f"  {pos+'/'+neg+' '+label:<48}" + "".join(f"{row[k]:>10.3f}" for k in KEYS))

    rs, ss_, rsc = out["rotate_vs_stable"], out["scale_vs_stable"], out["rotate_vs_scale"]
    sub = lambda d: max(d["sub_vel"], d["eig_angle"]); nul2 = lambda d: max(d["cov_frob"], d["var_vel"])
    print("\n  === VERDICT ===")
    print(f"  detect structural (rotate/stable): subspace={sub(rs):.3f} 2nd-order-null={nul2(rs):.3f} mean={max(rs['sc_d2'],rs['scn_d2']):.3f}")
    print(f"  scale-invariance (scale/stable):   subspace={sub(ss_):.3f} (want ~0.5)  2nd-order-null={nul2(ss_):.3f} (will fire)")
    print(f"  NOVELTY (rotate/scale):            subspace={sub(rsc):.3f} (want high)  2nd-order-null={nul2(rsc):.3f} (want ~0.5)")
    win = sub(rs) > 0.85 and sub(ss_) < 0.65 and sub(rsc) > 0.80 and nul2(rsc) < 0.65
    print(f"  => {'SUBSPACE UNIQUELY isolates scale-invariant STRUCTURAL change (real lead)' if win else 'lead weaker than it looked — see numbers'}")
    json.dump(out, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
