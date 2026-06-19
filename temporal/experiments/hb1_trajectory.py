"""H-B-1 — Does the GEOMETRY of the change-trajectory recover the TYPE of change (speed-change vs
direction-change) that scalar dynamics cannot?  (Fukui 2nd-order Difference Subspace, satellite-style.)

PRE-REGISTRATION (written before looking at matched-speed results):
  Object: a multi-date spectral "patch" trajectory whose per-date mean spectrum m(t) follows a controlled
  path; per-date subspace S(t) = uncentered top-k PCA of the patch (tracks the dominant material direction).
  THREE regimes built on the SAME base A and two unit step-directions u1 (toward B) and u2 (orthogonal,
  toward C):
    - straight : constant step along u1            (constant-velocity geodesic; no event)
    - accel    : along u1, step size slow->fast     (SPEED change, same direction)
    - turn     : constant step, u1 then u2 at T0     (DIRECTION change, SAME speed as straight)
  Because straight and turn move with IDENTICAL per-step speed, every SCALAR speed/acceleration statistic
  is the same for them -> scalars are blind to the turn. accel changes speed -> scalars see it.
  Statistics (peak over valid t): subspace = velocity d1, 2nd-order DS magnitude d2 (Karcher), geodesic
  along/orth, velocity-GATED DS-of-DS curvature kappa(t)=magnitude(DS(S_{t-1},S_t),DS(S_t,S_{t+1}));
  scalar nulls = |2nd-diff mean spectrum| sc_d2, scalar accel sc_acc, numeric 2nd-diff of subspace
  velocity numd2_vel, and TOTAL change (matched control).
  HYPOTHESES / FALSIFIERS:
    H1 turn vs straight: kappa (and d2) HIGH, all SCALARS ~0.5 (matched speed). [geometry sees direction]
       Falsifier: a scalar separates turn/straight => geometry adds nothing.
    H2 turn vs accel:    kappa HIGH (direction), d2 ~0.5 (both accelerate), sc_acc HIGH (speed).
       => the 2-D signature (d2 magnitude, kappa direction) recovers the change TYPE no scalar can.
    H3 accel vs straight (sanity): scalars OK (speed change is a scalar 2nd-difference).
  LADDER: L0 clean orthogonal synth (mechanism?) -> L1 realistic correlated Salinas (survives realism?).
  Multi-seed; every scalar null reported beside every subspace statistic (no single-seed cherry-pick).

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.hb1_trajectory
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

from temporal import dynamics as dyn
from temporal import subspace as ss

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "hb1_trajectory")
os.makedirs(OUT, exist_ok=True)

REGIMES = ["straight", "accel", "turn"]
STAT_KEYS = ["vel", "d2", "along", "orth", "kappa", "numd2_vel", "sc_acc", "sc_d2", "total_change"]
SUBSPACE_KEYS = ["kappa", "orth", "along", "d2"]
SCALAR_KEYS = ["sc_acc", "sc_d2", "numd2_vel", "total_change"]


def salinas_endmembers():
    m = loadmat("data_hsi/salinas.mat")
    cube = [v for k, v in m.items() if not k.startswith("__") and v.ndim == 3][0].astype(float)
    g = loadmat("data_hsi/salinas_gt.mat")
    lab = [v for k, v in g.items() if not k.startswith("__") and v.ndim == 2][0]
    B = cube.shape[2]; X = cube.reshape(-1, B); y = lab.reshape(-1)
    means = [X[y == c].mean(0) for c in np.unique(y) if c > 0]
    return np.array(means)


def synth_endmembers(B, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((B, B)))
    return Q.T[: min(B, 24)] + 2.0                              # offset so spectra are positive-ish


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def pick_frame(E, rng):
    """A (base), u1 (toward B), u2 (orthogonalized toward C); u1,u2 unit & orthogonal => a true turn."""
    n = E.shape[0]
    a, b, c = rng.choice(n, 3, replace=False)
    A = E[a].astype(float)
    u1 = unit(E[b] - A)
    u2 = E[c] - A
    u2 = unit(u2 - (u2 @ u1) * u1)                              # Gram-Schmidt: u2 ⟂ u1
    spread = E[(c + 1) % n] - A                                 # within-patch variation direction
    spread = unit(spread - (spread @ u1) * u1 - (spread @ u2) * u2)
    return A, u1, u2, spread


def mean_path(regime, A, u1, u2, cfg):
    T, T0, step = cfg["T"], cfg["T0"], cfg["STEP"]
    m = [A.copy()]
    for t in range(1, T):
        if regime == "straight":
            d, sc = u1, 1.0
        elif regime == "accel":                                # same direction, speed ramps slow->fast
            d, sc = u1, (0.25 if t <= T0 else 1.75)
        elif regime == "turn":                                 # constant speed, direction flips at T0
            d, sc = (u1 if t <= T0 else u2), 1.0
        m.append(m[-1] + step * sc * d)
    return np.array(m)


def make_patches(mpath, spread, cfg, rng):
    patches = []
    for t in range(len(mpath)):
        coeff = cfg["SPREAD"] * rng.standard_normal((cfg["N"], 1))
        P = mpath[t][None, :] + coeff * spread[None, :] + cfg["SIGMA"] * rng.standard_normal((cfg["N"], mpath.shape[1]))
        patches.append(P.T)                                    # (B,N)
    return patches


def ds_curvature(bases, vel, gate=0.25):
    """Velocity-gated turning of the change direction. Zero where motion is below the noise floor."""
    vmax = float(np.max(vel)) + 1e-12
    kap = []
    for t in range(1, len(bases) - 1):
        dp = ss.difference_subspace(bases[t - 1], bases[t]); dn = ss.difference_subspace(bases[t], bases[t + 1])
        moving = (vel[t - 1] > gate * vmax) and (vel[t] > gate * vmax)
        kap.append(ss.magnitude(dp, dn) if (moving and dp.shape[1] and dn.shape[1]) else 0.0)
    return np.array(kap)


def trajectory_curves(A, u1, u2, spread, regime, cfg, rng):
    mpath = mean_path(regime, A, u1, u2, cfg)
    patches = make_patches(mpath, spread, cfg, rng)
    bases = [ss.pca_subspace(P, dim=cfg["K"], center=False) for P in patches]
    mean_series = np.array([P.mean(1) for P in patches])

    vel = dyn.velocity_curve(bases)
    acc = dyn.acceleration_curve(bases)
    kappa = ds_curvature(bases, vel)
    numd2_vel = np.abs(np.diff(vel))
    sc_vel = np.linalg.norm(np.diff(mean_series, axis=0), axis=1)
    sc_acc = np.abs(np.diff(sc_vel))
    sc_d2 = np.linalg.norm(mean_series[2:] - 2 * mean_series[1:-1] + mean_series[:-2], axis=1)
    total = float(np.linalg.norm(mean_series[-1] - mean_series[0]))
    return {"vel": vel, "d2": acc["d2"], "along": acc["along"], "orth": acc["orth"], "kappa": kappa,
            "numd2_vel": numd2_vel, "sc_acc": sc_acc, "sc_d2": sc_d2, "total_change": np.array([total])}


def run_rung(name, cfg):
    print(f"\n################  {name}  ################\n  cfg: {cfg}")
    rng = np.random.default_rng(cfg["seed"])
    E = salinas_endmembers() if cfg["endmembers"] == "salinas" else synth_endmembers(cfg["B"], rng)
    M = cfg["M"]
    peaks = {r: {k: [] for k in STAT_KEYS} for r in REGIMES}
    curves = {r: {k: [] for k in ["kappa", "d2", "sc_acc", "vel"]} for r in REGIMES}
    for r in REGIMES:
        for _ in range(M):
            A, u1, u2, spread = pick_frame(E, np.random.default_rng(rng.integers(1 << 30)))
            cur = trajectory_curves(A, u1, u2, spread, r, cfg, np.random.default_rng(rng.integers(1 << 30)))
            for k in STAT_KEYS:
                peaks[r][k].append(float(np.max(cur[k])))
            for k in curves[r]:
                curves[r][k].append(cur[k])
    peaks = {r: {k: np.array(v) for k, v in d.items()} for r, d in peaks.items()}

    print("  --- mean curves over M (mechanism visible) ---")
    for k in ["kappa", "d2", "sc_acc"]:
        print(f"  {k:>8}: " + " | ".join(
            f"{r[:5]}=[" + ",".join(f"{x:.2f}" for x in np.mean(np.array(curves[r][k]), axis=0)) + "]"
            for r in REGIMES))

    def auc(pos, neg, k):
        y = np.r_[np.ones(M), np.zeros(M)]; s = np.r_[peaks[pos][k], peaks[neg][k]]
        return float(max(roc_auc_score(y, s), 1 - roc_auc_score(y, s)))

    comparisons = [("turn", "straight", "DIRECTION-change @matched speed (KEY: scalars must fail)"),
                   ("turn", "accel", "direction vs speed (KEY: d2 must fail, kappa win)"),
                   ("accel", "straight", "speed-change (sanity: scalars OK)")]
    out = {"config": cfg}
    print(f"\n  {'comparison':<55}" + "".join(f"{k:>10}" for k in STAT_KEYS))
    for pos, neg, label in comparisons:
        row = {k: auc(pos, neg, k) for k in STAT_KEYS}
        out[f"{pos}_vs_{neg}"] = row
        print(f"  {pos+'/'+neg+' '+label:<55}" + "".join(f"{row[k]:>10.3f}" for k in STAT_KEYS))

    ts = out["turn_vs_straight"]; ta = out["turn_vs_accel"]
    best_scalar_ts = max(SCALAR_KEYS, key=lambda k: ts[k])
    h1 = ts["kappa"] > 0.80 and ts[best_scalar_ts] < 0.65 and ts["total_change"] < 0.65
    h2 = ta["kappa"] > 0.80 and ta["d2"] < 0.70
    print(f"\n  H1 turn/straight: kappa={ts['kappa']:.3f}  best-scalar({best_scalar_ts})={ts[best_scalar_ts]:.3f} "
          f"total_change={ts['total_change']:.3f}  -> {'PASS' if h1 else 'FAIL'}")
    print(f"  H2 turn/accel:    kappa={ta['kappa']:.3f}  d2={ta['d2']:.3f}  sc_acc={ta['sc_acc']:.3f} "
          f"-> {'PASS' if h2 else 'FAIL'}")
    print(f"  => {name}: {'SUPPORTED (geometry recovers change-TYPE; scalars cannot)' if (h1 and h2) else 'NOT fully supported'}")
    out["verdict"] = {"H1_turn_vs_straight": bool(h1), "H2_turn_vs_accel": bool(h2)}
    return out


def main():
    rungs = {
        "L0_clean_synth": dict(endmembers="synth", B=30, T=9, T0=4, N=60, K=2,
                               STEP=0.5, SPREAD=0.15, SIGMA=0.01, M=40, seed=0),
        "L1_realistic_salinas": dict(endmembers="salinas", B=204, T=9, T0=4, N=50, K=2,
                                     STEP=0.25, SPREAD=0.25, SIGMA=0.02, M=40, seed=0),
    }
    results = {name: run_rung(name, cfg) for name, cfg in rungs.items()}
    json.dump(results, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
