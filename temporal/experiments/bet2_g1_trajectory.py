"""BET 2 / G1 — recovery-TRAJECTORY characterization gate. Classify recovery PATH-type (fast / slow / detour)
from the time series, where types share the endpoint (recovered material V) but differ in PATH. Does the
trajectory GEOMETRY (subspace velocity / 2nd-order DS / geodesic along-orth / distance-to-baseline) characterize
the path better than (Null A) the scalar index recovery CURVE and (Null B, the killer) the full MULTI-BAND
per-band trajectory? Pre-reg: docs/research/bet2_design.md. Falsifier: geometry must beat the multi-band null.

Semi-synthetic on real Salinas spectra. D=damaged class, V=recovered class, M=intermediate (detour) class.
Path types over T dates (patch mean follows D -> target via frac(t)):
  fast  : D->V, fast saturating;  slow: D->V, slow;  detour: D->M (early) -> V (late).
Per-date patch = mean-path(t) + per-pixel spread + noise (+ illumination nuisance on test).

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.bet2_g1_trajectory
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score

from temporal import dynamics as dyn
from temporal import subspace as ss

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "bet2_g1_trajectory")
os.makedirs(OUT, exist_ok=True)
T, NB, N, K, MPER = 12, 100, 49, 6, 80
TYPES = ["fast", "slow", "detour"]


def frac_curve(kind):
    t = np.arange(T)
    if kind == "fast":
        return 1 - np.exp(-t / 1.8)
    if kind == "slow":
        return 1 - np.exp(-t / 6.0)
    return None  # detour handled separately


def make_traj(kind, D, V, Mid, spread_dirs, rng):
    patches = []
    for t in range(T):
        if kind == "detour":
            if t <= T // 2:
                f = (t / (T // 2)); m = D + f * (Mid - D)                 # D -> M
            else:
                f = (t - T // 2) / (T - 1 - T // 2); m = Mid + f * (V - Mid)  # M -> V
        else:
            f = frac_curve(kind)[t]; m = D + f * (V - D)
        coeff = 0.3 * rng.standard_normal((N, spread_dirs.shape[0]))
        P = m[None, :] + coeff @ spread_dirs + 0.05 * np.abs(m).mean() * rng.standard_normal((N, NB))
        patches.append(P)
    return patches


def geom_descriptor(patches):
    bases = [ss.pca_subspace(p.T, dim=K, center=False) for p in patches]
    vel = dyn.velocity_curve(bases)
    acc = dyn.acceleration_curve(bases)
    dbase = dyn.distance_to_baseline(bases, bases[0])
    return np.concatenate([vel, acc["d2"], acc["along"], acc["orth"], dbase])


def index_descriptor(patches, nir, swir, red):
    m = np.array([p.mean(0) for p in patches])
    nbr = (m[:, nir] - m[:, swir]) / (m[:, nir] + m[:, swir] + 1e-9)
    ndvi = (m[:, nir] - m[:, red]) / (m[:, nir] + m[:, red] + 1e-9)
    return np.concatenate([nbr, ndvi])                                    # 2T-dim index curves


def multiband_descriptor(patches):
    m = np.array([p.mean(0) for p in patches])                            # (T, NB)
    slope = np.array([np.polyfit(np.arange(T), m[:, b], 1)[0] for b in range(NB)])
    return np.concatenate([slope, m[-1], m[T // 2]])                      # per-band rate + final + mid (3*NB)


def nc_accuracy(Xtr, ytr, Xte, yte):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    cents = {c: Xtr[ytr == c].mean(0) for c in np.unique(ytr)}
    pred = [min(cents, key=lambda c: np.linalg.norm(x - cents[c])) for x in Xte]
    return float(accuracy_score(yte, pred))


def main():
    m = loadmat("data_hsi/salinas.mat")
    cube = [v for k, v in m.items() if not k.startswith("__") and v.ndim == 3][0].astype(float)
    gt = loadmat("data_hsi/salinas_gt.mat")
    lab = [v for k, v in gt.items() if not k.startswith("__") and v.ndim == 2][0]
    B = cube.shape[2]; X = cube.reshape(-1, B); y = lab.reshape(-1)
    bands = np.linspace(0, B - 1, NB).astype(int); X = X[:, bands]
    cnt = {c: int((y == c).sum()) for c in np.unique(y) if c > 0}
    top = sorted(cnt, key=cnt.get, reverse=True)
    D = X[y == top[0]].mean(0); V = X[y == top[1]].mean(0); Mid = X[y == top[2]].mean(0)
    sdirs = ss.pca_subspace(X[y == top[1]].T, dim=4, center=True).T       # real within-class spread dirs
    nir, swir, red = int(0.55 * NB), int(0.9 * NB), int(0.4 * NB)
    rng = np.random.default_rng(0)
    print(f"Bet2 G1 trajectory: T={T}, N={N}/patch, B={NB}; path types {TYPES}; {MPER} traj/type")

    for nuis in [0.0, 0.1]:
        feats = {"scalar_index": [], "multiband": [], "geometry": []}; ys = []
        for ci, kind in enumerate(TYPES):
            for _ in range(MPER):
                r = np.random.default_rng(rng.integers(1 << 30))
                patches = make_traj(kind, D, V, Mid, sdirs, r)
                if nuis > 0:
                    patches = [p * (1 + nuis * r.standard_normal()) for p in patches]  # per-date illumination
                feats["scalar_index"].append(index_descriptor(patches, nir, swir, red))
                feats["multiband"].append(multiband_descriptor(patches))
                feats["geometry"].append(geom_descriptor(patches))
                ys.append(ci)
        ys = np.array(ys)
        idx = rng.permutation(len(ys)); tr, te = idx[:len(idx) // 2], idx[len(idx) // 2:]
        print(f"\n  illum_nuisance={nuis}  (3-class path accuracy; chance=0.33)")
        accs = {}
        for meth in feats:
            Xf = np.array(feats[meth])
            accs[meth] = nc_accuracy(Xf[tr], ys[tr], Xf[te], ys[te])
            print(f"    {meth:>14}: {accs[meth]:.3f}")
        win = accs["geometry"] > accs["multiband"] + 0.03
        print(f"    => geometry beats multi-band: {win} "
              f"({'GEOMETRY ADDS' if win else 'redundant — multi-band matches/beats geometry'})")

    print("saved -> " + OUT)
    json.dump({"note": "see stdout"}, open(os.path.join(OUT, "metrics.json"), "w"))


if __name__ == "__main__":
    main()
