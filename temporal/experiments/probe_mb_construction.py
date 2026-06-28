"""ADVERSARIAL probe: CONSTRUCTION ROBUSTNESS of the multiband harmonic_dropout DS claim.

Claim under attack (from multiband_gate.py / metrics.json):
  full DS detects harmonic_dropout at AUC 1.0, min-angle at chance (~0.5), robust to noise.

My angle: is DS-AUC=1.0 and the DS>min-angle margin a STABLE plateau, or a knife-edge on
construction knobs (rank, energy, W, L, n_regions)? Does FIXED rank (no energy selection) change
the story?

Reuses committed builders: temporal.synth_multiband.make_panel, and a local copy of the committed
mssa_subspace / scoring logic (faithful to temporal.experiments.multiband_gate) so we measure the
SAME estimator the claim used, just sweeping its knobs.

Outputs: temporal/outputs/_mbprobe_construction/
Run: ./.venv/Scripts/python.exe -m temporal.experiments.probe_mb_construction
"""
from __future__ import annotations

import json
import os
import itertools

import numpy as np
from sklearn.metrics import roc_auc_score

from temporal import subspace as ss
from temporal.synth_multiband import MBConfig, make_panel

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "_mbprobe_construction")
os.makedirs(OUT, exist_ok=True)


# ---- faithful copies of the committed estimator (multiband_gate.py) -------------------
def hankel(x, L):
    N = len(x); K = N - L + 1
    return np.stack([x[i:i + K] for i in range(L)])


def mssa_subspace(Xwin, L, rank=8, energy=0.99):
    blocks = [hankel(Xwin[b], L) for b in range(Xwin.shape[0])]
    H = np.concatenate(blocks, axis=0)
    return ss.pca_subspace(H, rank, center=False, energy=energy)


def acf_multilag(x, lags):
    x = np.asarray(x, float) - np.mean(x); d = np.dot(x, x)
    if d < 1e-12:
        return np.zeros(len(lags))
    return np.array([np.dot(x[:-k], x[k:]) / d for k in lags])


def region_scores(X, W, L, rank, energy, lags):
    """Max-over-boundary DS, min-angle, and best-scalar (acf_ml) scores for one region (B,T)."""
    B, T = X.shape
    ds, ma, acf = [], [], []
    for t in range(W, T - W + 1):
        past = X[:, t - W:t]; pres = X[:, t:t + W]
        Sp = mssa_subspace(past, L, rank, energy); Sq = mssa_subspace(pres, L, rank, energy)
        cos = ss.canonical_cosines(Sp, Sq)
        ds.append(ss.magnitude(Sp, Sq)); ma.append(1.0 - np.max(cos))
        acf_d = np.mean([np.sum(np.abs(acf_multilag(pres[b], lags) - acf_multilag(past[b], lags)))
                         for b in range(B)])
        acf.append(acf_d)
    return float(np.max(ds)), float(np.max(ma)), float(np.max(acf))


def eval_config(mode, noise, W, L, rank, energy, n_regions, seeds):
    """Returns dict of mean/std AUC for ds, minangle, acf_ml across seeds, plus rank diagnostics."""
    lags = list(range(1, L))
    a_ds, a_ma, a_acf, ranks = [], [], [], []
    for sd in seeds:
        cfg = MBConfig(mode=mode, seed=sd, noise=noise, n_regions=n_regions)
        X, labels = make_panel(cfg)
        ds_s, ma_s, acf_s = [], [], []
        for x in X:
            d, m, a = region_scores(x, W, L, rank, energy, lags)
            ds_s.append(d); ma_s.append(m); acf_s.append(a)
        # rank actually realized on a representative window (first region, central boundary)
        x0 = X[0]; t = (W + (X.shape[2] - W)) // 2
        S0 = mssa_subspace(x0[:, t - W:t], L, rank, energy)
        ranks.append(S0.shape[1])
        a_ds.append(roc_auc_score(labels, ds_s))
        a_ma.append(roc_auc_score(labels, ma_s))
        a_acf.append(roc_auc_score(labels, acf_s))
    return {
        "ds": (float(np.mean(a_ds)), float(np.std(a_ds))),
        "minangle": (float(np.mean(a_ma)), float(np.std(a_ma))),
        "acf_ml": (float(np.mean(a_acf)), float(np.std(a_acf))),
        "realized_rank": float(np.mean(ranks)),
        "ambient_BL": MBConfig().B * L,
        "K_cols": W - L + 1,
    }


def main():
    seeds = (0, 1, 2, 3)
    noises = [0.25, 0.8]
    ranks = [4, 6, 8, 12]
    energies = [0.9, 0.95, 0.99, 0.999]
    Ws = [16, 24, 32]
    Ls = [8, 12, 16]
    n_regions_list = [60, 120]

    grid = []
    # ---- main 5-knob sweep, holding the OTHERS at the committed defaults while varying one axis
    # plus a focused full cross of (rank x energy) and (W x L) to find interactions.
    base = dict(W=24, L=12, rank=8, energy=0.99, n_regions=120)

    def run(tag, **kw):
        p = dict(base); p.update(kw)
        for noise in noises:
            # guard: rank cannot exceed available columns K=W-L+1 or ambient B*L
            r = eval_config("harmonic_dropout", noise, p["W"], p["L"], p["rank"],
                            p["energy"], p["n_regions"], seeds)
            row = dict(tag=tag, noise=noise, **p,
                       ds_auc=r["ds"][0], ds_std=r["ds"][1],
                       minangle_auc=r["minangle"][0], acf_auc=r["acf_ml"][0],
                       ds_minus_minangle=r["ds"][0] - r["minangle"][0],
                       ds_minus_acf=r["ds"][0] - r["acf_ml"][0],
                       realized_rank=r["realized_rank"], K_cols=r["K_cols"],
                       ambient_BL=r["ambient_BL"])
            grid.append(row)
            print(f"[{tag:14}] noise={noise} W={p['W']} L={p['L']} rank={p['rank']} "
                  f"energy={p['energy']} nreg={p['n_regions']} | "
                  f"DS={r['ds'][0]:.3f} minA={r['minangle'][0]:.3f} acf={r['acf_ml'][0]:.3f} "
                  f"DS-minA={r['ds'][0]-r['minangle'][0]:+.3f} rrank={r['realized_rank']:.1f} K={r['K_cols']}")

    # axis sweeps
    for rk in ranks:
        run("rank", rank=rk)
    for en in energies:
        run("energy", energy=en)
    for W in Ws:
        run("window_W", W=W)
    for L in Ls:
        run("embed_L", L=L)
    for nr in n_regions_list:
        run("n_regions", n_regions=nr)

    # FIXED rank (no energy selection): energy=None forces k = min(rank, cols)
    print("\n--- FIXED RANK (energy=None) ---")
    for rk in ranks:
        run("fixed_rank", rank=rk, energy=None)

    # interaction crosses
    print("\n--- rank x energy cross ---")
    for rk, en in itertools.product(ranks, energies):
        run("rank_x_energy", rank=rk, energy=en)

    print("\n--- W x L cross ---")
    for W, L in itertools.product(Ws, Ls):
        if L >= W:
            continue
        run("W_x_L", W=W, L=L)

    with open(os.path.join(OUT, "grid.json"), "w") as f:
        json.dump(grid, f, indent=2)

    # ---- summary: stable region vs collapse ----
    def summarize(noise):
        rows = [g for g in grid if g["noise"] == noise]
        ds1 = [g for g in rows if g["ds_auc"] >= 0.999]
        win_minA = [g for g in rows if g["ds_minus_minangle"] >= 0.30]
        ds_lt_acf = [g for g in rows if g["ds_minus_acf"] < -0.02]
        ds_collapse = sorted(rows, key=lambda g: g["ds_auc"])[:5]
        print(f"\n=== noise={noise} ===")
        print(f"  configs: {len(rows)}  DS-AUC>=0.999: {len(ds1)}  "
              f"DS beats minangle by >=0.30: {len(win_minA)}  "
              f"DS clearly < acf (best scalar): {len(ds_lt_acf)}")
        print(f"  DS range: [{min(g['ds_auc'] for g in rows):.3f}, {max(g['ds_auc'] for g in rows):.3f}]")
        print(f"  worst-5 DS configs:")
        for g in ds_collapse:
            print(f"    DS={g['ds_auc']:.3f} minA={g['minangle_auc']:.3f} acf={g['acf_auc']:.3f} "
                  f"[{g['tag']} W={g['W']} L={g['L']} rank={g['rank']} energy={g['energy']} "
                  f"nreg={g['n_regions']} rrank={g['realized_rank']:.1f}]")
        return dict(n=len(rows), ds_auc_1=len(ds1), win_minA=len(win_minA),
                    ds_lt_acf=len(ds_lt_acf),
                    ds_min=min(g["ds_auc"] for g in rows),
                    ds_max=max(g["ds_auc"] for g in rows))

    summ = {str(n): summarize(n) for n in noises}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(f"\nSaved {len(grid)} configs to {OUT}")


if __name__ == "__main__":
    main()
