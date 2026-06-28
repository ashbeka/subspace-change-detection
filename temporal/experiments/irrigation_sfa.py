"""Does SFA's order-sensitivity fix the temporal-DS failure? Head-to-head on the CACHED irrigation sites
(no GEE re-fetch). M-SSA-DS scored AUC 0.417 (below chance) separating switched-vs-constant; NDVI-amplitude
scored 0.729. Test whether a Slow-Feature DS (SFA-DS) and an SFA-slowness-change score do better.

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.irrigation_sfa
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from temporal import gee_fetch as gf
from temporal import slowfeature as sf
from temporal import subspace as ss
from temporal.experiments.multiband_gate import mssa_subspace
from temporal.experiments.real_irrigation import sample_sites, BANDS, CFG, W, L, RANK, ENERGY


def zscore(X):
    return (X - np.nanmean(X, 1, keepdims=True)) / (np.nanstd(X, 1, keepdims=True) + 1e-9)


def scores(ts):
    X, _ = gf.as_matrix(ts, BANDS)
    Xz = zscore(X); T = Xz.shape[1]
    sfa_d1, mssa_d1 = [], []
    for t in range(W, T - W + 1):
        past, pres = Xz[:, t - W:t], Xz[:, t:t + W]
        if np.isnan(past).any() or np.isnan(pres).any():
            continue
        sfa_d1.append(sf.sfa_ds(past, pres, n_slow=4))
        Sp, Sq = mssa_subspace(past, L, RANK, energy=ENERGY), mssa_subspace(pres, L, RANK, energy=ENERGY)
        mssa_d1.append(ss.magnitude(Sp, Sq))
    # SFA slowness change: first-third vs last-third of the series
    third = T // 3
    sl_pre = sf.slowness(Xz[:, :third]) if third >= 4 else np.nan
    sl_post = sf.slowness(Xz[:, -third:]) if third >= 4 else np.nan
    # SFA-DS pre vs post slow-feature subspace (whole-period)
    sfa_prepost = sf.sfa_ds(Xz[:, :T // 2], Xz[:, T // 2:], n_slow=4)
    return {"sfa_d1_max": float(np.max(sfa_d1)) if sfa_d1 else 0.0,
            "mssa_d1_max": float(np.max(mssa_d1)) if mssa_d1 else 0.0,
            "sfa_slowness_change": float(abs(sl_post - sl_pre)) if np.isfinite(sl_pre) and np.isfinite(sl_post) else 0.0,
            "sfa_prepost_ds": float(sfa_prepost)}


def main():
    sites = sample_sites(CFG["n_sites_per_category"])
    rows, y = [], []
    for i, s in enumerate(sites):
        name = f"irr_{CFG['region_name']}_{i}_cat{s['cat']}"
        aoi = gf.point_aoi(s["lon"], s["lat"], buffer_m=120)
        try:
            ts = gf.fetch_timeseries(name, aoi, CFG["s2_fetch_range"][0], CFG["s2_fetch_range"][1])
        except Exception:
            continue
        if len(ts["dates"]) < 2 * W + 5:
            continue
        sc = scores(ts); sc["cat"] = s["cat"]; sc["switched"] = s["switched"]
        rows.append(sc); y.append(1 if s["switched"] else 0)
        print(f"  site {i} cat{s['cat']} sw={s['switched']}: sfa_d1={sc['sfa_d1_max']:.2f} "
              f"mssa_d1={sc['mssa_d1_max']:.2f} sfa_slow_chg={sc['sfa_slowness_change']:.3f} sfa_prepost={sc['sfa_prepost_ds']:.2f}")
    y = np.array(y)
    methods = ["sfa_d1_max", "mssa_d1_max", "sfa_slowness_change", "sfa_prepost_ds"]
    aucs = {m: roc_auc_score(y, [r[m] for r in rows]) for m in methods}
    print("\n=== SFA vs M-SSA on irrigation switch (AUC switched-vs-constant; NDVI-amp baseline was 0.729) ===")
    for m in methods:
        print(f"  {m:22}: {aucs[m]:.3f}")
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "irrigation_sfa")
    os.makedirs(out, exist_ok=True)
    json.dump({"auc": aucs, "n": len(rows), "n_switched": int(y.sum())}, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    print(f"\nbest: {max(aucs, key=aucs.get)} = {max(aucs.values()):.3f}  (M-SSA-DS was 0.417, NDVI-amp 0.729)")


if __name__ == "__main__":
    main()
