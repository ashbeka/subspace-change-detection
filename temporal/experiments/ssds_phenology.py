"""SSDS-phenology — the FINAL FAIR test of a_hat on its actual niche: a change in the seasonal OSCILLATION
STRUCTURE (not a level/spectral shift). Real Saudi irrigation-onset sites: flat desert (amp~0.02, 2018-2022) ->
a NEW seasonal cycle emerges (amp 0.1-0.26, 2023-2024). This is the kind of change a_hat is built for and that
the fire (abrupt) / reservoir (signal-loss) did NOT test.

QUESTION: does a_hat detect the EMERGENCE of the seasonal cycle (post-onset windows >> flat pre-onset windows),
and does it match/beat the standard harmonic-deseasonalization baseline? Metric = AUC separating post-onset vs
pre-onset windows + post/pre mean ratio. D_N + z-stats from the flat pre-onset years (no leakage).

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.ssds_phenology
"""
from __future__ import annotations

import json
import os
from datetime import date

import numpy as np
from sklearn.metrics import roc_auc_score

from temporal import signal_subspace_ds as S
from temporal.experiments.ssds_longseries import W, M, R2, STEP, TAU, build_scores, load_resampled, raw_tds
from temporal.experiments.ssds_validate import harmonic_residual, meanshift

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "ssds_phenology")
os.makedirs(OUT, exist_ok=True)

SITES = [("saudi2", "2023-01-01"), ("saudi3", "2023-01-01"), ("saudi_jawf", "2023-01-01")]


def run(name, change_iso):
    grid, Xb, ndvi = load_resampled(name)
    T = len(grid); span = W + M - 1
    chg = int(np.argmin(np.abs(grid - date.fromisoformat(change_iso).toordinal())))
    normal_end = chg
    mu, sd = np.nanmean(Xb[:normal_end], 0), np.nanstd(Xb[:normal_end], 0) + 1e-9
    Xz = (Xb - mu) / sd
    ta, aa, _ = build_scores(lambda t: S.mssa_signal_subspace(Xz[:t + 1], W, M, R2), T, span, TAU, normal_end)
    th, hh = harmonic_residual(Xb, normal_end, grid)
    rt_ts, rt = raw_tds(Xz, T)
    methods = [("a_hat(D_N)", ta, aa), ("harmonic_resid", th, hh), ("harmonic_cp", th, meanshift(hh)),
               ("crude_TDS", rt_ts, rt), ("NDVI_meanshift", np.arange(T), meanshift(ndvi))]
    print(f"\n=== {name}: seasonality-onset ~{change_iso} (grid idx {chg}); pre {normal_end} pts / post {T-chg} ===")
    print(f"{'method':>16} {'AUC(post/pre)':>14} {'post/pre ratio':>15}")
    out = {}
    for nm, ts, sc in methods:
        pre = ts < chg; post = ts >= chg
        y = np.r_[np.zeros(pre.sum()), np.ones(post.sum())]; s = np.r_[sc[pre], sc[post]]
        auc = float(max(roc_auc_score(y, s), 1 - roc_auc_score(y, s)))
        ratio = float(np.mean(sc[post]) / (np.mean(sc[pre]) + 1e-12))
        out[nm] = {"auc": auc, "ratio": ratio}
        print(f"{nm:>16} {auc:>14.3f} {ratio:>15.2f}")
    return out


def main():
    res = {name: run(name, chg) for name, chg in SITES}
    print("\n=== SSDS-PHENOLOGY SUMMARY (a_hat on its actual niche; AUC post-onset vs flat pre) ===")
    wins = 0
    for name, r in res.items():
        a, h = r["a_hat(D_N)"], r["harmonic_resid"]
        tag = ("a_hat BEATS harmonic" if a["auc"] > h["auc"] + 0.03
               else "a_hat MATCHES harmonic" if a["auc"] >= h["auc"] - 0.03 else "harmonic better")
        wins += a["auc"] >= h["auc"] - 0.03
        print(f"  {name}: a_hat AUC={a['auc']:.3f} (x{a['ratio']:.1f}) | harmonic AUC={h['auc']:.3f} -> {tag}")
    a_detects = sum(1 for r in res.values() if r["a_hat(D_N)"]["auc"] > 0.7)
    print(f"\n  a_hat detects the seasonality-onset (AUC>0.7) on {a_detects}/{len(res)} sites; "
          f"matches/beats harmonic on {wins}/{len(res)}.")
    json.dump(res, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
