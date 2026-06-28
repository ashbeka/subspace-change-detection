"""SSDS-natural — convert the lead to a result on REAL, CONTINUOUS natural changes (no splice artifact).
Documented multi-year S2 changes: GERD reservoir filling (~2020, gradual land->water) and Creek Fire CA
(Sept 2020, + recovery). Learn D_N from the pre-change normal years; test whether a_hat (M-SSA signal-subspace
DS) localizes the documented change and beats/matches the standard harmonic-deseasonalization change-point
baseline + crude temporal-DS + NDVI mean-shift. This removes the splice-concatenation caveat of ssds_validate.

CONSTRUCTION (ledger): a_hat = generalized DS between past/present M-SSA signal subspaces, referenced to a
learned non-anomalous D_N (pre-change normal). Regular 10-day grid; z-stats + D_N from pre-change only (no leak).

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.ssds_natural
"""
from __future__ import annotations

import json
import os
from datetime import date

import numpy as np

from temporal import signal_subspace_ds as S
from temporal import subspace as ss
from temporal.experiments.ssds_longseries import W, M, R2, STEP, TAU, build_scores, load_resampled, raw_tds
from temporal.experiments.ssds_validate import harmonic_residual, margin, meanshift

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "ssds_natural")
os.makedirs(OUT, exist_ok=True)

LOCATIONS = [("gerd_reservoir", "2020-07-01"), ("creek_fire_ca", "2020-09-05")]
TOL = 12                       # grid pts (~120 days) — long window => coarse localization


def run_location(name, change_iso):
    grid, Xb, ndvi = load_resampled(name)
    T = len(grid); span = W + M - 1
    chg = int(np.argmin(np.abs(grid - date.fromisoformat(change_iso).toordinal())))
    normal_end = chg
    if normal_end < span - 1 + TAU + 6:
        print(f"  {name}: SKIP (only {normal_end} pre-change pts, need >{span-1+TAU+6})")
        return None
    mu, sd = np.nanmean(Xb[:normal_end], 0), np.nanstd(Xb[:normal_end], 0) + 1e-9
    Xz = (Xb - mu) / sd

    ta, aa, _ = build_scores(lambda t: S.mssa_signal_subspace(Xz[:t + 1], W, M, R2), T, span, TAU, normal_end)
    th, hh = harmonic_residual(Xb, normal_end, grid); hcp = meanshift(hh)
    rt_ts, rt = raw_tds(Xz, T)
    nd_ts = np.arange(len(ndvi)); nd_ms = meanshift(ndvi)

    methods = [("a_hat(D_N)", ta, aa), ("harmonic_cp", th, hcp),
               ("crude_TDS", rt_ts, rt), ("NDVI_meanshift", nd_ts, nd_ms)]
    print(f"\n=== {name}: change ~{change_iso} (grid idx {chg}); {normal_end} pre-change pts ===")
    print(f"{'method':>16} {'peak':>12} {'days_off':>9} {'localized':>10} {'margin':>8}")
    out = {}
    for nm, ts, sc in methods:
        loc, mg = margin(ts, sc, chg, tol=TOL)
        peak = int(ts[np.argmax(sc)]); days_off = abs(peak - chg) * STEP
        out[nm] = {"localized": loc, "margin": mg, "days_off": days_off}
        print(f"{nm:>16} {date.fromordinal(int(grid[peak])).isoformat():>12} {days_off:>9} {str(loc):>10} {mg:>8.2f}")
    return out


def main():
    res = {}
    for name, chg in LOCATIONS:
        r = run_location(name, chg)
        if r:
            res[name] = r
    print("\n=== SSDS-NATURAL SUMMARY (real continuous changes; a_hat vs standard harmonic_cp) ===")
    for name, r in res.items():
        a, h = r["a_hat(D_N)"], r["harmonic_cp"]
        verdict = ("a_hat BEATS harmonic" if a["localized"] and a["margin"] > h["margin"]
                   else "a_hat MATCHES harmonic" if a["localized"] and h["localized"]
                   else "a_hat localizes, harmonic doesn't" if a["localized"] and not h["localized"]
                   else "a_hat FAILS to localize")
        print(f"  {name}: a_hat loc={a['localized']} margin={a['margin']:.2f} (off {a['days_off']}d) | "
              f"harmonic_cp loc={h['localized']} margin={h['margin']:.2f} -> {verdict}")
    n_loc = sum(1 for r in res.values() if r["a_hat(D_N)"]["localized"])
    print(f"\n  a_hat localized {n_loc}/{len(res)} real natural changes.")
    json.dump(res, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
