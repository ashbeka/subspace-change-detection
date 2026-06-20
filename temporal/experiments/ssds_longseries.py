"""SSDS long-series FAIR test of the faithful Kanai signal-subspace DS (with learned non-anomalous D_N) on
MULTI-YEAR real S2 — the test the 84-date Tenerife series could not support.

THE decisive claim: D_N (the normal seasonal difference-subspace) models out the RECURRING seasonal cycle, so
normal seasons do NOT false-alarm — exactly what killed the crude temporal-DS. Two tests:

  TEST 1 (seasonality false-alarm).  Long strongly-seasonal series (Iowa Corn Belt, 7 yr). Learn D_N from the
    first ~3 normal years; score the remaining NORMAL years. A seasonality-ROBUST method keeps a_hat FLAT and
    DECORRELATED from the seasonal cycle; a fooled method spikes every green-up/senescence. Metric: |corr(score,
    |dNDVI/dt|)| and coefficient-of-variation over the held-out normal period. Compare a_hat (D_N) vs SSA
    mean-angle (no D_N) vs crude band-subspace velocity (the version that FAILED) vs NDVI |diff|.
  TEST 2 (controlled detection).  Splice Corn Belt (seasonal) -> Amazon (different phenology) at a change point;
    train D_N on Corn Belt normal. Does a_hat spike at the splice while staying flat on the seasons? margin metric.

CONSTRUCTION (ledger): NDVI-SSA = Hankel(w x M) of 1D NDVI -> top-r signal subspace (temporal structure);
M-SSA = stacked per-band Hankel (B*w x M) -> top-r (joint spectral-temporal). w=36 @10-day step = ~1 yr window.
Regular-grid resampling first (SSA needs regular sampling). z-scoring uses NORMAL-period stats only (no leakage).

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.ssds_longseries
"""
from __future__ import annotations

import json
import os
from datetime import date

import numpy as np

from temporal import signal_subspace_ds as S
from temporal import subspace as ss

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "ssds_longseries")
os.makedirs(OUT, exist_ok=True)

STEP = 10                       # day grid
W, M, R1, R2, TAU = 36, 18, 4, 8, 6      # span = 53 (~1.45 yr); window ~1 yr
WTD, KTD = 9, 2                # crude raw-temporal-DS window/rank


def load_resampled(name):
    d = json.load(open(f"temporal/data/{name}_2018-01-01_2024-12-31.json"))
    bands = list(d["bands"].keys())
    Xb = np.array([d["bands"][b] for b in bands], float).T
    ndvi = np.array(d["NDVI"], float)
    ords = np.array([date.fromisoformat(s).toordinal() for s in d["dates"]], float)
    grid = np.arange(ords[0], ords[-1], STEP)
    rs = lambda a: np.interp(grid, ords, a)
    Xb = np.stack([rs(Xb[:, i]) for i in range(Xb.shape[1])], axis=1)
    return grid, Xb, rs(ndvi)


def zfit(a, end):
    return np.nanmean(a[:end], 0), np.nanstd(a[:end], 0) + 1e-9


def build_scores(subspace_at, T, span, tau, normal_end, c=None):
    cache = {}
    P = lambda t: cache.setdefault(t, subspace_at(t))
    valid = list(range(span - 1 + tau, T))
    Dn, mun = [], []
    for t in valid:
        if t < normal_end:
            D = ss.difference_subspace(P(t - tau), P(t))
            if D.shape[1]:
                Dn.append(D); mun.append(S._mu(P(t - tau), P(t)))
    G = sum(D @ D.T for D in Dn); vals, vecs = np.linalg.eigh(G)
    nor = max(1, int(np.median([D.shape[1] for D in Dn])))
    D_N = vecs[:, ::-1][:, :nor]; mu_N = float(np.mean(mun))
    ts, a, ma = [], [], []
    for t in valid:
        Pp, Pc = P(t - tau), P(t)
        ma.append(float(np.mean(1 - ss.canonical_cosines(Pp, Pc))))
        D = ss.difference_subspace(Pp, Pc)
        if D.shape[1] == 0:
            a.append(0.0); ts.append(t); continue
        cref = ss.canonical_cosines(D, D_N); cc = c or len(cref)
        a.append(((S._mu(Pp, Pc) - mu_N) ** 2) * float(np.mean(1 - cref[:cc]))); ts.append(t)
    return np.array(ts), np.array(a), np.array(ma)


def raw_tds(Xz, T):
    subs = [ss.pca_subspace(Xz[t - WTD + 1:t + 1].T, dim=KTD, center=False) for t in range(WTD - 1, T)]
    vel = np.array([ss.magnitude(subs[i], subs[i + 1]) for i in range(len(subs) - 1)])
    return np.arange(WTD, T), vel


def seasonality_metrics(ts, score, dndvi_full, mask):
    """|corr with |dNDVI|| and coefficient-of-variation over the held-out-normal window `mask` (lower=robuster)."""
    s = score[mask]; dd = dndvi_full[ts][mask]
    s = s - s.mean(); dd = dd - dd.mean()
    corr = abs(float(np.sum(s * dd) / (np.sqrt(np.sum(s * s) * np.sum(dd * dd)) + 1e-12)))
    sc = score[mask]
    cv = float(np.std(sc) / (np.mean(sc) + 1e-12))
    return corr, cv


def main():
    grid, Xb, ndvi = load_resampled("cornbelt_iowa")
    T = len(grid); span = W + M - 1
    normal_end = int(0.42 * T)                       # first ~3 years used for D_N + z-fit
    print(f"cornbelt: T={T} pts @ {STEP}d ({(T*STEP/365):.1f} yr); D_N/normal from first {normal_end} ({normal_end*STEP/365:.1f} yr)")

    mu, sd = zfit(Xb, normal_end); Xz = (Xb - mu) / sd
    nmu, nsd = zfit(ndvi.reshape(-1, 1), normal_end); ndz = ((ndvi.reshape(-1, 1) - nmu) / nsd).ravel()
    dndvi = np.abs(np.gradient(ndvi))                # seasonal-transition proxy (full grid)

    ts_n, a_ndvi, ma_ndvi = build_scores(lambda t: S.signal_subspace(ndz[:t + 1], W, M, R1), T, span, TAU, normal_end)
    ts_m, a_mssa, ma_mssa = build_scores(lambda t: S.mssa_signal_subspace(Xz[:t + 1], W, M, R2), T, span, TAU, normal_end)
    rt_ts, rt = raw_tds(Xz, T)
    dts = np.arange(1, T); ndvidiff = np.abs(np.diff(ndvi))

    # ---- TEST 1: seasonality false-alarm over HELD-OUT normal years (t >= normal_end) ----
    print("\n=== TEST 1: seasonality false-alarm (held-out normal years; LOWER = robuster) ===")
    print(f"{'method':>16} {'|corr dNDVI|':>12} {'CV':>8}")
    rows = {}
    for name, ts, sc in [("a_hat_ndvi", ts_n, a_ndvi), ("a_hat_mssa", ts_m, a_mssa),
                         ("meanang_ndvi", ts_n, ma_ndvi), ("meanang_mssa", ts_m, ma_mssa),
                         ("raw_TDS(failed)", rt_ts, rt), ("NDVI_absdiff", dts, ndvidiff)]:
        mask = ts >= normal_end
        corr, cv = seasonality_metrics(ts, sc, dndvi, mask)
        rows[name] = {"corr": corr, "cv": cv}
        print(f"{name:>16} {corr:>12.3f} {cv:>8.2f}")

    # ---- TEST 2: controlled detection via splice (cornbelt -> amazon) ----
    g2, Xa, nda = load_resampled("amazon_evergreen")
    Xa_i = np.stack([np.interp(grid, g2, Xa[:, i]) for i in range(Xa.shape[1])], axis=1)  # amazon on cornbelt grid
    sp = int(0.62 * T)
    Xspl = np.vstack([Xb[:sp], Xa_i[sp:]])
    Xsz = (Xspl - mu) / sd
    ts_s, a_s, ma_s = build_scores(lambda t: S.mssa_signal_subspace(Xsz[:t + 1], W, M, R2), T, span, TAU, normal_end)
    rts_ts, rts = raw_tds(Xsz, T)

    def margin(ts, sc, chg, tol=6):
        ts = np.asarray(ts); near = np.abs(ts - chg) <= tol; far = np.abs(ts - chg) > tol + 2
        fr = float(np.max(sc[near])) if near.any() else 0.0
        bg = float(np.percentile(sc[far], 90)) if far.any() else 1e-9
        return bool(abs(int(ts[np.argmax(sc)]) - chg) <= tol), fr / (bg + 1e-12)

    print(f"\n=== TEST 2: controlled detection (splice at idx {sp}, {sp*STEP/365:.1f} yr) ===")
    print(f"{'method':>16} {'localized':>10} {'margin':>8}")
    for name, ts, sc in [("a_hat_mssa", ts_s, a_s), ("meanang_mssa", ts_s, ma_s), ("raw_TDS", rts_ts, rts)]:
        loc, mg = margin(ts, sc, sp)
        rows[f"splice_{name}"] = {"localized": loc, "margin": mg}
        print(f"{name:>16} {str(loc):>10} {mg:>8.2f}")

    # ---- NULL-SPLICE CONTROL: splice Corn Belt to a PHASE-ALIGNED earlier Corn Belt stretch (NO real change) ----
    period = int(round(365.25 / STEP))                # ~1 year in grid points
    seg2 = Xb[sp - period: sp - period + (T - sp)]    # same phenology, phase-aligned at the join
    Xnull = np.vstack([Xb[:sp], seg2]); Xnz = (Xnull - mu) / sd
    ts_z, a_z, _ = build_scores(lambda t: S.mssa_signal_subspace(Xnz[:t + 1], W, M, R2), T, span, TAU, normal_end)
    loc_z, mg_z = margin(ts_z, a_z, sp)
    rows["nullsplice_a_hat_mssa"] = {"localized": loc_z, "margin": mg_z}
    print(f"\n=== NULL-SPLICE CONTROL (no real change at the join; a_hat must NOT fire) ===")
    print(f"     a_hat_mssa localized={loc_z} margin={mg_z:.2f}  (real-splice margin was {rows['splice_a_hat_mssa']['margin']:.2f})")

    print("\n=== VERDICT ===")
    rob_vs_scalar = rows["a_hat_mssa"]["corr"] < 0.3 < rows["NDVI_absdiff"]["corr"]
    dn_detect = rows["splice_a_hat_mssa"]["margin"] > 1.5 * max(rows["splice_meanang_mssa"]["margin"],
                                                                rows["splice_raw_TDS"]["margin"])
    det = rows["splice_a_hat_mssa"]["localized"]
    null_ok = (not rows["nullsplice_a_hat_mssa"]["localized"]) or \
        rows["nullsplice_a_hat_mssa"]["margin"] < 0.6 * rows["splice_a_hat_mssa"]["margin"]
    print(f"  a_hat seasonality-robust vs scalar NDVI-diff: {rob_vs_scalar}")
    print(f"  D_N detection advantage (a_hat margin >> no-D_N / crude): {dn_detect}")
    print(f"  localizes the real change: {det}")
    print(f"  NULL-SPLICE control PASSES (no false fire on no-change join): {null_ok}")
    print(f"  => {'GENUINE LEAD (real change detected, null-splice clean)' if (det and dn_detect and null_ok) else 'NOT clean — see null-splice / margins'}")
    json.dump(rows, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
