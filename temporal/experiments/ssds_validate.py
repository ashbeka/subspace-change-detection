"""SSDS-validate — the make-or-break test of the signal-subspace DS (a_hat, with D_N) lead against STRONG
baselines on the 7-yr S2 data. The natural competitor for time-series SATELLITE change detection is HARMONIC
DESEASONALIZATION (BFAST/CCDC family): fit the seasonal cycle on the normal period, flag the residual — the
scalar-world version of exactly what D_N does. Plus a windowed SFA-CD. (IR-MAD is a spatial/per-pixel method and
my impl has a known additive-change bug, so it is not a fair time-series bar here; harmonic + SFA are the bars.)

Three tests (all on Iowa Corn Belt 7-yr, regular 10-day grid; Amazon for the splice):
  T1 seasonality false-alarm: |corr(score, |dNDVI/dt|)| over held-out normal years (lower = robuster).
  T2 controlled detection: splice CornBelt->Amazon; margin + localization.
  T3 null-splice control: CornBelt->phase-aligned CornBelt (no real change); must NOT localize.
QUESTION: does a_hat (subspace D_N) BEAT or MATCH harmonic deseasonalization? If harmonic matches it, the
subspace machinery adds nothing over the standard method (the project's recurring pattern). If a_hat wins, real edge.

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.ssds_validate
"""
from __future__ import annotations

import json
import os

import numpy as np

from temporal import signal_subspace_ds as S
from temporal import subspace as ss
from temporal.experiments.hsi_nuisance_invariance import sfa_cd
from temporal.experiments.ssds_longseries import (W, M, R2, STEP, TAU, build_scores, load_resampled)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "ssds_validate")
os.makedirs(OUT, exist_ok=True)


def harmonic_residual(X, normal_end, grid, K=3):
    """Multivariate harmonic deseasonalization (BFAST/CCDC-style). Fit a+bt+sum_k(sin,cos) per band on the
    NORMAL period; change score = L2 norm over bands of the residual. Models the seasonal cycle (like D_N)."""
    t = (grid - grid[0]) / 365.25
    cols = [np.ones_like(t), t]
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * t), np.cos(2 * np.pi * k * t)]
    A = np.stack(cols, axis=1)
    res = np.zeros_like(X)
    for b in range(X.shape[1]):
        coef, *_ = np.linalg.lstsq(A[:normal_end], X[:normal_end, b], rcond=None)
        res[:, b] = X[:, b] - A @ coef
    return np.arange(len(grid)), np.linalg.norm(res, axis=1)


def sfa_windows(Xz, span, tau):
    ts, sc = [], []
    for t in range(span - 1 + tau, len(Xz)):
        Pw = Xz[t - tau - span + 1:t - tau + 1]; Qw = Xz[t - span + 1:t + 1]
        sc.append(float(np.mean(sfa_cd(Pw, Qw)))); ts.append(t)
    return np.array(ts), np.array(sc)


def meanshift(s, k=6):
    """Sliding mean-shift |mean(after)-mean(before)| — converts a STEP (harmonic residual) into a change-POINT
    spike, the fair BFAST/CCDC-style breakpoint form. a_hat is already a change-point score."""
    out = np.zeros_like(s, dtype=float)
    for t in range(len(s)):
        a = s[max(0, t - k):t]; b = s[t:t + k]
        if len(a) and len(b):
            out[t] = abs(float(b.mean()) - float(a.mean()))
    return out


def margin(ts, sc, chg, tol=6):
    ts = np.asarray(ts); near = np.abs(ts - chg) <= tol; far = np.abs(ts - chg) > tol + 2
    fr = float(np.max(sc[near])) if near.any() else 0.0
    bg = float(np.percentile(sc[far], 90)) if far.any() else 1e-9
    return bool(abs(int(ts[np.argmax(sc)]) - chg) <= tol), fr / (bg + 1e-12)


def seas_corr(ts, sc, dndvi, mask):
    s = sc[mask] - sc[mask].mean(); dd = dndvi[ts][mask] - dndvi[ts][mask].mean()
    return abs(float(np.sum(s * dd) / (np.sqrt(np.sum(s * s) * np.sum(dd * dd)) + 1e-12)))


def main():
    grid, Xb, ndvi = load_resampled("cornbelt_iowa")
    T = len(grid); span = W + M - 1; normal_end = int(0.42 * T)
    mu, sd = np.nanmean(Xb[:normal_end], 0), np.nanstd(Xb[:normal_end], 0) + 1e-9
    Xz = (Xb - mu) / sd
    nmu, nsd = ndvi[:normal_end].mean(), ndvi[:normal_end].std() + 1e-9
    ndz = (ndvi - nmu) / nsd
    dndvi = np.abs(np.gradient(ndvi))
    print(f"T={T}@{STEP}d ({T*STEP/365:.1f}yr); normal/D_N first {normal_end}; methods: a_hat(D_N), harmonic, SFA-win")

    # scores on the plain Corn Belt series (for seasonality test)
    ts_a, a_hat, _ = build_scores(lambda t: S.mssa_signal_subspace(Xz[:t + 1], W, M, R2), T, span, TAU, normal_end)
    ts_h, harm = harmonic_residual(Xb, normal_end, grid)
    ts_s, sfa = sfa_windows(Xz, span, tau=TAU)
    mask_a = ts_a >= normal_end; mask_h = ts_h >= normal_end; mask_s = ts_s >= normal_end

    print("\n=== T1 seasonality false-alarm (|corr dNDVI| over held-out normal; LOWER=robuster) ===")
    rows = {}
    rows["a_hat"] = {"seas_corr": seas_corr(ts_a, a_hat, dndvi, mask_a)}
    rows["harmonic"] = {"seas_corr": seas_corr(ts_h, harm, dndvi, mask_h)}
    rows["sfa_win"] = {"seas_corr": seas_corr(ts_s, sfa, dndvi, mask_s)}
    for k in rows:
        print(f"  {k:>10}: {rows[k]['seas_corr']:.3f}")

    # splice + null-splice
    g2, Xa, _ = load_resampled("amazon_evergreen")
    Xa_i = np.stack([np.interp(grid, g2, Xa[:, i]) for i in range(Xa.shape[1])], axis=1)
    sp = int(0.62 * T)
    period = int(round(365.25 / STEP))
    Xspl = np.vstack([Xb[:sp], Xa_i[sp:]])
    Xnull = np.vstack([Xb[:sp], Xb[sp - period:sp - period + (T - sp)]])

    def run_change(Xchg, label):
        Xzc = (Xchg - mu) / sd
        ta, aa, _ = build_scores(lambda t: S.mssa_signal_subspace(Xzc[:t + 1], W, M, R2), T, span, TAU, normal_end)
        th, hh = harmonic_residual(Xchg, normal_end, grid)
        tsf, sf = sfa_windows(Xzc, span, tau=TAU)
        out = {}
        for nm, ts, sc in [("a_hat", ta, aa), ("harmonic_resid", th, hh),
                           ("harmonic_cp", th, meanshift(hh)), ("sfa_win", tsf, sf)]:
            loc, mg = margin(ts, sc, sp); out[nm] = {"localized": loc, "margin": mg}
        print(f"\n=== {label} (change idx {sp}, {sp*STEP/365:.1f}yr) ===")
        print(f"{'method':>10} {'localized':>10} {'margin':>8}")
        for nm in out:
            print(f"{nm:>10} {str(out[nm]['localized']):>10} {out[nm]['margin']:>8.2f}")
        return out

    spl = run_change(Xspl, "T2 detection (splice CornBelt->Amazon)")
    nul = run_change(Xnull, "T3 null-splice control (no real change)")
    for nm in spl:
        rows.setdefault(nm, {}).update({"splice_loc": spl[nm]["localized"], "splice_margin": spl[nm]["margin"],
                                        "null_loc": nul[nm]["localized"], "null_margin": nul[nm]["margin"]})

    print("\n=== VERDICT (vs the FAIR change-point harmonic baseline = mean-shift of the deseasonalized residual) ===")
    a, h = rows["a_hat"], rows["harmonic_cp"]
    print(f"  seasonality:  a_hat {a['seas_corr']:.3f} vs harmonic-resid {rows['harmonic']['seas_corr']:.3f} (lower=robuster)")
    print(f"  splice detect: a_hat loc={a['splice_loc']} margin={a['splice_margin']:.2f} | "
          f"harmonic_cp loc={h['splice_loc']} margin={h['splice_margin']:.2f}")
    print(f"  null-splice:  a_hat loc={a['null_loc']} margin={a['null_margin']:.2f} | "
          f"harmonic_cp loc={h['null_loc']} margin={h['null_margin']:.2f}")
    a_wins = a["splice_loc"] and a["splice_margin"] >= h["splice_margin"]
    a_matches = a["splice_loc"] and a["splice_margin"] >= 0.8 * h["splice_margin"]
    print(f"  => a_hat BEATS fair harmonic change-point: {a_wins} | MATCHES (>=80%): {a_matches}")
    print(f"     (if harmonic_cp clearly wins, the subspace machinery adds nothing over the standard method)")
    json.dump(rows, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
