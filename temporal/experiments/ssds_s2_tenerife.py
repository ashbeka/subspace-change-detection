"""SSDS-S2 — the FAITHFUL Kanai signal-subspace DS (with learned non-anomalous reference D_N) on a REAL
Sentinel-2 series (Tenerife Arafo 2023 fire), vs the crude temporal-DS that FAILED and trivial nulls.

THE QUESTION (not "can you detect a big fire" — NBR does that trivially): does the D_N-referenced subspace
method separate the fire from the SEASONAL background better than (a) the crude raw temporal-DS velocity that
failed before, and (b) plain SSA angles WITHOUT D_N? i.e. does "model-the-normal, flag-the-residual" earn its keep?

PRE-REGISTRATION:
  Data: 84 cloud-masked S2 dates (2023-01..2024-06), 13 bands + NDVI + NBR, single Arafo burn region (n=1 event).
  Ground-truth fire date = date of the largest NBR drop (NBR = burn index). D_N learned ONLY from pre-fire dates;
  per-band z-scoring uses pre-fire stats only (no leakage).
  Subspace constructions (ledger rows):
    - NDVI-SSA : 1D NDVI -> Hankel (w x M) -> top-r signal subspace. Represents NDVI temporal structure.
    - M-SSA    : 13-band -> stacked per-band Hankel (B*w x M) -> top-r. Represents joint spectral-temporal structure.
    - raw-TDS (the FAILED crude version): per-date-window band-subspace (PCA of B x W_td), velocity = Mag(S_t,S_{t+1}).
  Scores: a_hat = beta*delta (Kanai, with D_N); plus SSA mean-angle & min-angle (NO D_N) to isolate D_N's effect;
          plus trivial nulls NBR|diff|, NDVI|diff|, all-band CVA|diff|.
  METRICS (single series): (1) localization = global peak within +/-2 dates of the fire; (2) margin =
  fire-response / 90th-pct of NON-fire response (seasonality false-alarm margin; higher = more seasonality-robust).
  HYPOTHESIS: a_hat localizes AND has a higher margin than raw-TDS and than SSA-mean-angle (D_N helps).
  FALSIFIERS: a_hat mislocalizes, OR its margin <= raw-TDS / mean-angle (D_N adds nothing) -> faithful method
  does not beat the crude one on real data either.

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.ssds_s2_tenerife
"""
from __future__ import annotations

import json
import os
from datetime import date

import numpy as np

from temporal import signal_subspace_ds as S
from temporal import subspace as ss

DATA = "temporal/data/burn_tenerife_arafo_2023-01-01_2024-06-30.json"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "ssds_s2_tenerife")
os.makedirs(OUT, exist_ok=True)

W, M, R_1D, R_MS, TAU = 12, 6, 3, 6, 2        # span = w+M-1 = 17
WTD, KTD = 8, 2                                # crude raw-temporal-DS window / rank


def zscore_prefire(a, end):
    mu = np.nanmean(a[:end], 0); sd = np.nanstd(a[:end], 0) + 1e-9
    return (a - mu) / sd


def build_scores(subspace_at, T, span, tau, normal_end, c=None):
    cache = {}
    P = lambda t: cache.setdefault(t, subspace_at(t))
    valid = list(range(span - 1 + tau, T))
    Dn, mun = [], []
    for t in valid:
        if t < normal_end:                                         # present window normal (pre-fire)
            D = ss.difference_subspace(P(t - tau), P(t))
            if D.shape[1]:
                Dn.append(D); mun.append(S._mu(P(t - tau), P(t)))
    if not Dn:
        raise RuntimeError("no normal DS for D_N")
    G = sum(D @ D.T for D in Dn); vals, vecs = np.linalg.eigh(G)
    nor = max(1, int(np.median([D.shape[1] for D in Dn])))
    D_N = vecs[:, ::-1][:, :nor]; mu_N = float(np.mean(mun))
    ts, a, ma, mi = [], [], [], []
    for t in valid:
        Pp, Pc = P(t - tau), P(t)
        cab = ss.canonical_cosines(Pp, Pc)
        mi.append(1 - float(np.max(cab))); ma.append(float(np.mean(1 - cab)))
        D = ss.difference_subspace(Pp, Pc)
        if D.shape[1] == 0:
            a.append(0.0); ts.append(t); continue
        cref = ss.canonical_cosines(D, D_N); cc = c or len(cref)
        a.append(((S._mu(Pp, Pc) - mu_N) ** 2) * float(np.mean(1 - cref[:cc]))); ts.append(t)
    return np.array(ts), {"a_hat": np.array(a), "mean_angle": np.array(ma), "min_angle": np.array(mi)}


def raw_tds(Xz, T):
    subs = [ss.pca_subspace(Xz[t - WTD + 1:t + 1].T, dim=KTD, center=False) for t in range(WTD - 1, T)]
    vel = np.array([ss.magnitude(subs[i], subs[i + 1]) for i in range(len(subs) - 1)])
    return np.arange(WTD, T), vel


def metrics(ts, score, fire, tol=2):
    ts = np.asarray(ts); score = np.asarray(score)
    peak = int(ts[np.argmax(score)])
    near = np.abs(ts - fire) <= tol
    far = np.abs(ts - fire) > tol + 1
    fire_resp = float(np.max(score[near])) if near.any() else 0.0
    bg = float(np.percentile(score[far], 90)) if far.any() else 1e-9
    return {"peak": peak, "localized": bool(abs(peak - fire) <= tol), "margin": fire_resp / (bg + 1e-12)}


def main():
    d = json.load(open(DATA))
    raw_dates = d["dates"]; T = len(raw_dates)
    bands = list(d["bands"].keys())
    Xb = np.array([d["bands"][b] for b in bands], float).T          # (T, B)
    ndvi = np.array(d["NDVI"], float); nbr = np.array(d["NBR"], float)

    # FIX: SSA needs a REGULAR grid; cloud-masked S2 dates are irregular -> resample by linear interpolation
    ords = np.array([date.fromisoformat(s).toordinal() for s in raw_dates], float)
    orig_fire_ord = ords[int(np.argmax(-np.diff(nbr)) + 1)]
    grid = np.linspace(ords[0], ords[-1], T)                        # T evenly-spaced points
    resamp = lambda a: np.interp(grid, ords, a)
    Xb = np.stack([resamp(Xb[:, b]) for b in range(Xb.shape[1])], axis=1)
    ndvi = resamp(ndvi); nbr = resamp(nbr)
    dates = [date.fromordinal(int(round(g))).isoformat() for g in grid]
    fire = int(np.argmin(np.abs(grid - orig_fire_ord)))            # fire mapped onto the regular grid
    print(f"T={T} dates (resampled to regular grid, step={(grid[1]-grid[0]):.1f} days), bands={bands}")
    print(f"fire date index={fire} ({dates[fire]}); NBR {nbr[fire-1]:.3f}->{nbr[fire]:.3f}")

    Xz = zscore_prefire(Xb, fire); ndviz = zscore_prefire(ndvi.reshape(-1, 1), fire).ravel()
    span = W + M - 1
    res = {}
    res["a_hat_ndvi"] = build_scores(lambda t: S.signal_subspace(ndviz[:t + 1], W, M, R_1D), T, span, TAU, fire)
    res["a_hat_mssa"] = build_scores(lambda t: S.mssa_signal_subspace(Xz[:t + 1], W, M, R_MS), T, span, TAU, fire)
    # raw temporal-DS (crude, failed version) + trivial nulls
    rt_ts, rt = raw_tds(Xz, T)
    nbr_d = np.abs(np.diff(nbr)); ndvi_d = np.abs(np.diff(ndvi))
    cva = np.linalg.norm(np.diff(Xz, axis=0), axis=1)
    diff_ts = np.arange(1, T)

    print(f"\n{'method':>16} {'peak(date)':>16} {'localized':>10} {'margin':>8}")
    table = {}
    def show(name, ts, sc):
        m = metrics(ts, sc, fire); table[name] = m
        print(f"{name:>16} {dates[m['peak']]:>16} {str(m['localized']):>10} {m['margin']:>8.2f}")
    show("a_hat_ndvi", res["a_hat_ndvi"][0], res["a_hat_ndvi"][1]["a_hat"])
    show("meanang_ndvi", res["a_hat_ndvi"][0], res["a_hat_ndvi"][1]["mean_angle"])
    show("a_hat_mssa", res["a_hat_mssa"][0], res["a_hat_mssa"][1]["a_hat"])
    show("meanang_mssa", res["a_hat_mssa"][0], res["a_hat_mssa"][1]["mean_angle"])
    show("raw_TDS(failed)", rt_ts, rt)
    show("CVA_allband", diff_ts, cva)
    show("NDVI_absdiff", diff_ts, ndvi_d)
    show("NBR_absdiff", diff_ts, nbr_d)

    print("\n=== SSDS-S2 VERDICT (n=1 case study) ===")
    ah = table["a_hat_mssa"]; rt_m = table["raw_TDS(failed)"]; ma = table["meanang_mssa"]
    print(f"  a_hat(M-SSA) localized={ah['localized']} margin={ah['margin']:.2f} | "
          f"raw-TDS localized={rt_m['localized']} margin={rt_m['margin']:.2f} | "
          f"mean-angle margin={ma['margin']:.2f}")
    beats_crude = ah["localized"] and ah["margin"] > rt_m["margin"]
    dn_helps = ah["margin"] > ma["margin"]
    print(f"  faithful beats crude temporal-DS: {beats_crude}")
    print(f"  D_N reference helps over plain mean-angle: {dn_helps}")
    json.dump({k: v for k, v in table.items()}, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
