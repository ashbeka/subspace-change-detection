"""REAL-DATA Gate-0: does the frozen M-SSA d1(t) spike at a documented wildfire date?

Event: 2023 Tenerife wildfire, ignition night of 2023-08-15, Arafo highlands / Corona Forestal pine
forest (>14,000 ha). Burn AOI = pine forest above Arafo; control AOI = unburned southern pine forest.

Construction is FROZEN from temporal/configs/L2_preregistered.json (W=24,L=12,rank=8,energy=0.99). We
tune nothing here. Baselines on the same series: conventional SSA min-angle, trivial reflectance-diff
null, NBR step. Plus per-band attribution (expect SWIR for a burn) and the control-quiet check.

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.real_gate0
"""
from __future__ import annotations

import datetime as dt
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from temporal import gee_fetch as gf
from temporal import subspace as ss
from temporal.experiments.multiband_gate import mssa_subspace, hankel  # frozen construction

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "real_gate0")
os.makedirs(OUT, exist_ok=True)
CFG = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "configs", "L2_preregistered.json")))
W, L, RANK, ENERGY = CFG["window_W_dates"], CFG["embedding_L"], CFG["rank_dim"], CFG["energy"]
BANDS = CFG["bands_for_subspace"]
EVENT = dt.date(2023, 8, 15)
START, END = "2023-01-01", "2024-06-30"


def _days(d):
    return (dt.date.fromisoformat(d) - EVENT).days


def zscore_rows(X):
    mu = np.nanmean(X, axis=1, keepdims=True); sd = np.nanstd(X, axis=1, keepdims=True)
    return (X - mu) / (sd + 1e-9)


def curves(X, dates):
    """Frozen M-SSA sliding-window d1, min-angle; trivial reflectance-diff null; boundary dates."""
    Xz = zscore_rows(X)
    T = Xz.shape[1]
    bd, d1, ma, triv = [], [], [], []
    for t in range(W, T - W + 1):
        past, pres = Xz[:, t - W:t], Xz[:, t:t + W]
        if np.isnan(past).any() or np.isnan(pres).any():
            continue
        Sp = mssa_subspace(past, L, RANK, energy=ENERGY)
        Sq = mssa_subspace(pres, L, RANK, energy=ENERGY)
        cos = ss.canonical_cosines(Sp, Sq)
        bd.append(dates[t]); d1.append(ss.magnitude(Sp, Sq)); ma.append(1.0 - np.max(cos))
        triv.append(float(np.mean(np.abs(pres.mean(1) - past.mean(1)))))  # gain-naive null on z-scored bands
    return {"date": bd, "d1": np.array(d1), "minangle": np.array(ma), "trivial": np.array(triv)}


def attribution(X, dates):
    """Per-band difference-subspace energy at the boundary nearest the event -> which band drives it."""
    Xz = zscore_rows(X); T = Xz.shape[1]
    cand = [t for t in range(W, T - W + 1) if not np.isnan(Xz[:, t - W:t + W]).any()]
    t = min(cand, key=lambda i: abs(_days(dates[i])))
    Sp = mssa_subspace(Xz[:, t - W:t], L, RANK, energy=ENERGY)
    Sq = mssa_subspace(Xz[:, t:t + W], L, RANK, energy=ENERGY)
    D = ss.difference_subspace(Sp, Sq)
    if D.shape[1] == 0:
        return None
    energy = np.array([np.sum(D[b * L:(b + 1) * L, :] ** 2) for b in range(len(BANDS))])
    return {"top_band": BANDS[int(np.argmax(energy))],
            "energy_by_band": {BANDS[i]: float(energy[i]) for i in range(len(BANDS))}}


def peak_localization(c):
    if not len(c["d1"]):
        return None
    i = int(np.argmax(c["d1"]))
    return {"peak_date": c["date"][i], "localization_days": _days(c["date"][i])}


def main():
    sites = {
        "burn_tenerife_arafo": gf.point_aoi(-16.452, 28.370, buffer_m=400),
        "control_vilaflor":    gf.point_aoi(-16.630, 28.150, buffer_m=400),
    }
    results = {}
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for ax, (name, aoi) in zip(axes, sites.items()):
        ts = gf.fetch_timeseries(name, aoi, START, END)
        X, dates = gf.as_matrix(ts, BANDS)
        nbr = np.array([np.nan if v is None else float(v) for v in ts["NBR"]])
        c = curves(X, dates)
        loc = peak_localization(c)
        attr = attribution(X, dates) if name.startswith("burn") else None
        results[name] = {"n_dates": len(dates), "n_boundaries": len(c["d1"]),
                         "peak": loc, "attribution": attr,
                         "d1_peak_over_median": float(np.max(c["d1"]) / (np.median(c["d1"]) + 1e-9)) if len(c["d1"]) else None}
        # plot
        dd = [_days(d) for d in c["date"]]
        def nz(a): a = np.asarray(a, float); return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a) + 1e-12)
        ax.plot(dd, nz(c["d1"]), color="C3", lw=2.3, label="M-SSA d1 (full DS)")
        ax.plot(dd, nz(c["minangle"]), color="C1", lw=1.4, ls="--", label="min-angle SSA")
        ax.plot(dd, nz(c["trivial"]), color="C0", lw=1.2, ls=":", label="trivial reflectance-diff")
        ax.axvline(0, color="k", lw=1.4, label="event 2023-08-15"); ax.axvspan(-5, 5, color="0.85", alpha=.6)
        ax2 = ax.twinx()
        ddn = [_days(d) for d in dates]
        ax2.plot(ddn, nbr, color="#2a8f5a", lw=1.0, alpha=.6); ax2.set_ylabel("NBR", color="#2a8f5a")
        ttl = name
        if loc:
            ttl += f"   |  d1 peak @ {loc['localization_days']:+d} d (peak/median={results[name]['d1_peak_over_median']:.1f})"
        if attr:
            ttl += f"  |  attribution -> {attr['top_band']}"
        ax.set_title(ttl, fontsize=10); ax.set_ylabel("normalized score"); ax.legend(fontsize=7, loc="upper left")
    axes[1].set_xlabel("days relative to event (2023-08-15)")
    fig.suptitle("REAL Gate-0: Tenerife 2023 wildfire — frozen M-SSA d1(t) vs baselines (burn vs control)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, "fig_real_gate0.png"), dpi=140); plt.close(fig)

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    # verdict
    b = results["burn_tenerife_arafo"]; ctl = results["control_vilaflor"]
    print("\n=== GATE-0 VERDICT ===")
    if b["peak"]:
        hit = abs(b["peak"]["localization_days"]) <= CFG["delta_days"]
        print(f"burn d1 peak localization: {b['peak']['localization_days']:+d} days  -> "
              f"{'WITHIN' if hit else 'OUTSIDE'} +-{CFG['delta_days']}d")
        print(f"burn attribution top band: {b['attribution']['top_band'] if b['attribution'] else 'n/a'} "
              f"(SWIR B11/B12 expected for fire)")
        print(f"burn peak/median={b['d1_peak_over_median']:.1f} vs control peak/median={ctl['d1_peak_over_median']:.1f}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
