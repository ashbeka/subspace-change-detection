"""PRE-REGISTERED recovery/dynamics test (config: temporal/configs/L2b_recovery_preregistered.json).

Does temporal DS characterize post-fire RECOVERY / seasonal-dynamics disruption: sustained (unlike a
single NBR step), beating min-angle, and NOT just re-deriving NBR (bands exclude B8/B12)?

d_pre(t) = ||DS(S(t), S_pre)||  with S_pre a pre-fire subspace. Rising at fire = disruption; the post-fire
trajectory = recovery. We compute it on (a) main bands [no B8/B12], (b) strict visible-only bands, vs
min-angle, NBR, and raw mean-reflectance. Burn AOI vs unburned control.

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.real_recovery
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
from temporal.experiments.multiband_gate import mssa_subspace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "real_recovery"); os.makedirs(OUT, exist_ok=True)
CFG = json.load(open(os.path.join(ROOT, "configs", "L2b_recovery_preregistered.json")))
W, L, RANK, ENERGY = CFG["window_W_dates"], CFG["embedding_L"], CFG["rank_dim"], CFG["energy"]
MAIN = CFG["circularity_guard"]["ds_bands_main"]
STRICT = CFG["circularity_guard"]["ds_bands_strict"]
EVENT = dt.date.fromisoformat(CFG["event"]["date"])
START, END = CFG["fetch_range"]


def _days(d): return (dt.date.fromisoformat(d) - EVENT).days


def zscore(X):
    return (X - np.nanmean(X, 1, keepdims=True)) / (np.nanstd(X, 1, keepdims=True) + 1e-9)


def dpre_curve(ts, bands):
    """S_pre from pre-event window; d_pre(t)=||DS(S(t),S_pre)|| and min-angle, over sliding windows."""
    X, dates = gf.as_matrix(ts, bands); Xz = zscore(X); T = Xz.shape[1]
    dd = np.array([_days(d) for d in dates])
    pre_idx = [i for i in range(T) if dd[i] < -30]
    if len(pre_idx) < W:
        return None
    p_end = pre_idx[-1] + 1
    S_pre = mssa_subspace(Xz[:, p_end - W:p_end], L, RANK, energy=ENERGY)
    bd, dpre, ma = [], [], []
    for t in range(0, T - W + 1):
        win = Xz[:, t:t + W]
        if np.isnan(win).any():
            continue
        S = mssa_subspace(win, L, RANK, energy=ENERGY)
        cos = ss.canonical_cosines(S, S_pre)
        bd.append(dates[t]); dpre.append(ss.magnitude(S, S_pre)); ma.append(1.0 - np.max(cos))
    return {"date": bd, "days": np.array([_days(d) for d in bd]), "dpre": np.array(dpre), "minangle": np.array(ma)}


def baseline_stats(days, vals, lo, hi):
    m = (days >= lo) & (days < hi)
    return np.nanmedian(vals[m]) if m.any() else np.nan


def sustained_days(c):
    """Days post-event for which d_pre stays above pre-fire baseline + 2*MAD."""
    pre = c["dpre"][c["days"] < -30]
    if pre.size < 3:
        return None
    thr = np.median(pre) + 2 * (np.median(np.abs(pre - np.median(pre))) + 1e-9)
    post = (c["days"] >= 0)
    above = c["days"][post][c["dpre"][post] > thr]
    return int(above.max() - 0) if above.size else 0


def recovery_slope(days, vals, lo=0, hi=600):
    m = (days >= lo) & (days < hi) & np.isfinite(vals)
    if m.sum() < 4:
        return np.nan
    return float(np.polyfit(days[m], vals[m], 1)[0])


def main():
    sites = {"burn": gf.point_aoi(-16.452, 28.370, 400), "control": gf.point_aoi(-16.630, 28.150, 400)}
    res, series = {}, {}
    for name, aoi in sites.items():
        ts = gf.fetch_timeseries(f"rec_{name}", aoi, START, END)
        nbr_days = np.array([_days(d) for d in ts["dates"]])
        nbr = np.array([np.nan if v is None else float(v) for v in ts["NBR"]])
        rawmean_days = nbr_days
        Xall, _ = gf.as_matrix(ts, MAIN); rawmean = np.nanmean(zscore(Xall), axis=0)
        cm = dpre_curve(ts, MAIN); csg = dpre_curve(ts, STRICT)
        series[name] = {"main": cm, "strict": csg, "nbr_days": nbr_days, "nbr": nbr,
                        "raw_days": rawmean_days, "raw": rawmean}
        res[name] = {
            "n_dates": len(ts["dates"]),
            "dpre_sustained_days_main": sustained_days(cm) if cm else None,
            "dpre_postfire_median_main": baseline_stats(cm["days"], cm["dpre"], 0, 365) if cm else None,
            "dpre_prefire_median_main": baseline_stats(cm["days"], cm["dpre"], -600, -30) if cm else None,
            "minangle_postfire_median_main": baseline_stats(cm["days"], cm["minangle"], 0, 365) if cm else None,
            "minangle_prefire_median_main": baseline_stats(cm["days"], cm["minangle"], -600, -30) if cm else None,
            "dpre_recovery_slope_main": recovery_slope(cm["days"], cm["dpre"]) if cm else None,
            "dpre_recovery_slope_strict": recovery_slope(csg["days"], csg["dpre"]) if csg else None,
            "nbr_recovery_slope": recovery_slope(nbr_days, nbr),
            "rawmean_recovery_slope": recovery_slope(rawmean_days, rawmean),
        }

    # figure
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for ax, name in zip(axes, ["burn", "control"]):
        s = series[name]
        def nz(a): a = np.asarray(a, float); return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a) + 1e-12)
        if s["main"]:
            ax.plot(s["main"]["days"], nz(s["main"]["dpre"]), "C3", lw=2.3, label="d_pre full DS (no B8/B12)")
            ax.plot(s["main"]["days"], nz(s["main"]["minangle"]), "C1", lw=1.3, ls="--", label="d_pre min-angle")
        if s["strict"]:
            ax.plot(s["strict"]["days"], nz(s["strict"]["dpre"]), "C4", lw=1.1, ls=":", label="d_pre DS (visible-only)")
        ax.axvline(0, color="k", lw=1.4, label="fire 2023-08-15")
        ax2 = ax.twinx(); ax2.plot(s["nbr_days"], s["nbr"], "#2a8f5a", lw=1.0, alpha=.6); ax2.set_ylabel("NBR", color="#2a8f5a")
        sd = res[name]["dpre_sustained_days_main"]
        ax.set_title(f"{name}: d_pre sustained {sd} d post-fire | DS slope {res[name]['dpre_recovery_slope_main']:+.2e} "
                     f"| NBR slope {res[name]['nbr_recovery_slope']:+.2e}", fontsize=9.5)
        ax.set_ylabel("normalized"); ax.legend(fontsize=7, loc="upper left")
    axes[1].set_xlabel("days relative to fire (2023-08-15)")
    fig.suptitle("PRE-REGISTERED recovery test: post-fire DS d_pre trajectory (burn vs control), bands exclude NBR's B8/B12", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(OUT, "fig_recovery.png"), dpi=140); plt.close(fig)

    json.dump(res, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))
    # verdict vs pre-registered claims
    b, c = res["burn"], res["control"]
    print("\n=== PRE-REGISTERED CLAIMS ===")
    print(f"C1 sustained: burn d_pre sustained {b['dpre_sustained_days_main']}d vs control {c['dpre_sustained_days_main']}d")
    sep_ds = (b['dpre_postfire_median_main'] - b['dpre_prefire_median_main'])
    sep_ma = (b['minangle_postfire_median_main'] - b['minangle_prefire_median_main'])
    print(f"C3 beats min-angle: burn post-pre rise  DS={sep_ds:+.3f}  min-angle={sep_ma:+.3f}")
    print(f"C2 non-circular: DS recovery slope main={b['dpre_recovery_slope_main']:+.2e} strict={b['dpre_recovery_slope_strict']:+.2e}; "
          f"NBR slope={b['nbr_recovery_slope']:+.2e} (NBR should rise=+, d_pre should fall=- during recovery)")
    print(f"C4 control quiet: control sustained {c['dpre_sustained_days_main']}d (want ~0)")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
