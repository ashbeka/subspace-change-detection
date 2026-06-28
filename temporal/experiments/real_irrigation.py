"""L3 PRE-REGISTERED test: DS in its verified niche on a REAL LABELED dynamics change = irrigation
start/stop (IrrMapper). Irrigation onset adds a growing-season cycle (multi-harmonic gain); cessation
removes it. Config FROZEN: temporal/configs/L3_irrigation_preregistered.json.

Pipeline: sample switched/constant pixels from IrrMapper (GEE) -> fetch S2 series each -> M-SSA DS
change-score vs min-angle / NDVI-amplitude / trivial -> AUC(switched vs constant) + year localization +
attribution. Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.real_irrigation
"""
from __future__ import annotations

import datetime as dt
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from temporal import gee_fetch as gf
from temporal import subspace as ss
from temporal.experiments.multiband_gate import mssa_subspace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "real_irrigation"); os.makedirs(OUT, exist_ok=True)
CFG = json.load(open(os.path.join(ROOT, "configs", "L3_irrigation_preregistered.json")))
W, L, RANK, ENERGY = CFG["window_W_dates"], CFG["embedding_L"], CFG["rank_dim"], CFG["energy"]
BANDS = CFG["bands_for_subspace"]
YEARS = CFG["years"]
SITES_CACHE = os.path.join(gf.DATA_DIR, f"irr_sites_{CFG['region_name']}.json")


def sample_sites(n_per=6):
    if os.path.exists(SITES_CACHE):
        return json.load(open(SITES_CACHE))
    ee = gf._init()
    ic = ee.ImageCollection("UMT/Climate/IrrMapper_RF/v1_2")
    region = ee.Geometry.Rectangle(CFG["region_bbox"])

    def irr(yr):
        im = ic.filterBounds(region).filterDate(f"{yr}-01-01", f"{yr}-12-31").mosaic()
        return im.select("classification").eq(0).rename(f"y{yr}").unmask(0)
    stack = ee.Image.cat([irr(y) for y in YEARS])
    early = stack.select([f"y{YEARS[0]}", f"y{YEARS[1]}"]).reduce(ee.Reducer.sum())
    late = stack.select([f"y{YEARS[-2]}", f"y{YEARS[-1]}"]).reduce(ee.Reducer.sum())
    tot = stack.reduce(ee.Reducer.sum())
    cat = (early.eq(0).And(late.eq(2)).multiply(1)            # switched ON
           .where(early.eq(2).And(late.eq(0)), 2)             # switched OFF
           .where(tot.eq(len(YEARS)), 3)                      # constant irrigated
           .where(tot.eq(0), 4)                               # constant dryland
           .rename("cat").selfMask())
    samp = (cat.addBands(stack).addBands(ee.Image.pixelLonLat())
            .stratifiedSample(numPoints=n_per, classBand="cat", region=region,
                              scale=30, seed=42, geometries=False).getInfo()["features"])
    sites = []
    for f in samp:
        p = f["properties"]; catv = int(p["cat"])
        yrs = [int(p[f"y{y}"]) for y in YEARS]
        sw = None
        if catv in (1, 2):
            for i in range(1, len(yrs)):
                if yrs[i] != yrs[i - 1]:
                    sw = YEARS[i]; break
        sites.append({"lon": p["longitude"], "lat": p["latitude"], "cat": catv,
                      "switch_year": sw, "yrs": yrs,
                      "switched": catv in (1, 2)})
    json.dump(sites, open(SITES_CACHE, "w"), indent=1)
    return sites


def site_scores(ts, switch_year):
    """Per-site change scores: max DS d1, min-angle; NDVI-amplitude step; trivial. + d1 peak year."""
    X, dates = gf.as_matrix(ts, BANDS)
    Xz = (X - np.nanmean(X, 1, keepdims=True)) / (np.nanstd(X, 1, keepdims=True) + 1e-9)
    T = Xz.shape[1]
    d1, ma, triv, bdates = [], [], [], []
    for t in range(W, T - W + 1):
        past, pres = Xz[:, t - W:t], Xz[:, t:t + W]
        if np.isnan(past).any() or np.isnan(pres).any():
            continue
        Sp, Sq = mssa_subspace(past, L, RANK, energy=ENERGY), mssa_subspace(pres, L, RANK, energy=ENERGY)
        d1.append(ss.magnitude(Sp, Sq)); ma.append(1.0 - np.max(ss.canonical_cosines(Sp, Sq)))
        triv.append(float(np.mean(np.abs(pres.mean(1) - past.mean(1))))); bdates.append(dates[t])
    d1 = np.array(d1); ma = np.array(ma); triv = np.array(triv)
    # NDVI seasonal amplitude per calendar year, then max year-over-year step
    nbr = np.array([np.nan if v is None else float(v) for v in ts["NDVI"]])
    ndates = [dt.date.fromisoformat(d) for d in ts["dates"]]
    amps = {}
    for y in range(ndates[0].year, ndates[-1].year + 1):
        vals = nbr[[i for i, d in enumerate(ndates) if d.year == y]]
        if vals.size >= 5:
            amps[y] = np.nanmax(vals) - np.nanmin(vals)
    ay = sorted(amps); ndvi_step = max([abs(amps[ay[i]] - amps[ay[i - 1]]) for i in range(1, len(ay))], default=0.0)
    # d1 peak year & localization
    peak_year = dt.date.fromisoformat(bdates[int(np.argmax(d1))]).year if d1.size else None
    loc = (abs(peak_year - switch_year) if (peak_year and switch_year) else None)
    return {"d1_max": float(np.max(d1)) if d1.size else 0.0,
            "minangle_max": float(np.max(ma)) if ma.size else 0.0,
            "trivial_max": float(np.max(triv)) if triv.size else 0.0,
            "ndvi_amp_step": float(ndvi_step), "d1_peak_year": peak_year, "loc_years": loc}


def main():
    sites = sample_sites(CFG["n_sites_per_category"])
    print(f"sampled {len(sites)} sites: " + ", ".join(f"cat{c}={sum(1 for s in sites if s['cat']==c)}" for c in (1,2,3,4)))
    rows = []
    for i, s in enumerate(sites):
        aoi = gf.point_aoi(s["lon"], s["lat"], buffer_m=120)
        name = f"irr_{CFG['region_name']}_{i}_cat{s['cat']}"
        try:
            ts = gf.fetch_timeseries(name, aoi, CFG["s2_fetch_range"][0], CFG["s2_fetch_range"][1])
        except Exception as e:
            print(f"  site {i} fetch fail: {str(e)[:80]}"); continue
        if len(ts["dates"]) < 2 * W + 5:
            print(f"  site {i}: too few dates ({len(ts['dates'])}), skip"); continue
        sc = site_scores(ts, s["switch_year"])
        sc.update({"i": i, "cat": s["cat"], "switched": s["switched"], "switch_year": s["switch_year"],
                   "n_dates": len(ts["dates"])})
        rows.append(sc)
        print(f"  site {i} cat{s['cat']} sw{s['switch_year']}: d1_max={sc['d1_max']:.2f} "
              f"minang={sc['minangle_max']:.3f} ndvi_step={sc['ndvi_amp_step']:.3f} loc={sc['loc_years']}")

    y = np.array([1 if r["switched"] else 0 for r in rows])
    if y.sum() == 0 or y.sum() == len(y):
        print("not enough of each class; abort eval"); return
    aucs = {m: roc_auc_score(y, [r[m] for r in rows]) for m in ["d1_max", "minangle_max", "ndvi_amp_step", "trivial_max"]}
    locs = [r["loc_years"] for r in rows if r["switched"] and r["loc_years"] is not None]
    res = {"n_sites": len(rows), "n_switched": int(y.sum()), "auc": aucs,
           "median_loc_years": float(np.median(locs)) if locs else None,
           "within_tol_frac": float(np.mean([l <= CFG["tolerance_years"] for l in locs])) if locs else None}
    json.dump({"summary": res, "rows": rows}, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print("\n=== L3 IRRIGATION VERDICT (vs pre-registered claims) ===")
    print(f"AUC(switched vs constant): DS d1={aucs['d1_max']:.3f}  min-angle={aucs['minangle_max']:.3f}  "
          f"NDVI-amp={aucs['ndvi_amp_step']:.3f}  trivial={aucs['trivial_max']:.3f}")
    print(f"C1 detect (DS AUC>0.7): {'PASS' if aucs['d1_max']>0.7 else 'FAIL'}")
    print(f"C3 DS>=min-angle & >=NDVI: {'PASS' if aucs['d1_max']>=aucs['minangle_max'] and aucs['d1_max']>=aucs['ndvi_amp_step'] else 'FAIL'}")
    print(f"C2 localize (median loc {res['median_loc_years']}y, within +-1y frac {res['within_tol_frac']})")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
