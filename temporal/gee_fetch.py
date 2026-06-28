"""Fetch cloud-masked Harmonized Sentinel-2 surface-reflectance time series for an AOI via Earth Engine.

Stage 2 of the L2 pipeline (docs/DESIGN_TEMPORAL_DS_ACCV2026.md, METHOD.md). Returns a per-date table of
band/index means over a small AOI, cloud-masked with s2cloudless + SCL, scaled to reflectance. Caches to
temporal/data/ so we never re-hit the GEE quota for the same AOI/range.

Auth: a one-time `earthengine authenticate`; this module calls ee.Initialize(project=PROJECT).
Project: subspace-change-detection.

Run smoke: cd <worktree> && <venv>/python.exe -m temporal.gee_fetch
"""
from __future__ import annotations

import json
import os

import numpy as np

PROJECT = "subspace-change-detection"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temporal", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Reflectance bands kept (10/20 m). NBR=(B8-B12)/(B8+B12); NDVI=(B8-B4)/(B8+B4).
REFLECT_BANDS = ["B2", "B3", "B4", "B5", "B8", "B8A", "B11", "B12"]
_SCL_BAD = [3, 8, 9, 10]          # cloud-shadow, cloud med/high, cirrus
_CLOUD_PROB_THRESH = 40           # s2cloudless probability %


def _init():
    import ee
    try:
        ee.Initialize(project=PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=PROJECT)
    return ee


def _mask_clouds(ee, img):
    scl = img.select("SCL")
    scl_ok = scl.remap(_SCL_BAD, [0] * len(_SCL_BAD), 1)            # 1 where good, 0 where bad class
    prob = ee.Image(img.get("s2cloudless")).select("probability")
    prob_ok = prob.lt(_CLOUD_PROB_THRESH)
    return img.updateMask(scl_ok).updateMask(prob_ok)


def fetch_timeseries(name, geometry, start, end, scale=20, cloud_pct_scene=60, force=False):
    """Fetch a cloud-masked S2 time series over `geometry`.

    geometry: ee.Geometry (or call helper point_aoi / bbox_aoi). start/end: 'YYYY-MM-DD'.
    Returns dict: {dates: [...], bands: {B2: [...], ...}, NBR: [...], NDVI: [...]} with NaN for fully
    masked dates dropped. Cached to temporal/data/<name>_<start>_<end>.json.
    """
    cache = os.path.join(DATA_DIR, f"{name}_{start}_{end}.json")
    if os.path.exists(cache) and not force:
        with open(cache) as f:
            return json.load(f)

    ee = _init()
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(geometry).filterDate(start, end)
          .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_pct_scene)))
    clouds = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
              .filterBounds(geometry).filterDate(start, end))
    joined = ee.Join.saveFirst("s2cloudless").apply(
        s2, clouds, ee.Filter.equals(leftField="system:index", rightField="system:index"))
    col = ee.ImageCollection(joined).map(lambda im: _mask_clouds(ee, im))

    def to_feat(img):
        refl = img.select(REFLECT_BANDS).divide(10000.0)
        nbr = refl.normalizedDifference(["B8", "B12"]).rename("NBR")
        ndvi = refl.normalizedDifference(["B8", "B4"]).rename("NDVI")
        stats = refl.addBands([nbr, ndvi]).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometry, scale=scale, maxPixels=int(1e9))
        return ee.Feature(None, stats).set("date", img.date().format("YYYY-MM-dd"))

    feats = ee.FeatureCollection(col.map(to_feat)).getInfo()["features"]
    rows = []
    for ft in feats:
        p = ft["properties"]
        if p.get("NBR") is None:                      # fully masked date
            continue
        rows.append(p)
    rows.sort(key=lambda r: r["date"])

    out = {"name": name, "start": start, "end": end, "scale": scale,
           "dates": [r["date"] for r in rows],
           "bands": {b: [r.get(b) for r in rows] for b in REFLECT_BANDS},
           "NBR": [r.get("NBR") for r in rows], "NDVI": [r.get("NDVI") for r in rows]}
    with open(cache, "w") as f:
        json.dump(out, f)
    return out


def point_aoi(lon, lat, buffer_m=300):
    ee = _init()
    return ee.Geometry.Point([lon, lat]).buffer(buffer_m)


def bbox_aoi(lon_min, lat_min, lon_max, lat_max):
    ee = _init()
    return ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])


def as_matrix(ts, bands=None):
    """Return (B, T) float array of selected band series with NaNs, plus the date list."""
    bands = bands or REFLECT_BANDS
    X = np.array([[np.nan if v is None else float(v) for v in ts["bands"][b]] for b in bands], float)
    return X, ts["dates"]


if __name__ == "__main__":
    # SMOKE: ~7 months over a tiny AOI (central California), confirm we get a cloud-masked series.
    aoi = point_aoi(-119.7, 36.7, buffer_m=300)
    ts = fetch_timeseries("smoke_ca", aoi, "2023-05-01", "2023-12-01", force=True)
    X, dates = as_matrix(ts, ["B4", "B8", "B12"])
    print(f"valid cloud-free dates: {len(dates)}")
    print(f"first/last: {dates[0]} .. {dates[-1]}")
    print(f"band matrix shape (B,T): {X.shape}; NBR range "
          f"[{np.nanmin(ts['NBR']):.3f}, {np.nanmax(ts['NBR']):.3f}]")
    print("OK -> cached to temporal/data/")
