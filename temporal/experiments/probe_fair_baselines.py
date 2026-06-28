"""ADVERSARIAL PROBE: fair confound-robust baselines vs temporal DS.

Angle: the original L1 (temporal/experiments/synth_injection.py) compares DS distance-to-baseline
against a CRIPPLED baseline -- "trivial raw distance" = mean |x_t - x_ref| on raw reflectance, which
is by construction destroyed by global multiplicative gain + additive haze. That is not a fair fight:
a competent remote-sensing practitioner would never difference raw reflectance under seasonal
illumination; they would normalize or use a gain-invariant index FIRST.

This probe drops in FAIRER baselines on the EXACT SAME tiles / labels / pooled-AUC machinery as the
original evaluate(), on the EXACT SAME confound sweep and magnitude sweep, and asks: does DS still win?

Fair baselines (all per-tile, per-date -> distance-to-baseline against the reference date, scored with
the same roc_auc_score over (tile x date) just like DS d_pre):
  norm_l1   : per-date global normalization (subtract per-date scene mean, divide by per-date scene std)
              of the cube, then mean |x_t - x_ref| over the tile.  Kills global gain+haze.
  zscore_tile: per-tile per-date z-score of the band vector (subtract mean over the tile's flattened
              vector, divide by std) then L1 distance. Removes per-date offset+scale locally.
  ndvi_l1   : mean over adjacent-band normalized differences (b_i - b_j)/(b_i + b_j) -- gain invariant
              by construction (multiplicative gain cancels in the ratio). Distance-to-baseline of that.
  cosine    : 1 - cosine similarity between the tile's mean spectrum at date t and at the ref date.
              Pure-gain invariant (scaling a vector does not change its direction).
  spec_angle: spectral angle = arccos(cosine) between per-pixel-mean spectra. (cosine's monotone twin)

Also reruns the gain_invariance_check analog for each baseline (pure global gain -> should be ~0 for
the gain-invariant ones, exposing that DS's "DS=0 under pure gain" is NOT unique).

Outputs: temporal/outputs/_probe_fair_baselines/
Run: ./.venv/Scripts/python.exe -m temporal.experiments.probe_fair_baselines
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from temporal import subspace as ss
from temporal.synth import SynthConfig, make_scene
from temporal.experiments import synth_injection as L1

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "_probe_fair_baselines")
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------- fair baseline scorers
# Each returns, for a tile (r,c), an array of length T: distance-to-baseline at every date.
# These mirror L1.trivial_distance signature so we can swap them in cleanly.

def _tile_raw(cube, r, c, tile):
    T = cube.shape[0]
    return cube[:, :, r:r + tile, c:c + tile].reshape(T, -1)  # (T, B*tile*tile)


def dist_norm_l1(cube_norm, r, c, tile, ref_t=0):
    """L1 distance to ref date on a PER-DATE GLOBALLY-NORMALIZED cube (passed in already normalized)."""
    blk = _tile_raw(cube_norm, r, c, tile)
    ref = blk[ref_t]
    return np.array([np.mean(np.abs(blk[t] - ref)) for t in range(blk.shape[0])])


def dist_zscore_tile(cube, r, c, tile, ref_t=0):
    """Per-tile per-date z-score of the flattened band vector, then L1 distance to ref."""
    blk = _tile_raw(cube, r, c, tile).astype(np.float64)  # (T, D)
    mu = blk.mean(axis=1, keepdims=True)
    sd = blk.std(axis=1, keepdims=True) + 1e-9
    z = (blk - mu) / sd
    ref = z[ref_t]
    return np.array([np.mean(np.abs(z[t] - ref)) for t in range(z.shape[0])])


def dist_ndvi_l1(cube, r, c, tile, ref_t=0):
    """Adjacent-band normalized-difference index distance. Gain-invariant by construction."""
    T, B = cube.shape[0], cube.shape[1]
    blk = cube[:, :, r:r + tile, c:c + tile].reshape(T, B, -1)  # (T, B, P)
    bi = blk[:, :-1, :]
    bj = blk[:, 1:, :]
    nd = (bi - bj) / (bi + bj + 1e-9)            # (T, B-1, P), invariant to multiplicative gain
    nd = nd.reshape(T, -1)
    ref = nd[ref_t]
    return np.array([np.mean(np.abs(nd[t] - ref)) for t in range(T)])


def dist_cosine_mean(cube, r, c, tile, ref_t=0):
    """1 - cosine between tile MEAN spectrum at date t and at ref. Scale invariant."""
    T, B = cube.shape[0], cube.shape[1]
    blk = cube[:, :, r:r + tile, c:c + tile].reshape(T, B, -1).mean(axis=2)  # (T, B) mean spectrum
    ref = blk[ref_t]
    rn = np.linalg.norm(ref) + 1e-12
    out = []
    for t in range(T):
        v = blk[t]
        cos = float(np.dot(v, ref) / (np.linalg.norm(v) * rn + 1e-12))
        out.append(1.0 - np.clip(cos, -1.0, 1.0))
    return np.array(out)


def dist_spec_angle(cube, r, c, tile, ref_t=0):
    """Spectral angle (arccos cosine) between tile mean spectra. Monotone twin of cosine; scale invariant."""
    T, B = cube.shape[0], cube.shape[1]
    blk = cube[:, :, r:r + tile, c:c + tile].reshape(T, B, -1).mean(axis=2)
    ref = blk[ref_t]
    rn = np.linalg.norm(ref) + 1e-12
    out = []
    for t in range(T):
        v = blk[t]
        cos = float(np.dot(v, ref) / (np.linalg.norm(v) * rn + 1e-12))
        out.append(float(np.arccos(np.clip(cos, -1.0, 1.0))))
    return np.array(out)


def per_date_normalize(cube):
    """Subtract per-date scene mean, divide by per-date scene std (global radiometric normalization)."""
    T = cube.shape[0]
    flat = cube.reshape(T, -1)
    mu = flat.mean(axis=1, keepdims=True)
    sd = flat.std(axis=1, keepdims=True) + 1e-9
    return ((flat - mu) / sd).reshape(cube.shape)


# ----------------------------------------------------------------- pooled-AUC evaluation (mirrors L1.evaluate)
BASELINES = ["trivial", "norm_l1", "zscore_tile", "ndvi_l1", "cosine", "spec_angle"]


def evaluate_all(cube, scene, cfg, tile, mode, rank, T_w, center):
    """Same pooled (tile x date) AUC machinery as L1.evaluate, but for DS + every fair baseline.

    Positive label = event tile AND changed date, identical to the original.
    """
    t0 = cfg.event_t0
    series = L1.build_series(cube, tile, mode, rank, T_w, center)
    cube_norm = per_date_normalize(cube)

    labels = []
    scores = {"ds": []}
    for b in BASELINES:
        scores[b] = []

    for (r, c), (centers, bases) in series.items():
        is_evt = L1.event_overlap(r, c, tile, cfg.event_box)
        _, dpre = L1.dist_to_ref(centers, bases)            # DS distance-to-baseline
        # precompute per-baseline full-T distance arrays
        d_triv = L1.trivial_distance(cube, r, c, tile)[1]
        d_norm = dist_norm_l1(cube_norm, r, c, tile)
        d_z = dist_zscore_tile(cube, r, c, tile)
        d_ndvi = dist_ndvi_l1(cube, r, c, tile)
        d_cos = dist_cosine_mean(cube, r, c, tile)
        d_sa = dist_spec_angle(cube, r, c, tile)
        bl_arrays = {"trivial": d_triv, "norm_l1": d_norm, "zscore_tile": d_z,
                     "ndvi_l1": d_ndvi, "cosine": d_cos, "spec_angle": d_sa}
        for k, t in enumerate(centers):
            labels.append(1 if (is_evt and scene.changed_dates[t]) else 0)
            scores["ds"].append(dpre[k])
            for b in BASELINES:
                scores[b].append(bl_arrays[b][t])

    labels = np.array(labels)
    aucs = {}
    if 0 < labels.sum() < len(labels):
        for key, sc in scores.items():
            aucs[key] = float(roc_auc_score(labels, np.array(sc)))
    return aucs


# ----------------------------------------------------------------- sweeps
def confound_sweep(mode, center, rank, T_w, tile=8):
    amps = [0.0, 0.05, 0.1, 0.2, 0.3, 0.45]
    rows = {"gain_amp": amps}
    keys = ["ds"] + BASELINES
    for kk in keys:
        rows[kk] = []
    for amp in amps:
        c = SynthConfig(event_box=(16, 32, 16, 32), event_t0=30, recovery_len=0, event_magnitude=0.6,
                        season_gain_amp=amp, haze_amp=amp * 0.3, season_drift_amp=0.05)
        s = make_scene(c)
        aucs = evaluate_all(s.cube, s, c, tile, mode, rank, T_w, center)
        for kk in keys:
            rows[kk].append(round(aucs.get(kk, float("nan")), 3))
    return rows


def magnitude_sweep(mode, center, rank, T_w, tile=8, gain_amp=0.15):
    mags = [0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
    rows = {"magnitude": mags}
    keys = ["ds"] + BASELINES
    for kk in keys:
        rows[kk] = []
    for mag in mags:
        c = SynthConfig(event_box=(16, 32, 16, 32), event_t0=30, recovery_len=0, event_magnitude=mag,
                        season_gain_amp=gain_amp)
        s = make_scene(c)
        aucs = evaluate_all(s.cube, s, c, tile, mode, rank, T_w, center)
        for kk in keys:
            rows[kk].append(round(aucs.get(kk, float("nan")), 3))
    return rows


def pure_gain_invariance():
    """Analog of L1.gain_invariance_check for the fair baselines: pure global gain ramp, one structure.

    Builds a tiny synthetic tile, applies a pure multiplicative gain ramp (NO structural change), and
    reports each descriptor's response. Gain-invariant descriptors should be ~0, exposing that DS=0 is
    not a unique property.
    """
    rng = np.random.default_rng(7)
    B, P, T = 10, 64, 8
    base = rng.uniform(0.1, 0.8, (B, P))             # bands x pixels, one structure
    gains = np.linspace(1.0, 1.6, T)
    # cube shape (T, B, h, w) with h*w = P; use a single tile 8x8
    h = w = 8
    cube = np.stack([g * base.reshape(B, h, w) for g in gains], axis=0)  # (T,B,h,w)
    cube = cube + rng.normal(0, 1e-3, cube.shape)
    r = c = 0
    tile = 8
    ref_t = 0
    res = {}
    # raw L1
    res["trivial"] = round(float(L1.trivial_distance(cube, r, c, tile)[1][-1]), 5)
    res["norm_l1"] = round(float(dist_norm_l1(per_date_normalize(cube), r, c, tile)[-1]), 5)
    res["zscore_tile"] = round(float(dist_zscore_tile(cube, r, c, tile)[-1]), 5)
    res["ndvi_l1"] = round(float(dist_ndvi_l1(cube, r, c, tile)[-1]), 5)
    res["cosine"] = round(float(dist_cosine_mean(cube, r, c, tile)[-1]), 5)
    res["spec_angle"] = round(float(dist_spec_angle(cube, r, c, tile)[-1]), 5)
    # DS (spatial, center=False, rank 3) date 0 vs date T-1
    S0 = ss.pca_subspace(cube[0].reshape(B, -1), 3, center=False, energy=0.99)
    S1 = ss.pca_subspace(cube[-1].reshape(B, -1), 3, center=False, energy=0.99)
    res["ds"] = round(float(ss.magnitude(S0, S1)), 5)
    return res


def main():
    # Use the SAME "best variant" the original picks: spatial, center=False, rank=3, T_w=6
    mode, center, rank, T_w = "spatial", False, 3, 6

    inv = pure_gain_invariance()
    cs = confound_sweep(mode, center, rank, T_w)
    ms = magnitude_sweep(mode, center, rank, T_w)

    print("=== PURE-GAIN INVARIANCE (pure multiplicative gain ramp, no structural change) ===")
    print(json.dumps(inv, indent=2))
    print("\n=== CONFOUND SWEEP (state-AUC vs seasonal-gain amplitude) ===")
    print("gain_amp   :", cs["gain_amp"])
    for kk in ["ds"] + BASELINES:
        print(f"{kk:11}:", cs[kk])
    print("\n=== MAGNITUDE SWEEP (state-AUC vs event magnitude, gain_amp=0.15) ===")
    print("magnitude  :", ms["magnitude"])
    for kk in ["ds"] + BASELINES:
        print(f"{kk:11}:", ms[kk])

    summary = {"variant": {"mode": mode, "center": center, "rank": rank, "T_w": T_w},
               "pure_gain_invariance": inv, "confound_sweep": cs, "magnitude_sweep": ms}
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
