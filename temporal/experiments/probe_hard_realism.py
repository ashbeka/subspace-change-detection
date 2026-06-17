"""ADVERSARIAL PROBE (not committed by the original authors).

Goal: attack the L1 claim that temporal-DS distance-to-baseline detects the changed STATE
far more robustly to radiometric confounds than raw differencing. The committed synth.py uses
HOMOGENEOUS quadrant tiles, so each 8x8 tile is ~rank-1 and a material swap is a trivial rank
1->2 jump -- exactly the regime where a subspace descriptor wins for free. This probe rebuilds
the scene to be harder/more realistic and re-runs the SAME state-AUC + onset metrics for
DS distance-to-baseline vs trivial raw-distance under each stressor:

  (1) TEXTURE: multiple materials mixed inside every tile (stable rank > 1), so the event is not
      a trivial rank jump.
  (2) REGISTRATION JITTER: sub-pixel per-date shift (scipy.ndimage.shift) -- classic pseudo-change.
  (3) PARTIAL events: only a fraction of the event region actually changes.
  (4) HETEROSCEDASTIC / higher noise: signal-dependent + per-band noise.
  (5) NON-STATIONARY confounds: linear trend + seasonal gain (not purely periodic).

It reuses the committed builders (build_series, dist_to_ref, trivial_distance, evaluate machinery)
by importing them, but feeds them a custom cube. Nothing committed is edited.

Run: ./.venv/Scripts/python.exe -m temporal.experiments.probe_hard_realism
Outputs: temporal/outputs/_probe_hard_realism/
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import shift as nd_shift
from sklearn.metrics import roc_auc_score

from temporal import subspace as ss
from temporal.synth import SynthConfig, make_scene, _endmembers
from temporal.experiments.synth_injection import (
    tile_grid, event_overlap, build_series, velocity, minangle,
    dist_to_ref, trivial_distance, trivial_velocity, loc_err, contrast,
)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "_probe_hard_realism")
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------------------- harder scene builder
@dataclass
class HardConfig:
    H: int = 64
    W: int = 64
    B: int = 10
    T: int = 60
    n_materials: int = 4
    event_t0: int = 30
    event_box: tuple = (16, 32, 16, 32)
    event_magnitude: float = 1.0
    # ---- confounds (match SynthConfig defaults where shared) ----
    season_gain_amp: float = 0.15
    season_drift_amp: float = 0.08
    haze_amp: float = 0.03
    noise_sigma: float = 0.02
    seed: int = 0
    # ---- HARD knobs ----
    texture: bool = False          # mix multiple materials inside each tile (stable rank>1)
    n_texture_mats: int = 3        # how many endmembers mixed per pixel when texture=True
    jitter_px: float = 0.0         # std of per-date sub-pixel registration shift (in pixels)
    partial_frac: float = 1.0      # fraction of event-region pixels that actually change
    hetero_noise: float = 0.0      # signal-dependent noise coefficient (added to noise_sigma)
    trend_amp: float = 0.0         # linear (non-stationary) multiplicative trend over the series


def make_hard_scene(cfg: HardConfig):
    """Build a (T,B,H,W) cube with optional texture, jitter, partial events, hetero noise, trend.

    Returns (cube, region_mask, changed_dates, changed_pixel_mask).
    """
    rng = np.random.default_rng(cfg.seed)
    H, W, B, T = cfg.H, cfg.W, cfg.B, cfg.T
    endm = _endmembers(rng, B, cfg.n_materials + 1)
    burn = endm[-1]
    mats = endm[:-1]  # (n_materials, B)

    # ---- base structural cube (B,H,W): textured (per-pixel mixture) or blocky quadrants ----
    if cfg.texture:
        # every pixel is a random convex mixture of n_texture_mats endmembers, with a smooth
        # spatial component so neighbouring pixels correlate (still multi-material per tile).
        nmix = min(cfg.n_texture_mats, cfg.n_materials)
        # smooth random weight fields per material (low-freq) -> textured but spatially coherent
        weights = np.zeros((cfg.n_materials, H, W))
        for m in range(cfg.n_materials):
            field = rng.standard_normal((H, W))
            # cheap smoothing: average with shifted copies
            sm = field.copy()
            for _ in range(3):
                sm = (sm + np.roll(sm, 1, 0) + np.roll(sm, -1, 0)
                      + np.roll(sm, 1, 1) + np.roll(sm, -1, 1)) / 5.0
            weights[m] = sm
        # keep only top-nmix materials per pixel, softmax to convex weights
        order = np.argsort(-weights, axis=0)
        keep = np.zeros_like(weights, dtype=bool)
        for k in range(nmix):
            idx = order[k]
            keep[idx, np.arange(H)[:, None], np.arange(W)[None, :]] = True
        weights = np.where(keep, weights, -1e9)
        w = np.exp(weights - weights.max(0, keepdims=True))
        w = w / w.sum(0, keepdims=True)              # (n_materials, H, W) convex
        base = np.einsum("mhw,mb->bhw", w, mats)     # (B,H,W)
    else:
        label = np.zeros((H, W), dtype=int)
        label[: H // 2, : W // 2] = 0
        label[: H // 2, W // 2:] = 1
        label[H // 2:, : W // 2] = 2 % cfg.n_materials
        label[H // 2:, W // 2:] = 3 % cfg.n_materials
        base = np.moveaxis(mats[label], 2, 0)        # (B,H,W)

    r0, r1, c0, c1 = cfg.event_box
    region_mask = np.zeros((H, W), dtype=bool)
    region_mask[r0:r1, c0:c1] = True

    # partial-region: only a random subset of region pixels actually change
    changed_pixel_mask = region_mask.copy()
    if cfg.partial_frac < 1.0:
        rin = np.argwhere(region_mask)
        n_keep = int(round(cfg.partial_frac * len(rin)))
        sel = rng.choice(len(rin), size=n_keep, replace=False)
        changed_pixel_mask = np.zeros((H, W), dtype=bool)
        changed_pixel_mask[rin[sel, 0], rin[sel, 1]] = True

    drift_dir = rng.standard_normal(B)
    drift_dir = drift_dir / np.linalg.norm(drift_dir)

    cube = np.zeros((T, B, H, W), dtype=np.float64)
    changed_dates = np.zeros(T, dtype=bool)

    for t in range(T):
        struct = base.copy()
        if t >= cfg.event_t0:
            alpha = cfg.event_magnitude
            if alpha > 1e-6:
                changed_dates[t] = True
                cm = changed_pixel_mask
                struct[:, cm] = (1 - alpha) * struct[:, cm] + alpha * burn[:, None]

        # gradual seasonal spectral drift
        season_phase = np.sin(2 * np.pi * t / max(2, T // 1.5))
        struct = struct + cfg.season_drift_amp * season_phase * drift_dir[:, None, None]

        # multiplicative seasonal gain + NON-STATIONARY linear trend
        gain = 1.0 + cfg.season_gain_amp * np.sin(2 * np.pi * t / max(2, T // 1.2) + 0.7)
        gain = gain + cfg.trend_amp * (t / max(1, T - 1))      # monotonic drift on top
        haze = cfg.haze_amp * np.sin(2 * np.pi * t / max(2, T // 0.9) + 1.3)
        img = gain * struct + haze

        # noise: homoscedastic base + heteroscedastic (signal-dependent) term
        n = rng.normal(0, cfg.noise_sigma, img.shape)
        if cfg.hetero_noise > 0:
            n = n + rng.normal(0, 1, img.shape) * cfg.hetero_noise * np.sqrt(np.clip(img, 0, None))
        img = img + n

        # SUB-PIXEL REGISTRATION JITTER: shift the whole frame by a small random offset per date
        if cfg.jitter_px > 0:
            dr = rng.normal(0, cfg.jitter_px)
            dc = rng.normal(0, cfg.jitter_px)
            # shift along (H,W) axes only (axes 1,2 of (B,H,W)); reflect to avoid edge zeros
            img = nd_shift(img, shift=(0.0, dr, dc), order=1, mode="reflect")

        cube[t] = np.clip(img, 0.0, None)

    return cube, region_mask, changed_dates, changed_pixel_mask


# --------------------------------------------------------------------------- metric (probe copy)
def evaluate_state(cube, region_mask, changed_dates, cfg, tile, mode, rank, T_w, center, lag=1):
    """Re-implements synth_injection.evaluate's state-AUC + localization, with an arbitrary
    region_mask/changed_dates (so partial events and textured scenes work). Tile is positive if
    it overlaps the event_box (same convention as the committed code)."""
    t0 = cfg.event_t0
    T, B, H, W = cube.shape
    series = build_series(cube, tile, mode, rank, T_w, center)

    # event-tile curves (for onset localization + contrast)
    etile = next((rc for rc in series if event_overlap(*rc, tile, cfg.event_box)), (0, 0))
    cen, d1 = velocity(*series[etile], lag)
    _, ma = minangle(*series[etile], lag)
    tcen, tv = trivial_velocity(cube, *etile, tile, lag)

    # distance-to-baseline state AUC across all tiles x dates
    labels, sc_ds, sc_tv = [], [], []
    for (r, c), (centers, bases) in series.items():
        is_evt = event_overlap(r, c, tile, cfg.event_box)
        _, dpre = dist_to_ref(centers, bases)
        _, tdist = trivial_distance(cube, r, c, tile)
        for k, t in enumerate(centers):
            labels.append(1 if (is_evt and changed_dates[t]) else 0)
            sc_ds.append(dpre[k]); sc_tv.append(tdist[t])
    labels = np.array(labels)
    auc_ds = auc_tv = None
    if 0 < labels.sum() < len(labels):
        auc_ds = float(roc_auc_score(labels, sc_ds))
        auc_tv = float(roc_auc_score(labels, sc_tv))

    return {
        "loc_err_ds": loc_err(cen, d1, t0),
        "loc_err_trivial": loc_err(tcen, tv, t0),
        "contrast_ds": round(contrast(cen, d1, t0), 3),
        "contrast_trivial": round(contrast(tcen, tv, t0), 3),
        "state_auc_ds": round(auc_ds, 4) if auc_ds is not None else None,
        "state_auc_trivial": round(auc_tv, 4) if auc_tv is not None else None,
    }


# --------------------------------------------------------------------------- driver
# Use the SAME winning DS variant the committed main() selects: spatial, center=False, rank=3, T_w=6
DS_MODE, DS_CENTER, DS_RANK, DS_TW, TILE = "spatial", False, 3, 6, 8


def run_grid():
    base = dict(event_box=(16, 32, 16, 32), event_t0=30, event_magnitude=0.6,
                season_gain_amp=0.15, haze_amp=0.045, season_drift_amp=0.05, noise_sigma=0.02)

    scenarios = {
        "S0_baseline_homog":      dict(),
        "S1_texture":             dict(texture=True),
        "S2_jitter0.3":           dict(jitter_px=0.3),
        "S3_jitter0.7":           dict(jitter_px=0.7),
        "S4_texture+jitter0.5":   dict(texture=True, jitter_px=0.5),
        "S5_partial0.4":          dict(partial_frac=0.4),
        "S6_texture+partial0.4":  dict(texture=True, partial_frac=0.4),
        "S7_hetero_noise":        dict(hetero_noise=0.06, noise_sigma=0.04),
        "S8_nonstat_trend":       dict(trend_amp=0.4),
        "S9_ALL_combined":        dict(texture=True, jitter_px=0.5, partial_frac=0.5,
                                       hetero_noise=0.05, trend_amp=0.3, season_gain_amp=0.3),
    }

    rows = []
    for name, over in scenarios.items():
        # average over several seeds to be fair (jitter/texture are random)
        aucs_ds, aucs_tv, le_ds, le_tv = [], [], [], []
        for seed in range(5):
            cfg = HardConfig(seed=seed, **{**base, **over})
            cube, rmask, cdates, cpix = make_hard_scene(cfg)
            m = evaluate_state(cube, rmask, cdates, cfg, TILE, DS_MODE, DS_RANK, DS_TW, DS_CENTER)
            if m["state_auc_ds"] is not None:
                aucs_ds.append(m["state_auc_ds"]); aucs_tv.append(m["state_auc_trivial"])
            le_ds.append(m["loc_err_ds"]); le_tv.append(m["loc_err_trivial"])
        row = {
            "scenario": name,
            "ds_auc_mean": round(float(np.mean(aucs_ds)), 4) if aucs_ds else None,
            "ds_auc_std": round(float(np.std(aucs_ds)), 4) if aucs_ds else None,
            "trivial_auc_mean": round(float(np.mean(aucs_tv)), 4) if aucs_tv else None,
            "ds_locerr_med": float(np.median(le_ds)),
            "trivial_locerr_med": float(np.median(le_tv)),
        }
        rows.append(row)
        print(f"{name:24} DS-AUC={row['ds_auc_mean']}+-{row['ds_auc_std']}  "
              f"triv-AUC={row['trivial_auc_mean']}  "
              f"DS-locErr(med)={row['ds_locerr_med']}  triv-locErr(med)={row['trivial_locerr_med']}")
    return rows


def run_confound_sweep_hard(texture=True, jitter_px=0.5):
    """The headline H2 sweep, but on the HARD scene (textured + jittered). Does DS still hold
    1.0 while trivial collapses?"""
    out = {"gain_amp": [], "ds_auc": [], "trivial_auc": []}
    for amp in [0.0, 0.05, 0.1, 0.2, 0.3, 0.45]:
        aucs_ds, aucs_tv = [], []
        for seed in range(5):
            cfg = HardConfig(seed=seed, event_box=(16, 32, 16, 32), event_t0=30,
                             event_magnitude=0.6, season_gain_amp=amp, haze_amp=amp * 0.3,
                             season_drift_amp=0.05, texture=texture, jitter_px=jitter_px)
            cube, rmask, cdates, _ = make_hard_scene(cfg)
            m = evaluate_state(cube, rmask, cdates, cfg, TILE, DS_MODE, DS_RANK, DS_TW, DS_CENTER)
            if m["state_auc_ds"] is not None:
                aucs_ds.append(m["state_auc_ds"]); aucs_tv.append(m["state_auc_trivial"])
        out["gain_amp"].append(amp)
        out["ds_auc"].append(round(float(np.mean(aucs_ds)), 4))
        out["trivial_auc"].append(round(float(np.mean(aucs_tv)), 4))
    print("HARD confound sweep (texture+jitter):", json.dumps(out))
    return out


def main():
    print("=== HARD-REALISM PROBE (DS variant: spatial, center=False, rank=3, T_w=6) ===\n")
    print("--- scenario grid (5 seeds each) ---")
    rows = run_grid()
    print("\n--- confound sweep on HARD (textured+jittered) scene ---")
    cs_hard = run_confound_sweep_hard(texture=True, jitter_px=0.5)
    print("\n--- confound sweep on JITTER-ONLY (homog+jitter) scene ---")
    cs_jit = run_confound_sweep_hard(texture=False, jitter_px=0.5)

    summary = {"variant": dict(mode=DS_MODE, center=DS_CENTER, rank=DS_RANK, T_w=DS_TW, tile=TILE),
               "scenario_grid": rows, "confound_sweep_hard": cs_hard,
               "confound_sweep_jitteronly": cs_jit}
    with open(os.path.join(OUT, "probe_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
