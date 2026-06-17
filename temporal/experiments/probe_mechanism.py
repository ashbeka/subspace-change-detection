"""ADVERSARIAL PROBE 2 -- mechanism / where-does-DS-break.

Probe 1 (probe_hard_realism) showed DS survives texture/jitter/partial but its ROBUSTNESS GAP over
trivial in the confound sweep shrinks a lot on realistic data. This probe isolates the mechanism and
hunts for genuine breakage, with strict apples-to-apples controls so we are FAIR to the baseline:

  A. Committed-config confound sweep, homog vs hard, 5-seed avg, reporting the DS-minus-trivial GAP
     (the actual claim is about the GAP, not DS alone).
  B. Registration-jitter-only stress at increasing amplitude with NO radiometric confound, to see if
     jitter alone manufactures DS pseudo-change (false onset) -- a real-data failure mode.
  C. "Subspace-preserving" event: a brightness-only / within-line material tweak that should fool an
     angle-based descriptor (DS) while raw differencing still sees it. This is the adversarial case
     DESIGNED to make DS lose.
  D. Reference-choice fragility: the committed dist_to_ref uses a SINGLE date (idx 0) as baseline.
     Test multi-date baseline vs single-date under jitter -- is the single-date ref cherry-picked?

Run: ./.venv/Scripts/python.exe -m temporal.experiments.probe_mechanism
Outputs: temporal/outputs/_probe_mechanism/
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from temporal import subspace as ss
from temporal.synth import SynthConfig, make_scene
from temporal.experiments.synth_injection import (
    tile_grid, event_overlap, build_series, velocity, dist_to_ref, trivial_distance, loc_err,
)
from temporal.experiments.probe_hard_realism import HardConfig, make_hard_scene

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "_probe_mechanism")
os.makedirs(OUT, exist_ok=True)

DS_MODE, DS_CENTER, DS_RANK, DS_TW, TILE = "spatial", False, 3, 6, 8


def state_auc(cube, region_box, changed_dates, tile, mode, rank, T_w, center, ref="first"):
    """state-AUC for DS dist-to-baseline and trivial dist-to-baseline.

    ref="first": baseline = date 0 (committed). ref="premean": baseline subspace = PCA over all
    pre-event dates' tile data (a fairer, less fragile baseline)."""
    series = build_series(cube, tile, mode, rank, T_w, center)
    labels, sc_ds, sc_tv = [], [], []
    # precompute pre-event mean reference per tile if needed
    for (r, c), (centers, bases) in series.items():
        is_evt = event_overlap(r, c, tile, region_box)
        if ref == "first":
            _, dpre = dist_to_ref(centers, bases, 0)
        else:
            # build a baseline subspace from the stacked pre-event tile pixels
            T, B = cube.shape[0], cube.shape[1]
            pre_t = [t for t in range(T) if not changed_dates[t] and t < 30]
            blk = cube[:, :, r:r + tile, c:c + tile]
            Xpre = np.concatenate([blk[t].reshape(B, -1) for t in pre_t], axis=1)
            refb = ss.pca_subspace(Xpre, rank, center=center, energy=0.99)
            dpre = np.array([ss.magnitude(b, refb) for b in bases])
        _, tdist = trivial_distance(cube, r, c, tile)
        for k, t in enumerate(centers):
            labels.append(1 if (is_evt and changed_dates[t]) else 0)
            sc_ds.append(dpre[k]); sc_tv.append(tdist[t])
    labels = np.array(labels)
    if not (0 < labels.sum() < len(labels)):
        return None, None
    return float(roc_auc_score(labels, sc_ds)), float(roc_auc_score(labels, sc_tv))


# ----------------------------------------------------------------- A: gap, homog vs hard
def sweep_gap(scene_kind, seeds=8):
    box = (16, 32, 16, 32)
    res = {"gain_amp": [], "ds": [], "triv": [], "gap": []}
    for amp in [0.0, 0.05, 0.1, 0.2, 0.3, 0.45]:
        ds_l, tv_l = [], []
        for seed in range(seeds):
            if scene_kind == "homog":
                c = SynthConfig(event_box=box, event_t0=30, recovery_len=0, event_magnitude=0.6,
                                season_gain_amp=amp, haze_amp=amp * 0.3, season_drift_amp=0.05, seed=seed)
                s = make_scene(c)
                cube, cdates = s.cube, s.changed_dates
            else:  # hard = texture + jitter
                hc = HardConfig(seed=seed, event_box=box, event_t0=30, event_magnitude=0.6,
                                season_gain_amp=amp, haze_amp=amp * 0.3, season_drift_amp=0.05,
                                texture=True, jitter_px=0.5)
                cube, _, cdates, _ = make_hard_scene(hc)
            a_ds, a_tv = state_auc(cube, box, cdates, TILE, DS_MODE, DS_RANK, DS_TW, DS_CENTER)
            if a_ds is not None:
                ds_l.append(a_ds); tv_l.append(a_tv)
        res["gain_amp"].append(amp)
        res["ds"].append(round(float(np.mean(ds_l)), 3))
        res["triv"].append(round(float(np.mean(tv_l)), 3))
        res["gap"].append(round(float(np.mean(ds_l)) - float(np.mean(tv_l)), 3))
    return res


# ----------------------------------------------------------------- B: jitter-only false onset
def jitter_false_onset(seeds=8):
    """No event, no radiometric confound -- ONLY registration jitter. Does DS velocity manufacture a
    spurious 'change'? Report mean DS velocity (pseudo-change energy) vs jitter amplitude, and the
    onset-localization error when a real event IS present but jitter is high."""
    box = (16, 32, 16, 32)
    res = {"jitter_px": [], "ds_pseudo_vel": [], "triv_pseudo": [],
           "ds_locerr_with_event": [], "triv_locerr_with_event": []}
    for jit in [0.0, 0.25, 0.5, 1.0, 1.5]:
        pv_ds, pv_tv, le_ds, le_tv = [], [], [], []
        for seed in range(seeds):
            # (i) NO event: pure jitter pseudo-change
            hc0 = HardConfig(seed=seed, event_box=box, event_t0=30, event_magnitude=0.0,
                             season_gain_amp=0.0, haze_amp=0.0, season_drift_amp=0.0,
                             noise_sigma=0.01, texture=True, jitter_px=jit)
            cube0, _, _, _ = make_hard_scene(hc0)
            series = build_series(cube0, TILE, DS_MODE, DS_RANK, DS_TW, DS_CENTER)
            etile = next(rc for rc in series if event_overlap(*rc, TILE, box))
            cen, dv = velocity(*series[etile])
            pv_ds.append(float(np.mean(dv)))
            blk = cube0[:, :, etile[0]:etile[0] + TILE, etile[1]:etile[1] + TILE].reshape(cube0.shape[0], -1)
            pv_tv.append(float(np.mean([np.mean(np.abs(blk[t + 1] - blk[t])) for t in range(cube0.shape[0] - 1)])))
            # (ii) WITH event under same jitter: onset localization error
            hc1 = HardConfig(seed=seed, event_box=box, event_t0=30, event_magnitude=0.6,
                             season_gain_amp=0.0, haze_amp=0.0, season_drift_amp=0.0,
                             noise_sigma=0.01, texture=True, jitter_px=jit)
            cube1, _, _, _ = make_hard_scene(hc1)
            s2 = build_series(cube1, TILE, DS_MODE, DS_RANK, DS_TW, DS_CENTER)
            cen2, dv2 = velocity(*s2[etile])
            le_ds.append(loc_err(cen2, dv2, 30))
            blk1 = cube1[:, :, etile[0]:etile[0] + TILE, etile[1]:etile[1] + TILE].reshape(cube1.shape[0], -1)
            tv2 = np.array([np.mean(np.abs(blk1[t + 1] - blk1[t])) for t in range(cube1.shape[0] - 1)])
            le_tv.append(int(abs((np.argmax(tv2) + 0) - 30)))
        res["jitter_px"].append(jit)
        res["ds_pseudo_vel"].append(round(float(np.mean(pv_ds)), 4))
        res["triv_pseudo"].append(round(float(np.mean(pv_tv)), 4))
        res["ds_locerr_with_event"].append(round(float(np.median(le_ds)), 1))
        res["triv_locerr_with_event"].append(round(float(np.median(le_tv)), 1))
    return res


# ----------------------------------------------------------------- C: subspace-preserving event
def subspace_preserving_event(seeds=8):
    """Adversarial event chosen to be (nearly) WITHIN the existing dominant direction:
    a pure brightness scaling of the event region (multiply region spectra by a constant>1).
    This rotates the dominant line very little -> DS angle ~ 0, but raw differencing sees a big
    magnitude change. If DS is truly 'detecting the changed state', it should FAIL here while
    trivial succeeds. We compare to a normal material-swap event as control."""
    box = (16, 32, 16, 32)
    H = W = 64; B = 10; T = 60; t0 = 30
    out = {}
    for kind in ["brightness_only", "material_swap"]:
        ds_l, tv_l = [], []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            # textured base
            hc = HardConfig(seed=seed, event_box=box, event_t0=t0, event_magnitude=0.0,
                            season_gain_amp=0.0, haze_amp=0.0, season_drift_amp=0.0,
                            noise_sigma=0.015, texture=True, jitter_px=0.0)
            cube, rmask, _, _ = make_hard_scene(hc)  # event_magnitude 0 => no event yet
            changed = np.zeros(T, dtype=bool)
            cube = cube.copy()
            if kind == "brightness_only":
                for t in range(t0, T):
                    cube[t][:, rmask] *= 1.6   # pure scalar gain in the region (stays on same line)
                    changed[t] = True
            else:
                # reuse a material swap by adding a new endmember direction
                from temporal.synth import _endmembers
                burn = _endmembers(rng, B, 1)[0]
                for t in range(t0, T):
                    cube[t][:, rmask] = 0.4 * cube[t][:, rmask] + 0.6 * burn[:, None]
                    changed[t] = True
            a_ds, a_tv = state_auc(cube, box, changed, TILE, DS_MODE, DS_RANK, DS_TW, DS_CENTER)
            if a_ds is not None:
                ds_l.append(a_ds); tv_l.append(a_tv)
        out[kind] = {"ds_auc": round(float(np.mean(ds_l)), 3), "triv_auc": round(float(np.mean(tv_l)), 3)}
    return out


# ----------------------------------------------------------------- D: reference choice fragility
def ref_choice(seeds=8):
    box = (16, 32, 16, 32)
    res = {}
    for jit in [0.0, 0.5, 1.0]:
        first_ds, premean_ds = [], []
        for seed in range(seeds):
            hc = HardConfig(seed=seed, event_box=box, event_t0=30, event_magnitude=0.6,
                            season_gain_amp=0.2, haze_amp=0.06, season_drift_amp=0.05,
                            texture=True, jitter_px=jit)
            cube, _, cdates, _ = make_hard_scene(hc)
            a1, _ = state_auc(cube, box, cdates, TILE, DS_MODE, DS_RANK, DS_TW, DS_CENTER, ref="first")
            a2, _ = state_auc(cube, box, cdates, TILE, DS_MODE, DS_RANK, DS_TW, DS_CENTER, ref="premean")
            if a1 is not None:
                first_ds.append(a1); premean_ds.append(a2)
        res[f"jitter{jit}"] = {"ds_ref_first_date": round(float(np.mean(first_ds)), 3),
                               "ds_ref_premean": round(float(np.mean(premean_ds)), 3)}
    return res


def main():
    print("=== PROBE 2: mechanism / breakage ===\n")
    print("[A] DS-minus-trivial GAP, committed config, homog vs hard (texture+jitter):")
    g_homog = sweep_gap("homog")
    g_hard = sweep_gap("hard")
    print("   homog:", json.dumps(g_homog))
    print("   hard :", json.dumps(g_hard))

    print("\n[B] jitter-only pseudo-change + onset localization under jitter:")
    jb = jitter_false_onset()
    print("   ", json.dumps(jb))

    print("\n[C] subspace-preserving (brightness-only) event vs material swap:")
    sp = subspace_preserving_event()
    print("   ", json.dumps(sp))

    print("\n[D] reference-choice fragility (single date vs pre-event mean subspace):")
    rc = ref_choice()
    print("   ", json.dumps(rc))

    summary = {"gap_homog": g_homog, "gap_hard": g_hard, "jitter": jb,
               "subspace_preserving": sp, "ref_choice": rc}
    with open(os.path.join(OUT, "mechanism_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
