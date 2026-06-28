"""ADVERSARIAL PROBE 2: is the DS win literally a rank-1 (spectral-angle) effect?

Probe A showed DS-AUC is 1.0->0.86 ONLY at fixed rank k=1, and collapses to chance (~0.52)
at k>=2 even with NO confound. The energy=0.99 threshold in the original keeps the selected
rank near 1-2. Hypothesis: at the operating point DS magnitude is monotone in the angle between
the rank-1 dominant directions == spectral angle == SAM, which is gain-invariant for free.

Here we (1) compute, per (tile,date), the rank-1 DS distance and the SAM-to-ref distance and
correlate them; (2) compare DS-AUC vs SAM-AUC across the confound sweep at the OPERATING point;
(3) show that center=True (which the design also offers) kills the win, isolating the effect to
the autocorrelation-PCA-on-uncentered-data choice (a normalized mean-direction proxy).
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from temporal import subspace as ss
from temporal.synth import SynthConfig, make_scene
from temporal.experiments import synth_injection as si

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "_probe_construction")
os.makedirs(OUT, exist_ok=True)
TILE = 8


def scene_for(amp):
    c = SynthConfig(event_box=(16, 32, 16, 32), event_t0=30, recovery_len=0,
                    event_magnitude=0.6, season_gain_amp=amp, haze_amp=amp * 0.3,
                    season_drift_amp=0.05)
    return make_scene(c), c


def per_date_scores(cube, scene, cfg, center, energy):
    """Return arrays: labels, ds_rank1, ds_orig(E=energy), sam, for all (tile,date)."""
    T, B, H, W = cube.shape
    labels, ds1, dso, sam = [], [], [], []
    for (r, c) in si.tile_grid(H, W, TILE):
        blk = cube[:, :, r:r + TILE, c:c + TILE]
        is_evt = si.event_overlap(r, c, TILE, cfg.event_box)
        # rank-1 bases (dominant direction only)
        b1 = [ss.pca_subspace(blk[t].reshape(B, -1), 1, center=center, energy=None) for t in range(T)]
        # original variable-rank bases
        bo = [ss.pca_subspace(blk[t].reshape(B, -1), 3, center=center, energy=energy) for t in range(T)]
        ref1, refo = b1[0], bo[0]
        # SAM on mean spectrum
        spec = blk.reshape(T, B, -1).mean(axis=2)
        sref = spec[0]
        for t in range(T):
            labels.append(1 if (is_evt and scene.changed_dates[t]) else 0)
            ds1.append(ss.magnitude(b1[t], ref1))
            dso.append(ss.magnitude(bo[t], refo))
            na = np.linalg.norm(spec[t]) * np.linalg.norm(sref) + 1e-12
            sam.append(float(np.arccos(np.clip(np.dot(spec[t], sref) / na, -1, 1))))
    return (np.array(labels), np.array(ds1), np.array(dso), np.array(sam))


def main():
    out = {"sweep": [], "corr": {}}
    for amp in [0.0, 0.1, 0.2, 0.3, 0.45]:
        s, c = scene_for(amp)
        lab, ds1, dso, sam = per_date_scores(s.cube, s, c, center=False, energy=0.99)
        # center=True variant (design alternative)
        _, ds1c, dsoc, _ = per_date_scores(s.cube, s, c, center=True, energy=0.99)
        def auc(x):
            return round(float(roc_auc_score(lab, x)), 3) if 0 < lab.sum() < len(lab) else None
        row = {"amp": amp,
               "auc_ds_orig_E99_centerFalse": auc(dso),
               "auc_ds_rank1_centerFalse": auc(ds1),
               "auc_sam": auc(sam),
               "auc_ds_orig_E99_centerTrue": auc(dsoc),
               "auc_ds_rank1_centerTrue": auc(ds1c)}
        out["sweep"].append(row)
        if amp == 0.45:
            # correlation between DS(orig) and SAM, and DS(orig) and DS(rank1)
            rho_sam = float(spearmanr(dso, sam).correlation)
            rho_r1 = float(spearmanr(dso, ds1).correlation)
            out["corr"] = {"amp": amp,
                           "spearman_DSorig_vs_SAM": round(rho_sam, 4),
                           "spearman_DSorig_vs_DSrank1": round(rho_r1, 4)}
    with open(os.path.join(OUT, "rank1_equivalence.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("amp | DSorig(cF) DSrank1(cF)  SAM | DSorig(cT) DSrank1(cT)")
    for r in out["sweep"]:
        print(f" {r['amp']:.2f}| {str(r['auc_ds_orig_E99_centerFalse']):>9} "
              f"{str(r['auc_ds_rank1_centerFalse']):>10} {str(r['auc_sam']):>5} | "
              f"{str(r['auc_ds_orig_E99_centerTrue']):>9} {str(r['auc_ds_rank1_centerTrue']):>10}")
    print("\nAt amp=0.45:", json.dumps(out["corr"]))
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
