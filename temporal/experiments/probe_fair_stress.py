"""ADVERSARIAL PROBE 2: stress the fair-baseline finding across seeds, harsher confounds, drift.

Confirms that the probe_fair_baselines result (norm_l1 dominates DS; cosine matches DS) is not a
fluke of one seed / one box / the gain:haze ratio. Sweeps multiple seeds and reports mean +/- std AUC,
plus a harsher confound regime (bigger haze, bigger drift, extreme gain).

Run: ./.venv/Scripts/python.exe -m temporal.experiments.probe_fair_stress
"""
from __future__ import annotations

import json
import os

import numpy as np

from temporal.synth import SynthConfig, make_scene
from temporal.experiments import probe_fair_baselines as P

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "_probe_fair_baselines")
os.makedirs(OUT, exist_ok=True)

KEYS = ["ds"] + P.BASELINES


def multi_seed_confound(seeds, amp, haze_mult, drift, mode="spatial", center=False, rank=3, T_w=6, tile=8):
    acc = {k: [] for k in KEYS}
    for sd in seeds:
        c = SynthConfig(event_box=(16, 32, 16, 32), event_t0=30, recovery_len=0, event_magnitude=0.6,
                        season_gain_amp=amp, haze_amp=amp * haze_mult, season_drift_amp=drift, seed=sd)
        s = make_scene(c)
        aucs = P.evaluate_all(s.cube, s, c, tile, mode, rank, T_w, center)
        for k in KEYS:
            acc[k].append(aucs.get(k, float("nan")))
    return {k: (round(float(np.mean(v)), 3), round(float(np.std(v)), 3)) for k, v in acc.items()}


def main():
    seeds = list(range(8))

    print("=== MULTI-SEED, MODERATE CONFOUND (amp=0.3, haze=0.3*amp, drift=0.05) mean+/-std over 8 seeds ===")
    r1 = multi_seed_confound(seeds, amp=0.3, haze_mult=0.3, drift=0.05)
    for k in KEYS:
        print(f"{k:11}: {r1[k][0]:.3f} +/- {r1[k][1]:.3f}")

    print("\n=== MULTI-SEED, HARSH CONFOUND (amp=0.45, haze=1.0*amp, drift=0.15) mean+/-std over 8 seeds ===")
    r2 = multi_seed_confound(seeds, amp=0.45, haze_mult=1.0, drift=0.15)
    for k in KEYS:
        print(f"{k:11}: {r2[k][0]:.3f} +/- {r2[k][1]:.3f}")

    print("\n=== MULTI-SEED, EXTREME GAIN (amp=0.7, haze=0.3*amp, drift=0.05) mean+/-std over 8 seeds ===")
    r3 = multi_seed_confound(seeds, amp=0.7, haze_mult=0.3, drift=0.05)
    for k in KEYS:
        print(f"{k:11}: {r3[k][0]:.3f} +/- {r3[k][1]:.3f}")

    summary = {"moderate": r1, "harsh": r2, "extreme_gain": r3, "seeds": seeds}
    with open(os.path.join(OUT, "stress_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
