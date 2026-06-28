"""ADVERSARIAL PROBE (mbprobe) -- is the multi-band "full DS beats min-angle" win REAL or a
SPECIAL-CASE (harmonic_dropout cherry-pick) + MIN-ANGLE STRAWMAN.

Two angles, both reusing the COMMITTED M-SSA machinery (temporal/experiments/multiband_gate.py) and
the COMMITTED builders where possible:

  ANGLE 1 (special-case): design extra DISTRIBUTED multi-mode changes that keep per-band mean and
  variance ~preserved, and ask whether full DS > min-angle ROBUSTLY across them, or whether only
  harmonic_dropout (the committed mode) shows the gap.
     amp_redistribute : move variance between harmonic 1 and harmonic 2, TOTAL variance held fixed.
     attenuate2_half  : 2nd harmonic partially attenuated (not fully dropped).
     phase_flip2      : 2nd harmonic phase flipped (pi) across all bands -- modes rotate, var fixed.
     period_stretch   : fundamental period stretched (both harmonics move) -- new modes, var fixed.

  ANGLE 2 (strawman): replace the committed min-angle (1 - max cos) with stronger ANGLE-AGGREGATE
  variants that are NOT the DS construction:
     minangle    : 1 - max(cos)                              (committed; weakest)
     mean_top2   : mean over the 2 largest angles (1-cos)
     mean_all    : mean over ALL canonical angles (1-cos)
     sum_1mcos   : sum over ALL (1-cos)   == 0.5 * DS magnitude when ranks equal
     ds_mag      : the committed DS magnitude (2*sum(1-cos), full rank-mismatch weighting)
  If mean_all or sum_1mcos matches ds_mag, the advantage is "aggregate all angles" (a 1-line SSA
  fix), NOT the DS construction. We measure that head to head, incl. under noise and rank mismatch.

Run: ./.venv/Scripts/python.exe -m temporal.experiments.probe_mbprobe_angles
Outputs: temporal/outputs/_mbprobe_angles/
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from temporal import subspace as ss
from temporal.synth_multiband import MBConfig, make_panel, _two_harmonic
from temporal.experiments.multiband_gate import mssa_subspace, acf_multilag

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "_mbprobe_angles")
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------------------------------
# NEW distributed multi-mode changes. We MONKEY-BUILD a panel by reusing the committed
# config + noise + label machinery but swapping the per-region signal generator, so the
# only thing that changes vs the committed harmonic_dropout panel is the event shape.
# All keep per-band mean ~0 and per-band variance ~preserved (verified numerically below).
# --------------------------------------------------------------------------------------
A1, A2 = 1.0, 0.7
BASE_VAR = (A1 ** 2 + A2 ** 2) / 2.0


def _region(mode, is_event, cfg, rng):
    t = np.arange(cfg.T)
    X = np.zeros((cfg.B, cfg.T))
    for b in range(cfg.B):
        ph = rng.uniform(0, 2 * np.pi)
        pre = _two_harmonic(t, cfg.period, A1, A2, ph)
        x = pre.copy()
        if is_event:
            tt = t[cfg.t0:]
            P = cfg.period
            if mode == "harmonic_dropout":          # committed reference: 2nd harmonic vanishes, var preserved
                a = np.sqrt(2 * BASE_VAR)
                x[cfg.t0:] = a * np.sin(2 * np.pi * tt / P + ph)
            elif mode == "amp_redistribute":        # swap energy between h1 and h2, TOTAL var fixed
                # new (a1',a2') with a1'^2+a2'^2 == A1^2+A2^2 but roles swapped-ish
                a1n, a2n = 0.7, 1.0
                x[cfg.t0:] = _two_harmonic(tt, P, a1n, a2n, ph)
            elif mode == "attenuate2_half":         # 2nd harmonic halved, then renormalize to keep var
                a1n, a2n = A1, A2 * 0.4
                scale = np.sqrt((A1 ** 2 + A2 ** 2) / (a1n ** 2 + a2n ** 2))
                x[cfg.t0:] = _two_harmonic(tt, P, a1n * scale, a2n * scale, ph)
            elif mode == "phase_flip2":             # 2nd harmonic phase flipped by pi (var identical)
                x[cfg.t0:] = A1 * np.sin(2 * np.pi * tt / P + ph) + \
                             A2 * np.sin(4 * np.pi * tt / P + ph + np.pi)
            elif mode == "period_stretch":          # fundamental period stretched 1.35x (both harmonics move)
                x[cfg.t0:] = _two_harmonic(tt, P * 1.35, A1, A2, ph)
            else:
                raise ValueError(mode)
        X[b] = x
    X = X + rng.normal(0, cfg.noise, X.shape)
    return X


def make_panel_mode(mode, cfg: MBConfig):
    rng = np.random.default_rng(cfg.seed)
    n_evt = int(cfg.n_regions * cfg.event_frac)
    labels = np.array([True] * n_evt + [False] * (cfg.n_regions - n_evt))
    rng.shuffle(labels)
    X = np.stack([_region(mode, bool(lb), cfg, rng) for lb in labels])
    return X, labels


# --------------------------------------------------------------------------------------
# Angle-aggregate descriptors. All operate on the SAME M-SSA past/present subspaces.
# --------------------------------------------------------------------------------------
def angle_descriptors(Sp, Sq):
    cos = ss.canonical_cosines(Sp, Sq)          # length min(kp,kq), in [0,1]
    one_m = 1.0 - cos
    kp, kq = Sp.shape[1], Sq.shape[1]
    srt = np.sort(one_m)[::-1]                   # descending (largest angle first)
    out = {
        "minangle":  float(srt[0]) if srt.size else 0.0,
        "mean_top2": float(np.mean(srt[:2])) if srt.size else 0.0,
        "mean_all":  float(np.mean(one_m)) if one_m.size else 0.0,
        "sum_1mcos": float(np.sum(one_m)),
        "ds_mag":    float(2.0 * (max(kp, kq) - np.sum(cos))),   # committed magnitude (rank-mismatch weighted)
    }
    return out


ANGLE_METHODS = ["minangle", "mean_top2", "mean_all", "sum_1mcos", "ds_mag"]
SCALARS = ["acf_ml"]
ALLM = ANGLE_METHODS + SCALARS


def region_scores(X, W, L, rank, lags, energy=0.99):
    B, T = X.shape
    series = {m: [] for m in ALLM}
    for t in range(W, T - W + 1):
        past = X[:, t - W:t]; pres = X[:, t:t + W]
        Sp = mssa_subspace(past, L, rank, energy=energy)
        Sq = mssa_subspace(pres, L, rank, energy=energy)
        d = angle_descriptors(Sp, Sq)
        for m in ANGLE_METHODS:
            series[m].append(d[m])
        acf_d = np.mean([np.sum(np.abs(acf_multilag(pres[b], lags) - acf_multilag(past[b], lags)))
                         for b in range(B)])
        series["acf_ml"].append(acf_d)
    return {m: np.array(series[m]) for m in ALLM}


def evaluate(mode, seeds, noise, W=24, L=12, rank=8, n_regions=120, energy=0.99):
    lags = list(range(1, L))
    aucs = {m: [] for m in ALLM}
    for sd in seeds:
        cfg = MBConfig(mode="harmonic_dropout", seed=sd, noise=noise, n_regions=n_regions)
        X, labels = make_panel_mode(mode, cfg)
        det = {m: [] for m in ALLM}
        for x in X:
            s = region_scores(x, W, L, rank, lags, energy=energy)
            for m in ALLM:
                det[m].append(float(np.max(s[m])))
        for m in ALLM:
            aucs[m].append(roc_auc_score(labels, det[m]))
    return {m: (float(np.mean(aucs[m])), float(np.std(aucs[m]))) for m in ALLM}


def variance_check(modes, seed=0):
    """Confirm per-band variance is ~preserved across the event boundary for each mode."""
    cfg = MBConfig(seed=seed, noise=0.0, n_regions=2)
    rng = np.random.default_rng(seed)
    out = {}
    for mode in modes:
        x = _region(mode, True, cfg, np.random.default_rng(123))
        pre = x[:, cfg.t0 - 24:cfg.t0]; post = x[:, cfg.t0:cfg.t0 + 24]
        out[mode] = {
            "var_pre": float(np.mean(np.var(pre, axis=1))),
            "var_post": float(np.mean(np.var(post, axis=1))),
            "mean_pre": float(np.mean(pre)),
            "mean_post": float(np.mean(post)),
        }
    return out


def main():
    modes = ["harmonic_dropout", "amp_redistribute", "attenuate2_half", "phase_flip2", "period_stretch"]
    seeds = (0, 1, 2, 3, 4)

    print("=== VARIANCE PRESERVATION CHECK (per-band var pre vs post, noise=0) ===")
    vc = variance_check(modes)
    for m in modes:
        v = vc[m]
        print(f"  {m:18} var {v['var_pre']:.3f}->{v['var_post']:.3f}  mean {v['mean_pre']:+.3f}->{v['mean_post']:+.3f}")

    results = {}
    print("\n=== ANGLE 1+2: AUC per mode at noise=0.25 ===")
    print(f"{'mode':18} | " + " ".join(f"{m:>10}" for m in ALLM))
    for mode in modes:
        r = evaluate(mode, seeds, noise=0.25)
        results[mode] = {m: {"auc": r[m][0], "std": r[m][1]} for m in ALLM}
        print(f"{mode:18} | " + " ".join(f"{r[m][0]:.2f}" for m in ALLM))

    # Noise sweep for every mode: track ds_mag vs minangle vs mean_all vs sum_1mcos
    print("\n=== NOISE SWEEP per mode (ds_mag | minangle | mean_all | sum_1mcos | acf_ml) ===")
    noises = (0.1, 0.5, 0.8, 1.2)
    sweep = {}
    for mode in modes:
        sweep[mode] = {"noise": list(noises)}
        for m in ALLM:
            sweep[mode][m] = []
        print(f"-- {mode}")
        for nz in noises:
            r = evaluate(mode, seeds=(0, 1, 2, 3), noise=nz)
            for m in ALLM:
                sweep[mode][m].append(r[m][0])
            print(f"   noise={nz:>4}: ds={r['ds_mag'][0]:.2f} min={r['minangle'][0]:.2f} "
                  f"meanAll={r['mean_all'][0]:.2f} sum={r['sum_1mcos'][0]:.2f} acf={r['acf_ml'][0]:.2f}")

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump({"variance_check": vc, "auc_noise025": results, "noise_sweep": sweep}, f, indent=2)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
