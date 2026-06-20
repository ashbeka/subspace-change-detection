"""L0 sanity for the FAITHFUL Kanai signal-subspace DS (temporal/signal_subspace_ds.py).
Goal: (a) confirm construction (a_hat spikes AT a transition), (b) reproduce the paper's claim that DS beats
the SSA MINIMUM-ANGLE baseline when the change is in a SECONDARY component (dominant persists).

Fixes vs first attempt: longer windows (stable signal subspace over several periods); phase-continuous
frequency-segment anomaly (cumulative-phase integration, no artificial joins); change-POINT evaluation
(a window is positive iff its present window overlaps the anomalous segment -> the transition region).

CONSTRUCTION (ledger): object = SSA signal subspace of a 1D window; data matrix = Hankel (w x M); built =
top-r left singular vectors; represents = dominant temporal structure. Change = generalized DS(past,present);
reference D_N = principal subspace of the DSs over a normal period.

Run: cd <worktree> && <venv>/python.exe -m temporal.experiments.ssds_l0
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from temporal import signal_subspace_ds as S

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "ssds_l0")
os.makedirs(OUT, exist_ok=True)

W, M, R, TAU = 120, 60, 4, 16
FS = 100.0
SEG = (380, 460)            # anomalous segment (length 80)
L = 820


def gen(kind, rng, anomalous):
    """Phase-continuous two-tone signal; during SEG one tone's frequency shifts (if anomalous)."""
    f0 = np.full(L, 5.0); f1 = np.full(L, 11.0); a1 = 0.8
    if anomalous:
        if kind == "secondary":
            f1[SEG[0]:SEG[1]] = 11.0 * 1.7        # secondary tone shifts; dominant persists
        else:
            f0[SEG[0]:SEG[1]] = 5.0 * 1.35        # dominant tone shifts
    ph0 = np.cumsum(2 * np.pi * f0 / FS); ph1 = np.cumsum(2 * np.pi * f1 / FS)
    return np.sin(ph0) + a1 * np.sin(ph1) + 0.15 * rng.standard_normal(L)


def eval_once(kind, rng, want_curve=False):
    normal = gen(kind, np.random.default_rng(rng.integers(1 << 30)), anomalous=False)
    test = gen(kind, np.random.default_rng(rng.integers(1 << 30)), anomalous=True)
    D_N, mu_N = S.learn_reference(normal, W, M, R, TAU)
    ts, a_hat, b_min, b_mean = S.change_degree(test, W, M, R, TAU, D_N, mu_N)
    span = W + M - 1
    # positive = present window [t-span, t] overlaps the anomalous segment (the transition region)
    pos = (ts > SEG[0]) & ((ts - span) < SEG[1])
    neg = ~pos
    y = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
    out = {}
    for name, sc in [("a_hat", a_hat), ("min_angle", b_min), ("mean_angle", b_mean)]:
        s = np.r_[sc[pos], sc[neg]]
        out[name] = float(max(roc_auc_score(y, s), 1 - roc_auc_score(y, s)))
    out["peak_in_seg"] = bool(SEG[0] - span <= ts[np.argmax(a_hat)] <= SEG[1])
    if want_curve:
        order = np.argsort(a_hat)[::-1][:5]
        out["top5_peaks"] = [int(ts[i]) for i in order]
    return out


def main():
    rng = np.random.default_rng(0)
    SEEDS = 15
    res = {}
    print(f"cfg: w={W} M={M} r={R} tau={TAU} span={W+M-1}; seg={SEG}; {SEEDS} seeds/type")
    ex = eval_once("secondary", np.random.default_rng(7), want_curve=True)
    print(f"  example (secondary) top-5 a_hat peak indices: {ex['top5_peaks']}  (segment {SEG})")
    print(f"{'anomaly':>10} {'a_hat':>8} {'min_ang':>8} {'mean_ang':>9} {'peak_in_seg':>12}")
    for kind in ["dominant", "secondary"]:
        accs = [eval_once(kind, np.random.default_rng(rng.integers(1 << 30))) for _ in range(SEEDS)]
        agg = {k: float(np.mean([a[k] for a in accs])) for k in ["a_hat", "min_angle", "mean_angle"]}
        agg["peak_in_seg"] = float(np.mean([a["peak_in_seg"] for a in accs]))
        res[kind] = agg
        print(f"{kind:>10} {agg['a_hat']:>8.3f} {agg['min_angle']:>8.3f} {agg['mean_angle']:>9.3f} {agg['peak_in_seg']:>12.2f}")

    print("\n=== L0 VERDICT ===")
    okc = res["dominant"]["a_hat"] > 0.8 and res["secondary"]["a_hat"] > 0.8
    adv = res["secondary"]["a_hat"] - res["secondary"]["min_angle"]
    print(f"  construction valid (a_hat detects both, AUC>0.8): {okc}")
    print(f"  DS advantage over min-angle on SECONDARY change: {adv:+.3f} "
          f"({'reproduces paper claim' if adv > 0.05 else 'no advantage here'})")
    json.dump(res, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
