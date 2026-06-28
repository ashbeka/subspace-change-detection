"""GATE experiment (post-verdict): does SSA temporal-subspace DS detect a DYNAMICS change that
mean/variance/spectral-angle baselines fundamentally cannot?

This is the decisive test the L1 adversarial verdict demanded before touching real data:
demonstrate a regime where genuinely multi-dimensional subspace structure beats the fair scalar
baselines (mean-shift, variance-shift, lag-1 autocorrelation-shift) and the SSA minimum-angle.

Construction: per region (scalar index series), build SSA trajectory (Hankel) subspaces over a past
window and a present window; anomaly score = DS magnitude between them (Kanai et al. 2023). Compare to:
  mean_shift  |mean(present)-mean(past)|            (the "spectral / level" analog -- should be blind)
  var_shift   |std(present)-std(past)|              (catches amplitude collapse, not structure)
  acf1_shift  |acf1(present)-acf1(past)|            (a temporal scalar)
  ssa_minangle 1 - max canonical cosine             (conventional SSA; Kanai's baseline)
  ssa_ds      DS magnitude (ours)
Reported per dynamics-change mode, multi-seed, as detection AUC over regions + onset localization.

Run: ./.venv/Scripts/python.exe -m temporal.experiments.dynamics_gate
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from temporal import subspace as ss
from temporal.synth_dynamics import DynConfig, make_panel

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "dynamics_gate")
os.makedirs(OUT, exist_ok=True)


def hankel(x, L):
    N = len(x); K = N - L + 1
    return np.stack([x[i:i + K] for i in range(L)])          # (L, K)


def ssa_subspace(x_win, L, rank=4):
    H = hankel(np.asarray(x_win, float), L)
    return ss.pca_subspace(H, rank, center=False, energy=0.99)


def acf1(x):
    x = np.asarray(x, float) - np.mean(x)
    d = np.dot(x, x)
    return float(np.dot(x[:-1], x[1:]) / d) if d > 1e-12 else 0.0


def score_series(x, W, L, rank, lag_present=0):
    """Slide a boundary t; compare past window [t-W,t) vs present [t,t+W). Return dict of method->(t, score)."""
    T = len(x)
    ts, ds, ma, msh, vsh, ash = [], [], [], [], [], []
    for t in range(W, T - W + 1):
        past = x[t - W:t]; pres = x[t:t + W]
        Sp = ssa_subspace(past, L, rank); Sq = ssa_subspace(pres, L, rank)
        cos = ss.canonical_cosines(Sp, Sq)
        ts.append(t)
        ds.append(ss.magnitude(Sp, Sq))
        ma.append(1.0 - np.max(cos))
        msh.append(abs(np.mean(pres) - np.mean(past)))
        vsh.append(abs(np.std(pres) - np.std(past)))
        ash.append(abs(acf1(pres) - acf1(past)))
    return {"t": np.array(ts), "ssa_ds": np.array(ds), "ssa_minangle": np.array(ma),
            "mean_shift": np.array(msh), "var_shift": np.array(vsh), "acf1_shift": np.array(ash)}


METHODS = ["ssa_ds", "ssa_minangle", "mean_shift", "var_shift", "acf1_shift"]


def evaluate_mode(mode, seeds=(0, 1, 2, 3, 4, 5), W=18, L=9, rank=4, n_regions=120):
    per_seed = {m: [] for m in METHODS}
    locerr = {m: [] for m in METHODS}
    for sd in seeds:
        cfg = DynConfig(mode=mode, seed=sd, n_regions=n_regions)
        X, labels = make_panel(cfg)
        det = {m: [] for m in METHODS}
        for x in X:
            sc = score_series(x, W, L, rank)
            for m in METHODS:
                det[m].append(float(np.max(sc[m])))            # unsupervised detection score per region
        for m in METHODS:
            per_seed[m].append(roc_auc_score(labels, det[m]))
        # onset localization on event regions
        for x, lb in zip(X, labels):
            if not lb:
                continue
            sc = score_series(x, W, L, rank)
            for m in METHODS:
                locerr[m].append(abs(sc["t"][int(np.argmax(sc[m]))] - cfg.t0))
    out = {m: {"auc_mean": float(np.mean(per_seed[m])), "auc_std": float(np.std(per_seed[m])),
               "locerr_med": float(np.median(locerr[m]))} for m in METHODS}
    return out


def main():
    modes = ["amp_collapse", "freq_shift", "noise_replace", "trend_onset"]
    results = {}
    print(f"{'mode':14} | " + " ".join(f"{m:>13}" for m in METHODS))
    for mode in modes:
        r = evaluate_mode(mode)
        results[mode] = r
        print(f"{mode:14} | " + " ".join(f"{r[m]['auc_mean']:.3f}±{r[m]['auc_std']:.2f}" for m in METHODS))

    # decisive contrast: on mean&variance-preserving structural changes, does ssa_ds beat the scalar stats?
    structural = ["freq_shift", "noise_replace"]
    verdict = {}
    for mode in structural:
        r = results[mode]
        scalar_best = max(r["mean_shift"]["auc_mean"], r["var_shift"]["auc_mean"])
        verdict[mode] = {
            "ssa_ds_auc": r["ssa_ds"]["auc_mean"],
            "best_scalar_auc": round(scalar_best, 3),
            "ssa_ds_minus_best_scalar": round(r["ssa_ds"]["auc_mean"] - scalar_best, 3),
            "ssa_ds_vs_minangle": round(r["ssa_ds"]["auc_mean"] - r["ssa_minangle"]["auc_mean"], 3),
        }
    print("\nDECISIVE (mean&var-preserving structural change):")
    print(json.dumps(verdict, indent=2))

    _fig_auc(results, modes, os.path.join(OUT, "fig_dynamics_auc.png"))
    _fig_example(os.path.join(OUT, "fig_example_series.png"))
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump({"results": results, "decisive": verdict}, f, indent=2)
    print(f"\nSaved to {OUT}")


def _fig_auc(results, modes, path):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(modes)); w = 0.16
    colors = {"ssa_ds": "C3", "ssa_minangle": "C1", "mean_shift": "C0", "var_shift": "C2", "acf1_shift": "C4"}
    for i, m in enumerate(METHODS):
        means = [results[mo][m]["auc_mean"] for mo in modes]
        errs = [results[mo][m]["auc_std"] for mo in modes]
        ax.bar(x + (i - 2) * w, means, w, yerr=errs, capsize=2, label=m, color=colors[m])
    ax.axhline(0.5, color="k", ls=":", lw=0.8, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(modes); ax.set_ylabel("detection AUC (mean±std over seeds)")
    ax.set_title("GATE: SSA temporal-DS vs scalar baselines per dynamics-change mode")
    ax.legend(ncol=5, fontsize=8, loc="lower center"); ax.set_ylim(0.3, 1.02)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _fig_example(path):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.2), sharex=True)
    from temporal.synth_dynamics import _make_one
    for ax, mode in zip(axes.ravel(), ["amp_collapse", "freq_shift", "noise_replace", "trend_onset"]):
        cfg = DynConfig(mode=mode, seed=1)
        ax.plot(_make_one(cfg, True, np.random.default_rng(1)), color="C3", label="event")
        ax.plot(_make_one(cfg, False, np.random.default_rng(1)), color="0.6", label="no-event", alpha=0.8)
        ax.axvline(cfg.t0, color="k", ls="--", lw=0.8); ax.set_title(mode, fontsize=9)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Dynamics-change scenarios (event vs no-event); mean preserved in amp/freq/noise modes")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
