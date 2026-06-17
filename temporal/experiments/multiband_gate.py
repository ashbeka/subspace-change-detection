"""DECISIVE battery: does the FULL multivariate Difference Subspace beat the single min-angle AND
all simple scalars, in a genuinely multi-dimensional (multi-mode, multi-band) regime?

Construction: multivariate SSA (M-SSA). For a window of a multi-band series, stack per-band Hankel
trajectory matrices -> a joint trajectory matrix -> a joint signal subspace (genuinely multi-dim).
Compare past vs present windows with the full DS magnitude.

Tests:
  1. Detection AUC per mode vs baselines (min-angle, multi-lag autocorrelation, variance, mean), multi-seed.
  2. NOISE SWEEP: full-DS vs min-angle margin as noise grows (Kanai's claim: DS aggregates across angles).
  3. ATTRIBUTION: does the difference-subspace per-band energy localize the changed band? (no scalar can).
  4. RECOVERY (2nd-order/geodesic): distance-to-baseline + off-geodesic onset on a recovery trajectory.

Run: ./.venv/Scripts/python.exe -m temporal.experiments.multiband_gate
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
from temporal.synth_multiband import MBConfig, make_panel

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "temporal", "outputs", "multiband_gate")
os.makedirs(OUT, exist_ok=True)


def hankel(x, L):
    N = len(x); K = N - L + 1
    return np.stack([x[i:i + K] for i in range(L)])


def mssa_subspace(Xwin, L, rank=8, energy=0.99):
    """Xwin (B, W) -> joint stacked Hankel subspace (B*L, .)."""
    blocks = [hankel(Xwin[b], L) for b in range(Xwin.shape[0])]   # each (L, K)
    H = np.concatenate(blocks, axis=0)                            # (B*L, K)
    return ss.pca_subspace(H, rank, center=False, energy=energy)


def acf_multilag(x, lags):
    x = np.asarray(x, float) - np.mean(x); d = np.dot(x, x)
    if d < 1e-12:
        return np.zeros(len(lags))
    return np.array([np.dot(x[:-k], x[k:]) / d for k in lags])


SCALARS = ["acf_ml", "var", "mean"]


def region_scores(X, W, L, rank, lags):
    """Return per-method anomaly-score series over boundary t for one region X (B,T)."""
    B, T = X.shape
    ts, ds, ma = [], [], []
    sc = {m: [] for m in SCALARS}
    for t in range(W, T - W + 1):
        past = X[:, t - W:t]; pres = X[:, t:t + W]
        Sp = mssa_subspace(past, L, rank); Sq = mssa_subspace(pres, L, rank)
        cos = ss.canonical_cosines(Sp, Sq)
        ts.append(t); ds.append(ss.magnitude(Sp, Sq)); ma.append(1.0 - np.max(cos))
        # scalar baselines aggregated over bands
        acf_d = np.mean([np.sum(np.abs(acf_multilag(pres[b], lags) - acf_multilag(past[b], lags))) for b in range(B)])
        var_d = np.mean([abs(np.std(pres[b]) - np.std(past[b])) for b in range(B)])
        mean_d = np.mean([abs(np.mean(pres[b]) - np.mean(past[b])) for b in range(B)])
        sc["acf_ml"].append(acf_d); sc["var"].append(var_d); sc["mean"].append(mean_d)
    out = {"t": np.array(ts), "mssa_ds": np.array(ds), "mssa_minangle": np.array(ma)}
    out.update({m: np.array(sc[m]) for m in SCALARS})
    return out


METHODS = ["mssa_ds", "mssa_minangle"] + SCALARS


def evaluate(mode, seeds=(0, 1, 2, 3, 4), noise=0.25, W=24, L=12, rank=8, n_regions=120):
    lags = list(range(1, L))
    aucs = {m: [] for m in METHODS}
    for sd in seeds:
        cfg = MBConfig(mode=mode, seed=sd, noise=noise, n_regions=n_regions)
        X, labels = make_panel(cfg)
        det = {m: [] for m in METHODS}
        for x in X:
            s = region_scores(x, W, L, rank, lags)
            for m in METHODS:
                det[m].append(float(np.max(s[m])))
        for m in METHODS:
            aucs[m].append(roc_auc_score(labels, det[m]))
    return {m: (float(np.mean(aucs[m])), float(np.std(aucs[m]))) for m in METHODS}


def noise_sweep(mode="harmonic_dropout", noises=(0.1, 0.25, 0.5, 0.8, 1.2), seeds=(0, 1, 2, 3)):
    res = {"noise": list(noises), "mssa_ds": [], "mssa_minangle": [], "best_scalar": []}
    for nz in noises:
        r = evaluate(mode, seeds=seeds, noise=nz)
        res["mssa_ds"].append(r["mssa_ds"][0]); res["mssa_minangle"].append(r["mssa_minangle"][0])
        res["best_scalar"].append(max(r["acf_ml"][0], r["var"][0], r["mean"][0]))
    return res


def attribution(seeds=(0, 1, 2, 3, 4), W=24, L=12, rank=8, noise=0.25):
    """freq_split: does the difference subspace's per-band energy point to the changed band?"""
    hits, total = 0, 0
    for sd in seeds:
        cfg = MBConfig(mode="freq_split", seed=sd, noise=noise, n_regions=60)
        X, labels = make_panel(cfg)
        for x, lb in zip(X, labels):
            if not lb:
                continue
            past = x[:, cfg.t0 - W:cfg.t0]; pres = x[:, cfg.t0:cfg.t0 + W]
            Sp = mssa_subspace(past, L, rank); Sq = mssa_subspace(pres, L, rank)
            D = ss.difference_subspace(Sp, Sq)               # (B*L, d)
            if D.shape[1] == 0:
                continue
            energy = np.array([np.sum(D[b * L:(b + 1) * L, :] ** 2) for b in range(cfg.B)])
            hits += int(np.argmax(energy) == cfg.changed_band); total += 1
    return {"attribution_accuracy": (hits / total) if total else None, "n": total, "chance": 1.0 / 6}


def recovery_demo(W=24, L=12, rank=8, seed=1):
    cfg = MBConfig(mode="recovery", seed=seed, noise=0.2, n_regions=2)
    X, labels = make_panel(cfg)
    x = X[np.argmax(labels)]                                  # an event region
    T = x.shape[1]
    centers, dpre, orth = [], [], []
    # sliding present-window subspaces, ref = first window
    subs = []
    for start in range(0, T - W + 1):
        subs.append((start + W // 2, mssa_subspace(x[:, start:start + W], L, rank)))
    ref = subs[0][1]
    for c, S in subs:
        centers.append(c); dpre.append(ss.magnitude(S, ref))
    for i in range(len(subs) - 2):
        o = ss.second_order_decomposed(subs[i][1], subs[i + 1][1], subs[i + 2][1])[1]
        orth.append((subs[i + 1][0], o))
    return cfg, np.array(centers), np.array(dpre), np.array(orth)


def main():
    modes = ["harmonic_dropout", "freq_split", "recovery"]
    results = {}
    print(f"{'mode':17} | " + " ".join(f"{m:>16}" for m in METHODS))
    for mode in modes:
        r = evaluate(mode)
        results[mode] = {m: {"auc": r[m][0], "std": r[m][1]} for m in METHODS}
        print(f"{mode:17} | " + " ".join(f"{r[m][0]:.3f}+-{r[m][1]:.2f}" for m in METHODS))

    ns = noise_sweep()
    print("\nNOISE SWEEP (harmonic_dropout) full-DS vs min-angle vs best-scalar:")
    for i, nz in enumerate(ns["noise"]):
        print(f"  noise={nz:>4}: DS={ns['mssa_ds'][i]:.3f}  minangle={ns['mssa_minangle'][i]:.3f}  "
              f"scalar={ns['best_scalar'][i]:.3f}  (DS-minangle={ns['mssa_ds'][i]-ns['mssa_minangle'][i]:+.3f})")

    attr = attribution()
    print(f"\nATTRIBUTION (freq_split): changed-band accuracy = {attr['attribution_accuracy']:.3f} "
          f"(chance {attr['chance']:.3f}, n={attr['n']})")

    cfgr, cen, dpre, orth = recovery_demo()

    _fig_auc(results, modes, os.path.join(OUT, "fig_multiband_auc.png"))
    _fig_noise(ns, os.path.join(OUT, "fig_noise_sweep.png"))
    _fig_recovery(cfgr, cen, dpre, orth, os.path.join(OUT, "fig_recovery.png"))

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump({"results": results, "noise_sweep": ns, "attribution": attr}, f, indent=2)
    print(f"\nSaved to {OUT}")


def _fig_auc(results, modes, path):
    fig, ax = plt.subplots(figsize=(10, 4.6)); x = np.arange(len(modes)); w = 0.16
    colors = {"mssa_ds": "C3", "mssa_minangle": "C1", "acf_ml": "C0", "var": "C2", "mean": "C4"}
    for i, m in enumerate(METHODS):
        ax.bar(x + (i - 2) * w, [results[mo][m]["auc"] for mo in modes], w,
               yerr=[results[mo][m]["std"] for mo in modes], capsize=2, label=m, color=colors[m])
    ax.axhline(0.5, color="k", ls=":", lw=0.8, alpha=.6); ax.set_xticks(x); ax.set_xticklabels(modes)
    ax.set_ylabel("detection AUC (mean±std)"); ax.set_ylim(0.3, 1.02)
    ax.set_title("Multivariate SSA: full DS vs min-angle vs scalars"); ax.legend(ncol=5, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _fig_noise(ns, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(ns["noise"], ns["mssa_ds"], "o-", color="C3", lw=2.2, label="full DS")
    ax.plot(ns["noise"], ns["mssa_minangle"], "s--", color="C1", lw=1.8, label="min-angle (conventional SSA)")
    ax.plot(ns["noise"], ns["best_scalar"], "^:", color="C0", lw=1.6, label="best scalar")
    ax.axhline(0.5, color="k", ls=":", lw=.8, alpha=.6, label="chance")
    ax.set_xlabel("noise level"); ax.set_ylabel("detection AUC")
    ax.set_title("Full DS aggregates across angles -> pulls ahead of min-angle under noise")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _fig_recovery(cfg, cen, dpre, orth, path):
    def nz(a): a = np.asarray(a, float); return (a - a.min()) / (np.ptp(a) + 1e-12)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(cen, nz(dpre), color="C3", lw=2.3, label="d_pre (distance to baseline)")
    if len(orth):
        ax.plot(orth[:, 0], nz(orth[:, 1]), color="C1", lw=1.6, label="2nd-order off-geodesic (onset)")
    ax.axvline(cfg.t0, color="k", ls="--", lw=.9, alpha=.6, label=f"onset t0={cfg.t0}")
    half = cfg.t0 + (cfg.T - cfg.t0) // 2
    ax.axvline(half, color="g", ls="--", lw=.9, alpha=.6, label="recovery turn")
    ax.set_xlabel("date index"); ax.set_ylabel("normalized")
    ax.set_title("Recovery (multi-band): d_pre rises then falls; off-geodesic marks abrupt onset")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
