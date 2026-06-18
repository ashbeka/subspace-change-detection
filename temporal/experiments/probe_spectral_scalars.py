"""ADVERSARIAL PROBE: strong FREQUENCY-DOMAIN scalar baselines vs the full multivariate DS.

The committed multiband_gate.py only pits the full DS against weak scalars (multi-lag autocorrelation,
variance, mean). But the headline change `harmonic_dropout` is a SPECTRAL event: the 2nd harmonic
(period P/2) vanishes while the 1st harmonic is rescaled to preserve total variance. A competent analyst
would NOT reach for variance/mean -- they would reach for a periodogram. The committed "best scalar"
is therefore a strawman.

This probe drops in STRONG, one-line-ish, frequency-domain scalars on the EXACT SAME panels / labels /
windows / noise sweep / AUC machinery as multiband_gate.evaluate(), reusing the committed builders
(make_panel, MBConfig). Each scalar produces a past-vs-present "change score" per boundary t per region,
max-pooled over t (identical to the committed protocol), then pooled into an AUC over the panel.

Strong scalars added (all per-band, aggregated over bands like the committed acf_ml):
  psd_l2     : L2 distance between past & present windowed periodograms (rFFT |.|^2).
  psd_l1     : L1 distance between past & present periodograms.
  spec_ent   : |spectral entropy(present) - spectral entropy(past)| (Shannon entropy of normalized PSD).
  acf_full   : L2 distance of the FULL autocorrelation vector (all lags), not just multi-lag sum.
  h2_power   : |2nd-harmonic-band power(present) - 2nd-harmonic-band power(past)|, band centered at P/2.
               This is the ORACLE scalar -- it knows exactly which frequency drops out.
  h2_ratio   : |power_ratio(present) - power_ratio(past)| where ratio = P(2nd harm)/P(1st harm).
               Dimensionless; should be maximally sensitive to harmonic dropout, gain-robust.
  dom_freq   : |argmax-frequency(present) - argmax-frequency(past)| (dominant-frequency shift).

Also re-runs the committed weak scalars (acf_ml, var, mean) and DS/min-angle so the comparison is
apples-to-apples within one script. The decisive question: across the noise sweep on harmonic_dropout,
does the full DS still beat the BEST of these strong spectral scalars? If a one-line PSD/spectral
detector matches or beats DS, that is a major threat to the DS contribution.

Outputs: temporal/outputs/_mbprobe_spectral/
Run: ./.venv/Scripts/python.exe -m temporal.experiments.probe_spectral_scalars
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from temporal import subspace as ss
from temporal.synth_multiband import MBConfig, make_panel
from temporal.experiments.multiband_gate import mssa_subspace, acf_multilag, hankel  # reuse committed builders

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "_mbprobe_spectral")
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------- spectral primitives
def _psd(x):
    """One-sided power spectral density of a 1D window via rFFT. Mean-subtracted, Hann-tapered."""
    x = np.asarray(x, float)
    x = x - x.mean()
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    return (np.abs(X) ** 2)


def _spec_entropy(x):
    p = _psd(x)
    s = p.sum()
    if s < 1e-12:
        return 0.0
    p = p / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _band_power(x, P, harmonic, half_width=1):
    """Power in a narrow bin centered on the `harmonic`-th harmonic of period P.

    rFFT bin k corresponds to frequency k/len(x); the m-th harmonic of period P sits at
    frequency m/P, i.e. bin index m*len(x)/P. We sum |X|^2 in a small window around that bin.
    """
    x = np.asarray(x, float)
    x = x - x.mean()
    N = len(x)
    w = np.hanning(N)
    Xp = np.abs(np.fft.rfft(x * w)) ** 2
    k = harmonic * N / P
    lo = max(0, int(round(k)) - half_width)
    hi = min(len(Xp) - 1, int(round(k)) + half_width)
    return float(np.sum(Xp[lo:hi + 1]))


def _dom_freq(x):
    p = _psd(x)
    if p.sum() < 1e-12:
        return 0.0
    return float(np.argmax(p) / len(x))


def _acf_full(x):
    """Full (biased) autocorrelation vector for lags 0..N-1, normalized by lag-0."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    ac = np.correlate(x, x, mode="full")[n - 1:]
    if ac[0] < 1e-12:
        return np.zeros(n)
    return ac / ac[0]


# ----------------------------------------------------------------- per-region score series
WEAK = ["acf_ml", "var", "mean"]
STRONG = ["psd_l2", "psd_l1", "spec_ent", "acf_full", "h2_power", "h2_ratio", "dom_freq"]
SUB = ["mssa_ds", "mssa_minangle"]
METHODS = SUB + WEAK + STRONG


def region_scores(X, W, L, rank, lags, period):
    """Per-method change-score series over boundary t for one region X (B,T). Mirrors committed loop."""
    B, T = X.shape
    acc = {m: [] for m in METHODS}
    for t in range(W, T - W + 1):
        past = X[:, t - W:t]
        pres = X[:, t:t + W]
        # --- subspace methods (committed) ---
        Sp = mssa_subspace(past, L, rank)
        Sq = mssa_subspace(pres, L, rank)
        cos = ss.canonical_cosines(Sp, Sq)
        acc["mssa_ds"].append(ss.magnitude(Sp, Sq))
        acc["mssa_minangle"].append(1.0 - np.max(cos))
        # --- weak scalars (committed) ---
        acc["acf_ml"].append(np.mean([np.sum(np.abs(acf_multilag(pres[b], lags) - acf_multilag(past[b], lags))) for b in range(B)]))
        acc["var"].append(np.mean([abs(np.std(pres[b]) - np.std(past[b])) for b in range(B)]))
        acc["mean"].append(np.mean([abs(np.mean(pres[b]) - np.mean(past[b])) for b in range(B)]))
        # --- strong spectral scalars (this probe) ---
        psd_p = [_psd(past[b]) for b in range(B)]
        psd_q = [_psd(pres[b]) for b in range(B)]
        acc["psd_l2"].append(np.mean([np.linalg.norm(psd_q[b] - psd_p[b]) for b in range(B)]))
        acc["psd_l1"].append(np.mean([np.sum(np.abs(psd_q[b] - psd_p[b])) for b in range(B)]))
        acc["spec_ent"].append(np.mean([abs(_spec_entropy(pres[b]) - _spec_entropy(past[b])) for b in range(B)]))
        acc["acf_full"].append(np.mean([np.linalg.norm(_acf_full(pres[b]) - _acf_full(past[b])) for b in range(B)]))
        h2p = np.array([_band_power(past[b], period, 2) for b in range(B)])
        h2q = np.array([_band_power(pres[b], period, 2) for b in range(B)])
        h1p = np.array([_band_power(past[b], period, 1) for b in range(B)])
        h1q = np.array([_band_power(pres[b], period, 1) for b in range(B)])
        acc["h2_power"].append(np.mean(np.abs(h2q - h2p)))
        rp = h2p / (h1p + 1e-9)
        rq = h2q / (h1q + 1e-9)
        acc["h2_ratio"].append(np.mean(np.abs(rq - rp)))
        acc["dom_freq"].append(np.mean([abs(_dom_freq(pres[b]) - _dom_freq(past[b])) for b in range(B)]))
    return {m: np.array(acc[m]) for m in METHODS}


def evaluate(mode, seeds=(0, 1, 2, 3, 4), noise=0.25, W=24, L=12, rank=8, n_regions=120, period=16.0):
    lags = list(range(1, L))
    aucs = {m: [] for m in METHODS}
    for sd in seeds:
        cfg = MBConfig(mode=mode, seed=sd, noise=noise, n_regions=n_regions, period=period)
        X, labels = make_panel(cfg)
        det = {m: [] for m in METHODS}
        for x in X:
            s = region_scores(x, W, L, rank, lags, cfg.period)
            for m in METHODS:
                det[m].append(float(np.max(s[m])))
        for m in METHODS:
            aucs[m].append(roc_auc_score(labels, det[m]))
    return {m: (float(np.mean(aucs[m])), float(np.std(aucs[m]))) for m in METHODS}


def noise_sweep(mode="harmonic_dropout", noises=(0.1, 0.25, 0.5, 0.8, 1.2), seeds=(0, 1, 2, 3)):
    res = {"noise": list(noises)}
    for m in METHODS:
        res[m] = []
    res["best_weak"] = []
    res["best_strong"] = []
    res["best_strong_name"] = []
    for nz in noises:
        r = evaluate(mode, seeds=seeds, noise=nz)
        for m in METHODS:
            res[m].append(round(r[m][0], 4))
        res["best_weak"].append(round(max(r[w][0] for w in WEAK), 4))
        strong_vals = {sname: r[sname][0] for sname in STRONG}
        best_s = max(strong_vals, key=strong_vals.get)
        res["best_strong"].append(round(strong_vals[best_s], 4))
        res["best_strong_name"].append(best_s)
    return res


def main():
    print("=== FULL-PANEL AUC per method (harmonic_dropout, noise=0.25, 5 seeds) ===")
    r = evaluate("harmonic_dropout")
    for m in METHODS:
        print(f"  {m:14}: {r[m][0]:.3f} +- {r[m][1]:.3f}")

    ns = noise_sweep("harmonic_dropout")
    print("\n=== NOISE SWEEP (harmonic_dropout): DS vs strong spectral scalars ===")
    hdr = f"{'noise':>6} | {'DS':>6} {'minang':>6} | " + " ".join(f"{s:>8}" for s in STRONG) + f" | {'bestWeak':>8} {'bestStrong':>10}"
    print(hdr)
    for i, nz in enumerate(ns["noise"]):
        row = f"{nz:>6} | {ns['mssa_ds'][i]:>6.3f} {ns['mssa_minangle'][i]:>6.3f} | " + \
              " ".join(f"{ns[s][i]:>8.3f}" for s in STRONG) + \
              f" | {ns['best_weak'][i]:>8.3f} {ns['best_strong'][i]:>10.3f} ({ns['best_strong_name'][i]})"
        print(row)

    print("\n=== DS vs BEST-STRONG margin per noise ===")
    for i, nz in enumerate(ns["noise"]):
        margin = ns["mssa_ds"][i] - ns["best_strong"][i]
        verdict = "DS wins" if margin > 0.01 else ("TIE" if abs(margin) <= 0.01 else "SCALAR WINS")
        print(f"  noise={nz:>4}: DS={ns['mssa_ds'][i]:.3f}  best_strong={ns['best_strong'][i]:.3f} "
              f"({ns['best_strong_name'][i]})  margin={margin:+.3f}  -> {verdict}")

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump({"full_panel_auc": {m: r[m] for m in METHODS}, "noise_sweep": ns}, f, indent=2)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
